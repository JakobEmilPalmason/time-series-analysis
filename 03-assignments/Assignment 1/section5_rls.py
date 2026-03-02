"""Assignment 1 — Section 5: Recursive Estimation and Optimization of Lambda"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Data preparation (same as other sections) ────────────────────────
D = pd.read_csv("DST_BIL54.csv")
year = D["time"].str[:4].astype(int)
month = D["time"].str[5:7].astype(int)
D["x"] = year + (month - 1) / 12
D["total"] = D["total"] / 1e6

train = D[D["x"] < 2024].copy()
test = D[D["x"] >= 2024].copy()

y = train["total"].values
x = train["x"].values
X = np.column_stack([np.ones(len(train)), x])
N = len(y)

x_test = test["x"].values
y_test = test["total"].values
X_test = np.column_stack([np.ones(len(test)), x_test])

# OLS estimates (baseline comparison)
theta_ols = np.linalg.solve(X.T @ X, X.T @ y)
y_hat_ols = X @ theta_ols
resid_ols = y - y_hat_ols
sigma2_ols = np.sum(resid_ols**2) / (N - 2)


# ── RLS function (reusable for 5.2–5.7) ─────────────────────────────
def rls(y, X, lam=1.0, R0=None, theta0=None, return_details=False):
    """Recursive Least Squares with forgetting factor.

    Returns:
        thetas: (N, p) array of parameter estimates at each time step
        If return_details=True, also returns Rs (list of R_t matrices)
    """
    N, p = X.shape
    if R0 is None:
        R0 = 0.1 * np.eye(p)
    if theta0 is None:
        theta0 = np.zeros(p)

    thetas = np.zeros((N, p))
    R = R0.copy()
    theta = theta0.copy()
    Rs = [R0.copy()] if return_details else None

    for t in range(N):
        x_t = X[t].reshape(-1, 1)       # (p, 1)
        R = lam * R + x_t @ x_t.T       # (p, p)
        e_t = y[t] - (X[t] @ theta)     # scalar residual
        theta = theta + np.linalg.solve(R, X[t]) * e_t
        thetas[t] = theta
        if return_details:
            Rs.append(R.copy())

    if return_details:
        return thetas, Rs
    return thetas


# ══════════════════════════════════════════════════════════════════════
# 5.1 + 5.2 — RLS without forgetting: hand-calc verification + loop
# ══════════════════════════════════════════════════════════════════════
print("=== 5.1 & 5.2 RLS Without Forgetting (First Iterations) ===")

R0 = 0.1 * np.eye(2)
theta0 = np.zeros(2)

thetas_noforgetting, Rs = rls(y, X, lam=1.0, R0=R0, theta0=theta0,
                               return_details=True)

# Print R_1 and R_2 for hand-calculation verification
print(f"\nInitial values:")
print(f"  R_0 = [[0.1, 0], [0, 0.1]]")
print(f"  θ_0 = [0, 0]")
print(f"  x_1 = [1, {x[0]:.3f}]^T,  Y_1 = {y[0]:.6f}")
print(f"  x_2 = [1, {x[1]:.3f}]^T,  Y_2 = {y[1]:.6f}")

print(f"\nR_1 = R_0 + x_1 x_1^T:")
print(f"  [[{Rs[1][0,0]:.4f}, {Rs[1][0,1]:.4f}],")
print(f"   [{Rs[1][1,0]:.4f}, {Rs[1][1,1]:.4f}]]")

print(f"\nR_2 = R_1 + x_2 x_2^T:")
print(f"  [[{Rs[2][0,0]:.4f}, {Rs[2][0,1]:.4f}],")
print(f"   [{Rs[2][1,0]:.4f}, {Rs[2][1,1]:.4f}]]")

print(f"\nParameter estimates:")
for t in range(3):
    print(f"  θ̂_{t+1} = [{thetas_noforgetting[t, 0]:>12.6f}, {thetas_noforgetting[t, 1]:>10.8f}]")
print()


# ══════════════════════════════════════════════════════════════════════
# 5.3 — RLS at t=N vs OLS; effect of initial values
# ══════════════════════════════════════════════════════════════════════
print("=== 5.3 RLS at t=N vs OLS ===")

theta_rls_N = thetas_noforgetting[-1]
print(f"RLS  (R0=0.1·I): θ̂₁ = {theta_rls_N[0]:.6f},  θ̂₂ = {theta_rls_N[1]:.8f}")
print(f"OLS:              θ̂₁ = {theta_ols[0]:.6f},  θ̂₂ = {theta_ols[1]:.8f}")
print(f"Difference:       Δθ̂₁ = {abs(theta_rls_N[0] - theta_ols[0]):.6e},  "
      f"Δθ̂₂ = {abs(theta_rls_N[1] - theta_ols[1]):.6e}")

# Re-run with progressively smaller R0 to reduce prior influence
for r0_scale in [1e-6, 1e-9]:
    R0_small = r0_scale * np.eye(2)
    thetas_small = rls(y, X, lam=1.0, R0=R0_small, theta0=theta0)
    theta_N = thetas_small[-1]
    print(f"\nRLS  (R0={r0_scale:.0e}·I): θ̂₁ = {theta_N[0]:.6f},  θ̂₂ = {theta_N[1]:.8f}")
    print(f"OLS:                θ̂₁ = {theta_ols[0]:.6f},  θ̂₂ = {theta_ols[1]:.8f}")
    print(f"Difference:         Δθ̂₁ = {abs(theta_N[0] - theta_ols[0]):.6e},  "
          f"Δθ̂₂ = {abs(theta_N[1] - theta_ols[1]):.6e}")

print()
print("→ With smaller R_0, the prior has less influence and RLS converges")
print("  closer to OLS. R_0 acts as a regularization/prior on the information")
print("  matrix; making it smaller means we start with 'less prior information'.")
print("  The remaining gap is due to the ill-conditioned design matrix (x ≈ 2018),")
print("  which amplifies even tiny R_0 values in the intercept estimate.")
print()


# ══════════════════════════════════════════════════════════════════════
# 5.4 — RLS with forgetting; plot parameter trajectories
# ══════════════════════════════════════════════════════════════════════
print("=== 5.4 RLS With Forgetting Factor ===")

lambdas_54 = [0.7, 0.99]
colors = {"0.7": "tab:blue", "0.99": "tab:red"}
burn_in = 5  # skip first few points in plots

# Also compute WLS for comparison
def wls_estimate(y, X, lam):
    N = len(y)
    weights = np.array([lam ** (N - i) for i in range(1, N + 1)])
    W = np.diag(weights)
    return np.linalg.solve(X.T @ W @ X, X.T @ W @ y)

# Run RLS for each lambda
rls_results = {}
for lam in lambdas_54:
    thetas_lam = rls(y, X, lam=lam, R0=R0, theta0=theta0)
    rls_results[lam] = thetas_lam
    theta_wls = wls_estimate(y, X, lam)
    print(f"λ = {lam}:")
    print(f"  RLS θ̂_N = [{thetas_lam[-1, 0]:.6f}, {thetas_lam[-1, 1]:.8f}]")
    print(f"  WLS θ̂   = [{theta_wls[0]:.6f}, {theta_wls[1]:.8f}]")
    delta1 = abs(thetas_lam[-1, 0] - theta_wls[0])
    delta2 = abs(thetas_lam[-1, 1] - theta_wls[1])
    print(f"  → Match: Δθ̂₁ = {delta1:.2e}, Δθ̂₂ = {delta2:.2e}")
    if delta1 > 1:
        print(f"    (Large gap due to R_0 influence; with slow forgetting,")
        print(f"     the initial R_0 = 0.1·I persists across all iterations.)")
print()

# Plot θ̂₁ (intercept) over time
plt.figure(figsize=(10, 5))
for lam in lambdas_54:
    label = f"λ = {lam}"
    plt.plot(x[burn_in:], rls_results[lam][burn_in:, 0], linewidth=1.5,
             color=colors[str(lam)], label=label)
plt.axhline(theta_ols[0], color="gray", linestyle="--", linewidth=1, label="OLS")
plt.xlabel("Time (year)")
plt.ylabel("θ̂₁ (intercept)")
plt.title("5.4 — RLS intercept estimates over time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_5_4a.png", dpi=150)
plt.close()

# Plot θ̂₂ (slope) over time
plt.figure(figsize=(10, 5))
for lam in lambdas_54:
    label = f"λ = {lam}"
    plt.plot(x[burn_in:], rls_results[lam][burn_in:, 1], linewidth=1.5,
             color=colors[str(lam)], label=label)
plt.axhline(theta_ols[1], color="gray", linestyle="--", linewidth=1, label="OLS")
plt.xlabel("Time (year)")
plt.ylabel("θ̂₂ (slope)")
plt.title("5.4 — RLS slope estimates over time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_5_4b.png", dpi=150)
plt.close()

print("Plots saved: plot_5_4a.png, plot_5_4b.png")
print()


# ══════════════════════════════════════════════════════════════════════
# 5.5 — One-step predictions and residuals
# ══════════════════════════════════════════════════════════════════════
print("=== 5.5 One-Step Predictions and Residuals ===")

plt.figure(figsize=(10, 5))
for lam in lambdas_54:
    thetas_lam = rls_results[lam]
    # One-step prediction: y_hat_{t+1|t} = x_{t+1}^T @ theta_t
    # For t = 0, ..., N-2: predict y[t+1] using theta[t]
    y_pred_1step = np.array([X[t + 1] @ thetas_lam[t] for t in range(N - 1)])
    resid_1step = y[1:] - y_pred_1step  # epsilon_{t+1|t}

    rmse_1step = np.sqrt(np.mean(resid_1step[burn_in:]**2))
    print(f"λ = {lam}: 1-step RMSE (after burn-in) = {rmse_1step:.6f}")

    plt.plot(x[burn_in + 1:], resid_1step[burn_in:], linewidth=1,
             color=colors[str(lam)], label=f"λ = {lam}", alpha=0.8)

plt.axhline(0, color="k", linestyle="--", linewidth=0.5)
plt.xlabel("Time (year)")
plt.ylabel("Residual (millions)")
plt.title("5.5 — One-step prediction residuals")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_5_5.png", dpi=150)
plt.close()

print("Plot saved: plot_5_5.png")
print()


# ══════════════════════════════════════════════════════════════════════
# 5.6 — Optimize lambda for horizons k = 1, ..., 12
# ══════════════════════════════════════════════════════════════════════
print("=== 5.6 Lambda Optimization ===")

lambda_grid = np.arange(0.30, 1.00, 0.01)
horizons = range(1, 13)
burn_in_opt = 10  # discard first 10 observations for RMSE calculation

# Precompute RLS for each lambda
rmse_matrix = np.zeros((len(lambda_grid), len(horizons)))

for i, lam in enumerate(lambda_grid):
    thetas_lam = rls(y, X, lam=lam, R0=R0, theta0=theta0)

    for j, k in enumerate(horizons):
        # k-step residual: epsilon_{t|t-k} = Y_t - x_t^T @ theta_{t-k}
        # Valid range: t = k, k+1, ..., N-1 (0-indexed)
        # But theta_{t-k} needs t-k >= 0, so t >= k
        # Also skip burn-in: t-k >= burn_in_opt, so t >= k + burn_in_opt
        start_t = k + burn_in_opt
        if start_t >= N:
            rmse_matrix[i, j] = np.nan
            continue

        residuals_k = []
        for t in range(start_t, N):
            y_pred_k = X[t] @ thetas_lam[t - k]
            residuals_k.append(y[t] - y_pred_k)
        residuals_k = np.array(residuals_k)
        rmse_matrix[i, j] = np.sqrt(np.mean(residuals_k**2))

# Find optimal lambda for each horizon
print(f"{'Horizon k':<12} {'Optimal λ':<12} {'RMSE':>10}")
print("-" * 36)
optimal_lambdas = {}
for j, k in enumerate(horizons):
    best_idx = np.nanargmin(rmse_matrix[:, j])
    optimal_lambdas[k] = lambda_grid[best_idx]
    print(f"  k = {k:<6} λ* = {lambda_grid[best_idx]:.2f}    "
          f"RMSE = {rmse_matrix[best_idx, j]:.6f}")
print()

# Plot RMSE vs lambda for all horizons
plt.figure(figsize=(10, 6))
cmap = plt.cm.viridis
for j, k in enumerate(horizons):
    color = cmap(j / (len(horizons) - 1))
    plt.plot(lambda_grid, rmse_matrix[:, j], linewidth=1.2, color=color,
             label=f"k={k}")
plt.xlabel("Forgetting factor λ")
plt.ylabel("RMSE (millions)")
plt.title("5.6 — RMSE vs λ for prediction horizons k = 1, ..., 12")
plt.legend(ncol=3, fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_5_6.png", dpi=150)
plt.close()

print("Plot saved: plot_5_6.png")
print()


# ══════════════════════════════════════════════════════════════════════
# 5.7 — Predict test set using RLS
# ══════════════════════════════════════════════════════════════════════
print("=== 5.7 Test Set Predictions ===")

# Use optimal lambda for k=1 as single lambda
lam_opt = optimal_lambdas[1]
print(f"Using optimal λ* = {lam_opt:.2f} (from k=1 optimization)")

thetas_opt = rls(y, X, lam=lam_opt, R0=R0, theta0=theta0)
theta_rls_final = thetas_opt[-1]

# RLS predictions: y_hat_{N+k|N} = [1, x_{N+k}] @ theta_N
pred_rls = X_test @ theta_rls_final

# Also compute per-horizon optimal predictions
pred_rls_perh = np.zeros(len(x_test))
for k in range(1, 13):
    lam_k = optimal_lambdas[k]
    thetas_k = rls(y, X, lam=lam_k, R0=R0, theta0=theta0)
    pred_rls_perh[k - 1] = X_test[k - 1] @ thetas_k[-1]

# OLS predictions
pred_ols = X_test @ theta_ols

# WLS predictions (lambda = 0.9, same as Part 4)
theta_wls_09 = wls_estimate(y, X, 0.9)
pred_wls = X_test @ theta_wls_09

# RMSE comparison
rmse_ols_test = np.sqrt(np.mean((y_test - pred_ols)**2))
rmse_wls_test = np.sqrt(np.mean((y_test - pred_wls)**2))
rmse_rls_test = np.sqrt(np.mean((y_test - pred_rls)**2))
rmse_rls_perh_test = np.sqrt(np.mean((y_test - pred_rls_perh)**2))

print(f"\nTest RMSE comparison:")
print(f"  OLS:                          {rmse_ols_test:.6f}")
print(f"  WLS (λ=0.9):                  {rmse_wls_test:.6f}")
print(f"  RLS (λ*={lam_opt:.2f}, single):     {rmse_rls_test:.6f}")
print(f"  RLS (per-horizon optimal λ):  {rmse_rls_perh_test:.6f}")
print()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, "k.", markersize=6, label="Training data")
plt.plot(x_test, y_test, "kx", markersize=7, label="Test data")
plt.plot(x_test, pred_ols, "b-o", markersize=4, linewidth=1.5, label="OLS")
plt.plot(x_test, pred_wls, "g-s", markersize=4, linewidth=1.5, label="WLS (λ=0.9)")
plt.plot(x_test, pred_rls, "r-^", markersize=4, linewidth=1.5,
         label=f"RLS (λ*={lam_opt:.2f})")
plt.plot(x_test, pred_rls_perh, "m-d", markersize=4, linewidth=1.5,
         label="RLS (per-horizon λ*)")
plt.xlabel("Time (year)")
plt.ylabel("Total vehicles (millions)")
plt.title("5.7 — OLS vs WLS vs RLS forecasts for 2024")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_5_7.png", dpi=150)
plt.close()

print("Plot saved: plot_5_7.png")
print()


# ══════════════════════════════════════════════════════════════════════
# 5.8 — Reflections (printed summary; full discussion in notes)
# ══════════════════════════════════════════════════════════════════════
print("=== 5.8 Reflections ===")
print("See assignment1-notes.md for full discussion.")
print()
print("Key points:")
print("- Small λ → short memory → overfitting (tracks noise)")
print("- Large λ / OLS → long memory → underfitting (misses regime changes)")
print("- Optimal λ typically depends on prediction horizon")
print("- RLS one-step residuals provide built-in cross-validation")
print("- Alternative methods: Kalman filter, exponential smoothing, state-space models")
