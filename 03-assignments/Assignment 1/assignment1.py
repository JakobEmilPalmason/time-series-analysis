import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Read data
D = pd.read_csv("DST_BIL54.csv")

# Parse time: "2018-01" -> year + (month-1)/12
year = D["time"].str[:4].astype(int)
month = D["time"].str[5:7].astype(int)
D["x"] = year + (month - 1) / 12

# Total vehicles in millions
D["total"] = D["total"] / 1e6

# Train/test split
train = D[D["x"] < 2024].copy()
test = D[D["x"] >= 2024].copy()

# --- 1.1 Plot training data ---
plt.figure(figsize=(10, 5))
plt.plot(train["x"], train["total"], "k.", markersize=6)
plt.xlabel("Time (year)")
plt.ylabel("Total vehicles (millions)")
plt.title("1.1 — Registered motor vehicles in Denmark (training set)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_1_1.png", dpi=150)
plt.show()

# --- 3.1 OLS estimation ---
y = train["total"].values
X = np.column_stack([np.ones(len(train)), train["x"].values])
N = len(y)

# OLS: theta_hat = (X'X)^{-1} X'y
theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
print("=== 3.1 OLS Estimates ===")
print(f"theta_1 (intercept) = {theta_hat[0]:.6f}")
print(f"theta_2 (slope)     = {theta_hat[1]:.6f}")

# Residuals and sigma^2
y_hat = X @ theta_hat
residuals = y - y_hat
sigma2 = np.sum(residuals**2) / (N - 2)
print(f"sigma^2             = {sigma2:.6e}")
print(f"sigma               = {np.sqrt(sigma2):.6f}")

# --- 3.2 Std errors and fitted line plot ---
cov_theta = sigma2 * np.linalg.inv(X.T @ X)
se_theta = np.sqrt(np.diag(cov_theta))

print("\n=== 3.2 Parameter Estimates with Std Errors ===")
print(f"theta_1 = {theta_hat[0]:.4f}  (se = {se_theta[0]:.4f})")
print(f"theta_2 = {theta_hat[1]:.6f}  (se = {se_theta[1]:.6f})")

plt.figure(figsize=(10, 5))
plt.plot(train["x"], train["total"], "k.", markersize=6, label="Training data")
plt.plot(train["x"], y_hat, "r-", linewidth=2, label="OLS fit")
plt.xlabel("Time (year)")
plt.ylabel("Total vehicles (millions)")
plt.title("3.2 — OLS linear trend model")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_3_2.png", dpi=150)
plt.show()

# --- 3.3 Forecast test set with prediction intervals ---
x_test = test["x"].values
X_test = np.column_stack([np.ones(len(test)), x_test])
y_test = test["total"].values

y_pred = X_test @ theta_hat

# Prediction variance: sigma^2 * (1 + x'(X'X)^{-1}x) for each test point
XtX_inv = np.linalg.inv(X.T @ X)
pred_var = np.array([sigma2 * (1 + x_i @ XtX_inv @ x_i) for x_i in X_test])
pred_se = np.sqrt(pred_var)

# 95% prediction interval using t-distribution
t_crit = stats.t.ppf(0.975, df=N - 2)
pred_lower = y_pred - t_crit * pred_se
pred_upper = y_pred + t_crit * pred_se

print("\n=== 3.3 Forecast Table (2024) ===")
print(f"{'Month':<10} {'Predicted':>10} {'Lower 95%':>10} {'Upper 95%':>10}")
print("-" * 42)
months_2024 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for i in range(len(x_test)):
    print(f"{months_2024[i]:<10} {y_pred[i]:>10.4f} {pred_lower[i]:>10.4f} {pred_upper[i]:>10.4f}")

# --- 3.4 Plot training + forecast + prediction intervals ---
plt.figure(figsize=(10, 5))
plt.plot(train["x"], train["total"], "k.", markersize=6, label="Training data")
x_line = np.linspace(train["x"].min(), test["x"].max(), 200)
X_line = np.column_stack([np.ones(len(x_line)), x_line])
plt.plot(x_line, X_line @ theta_hat, "r-", linewidth=1.5, label="OLS fit")
plt.plot(x_test, y_test, "bs", markersize=5, label="Test data (actual)")
plt.plot(x_test, y_pred, "r^", markersize=5, label="Forecast")
plt.fill_between(x_test, pred_lower, pred_upper, color="red", alpha=0.15,
                 label="95% prediction interval")
plt.xlabel("Time (year)")
plt.ylabel("Total vehicles (millions)")
plt.title("3.4 — OLS forecast vs actual test data")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_3_4.png", dpi=150)
plt.show()

# --- 3.5 Comment on forecast ---
rmse_test = np.sqrt(np.mean((y_pred - y_test)**2))
print(f"\n=== 3.5 Forecast Quality ===")
print(f"Test RMSE = {rmse_test:.4f} million vehicles")
print(f"Mean absolute error = {np.mean(np.abs(y_pred - y_test)):.4f}")

# --- 3.6 Residual diagnostics ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residuals vs time
axes[0, 0].plot(train["x"], residuals, "k.", markersize=4)
axes[0, 0].axhline(0, color="r", linestyle="--", linewidth=0.8)
axes[0, 0].set_xlabel("Time (year)")
axes[0, 0].set_ylabel("Residual")
axes[0, 0].set_title("Residuals vs time")

# Residuals vs fitted
axes[0, 1].plot(y_hat, residuals, "k.", markersize=4)
axes[0, 1].axhline(0, color="r", linestyle="--", linewidth=0.8)
axes[0, 1].set_xlabel("Fitted values")
axes[0, 1].set_ylabel("Residual")
axes[0, 1].set_title("Residuals vs fitted")

# QQ plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Normal Q-Q plot")

# ACF
max_lag = 20
acf_vals = [np.corrcoef(residuals[:-k], residuals[k:])[0, 1] for k in range(1, max_lag + 1)]
axes[1, 1].bar(range(1, max_lag + 1), acf_vals, color="steelblue", width=0.6)
axes[1, 1].axhline(1.96 / np.sqrt(N), color="r", linestyle="--", linewidth=0.8)
axes[1, 1].axhline(-1.96 / np.sqrt(N), color="r", linestyle="--", linewidth=0.8)
axes[1, 1].set_xlabel("Lag")
axes[1, 1].set_ylabel("ACF")
axes[1, 1].set_title("Autocorrelation of residuals")

fig.suptitle("3.6 — Residual diagnostics", fontsize=13)
plt.tight_layout()
plt.savefig("plot_3_6.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════
# Section 4: WLS Local Linear Trend Model (λ = 0.9)
# ══════════════════════════════════════════════════════════════════════

lam = 0.9

# ── 4.1 Variance-covariance matrix ────────────────────────────────────
print("=== 4.1 Variance-Covariance Matrix ===")

# Weight for observation i (1-indexed): w_i = λ^{N-i}
# Most recent (i=N) has weight λ^0 = 1, oldest (i=1) has weight λ^{N-1}
weights = np.array([lam ** (N - i) for i in range(1, N + 1)])
W = np.diag(weights)

print(f"Weight matrix W is {N}×{N} diagonal.")
print(f"  W[1,1]   = λ^{N-1} = {weights[0]:.6e}  (oldest observation)")
print(f"  W[2,2]   = λ^{N-2} = {weights[1]:.6e}")
print(f"  ...")
print(f"  W[{N-1},{N-1}] = λ^1   = {weights[-2]:.6f}")
print(f"  W[{N},{N}] = λ^0   = {weights[-1]:.6f}  (most recent)")
print()
print("In the global OLS model, Σ = σ²·I (identity), so all observations")
print("have equal weight. In the local WLS model, Σ = σ²·W⁻¹, meaning")
print("older observations have larger variance (less certainty) and thus")
print("contribute less to the parameter estimates.")
print()

# ── 4.2 Plot λ-weights vs time ────────────────────────────────────────
print("=== 4.2 λ-Weights Plot ===")

x = train["x"].values

plt.figure(figsize=(10, 5))
plt.plot(x, weights, "k.-", markersize=6)
plt.xlabel("Time (year)")
plt.ylabel("Weight λ^(N−i)")
plt.title("4.2 — WLS weights (λ = 0.9) vs. time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_4_2.png", dpi=150)
plt.close()

max_idx = np.argmax(weights)
print(f"Highest weight: {weights[max_idx]:.4f} at time x = {x[max_idx]:.3f}")
print(f"  → The most recent observation (Dec 2023) has the highest weight.")
print()

# ── 4.3 Sum of λ-weights ──────────────────────────────────────────────
print("=== 4.3 Sum of λ-Weights ===")

weight_sum = np.sum(weights)
weight_sum_formula = (1 - lam**N) / (1 - lam)
print(f"Sum of λ-weights = Σ λ^(N-i) = (1 - λ^N)/(1 - λ) = {weight_sum_formula:.4f}")
print(f"  (numerical check: {weight_sum:.4f})")
print(f"Corresponding OLS sum of weights = N = {N}")
print(f"  → The effective sample size in WLS ({weight_sum:.1f}) is much smaller than")
print(f"    in OLS ({N}), because older observations are down-weighted.")
print()

# ── 4.4 WLS parameter estimates ───────────────────────────────────────
print("=== 4.4 WLS Parameter Estimates (λ = 0.9) ===")

# θ̂_WLS = (X'WX)^{-1} X'Wy
theta_wls = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
print(f"θ̂₁ (intercept) = {theta_wls[0]:.6f}")
print(f"θ̂₂ (slope)     = {theta_wls[1]:.6f}")

# WLS residuals and sigma^2
y_hat_wls = X @ theta_wls
resid_wls = y - y_hat_wls
sigma2_wls = np.sum(weights * resid_wls**2) / (np.sum(weights) - 2)
print(f"σ²_WLS          = {sigma2_wls:.6e}")
print(f"σ_WLS           = {np.sqrt(sigma2_wls):.6f}")
print()
print(f"Comparison with OLS:")
print(f"  OLS:  θ̂₁ = {theta_hat[0]:.6f},  θ̂₂ = {theta_hat[1]:.6f}")
print(f"  WLS:  θ̂₁ = {theta_wls[0]:.6f},  θ̂₂ = {theta_wls[1]:.6f}")
print()

# ── 4.5 Forecast and comparison plot ──────────────────────────────────
print("=== 4.5 Forecast for Next 12 Months ===")

# Predictions
pred_ols = X_test @ theta_hat
pred_wls = X_test @ theta_wls

# Prediction intervals for OLS
C_ols = np.linalg.inv(X.T @ X)
pred_var_ols = sigma2 * (1 + np.array([xrow @ C_ols @ xrow for xrow in X_test]))
pred_se_ols = np.sqrt(pred_var_ols)

# Prediction intervals for WLS
C_wls = np.linalg.inv(X.T @ W @ X)
pred_var_wls = sigma2_wls * (1 + np.array([xrow @ C_wls @ xrow for xrow in X_test]))
pred_se_wls = np.sqrt(pred_var_wls)

# 95% prediction intervals (approx z=1.96)
z = 1.96

plt.figure(figsize=(10, 5))
# Training data
plt.plot(x, y, "k.", markersize=6, label="Training data")
# Test data
plt.plot(x_test, y_test, "kx", markersize=7, label="Test data")
# OLS forecast
plt.plot(x_test, pred_ols, "b-", linewidth=1.5, label="OLS forecast")
plt.fill_between(x_test, pred_ols - z * pred_se_ols, pred_ols + z * pred_se_ols,
                 color="blue", alpha=0.15, label="OLS 95% PI")
# WLS forecast
plt.plot(x_test, pred_wls, "r-", linewidth=1.5, label="WLS forecast (λ=0.9)")
plt.fill_between(x_test, pred_wls - z * pred_se_wls, pred_wls + z * pred_se_wls,
                 color="red", alpha=0.15, label="WLS 95% PI")

plt.xlabel("Time (year)")
plt.ylabel("Total vehicles (millions)")
plt.title("4.5 — OLS vs WLS (λ=0.9) forecasts for 2024")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_4_5.png", dpi=150)
plt.close()

# RMSE comparison
rmse_ols = np.sqrt(np.mean((y_test - pred_ols)**2))
rmse_wls = np.sqrt(np.mean((y_test - pred_wls)**2))
print(f"OLS forecast RMSE: {rmse_ols:.6f} million vehicles")
print(f"WLS forecast RMSE: {rmse_wls:.6f} million vehicles")
print()
if rmse_wls < rmse_ols:
    print("→ WLS (λ=0.9) produces better forecasts, as it adapts to the")
    print("  more recent trend rather than fitting the entire history equally.")
else:
    print("→ OLS produces better forecasts in this case.")
print()
print("Plots saved: plot_4_2.png, plot_4_5.png")

# ══════════════════════════════════════════════════════════════════════
# Section 5: Recursive Estimation and Optimization of Lambda
# ══════════════════════════════════════════════════════════════════════


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


def wls_estimate(y, X, lam):
    """Batch WLS estimate for comparison with RLS."""
    N = len(y)
    weights = np.array([lam ** (N - i) for i in range(1, N + 1)])
    W = np.diag(weights)
    return np.linalg.solve(X.T @ W @ X, X.T @ W @ y)


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
print(f"OLS:              θ̂₁ = {theta_hat[0]:.6f},  θ̂₂ = {theta_hat[1]:.8f}")
print(f"Difference:       Δθ̂₁ = {abs(theta_rls_N[0] - theta_hat[0]):.6e},  "
      f"Δθ̂₂ = {abs(theta_rls_N[1] - theta_hat[1]):.6e}")

# Re-run with progressively smaller R0 to reduce prior influence
for r0_scale in [1e-6, 1e-9]:
    R0_small = r0_scale * np.eye(2)
    thetas_small = rls(y, X, lam=1.0, R0=R0_small, theta0=theta0)
    theta_N = thetas_small[-1]
    print(f"\nRLS  (R0={r0_scale:.0e}·I): θ̂₁ = {theta_N[0]:.6f},  θ̂₂ = {theta_N[1]:.8f}")
    print(f"OLS:                θ̂₁ = {theta_hat[0]:.6f},  θ̂₂ = {theta_hat[1]:.8f}")
    print(f"Difference:         Δθ̂₁ = {abs(theta_N[0] - theta_hat[0]):.6e},  "
          f"Δθ̂₂ = {abs(theta_N[1] - theta_hat[1]):.6e}")

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

# Run RLS for each lambda
rls_results = {}
for lam in lambdas_54:
    thetas_lam = rls(y, X, lam=lam, R0=R0, theta0=theta0)
    rls_results[lam] = thetas_lam
    theta_wls_cmp = wls_estimate(y, X, lam)
    print(f"λ = {lam}:")
    print(f"  RLS θ̂_N = [{thetas_lam[-1, 0]:.6f}, {thetas_lam[-1, 1]:.8f}]")
    print(f"  WLS θ̂   = [{theta_wls_cmp[0]:.6f}, {theta_wls_cmp[1]:.8f}]")
    delta1 = abs(thetas_lam[-1, 0] - theta_wls_cmp[0])
    delta2 = abs(thetas_lam[-1, 1] - theta_wls_cmp[1])
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
plt.axhline(theta_hat[0], color="gray", linestyle="--", linewidth=1, label="OLS")
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
plt.axhline(theta_hat[1], color="gray", linestyle="--", linewidth=1, label="OLS")
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
pred_ols_57 = X_test @ theta_hat

# WLS predictions (lambda = 0.9, same as Part 4)
theta_wls_09 = wls_estimate(y, X, 0.9)
pred_wls_57 = X_test @ theta_wls_09

# RMSE comparison
rmse_ols_test = np.sqrt(np.mean((y_test - pred_ols_57)**2))
rmse_wls_test = np.sqrt(np.mean((y_test - pred_wls_57)**2))
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
plt.plot(x_test, pred_ols_57, "b-o", markersize=4, linewidth=1.5, label="OLS")
plt.plot(x_test, pred_wls_57, "g-s", markersize=4, linewidth=1.5, label="WLS (λ=0.9)")
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
