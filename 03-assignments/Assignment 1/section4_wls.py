"""Assignment 1 — Section 4: WLS Local Linear Trend Model (λ = 0.9)"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Data preparation (same as assignment1.py) ──────────────────────────
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

# OLS estimates (needed for comparison in 4.5)
theta_ols = np.linalg.solve(X.T @ X, X.T @ y)
y_hat_ols = X @ theta_ols
resid_ols = y - y_hat_ols
sigma2_ols = np.sum(resid_ols**2) / (N - 2)

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
print(f"  OLS:  θ̂₁ = {theta_ols[0]:.6f},  θ̂₂ = {theta_ols[1]:.6f}")
print(f"  WLS:  θ̂₁ = {theta_wls[0]:.6f},  θ̂₂ = {theta_wls[1]:.6f}")
print()

# ── 4.5 Forecast and comparison plot ──────────────────────────────────
print("=== 4.5 Forecast for Next 12 Months ===")

x_test = test["x"].values
y_test = test["total"].values
X_test = np.column_stack([np.ones(len(test)), x_test])

# Predictions
pred_ols = X_test @ theta_ols
pred_wls = X_test @ theta_wls

# Prediction intervals for OLS
C_ols = np.linalg.inv(X.T @ X)
pred_var_ols = sigma2_ols * (1 + np.array([xrow @ C_ols @ xrow for xrow in X_test]))
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
