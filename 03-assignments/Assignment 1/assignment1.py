import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from scipy import stats

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
