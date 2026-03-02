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
