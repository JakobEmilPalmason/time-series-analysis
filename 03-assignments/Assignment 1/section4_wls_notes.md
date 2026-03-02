# Assignment 1 — Section 4: WLS Local Linear Trend Model

## Overview

We now fit the same linear trend model Y_t = θ₁ + θ₂·x_t + ε_t, but using **Weighted Least Squares (WLS)** to make it *local* — recent observations matter more than older ones. The forgetting factor is λ = 0.9.

---

## 4.1 Variance-Covariance Matrix

The WLS weight for observation i (i = 1, …, N) is:

```
w_i = λ^(N−i)
```

This means the most recent observation (i = N) has weight λ⁰ = 1, while the oldest (i = 1) has weight λ^(N−1) = 0.9^71 ≈ 5.6 × 10⁻⁴.

The **weight matrix** W is N × N diagonal:

```
        ⎡λ^(N-1)   0      ⋯    0  ⎤
W  =    ⎢  0     λ^(N-2)  ⋯    0  ⎥
        ⎢  ⋮       ⋮      ⋱    ⋮  ⎥
        ⎣  0       0      ⋯    1  ⎦
```

Specifically for our 72 × 72 matrix:
- W[1,1] = 0.9^71 = 5.64 × 10⁻⁴ (oldest — Jan 2018)
- W[71,71] = 0.9^1 = 0.90
- W[72,72] = 0.9^0 = 1.00 (most recent — Dec 2023)

**Comparison with OLS:** In the global OLS model the variance-covariance matrix of the errors is Σ_OLS = σ²·I, so every observation has equal influence. In the local WLS model, Σ_WLS = σ²·W⁻¹, which means older observations have *larger* variance (less certainty) and contribute less to the fit. The model effectively "forgets" old data, focusing on the recent trend.

---

## 4.2 λ-Weights vs. Time

![λ-weights plot](plot_4_2.png)

The plot shows exponential growth of the weights toward the present. With λ = 0.9, data older than about 2 years (roughly 20 months back, since 0.9^20 ≈ 0.12) contributes very little.

**The most recent time point (Dec 2023, x = 2023.917) has the highest weight of 1.0.**

---

## 4.3 Sum of λ-Weights

The sum of all weights is a geometric series:

```
Σ w_i = Σ_{k=0}^{N-1} λ^k = (1 − λ^N) / (1 − λ) = (1 − 0.9^72) / 0.1 = 9.9949
```

For comparison, the OLS sum of weights is simply N = 72 (each observation has weight 1).

The **effective sample size** of the WLS model is ≈ 10, much smaller than 72. This means WLS is essentially fitting a line through only the most recent ~10 "equivalent" observations, making it far more responsive to recent changes in the trend.

---

## 4.4 WLS Parameter Estimates (λ = 0.9)

Using the WLS formula θ̂_WLS = (X'WX)⁻¹ X'Wy:

| Parameter | OLS | WLS (λ = 0.9) |
|-----------|-----|----------------|
| θ̂₁ (intercept) | −110.355428 | −52.482862 |
| θ̂₂ (slope) | 0.056145 | 0.027530 |
| σ² | 6.828 × 10⁻⁴ | 3.415 × 10⁻⁴ |
| σ | 0.02613 | 0.01848 |

**Interpretation:** The WLS slope (0.0275, i.e. ~27,500 vehicles/year) is much lower than the OLS slope (0.0561, i.e. ~56,100 vehicles/year). This makes sense: the recent trend (2022–2023) shows the growth rate slowing/plateauing, while OLS is heavily influenced by the steep increase around 2021. WLS captures the *current* local trend better.

---

## 4.5 Forecast for 2024 and Comparison

![OLS vs WLS forecast](plot_4_5.png)

Forecasting the next 12 months (Jan–Dec 2024):

| Metric | OLS | WLS (λ = 0.9) |
|--------|-----|----------------|
| RMSE | 0.0617 M | 0.0082 M |
| RMSE (vehicles) | ~61,700 | ~8,200 |

**The WLS model with λ = 0.9 produces dramatically better forecasts** — its RMSE is roughly 7.5× smaller than OLS.

**Why?** The OLS forecast extrapolates the average slope across the entire 2018–2023 period, which includes the steep 2020–2021 growth. This causes it to **overshoot** the actual 2024 values (blue line above the test data). The WLS model, by contrast, focuses on the recent leveling-off trend and produces a gentler slope that matches the 2024 reality much more closely (red line through the test data).

**Conclusion:** For this data, the WLS forecast (λ = 0.9) is clearly preferable, as the time series exhibits a changing growth rate. A global model is too rigid when the underlying dynamics shift over time.
