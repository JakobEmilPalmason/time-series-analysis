# Assignment 1 — Notes

## 1. Plot Data

### 1.1 Time variable and plot
- Time variable x: 2018-Jan = 2018.000, 2018-Feb = 2018.083, ..., 2023-Dec = 2023.917
- Total vehicles converted to millions (divided by 1e6)
- Training set: x < 2024 (72 months), Test set: x >= 2024 (12 months)
- Plot saved as `plot_1_1.png`

### 1.2 Description of the time series
- **Overall trend:** Clear upward trend from ~2.93M to ~3.23M vehicles over 2018–2023, an increase of roughly 300,000 vehicles (~10%)
- **Growth rate change:** Growth is roughly linear 2018–2020, then accelerates sharply in early 2021 (a jump of ~40,000 in one month), after which it levels off into a slower/plateauing trend from 2022 onward
- **Seasonality:** A mild seasonal pattern is visible — slight increases in spring/summer months and small dips in autumn/winter (likely corresponding to registration/deregistration cycles)
- **No obvious outliers** beyond the 2021 jump, which may reflect a change in registration methodology or a real surge in new registrations

## 2. Linear Trend Model

### 2.1 Matrix form (first 3 time points)

Model: Y_t = θ₁ + θ₂·x_t + ε_t

**Generic matrix form:**
```
Y = Xθ + ε
```

**With symbolic elements:**
```
⎡Y₁⎤   ⎡1  x₁⎤ ⎡θ₁⎤   ⎡ε₁⎤
⎢Y₂⎥ = ⎢1  x₂⎥ ⎢  ⎥ + ⎢ε₂⎥
⎣Y₃⎦   ⎣1  x₃⎦ ⎣θ₂⎦   ⎣ε₃⎦
```

**With actual values (3 digits):**
```
⎡2.930⎤   ⎡1  2018.000⎤ ⎡θ₁⎤   ⎡ε₁⎤
⎢2.934⎥ = ⎢1  2018.083⎥ ⎢  ⎥ + ⎢ε₂⎥
⎣2.941⎦   ⎣1  2018.167⎦ ⎣θ₂⎦   ⎣ε₃⎦
```

Each group member writes this by hand and includes a photo in the report.

## 3. OLS — Global Linear Trend Model

### 3.1 OLS estimation

Method: Minimize sum of squared residuals → closed-form solution θ̂ = (X'X)⁻¹X'y

Results:
- θ̂₁ (intercept) = **-110.355428**
- θ̂₂ (slope) = **0.056145** (millions of vehicles per year, i.e. ~56,100 vehicles/year)
- σ² = 6.828e-04
- σ = 0.02613

### 3.2 Parameter estimates with standard errors

| Parameter | Estimate | Std Error |
|-----------|----------|-----------|
| θ̂₁ (intercept) | -110.3554 | 3.5936 |
| θ̂₂ (slope) | 0.056145 | 0.001778 |

- Covariance matrix: Cov(θ̂) = σ² · (X'X)⁻¹, std errors are sqrt of diagonal
- Plot `plot_3_2.png`: training data with OLS regression line overlaid
- The line fits well in 2018–2020 but undershoots 2021 and overshoots 2022–2023 due to the non-linear growth pattern

### 3.3 Forecast table (2024, test set)

| Month | Predicted | Lower 95% | Upper 95% |
|-------|-----------|-----------|-----------|
| Jan | 3.2812 | 3.2276 | 3.3347 |
| Feb | 3.2858 | 3.2322 | 3.3395 |
| Mar | 3.2905 | 3.2368 | 3.3442 |
| Apr | 3.2952 | 3.2414 | 3.3489 |
| May | 3.2999 | 3.2460 | 3.3537 |
| Jun | 3.3045 | 3.2507 | 3.3584 |
| Jul | 3.3092 | 3.2553 | 3.3632 |
| Aug | 3.3139 | 3.2599 | 3.3679 |
| Sep | 3.3186 | 3.2645 | 3.3727 |
| Oct | 3.3233 | 3.2691 | 3.3774 |
| Nov | 3.3279 | 3.2737 | 3.3822 |
| Dec | 3.3326 | 3.2783 | 3.3869 |

- Prediction variance: σ² · (1 + x'(X'X)⁻¹x) for each test point
- 95% intervals use t-distribution with N-2 = 70 degrees of freedom

### 3.4 Forecast plot
- Plot `plot_3_4.png`: training data, OLS line extended, forecast (red triangles), actual test data (blue squares), and 95% prediction band
- The OLS forecast clearly overshoots — it predicts ~3.28–3.33M but actual values are ~3.22–3.26M

### 3.5 Comment on forecast
- Test RMSE = **0.0617** million vehicles (~61,700 vehicles)
- Mean absolute error = **0.0612** million
- The forecast is **not good** — the linear model extrapolates the average growth rate from 2018–2023, but the actual trend has flattened since 2022. The OLS model overestimates because it weights all historical data equally, including the steep growth in 2018–2021 which no longer represents recent dynamics.
- The actual test values fall mostly outside or near the edge of the 95% prediction intervals.

### 3.6 Residual diagnostics
Plot `plot_3_6.png` — four diagnostic plots:

- **Residuals vs time:** Clear systematic pattern — residuals are negative early (2018–2020), become positive around 2021, then turn negative again. This is not random scatter; it shows the linear model doesn't capture the changing growth rate.
- **Residuals vs fitted:** Same non-random pattern (curved), confirming model misspecification.
- **Q-Q plot:** Deviations from normality in the tails — the residuals are not perfectly normally distributed.
- **ACF plot:** Very high autocorrelation at all lags (ACF > 0.9 at lag 1), decaying slowly. This strongly violates the i.i.d. assumption — residuals are highly correlated over time.

**Conclusion:** The model assumptions are **not fulfilled**. The residuals show systematic structure (non-zero mean pattern), non-normality, and strong autocorrelation. A simple global linear trend is insufficient for this data.

## 5. Recursive Estimation and Optimization of λ

### 5.1 & 5.2 — RLS update equations and first iterations

Update equations:
- R_t = λ·R_{t-1} + x_t · x_t'
- θ̂_t = θ̂_{t-1} + R_t⁻¹ · x_t · (y_t - x_t' · θ̂_{t-1})

Initialized with R₀ = 0.1·I, θ₀ = [0, 0]', λ = 1 (no forgetting).

**R matrices (first 2 iterations):**
```
R₁ = [[1.1000,    2018.0000],
      [2018.0000, 4072324.1000]]

R₂ = [[2.1000,    4036.0833],
      [4036.0833, 8144984.4403]]
```

**Parameter estimates for t = 1, 2, 3:**

| t | θ̂₁ | θ̂₂ |
|---|-----|-----|
| 1 | 0.000001 | 0.001452 |
| 2 | 0.000000 | 0.001453 |
| 3 | -0.000004 | 0.001455 |

The estimates are very small initially because R₀ = 0.1·I gives a strong prior pulling toward θ₀ = 0. Each group member should do the first 2 R iterations by hand.

### 5.3 — RLS at t=N vs OLS

| Method | θ̂₁ | θ̂₂ |
|--------|-----|-----|
| RLS (R₀ = 0.1·I) | -0.058 | 0.00157 |
| RLS (R₀ = 1e-6·I) | -108.307 | 0.05513 |
| RLS (R₀ = 1e-9·I) | -110.353 | 0.05614 |
| OLS | -110.355 | 0.05614 |

- With R₀ = 0.1·I, RLS is far from OLS — the prior dominates even after 72 observations
- Shrinking R₀ progressively: Δθ̂₁ goes from 110 → 2.0 → 0.002
- R₀ = 1e-9·I effectively matches OLS to 3 decimal places
- R₀ acts as prior information; smaller R₀ = "we know less initially" → data dominates faster
- The ill-conditioned design matrix (x ≈ 2018) amplifies even tiny R₀ in the intercept

### 5.4 — RLS with forgetting (λ = 0.7 and λ = 0.99)

Plots `plot_5_4a.png` (intercept) and `plot_5_4b.png` (slope):

- **λ = 0.7** (blue): Very reactive — the slope spikes to ~0.09 during the 2021 growth surge, then drops to near-zero or negative as growth plateaus in 2022–2023. The intercept swings wildly (-180 to +30). This captures local dynamics but is noisy.
- **λ = 0.99** (red): Nearly flat, close to zero — behaves almost like OLS with slight adaptation. Very stable but slow to respond to regime changes.
- The OLS estimate (gray dashed) is a fixed horizontal line for comparison.

**RLS at t=N vs WLS comparison:**
- λ = 0.7: RLS ≈ WLS (Δθ̂₁ = 2.5e-4) — excellent match
- λ = 0.99: RLS ≠ WLS (large gap) — this is because R₀ = 0.1·I still has influence when λ is close to 1 (the forgetting is slow, so the prior doesn't wash out)

### 5.5 — One-step prediction residuals

Plot `plot_5_5.png`:
- **λ = 0.7:** Small residuals (±0.02M), roughly centered around zero. Adapts quickly. 1-step RMSE = **0.0149**
- **λ = 0.99:** Large, systematically positive residuals (0.02–0.15M), especially after 2021. The model is always "behind" because it adapts too slowly. 1-step RMSE = **0.0944**

### 5.6 — Optimal λ per prediction horizon

Plot `plot_5_6.png` — RMSE vs λ for k = 1, ..., 12:

| Horizon k | Optimal λ* | RMSE |
|-----------|-----------|------|
| 1 | 0.30 | 0.0067 |
| 2 | 0.30 | 0.0112 |
| 3 | 0.30 | 0.0164 |
| 4 | 0.46 | 0.0211 |
| 5 | 0.47 | 0.0255 |
| 6 | 0.48 | 0.0299 |
| 7 | 0.47 | 0.0343 |
| 8 | 0.47 | 0.0390 |
| 9 | 0.46 | 0.0439 |
| 10 | 0.45 | 0.0487 |
| 11 | 0.44 | 0.0532 |
| 12 | 0.42 | 0.0585 |

- Short horizons favor smaller λ (more forgetting, faster adaptation)
- Longer horizons favor slightly larger λ (~0.42–0.48) for stability
- All optimal λ values are well below 1, confirming the data has time-varying dynamics
- Yes, it makes sense to let λ depend on the horizon

### 5.7 — Test set predictions comparison

| Method | Test RMSE |
|--------|-----------|
| OLS | 0.0617 |
| WLS (λ = 0.9) | 0.0082 |
| RLS (λ* = 0.30, single) | 0.0079 |
| RLS (per-horizon λ*) | 0.0169 |

- OLS is the worst — overshoots significantly due to global averaging
- WLS and single-λ RLS perform similarly well (~0.008)
- Per-horizon RLS is slightly worse here because the very small λ values (0.30) lead to more volatile final estimates that don't extrapolate as stably over 12 months
- Plot `plot_5_7.png` shows OLS (blue) far above actual, while WLS/RLS track closely

### 5.8 — Reflections on time-adaptive models

- **Overfitting vs underfitting:** Small λ → short memory → tracks noise (overfitting). Large λ / OLS → long memory → misses regime changes (underfitting). The optimal λ balances these.
- **Test set challenges with time-dependent data:** Cannot shuffle or randomly split — must respect temporal ordering. Future data cannot be used to estimate past parameters. This makes traditional cross-validation invalid.
- **RLS and test sets:** One-step prediction residuals provide a natural form of sequential cross-validation — at each time step, we predict before observing, creating "out-of-sample" errors without a separate test set.
- **Alternative methods:** Kalman filter (state-space framework), exponential smoothing, ARIMA, sliding window regression, change-point detection models.
