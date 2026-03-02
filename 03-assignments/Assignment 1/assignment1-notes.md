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
