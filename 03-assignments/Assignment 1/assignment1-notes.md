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
