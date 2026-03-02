# 02417 Time Series Analysis — Spring 2026

## Overview

**Book:** [Time Series Analysis (PDF)](https://www2.imm.dtu.dk/~hm/time.series.analysis/)
**Sample Solutions:** [Solutions](https://www2.imm.dtu.dk/~hm/time.series.analysis/) *(from the book website, some corrected)*

---

## Weekly Schedule

### Week 1 — Introduction

**Book**
- Chapter 1: Introduction
- Chapter 2 (not 2.8): Multivariate random variables

**Exercises**
- 2.1, 2.2

**Simulation examples**
- 2.3

---

### Week 2 — Regression Based Methods, 1st Part

> The General Linear Model (GLM): Ordinary Least Squares (OLS) and Maximum Likelihood (ML) estimation. Recursive Least Squares (RLS) and global trend models, as well as some principles needed to find a good model, such as model selection.

**Book**
- 3.1: Introduction
- 3.2: The GLM, including OLS- and ML-estimates *(skip Section 3.2.1.4 on WLS)*
- 3.3: Prediction in the GLM
- 11.1 (with introduction): RLS

**Additional material**
- `ModellingReference.pdf` — slides about GLM and RLS (without forgetting)
- `ModelExamples.pdf` — slides about the global trend models

**Exercises**
- 3.1 *(try solving by hand as in the book, then check with a computer)*
- 3.4
- You can start working on **Assignment 1**: Sections 1–3 are covered

---

### Week 3 — Regression Based Methods, 2nd Part

**Book**
- 3.2.1.4: WLS
- 11.1.1: RLS with forgetting

**Additional material**
- More on local trend models

**Exercises**
- Work on **Assignment 1**: Sections 4–5 (local trend models using WLS and RLS)

---

### Week 4 — Introduction to Stochastic Processes, Operators and Linear Systems

**Book**
- 4.5: Shift operators *(for understanding 5.3)*
- 5.1 and 5.2: Stochastic processes in general
- 5.3 *(only slightly touch 5.3.2)*: Linear processes

**Exercises**
- 5.1 (for c ≠ 0) *(Q2: see page 117 for MA(1) process)*
- 5.4
- 5.7

---

### Week 5 — AR, MA and ARMA Processes

**Book**
- 5.5 *(disregard spectra like (5.67), (5.72), (5.85), (5.86), (5.112))*: MA, AR, and ARMA processes
- 5.3.2: Cursory material
- 5.6: Non-stationary models
- 5.7: Optimal Prediction
- 6.4: Estimation of parameters in ARMA models

**Exercises**
- Identification game
- 5.5
- 5.6 *(assume the process is stationary and invertible)*
- 5.10 *(if time allows — find a recursion for γ(k) for k > 2; skip Q3)*

---

### Week 6 — ACF and PACF with a Focus on Model Order Selection

> Identification of univariate time series models, 1st part.

**Book**
- 6.1 (with intro): Introduction
- 6.2.1 (and the introduction to 6.2, Sec. 6.2.1(a)): Estimation of auto-covariance and -correlation
- 6.3 (not 6.3.3): Using the SACF and SPACF for model order selection
- 6.5: Model order selection
- 6.6: Model validation

**Exercises**
- ARIMA model identification
- 6.1
- 6.6
- Carry on with **Assignment 2**

---

### Week 7 — Linear Systems

**Book**
- 4.1 (with intro): Linear Systems
- 4.4 *(disregard Theorem 4.10 and following example; Theorem 4.12 postponed to multivariate time series)*
- 6.2.2: Cross-correlation functions
- Chapter 8: Linear systems and stochastic processes

**Exercises**
- R `arima` with and without external regressor
- 4.1 (Q1–2)
- 8.2
- 6.6

---

### Week 8 — Multivariate Time Series

**Book**
- Chapter 9 *(read coarsely — the bivariate ARMA is good to think about; the rest extends univariate concepts to multi-variable VARMAX models)*

**Additional material**
- MARIMA: Spliid's method for parameter estimation in multi-variable ARMAX models
- Original paper: `Marima_paper` *(don't go into too much detail)*
- R package vignette: `Marima_vignette` *(again, don't go into details)*

**Exercises**
- Bivariate-output ARMAX exercise: `exercise_bivariate_waterheating`
- If time allows: 9.1 and its solutions *(theoretical ACF in a bivariate model)*

---

### Week 9 — TBD

*To be determined.*

---

### Week 10 — State Space Models, 1st Part

**Book**
- 10.1: The Linear Stochastic State Space Model
- 10.3: The Kalman filter

**Exercises**
- 10.1
- Kalman filter

---

### Week 11 — State Space Models, 2nd Part

**Book**
- 10.4 (not 10.4.1): ARMA models on state space form
- 10.6: ML estimates of state space models

**Exercises**
- Follow up on Kalman Filter exercise from last week *(estimate gravity in the "throw" model)*
- Estimation in state-space model
- 10.2
- 10.3

---

### Week 12 — Recursive and Adaptive Estimation

**Book**
- Chapter 11: Recursive and adaptive estimation

**Exercises**
- 10.4
- Get help on **Assignment 4**

---

### Week 13 — Final Lecture

> During the final lecture we will consider a number of real-world problems and discuss what tools from the course can be used to solve them.

**Exercises**
- Get help on **Assignment 4**
