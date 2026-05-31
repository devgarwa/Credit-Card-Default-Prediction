# Credit Card Default Prediction

Predicting next-month credit card default from financial, behavioral, and demographic data, with a risk-based threshold tuned for catching high-risk customers rather than chasing raw accuracy.

## Overview

Given an imbalanced dataset of ~25K credit card holders (30+ features), the goal is to flag customers likely to default next month so a lender can intervene early. Because missing a default is far costlier than a false alarm, the project optimizes for **recall on the default class** and uses an **F2-oriented** evaluation rather than accuracy.

## Approach

- **Feature engineering** — built risk-signal features on top of the raw bill/payment history:
  - `utilization_ratio` (average bill ÷ credit limit)
  - `max_delay`, `count_delays`, `on_time_pays` (payment-behavior aggregates)
  - `std_bill_amt` (bill volatility)
- **EDA & risk stratification** — default-rate analysis across utilization bins and payment-delay buckets, plus a utilization × delay risk heatmap to expose interaction effects.
- **Pipeline** — median imputation → standardization → **SMOTE** to address class imbalance → model training.
- **Models compared** — Logistic Regression, Random Forest, XGBoost, LightGBM, ranked by ROC-AUC.
- **Model selection** — XGBoost chosen on validation AUC.
- **Threshold tuning** — decision threshold optimized from the precision–recall curve (≈ 0.27) to prioritize catching defaulters.

## Results

| Metric | Value |
|---|---|
| Model | XGBoost (SMOTE-balanced) |
| F2-score | ~0.72 |
| Recall (default class) | ~75% |
| Decision threshold | ~0.27 (tuned, not default 0.5) |

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · imbalanced-learn (SMOTE) · Matplotlib · Seaborn

## Repository Structure

```
├── Submission_21410010.ipynb              # Full pipeline: EDA → FE → modeling → threshold tuning
├── Submission_21410010_ProjectReport.pdf  # Written project report
└── README.md
```

## Key Takeaway

Optimizing the decision threshold and the right metric (recall / F2) — not accuracy — is what makes an imbalanced-classification model actually useful for risk management.
