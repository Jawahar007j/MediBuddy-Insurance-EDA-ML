# MediBuddy-Insurance-EDA-ML
EDA and predictive modelling on insurance data — Python, Seaborn, Scikit-learn

# MediBuddy Insurance Data — EDA & Predictive Modelling

**Tools:** Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn) | Statistical Analysis  
**Domain:** Insurance Analytics | Healthcare Data | Business Intelligence  
**Dataset:** MediBuddy insurance policyholder data — personal details + pricing (merged across two sources)

---

## Project Overview

An end-to-end Exploratory Data Analysis (EDA) and predictive modelling project on insurance claim data — combining multi-source data merging, statistical hypothesis testing, business-driven analysis, and machine learning to answer real insurance business questions.

---

## Business Questions Answered

| # | Business Question | Key Finding |
|---|---|---|
| 1 | Does gender influence insurance policy cost? | Gender alone is NOT a significant constraint — smoking status dominates |
| 2 | What is the average company spend per policy? | ₹13,270 per policy on average |
| 3 | Should the company offer region-specific policies? | Regional charge variation exists — Southeast shows highest average costs |
| 4 | Do number of dependents affect claim amount? | Modest variation — not a primary cost driver |
| 5 | Does BMI help predict insurance claims? | Obese policyholders incur significantly higher charges |
| 6 | Is smoking status critical for policy decisions? | YES — smokers cost nearly 4x more than non-smokers (₹32,050 vs ₹8,434) |
| 7 | Does age create a barrier on insurance claimed? | Positive correlation — charges increase with age |
| 8 | Can the company extend discounts based on BMI/health status? | Normal BMI policyholders are low-risk — discount eligibility supported |

---

## Key Findings & Business Insights

### Smoking Status — Strongest Cost Driver
- Smokers average **₹32,050** vs non-smokers at **₹8,434** — nearly **4x higher**
- Smoking dominates cost regardless of gender or region
- **Recommendation:** Smoking status should be a mandatory underwriting variable

### Gender — Limited Business Impact
- Male average: ₹13,957 | Female average: ₹12,570
- t-test confirms statistical significance (p < 0.05) but **effect size is small**
- When controlling for smoking status, gender difference nearly disappears
- **Recommendation:** Gender alone should not constrain policy extension

### BMI Categories & Insurance Risk
- Obese policyholders show highest average charges
- BMI × Smoking interaction reveals compounded risk
- **Recommendation:** BMI-based tiered pricing or wellness discounts for Normal BMI holders

### Average Policy Spend Benchmark
- Company spends **₹13,270 per policy** on average
- Any segment significantly above this threshold represents elevated financial risk

---

## Analysis Performed

**Exploratory Data Analysis:**
- Age distribution | BMI distribution | Smoker count | Region-wise policy distribution
- Correlation heatmap (age, BMI, children, charges)
- Scatter plots: Age vs Charges | BMI vs Charges (with regression line)
- Violin plot & strip plot: Smoker vs Charges distribution

**Statistical Testing:**
- Independent t-test (gender vs charges) — p-value analysis
- Correlation matrix — age, BMI, children vs charges

**Machine Learning — Insurance Charge Prediction:**

| Model | R² Score | Notes |
|---|---|---|
| Linear Regression (Baseline) | — | Preprocessing pipeline with OneHotEncoding |
| Random Forest Regressor | — | Significantly higher R² than baseline |
| Tuned Random Forest (GridSearchCV) | Best | Hyperparameter tuning: n_estimators, max_depth, min_samples |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data Processing | Python — Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Statistical Testing | SciPy (ttest_ind) |
| Machine Learning | Scikit-learn (LinearRegression, RandomForestRegressor, GridSearchCV) |
| Data Sources | Two Excel files merged on Policy No. |

---

## Data Pipeline

```python
# Multi-source merge
df_personal = pd.read_excel("Medibuddy insurance data personal details.xlsx")
df_price    = pd.read_excel("Medibuddy Insurance Data Price.xlsx")
df_merged   = pd.merge(df_personal, df_price, on="Policy no.", how="inner")

# BMI categorisation
df['bmi_category'] = pd.cut(df['bmi'],
    bins=[0, 18.5, 25, 30, 100],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# ML Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])
```

---

## Files in This Repo

| File | Description |
|---|---|
| `EDA.py` | Full EDA + ML Python script |
| `REPORT - EDA - QUESTIONS & INSIGHTS.pdf` | Business insights report with visualisation interpretations |
| `Medibuddy insurance data personal details.xlsx` | Source dataset 1 |
| `Medibuddy Insurance Data Price.xlsx` | Source dataset 2 |

---

## Connect

[LinkedIn](https://linkedin.com/in/jawahar-ranganathan) | [GitHub](https://github.com/Jawahar007j)
