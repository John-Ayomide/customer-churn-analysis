# Executive Summary — Customer Churn Analysis

**Author:** John Ayomide  
**Date:** June 2026  
**Tools:** Python, XGBoost, SHAP, Streamlit

---

## Business Problem

A telecom company is experiencing a customer churn rate of 26.5% —
nearly double the industry average of 10-15%. This represents
significant recurring revenue loss. The objective was to identify
the key drivers of churn and build a predictive tool to flag
at-risk customers before they leave.

---

## Key Findings

**1. Contract type is the strongest predictor**
Month-to-month customers churn at 42.7% — almost 1 in 2.
Two-year contract customers show dramatically better retention.

**2. Early tenure is the critical danger window**
Churn is highest in the first 12 months. Approximately 600
customers churned within their first few months — the largest
concentration in the dataset.

**3. Pricing sensitivity is real**
Monthly charges and fibre optic internet service both appear
in the top 5 SHAP churn drivers — suggesting price is a
significant factor in churn decisions.

---

## Model Performance

| Metric                        | Score              |
| ----------------------------- | ------------------ |
| Accuracy                      | 79.1%              |
| AUC-ROC                       | 0.841              |
| Churners correctly identified | 195 of 373 (52.3%) |
| F1 Score                      | 0.570              |

---

## Business Recommendations

1. **Convert month-to-month customers** to annual contracts
   through targeted incentives — this addresses the highest
   risk segment directly

2. **Launch an early tenure programme** — dedicated onboarding,
   loyalty rewards, and check-in calls for customers in months
   1 through 12

3. **Review fibre optic pricing** — fibre customers show
   disproportionately high churn, suggesting price or service
   quality issues

4. **Deploy the predictive model** operationally — flag
   high-risk customers monthly for proactive retention outreach

---

## Deliverables

- 5 Jupyter notebooks with full methodology documentation
- 5 professional visualisations
- Trained XGBoost model with SHAP explainability
- Interactive Streamlit dashboard (live)
- Full GitHub repository with README

---

_Built as part of a data analytics portfolio targeting
football/sports analytics and general business intelligence roles._
