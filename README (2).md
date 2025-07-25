
#  Financial Goal Prediction Using Behavioral and Investment Signals

**Can your financial behavior predict your success? This project builds a model to find out.**

---

##  Overview

This repository contains a full machine learning pipeline that predicts an individual’s likelihood of attaining their financial goals (`GOAL_scaled`). We bridge classical financial data with behavioral economics, aiming to improve predictive insights in personal finance using demographic, behavioral, and investment indicators.

---

##  Project Objectives

- Model the relationship between **behavioral traits**, **investment habits**, and **demographics** in predicting financial outcomes.
- Test whether behavioral indicators (like risk tolerance and planning behavior) significantly explain financial goal achievement.
- Evaluate model transparency, fairness, and deployment feasibility.

---

##  Why This Matters

Traditional financial models rarely account for **human behavior**. This project explores the integration of:
- **Behavioral signals** (e.g., `RA_total`, `PBC_total`, `BI_total`)
- **Financial habits** (e.g., `CASH`, `EQUITY`, `INSURANCE`)
- **Demographic profiles** (e.g., `AGE`, `EDUCATION`, `OCCUPATION`)

to uncover **behavior-driven prediction pipelines** that could inform future financial advisory platforms, credit scoring models, or fintech apps.

---

##  Dataset Summary

**Source**: Processed proprietary dataset  
**File**: `CODED_DATA_cleaned_transformed.csv`  
**Size**: 423 records × 41 features  

| Category           | Sample Columns                                               |
|--------------------|--------------------------------------------------------------|
| Demographics       | `AGE`, `GENDER`, `EDUCATION`, `MARITAL_STATUS`, `OCCUPATION` |
| Financial Products | `CASH`, `INSURANCE`, `EQUITY`, `GOLD`, `MUTUAL_FUNDS`        |
| Behavioral Scores  | `RA1-3`, `RP1-3`, `PBC1-4`, `BI1-2`, `FQPB`, `BEH`            |
| Aggregates         | `RA_total`, `RP_total`, `PBC_total`, `BI_total`              |
| Encoded/Scaled     | `*_scaled`, `*_encoded`                                      |
| Target             | `GOAL_scaled`                                                |

> All features are numeric and machine-learning-ready.

---

##  Modeling Strategy

###  Phase 1: Regression (current)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
```
Trained on `GOAL_scaled` as a continuous target. Early results suggest moderate predictive strength.

###  Phase 2: Classification (recommended)
Convert `GOAL_scaled` into binary or categorical buckets:
- **Binary**: Above or below population mean
- **Tertiles**: Low / Medium / High financial attainment

This increases interpretability for business decisions and public-facing tools.

---

##  ML Pipeline Summary

```python
# Load and split
df = pd.read_csv("CODED_DATA_cleaned_transformed.csv")
X = df.drop(columns=["GOAL_scaled"])
y = df["GOAL_scaled"]

# Train
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
print("R²:", r2_score(y_test, model.predict(X_test)))
```

> 📎 Next Steps:
> - Evaluate baseline models (e.g., Dummy Regressor)
> - Use SHAP for feature attribution
> - Track experiments via MLflow or Weights & Biases

---

## 🧪 Evaluation Metrics

| Metric       | Status      | Notes |
|--------------|-------------|-------|
| R² Score     |  Reported | Needs baseline for context |
| RMSE         |  Reported | Add MAE + visualization |
| Feature Importances |  Missing | Required for trust and deployment |
| Bias Testing |  Missing | Consider subgroup fairness |

---

##  Explainability & Fairness

| Tool  | Use Case              | Status |
|-------|------------------------|--------|
| SHAP  | Global & local explanations | 🔄 Planned |
| LIME  | Local interpretability       | 🔄 Planned |
| Bias Audit | Gender, age fairness check | 🔄 Needed |

>  Risk: Models trained on demographic behavior data may reflect or amplify societal biases unless audited properly.

---

##  Ethical Checklist

| Concern                         | Covered? | Comment |
|----------------------------------|----------|---------|
| Anonymized data                  |        | No direct identifiers present |
| Informed consent                 |        | Unclear from metadata |
| Bias and fairness mitigation     |        | Needs implementation |
| Explainability for end-users    |        | SHAP planned |
| Misuse prevention disclaimer     |       | Should be added before deployment |

---

##  Future Roadmap

1.  Add classification pipeline for clearer output
2.  Introduce SHAP & fairness dashboard
3.  Deploy model using Streamlit + API
4.  Publish model card with ethical analysis
5.  Compare RandomForest vs XGBoost vs Lasso

---

##  File Structure

```
financial-goal-prediction/
├── data/
│   └── CODED_DATA_cleaned_transformed.csv
├── notebooks/
│   └── model_training_script.ipynb
├── src/
│   └── train_model.py
├── app/
│   └── streamlit_app.py (🔄 planned)
├── README.md
├── requirements.txt
└── LICENSE
```

---

##  Author

**Elvis Oduor Ong'injo**  
Finance Specialist | Data Analyst | CRISP-DM Enthusiast  
[LinkedIn Profile →](https://www.linkedin.com/in/levi-oduor-004467308)

---

## 🚀 TL;DR

> Financial goals aren't just about income—they're shaped by behavior.  
> This model explores how you *think* and *plan* to predict how far you'll go.
