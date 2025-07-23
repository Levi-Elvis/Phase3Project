


# Financial Goal Prediction Using Behavioral and Investment Indicators

This repository contains a complete data science project focused on predicting individual financial goal attainment using a dataset of behavioral economics indicators, financial investment habits, and demographic information. The goal is to model the `GOAL_scaled` variable—a normalized representation of an individual’s financial outcome—using structured machine learning pipelines.

---

## Project Purpose

In the modern financial ecosystem, behavioral and attitudinal variables significantly influence how individuals save, invest, and plan. This project attempts to bridge classical finance and behavioral signals through predictive modeling. Using the processed dataset provided, we evaluate the feasibility of predicting a person’s financial goal status (`GOAL_scaled`) using features like:

- Product ownership (e.g., CASH, EQUITY, INSURANCE)
- Risk perception (`RP1`–`RP3`)
- Risk attitude (`RA1`–`RA3`)
- Planning behavior (`PBC1`–`PBC4`)
- Behavioral intent and frequency (`BI1`, `BI2`, `FQPB`)
- Categorical demographics

---

## Dataset Summary

**Filename**: `CODED_DATA_cleaned_transformed.csv`  
**Observations**: 423  
**Features**: 41

### 🔍 Feature Breakdown

| Type               | Columns                                                                 |
|--------------------|-------------------------------------------------------------------------|
| Demographic        | `AGE`, `GENDER`, `MARITAL_STATUS`, `EDUCATION`, `OCCUPATION`           |
| Financial Products | `CASH`, `EQUITY`, `INSURANCE`, `FIXED_DEPOSIT`, `GOLD`, `MUTUAL_FUNDS` |
| Behavioral Scores  | `RA1`-`RA3`, `RP1`-`RP3`, `PBC1`-`PBC4`, `BI1`, `BI2`, `FQPB`, `BEH`     |
| Aggregated Scores  | `RA_total`, `RP_total`, `PBC_total`, `BI_total`                         |
| Scaled Variables   | `*_scaled` versions (used for ML stability)                             |
| Encoded Variables  | `GENDER_encoded`, `EDUCATION_encoded`, etc.                             |
| Target Variable    | `GOAL_scaled`                                                           |

All features are numeric (float/int), making the dataset ready for supervised modeling without additional transformations.

---

## Sample Rows

Example:
```
AGE: 3, GENDER: 2, EDUCATION: 4, CASH: 4, RA_total_scaled: 0.246, RP_total_scaled: 2.45, GOAL_scaled: -0.85
```

This implies the first respondent is middle-aged, highly exposed to risk perception, and currently below expected financial goal levels.

---

## Recommended Modeling Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load and split
df = pd.read_csv("CODED_DATA_cleaned_transformed.csv")
X = df.drop(columns=['GOAL_scaled'])
y = df['GOAL_scaled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("R² Score:", r2_score(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds, squared=False))
```

---

## Future Directions

- Build a classification version by thresholding `GOAL_scaled` (e.g., above or below mean).
- Train multiple models: XGBoost, Lasso, SVR.
- Apply hyperparameter tuning using `GridSearchCV`.
- Introduce LIME or SHAP for interpretability.
- Convert to Streamlit dashboard for interactive prediction.

---

## Ethical Considerations

- All data appears anonymized with no direct identifiers.
- Ensure use aligns with ethical data science practices, especially for financial modeling.

---

## File Structure

```
 financial-goal-prediction/
├── CODED_DATA_cleaned_transformed.csv      # Main dataset
├── README_FINANCIAL_GOAL_PROJECT.md        # This file
├── model_training_script.ipynb             # (recommended next step)
```

---

##  Author

**Elvis Oduor Ong'injo**  
Finance Specialist | Data Analyst | CRISP-DM Enthusiast  
[LinkedIn →](https://www.linkedin.com/in/levi-oduor-004467308)

---

##  TL;DR

> If behavior drives financial outcomes, this project models it.  
> Scalable. Extensible. Explainable. Financial analytics made smarter.

