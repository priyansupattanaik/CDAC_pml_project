# HR Employee Attrition Prediction Project

## Project Overview

This project predicts employee attrition using the `Balanced_HR_Employee_Attrition.csv` dataset. It contains two user interfaces:

- `streamlit_app.py`: Python Streamlit dashboard with EDA, model training, metrics, and prediction.
- `hr_attrition_dashboard.html`: One-page HTML dashboard with themed UI, interactive analytics filters, charts, dataset preview, and browser-side prediction.

## Dataset

Data file: `Balanced_HR_Employee_Attrition.csv`

Target column:

- `Attrition`: `Yes` means the employee left, `No` means the employee stayed.

Important features used in analysis and prediction include:

- Demographics: `Age`, `Gender`, `MaritalStatus`, `EducationField`
- Work profile: `Department`, `JobRole`, `BusinessTravel`, `OverTime`
- Compensation: `MonthlyIncome`, `DailyRate`, `HourlyRate`, `PercentSalaryHike`
- Experience: `TotalWorkingYears`, `YearsAtCompany`, `YearsInCurrentRole`
- Satisfaction: `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`

## Streamlit Dashboard

Run:

```powershell
streamlit run streamlit_app.py
```

Main features:

- Dataset summary metrics
- Six EDA graphs before model training
- Model selector: Random Forest, Gradient Boosting, Logistic Regression
- Train/test split control
- Accuracy, precision, recall, ROC AUC
- Confusion matrix and feature importance
- Employee attrition prediction form

## HTML Dashboard

Run a local static server from this folder:

```powershell
python -m http.server 8010 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8010/hr_attrition_dashboard.html
```

Main features:

- One-page simplified themed dashboard
- Four KPI cards
- Interactive filters for department, job role, overtime, and attrition
- Six EDA charts that update when filters change
- Dataset preview that updates with filters
- Working browser-side attrition prediction form
- Live risk score, clear risk level, and top risk drivers

## Prediction Logic

The Streamlit app trains actual machine-learning models using scikit-learn.

The HTML dashboard uses a browser-side risk scoring method derived from dataset patterns. It calculates a baseline attrition rate and adjusts risk based on selected categorical patterns and numeric conditions such as overtime, job role, income, commute distance, tenure, stock options, travel, work-life balance, and satisfaction. The result updates immediately when any prediction input changes.

## Files

- `Balanced_HR_Employee_Attrition.csv`: Dataset
- `streamlit_app.py`: Streamlit ML application
- `hr_attrition_dashboard.html`: One-page HTML analytics and prediction dashboard
- `PROJECT_DOCUMENTATION.md`: Project documentation

## Notes

- Open the HTML dashboard through a local server, not by double-clicking the file, because the browser must fetch the CSV file.
- The HTML dashboard is intentionally self-contained with inline CSS and JavaScript.
- The Streamlit app provides the stronger ML workflow because it trains real models.
