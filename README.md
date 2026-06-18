# CDAC PML Project — HR Employee Attrition Prediction

Machine learning system that predicts employee attrition risk using a **Random Forest + SMOTE** pipeline trained on HR workforce data. Built as a CDAC pair-programming / viva project.

**Repository:** [github.com/priyansupattanaik/CDAC_pml_project](https://github.com/priyansupattanaik/CDAC_pml_project)

---

## Features

- **Exploratory Data Analysis (EDA)** — attrition trends by department, overtime, job role, marital status, and work-life balance
- **Real-time attrition prediction** — 29-feature employee profile → retention/attrition probability
- **Rule-based HR recommendations** — actionable retention steps based on risk drivers (Flask API)
- **Three interfaces** — HTML dashboard, Flask REST API, and Streamlit app

**Model accuracy (test set):** 93.35%

---

## Project Structure

```
CDAC_pml_project/
├── server.py                    # Flask REST API (port 5000)
├── train_model.py               # Training script → generates model_data.joblib
├── app_streamlit.py             # Streamlit interactive dashboard
├── index.html                   # Web dashboard (EDA + predictor + model info)
├── script.js                    # Frontend logic, Chart.js, API calls
├── style.css                    # Dashboard styling
├── model_data.joblib            # Serialized model + preprocessors
├── HR_Attrition.csv             # Source dataset (1749 rows, 35 columns)
├── full_finalcodemlproject.ipynb # Original ML notebook (EDA, model comparison)
├── PROJECT_DOCUMENTATION.md     # Full technical documentation
├── requirements.txt             # Python dependencies
├── HR_Attrition_Project_Explanation.pdf
└── Ultimate_HR_Attrition_Project_Viva_Handbook.pdf
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/priyansupattanaik/CDAC_pml_project.git
cd CDAC_pml_project
pip install -r requirements.txt
```

### 2. Run the Flask API (required for HTML dashboard)

```bash
python server.py
```

Server starts at `http://127.0.0.1:5000` with endpoints:
- `GET /health` — server status
- `GET /api/eda` — aggregated EDA statistics from CSV
- `POST /predict` — attrition prediction (JSON body with 29 features)

### 3. Open the HTML dashboard

Open `index.html` in a browser. The dashboard fetches live data from the Flask API. If the server is offline, EDA charts fall back to pre-computed statistics.

### 4. Run the Streamlit app (alternative interface)

```bash
streamlit run app_streamlit.py
```

Streamlit loads the model and CSV directly — no Flask server needed.

### 5. Retrain the model (optional)

```bash
python train_model.py
```

Reads `HR_Attrition.csv`, trains the pipeline, and overwrites `model_data.joblib`. Restart `server.py` after retraining.

---

## Machine Learning Pipeline

| Step | Method |
|---|---|
| Duplicate removal | `drop_duplicates()` (1749 → 1652 rows) |
| Column dropping | EmployeeCount, EmployeeNumber, Over18, StandardHours, Education |
| Null imputation | Median (numeric), most frequent (categorical) |
| Encoding | LabelEncoder per categorical column (7 cols) |
| Train/test split | 70/30, stratified, `random_state=42` |
| Class balancing | SMOTE on training split only |
| Model | StandardScaler → RandomForestClassifier (`n_estimators=100`) |
| Serialization | `joblib.dump()` → `model_data.joblib` |

**Features used (29):** Age, BusinessTravel, DailyRate, Department, DistanceFromHome, EducationField, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

**Target:** Attrition (`No` → Retention, `Yes` → Attrition)

---

## API Example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Sales",
    "DistanceFromHome": 9,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 2,
    "Gender": "Male",
    "HourlyRate": 65,
    "JobInvolvement": 2,
    "JobLevel": 2,
    "JobRole": "Sales Executive",
    "JobSatisfaction": 2,
    "MaritalStatus": "Married",
    "MonthlyIncome": 6500,
    "MonthlyRate": 14000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3
  }'
```

**Response:**

```json
{
  "success": true,
  "prediction": 0,
  "class": "Retention",
  "probability": 0.07,
  "recommendations": ["Low risk. Maintain standard positive engagement policies."]
}
```

---

## Dataset

| Property | Value |
|---|---|
| File | `HR_Attrition.csv` |
| Raw rows | 1,749 |
| After deduplication | 1,652 |
| Columns | 35 |
| Null values | 1,204 |
| Class distribution (deduped) | No: 1,256 / Yes: 396 |

---

## Technology Stack

| Layer | Tools |
|---|---|
| Backend | Python 3.11, Flask, flask-cors |
| ML | scikit-learn, imbalanced-learn (SMOTE), joblib |
| Data | pandas, numpy |
| Streamlit UI | streamlit, matplotlib, seaborn |
| Web UI | HTML5, CSS3, Vanilla JavaScript, Chart.js |
| CDN | Google Fonts, Font Awesome 6.4.0 |

---

## Documentation

For exhaustive technical details — architecture, data flow, module I/O, edge cases, and interview questions — see **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**.

---

## Team / Author

**Priyansu Pattanaik** — [GitHub](https://github.com/priyansupattanaik)

CDAC PML (Predictive Machine Learning) Project