# HR Employee Attrition Predictive Analytics — Technical Project Documentation

**Document Version:** 1.0  
**Audit Date:** June 18, 2026  
**Evidence Basis:** Exhaustive static analysis of all files in `D:\CDAC_PROJECT\3.PML_Project`  
**Hallucination Policy:** Every claim in this document is traceable to a specific file, line, or runtime inspection. Items not present in the codebase are explicitly marked as **NOT PRESENT**.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Inventory](#2-project-inventory)
3. [System Architecture](#3-system-architecture)
4. [End-to-End Data Flow](#4-end-to-end-data-flow)
5. [Module Reference (Inputs, Outputs, Logic)](#5-module-reference-inputs-outputs-logic)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Dataset Specification](#7-dataset-specification)
8. [API Contract](#8-api-contract)
9. [Frontend Architecture](#9-frontend-architecture)
10. [Technology Stack](#10-technology-stack)
11. [Configuration & Deployment](#11-configuration--deployment)
12. [Known Discrepancies & Edge Cases](#12-known-discrepancies--edge-cases)
13. [Operational Runbook](#13-operational-runbook)
14. [Senior Engineer Interview Questions](#14-senior-engineer-interview-questions)

---

## 1. Executive Summary

This project is an **HR Employee Attrition** analytics and prediction system built for a CDAC pair-programming / viva context. It predicts whether an employee is likely to leave (`Attrition = Yes`) or stay (`Attrition = No`) using a **Random Forest Classifier** trained on imbalanced employee HR data, with **SMOTE** applied during training to address class imbalance.

The system exposes **three independent client interfaces** over the same underlying model artifact (`model_data.joblib`):

| Interface | Entry Point | Transport |
|---|---|---|
| **Web Dashboard** | `index.html` + `script.js` | HTTP REST via Flask (`server.py`) |
| **Streamlit Portal** | `app_streamlit.py` | In-process Python (no HTTP API) |
| **Training Pipeline** | `train_model.py` | CLI script, writes `model_data.joblib` |

A Jupyter notebook (`full_finalcodemlproject.ipynb`) contains the original exploratory analysis, model comparison, and SMOTE evaluation that `train_model.py` was derived from.

**Claimed production accuracy:** 93.35% (referenced in `server.py`, `index.html`, `app_streamlit.py`, and computed as `0.933535` in the notebook's RF+SMOTE evaluation table).

---

## 2. Project Inventory

### 2.1 Application & Runtime Files

| File | Size (approx.) | Role |
|---|---|---|
| `server.py` | 5.7 KB | Flask REST API server |
| `train_model.py` | 3.7 KB | Model training & serialization script |
| `app_streamlit.py` | 18.3 KB | Streamlit interactive dashboard |
| `index.html` | 37.0 KB | Static HTML dashboard shell |
| `script.js` | 17.6 KB | Frontend logic, Chart.js, API calls |
| `style.css` | 14.8 KB | Frontend styling (755 lines) |
| `model_data.joblib` | 3.1 MB | Serialized model + preprocessors |
| `HR_Attrition.csv` | 303.7 KB | Source dataset |

### 2.2 Development & Documentation Artifacts

| File | Role | Code References |
|---|---|---|
| `full_finalcodemlproject.ipynb` | Original ML notebook (EDA, model comparison, SMOTE) | **None** — not imported by runtime |
| `HR_Attrition_Project_Explanation.pdf` | Project explanation document | **None** |
| `Ultimate_HR_Attrition_Project_Viva_Handbook.pdf` | Viva handbook | **None** |

### 2.3 NOT PRESENT in Codebase

The following are **absent** and were verified by filesystem scan:

- `requirements.txt` / `pyproject.toml` / `Pipfile`
- `package.json` / `node_modules`
- `Dockerfile` / `docker-compose.yml`
- CI/CD configuration (`.github/`, Jenkins, etc.)
- `.git/` repository
- `.env` / `.env.example`
- Unit test files (`test_*.py`, `*_test.py`)
- Authentication / authorization layer
- Database layer (all data is file-based)

### 2.4 Generated Artifacts (not source)

- `__pycache__/*.pyc` — Python bytecode cache
- `terminals/` — IDE terminal session logs

---

## 3. System Architecture

### 3.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HR ATTRITION PROJECT                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐         HTTP (port 5000)        ┌──────────────────┐  │
│  │  index.html      │ ──────────────────────────────► │   server.py      │  │
│  │  + style.css     │   GET  /health                  │   (Flask + CORS) │  │
│  │  + script.js     │   GET  /api/eda                 │                  │  │
│  │                  │   POST /predict                 │                  │  │
│  │  CDN: Chart.js   │                                 │        │         │  │
│  │  CDN: FontAwesome│                                 │        ▼         │  │
│  │  CDN: GoogleFonts│                          ┌──────┴──────────────┐  │  │
│  └──────────────────┘                          │  model_data.joblib  │  │  │
│                                                │  HR_Attrition.csv   │  │  │
│  ┌──────────────────┐                          └─────────────────────┘  │  │
│  │ app_streamlit.py │ ──── direct load ────────────────────────────────►│  │
│  │ (Streamlit UI)   │                                                    │  │
│  └──────────────────┘                                                    │  │
│                                                                             │
│  ┌──────────────────┐         writes                  ┌──────────────────┐  │
│  │  train_model.py  │ ──────────────────────────────► │ model_data.joblib│  │
│  │  (CLI training)  │         reads                   │                  │  │
│  └────────┬─────────┘ ◄───────────────────────────────┤ HR_Attrition.csv │  │
│           │                                           └──────────────────┘  │
│           │ derived from                                                    │
│           ▼                                                                 │
│  ┌──────────────────────────┐                                               │
│  │ full_finalcodemlproject  │  (standalone notebook, not executed at runtime) │
│  │        .ipynb            │                                               │
│  └──────────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Architectural Pattern

- **Pattern:** Multi-interface monolith with file-based persistence
- **Coupling:** `index.html` dashboard is **tightly coupled** to `server.py` via hardcoded `http://127.0.0.1:5000` URLs
- **Decoupling:** `app_streamlit.py` operates independently — loads model and CSV directly, no Flask dependency
- **Model serving:** Eager loading at process startup (`joblib.load` in `server.py` lines 12–21); failure to load causes `exit(1)`

### 3.3 Dual Frontend Strategy

| Aspect | HTML Dashboard | Streamlit App |
|---|---|---|
| Rendering | Client-side (browser) | Server-side (Python) |
| Charts | Chart.js (6 charts) | Matplotlib + Seaborn (10 plots) |
| Prediction | Via Flask `/predict` | In-process `pipeline.predict()` |
| EDA data | Via Flask `/api/eda` or JS fallback | Direct `pd.read_csv()` |
| Styling | Custom CSS (`style.css`) | Inline `<style>` block in Python |

---

## 4. End-to-End Data Flow

### 4.1 Training Flow (`train_model.py`)

```
HR_Attrition.csv
    │
    ▼
[1] pd.read_csv()
    │
    ▼
[2] drop_duplicates()                    → removes 97 duplicate rows (1749 → 1652)
    │
    ▼
[3] Drop columns:                        → EmployeeCount, EmployeeNumber, Over18,
    EmployeeCount, EmployeeNumber,          StandardHours, Education
    Over18, StandardHours, Education
    │
    ▼
[4] Separate X (29 features) / y         → y mapped: 'No'→0, 'Yes'→1
    │
    ▼
[5] SimpleImputer (median / most_freq)   → fits on full X before split
    │
    ▼
[6] LabelEncoder per categorical col     → 7 categorical columns encoded
    │
    ▼
[7] train_test_split(test_size=0.3,      → 70/30 stratified split
    stratify=y, random_state=42)
    │
    ▼
[8] SMOTE(random_state=42)               → applied ONLY to training split
    │                                      .fit_resample(X_train, y_train)
    ▼
[9] Pipeline.fit():                      → StandardScaler + RandomForestClassifier
    StandardScaler →                       (n_estimators=100, random_state=42)
    RandomForestClassifier
    │
    ▼
[10] joblib.dump() → model_data.joblib
```

### 4.2 Inference Flow — Flask API (`server.py` `/predict`)

```
JSON POST body (29 feature keys)
    │
    ▼
[1] Validate all feature_names present   → 400 if any missing
    │
    ▼
[2] Type coercion:                       → numeric_cols → float()
    numeric → float, categorical → str   → categorical → raw string
    │
    ▼
[3] pd.DataFrame([input_data])           → single-row DataFrame
    │
    ▼
[4] LabelEncoder.transform() per col     → fallback to classes_[0] on unknown
    │                                      ⚠ num_imputer / cat_imputer NOT applied
    ▼
[5] Reorder columns to feature_names
    │
    ▼
[6] pipeline.predict_proba()[0][1]       → P(Attrition=1)
    pipeline.predict()[0]                → class 0 or 1
    │
    ▼
[7] Rule-based recommendations           → if prob > 0.4, check OverTime,
    (OverTime, MonthlyIncome,              MonthlyIncome, WorkLifeBalance,
     WorkLifeBalance, JobSatisfaction)       JobSatisfaction thresholds
    │
    ▼
JSON response: {success, prediction, class, probability, recommendations}
```

### 4.3 Inference Flow — Streamlit (`app_streamlit.py`)

Identical preprocessing steps 1–6 as Flask, executed in-process when user clicks **"Calculate Attrition Risk"**. Display logic uses three probability tiers:

- `prob > 0.7` → "extremely high" recommendation
- `prob > 0.4` → "moderate" recommendation
- else → "healthy retention index"

**Note:** Streamlit does **not** generate the rule-based `recommendations` list that Flask produces.

### 4.4 EDA Data Flow

**Flask `/api/eda`:**
```
HR_Attrition.csv → read_csv → drop_duplicates → groupby aggregations → JSON
```

**HTML Dashboard:**
```
fetch('http://127.0.0.1:5000/api/eda')
    ├─ success → renderAllCharts(liveData)
    └─ failure → renderAllCharts(fallbackData)   ← hardcoded in script.js lines 36-68
```

**Streamlit EDA tab:**
```
HR_Attrition.csv → copy → drop_duplicates → median/mode fillna → 10 seaborn plots
```

---

## 5. Module Reference (Inputs, Outputs, Logic)

### 5.1 `train_model.py`

| Attribute | Detail |
|---|---|
| **Entry** | `python train_model.py` → calls `train_and_serialize_model()` |
| **Input** | `HR_Attrition.csv` (relative path, same directory) |
| **Output** | `model_data.joblib` (overwrites existing) |
| **Console Output** | Training progress logs, train/test accuracy percentages |

**Serialized `model_data` dictionary keys:**

| Key | Type | Used at Inference? |
|---|---|---|
| `pipeline` | `sklearn.pipeline.Pipeline` | ✅ Yes (server, streamlit) |
| `label_encoders` | `dict[str, LabelEncoder]` | ✅ Yes (server, streamlit) |
| `num_imputer` | `SimpleImputer` | ❌ No (saved but not loaded by server/streamlit) |
| `cat_imputer` | `SimpleImputer` | ❌ No (saved but not loaded by server/streamlit) |
| `numeric_cols` | `list[str]` (22 columns) | ✅ Yes (server type coercion) |
| `categorical_cols` | `list[str]` (7 columns) | ⚠ Stored but not explicitly used at inference |
| `feature_names` | `list[str]` (29 columns) | ✅ Yes (validation + column ordering) |
| `target_classes` | `['Retention', 'Attrition']` | ⚠ Loaded but not used in prediction logic |

### 5.2 `server.py`

| Route | Method | Input | Output | Status Codes |
|---|---|---|---|---|
| `/predict` | POST | JSON body with 29 features | `{success, prediction, class, probability, recommendations}` | 200, 400, 500 |
| `/api/eda` | GET | None | `{success, attrition_distribution, department_attrition, overtime_attrition, marital_attrition, job_role_attrition, wlb_attrition}` | 200, 500 |
| `/health` | GET | None | `{status: 'healthy', accuracy: 0.9335}` | 200 |

**Startup behavior:**
- Loads `model_data.joblib` at import time
- On load failure: prints error and `exit(1)` — server will not start
- Runs with `host='0.0.0.0'`, `port=5000`, `debug=True`

**Recommendation engine logic** (lines 66–80):

| Condition | Threshold | Message |
|---|---|---|
| `prob > 0.4` AND `OverTime == 'Yes'` | — | Overtime workload/bonus recommendation |
| `prob > 0.4` AND `MonthlyIncome < 4000` | default 5000 if missing | Compensation review |
| `prob > 0.4` AND `WorkLifeBalance >= 3` | default 3 | Schedule flexibility review |
| `prob > 0.4` AND `JobSatisfaction >= 3` | default 2 | Stay interview recommendation |
| `prob > 0.4` AND no rules matched | — | General engagement review |
| `prob <= 0.4` | — | Low risk, maintain policies |

### 5.3 `app_streamlit.py`

| Tab | Content | Data Source |
|---|---|---|
| **EDA** | 10 count/histogram plots with inference text | `HR_Attrition.csv` (median/mode imputed for viz only) |
| **Predictor** | 29-feature form → probability gauge | `model_data.joblib` pipeline |
| **Model Insights** | Before/After SMOTE metrics table + ASCII pipeline diagram | Hardcoded strings in Python (lines 350–376) |

**Caching:** `@st.cache_resource` on `load_assets()` — CSV and model loaded once per Streamlit session.

**Form input ranges** (verified from code):

| Feature | Widget | Range / Options |
|---|---|---|
| Age | slider | 18–60, default 35 |
| Gender | selectbox | Female, Male |
| MaritalStatus | selectbox | Single, Married, Divorced |
| Department | selectbox | R&D, Sales, HR |
| JobRole | selectbox | 9 roles |
| JobLevel | slider | 1–5 |
| BusinessTravel | selectbox | Travel_Rarely, Travel_Frequently, Non-Travel |
| DistanceFromHome | slider | 1–29 |
| EducationField | selectbox | 6 options |
| MonthlyIncome | number_input | 1000–100000, step 100 |
| DailyRate | slider | 100–1500 |
| HourlyRate | slider | 30–100 |
| MonthlyRate | slider | 2000–27000 |
| PerformanceRating | selectbox | 3, 4 only |
| All satisfaction/involvement | slider | 1–4 |
| TrainingTimesLastYear | slider | 0–6 |

### 5.4 `index.html`

| Section | ID | Content |
|---|---|---|
| Sidebar nav | `btn-eda`, `btn-predict`, `btn-model` | 3-tab navigation |
| EDA Dashboard | `tab-eda-content` | 3 stat cards + 6 chart canvases |
| Risk Predictor | `tab-predict-content` | 29-field form + result panel |
| Model Architecture | `tab-model-content` | Pipeline description + metrics comparison table |

**Form fields:** Mirror the 29 `feature_names` from the model. All use `name` attribute matching feature key for `FormData` collection in `script.js`.

**Static stat cards** (hardcoded, not API-driven):
- Total Records: 1,652
- Attritions: 396
- Retained: 1,256

### 5.5 `script.js`

| Functionality | Implementation |
|---|---|
| Tab switching | `switchTab(index)` toggles `.active` class |
| EDA charts | 6 Chart.js instances via `renderAllCharts(data)` |
| API health check | `fetch('http://127.0.0.1:5000/health')` on DOMContentLoaded |
| EDA data fetch | `fetch('http://127.0.0.1:5000/api/eda')` with `fallbackData` |
| Prediction | `fetch POST http://127.0.0.1:5000/predict` with JSON body |
| Random profile | `btn-random` randomizes all sliders and selects |
| Result UI | Conic-gradient gauge, risk badge, recommendations list |

**Chart types:**
1. `chart-attrition-dist` — doughnut
2. `chart-department-attr` — grouped bar
3. `chart-overtime-attr` — grouped bar
4. `chart-marital-attr` — grouped bar
5. `chart-job-role-attr` — horizontal bar (`indexAxis: 'y'`)
6. `chart-wlb-attr` — grouped bar

### 5.6 `style.css`

- **Theme:** Light mode, CSS custom properties (`:root` variables)
- **Layout:** Fixed 260px sidebar + fluid main content (`margin-left: 260px`)
- **Fonts:** Outfit (headings), Plus Jakarta Sans (body) — loaded via Google Fonts CDN
- **Color palette:** `--accent-blue: #2E5BFF`, `--success: #10b981`, `--error: #ef4444`
- **Components:** stat cards, chart cards, form groups, risk badges, circular progress gauge, metrics table
- **Animations:** `fadeIn` keyframe on tab switch (0.4s ease-out)
- **Responsive:** `@media (max-width: 1024px)` collapses sidebar (verified at line ~700+)

### 5.7 `full_finalcodemlproject.ipynb`

Standalone Jupyter notebook containing the complete ML experimentation workflow. **Not executed by any runtime component.**

**Notebook workflow (cell sequence):**

1. Import libraries (pandas, numpy, sklearn, imblearn, xgboost, matplotlib, seaborn)
2. Load CSV (`"Imperfect_HR_Attrition (1) (1).csv"` — **note:** filename differs from current `HR_Attrition.csv`)
3. `drop_duplicates()`
4. EDA visualizations (attrition distribution, department, age, income, etc.)
5. Drop constant/ID columns (same 5 columns as `train_model.py`)
6. Null imputation (median numeric, most_frequent categorical)
7. Label encoding per categorical column
8. `train_test_split(test_size=0.2, random_state=42)` — **note:** differs from `train_model.py` which uses `test_size=0.3` with `stratify=y`
9. Train 6 models BEFORE SMOTE: Logistic Regression, Random Forest, Decision Tree, KNN, XGBoost, SVM
10. Apply SMOTE to training data
11. Train 6 models AFTER SMOTE with same pipeline structure
12. Comparison tables and bar charts

**Notebook RF+SMOTE evaluation results (stored output in notebook):**

| Metric | Value |
|---|---|
| Accuracy | 0.933535 (93.35%) |
| Precision | 0.931034 |
| Recall | 0.835052 |
| F1 | 0.880435 |
| ROC AUC | 0.904705 |

---

## 6. Machine Learning Pipeline

### 6.1 Feature Engineering

**Dropped columns (5):** `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`, `Education`

**Rationale (from code comments):** "Re-implement exact column dropping from the notebook" — these are constant-value or identifier columns that provide no predictive signal.

**Final feature set (29):**

```
Age, BusinessTravel, DailyRate, Department, DistanceFromHome, EducationField,
EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole,
JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
```

**Categorical (7):** BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime

**Numeric (22):** All remaining features

### 6.2 Preprocessing Steps

| Step | Tool | Parameters | Applied When |
|---|---|---|---|
| Duplicate removal | `df.drop_duplicates()` | — | Training + EDA |
| Null imputation (numeric) | `SimpleImputer` | `strategy='median'` | Training only (full dataset before split) |
| Null imputation (categorical) | `SimpleImputer` | `strategy='most_frequent'` | Training only |
| Label encoding | `LabelEncoder` (per column) | `astype(str)` before fit | Training; transform at inference |
| Feature scaling | `StandardScaler` | default params | Inside pipeline, fit on SMOTE-resampled training data |
| Class balancing | `SMOTE` | `random_state=42` | Training split only (not test, not inference) |

### 6.3 Model Selection

**Selected model:** `RandomForestClassifier(n_estimators=100, random_state=42)`

**Why Random Forest (evidence from codebase):**

1. **Notebook comparison:** RF+SMOTE achieved highest accuracy (0.933535) among 6 post-SMOTE models evaluated in the notebook
2. **Ensemble robustness:** Referenced in `index.html` as "ensemble model of 100 individual decision trees"
3. **Class imbalance handling:** Combined with SMOTE, RF improved recall from 0.7835 (before SMOTE, per `app_streamlit.py` table) to 0.8351 (after SMOTE)
4. **Reproducibility:** Fixed `random_state=42` across SMOTE, split, and classifier

**Models evaluated in notebook but NOT deployed:**

| Model | Before SMOTE Accuracy (notebook) | After SMOTE Accuracy (notebook) |
|---|---|---|
| Logistic Regression | 0.815710 | 0.749245 |
| Random Forest | (trained) | **0.933535** ← selected |
| Decision Tree | 0.767372 | 0.725076 |
| KNN | (trained) | 0.731118 |
| XGBoost | 0.918429 | 0.915408 |
| SVM | 0.858006 | 0.854985 |

### 6.4 Class Imbalance Strategy

- **Problem:** 1256 No vs 396 Yes (76.0% / 24.0% after dedup) — confirmed by dataset inspection
- **Solution:** SMOTE (`imblearn.over_sampling.SMOTE`) applied only to `X_train, y_train`
- **Not used:** Class weights, undersampling, or threshold tuning (none present in code)

### 6.5 Serialization

```python
joblib.dump(model_data, "model_data.joblib")
```

**Model was trained with scikit-learn 1.5.1** (per `InconsistentVersionWarning` when loading with sklearn 1.8.0). The artifact remains functional but emits version warnings.

---

## 7. Dataset Specification

### 7.1 File: `HR_Attrition.csv`

| Property | Value |
|---|---|
| Raw shape | (1749, 35) |
| After `drop_duplicates()` | (1652, 35) |
| Total null values | 1204 |
| Duplicate rows | 97 |
| Target column | `Attrition` (values: `'No'`, `'Yes'`) |
| Raw class distribution | No: 1266, Yes: 483 |
| Deduped class distribution | No: 1256, Yes: 396 |

### 7.2 All 35 Columns

```
Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome,
Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction,
Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime,
PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours,
StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
```

### 7.3 Columns with Null Values

Confirmed present via `df.isnull().sum().sum() == 1204`. Specific per-column null counts are data-dependent; the dataset contains missing values in multiple numeric and categorical fields (visible in raw CSV rows with empty trailing fields).

### 7.4 Satisfaction / Balance Scale Convention

Documented inconsistently across files:

| Source | Convention |
|---|---|
| `index.html` line 154 | WLB: "1 = Best, 4 = Bad" |
| `app_streamlit.py` line 178 | WLB: "1=Best, 4=Bad" |
| `app_streamlit.py` line 187 | Job Satisfaction: "1=Very High, 4=Low" |
| `script.js` line 260 | WLB chart labels: `'1.0 (Bad)'` to `'4.0 (Best)'` — **inverted relative to HTML/Streamlit** |

---

## 8. API Contract

### 8.1 `POST /predict`

**Request:**
```json
{
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
}
```

**Success Response (200):**
```json
{
  "success": true,
  "prediction": 0,
  "class": "Retention",
  "probability": 0.07,
  "recommendations": ["Low risk. Maintain standard positive engagement policies."]
}
```

**Error Responses:**
- `400` — `{"error": "No input data provided"}` or `{"error": "Missing feature: <col>"}`
- `500` — `{"error": "<exception message>"}`

### 8.2 `GET /api/eda`

**Success Response (200):**
```json
{
  "success": true,
  "attrition_distribution": {"No": 1256, "Yes": 396},
  "department_attrition": {"Sales": {"No": 359, "Yes": 159}, ...},
  "overtime_attrition": {"No": {"No": 962, "Yes": 185}, "Yes": {"No": 294, "Yes": 211}},
  "marital_attrition": {...},
  "job_role_attrition": {...},
  "wlb_attrition": {"1.0": {"No": 51, "Yes": 40}, ...}
}
```

### 8.3 `GET /health`

```json
{
  "status": "healthy",
  "accuracy": 0.9335
}
```

### 8.4 CORS

`CORS(app)` applied globally with default permissive settings (all origins, all routes). No authentication headers required.

---

## 9. Frontend Architecture

### 9.1 HTML Dashboard Tab Structure

| Tab | Button ID | Section ID | Primary Interaction |
|---|---|---|---|
| EDA Dashboard | `btn-eda` | `tab-eda-content` | Auto-fetch EDA on load |
| Risk Predictor | `btn-predict` | `tab-predict-content` | Form submit → POST /predict |
| Model Architecture | `btn-model` | `tab-model-content` | Static content only |

### 9.2 CDN Dependencies (index.html)

| Resource | URL | Purpose |
|---|---|---|
| Google Fonts | `fonts.googleapis.com` | Outfit + Plus Jakarta Sans |
| Font Awesome 6.4.0 | `cdnjs.cloudflare.com` | Icons |
| Chart.js | `cdn.jsdelivr.net/npm/chart.js` | EDA visualizations |

### 9.3 Offline Resilience

`script.js` embeds complete `fallbackData` object (lines 36–68) with pre-computed aggregation statistics identical to deduped CSV values. Charts render immediately from fallback if Flask is unreachable. Prediction form **requires** live API — shows `alert()` on connection failure.

### 9.4 Result Panel UI Elements

| Element ID | Purpose |
|---|---|
| `risk-badge` | Text: "HIGH RISK OF ATTRITION" or "LOW RISK (RETENTION)" |
| `progress-circle` | Conic-gradient CSS gauge |
| `progress-val` | Percentage text |
| `rec-list` | Dynamic `<li>` elements from API recommendations |

---

## 10. Technology Stack

### 10.1 Backend (Python)

| Library | Version (installed env) | Usage | Rationale (from code evidence) |
|---|---|---|---|
| **Python** | 3.11 | Runtime | Verified via `__pycache__/cpython-311` |
| **Flask** | 3.1.0 | HTTP server (`server.py`) | Lightweight REST API for frontend communication |
| **flask-cors** | 6.0.2 | `CORS(app)` | Enables browser `fetch()` from `index.html` to `localhost:5000` |
| **pandas** | 3.0.3 | CSV I/O, DataFrame ops | Standard tabular data manipulation |
| **numpy** | 1.26.4 | dtype detection (`np.number`) | Feature type classification |
| **scikit-learn** | 1.8.0 (runtime) / 1.5.1 (training) | Pipeline, RF, Scaler, Imputer, Encoder, split | Core ML framework |
| **imbalanced-learn** | 0.14.1 | `SMOTE` oversampling | Address 76/24 class imbalance |
| **joblib** | 1.5.3 | Model serialization | Standard sklearn model persistence |
| **streamlit** | NOT INSTALLED (imported in code) | `app_streamlit.py` UI | Rapid interactive ML dashboard |
| **matplotlib** | 3.9.2 | `plt.subplots()` in Streamlit | Static plot rendering |
| **seaborn** | 0.13.2 | `sns.countplot()`, `sns.histplot()` | Statistical visualizations |

**Notebook-only dependencies (imported in `.ipynb`, not in production scripts):**
- `xgboost` — model comparison only
- `sklearn.neighbors.KNeighborsClassifier` — model comparison only
- `sklearn.svm.SVC` — model comparison only
- `sklearn.tree.DecisionTreeClassifier` — model comparison only
- `sklearn.linear_model.LogisticRegression` — model comparison only
- `sklearn.metrics` (accuracy_score, precision, recall, f1, roc_auc_score)

### 10.2 Frontend

| Technology | Usage |
|---|---|
| HTML5 | Semantic structure, form elements |
| CSS3 | Custom properties, flexbox, grid, animations, conic-gradient |
| Vanilla JavaScript (ES6+) | DOM manipulation, fetch API, FormData |
| Chart.js (CDN) | 6 responsive charts |
| Font Awesome 6.4.0 (CDN) | Icon system |
| Google Fonts (CDN) | Typography |

### 10.3 Data Storage

| Store | Format | Access Pattern |
|---|---|---|
| `HR_Attrition.csv` | CSV (flat file) | Read on each `/api/eda` request; cached in Streamlit |
| `model_data.joblib` | joblib binary | Eager load at process startup |

**No database, cache server, or message queue is used.**

---

## 11. Configuration & Deployment

### 11.1 Hardcoded Configuration Values

| Parameter | Value | Location |
|---|---|---|
| Flask host | `0.0.0.0` | `server.py:140` |
| Flask port | `5000` | `server.py:140` |
| Flask debug | `True` | `server.py:140` |
| API base URL | `http://127.0.0.1:5000` | `script.js` (3 fetch calls) |
| CSV filename | `HR_Attrition.csv` | `server.py`, `train_model.py`, `app_streamlit.py` |
| Model filename | `model_data.joblib` | All Python modules |
| Train/test split | `test_size=0.3` | `train_model.py:56` |
| Split random state | `42` | `train_model.py`, `train_model.py` SMOTE, RF |
| RF estimators | `100` | `train_model.py:68` |
| SMOTE random state | `42` | `train_model.py:61` |
| Risk threshold | `0.4` | `server.py:68`, `app_streamlit.py:327` |
| High risk threshold | `0.7` | `app_streamlit.py:325` |
| Sidebar width | `260px` | `style.css:62` |

### 11.2 Startup Commands

```bash
# Train model (generates model_data.joblib)
python train_model.py

# Start Flask API (required for HTML dashboard predictions)
python server.py

# Start Streamlit app (independent interface)
streamlit run app_streamlit.py

# Open HTML dashboard (static file — open in browser)
# Requires server.py running for live data
```

### 11.3 Dependency Installation (inferred from imports — no requirements.txt exists)

```bash
pip install flask flask-cors pandas numpy scikit-learn imbalanced-learn joblib streamlit matplotlib seaborn
```

---

## 12. Known Discrepancies & Edge Cases

### 12.1 Imputer Not Applied at Inference

`train_model.py` fits and saves `num_imputer` and `cat_imputer`, but `server.py` and `app_streamlit.py` **never apply them** during prediction. Since the HTML/Streamlit forms always provide complete values (no nulls), this gap is masked for the current UI. API consumers sending null/missing numeric values would produce incorrect predictions.

### 12.2 Metrics Inconsistency Across Files

| Source | RF+SMOTE Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| `full_finalcodemlproject.ipynb` (computed) | 93.35% | 93.10% | 83.51% | 88.04% | **90.47%** |
| `index.html` / `app_streamlit.py` (hardcoded table) | 93.35% | 88.04% | 83.51% | 88.04% | **96.69%** |

The accuracy and recall match; precision and ROC AUC differ between notebook output and UI-displayed tables.

### 12.3 Train/Test Split Mismatch

| Source | `test_size` | `stratify` |
|---|---|---|
| `train_model.py` | 0.3 | `y` |
| `full_finalcodemlproject.ipynb` | 0.2 | not specified |

### 12.4 CSV Filename History

| File | CSV Reference |
|---|---|
| `train_model.py` (current) | `HR_Attrition.csv` |
| `full_finalcodemlproject.ipynb` | `"Imperfect_HR_Attrition (1) (1).csv"` |

The notebook has not been updated to reflect the current CSV filename.

### 12.5 LabelEncoder Unknown Category Handling

Both `server.py` (line 54) and `app_streamlit.py` (line 298) silently fall back to `le.classes_[0]` when an unknown categorical value is encountered. This produces a valid prediction but may not reflect the input intent.

### 12.6 Work-Life Balance Chart Label Inversion

`script.js` line 260 labels WLB as `'1.0 (Bad)'` through `'4.0 (Best)'`, while `index.html` and `app_streamlit.py` document `1=Best, 4=Bad`. The underlying data mapping is unchanged; only the chart labels are inconsistent.

### 12.7 scikit-learn Version Warning

`model_data.joblib` was pickled with sklearn 1.5.1. Loading with sklearn 1.8.0 emits `InconsistentVersionWarning` for StandardScaler, DecisionTreeClassifier, RandomForestClassifier, Pipeline, LabelEncoder, and SimpleImputer.

### 12.8 Flask Debug Mode in Production

`debug=True` enables the Werkzeug debugger and auto-reloader. No production configuration exists.

### 12.9 No Input Validation Beyond Feature Presence

`/predict` validates feature key presence and numeric type coercion, but does not validate value ranges (e.g., Age 18–60 enforced only in HTML form, not API).

### 12.10 EDA Endpoint Reads CSV on Every Request

`/api/eda` performs `pd.read_csv()` + `drop_duplicates()` + 6 `groupby` operations on every GET request. No caching layer.

---

## 13. Operational Runbook

### 13.1 Minimum Viable Startup (HTML Dashboard)

1. Ensure `model_data.joblib` and `HR_Attrition.csv` exist in project root
2. `python server.py`
3. Open `index.html` in a web browser
4. Verify sidebar shows green "API Server Connected" dot

### 13.2 Failure Modes

| Symptom | Cause | Code Behavior |
|---|---|---|
| Server exits immediately | `model_data.joblib` missing/corrupt | `exit(1)` in `server.py:21` |
| Charts show static data | Flask not running | `script.js` falls back to `fallbackData` |
| Prediction alert | Flask not running | `script.js:389` alert message |
| Streamlit error page | Missing CSV or model | `st.error()` + `st.stop()` |
| 400 on predict | Missing feature in JSON | `server.py:35` |
| 500 on predict | Invalid data type / model error | `server.py:91` |

### 13.3 Retraining Procedure

```bash
python train_model.py
# Overwrites model_data.joblib
# Restart server.py to load new model
```

---

## 14. Senior Engineer Interview Questions

*All questions are grounded in actual design decisions, gaps, or implementation details found in this codebase.*

### Architecture & System Design

1. **This project has two frontends (HTML+Flask and Streamlit) serving the same model. What are the trade-offs of this dual-interface approach versus a single API-first architecture? When would you consolidate?**

2. **`server.py` loads the model eagerly at import time and calls `exit(1)` on failure. How would you redesign this for zero-downtime model updates in production?**

3. **The HTML frontend hardcodes `http://127.0.0.1:5000` in three `fetch()` calls. How would you make this configurable across development, staging, and production environments without modifying JavaScript?**

4. **There is no `requirements.txt`. How would you introduce reproducible builds for this project, and what risks exist given the sklearn version mismatch (1.5.1 trained, 1.8.0 runtime)?**

### Machine Learning Pipeline

5. **SMOTE is applied only to the training split in `train_model.py`, but the imputers are fit on the entire dataset before the split. Is this a data leakage concern? Defend or critique the current approach.**

6. **`num_imputer` and `cat_imputer` are saved in `model_data.joblib` but never used during inference in `server.py`. What impact does this have on API predictions when callers send null values? How would you fix the inference pipeline?**

7. **The project uses `LabelEncoder` for categorical features. What problems arise when the model encounters unseen categories at inference time? The code falls back to `classes_[0]` — is this acceptable for an HR attrition system?**

8. **`train_model.py` uses `test_size=0.3` with `stratify=y`, while the notebook uses `test_size=0.2` without explicit stratification. How does this affect the claimed 93.35% accuracy figure's reproducibility?**

9. **Random Forest was selected over XGBoost (91.54% in the UI table, 91.54% in notebook post-SMOTE). Was this the right choice considering accuracy alone? What other metrics and operational factors should drive model selection here?**

10. **The model uses accuracy as the headline metric (displayed in sidebar, health endpoint, subtitles). Why is accuracy potentially misleading for this imbalanced dataset, and what metric would you report instead?**

### Data Engineering

11. **`HR_Attrition.csv` contains 1204 null values across 1749 rows. Walk through exactly how nulls are handled in training versus EDA versus inference. Where are the inconsistencies?**

12. **The `/api/eda` endpoint re-reads and re-aggregates the entire CSV on every GET request. At what data volume does this become a bottleneck, and what caching strategy would you implement?**

13. **`drop_duplicates()` removes 97 rows. Should duplicate detection happen at ingestion time rather than at query/training time? What risks exist if new duplicates are added to the CSV?**

### API & Security

14. **CORS is enabled globally with default permissive settings. What security implications does this have if `server.py` is deployed with `host='0.0.0.0'` on a public network?**

15. **There is no authentication on `/predict` or `/api/eda`. How would you add API key authentication without breaking the static HTML frontend's `fetch()` calls?**

16. **Flask runs with `debug=True`. What specific vulnerabilities does this introduce, and what configuration changes are needed for production deployment?**

17. **The `/predict` endpoint accepts arbitrary JSON with no range validation. How could an attacker or buggy client send out-of-distribution values, and how would you add input schema validation?**

### Frontend & UX

18. **`script.js` maintains a hardcoded `fallbackData` object. How do you ensure this fallback stays synchronized with the actual CSV as data changes? What is the maintenance risk?**

19. **The Work-Life Balance chart in `script.js` labels 1.0 as "Bad" and 4.0 as "Best", but `index.html` and `app_streamlit.py` state the opposite. How would you detect and prevent this kind of cross-interface labeling inconsistency?**

20. **The recommendation engine in `server.py` uses hardcoded thresholds (`prob > 0.4`, `MonthlyIncome < 4000`). Are these clinically/HR-meaningful? How would you derive data-driven thresholds from the training set?**

### Scalability & Operations

21. **This system serves predictions synchronously via Flask's development server. What changes are needed to handle 1000 concurrent prediction requests?**

22. **`model_data.joblib` is 3.1 MB. How would the deployment strategy change if the model grew to 500 MB with deep learning architectures?**

23. **If the HR team wants real-time predictions integrated into their SAP/Workday system, what integration pattern would you recommend given the current architecture?**

### Testing & Quality

24. **There are zero automated tests in this project. Design a minimal test suite covering the training pipeline, API contract, and prediction consistency between Flask and Streamlit.**

25. **How would you write a regression test that verifies the 93.35% accuracy claim remains stable after retraining with new data?**

### Model Governance & Ethics

26. **The model uses `Gender`, `Age`, and `MaritalStatus` as features. What ethical and legal concerns (e.g., GDPR, bias) should be addressed before deploying this in a real HR context?**

27. **The rule-based recommendations in `server.py` are not derived from SHAP values or feature importances. How would you make the recommendations model-explainable rather than rule-based?**

28. **Precision and ROC AUC values differ between the notebook (90.47% ROC) and the UI table (96.69% ROC). How would you establish a single source of truth for model metrics in a production ML system?**

---

## Appendix A: File Dependency Graph

```
HR_Attrition.csv
├── train_model.py → model_data.joblib
├── server.py (EDA endpoint)
└── app_streamlit.py

model_data.joblib
├── server.py (startup load)
└── app_streamlit.py (cached load)

index.html
├── style.css
├── script.js
│   └── server.py (HTTP API)
├── Chart.js (CDN)
├── Font Awesome (CDN)
└── Google Fonts (CDN)

full_finalcodemlproject.ipynb (isolated — no runtime dependencies)
HR_Attrition_Project_Explanation.pdf (isolated)
Ultimate_HR_Attrition_Project_Viva_Handbook.pdf (isolated)
```

## Appendix B: Verified Runtime Test Results

Tests performed during documentation audit (June 18, 2026):

| Test | Result |
|---|---|
| `py_compile` on all `.py` files | Pass |
| `joblib.load('model_data.joblib')` | Pass (with sklearn version warnings) |
| `GET /health` | `{"status": "healthy", "accuracy": 0.9335}` |
| `GET /api/eda` | `success: true`, attrition `{"No": 1256, "Yes": 396}` |
| `POST /predict` (sample payload) | `success: true`, class `Retention`, prob `0.07` |

---

*End of documentation. All statements verified against codebase at `D:\CDAC_PROJECT\3.PML_Project`.*