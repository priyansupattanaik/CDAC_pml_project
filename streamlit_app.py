from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="HR",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = pathlib.Path(__file__).with_name("Balanced_HR_Employee_Attrition.csv")
TARGET = "Attrition"
DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.7rem;}
    .hero {
        border: 1px solid #d8e2ee;
        border-radius: 8px;
        padding: 22px 24px;
        background: linear-gradient(135deg, #f7fbff 0%, #f6f1ff 48%, #fffaf2 100%);
        margin-bottom: 18px;
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 2.1rem;
        line-height: 1.15;
        color: #172033;
    }
    .hero p {
        color: #48556a;
        font-size: 1.02rem;
        margin: 0;
    }
    .section-note {
        color: #5d6878;
        margin-top: -8px;
        margin-bottom: 14px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e5eaf0;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = df.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in df.columns])
    numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = features.select_dtypes(exclude=np.number).columns.tolist()
    return numeric_cols, categorical_cols


def build_model(model_name: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    if model_name == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "Gradient Boosting":
        estimator = GradientBoostingClassifier(random_state=42)
    else:
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


@st.cache_resource(show_spinner=False)
def train_model(model_name: str, test_size: float) -> dict:
    df = load_data()
    numeric_cols, categorical_cols = split_columns(df)
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in df.columns])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    pipeline = build_model(model_name, numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, target_names=["No", "Yes"], output_dict=True)

    return {
        "pipeline": pipeline,
        "X": X,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "metrics": {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob),
        },
        "report": report,
    }


def attrition_rate(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        data.assign(AttritionFlag=(data[TARGET] == "Yes").astype(int))
        .groupby(group_col, as_index=False)
        .agg(AttritionRate=("AttritionFlag", "mean"), Employees=(TARGET, "size"))
        .sort_values("AttritionRate", ascending=False)
    )
    grouped["AttritionRate"] = grouped["AttritionRate"] * 100
    return grouped


def plot_eda(df: pd.DataFrame) -> None:
    st.subheader("Exploratory Data Analysis")
    st.markdown(
        '<p class="section-note">Six quick views of attrition patterns before model training.</p>',
        unsafe_allow_html=True,
    )

    graph_1, graph_2 = st.columns(2)
    with graph_1:
        target_counts = df[TARGET].value_counts().reset_index()
        target_counts.columns = [TARGET, "Employees"]
        fig = px.pie(
            target_counts,
            names=TARGET,
            values="Employees",
            hole=0.48,
            color=TARGET,
            color_discrete_map={"Yes": "#de425b", "No": "#2f9e80"},
            title="1. Overall Attrition Balance",
        )
        st.plotly_chart(fig, use_container_width=True)

    with graph_2:
        dept_rate = attrition_rate(df, "Department")
        fig = px.bar(
            dept_rate,
            x="Department",
            y="AttritionRate",
            color="Employees",
            text=dept_rate["AttritionRate"].round(1).astype(str) + "%",
            color_continuous_scale="Tealrose",
            title="2. Attrition Rate by Department",
        )
        fig.update_layout(yaxis_title="Attrition rate (%)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    graph_3, graph_4 = st.columns(2)
    with graph_3:
        fig = px.histogram(
            df,
            x="Age",
            color=TARGET,
            nbins=24,
            barmode="overlay",
            opacity=0.72,
            color_discrete_map={"Yes": "#de425b", "No": "#2f9e80"},
            title="3. Age Distribution by Attrition",
        )
        fig.update_layout(yaxis_title="Employees")
        st.plotly_chart(fig, use_container_width=True)

    with graph_4:
        fig = px.box(
            df,
            x=TARGET,
            y="MonthlyIncome",
            color=TARGET,
            points="outliers",
            color_discrete_map={"Yes": "#de425b", "No": "#2f9e80"},
            title="4. Monthly Income Spread by Attrition",
        )
        st.plotly_chart(fig, use_container_width=True)

    graph_5, graph_6 = st.columns(2)
    with graph_5:
        overtime_rate = attrition_rate(df, "OverTime")
        fig = px.bar(
            overtime_rate,
            x="OverTime",
            y="AttritionRate",
            color="OverTime",
            text=overtime_rate["AttritionRate"].round(1).astype(str) + "%",
            color_discrete_sequence=["#607d8b", "#d67236"],
            title="5. Overtime vs Attrition Rate",
        )
        fig.update_layout(yaxis_title="Attrition rate (%)", xaxis_title="Overtime")
        st.plotly_chart(fig, use_container_width=True)

    with graph_6:
        corr_cols = [
            "Age",
            "MonthlyIncome",
            "TotalWorkingYears",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "JobSatisfaction",
            "WorkLifeBalance",
            "DistanceFromHome",
        ]
        corr = df[corr_cols].corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="6. Correlation Heatmap of Key Numeric Features",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Optional Deep Dive")
    left, right = st.columns([1, 1])
    with left:
        group_col = st.selectbox(
            "Compare attrition rate by category",
            ["JobRole", "EducationField", "BusinessTravel", "MaritalStatus", "Gender"],
        )
        custom_rate = attrition_rate(df, group_col)
        fig = px.bar(
            custom_rate,
            x=group_col,
            y="AttritionRate",
            color="Employees",
            text=custom_rate["AttritionRate"].round(1).astype(str) + "%",
            color_continuous_scale="Bluered",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Attrition rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(
            custom_rate.style.format({"AttritionRate": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )


def feature_importance(model_result: dict) -> pd.DataFrame | None:
    pipeline = model_result["pipeline"]
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    clean_names = [name.split("__", 1)[-1].replace("_", " ") for name in names]
    return (
        pd.DataFrame({"Feature": clean_names, "Importance": estimator.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(15)
    )


def plot_model_results(model_result: dict) -> None:
    st.subheader("Model Training Results")
    metric_cols = st.columns(4)
    for column, (label, value) in zip(metric_cols, model_result["metrics"].items()):
        column.metric(label, f"{value:.3f}")

    left, right = st.columns([0.92, 1.08])
    with left:
        cm = confusion_matrix(model_result["y_test"], model_result["y_pred"])
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            colorscale="Blues",
            showscale=True,
        )
        fig.update_layout(title="Confusion Matrix", height=390)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        importance = feature_importance(model_result)
        if importance is not None:
            fig = px.bar(
                importance.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top Feature Importance",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            report_df = pd.DataFrame(model_result["report"]).T
            st.dataframe(report_df.round(3), use_container_width=True)


def prediction_form(df: pd.DataFrame, model_result: dict) -> None:
    st.subheader("Predict Employee Attrition")
    st.markdown(
        '<p class="section-note">Change employee details and generate an attrition risk score.</p>',
        unsafe_allow_html=True,
    )

    X = model_result["X"]
    numeric_cols = model_result["numeric_cols"]
    categorical_cols = model_result["categorical_cols"]
    defaults = X.median(numeric_only=True).to_dict()
    input_data: dict[str, object] = {}

    tab_core, tab_work, tab_satisfaction = st.tabs(["Core Profile", "Work Details", "Satisfaction"])

    with tab_core:
        c1, c2, c3 = st.columns(3)
        input_data["Age"] = c1.slider("Age", int(df["Age"].min()), int(df["Age"].max()), int(defaults["Age"]))
        input_data["Gender"] = c2.selectbox("Gender", sorted(df["Gender"].unique()))
        input_data["MaritalStatus"] = c3.selectbox("Marital Status", sorted(df["MaritalStatus"].unique()))
        input_data["Education"] = c1.slider("Education", 1, 5, int(defaults["Education"]))
        input_data["EducationField"] = c2.selectbox("Education Field", sorted(df["EducationField"].unique()))
        input_data["DistanceFromHome"] = c3.slider(
            "Distance From Home", int(df["DistanceFromHome"].min()), int(df["DistanceFromHome"].max()), int(defaults["DistanceFromHome"])
        )

    with tab_work:
        c1, c2, c3 = st.columns(3)
        input_data["Department"] = c1.selectbox("Department", sorted(df["Department"].unique()))
        input_data["JobRole"] = c2.selectbox("Job Role", sorted(df["JobRole"].unique()))
        input_data["BusinessTravel"] = c3.selectbox("Business Travel", sorted(df["BusinessTravel"].unique()))
        input_data["OverTime"] = c1.selectbox("Overtime", sorted(df["OverTime"].unique()))
        input_data["JobLevel"] = c2.slider("Job Level", 1, 5, int(defaults["JobLevel"]))
        input_data["MonthlyIncome"] = c3.number_input(
            "Monthly Income", int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max()), int(defaults["MonthlyIncome"]), step=250
        )
        input_data["DailyRate"] = c1.number_input("Daily Rate", int(df["DailyRate"].min()), int(df["DailyRate"].max()), int(defaults["DailyRate"]))
        input_data["HourlyRate"] = c2.number_input("Hourly Rate", int(df["HourlyRate"].min()), int(df["HourlyRate"].max()), int(defaults["HourlyRate"]))
        input_data["MonthlyRate"] = c3.number_input("Monthly Rate", int(df["MonthlyRate"].min()), int(df["MonthlyRate"].max()), int(defaults["MonthlyRate"]))
        input_data["TotalWorkingYears"] = c1.slider("Total Working Years", 0, int(df["TotalWorkingYears"].max()), int(defaults["TotalWorkingYears"]))
        input_data["YearsAtCompany"] = c2.slider("Years At Company", 0, int(df["YearsAtCompany"].max()), int(defaults["YearsAtCompany"]))
        input_data["YearsInCurrentRole"] = c3.slider("Years In Current Role", 0, int(df["YearsInCurrentRole"].max()), int(defaults["YearsInCurrentRole"]))
        input_data["YearsSinceLastPromotion"] = c1.slider(
            "Years Since Last Promotion", 0, int(df["YearsSinceLastPromotion"].max()), int(defaults["YearsSinceLastPromotion"])
        )
        input_data["YearsWithCurrManager"] = c2.slider(
            "Years With Current Manager", 0, int(df["YearsWithCurrManager"].max()), int(defaults["YearsWithCurrManager"])
        )
        input_data["NumCompaniesWorked"] = c3.slider(
            "Companies Worked", 0, int(df["NumCompaniesWorked"].max()), int(defaults["NumCompaniesWorked"])
        )

    with tab_satisfaction:
        c1, c2, c3 = st.columns(3)
        input_data["EnvironmentSatisfaction"] = c1.slider("Environment Satisfaction", 1, 4, int(defaults["EnvironmentSatisfaction"]))
        input_data["JobInvolvement"] = c2.slider("Job Involvement", 1, 4, int(defaults["JobInvolvement"]))
        input_data["JobSatisfaction"] = c3.slider("Job Satisfaction", 1, 4, int(defaults["JobSatisfaction"]))
        input_data["RelationshipSatisfaction"] = c1.slider(
            "Relationship Satisfaction", 1, 4, int(defaults["RelationshipSatisfaction"])
        )
        input_data["WorkLifeBalance"] = c2.slider("Work Life Balance", 1, 4, int(defaults["WorkLifeBalance"]))
        input_data["StockOptionLevel"] = c3.slider("Stock Option Level", 0, 3, int(defaults["StockOptionLevel"]))
        input_data["TrainingTimesLastYear"] = c1.slider(
            "Training Times Last Year", 0, int(df["TrainingTimesLastYear"].max()), int(defaults["TrainingTimesLastYear"])
        )
        input_data["PercentSalaryHike"] = c2.slider(
            "Percent Salary Hike", int(df["PercentSalaryHike"].min()), int(df["PercentSalaryHike"].max()), int(defaults["PercentSalaryHike"])
        )
        input_data["PerformanceRating"] = c3.slider("Performance Rating", 3, 4, int(defaults["PerformanceRating"]))

    for column in X.columns:
        if column not in input_data:
            if column in numeric_cols:
                input_data[column] = defaults.get(column, float(df[column].median()))
            elif column in categorical_cols:
                input_data[column] = df[column].mode().iloc[0]

    input_df = pd.DataFrame([input_data], columns=X.columns)
    probability = model_result["pipeline"].predict_proba(input_df)[0, 1]
    prediction = "Yes" if probability >= 0.5 else "No"

    result_left, result_right = st.columns([0.6, 0.4])
    with result_left:
        st.progress(float(probability), text=f"Attrition risk: {probability:.1%}")
        if prediction == "Yes":
            st.error("Prediction: Employee is likely to leave.")
        else:
            st.success("Prediction: Employee is likely to stay.")
    with result_right:
        st.dataframe(input_df.T.rename(columns={0: "Selected value"}), use_container_width=True)


def main() -> None:
    df = load_data()

    with st.sidebar:
        st.title("HR Attrition")
        st.caption("EDA, training, and prediction dashboard")
        model_name = st.selectbox(
            "Choose model",
            ["Random Forest", "Gradient Boosting", "Logistic Regression"],
            index=0,
        )
        test_size = st.slider("Test data size", 0.15, 0.40, 0.20, 0.05)
        st.divider()
        st.caption(f"Data file: {DATA_PATH.name}")
        st.caption(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

    st.markdown(
        """
        <div class="hero">
            <h1>Employee Attrition Prediction Dashboard</h1>
            <p>Explore HR patterns first, train a model, then estimate whether an employee is at risk of attrition.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Employees", f"{len(df):,}")
    summary_cols[1].metric("Features", f"{df.shape[1] - 1}")
    summary_cols[2].metric("Attrition Yes", f"{(df[TARGET] == 'Yes').mean():.1%}")
    summary_cols[3].metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    with st.expander("View raw dataset sample", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    plot_eda(df)

    with st.spinner("Training model..."):
        model_result = train_model(model_name, test_size)

    plot_model_results(model_result)
    prediction_form(df, model_result)


if __name__ == "__main__":
    main()
