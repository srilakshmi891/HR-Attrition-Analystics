import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="HR Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Load Files
# ---------------------------
df = pd.read_csv("data/cleaned_dataset.csv")

model = joblib.load("models/hr_attrition_svc.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------------------
# Header
# ---------------------------
st.title("📊 HR Attrition Analytics Dashboard")

st.markdown("""
Analyze employee attrition patterns and predict employee retention using Machine Learning.
""")

# ---------------------------
# KPI Cards
# ---------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(df))

col2.metric(
    "Average Income",
    round(df["MonthlyIncome"].mean())
)

col3.metric(
    "Attrition Rate",
    f"{df['Attrition'].mean()*100:.2f}%"
)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(
    ["📈 Dashboard", "🤖 Prediction", "📋 Insights"]
)

# ==================================================
# DASHBOARD TAB
# ==================================================
with tab1:

    st.subheader("Attrition Distribution")

    fig = px.histogram(
        df,
        x="Attrition",
        title="Employee Attrition Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PREDICTION TAB
# ==================================================
with tab2:

    st.subheader("Employee Prediction")

    age = st.number_input(
        "Age",
        18,
        60,
        30
    )

    distance = st.number_input(
        "Distance From Home",
        1,
        50,
        5
    )

    daily_rate = st.number_input(
        "Daily Rate",
        100,
        1500,
        800
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )

    total_working_years = st.number_input(
        "Total Working Years",
        0,
        40,
        10
    )

    years_at_company = st.number_input(
        "Years At Company",
        0,
        40,
        5
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

    if st.button("Predict Attrition"):

        input_data = pd.DataFrame([[0]*44])

        input_data.iloc[0,0] = age
        input_data.iloc[0,1] = daily_rate
        input_data.iloc[0,2] = distance
        input_data.iloc[0,8] = job_level
        input_data.iloc[0,9] = job_satisfaction
        input_data.iloc[0,13] = 1 if overtime=="Yes" else 0
        input_data.iloc[0,18] = total_working_years
        input_data.iloc[0,21] = years_at_company

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("⚠ Employee is likely to leave the company")
        else:
            st.success("✅ Employee is likely to stay in the company")

# ==================================================
# INSIGHTS TAB
# ==================================================
with tab3:

    st.subheader("Business Insights")

    st.markdown("""
    ### Top Factors Affecting Attrition

    - OverTime
    - Frequent Business Travel
    - Marital Status (Single)
    - Years With Current Manager
    - Job Satisfaction
    - Total Working Years
    - Distance From Home

    ### Recommendations

    - Reduce excessive overtime.
    - Improve employee satisfaction.
    - Support employees with long commutes.
    - Improve manager-employee engagement.
    """)
