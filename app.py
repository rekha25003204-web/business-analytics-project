import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # trained in notebooks/model_development.ipynb


@st.cache_data
def load_data():
    return pd.read_csv("data/dataset.csv")


model = load_model()
df = load_data()
df["Left"] = (df["Attrition"] == "Yes").astype(int)

page = st.sidebar.radio("Navigate", ["1. Business Problem", "2. Data Insights", "3. Prediction"])

# ---------------------------------------------------------------- Page 1
if page == "1. Business Problem":
    st.title("👥 Employee Attrition Predictor")
    st.subheader("A decision tool for HR retention planning")
    st.markdown("""
**Business problem:** Employees leaving is costly (recruiting, training, lost productivity).
HR wants to identify who is at risk *before* they resign.

**Analytics objective:** Predict whether an employee is likely to leave (classification, target = `Attrition` Yes/No).

**Business decision:** High-risk employees are prioritised for retention actions such as stay interviews,
workload and overtime review, and compensation or career-path discussions.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", f"{len(df):,}")
    c2.metric("Columns", df.shape[1] - 1)
    c3.metric("Attrition rate", f"{df['Left'].mean():.1%}")
    c4.metric("Model", "Logistic Regression")
    st.markdown("""
**Dataset:** IBM HR Analytics Employee Attrition & Performance (fictional data published on Kaggle).
1,470 rows, 35 columns. **Target:** `Attrition` (1 = left, 0 = stayed).

**Model performance (20% held-out test set):** Recall about 77%, precision about 39%, ROC-AUC about 0.84.
Recall is the priority metric because missing a leaver is costlier than a needless check-in.
""")

# ---------------------------------------------------------------- Page 2
elif page == "2. Data Insights":
    st.title("📊 Data Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall attrition", f"{df['Left'].mean():.1%}")
    c2.metric("Median income (left)", f"{df[df.Left == 1].MonthlyIncome.median():,.0f}")
    c3.metric("Median income (stayed)", f"{df[df.Left == 0].MonthlyIncome.median():,.0f}")

    st.markdown("#### Attrition rate by overtime (%)")
    st.bar_chart((df.groupby("OverTime").Left.mean() * 100).round(1))
    ot = df.groupby("OverTime").Left.mean() * 100
    st.info(f"Employees working overtime leave at {ot['Yes']:.1f}% versus {ot['No']:.1f}% for those who do not. "
            "Overtime is a controllable lever for HR.")

    st.markdown("#### Attrition rate by tenure (%)")
    tenure = df.groupby(pd.cut(df.YearsAtCompany, [-1, 2, 5, 10, 50],
                               labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"]), observed=True).Left.mean() * 100
    st.bar_chart(tenure.round(1))
    st.info("The first two years are the highest-risk window, so onboarding and early engagement matter most.")

    st.markdown("#### Attrition rate by business travel (%)")
    st.bar_chart((df.groupby("BusinessTravel").Left.mean() * 100).round(1))
    st.info("Frequent travellers leave far more often than non-travellers.")

    st.markdown("#### Model: confusion matrix and key drivers")
    c1, c2 = st.columns(2)
    c1.image("images/confusion_matrix.png", caption="Confusion matrix (test set)")
    c2.image("images/feature_importance.png", caption="Factors raising / lowering risk")

# ---------------------------------------------------------------- Page 3
else:
    st.title("🔮 Predict Employee Attrition Risk")
    st.write("Enter the employee's details and click **Predict**.")

    with st.form("inputs"):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 18, 60, 35)
        income = c1.number_input("Monthly income", 1000, 20000, 5000, step=250)
        distance = c1.slider("Distance from home (km)", 1, 30, 8)
        years_company = c1.slider("Years at company", 0, 40, 5)
        total_years = c2.slider("Total working years", 0, 40, 10)
        num_companies = c2.slider("Number of companies worked at", 0, 9, 2)
        since_promo = c2.slider("Years since last promotion", 0, 15, 1)
        overtime = c2.selectbox("Works overtime?", ["No", "Yes"])
        travel = c3.selectbox("Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1)
        job_level = c3.selectbox("Job level (1 = junior, 5 = senior)", [1, 2, 3, 4, 5], index=1)
        stock = c3.selectbox("Stock option level", [0, 1, 2, 3], index=1)
        job_sat = c3.selectbox("Job satisfaction (1 = low, 4 = high)", [1, 2, 3, 4], index=2)
        wlb = c1.selectbox("Work-life balance (1 = bad, 4 = best)", [1, 2, 3, 4], index=2)
        env_sat = c2.selectbox("Environment satisfaction (1 = low, 4 = high)", [1, 2, 3, 4], index=2)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "Age": age, "MonthlyIncome": income, "DistanceFromHome": distance,
            "YearsAtCompany": years_company, "TotalWorkingYears": total_years,
            "NumCompaniesWorked": num_companies, "YearsSinceLastPromotion": since_promo,
            "OverTime": overtime, "BusinessTravel": travel, "JobSatisfaction": job_sat,
            "WorkLifeBalance": wlb, "JobLevel": job_level, "StockOptionLevel": stock,
            "EnvironmentSatisfaction": env_sat,
        }])
        prob = model.predict_proba(row)[0, 1]
        st.metric("Probability of leaving", f"{prob:.1%}")
        st.progress(float(prob))

        if prob >= 0.60:
            st.error("🔴 HIGH risk of leaving")
            st.markdown("**Recommended action:** schedule a stay interview this week, review workload/overtime and "
                        "pay position, and discuss a career path.")
        elif prob >= 0.30:
            st.warning("🟠 MEDIUM risk of leaving")
            st.markdown("**Recommended action:** manager check-in within the month; look at overtime, travel "
                        "and promotion timeline.")
        else:
            st.success("🟢 LOW risk of leaving")
            st.markdown("**Recommended action:** no urgent action; continue regular engagement.")
        st.caption("Risk bands: below 30% low, 30-60% medium, above 60% high. This is a decision-support "
                   "tool, not a final judgement. A human should always review before acting.")
