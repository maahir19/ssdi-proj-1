import streamlit as st

import pandas as pd

import numpy as np

from scipy.stats import t
 
 
# ---------- YOUR FUNCTION ----------

def t_test_independent_pooled(a, b, alpha=0.05, alternative="two-sided"):

    a = np.array(a)

    b = np.array(b)
 
    n1, n2 = len(a), len(b)

    xbar1, xbar2 = np.mean(a), np.mean(b)

    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
 
    sp2 = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2)

    se = np.sqrt(sp2 * (1/n1 + 1/n2))
 
    t_cal = (xbar1 - xbar2) / se

    df = n1 + n2 - 2
 
    if alternative == "two-sided":

        t_crit = t.ppf(1 - alpha/2, df)

        p_value = 2 * (1 - t.cdf(abs(t_cal), df))

        reject = abs(t_cal) > t_crit
 
    elif alternative == "greater":

        t_crit = t.ppf(1 - alpha, df)

        p_value = 1 - t.cdf(t_cal, df)

        reject = t_cal > t_crit
 
    elif alternative == "less":

        t_crit = t.ppf(alpha, df)

        p_value = t.cdf(t_cal, df)

        reject = t_cal < t_crit
 
    return {

        "t_cal": t_cal,

        "df": df,

        "p_value": p_value,

        "decision": "Reject H0" if reject else "Fail to Reject H0"

    }
 
 
# ---------- UI ----------

st.title("📊 Independent Pooled T-Test App")
 
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
 
if uploaded_file:

    df = pd.read_csv(uploaded_file)
 
    st.subheader("Dataset Preview")

    st.dataframe(df.head())
 
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
 
    if len(numeric_cols) < 2:

        st.error("Need at least 2 numeric columns.")

    else:

        col1 = st.selectbox("Select Sample 1 Column", numeric_cols)

        col2 = st.selectbox("Select Sample 2 Column", numeric_cols)
 
        alpha = st.number_input("Significance Level (alpha)", value=0.05)

        alternative = st.selectbox(

            "Alternative Hypothesis",

            ["two-sided", "greater", "less"]

        )
 
        if st.button("Run T-Test"):

            result = t_test_independent_pooled(

                df[col1].dropna(),

                df[col2].dropna(),

                alpha=alpha,

                alternative=alternative

            )
 
            st.subheader("Results")

            st.write("t statistic:", result["t_cal"])

            st.write("Degrees of Freedom:", result["df"])

            st.write("p-value:", result["p_value"])

            st.success(result["decision"])
 