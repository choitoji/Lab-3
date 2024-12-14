import streamlit as st
from sklearn.linear_model import LinearRegression

def linear_regression_ui():
    st.write("N/A")
    model = LinearRegression()
    st.session_state["regressors"]["Linear Regression"] = model
    st.write(model)
