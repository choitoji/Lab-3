import streamlit as st
from sklearn.ensemble import AdaBoostRegressor

def adaboost_regressor_ui():
    n_estimators = st.slider("Number of Estimators", 1, 200, 50, 1)
    learning_rate = st.slider("Learning Rate", 0.01, 5.0, 1.0, 0.01)
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    st.session_state["regressors"]["AdaBoost Regressor"] = model
    st.write(model)
