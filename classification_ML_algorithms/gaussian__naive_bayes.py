import streamlit as st
from sklearn.naive_bayes import GaussianNB

def gaussian_naive_bayes_ui():
    var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1, key="gaussian_naive_bayes_var_smoothing")
    model = GaussianNB(var_smoothing=10 ** var_smoothing)
    st.session_state["classifiers"]["Gaussian Naive Bayes"] = model
    st.write(model)


