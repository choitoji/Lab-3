import streamlit as st
from sklearn.linear_model import LogisticRegression

def logistic_regression_ui():
    random_seed = st.slider("Random Seed", 1, 100, 7, key="logistic_random_seed")
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="logistic_max_iter")
    solver = st.selectbox("Solver", options=["sag", "lbfgs", "liblinear", "saga", "newton-cg"], key="logistic_solver")
    C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0, key="logistic_C")
    model = LogisticRegression(max_iter=max_iter, solver=solver, C=C, random_state=random_seed)
    st.session_state["classifiers"]["Logistic Regression"] = model
    st.write(model)

