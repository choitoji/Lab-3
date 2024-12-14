import streamlit as st
from sklearn.linear_model import Lasso

def lasso_ui():
    alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key="lasso_alpha")
    max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100, key="lasso_max_iter")
    model = Lasso(alpha=alpha, max_iter=max_iter)
    st.session_state["regressors"]["Lasso"] = model
    st.write(model)
