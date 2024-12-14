import streamlit as st
from sklearn.linear_model import Ridge

def ridge_ui():
    alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key="ridge_alpha")
    max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100, key="ridge_max_iter")
    model = Ridge(alpha=alpha, max_iter=max_iter)
    st.session_state["regressors"]["Ridge"] = model
    st.write(model)
