import streamlit as st
from sklearn.linear_model import ElasticNet

def elastic_net_ui():
    alpha = st.slider("Alpha (Regularization Strength)", 0.0, 5.0, 1.0, 0.1)
    l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01)
    max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
    st.session_state["regressors"]["Elastic Net"] = model
    st.write(model)
