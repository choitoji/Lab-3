import streamlit as st
from sklearn.svm import SVR

def support_vector_regressor_ui():
    kernel = st.selectbox("Kernel", options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
    C = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
    epsilon = st.slider("Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    st.session_state["regressors"]["Support Vector Regressor"] = model
    st.write(model)
