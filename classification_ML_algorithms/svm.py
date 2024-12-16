import streamlit as st
from sklearn.svm import SVC

def svm_ui():
    random_seed = st.slider("Random Seed", 1, 100, 42, key="svm_random_seed")
    C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, key="svm_C")
    kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'], key="svm_kernel")
    model = SVC(kernel=kernel, C=C, random_state=random_seed)
    st.session_state["classifiers"]["Support Vector Machine (SVM)"] = model
    st.write(model)

