import streamlit as st
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

def perceptron_ui():
    random_seed = st.slider("Random Seed", 1, 100, 7, key="perceptron_random_seed")
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="perceptron_max_iter")
    eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0, key="perceptron_eta0")
    tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3, key="perceptron_tol")
    model = Perceptron(max_iter=max_iter, random_state=random_seed, eta0=eta0, tol=tol)
    st.session_state["classifiers"]["Perceptron"] = model
    st.write(model)

