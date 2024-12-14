import streamlit as st
from sklearn.neural_network import MLPRegressor

def mlp_regressor_ui():
    hidden_layer_sizes = st.slider("Hidden Layer Sizes", min_value=10, max_value=200, value=(100, 50), step=10)
    activation = st.selectbox("Activation Function", options=['identity', 'logistic', 'tanh', 'relu'], index=3)
    solver = st.selectbox("Solver", options=['adam', 'lbfgs', 'sgd'], index=0)
    learning_rate = st.selectbox("Learning Rate Schedule", options=['constant', 'invscaling', 'adaptive'], index=0)
    max_iter = st.slider("Max Iterations", min_value=100, max_value=2000, value=1000, step=100, key="max1")
    random_state = st.number_input("Random State", value=50)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, learning_rate=learning_rate, max_iter=max_iter, random_state=random_state)
    st.session_state["regressors"]["MLP Regressor"] = model
    st.write(model)
