import streamlit as st
from sklearn.neural_network import MLPClassifier

def mlp_classifier_ui():
    random_seed = st.slider("Random Seed", 1, 100, 7, key="mlp_classifier_random_seed")
    hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 65,32)", "65,32", key="mlp_classifier_hidden_layer_sizes")
    activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"], key="mlp_classifier_activation")
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="mlp_classifier_max_iter")
    hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=random_seed)
    st.session_state["classifiers"]["MLP Classifier"] = model
    st.write(model)

