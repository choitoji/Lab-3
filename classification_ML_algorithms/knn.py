import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

def knn_ui():
    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, key="knn_n_neighbors")
    weights = st.selectbox("Weights", options=["uniform", "distance"], key="knn_weights")
    algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"], key="knn_algorithm")
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    st.session_state["classifiers"]["K-Nearest Neighbors"] = model
    st.write(model)

