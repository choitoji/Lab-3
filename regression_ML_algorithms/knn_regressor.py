import streamlit as st
from sklearn.neighbors import KNeighborsRegressor

def knn_regressor_ui():
    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, 1)
    weights = st.selectbox("Weights", ["uniform", "distance"])
    algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    st.session_state["regressors"]["KNN Regressor"] = model
    st.write(model)
