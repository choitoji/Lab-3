import streamlit as st
from sklearn.ensemble import RandomForestRegressor

def random_forest_regressor_ui():
    n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
    max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=None)
    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
    min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1)
    random_state = st.number_input("Random State", value=42)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state)
    st.session_state["regressors"]["Random Forest Regressor"] = model
    st.write(model)
