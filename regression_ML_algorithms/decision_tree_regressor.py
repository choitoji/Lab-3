import streamlit as st
from sklearn.tree import DecisionTreeRegressor

def decision_tree_regressor_ui():
    max_depth = st.slider("Max Depth", 1, 20, None)
    min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1)
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    st.session_state["regressors"]["Decision Tree Regressor"] = model
    st.write(model)
