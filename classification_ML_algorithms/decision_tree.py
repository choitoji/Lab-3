import streamlit as st
from sklearn.tree import DecisionTreeClassifier

def decision_tree_ui():
    max_depth = st.slider("Max Depth", 1, 20, 5, key="decision_tree_max_depth")
    min_samples_split = st.slider("Min Samples Split", 2, 10, 2, key="decision_tree_min_samples_split")
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1, key="decision_tree_min_samples_leaf")
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    st.session_state["classifiers"]["Decision Tree"] = model
    st.write(model)

