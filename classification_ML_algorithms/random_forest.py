import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def random_forest_ui():
    random_seed = st.slider("Random Seed", 1, 100, 7, key="random_forest_random_seed")
    n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100, key="random_forest_n_estimators")
    max_depth = st.slider("Max Depth of Trees", 1, 50, None, key="random_forest_max_depth")  # Allows None for no limit
    min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2, key="random_forest_min_samples_split")
    min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1, key="random_forest_min_samples_leaf")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_seed)
    st.session_state["classifiers"]["Random Forest"] = model
    st.write(model)

