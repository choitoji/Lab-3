import streamlit as st
from sklearn.ensemble import AdaBoostClassifier

def adaboost_ui():
    random_seed = st.slider("Random Seed", 1, 100, 7, key="adaboost_random_seed")
    n_estimators = st.slider("Number of Estimators", 1, 100, 50, key="adaboost_n_estimators")
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed)
    st.session_state["classifiers"]["AdaBoost"] = model
    st.write(model)
