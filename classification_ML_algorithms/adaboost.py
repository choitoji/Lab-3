import streamlit as st
from sklearn.ensemble import AdaBoostClassifier

def adaboost_ui():
    random_seed = st.slider("Random Seed", 1, 100, 7, key="adaboost_random_seed")
    n_estimators = st.slider("Number of Estimators", 1, 100, 50, key="adaboost_n_estimators")
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed)
    st.session_state["classifiers"]["AdaBoost"] = model
    st.write(model)


# def adaboost_ui(X_train, Y_train,X_test,Y_test,random_seed):
#     st.subheader("AdaBoost Hyperparameters")

#     # Hyperparameters for AdaBoost
#     n_estimators = st.slider("Number of Estimators", 1, 100, 50)

#     # Split the dataset for classification
#     # Initialize AdaBoost classifier
#     model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed, algorithm="SAMME")

#     # Train the model
#     model.fit(X_train, Y_train)

#     # Evaluate classification accuracy
#     accuracy = model.score(X_test,Y_test)
#     st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
#     return model




# import streamlit as st
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import AdaBoostClassifier

# st.title("AdaBoost")

# filename = 'path_to/Breast Cancer.csv'
# dataframe = read_csv(filename)
# X = dataframe.iloc[:, :-1].values
# Y = dataframe.iloc[:, -1].values

# with st.sidebar.expander("AdaBoost Hyperparameters", expanded=True):
#     test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
#     random_seed = st.slider("Random Seed", 1, 100, 42)
#     n_estimators = st.slider("Number of Estimators", 10, 100, 50)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed)
# model.fit(X_train, Y_train)
# accuracy = model.score(X_test, Y_test)

# st.write(f"Accuracy: {accuracy * 100.0:.3f}%")