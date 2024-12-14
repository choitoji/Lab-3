import streamlit as st
from sklearn.linear_model import LogisticRegression

def logistic_regression_ui():
    random_seed = st.slider("Random Seed", 1, 100, 7, key="logistic_random_seed")
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="logistic_max_iter")
    solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"], key="logistic_solver")
    C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0, key="logistic_C")
    model = LogisticRegression(max_iter=max_iter, solver=solver, C=C, random_state=random_seed)
    st.session_state["classifiers"]["Logistic Regression"] = model
    st.write(model)

# def logistic_regression_ui(X_train, Y_train,X_test,Y_test):
#     st.subheader("Logistic Regression Hyperparameters")

#     max_iter = st.slider("Max Iterations", 100, 500, 200)
#     solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
#     C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0)


#     model = LogisticRegression(max_iter=max_iter, solver=solver, C=C)
#     model.fit(X_train, Y_train)
#     accuracy = model.score(X_test, Y_test)

#     st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
#     return model

# import streamlit as st
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# st.title("Logistic Regression")

# filename = 'path_to/Breast Cancer.csv'
# dataframe = read_csv(filename)
# X = dataframe.iloc[:, :-1].values
# Y = dataframe.iloc[:, -1].values

# with st.sidebar.expander("Logistic Regression Hyperparameters", expanded=True):
#     test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
#     random_seed = st.slider("Random Seed", 1, 100, 42)
#     max_iter = st.slider("Max Iterations", 100, 500, 200)
#     solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
#     C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# model = LogisticRegression(max_iter=max_iter, solver=solver, C=C)
# model.fit(X_train, Y_train)
# accuracy = model.score(X_test, Y_test)

# st.write(f"Accuracy: {accuracy * 100.0:.3f}%")