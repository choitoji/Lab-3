import streamlit as st
from sklearn.svm import SVC

def svm_ui():
    random_seed = st.slider("Random Seed", 1, 100, 42, key="svm_random_seed")
    C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, key="svm_C")
    kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'], key="svm_kernel")
    model = SVC(kernel=kernel, C=C, random_state=random_seed)
    st.session_state["classifiers"]["Support Vector Machine (SVM)"] = model
    st.write(model)

# def svm_ui(X_train, Y_train,X_test,Y_test,random_seed):
#     st.subheader("Support Vector Machine (SVM) Hyperparameters")

#     C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
#     kernel = st.selectbox("Kernel Type", options=["linear", "poly", "rbf", "sigmoid"])

#     model = SVC(kernel=kernel, C=C, random_state=random_seed)
#     model.fit(X_train, Y_train)
#     accuracy = model.score(X_test, Y_test)

#     st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
#     return model


# import streamlit as st
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC

# st.title("Support Vector Machine (SVM)")

# filename = 'path_to/Breast Cancer.csv'
# dataframe = read_csv(filename)
# X = dataframe.iloc[:, :-1].values
# Y = dataframe.iloc[:, -1].values

# with st.sidebar.expander("Support Vector Machine (SVM) Hyperparameters", expanded=True):
#     test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
#     random_seed = st.slider("Random Seed", 1, 100, 42)
#     C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
#     kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'])

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# model = SVC(kernel=kernel, C=C, random_state=random_seed)
# model.fit(X_train, Y_train)
# accuracy = model.score(X_test, Y_test)

# st.write(f"Accuracy: {accuracy * 100.0:.3f}%")