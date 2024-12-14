import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

def knn_ui():
    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, key="knn_n_neighbors")
    weights = st.selectbox("Weights", options=["uniform", "distance"], key="knn_weights")
    algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"], key="knn_algorithm")
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    st.session_state["classifiers"]["K-Nearest Neighbors"] = model
    st.write(model)

# def knn_ui(X_train, Y_train,X_test,Y_test):
#     st.subheader("K-Nearest Neighbors Hyperparameters")

#     n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
#     weights = st.selectbox("Weights", options=["uniform", "distance"])
#     algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"])

#     model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
#     model.fit(X_train, Y_train)
#     accuracy = model.score(X_test, Y_test)

#     st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
#     return model

# import streamlit as st
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier

# st.title("K-Nearest Neighbors")

# filename = 'path_to/Breast Cancer.csv'
# dataframe = read_csv(filename)
# X = dataframe.iloc[:, :-1].values
# Y = dataframe.iloc[:, -1].values

# with st.sidebar.expander("K-Nearest Neighbors Hyperparameters", expanded=True):
#     test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
#     random_seed = st.slider("Random Seed", 1, 100, 42)
#     n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
#     weights = st.selectbox("Weights", options=["uniform", "distance"])
#     algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"])

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
# model.fit(X_train, Y_train)
# accuracy = model.score(X_test, Y_test)

# st.write(f"Accuracy: {accuracy * 100.0:.3f}%")