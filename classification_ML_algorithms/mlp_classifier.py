def mlp_classifier_ui(X_train, Y_train,X_test,Y_test,random_seed):
    import streamlit as st
    from sklearn.neural_network import MLPClassifier

    st.subheader("MLP Classifier Hyperparameters")

    hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 64,32)", "64,32")
    activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"])
    max_iter = st.slider("Max Iterations", 100, 500, 200)

    hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))

    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=random_seed)
    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)

    st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
    return model

# import streamlit as st
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier

# st.title("MLP Classifier")

# filename = 'path_to/Breast Cancer.csv'
# dataframe = read_csv(filename)
# X = dataframe.iloc[:, :-1].values
# Y = dataframe.iloc[:, -1].values

# with st.sidebar.expander("MLP Classifier Hyperparameters", expanded=True):
#     test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
#     random_seed = st.slider("Random Seed", 1, 100, 42)
#     hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 64,32)", "64,32")
#     activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"])
#     max_iter = st.slider("Max Iterations", 100, 500, 200)

# hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=random_seed)
# model.fit(X_train, Y_train)
# accuracy = model.score(X_test, Y_test)

# st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
