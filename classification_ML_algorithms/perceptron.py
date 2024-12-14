def perceptron_ui(X_train, Y_train,X_test,Y_test,random_seed):
    import streamlit as st
    from sklearn.linear_model import Perceptron
    from sklearn.model_selection import train_test_split

    st.subheader("Perceptron Hyperparameters")


    max_iter = st.slider("Max Iterations", 100, 500, 200)
    eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0)
    tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3)


    model = Perceptron(
        max_iter=max_iter,
        random_state=random_seed,
        eta0=eta0,
        tol=tol
    )
    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)

    st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
    return model


# import streamlit as st
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron

# st.title("Perceptron Classifier")

# filename = 'path_to/Breast Cancer.csv'
# dataframe = read_csv(filename)
# X = dataframe.iloc[:, :-1].values
# Y = dataframe.iloc[:, -1].values

# with st.sidebar.expander("Perceptron Classifier Hyperparameters", expanded=True):
#     test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
#     random_seed = st.slider("Random Seed", 1, 100, 42)
#     max_iter = st.slider("Max Iterations", 100, 500, 200)
#     eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0)
#     tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# model = Perceptron(max_iter=max_iter, random_state=random_seed, eta0=eta0, tol=tol)
# model.fit(X_train, Y_train)
# accuracy = model.score(X_test, Y_test)

# st.write(f"Accuracy: {accuracy * 100.0:.3f}%")