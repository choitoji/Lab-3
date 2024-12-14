def gaussian_naive_bayes_ui(X_train, Y_train,X_test,Y_test):
    import streamlit as st
    from sklearn.naive_bayes import GaussianNB

    st.subheader("Gaussian Naive Bayes Hyperparameters")

    var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1)

    var_smoothing_value = 10 ** var_smoothing

    model = GaussianNB(var_smoothing=var_smoothing_value)
    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)

    st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
    return model

# import streamlit as st
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB

# st.title("Gaussian Naive Bayes")

# filename = 'path_to/Breast Cancer.csv'
# dataframe = read_csv(filename)
# X = dataframe.iloc[:, :-1].values
# Y = dataframe.iloc[:, -1].values

# with st.sidebar.expander("Gaussian Naive Bayes Hyperparameters", expanded=True):
#     test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
#     random_seed = st.slider("Random Seed", 1, 100, 7)
#     var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1)

# var_smoothing_value = 10 ** var_smoothing

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# model = GaussianNB(var_smoothing=var_smoothing_value)
# model.fit(X_train, Y_train)
# accuracy = model.score(X_test, Y_test)
# st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
