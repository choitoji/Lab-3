def decision_tree_ui(X_train, Y_train,X_test,Y_test,random_seed):
    import streamlit as st
    from sklearn.tree import DecisionTreeClassifier

    st.subheader("Decision Tree Hyperparameters")

    max_depth = st.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)


    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed
    )

    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)

    st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
    return model

# import streamlit as st
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier

# st.title("Decision Tree ML")

# filename = 'path_to/Breast Cancer.csv'
# dataframe = read_csv(filename)
# X = dataframe.iloc[:, :-1].values  # Assuming all but the last column are features
# Y = dataframe.iloc[:, -1].values   # Assuming the last column is the target

# with st.sidebar.expander("Decision Tree Hyperparameters", expanded=True):
#     test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
#     random_seed = st.slider("Random Seed", 1, 100, 50)
#     max_depth = st.slider("Max Depth", 1, 20, 5)
#     min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
#     min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# model = DecisionTreeClassifier(
#     max_depth=max_depth,
#     min_samples_split=min_samples_split,
#     min_samples_leaf=min_samples_leaf,
#     random_state=random_seed
# )

# model.fit(X_train, Y_train)
# accuracy = model.score(X_test, Y_test)
# st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
