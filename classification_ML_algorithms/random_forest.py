def random_forest_ui(X_train, Y_train,X_test,Y_test,random_seed):
    import streamlit as st
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    st.subheader("Random Forest Hyperparameters")

    n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100)
    max_depth = st.slider("Max Depth of Trees", 1, 50, None)
    min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2)
    min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
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
# from sklearn.ensemble import RandomForestClassifier

# st.title("Random Forest")

# filename = 'path_to/Breast Cancer.csv'
# dataframe = read_csv(filename)
# X = dataframe.iloc[:, :-1].values
# Y = dataframe.iloc[:, -1].values

# with st.sidebar.expander("Random Forest Hyperparameters", expanded=True):
#     test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
#     random_seed = st.slider("Random Seed", 1, 100, 42)
#     n_estimators = st.slider("Number of Estimators", 10, 200, 100)
#     max_depth = st.slider("Max Depth of Trees", 1, 50, None)
#     min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2)
#     min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# model = RandomForestClassifier(
#     n_estimators=n_estimators,
#     max_depth=max_depth,
#     min_samples_split=min_samples_split,
#     min_samples_leaf=min_samples_leaf,
#     random_state=random_seed
# )

# model.fit(X_train, Y_train)
# accuracy = model.score(X_test, Y_test)
# st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
