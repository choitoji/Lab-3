import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_val_score, PredefinedSplit, ShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Importing ML algorithm files
from classification_ML_algorithms.decision_tree import decision_tree_ui
from classification_ML_algorithms.gaussian__naive_bayes import gaussian_naive_bayes_ui
from classification_ML_algorithms.adaboost import adaboost_ui
from classification_ML_algorithms.knn import knn_ui
from classification_ML_algorithms.logistic_regression import logistic_regression_ui
from classification_ML_algorithms.mlp_classifier import mlp_classifier_ui
from classification_ML_algorithms.perceptron import perceptron_ui
from classification_ML_algorithms.random_forest import random_forest_ui
from classification_ML_algorithms.svm import svm_ui

from regression_ML_algorithms.adaboost_regressor import adaboost_regressor_ui
from regression_ML_algorithms.decision_tree_regressor import decision_tree_regressor_ui
from regression_ML_algorithms.elastic_net import elastic_net_ui
from regression_ML_algorithms.knn_regressor import knn_regressor_ui
from regression_ML_algorithms.lasso import lasso_ui
from regression_ML_algorithms.linear_regression import linear_regression_ui
from regression_ML_algorithms.mlp_regressor import mlp_regressor_ui
from regression_ML_algorithms.random_forest_regressor import random_forest_regressor_ui
from regression_ML_algorithms.ridge import ridge_ui
from regression_ML_algorithms.support_vector_regressor import support_vector_regressor_ui


# Title of the app
st.title('Lab 3: Comparison of Different Machine Learning Algorithms for Model Selection')

# File uploader for dataset
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

# Select task type
st.session_state['task'] = st.selectbox('Select Task', ['Classification', 'Regression'])

if "cv" not in st.session_state:
    st.session_state["cv"] = None
if "X" not in st.session_state:
    st.session_state["X"] = None
if "Y" not in st.session_state:
    st.session_state["Y"] = None
if "classifiers" not in st.session_state:
    st.session_state["classifiers"] = {}
if "regressors" not in st.session_state:
    st.session_state["regressors"] = {}
if "classif_scores" not in st.session_state:
    st.session_state["classif_scores"] = {}
if "regress_scores" not in st.session_state:
    st.session_state["regress_scores"] = {}

# Tabs for separating functionalities
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset Overview",
    "Resampling Techniques",
    "ML Model Training",
    "Visualizations",
    "Prediction"
])

# Dataset Overview
with tab1:
    st.header("Dataset Overview")

    if uploaded_file is not None:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")

        # Drop any columns that have all missing values
        df.dropna(axis=1, how='all', inplace=True)
        st.write(df.head())

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Missing Values Before Imputation")
        with c2: 
            st.subheader("Missing Values After Imputation")

        c1, c2 = st.columns(2)
        with c1:
            # Display missing values
            missing_values = df.isnull().sum().reset_index()
            missing_values.columns = ["Column", "Missing Count"]
            missing_values["Missing Percentage"] = (missing_values["Missing Count"] / len(df)) * 100
            st.write(missing_values)

        # Handle missing values using SimpleImputer with median strategy for numeric columns
        imputer = SimpleImputer(strategy='median')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        with c2:
            # show missing values again for new dataset
            missing_values = df.isnull().sum().reset_index()
            missing_values.columns = ["Column", "Missing Count"]
            missing_values["Missing Percentage"] = (missing_values["Missing Count"] / len(df)) * 100
            st.write(missing_values)

        st.subheader("Cleaned Dataset Preview:")
        st.write(df.head())

        # Preprocess categorical columns
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))

        # Preprocess target variable based on task type
        if st.session_state['task'] == 'Classification':
            if 'diagnosis' not in df.columns:
                st.error("The dataset must contain a 'diagnosis' column for classification.")
            else:
                # Check if 'diagnosis' is already numeric (mapped to 0 and 1)
                if pd.api.types.is_numeric_dtype(df['diagnosis']):
                    # Assume it's already mapped correctly to 0 and 1
                    st.write("The 'diagnosis' column is already numeric (0 for Benign, 1 for Malignant).")
                else:
                    # Explicit mapping of diagnosis to integers
                    unexpected_values = df[~df['diagnosis'].isin(['M', 'B'])]['diagnosis'].unique()
                    if len(unexpected_values) > 0:
                        st.error(f"Unexpected values found in 'diagnosis': {unexpected_values}. Ensure the column contains only 'M' or 'B'.")
                    else:
                        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # M -> 1, B -> 0
                        st.session_state['label_encoder'] = {'M': 1, 'B': 0}  # Save mapping in session state

                # Preparing the data
                X = df.drop(columns=['diagnosis']).values
                Y = df['diagnosis'].values
                st.session_state['X'] = X
                st.session_state['Y'] = Y

                # Ensure the target is discrete for classification tasks
                if pd.isnull(Y).any() or not np.issubdtype(Y.dtype, np.integer):
                    st.error("The target variable for classification must be discrete (integer values) and contain no missing values. Please clean your data and try again.")

        elif st.session_state['task'] == 'Regression':
            # Ensure 'pm2.5' column exists in the dataset
            if 'pm2.5' not in df.columns:
                st.error("The dataset must contain a 'pm2.5' column for regression.")
            else:
                # Preparing Data
                X = df.drop(columns=['pm2.5']).values
                Y = df['pm2.5'].values
                st.session_state['X'] = X
                st.session_state['Y'] = Y


# Resampling and Training
with tab2:
    st.header("Resampling Techniques")
    if st.session_state.X is None or st.session_state.Y is None:
        st.warning("Please upload a dataset in the 'Dataset Overview' tab.")
    else:
        st.selectbox("Select Resampling Technique", [
            "Train-Test Split",
            "K-Fold Cross Validation",
            "Leave-One-Out Cross Validation",
            "Repeated Random Sampling",
        ], key="chosen_cv")

        with st.form("resampling_form"):
            cv_object = None
            cv = st.session_state["chosen_cv"]
            if cv == "Train-Test Split":
                st.write("Train-Test Split")
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
                random_seed = st.slider("Random Seed", 1, 100, 42)

                X_train, X_test, Y_train, Y_test = train_test_split(st.session_state['X'], st.session_state['Y'], test_size=test_size, random_state=random_seed)
                predefined_split = PredefinedSplit(test_fold=[-1 if i in X_train else 0 for i in range(len(X))])
                cv_object = predefined_split

            elif cv == "K-Fold Cross Validation":
                st.write("K-Fold Cross Validation")
                k_folds = st.slider("Number of Folds", 2, 10, 5)
                shuffle = st.checkbox("Shuffle Data", value=True)
                random_seed = st.slider("Random Seed", 1, 100, 42)
                kfold = KFold(n_splits=k_folds, shuffle=shuffle, random_state=random_seed if shuffle else None)
                cv_object = kfold

            elif cv == "Leave-One-Out Cross Validation":
                st.write("Leave-One-Out Cross Validation")
                loo = LeaveOneOut()
                cv_object = loo

            elif cv == "Repeated Random Sampling":
                st.write("Repeated Random Sampling")
                n_splits = st.slider("Number of Splits", 2, 10, 5)
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
                random_seed = st.slider("Random Seed", 1, 100, 42)
                shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
                cv_object = shuffle_split

            else:
                st.warning("Please select a resampling technique to continue.")

            if cv_object is not None:
                # button = st.button("Apply Resampling Technique")
                button = st.form_submit_button("Apply Resampling Technique")
                if button:
                    st.session_state["cv"] = cv_object
                    st.success("Resampling technique applied successfully.")

        st.write("Chosen Sampling Technique:")
        st.write(st.session_state["cv"])

# Machine Learning Models
with tab3:
    st.header("Machine Learning Models")
    if st.session_state.X is None or st.session_state.Y is None:
        st.warning("Please upload a dataset in the 'Dataset Overview' tab.")
    elif st.session_state["cv"] is None:
        st.warning("Please apply a resampling technique in the 'Resampling Techniques' tab.")
    else:
        if st.session_state['task'] == 'Classification':
            model_ui = {
                "Decision Tree": decision_tree_ui,
                "Gaussian Naive Bayes": gaussian_naive_bayes_ui,
                "AdaBoost": adaboost_ui,
                "K-Nearest Neighbors": knn_ui,
                "Logistic Regression": logistic_regression_ui,
                "MLP Classifier": mlp_classifier_ui,
                "Perceptron": perceptron_ui,
                "Random Forest": random_forest_ui,
                "Support Vector Machine": svm_ui
            }
        else:
            model_ui = {
                "Decision Tree Regressor": decision_tree_regressor_ui,
                "Elastic Net": elastic_net_ui,
                "AdaBoost Regressor": adaboost_regressor_ui,
                "KNN Regressor": knn_regressor_ui,
                "Lasso": lasso_ui,
                "Ridge": ridge_ui,
                "Linear Regression": linear_regression_ui,
                "MLP Regressor": mlp_regressor_ui,
                "Random Forest Regressor": random_forest_regressor_ui,
                "Support Vector Regressor (SVR)": support_vector_regressor_ui
            }

        c1, c2 = st.container(border=True).columns(2)
        model_group = "classifiers" if st.session_state['task'] == 'Classification' else "regressors"
        for i, (model_name, model_ui_func) in enumerate(model_ui.items()):
            c = c1 if i % 2 == 0 else c2
            with c.expander(model_name):
                model_ui_func()


# Visualizations
with tab4:
    st.header("Visualizations")
    if st.session_state.X is None or st.session_state.Y is None:
        st.warning("Please upload a dataset in the 'Dataset Overview' tab.")
    elif st.session_state["cv"] is None:
        st.warning("Please apply a resampling technique in the 'Resampling Techniques' tab.")
    else:
        if st.session_state['task'] == 'Classification' and st.session_state['classifiers'] == {}:
            st.warning("Please train a classifier in the 'ML Models' tab.")
        elif st.session_state['task'] == 'Regression' and st.session_state['regressors'] == {}:
            st.warning("Please train a regressor in the 'ML Models' tab.")
        else:
            if st.button("Generate Scores"):
                if st.session_state['task'] == 'Classification':
                    models = st.session_state['classifiers']
                    try:
                        for model_name, model in models.items():
                            scores = cross_val_score(model, st.session_state['X'], st.session_state['Y'], cv=st.session_state["cv"], scoring='accuracy').mean()
                            st.session_state['classif_scores'][model_name] = scores
                    except ValueError as e:
                        print(e)
                else:
                    models = st.session_state['regressors']
                    try:
                        for model_name, model in models.items():
                            scores = cross_val_score(model, st.session_state['X'], st.session_state['Y'], cv=st.session_state["cv"], scoring='neg_mean_absolute_error').mean()
                            scores = -scores
                            st.session_state['regress_scores'][model_name] = scores
                    except ValueError as e:
                        print(e)

            if st.session_state['task'] == 'Classification':
                st.write("Classification Scores:")
                fig = px.bar(x=list(st.session_state['classif_scores'].keys()), y=list(st.session_state['classif_scores'].values()), labels={'x':'Model', 'y':'Accuracy Score'}, title='Classification Scores')
                st.plotly_chart(fig)

                best_model = max(st.session_state['classif_scores'], key=st.session_state['classif_scores'].get)
                st.write(f"Best Model: {best_model} with accuracy score of {st.session_state['classif_scores'][best_model]* 100:.2f}%")

            else:
                st.write("Regression Scores:")
                fig = px.bar(x=list(st.session_state['regress_scores'].keys()), y=list(st.session_state['regress_scores'].values()), labels={'x':'Model', 'y':'MAE Score'}, title='Regression Scores')
                st.plotly_chart(fig)

                best_model = min(st.session_state['regress_scores'], key=st.session_state['regress_scores'].get)
                st.write(f"Best Model: {best_model}, with MAE score of {st.session_state['regress_scores'][best_model]:.2f}%")

            st.write("Save best model")
            if st.button("Save Best Model"):
                if st.session_state['task'] == 'Classification':
                    model = st.session_state['classifiers'][best_model]
                else:
                    model = st.session_state['regressors'][best_model]
                joblib.dump(model.fit(st.session_state["X"], st.session_state["Y"]), f"models/{best_model}({st.session_state['task']}).joblib")
                st.success("Best model saved successfully.")

with tab5:
    st.header("Prediction")
    st.write("Predict using trained models.")

    st.subheader("Upload a Saved Model for Prediction")
    uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"], key="model_upload")

    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        st.write("Model Loaded Successfully!")

        # Fetch the feature count from the trained model
        try:
            n_features = model.n_features_in_
        except AttributeError:
            st.error("Model does not have attribute `n_features_in_`. Ensure it is a Scikit-learn model.")

        if st.session_state['task'] == 'Classification':
            # Input fields for breast cancer prediction based on the provided dataset headers
            st.subheader("Input Sample Data for Breast Cancer Prediction")

            # Map input features from the dataset
            input_features = [
                "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                "smoothness_mean", "compactness_mean", "concavity_mean", 
                "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
                # Add placeholders for missing features
                *(f"feature_{i}" for i in range(n_features - 10))  # Placeholder names
            ]

            # Dynamically create input fields for the required features
            input_data = []
            for feature in input_features[:10]:  # Adjusted for known dataset headers
                input_data.append(st.number_input(f"{feature}", value=0.0, step=0.1))
            input_data += [0] * (n_features - 10)  # Add placeholders for extra features

            input_data = np.array([input_data])

            # Predict using the trained model
            if st.button("Predict"):
                prediction = model.predict(input_data)

                # Decode prediction using the explicit mapping
                if prediction[0] == 1:
                    decoded_prediction = "Malignant"
                elif prediction[0] == 0:
                    decoded_prediction = "Benign"
                else:
                    decoded_prediction = "Unknown"

                st.subheader("Prediction Result")
                st.write(f"The predicted diagnosis is: **{decoded_prediction}**")

        elif st.session_state['task'] == 'Regression':
            # Input fields for PM2.5 prediction based on the provided dataset headers
            st.subheader("Input Sample Data for PM2.5 Prediction")

            # Map input features from the dataset
            input_features = [
                "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"
            ]

            # Replace date and hour inputs with a calendar and clock
            st.write("Select Date and Time:")
            date_time = st.date_input("Date", key="date_input")
            time = st.time_input("Time", key="time_input")

            # Dynamically create input fields for the other features
            input_data = []
            for feature in input_features:
                if feature == "cbwd":  # For categorical wind direction
                    cbwd_map = {"NW": 1, "NE": 2, "SW": 3, "SE": 4}
                    cbwd_input = st.selectbox(f"{feature} (Wind Direction)", options=list(cbwd_map.keys()), index=0)
                    input_data.append(cbwd_map[cbwd_input])  # Map wind direction to numeric values
                else:
                    input_data.append(st.number_input(f"{feature}", value=0.0, step=0.1))

            # Combine date and time into separate features
            input_data.insert(0, date_time.year)
            input_data.insert(1, date_time.month)
            input_data.insert(2, date_time.day)
            input_data.insert(3, time.hour)
            input_data.insert(4, time.minute)

            # Prepare the input array
            input_data = np.array([input_data])

            # Predict using the trained model
            if st.button("Predict"):
                try:
                    prediction = model.predict(input_data)
                    st.subheader("Prediction Result")
                    st.write(f"The predicted PM2.5 level is: **{prediction[0]:.2f}**")
                except ValueError as e:
                    st.error(f"Prediction failed: {e}")
