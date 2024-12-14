import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_val_score
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

# Title of the app
st.title('Lab 3: Comparison of Different Machine Learning Algorithms for Model Selection')

# File uploader for dataset
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

# Select task type
st.session_state['task'] = st.selectbox('Select Task', ['Classification', 'Regression'])

# Tabs for separating functionalities
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dataset Overview",
    "ML Models",
    "Resampling and Training",
    "Visualizations",
    "Model Comparison",
    "Prediction"
])

# Dataset Overview
with tab1:
    st.header("Dataset Overview")

    if uploaded_file is not None:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(df.head())

        # Drop any columns that have all missing values
        df.dropna(axis=1, how='all', inplace=True)

        # Display missing values
        st.subheader("Missing Values")
        missing_values = df.isnull().sum().reset_index()
        missing_values.columns = ["Column", "Missing Count"]
        missing_values["Missing Percentage"] = (missing_values["Missing Count"] / len(df)) * 100
        st.write(missing_values)

        # Handle missing values using SimpleImputer with median strategy for numeric columns
        imputer = SimpleImputer(strategy='median')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # Preprocess categorical columns
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))

        # Preprocess target variable based on task type
        if st.session_state['task'] == 'Classification':
            if 'diagnosis' not in df.columns:
                st.error("The dataset must contain a 'diagnosis' column for classification.")
            else:
                # Preparing Data
                le = LabelEncoder()
                df['diagnosis'] = le.fit_transform(df['diagnosis'])  # M -> 1, B -> 0
                st.session_state['label_encoder'] = le  # Save encoder in session state

                if le is not None:
                    label_mapping = {str(class_label): int(encoded_label) for class_label, encoded_label in zip(le.classes_, le.transform(le.classes_))}
                    st.write("Label Mapping (Original -> Encoded):", label_mapping)
                else:
                    st.warning("Label Encoder is not available. Ensure the dataset has been preprocessed.")
                            
                X = df.drop(columns=['diagnosis']).values
                Y = df['diagnosis'].values

                # Ensure the target is discrete for classification tasks
                if pd.isnull(Y).any() or not np.issubdtype(Y.dtype, np.integer):
                    st.error("The target variable for classification must be discrete (integer values) and contain no missing values. Please clean your data and try again.")
        # elif st.session_state['task'] == 'Regression':
        #     # Ensure 'pm2.5' column exists in the dataset
        #     if 'pm2.5' not in df.columns:
        #         st.error("The dataset must contain a 'pm2.5' column for regression.")
        #     else:
        #         # Preparing Data
        #         X = df.drop(columns=['pm2.5']).values
        #         Y = df['pm2.5'].values
        st.session_state['X'] = X
        st.session_state['Y'] = Y

        


# Machine Learning Models
with tab2:
    st.header("Machine Learning Models")

    if st.session_state['task'] == 'Classification':
        # Ensure X and Y are available in session state
        if 'X' not in st.session_state or 'Y' not in st.session_state:
            st.warning("Please upload a dataset in the 'Dataset Overview' tab.")
        else:
            X = st.session_state['X']
            Y = st.session_state['Y']

            # Dropdown to select ML Algorithm
            ml_algorithm = st.selectbox("Select Machine Learning Algorithm", [
                "Decision Tree",
                "Gaussian Naive Bayes",
                "AdaBoost",
                "K-Nearest Neighbors",
                "Logistic Regression",
                "MLP Classifier",
                "Perceptron",
                "Random Forest",
                "Support Vector Machine (SVM)"
            ])

            # Split dataset into training and testing sets
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
            random_seed = st.slider("Random Seed", 1, 100, 42)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Save split data in session state
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['Y_train'] = Y_train
            st.session_state['Y_test'] = Y_test

            # Train and save the selected model
            model = None
            if ml_algorithm == "Decision Tree":
                model = decision_tree_ui(X_train, Y_train,X_test,Y_test,random_seed)
                st.session_state['model'] = model
                st.success("Model trained and saved successfully.")
            elif ml_algorithm == "Gaussian Naive Bayes":
                model = gaussian_naive_bayes_ui(X_train, Y_train,X_test,Y_test)
                st.session_state['model'] = model
                st.success("Model trained and saved successfully.")
            elif ml_algorithm == "AdaBoost":
                model = adaboost_ui(X_train, Y_train,X_test,Y_test,random_seed)
                st.session_state['model'] = model
                st.success("Model trained and saved successfully.")
            elif ml_algorithm == "K-Nearest Neighbors":
                model = knn_ui(X_train, Y_train,X_test,Y_test)
                st.session_state['model'] = model
                st.success("Model trained and saved successfully.")
            elif ml_algorithm == "Logistic Regression":
                model = logistic_regression_ui(X_train, Y_train,X_test,Y_test)
                st.session_state['model'] = model
                st.success("Model trained and saved successfully.")
            elif ml_algorithm == "MLP Classifier":
                model = mlp_classifier_ui(X_train, Y_train,X_test,Y_test,random_seed)
                st.session_state['model'] = model
                st.success("Model trained and saved successfully.")
            elif ml_algorithm == "Perceptron":
                model = perceptron_ui(X_train, Y_train,X_test,Y_test,random_seed)
                st.session_state['model'] = model
                st.success("Model trained and saved successfully.")
            elif ml_algorithm == "Random Forest":
                model = random_forest_ui(X_train, Y_train,X_test,Y_test,random_seed)
                st.session_state['model'] = model
                st.success("Model trained and saved successfully.")
            elif ml_algorithm == "Support Vector Machine (SVM)":
                model = svm_ui(X_train, Y_train,X_test,Y_test,random_seed)
                st.session_state['model'] = model
                st.success("Model trained and saved successfully.")

            


# Resampling and Training
with tab3:
    st.header("Resampling and Training")

    if st.session_state['task'] == 'Classification':
        if 'X_test' not in st.session_state or 'Y_test' not in st.session_state or 'model' not in st.session_state:
            st.warning("Please upload a dataset, select a task, and train a model in the 'ML Models' tab.")
        else:
            X_train = st.session_state['X_test']
            Y_train = st.session_state['Y_test']
            model = st.session_state['model']

            st.subheader("Classification Resampling Techniques")

            # Dropdown to select resampling technique
            resampling_techniques = st.selectbox(
                "Select Resampling Technique",
                ["K-Fold Cross Validation", "Leave-One-Out Cross Validation"]
            )

            # Perform the selected resampling technique
            if resampling_techniques == "K-Fold Cross Validation":
                num_folds = st.slider("Select number of folds for K-Fold Cross Validation:", 2, 10, 5)
                kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
                results = cross_val_score(model, X_train, Y_train, cv=kfold)
                st.write(f"Accuracy: {results.mean() * 100:.2f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.2f}%")

            elif resampling_techniques == "Leave-One-Out Cross Validation":
                loocv = LeaveOneOut()
                results = cross_val_score(model, X_train, Y_train, cv=loocv)
                st.write(f"Accuracy: {results.mean() * 100:.2f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.2f}%")
            
            if st.button("Train Model"):
                # Train the model only if no missing values exist
                if not pd.isnull(Y).any():
                    model.fit(X, Y)
                    model_filename = f"{st.session_state['task'].lower()}_{ml_algorithm.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').lower()}.joblib"

                    try:
                        joblib.dump(model, model_filename)
                        st.success(f"Model saved as {model_filename}")

                        # Option to download the model
                        with open(model_filename, "rb") as f:
                            st.download_button("Download Trained Model", f, file_name=model_filename)
                    except Exception as e:
                        st.error(f"An error occurred while saving the model: {e}")
                else:
                    st.error("Cannot train model with missing values in the target variable.")



# Visualizations
with tab4:
    st.header("Visualizations")
    st.write("Generate plots and insights here.")

    if st.session_state['task'] == 'Classification':
        if 'model' not in st.session_state or 'X' not in st.session_state or 'Y' not in st.session_state:
            st.warning("Please train a model and split the data in the 'ML Models' tab.")
        else:
            X = st.session_state['X']
            Y = st.session_state['Y']
            model = st.session_state['model']

            # Generate predictions
            predictions = model.predict(X)

            # Retrieve label encoder and decode labels
            if 'label_encoder' in st.session_state:
                le = st.session_state['label_encoder']
                decoded_predictions = le.inverse_transform(predictions)  # Decode predictions
                decoded_Y = le.inverse_transform(Y)  # Decode true labels
                class_labels = le.classes_  # ['B', 'M']
            else:
                # Fallback if LabelEncoder is not found
                decoded_predictions = predictions
                decoded_Y = Y
                class_labels = np.unique(Y)

            # Classification Metrics
            st.subheader("Classification Metrics")
            accuracy = accuracy_score(decoded_Y, decoded_predictions)
            log_loss_value = log_loss(decoded_Y, model.predict_proba(X))
            st.write(f"Classification Accuracy: {accuracy * 100:.2f}%")
            st.write(f"Logarithmic Loss: {log_loss_value:.2f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            confusion_mat = confusion_matrix(decoded_Y, decoded_predictions, labels=class_labels)
            confusion_df = pd.DataFrame(
                confusion_mat,
                index=[f"True {label}" for label in class_labels],
                columns=[f"Pred {label}" for label in class_labels]
            )
            st.write(confusion_df)

            # Classification Report in Tabular Form
            st.subheader("Classification Report")
            report = classification_report(
                decoded_Y, decoded_predictions, output_dict=True, target_names=class_labels
            )
            report_df = pd.DataFrame(report).transpose()

            # Ensure all numeric columns are of type float for compatibility
            numeric_cols = report_df.select_dtypes(include=['number']).columns
            report_df[numeric_cols] = report_df[numeric_cols].astype(float)

            # Ensure all non-numeric columns are of type string for compatibility
            non_numeric_cols = report_df.select_dtypes(exclude=['number']).columns
            report_df[non_numeric_cols] = report_df[non_numeric_cols].astype(str)

            # Reset index to avoid serialization issues
            report_df.reset_index(inplace=True)

            # Display the DataFrame in Streamlit
            st.dataframe(report_df)

            # ROC Curve for binary classification
            if len(np.unique(Y)) == 2:  # Ensure binary classification for ROC AUC
                st.subheader("Receiver Operating Characteristic (ROC) Curve")
                roc_auc = roc_auc_score(Y, model.predict_proba(X)[:, 1])
                st.write(f"Area Under ROC Curve (AUC): {roc_auc:.2f}")
                fpr, tpr, _ = roc_curve(Y, model.predict_proba(X)[:, 1])
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label='ROC Curve')
                ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend()
                st.pyplot(fig)


# Model Comparison
with tab5:
    st.header("Model Comparison")
    st.write("Compare model performances here.")

with tab6:
    st.header("Prediction")
    st.write("Predict using trained models.")

    st.subheader("Upload a Saved Model for Prediction")
    uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"], key="model_upload")

    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        st.write("Model Loaded Successfully!")

        # Predict using entire dataset (X)
        if uploaded_file is not None:
            # Input fields for breast cancer prediction
            st.subheader("Input Sample Data for Breast Cancer Prediction")

            # Feature input fields
            radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
            area_mean = st.number_input("Area Mean", min_value=0.0, max_value=5000.0, value=0.0, step=0.1)
            smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

            # Combine input features into a single array
            input_data = np.array([[
                radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                fractal_dimension_mean
            ]])

            # Predict using the trained model
            if st.button("Predict"):
                if 'model' in st.session_state:
                    model = st.session_state['model']
                    prediction = model.predict(input_data)
                    
                    # Decode prediction if label encoder is available
                    if 'label_encoder' in st.session_state:
                        le = st.session_state['label_encoder']
                        decoded_prediction = le.inverse_transform(prediction)[0]  # Malignant (M) or Benign (B)
                    else:
                        decoded_prediction = prediction[0]  # Fallback to numeric

                    st.subheader("Prediction Result")
                    st.write(f"The predicted diagnosis is: {decoded_prediction}")
                else:
                    st.warning("Please train a model in the 'ML Models' tab first.")

