import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, ShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Title of the app
st.title('Utilizing Resampling Techniques and Performance Metrics for Classification, Regression, and Time Series Analysis')

# Task Selection
task = st.sidebar.selectbox('Select Task', ['Classification', 'Regression'])

# File uploader for dataset
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Drop any columns that have all missing values
    df.dropna(axis=1, how='all', inplace=True)

    # Handle missing values using SimpleImputer with median strategy for numeric columns
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Convert categorical columns to numeric using LabelEncoder
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

    # Model Selection based on Task
    if task == 'Classification':
        st.sidebar.subheader('Classification Models')
        model_option = st.sidebar.radio('Select Model', ['Model A (K-Fold Cross Validation)', 'Model B (Leave-One-Out Cross Validation)'])

        # Ensure 'diagnosis' column exists in the dataset
        if 'diagnosis' not in df.columns:
            st.error("The dataset must contain a 'diagnosis' column for classification.")
        else:
            # Preparing Data
            X = df.drop(columns=['diagnosis']).values
            Y = df['diagnosis'].values

            # Ensure the target is discrete for classification tasks
            if pd.isnull(Y).any() or not np.issubdtype(Y.dtype, np.integer):
                st.error("The target variable for classification must be discrete (integer values) and contain no missing values. Please clean your data and try again.")
            else:
                model = LogisticRegression(max_iter=200)

                if model_option == 'Model A (K-Fold Cross Validation)':
                    # K-Fold Cross Validation
                    num_folds = st.slider("Select number of folds for K-Fold Cross Validation:", 2, 10, 5)
                    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
                    results = cross_val_score(model, X, Y, cv=kfold)
                    st.write(f"Accuracy: {results.mean() * 100:.2f}%")
                    st.write(f"Standard Deviation: {results.std() * 100:.2f}%")

                elif model_option == 'Model B (Leave-One-Out Cross Validation)':
                    # Leave-One-Out Cross Validation
                    loocv = LeaveOneOut()
                    results = cross_val_score(model, X, Y, cv=loocv)
                    st.write(f"Accuracy: {results.mean() * 100:.2f}%")
                    st.write(f"Standard Deviation: {results.std() * 100:.2f}%")

                # Train model and display additional metrics
                model.fit(X, Y)
                predictions = model.predict(X)
                st.subheader("Classification Metrics")
                st.write(f"Classification Accuracy: {accuracy_score(Y, predictions) * 100:.2f}%")
                st.write(f"Logarithmic Loss: {log_loss(Y, model.predict_proba(X)):.2f}")
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(Y, predictions))
                st.write("Classification Report:")
                st.text(classification_report(Y, predictions))

                # ROC Curve for binary classification
                if len(np.unique(Y)) == 2:  # Ensure binary classification for ROC AUC
                    st.write(f"Area Under ROC Curve (AUC): {roc_auc_score(Y, model.predict_proba(X)[:, 1]):.2f}")
                    fpr, tpr, _ = roc_curve(Y, model.predict_proba(X)[:, 1])
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label='ROC Curve')
                    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                    ax.legend()
                    st.pyplot(fig)

    elif task == 'Regression':
        st.sidebar.subheader('Regression Models')
        model_option = st.sidebar.radio('Select Model', ['Model A (Train/Test Split)', 'Model B (Repeated Random Test-Train Splits)'])

        # Ensure 'pm2.5' column exists in the dataset
        if 'pm2.5' not in df.columns:
            st.error("The dataset must contain a 'pm2.5' column for regression.")
        else:
            # Preparing Data
            X = df.drop(columns=['pm2.5']).values
            Y = df['pm2.5'].values

            model = LinearRegression()

            # Ensure target for regression is continuous
            if pd.isnull(Y).any() or not (np.issubdtype(Y.dtype, np.floating) or np.issubdtype(Y.dtype, np.integer)):
                st.error("The target variable for regression must be continuous and contain no missing values. Please clean your data and try again.")
            else:
                if model_option == 'Model A (Train/Test Split)':
                    # Train/Test Split
                    test_size = st.slider("Select Test Size (as a percentage)", 10, 50, 20) / 100
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=1)
                    model.fit(X_train, Y_train)
                    predictions = model.predict(X_test)
                    st.subheader("Regression Metrics")
                    st.write(f"Mean Squared Error (MSE): {mean_squared_error(Y_test, predictions):.2f}")
                    st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(Y_test, predictions):.2f}")
                    st.write(f"R-squared (R²): {r2_score(Y_test, predictions):.2f}")

                elif model_option == 'Model B (Repeated Random Test-Train Splits)':
                    # Repeated Random Test-Train Splits
                    n_splits = st.slider("Select number of splits:", 2, 20, 10)
                    test_size = st.slider("Select test size proportion:", 0.1, 0.5, 0.33)
                    shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=1)
                    results_mse = cross_val_score(model, X, Y, cv=shuffle_split, scoring='neg_mean_squared_error')
                    results_mae = cross_val_score(model, X, Y, cv=shuffle_split, scoring='neg_mean_absolute_error')
                    results_r2 = cross_val_score(model, X, Y, cv=shuffle_split, scoring='r2')
                    st.subheader("Regression Metrics")
                    st.write(f"Mean Squared Error (MSE): {-results_mse.mean():.2f}")
                    st.write(f"Mean Absolute Error (MAE): {-results_mae.mean():.2f}")
                    st.write(f"R-squared (R²): {results_r2.mean():.2f}")

    # Train Model and Save
    st.subheader("Train and Save Model")
    if st.button("Train Model"):
        # Train the model only if no missing values exist
        if not pd.isnull(Y).any():
            model.fit(X, Y)
            model_filename = f"{task.lower()}_{model_option.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').lower()}.joblib"

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

    # Model upload for prediction
    st.subheader("Upload a Saved Model for Prediction")
    uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"], key="model_upload")

    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        st.write("Model Loaded Successfully!")

        # Predict using entire dataset (X)
        if uploaded_file is not None:
            predictions = model.predict(X)
            st.subheader("Prediction Results")
            # Creating a DataFrame to display predictions alongside original features
            prediction_df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
            prediction_df["Prediction"] = predictions
            st.write(prediction_df)
