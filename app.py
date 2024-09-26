import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set page configuration
st.set_page_config(layout="wide")

# Function Definitions
def train_model(algo_name, X_train, y_train, problem_type):
    """
    Trains the specified algorithm using GridSearchCV and returns the best model and parameters.

    Parameters:
    - algo_name (str): Name of the algorithm to train.
    - X_train (DataFrame): Training feature set.
    - y_train (Series): Training target variable.
    - problem_type (str): "Regression" or "Classification".

    Returns:
    - best_model: Trained model with the best parameters.
    - best_params: Best hyperparameters found during GridSearchCV.
    """
    if problem_type == "Regression":
        if algo_name == "Linear Regression":
       
            model = LinearRegression()
            params = {
                'fit_intercept': [True, False]
            }
        elif algo_name == "Decision Tree Regressor":

            model = DecisionTreeRegressor()
            params = {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif algo_name == "Random Forest Regressor":

            model = RandomForestRegressor()
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif algo_name == "Support Vector Regressor":

            model = SVR()
            params = {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        elif algo_name == "K-Nearest Neighbors Regressor":
    
            model = KNeighborsRegressor()
            params = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        else:
            st.error(f"Algorithm '{algo_name}' is not supported for regression.")
            return None, None

    elif problem_type == "Classification":
        if algo_name == "Logistic Regression":

            model = LogisticRegression(max_iter=1000)
            params = {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        elif algo_name == "Decision Tree Classifier":

            model = DecisionTreeClassifier()
            params = {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif algo_name == "Random Forest Classifier":

            model = RandomForestClassifier()
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif algo_name == "Support Vector Classifier":

            model = SVC()
            params = {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        elif algo_name == "K-Nearest Neighbors Classifier":

            model = KNeighborsClassifier()
            params = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        else:
            st.error(f"Algorithm '{algo_name}' is not supported for classification.")
            return None, None

    else:
        st.error("Invalid problem type specified.")
        return None, None

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params




st.title("Machine Learning Predictive Model Simulator")
st.markdown("""
This simulator allows you to upload your dataset, select variables, choose algorithms, and visualize the results.
""")

# Tabs for navigation
tabs = st.tabs(["Training", "Prediction"])

with tabs[0]:
    st.header("Training Section")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"], key='train_file')

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head())

        # Data Exploration
        if st.checkbox("Show Data Types"):
            st.write(data.dtypes)
        if st.checkbox("Show Missing Values"):
            st.write(data.isnull().sum())
        if st.checkbox("Show Descriptive Statistics"):
            st.write(data.describe())

        # Data Visualization
        if st.checkbox("Show Correlation Heatmap"):
            st.write("Correlation Heatmap:")
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        if st.checkbox("Show Pairplot"):
            st.write("Pairplot:")
            fig = sns.pairplot(data)
            st.pyplot(fig)

        # Data Preprocessing
        missing_value_option = st.selectbox(
            "Choose how to handle missing values:",
            ("Do nothing", "Drop rows with missing values", "Impute missing values")
        )

        if missing_value_option == "Drop rows with missing values":
            data = data.dropna()
            st.write("Dropped rows with missing values.")
        elif missing_value_option == "Impute missing values":
            imputer = SimpleImputer(strategy='mean')
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            st.write("Imputed missing values with mean.")

        # Encoding Categorical Variables
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) > 0:
            st.write("Encoding categorical variables using One-Hot Encoding:")
            data_encoded = pd.get_dummies(
                data, 
                columns=categorical_cols, 
                drop_first=True, 
                prefix=categorical_cols
            )
            st.write("Categorical variables encoded.")
            st.write("Updated Data Preview:")
            st.dataframe(data_encoded.head())
        else:
            data_encoded = data.copy()

        # Feature Scaling
        scaling_option = st.selectbox(
            "Choose how to scale features:",
            ("Do nothing", "StandardScaler", "MinMaxScaler")
        )

        scaler = None
        if scaling_option == "StandardScaler":
            scaler = StandardScaler()
            data_encoded[data_encoded.columns] = scaler.fit_transform(data_encoded[data_encoded.columns])
            st.write("Features scaled using StandardScaler.")
        elif scaling_option == "MinMaxScaler":
            scaler = MinMaxScaler()
            data_encoded[data_encoded.columns] = scaler.fit_transform(data_encoded[data_encoded.columns])
            st.write("Features scaled using MinMaxScaler.")

        # Variable Selection
        all_columns = data_encoded.columns.tolist()
        target_variable = st.selectbox("Select the Dependent Variable (Target)", all_columns)
        feature_variables = st.multiselect(
            "Select Independent Variables (Features)", 
            [col for col in all_columns if col != target_variable]
        )

        if len(feature_variables) == 0:
            st.error("Please select at least one independent variable.")
        else:
            X = data_encoded[feature_variables]
            y = data_encoded[target_variable]

            # Problem Type Selection
            problem_type = st.selectbox("Select Problem Type", ("Regression", "Classification"))

            # Algorithm Selection
            if problem_type == "Regression":
                algorithms = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor",
                              "Support Vector Regressor", "K-Nearest Neighbors Regressor"]
            else:
                algorithms = ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier",
                              "Support Vector Classifier", "K-Nearest Neighbors Classifier"]

            selected_algorithms = st.multiselect("Select Algorithms to Run", algorithms, default=algorithms)

            # Data Splitting
            test_size = st.slider("Select Test Data Size (%):", 10, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42
            )
            st.write(f"Data split into training and testing sets with test size {test_size}%.")

            # Training and Evaluation
            if st.button("Run Models"):
                model_performance = {}
                best_models = {}
                progress_bar = st.progress(0)
                total_algorithms = len(selected_algorithms)

                for idx, algo in enumerate(selected_algorithms):
                    st.write(f"Training **{algo}**...")
                    best_model, best_params = train_model(algo, X_train, y_train, problem_type)

                    if best_model is not None:
                        y_pred = best_model.predict(X_test)

                        if problem_type == "Regression":
                            # Regression Metrics
            

                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)

                            model_performance[algo] = {
                                'R-squared': r2,
                                'MSE': mse,
                                'MAE': mae,
                                'Best Parameters': best_params
                            }
                        else:
                            # Classification Metrics
         

                            acc = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            cm = confusion_matrix(y_test, y_pred)

                            model_performance[algo] = {
                                'Accuracy': acc,
                                'Precision': precision,
                                'Recall': recall,
                                'F1 Score': f1,
                                'Confusion Matrix': cm,
                                'Best Parameters': best_params
                            }

                        best_models[algo] = best_model

                        st.success(f"Training completed for {algo}.")
                    else:
                        st.error(f"Training failed for {algo}.")

                    # Update progress bar
                    progress = (idx + 1) / total_algorithms
                    progress_bar.progress(progress)

                progress_bar.empty()  # Remove the progress bar after completion

                # Display model performance
                st.header("Model Performance Results")

                for algo, metrics in model_performance.items():
                    st.subheader(f"Algorithm: {algo}")
                    if problem_type == "Regression":
                        st.write(f"**R-squared**: {metrics['R-squared']:.4f}")
                        st.write(f"**Mean Squared Error (MSE)**: {metrics['MSE']:.4f}")
                        st.write(f"**Mean Absolute Error (MAE)**: {metrics['MAE']:.4f}")
                    else:
                        st.write(f"**Accuracy**: {metrics['Accuracy']:.4f}")
                        st.write(f"**Precision**: {metrics['Precision']:.4f}")
                        st.write(f"**Recall**: {metrics['Recall']:.4f}")
                        st.write(f"**F1 Score**: {metrics['F1 Score']:.4f}")
                        st.write("**Confusion Matrix**:")
                        st.write(metrics['Confusion Matrix'])

                    st.write("**Best Hyperparameters**:")
                    st.write(metrics['Best Parameters'])

                    # Visualizations
                    st.subheader(f"Visualization for {algo}")
                    if problem_type == "Regression":
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred, alpha=0.5)
                        ax.plot(y_test, y_test, color='red', linestyle='--')
                        ax.set_xlabel('Actual Values')
                        ax.set_ylabel('Predicted Values')
                        ax.set_title(f"Actual vs Predicted Values for {algo}")
                        st.pyplot(fig)
                    else:
                        fig, ax = plt.subplots()
                        sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted Labels')
                        ax.set_ylabel('True Labels')
                        ax.set_title(f"Confusion Matrix for {algo}")
                        st.pyplot(fig)

                # Selecting the Best Model
                if problem_type == "Regression":
                    best_algo = max(model_performance, key=lambda x: model_performance[x]['R-squared'])
                    st.success(f"The best model is **{best_algo}** with R-squared: {model_performance[best_algo]['R-squared']:.4f}")
                else:
                    best_algo = max(model_performance, key=lambda x: model_performance[x]['Accuracy'])
                    st.success(f"The best model is **{best_algo}** with Accuracy: {model_performance[best_algo]['Accuracy']:.4f}")

                # Store the best model and its details in session state
                st.session_state['best_model'] = best_models[best_algo]
                st.session_state['best_model_name'] = best_algo
                st.session_state['problem_type'] = problem_type
                st.session_state['feature_variables'] = feature_variables
                st.session_state['scaler'] = scaler  # If scaling was applied
                st.session_state['categorical_cols'] = categorical_cols
                if len(categorical_cols) > 0:
                    st.session_state['training_dummy_columns'] = data_encoded.columns.tolist()
                else:
                    st.session_state['training_dummy_columns'] = None

    else:
        st.info("Please upload a CSV file to proceed.")

with tabs[1]:
    st.header("Prediction Section")
    # Check if a model has been trained and stored
    if 'best_model' in st.session_state:
        st.write(f"Using the best model: **{st.session_state['best_model_name']}**")
        
        prediction_option = st.radio(
            "Select Prediction Method:",
            ("Upload CSV File for Prediction", "Manual Data Entry")
        )

        if prediction_option == "Upload CSV File for Prediction":
            pred_file = st.file_uploader("Upload CSV file with independent variables", type=["csv"], key='pred_file')
            if pred_file is not None:
                pred_data = pd.read_csv(pred_file)
                st.write("Prediction Data Preview:")
                st.dataframe(pred_data.head())

                # Ensure that the prediction data has the necessary features
                missing_cols = set(st.session_state['feature_variables']) - set(pred_data.columns)
                if missing_cols:
                    st.error(f"The following required columns are missing in the uploaded data: {missing_cols}")
                else:
                    # Preprocess the prediction data
                    pred_data_processed = pred_data.copy()

                    # Encoding
                    categorical_cols = st.session_state['categorical_cols']
                    if len(categorical_cols) > 0:
                        for col in categorical_cols:
                            if col in pred_data_processed.columns:
                                dummies = pd.get_dummies(pred_data_processed[col], prefix=col, drop_first=True)
                                pred_data_processed = pd.concat([pred_data_processed, dummies], axis=1)
                                pred_data_processed.drop(col, axis=1, inplace=True)
                            else:
                                st.error(f"Column {col} is missing from the prediction data.")
                                st.stop()

                        # Ensure all dummy columns are present
                        training_dummy_columns = st.session_state['training_dummy_columns']
                        missing_dummies = set(training_dummy_columns) - set(pred_data_processed.columns)
                        for col in missing_dummies:
                            pred_data_processed[col] = 0

                        # Reorder columns to match training data
                        pred_data_processed = pred_data_processed[training_dummy_columns]

                    # Scaling
                    if st.session_state['scaler'] is not None:
                        pred_data_processed[st.session_state['feature_variables']] = st.session_state['scaler'].transform(pred_data_processed[st.session_state['feature_variables']])

                    # Make predictions
                    best_model = st.session_state['best_model']
                    predictions = best_model.predict(pred_data_processed[st.session_state['feature_variables']])

                    # Append predictions to the data
                    pred_data['Predicted_' + st.session_state['problem_type']] = predictions
                    st.write("Predictions:")
                    st.dataframe(pred_data.head())

                    # Provide option to download the predictions
                    @st.cache_data
                    def convert_df(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv = convert_df(pred_data)

                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv',
                    )
        else:
            st.write("Please input values for the following features:")
            input_data = {}
            for feature in st.session_state['feature_variables']:
                input_val = st.text_input(f"Enter value for {feature}", key=f"input_{feature}")
                input_data[feature] = input_val

            if st.button("Predict", key='manual_predict'):
                # Convert input data to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Data type conversion
                for col in input_df.columns:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

                if input_df.isnull().values.any():
                    st.error("Please ensure all inputs are valid numbers.")
                else:
                    # Preprocess input data
                    # Encoding
                    categorical_cols = st.session_state['categorical_cols']
                    if len(categorical_cols) > 0:
                        for col in categorical_cols:
                            if col in input_df.columns:
                                dummies = pd.get_dummies(input_df[col], prefix=col, drop_first=True)
                                input_df = pd.concat([input_df, dummies], axis=1)
                                input_df.drop(col, axis=1, inplace=True)
                            else:
                                # If the column was not entered (possible in manual entry), add zeros
                                for category in st.session_state['training_dummy_columns']:
                                    if category.startswith(col + '_'):
                                        input_df[category] = 0

                        # Ensure all dummy columns are present
                        training_dummy_columns = st.session_state['training_dummy_columns']
                        missing_dummies = set(training_dummy_columns) - set(input_df.columns)
                        for col in missing_dummies:
                            input_df[col] = 0

                        # Reorder columns to match training data
                        input_df = input_df[training_dummy_columns]

                    # Scaling
                    if st.session_state['scaler'] is not None:
                        input_df[st.session_state['feature_variables']] = st.session_state['scaler'].transform(input_df[st.session_state['feature_variables']])

                    # Make prediction
                    best_model = st.session_state['best_model']
                    prediction = best_model.predict(input_df[st.session_state['feature_variables']])

                    st.success(f"Predicted {st.session_state['problem_type']}: {prediction[0]}")

    else:
        st.warning("Please train a model in the Training section before making predictions.")



if st.button("Run Models", key="run_models_train"):
    model_performance = {}
    best_models = {}
    progress_bar = st.progress(0)
    total_algorithms = len(selected_algorithms)

    for idx, algo in enumerate(selected_algorithms):
        st.write(f"Training **{algo}**...")
        best_model, best_params = train_model(algo, X_train, y_train, problem_type)

        if best_model is not None:
            y_pred = best_model.predict(X_test)

            if problem_type == "Regression":
                # Regression Metrics
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                model_performance[algo] = {
                    'R-squared': r2,
                    'MSE': mse,
                    'MAE': mae,
                    'Best Parameters': best_params
                }
            else:
                # Classification Metrics
                acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)

                model_performance[algo] = {
                    'Accuracy': acc,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'Confusion Matrix': cm,
                    'Best Parameters': best_params
                }

            best_models[algo] = best_model

            st.success(f"Training completed for {algo}.")
        else:
            st.error(f"Training failed for {algo}.")

        # Update progress bar
        progress = (idx + 1) / total_algorithms
        progress_bar.progress(progress)

    progress_bar.empty()  # Remove the progress bar after completion

    # Display model performance
    st.header("Model Performance Results")

    for algo, metrics in model_performance.items():
        st.subheader(f"Algorithm: {algo}")
        if problem_type == "Regression":
            st.write(f"**R-squared**: {metrics['R-squared']:.4f}")
            st.write(f"**Mean Squared Error (MSE)**: {metrics['MSE']:.4f}")
            st.write(f"**Mean Absolute Error (MAE)**: {metrics['MAE']:.4f}")
        else:
            st.write(f"**Accuracy**: {metrics['Accuracy']:.4f}")
            st.write(f"**Precision**: {metrics['Precision']:.4f}")
            st.write(f"**Recall**: {metrics['Recall']:.4f}")
            st.write(f"**F1 Score**: {metrics['F1 Score']:.4f}")
            st.write("**Confusion Matrix**:")
            st.write(metrics['Confusion Matrix'])

        st.write("**Best Hyperparameters**:")
        st.write(metrics['Best Parameters'])

        # Visualizations
        st.subheader(f"Visualization for {algo}")
        if problem_type == "Regression":
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot(y_test, y_test, color='red', linestyle='--')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f"Actual vs Predicted Values for {algo}")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title(f"Confusion Matrix for {algo}")
            st.pyplot(fig)

    # Selecting the Best Model
    if problem_type == "Regression":
        best_algo = max(model_performance, key=lambda x: model_performance[x]['R-squared'])
        st.success(f"The best model is **{best_algo}** with R-squared: {model_performance[best_algo]['R-squared']:.4f}")
    else:
        best_algo = max(model_performance, key=lambda x: model_performance[x]['Accuracy'])
        st.success(f"The best model is **{best_algo}** with Accuracy: {model_performance[best_algo]['Accuracy']:.4f}")

    # Optionally, allow the user to download the best model or save it for future use
    # You can use joblib or pickle to save the model

# Identify categorical columns
if uploaded_file is not None: # This is the fix!
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) > 0:
        st.write("Encoding categorical variables using One-Hot Encoding:")
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True, prefix=categorical_cols)
        st.write("Categorical variables encoded.")
        st.write("Updated Data Preview:")
        st.dataframe(data_encoded.head())
    else:
        data_encoded = data.copy()

    # Update feature variables if they were affected by encoding
    if uploaded_file is not None and len(categorical_cols) > 0:
        # Reconstruct the feature variable list after encoding
        all_columns = data_encoded.columns.tolist()
        feature_variables = [col for col in all_columns if col != target_variable]
        X = data_encoded[feature_variables]
        y = data_encoded[target_variable]
    else:
        X = data[feature_variables]
        y = data[target_variable]