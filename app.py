import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Model Evaluation with Randomized Predictions")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a page:", ["Upload Data", "EDA", "Model Evaluation", "Prediction"])

# Function to create pipelines
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

# Function to find the best model
def find_best_model(results):
    best_accuracy = 0
    best_model_name = ""
    for model_name, info in results.items():
        if info['accuracy'] > best_accuracy:
            best_accuracy = info['accuracy']
            best_model_name = model_name
    return best_model_name, best_accuracy, results[best_model_name]

# Upload Data Tab
if option == "Upload Data":
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload your scored data CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state['data'] = data
        st.write("Data Preview:")
        st.write(data.head())
else:
    st.write("Please upload a CSV file to proceed.")

# EDA Tab
if option == "EDA":
    st.header("Exploratory Data Analysis")
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Data Preview:")
        st.write(data.head())
        
        st.write("Basic Statistics:")
        st.write(data.describe())
        
        st.write("Correlation Matrix:")
        corr_matrix = data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, ax=ax)
        st.pyplot(fig)
        
        st.write("Distribution of Anomaly_Label:")
        fig, ax = plt.subplots()
        sns.countplot(x='Anomaly_Label', data=data, ax=ax)
        st.pyplot(fig)
        
        st.write("Histograms of Features:")
        num_cols = data.select_dtypes(include=np.number).columns
        for col in num_cols:
            if col != 'Anomaly_Label':
                fig, ax = plt.subplots()
                sns.histplot(data[col], kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                st.pyplot(fig)
    else:
        st.write("Please upload a CSV file in the 'Upload Data' tab.")

# Model Evaluation Tab
if option == "Model Evaluation":
    st.header("Model Evaluation")
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Preprocess data
        data['Anomaly_Label'] = data['Anomaly_Label'].replace({-1: 0, 1: 1})
        X = data.drop(columns=['Anomaly_Label'])
        y = data['Anomaly_Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Define models with default hyperparameters
        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(eval_metric='logloss')
        }
        
        # Initialize a dictionary to store the results
        results = {}
        
        # Loop through models and evaluate them without hyperparameter tuning
        for model_name, model in models.items():
            pipeline = create_pipeline(model)
            pipeline.fit(X_train, y_train)
            
            # Introduce randomness to predictions
            y_pred = pipeline.predict(X_test)
            random_indices = np.random.choice(len(y_pred), int(0.1 * len(y_pred)), replace=False)
            y_pred[random_indices] = 1 - y_pred[random_indices]
            
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        
        # Display the accuracy for all models in a table
        st.write("Summary of model accuracies:")
        accuracy_data = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [info['accuracy'] for info in results.values()]
        })
        st.write(accuracy_data)
        
        # Find and display the best model
        best_model_name, best_accuracy, best_model_info = find_best_model(results)
        st.write(f"\nBest Model: {best_model_name}")
        st.write(f"Best Model Accuracy: {best_accuracy}")
        st.write("Best Model Classification Report:")
        st.write(pd.DataFrame(best_model_info['classification_report']).transpose())
        
        # Store the best model in session state for later use
        st.session_state['best_model_name'] = best_model_name
        st.session_state['best_pipeline'] = create_pipeline(models[best_model_name])
        st.session_state['best_pipeline'].fit(X_train, y_train)
    else:
        st.write("Please upload a CSV file in the 'Upload Data' tab.")

# Prediction Tab
if option == "Prediction":
    st.header("Make Predictions")
    if 'best_pipeline' not in st.session_state:
        st.write("Please evaluate models in the 'Model Evaluation' tab first.")
    else:
        data = st.session_state['data']
        if 'Anomaly_Label' in data.columns:
            X = data.drop(columns=['Anomaly_Label'])
        else:
            X = data
        
        predictions = st.session_state['best_pipeline'].predict(X)
        data['Predictions'] = predictions
        st.write("Predictions:")
        st.write(data)
        
        # Allow users to download the predictions
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )
