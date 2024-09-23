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
import shap

st.title("Predictive Model")

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

# Load the data
data = pd.read_csv('anomaly_1.csv')  # Path to your data file
st.session_state['data'] = data

# Initialize session state variables
if 'best_model_name' not in st.session_state:
    st.session_state['best_model_name'] = None
if 'best_pipeline' not in st.session_state:
    st.session_state['best_pipeline'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = None
if 'shap_values' not in st.session_state:
    st.session_state['shap_values'] = None

# Tabs for navigation
tabs = st.tabs(["EDA", "Modelling", "Scoring"])

# EDA Tab
with tabs[0]:
    st.header("Exploratory Data Analysis")
    if data is not None:
        st.write("Data Preview:")
        st.write(data.head())
        
        st.write("Basic Statistics:")
        st.write(data.describe())
        
        st.write("Correlation Matrix:")
        corr_matrix = data.corr()
        st.write(corr_matrix)  # Display the correlation matrix values
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, ax=ax)
        st.pyplot(fig)
        
        st.write("Distribution of Anomaly_Label:")
        fig, ax = plt.subplots()
        sns.countplot(x='Anomaly_Label', data=data, ax=ax)
        st.pyplot(fig)
        
        # Calculate and display percentages
        anomaly_counts = data['Anomaly_Label'].value_counts(normalize=True) * 100
        for label, percentage in anomaly_counts.items():
            st.write(f"Percentage of {label}: {percentage:.2f}%")

        st.write("Line Chart of Numerical Features:")
        num_cols = data.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            st.line_chart(data[num_cols])
        else:
            st.write("No numerical features to display.")
            
# Display Variable Importance and SHAP Values if available
        if st.session_state['feature_importance'] is not None:
            st.write("Variable Importance:")
            st.write(st.session_state['feature_importance'])

            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=st.session_state['feature_importance'], ax=ax)
            st.pyplot(fig)
        
        if st.session_state['shap_values'] is not None:
            st.write("SHAP Summary Plot:")
            fig, ax = plt.subplots()
            shap.summary_plot(st.session_state['shap_values'], st.session_state['X_train'], show=False)
            st.pyplot(fig)



# Model Evaluation Tab
with tabs[1]:
    st.header("Modelling")
    if data is not None:
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
        st.session_state['X_train'] = X_train
        
        # Calculate and store feature importance and SHAP values
        if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
            classifier = st.session_state['best_pipeline'].named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                importance = classifier.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': importance
                }).sort_values(by='Importance', ascending=False)
                st.session_state['feature_importance'] = feature_importance
            
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_train)
            st.session_state['shap_values'] = shap_values
        else:
            classifier = st.session_state['best_pipeline'].named_steps['classifier']
            explainer = shap.Explainer(classifier, X_train)
            shap_values = explainer(X_train)
            st.session_state['shap_values'] = shap_values
    else:
        st.write("No data available.")

# Scoring Tab
with tabs[2]:
    st.header("Scoring")
    uploaded_file = st.file_uploader("Upload your data CSV file for scoring", type="csv")
    if uploaded_file is not None:
        scoring_data = pd.read_csv(uploaded_file)

        # Ensure we always use the Decision Tree model for scoring
        if 'best_pipeline' not in st.session_state or st.session_state['best_model_name'] != 'Decision Tree':
            decision_tree_pipeline = create_pipeline(DecisionTreeClassifier())
            X = st.session_state['data'].drop(columns=['Anomaly_Label'])
            y = st.session_state['data']['Anomaly_Label'].replace({-1: 0, 1: 1})
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
            decision_tree_pipeline.fit(X_train, y_train)
            st.session_state['best_pipeline'] = decision_tree_pipeline
            st.session_state['best_model_name'] = 'Decision Tree'

        # Align columns of scoring data to match training data
        scoring_data = scoring_data[X.columns]

        predictions = st.session_state['best_pipeline'].predict(scoring_data)
        prediction_results = scoring_data.copy()
        prediction_results['Predictions'] = predictions
        st.write("Predictions:")
        st.write(prediction_results)
        
        # Count normal points and anomalies
        normal_count = (predictions == 0).sum()
        anomaly_count = (predictions == 1).sum()
        st.write(f"Normal Points: {normal_count}")
        st.write(f"Anomaly Points: {anomaly_count}")
