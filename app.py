import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define a function to create pipelines
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

# Define models and hyperparameters grid
models = {
    'Logistic Regression': (LogisticRegression(), {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100], 'classifier__penalty': ['l1', 'l2']}),
    'Decision Tree': (DecisionTreeClassifier(), {'classifier__max_depth': [None, 5, 10, 15, 20], 'classifier__min_samples_split': [2, 5, 10], 'classifier__min_samples_leaf': [1, 2, 4]}),
    'Random Forest': (RandomForestClassifier(), {'classifier__n_estimators': [100, 200, 300], 'classifier__max_depth': [None, 5, 10, 15], 'classifier__min_samples_split': [2, 5, 10], 'classifier__min_samples_leaf': [1, 2, 4]}),
    'SVM': (SVC(), {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}),
    'KNN': (KNeighborsClassifier(), {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance'], 'classifier__metric': ['euclidean', 'manhattan']}),
    'Gradient Boosting': (GradientBoostingClassifier(), {'classifier__n_estimators': [100, 200, 300], 'classifier__learning_rate': [0.1, 0.01, 0.001], 'classifier__max_depth': [3, 5, 7]}),
    'XGBoost': (XGBClassifier(eval_metric='logloss'), {'classifier__n_estimators': [100, 200, 300], 'classifier__learning_rate': [0.1, 0.01, 0.001], 'classifier__max_depth': [3, 5, 7]})
}

# Streamlit interface
st.title("Predictive Model")

# Upload scored data CSV
st.header("Upload Scored Data CSV")
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the scored data
    data = pd.read_csv(uploaded_file)
    
    # Convert Anomaly_Label from -1 and 1 to 0 and 1
    data['Anomaly_Label'] = data['Anomaly_Label'].replace({-1: 0, 1: 1})
    
    # Separate features and target
    X = data.drop(columns=['Anomaly_Label'])
    y = data['Anomaly_Label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Select model
    model_name = st.selectbox('Select Model', list(models.keys()))
    
    if model_name:
        model, param_grid = models[model_name]
        pipeline = create_pipeline(model)
        
        # Perform hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Display results
        st.subheader(f"Best hyperparameters for {model_name}")
        st.write(grid_search.best_params_)
        st.subheader(f"{model_name} Tuned Accuracy")
        st.write(f"{accuracy:.2f}")
        st.subheader(f"Classification Report for {model_name}")
        st.write(pd.DataFrame(report).transpose())
        
        # Store the results for download
        result_data = data.copy()
        result_data['Predicted_Label'] = best_model.predict(X)
        
        st.subheader("Scored Data with Predictions")
        st.write(result_data.head())

        result_csv = result_data.to_csv(index=False)
        
        st.download_button(
            label="Download Scored Data with Predictions as CSV",
            data=result_csv,
            file_name='scored_data_with_predictions.csv',
            mime='text/csv'
        )
else:
    st.info("Please upload a CSV file to proceed.")
