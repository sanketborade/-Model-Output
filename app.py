import os
import pandas as pd
import numpy as np
import streamlit as st
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

# Define a function to create pipelines
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

# Define models with tuned hyperparameters
models = {
    'Logistic Regression': LogisticRegression(C=100, penalty='l2'),
    'Decision Tree': DecisionTreeClassifier(max_depth='None', min_samples_split=2, min_samples_leaf=1),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth='None', min_samples_split=2, min_samples_leaf=1),
    'SVM': SVC(C=10, kernel='linear'),
    'KNN': KNeighborsClassifier(n_neighbors=9, weights='distance', metric='manhattan'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=100, max_depth=3),
    'XGBoost': XGBClassifier(eval_metric='logloss', n_estimators=300, learning_rate=0.01, max_depth=3)
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
        model = models[model_name]
        pipeline = create_pipeline(model)
        
        # Train the model
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Display results
        st.subheader(f"{model_name} Accuracy")
        st.write(f"{accuracy:.2f}")
        st.subheader(f"Classification Report for {model_name}")
        st.write(pd.DataFrame(report).transpose())
        
        # Store the results for download
        result_data = data.copy()
        result_data['Predicted_Label'] = pipeline.predict(X)
        
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
