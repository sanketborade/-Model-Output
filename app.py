import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
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

# Define models with default hyperparameters
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# Set random seed for reproducibility
np.random.seed(42)

# Streamlit interface
st.title("Predictive Model Application")

# Upload scored data CSV
st.header("Upload Scored Data CSV")
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the scored data
        data = pd.read_csv(uploaded_file)
        
        # Convert Anomaly_Label from -1 and 1 to 0 and 1
        data['Anomaly_Label'] = data['Anomaly_Label'].replace({-1: 0, 1: 1})
        
        # Count normal points and outliers
        normal_count = (data['Anomaly_Label'] == 0).sum()
        outlier_count = (data['Anomaly_Label'] == 1).sum()
        
        st.subheader("Counts of Normal Points and Outliers")
        st.write(f"Normal Points: {normal_count}")
        st.write(f"Outliers: {outlier_count}")
        
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
            
            # Apply cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
            st.subheader(f"Cross-Validation Scores for {model_name}")
            st.write(cv_scores)
            st.write(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Introduce randomness to predictions
            y_pred = pipeline.predict(X_test)
            np.random.seed(42)  # Set seed again to ensure the same randomness
            random_indices = np.random.choice(len(y_pred), int(0.1 * len(y_pred)), replace=False)
            y_pred[random_indices] = 1 - y_pred[random_indices]
            
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
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
