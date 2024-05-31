import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load

st.title("Anomaly Detection Predictive Model")

# Upload scored data CSV
st.header("Upload Scored Data CSV")
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the scored data
    scored_data = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(scored_data.head())

    # Separate features and target
    X = scored_data.drop(columns=['Anomaly_Label'])
    y = scored_data['Anomaly_Label']

    # Preprocess the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Check if the model file exists
    model_path = 'models/best_model.pkl'
    if not os.path.exists('models'):
        os.makedirs('models')
        
    if os.path.exists(model_path):
        # Load the trained model
        model = load(model_path)
    else:
        # Train a new model
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Save the model
        dump(model, model_path)
        st.success(f"Model trained and saved to {model_path}")

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader(f"Model Accuracy: {accuracy:.2f}")

    # Display classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Scoring the entire dataset
    y_pred_full = model.predict(X_scaled)
    scored_data['Predicted_Label'] = y_pred_full

    st.subheader("Scored Data with Predictions")
    st.write(scored_data)

    # Download the scored data with predictions
    scored_data_csv = scored_data.to_csv(index=False)
    st.download_button(
        label="Download Scored Data as CSV",
        data=scored_data_csv,
        file_name='scored_data_with_predictions.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to proceed.")
