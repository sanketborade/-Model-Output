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
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import scikitplot as skplt  # For plotting gain and lift charts

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
    'SVM': SVC(probability=True),  # Enable probability estimates
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
            
            # Predictions and probabilities
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
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
            
            # Count normal points and outliers after prediction
            normal_count_pred = (result_data['Predicted_Label'] == 0).sum()
            outlier_count_pred = (result_data['Predicted_Label'] == 1).sum()
            
            st.subheader("Counts of Normal Points and Outliers After Prediction")
            st.write(f"Normal Points: {normal_count_pred}")
            st.write(f"Outliers: {outlier_count_pred}")
            
            st.subheader("Scored Data with Predictions")
            st.write(result_data.head())

            result_csv = result_data.to_csv(index=False)
            
            st.download_button(
                label="Download Scored Data with Predictions as CSV",
                data=result_csv,
                file_name='scored_data_with_predictions.csv',
                mime='text/csv'
            )
            
            # EDA Tab
            st.header("Exploratory Data Analysis (EDA)")
            
            # ROC Curve
            st.subheader("ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            st.pyplot(plt)
            
            # Gain Chart
            st.subheader("Gain Chart")
            skplt.metrics.plot_cumulative_gain(y_test, pipeline.predict_proba(X_test))
            st.pyplot(plt)
            
            # Lift Chart
            st.subheader("Lift Chart")
            skplt.metrics.plot_lift_curve(y_test, pipeline.predict_proba(X_test))
            st.pyplot(plt)
            
            # KS Table
            st.subheader("KS Table")
            ks_table = pd.DataFrame({
                'Threshold': thresholds,
                'FPR': fpr,
                'TPR': tpr,
                'KS Statistic': tpr - fpr
            })
            ks_table = ks_table.sort_values(by='KS Statistic', ascending=False)
            st.write(ks_table)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
