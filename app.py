import streamlit as st
import pandas as pd

# Streamlit interface
st.title("Simple CSV Upload App")

# Upload scored data CSV
st.header("Upload Scored Data CSV")
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the scored data
        data = pd.read_csv(uploaded_file)
        
        st.subheader("Uploaded Data")
        st.write(data.head())
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
