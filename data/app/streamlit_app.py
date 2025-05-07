import streamlit as st
import pandas as pd
from recommender import NewsRecommender

# Title of the app
st.title("Content-Based News Recommender")

# File upload for the dataset
uploaded_file = st.file_uploader("Upload a CSV file with 'title' and 'content' columns", type=["csv"])

# Default dataset path
default_data_path = "data/test_data.csv"

if uploaded_file:
    # Try to read the uploaded file
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded file preview:", data.head())  # Debugging: Show the first few rows
        if data.empty:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        elif 'title' not in data.columns or 'content' not in data.columns:
            st.error("The uploaded file must contain 'title' and 'content' columns.")
        else:
            st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
else:
    # Use the default dataset
    st.info("Using the default dataset.")
    data = pd.read_csv(default_data_path)

# Validate the dataset
if 'title' in data.columns and 'content' in data.columns:
    # Initialize the recommender with the dataset
    recommender = NewsRecommender(data)

    # User input for the news content
    user_input = st.text_area("Enter a news headline or content to get recommendations:")

    if user_input:
        # Get recommendations
        recommendations = recommender.recommend(user_input)

        # Display recommendations
        st.subheader("Top Recommendations:")
        for index, row in recommendations.iterrows():
            st.write(f"**Title:** {row['title']}")
            st.write(f"**Content:** {row['content']}")
            st.write("---")
else:
    st.error("The dataset must contain 'title' and 'content' columns.")