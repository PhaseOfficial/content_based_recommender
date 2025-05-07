import streamlit as st
from app.recommender import NewsRecommender
from app.utils import clean_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Recommender
@st.cache_resource
def load_model():
    return NewsRecommender('data/news_data.csv')

# utils.py — can be used for text preprocessing (if needed)

def clean_text(text):
    return text.strip().lower()


recommender = load_model()

class NewsRecommender:
    def __init__(self, data_source):
        # Check if data_source is a DataFrame or a file path
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source
        else:
            self.df = pd.read_csv(data_source)
        
        # Ensure the dataset has the required columns
        if 'title' not in self.df.columns or 'content' not in self.df.columns:
            raise ValueError("The dataset must contain 'title' and 'content' columns.")
        
        # Combine title and content for text processing
        self.df['text'] = self.df['title'] + " " + self.df['content']
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text'])

    def recommend(self, input_text, top_n=5):
        input_vec = self.vectorizer.transform([input_text])
        similarity_scores = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        return self.df.iloc[top_indices][['title', 'content']]

# Streamlit UI
st.title("📰 Content-Based News Recommender")
st.write("Enter a news story or description, and get similar articles.")

user_input = st.text_area("📝 Enter a news story or topic:")

if st.button("Get Recommendations"):
    if user_input:
        cleaned = clean_text(user_input)
        results = recommender.recommend(cleaned)
        st.subheader("🔍 Recommended Articles:")
        for _, row in results.iterrows():
            st.markdown(f"**{row['title']}**")
            st.write(row['content'])
            st.markdown("---")
    else:
        st.warning("Please enter some text to receive recommendations.")
