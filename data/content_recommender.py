import streamlit as st
from app.recommender import NewsRecommender
from app.utils import clean_text

# Load Recommender
@st.cache_resource
def load_model():
    return NewsRecommender('data/news_data.csv')

recommender = load_model()

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
