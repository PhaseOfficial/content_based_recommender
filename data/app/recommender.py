import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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