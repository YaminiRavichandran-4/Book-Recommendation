import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\YAMINI RAVICHANDRAN\OneDrive\ドキュメント\book recommendation\data.csv")  
    return df

df = load_data()

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["Processed_Description"] = df["Description"].fillna("No Description").apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Processed_Description"])

# K-Means Clustering
num_clusters = 10  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(tfidf_matrix)

# Hybrid Recommendation System
def hybrid_recommend(book_name, top_n=5):
    book_idx = df[df["Book Name"].str.lower() == book_name.lower()].index
    if len(book_idx) == 0:
        return "Book not found. Try another title."
    book_idx = book_idx[0]

    book_cluster = df.loc[book_idx, "Cluster"]
    cluster_books = df[df["Cluster"] == book_cluster]

    cluster_tfidf_matrix = vectorizer.transform(cluster_books["Processed_Description"])
    book_vector = vectorizer.transform([df.loc[book_idx, "Processed_Description"]])

    similarity_scores = cosine_similarity(book_vector, cluster_tfidf_matrix).flatten()
    
    cluster_books = cluster_books.copy()  # Avoid SettingWithCopyWarning
    cluster_books["Similarity"] = similarity_scores

    recommended_books = cluster_books.sort_values(by=["Similarity", "Rating"], ascending=[False, False])

    return recommended_books[["Book Name", "Author", "Rating"]].head(top_n)

# Streamlit UI
st.title("📚 Audible Insights: Intelligent Book Recommendation")

st.write("Enter a book name to get recommendations based on similarity and clustering.")

book_input = st.text_input("🔍 Enter Book Name:", "")

if st.button("Get Recommendations"):
    if book_input.strip() == "":
        st.warning("Please enter a book name.")
    else:
        recommendations = hybrid_recommend(book_input)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.table(recommendations)
