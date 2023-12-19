import numpy as np
import pandas as pd
import sklearn as sk
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('Precily_Text_Similarity.csv')

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    return text

data['text1'] = data['text1'].apply(preprocess_text)
data['text2'] = data['text2'].apply(preprocess_text)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text columns
tfidf_matrix = vectorizer.fit_transform(data['text1'] + ' ' + data['text2'])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a new column "similarity score" with the similarity scores
data['similarity score'] = [similarity_score[1] for similarity_score in cosine_similarities]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, data['similarity score'], test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Text Similarity Prediction App")

# Input text
text1 = st.text_input("Enter text1:", "")
text2 = st.text_input("Enter text2:", "")

# Button to trigger similarity prediction
if st.button("Predict Similarity Score"):
    # Preprocess input text
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Transform input text using the TF-IDF vectorizer
    tfidf_vector = vectorizer.transform([text1 + ' ' + text2])

    # Predict similarity score
    similarity_score = model.predict(tfidf_vector)[0]

    # Display predicted similarity score
    st.success(f"Predicted Similarity Score: {similarity_score:.4f}")
