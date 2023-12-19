import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('songdata.csv')
df = df.sample(n=5000).drop('link', axis=1).reset_index(drop=True)

# Preprocess text
df['text'] = df['text'].str.lower().replace(r'[^\w\s]','').replace(r'\n',' ', regex=True)

# Tokenization and stemming
stemmer = PorterStemmer()
def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

df['text'] = df['text'].apply(lambda x: tokenization(x))

# TF-IDF Vectorization
tfidfvector = TfidfVectorizer(stop_words='english')
matrix = tfidfvector.fit_transform(df['text'])
similarity = cosine_similarity(matrix)

# Recommendation function
def recommendation(song_df):
    idx = df[df['song'] == song_df].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]]['song'])
        
    return songs

# Streamlit app
st.title("Song Recommendation App")

# Select a song
selected_song = st.selectbox("Select a song:", df['song'].values)

# Button to trigger recommendation
if st.button("Get Recommendations"):
    # Get recommendations
    recommendations = recommendation(selected_song)

    # Display recommendations
    st.success("Recommended Songs:")
    st.write(recommendations)
