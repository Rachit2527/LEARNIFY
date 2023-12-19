import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from string import punctuation
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# book recommendation system
book = pd.read_csv("Books.csv")
book = book[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
book.rename(columns={'Book-Title': 'Title', 'Book-Author': 'Author', 'Year-Of-Publication': 'Year'}, inplace=True)

ratings = pd.read_csv('Ratings.csv')
ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'Rating'}, inplace=True)

users = pd.read_csv('Users.csv')
users.rename(columns={'User-ID': 'user_id'}, inplace=True)
x = ratings['user_id'].value_counts() > 200
y = x[x].index
ratings = ratings[ratings['user_id'].isin(y)]

rating_with_book = ratings.merge(book, on='ISBN')
number_rating = rating_with_book.groupby('Title')['Rating'].count().reset_index()
number_rating.rename(columns={'Rating': 'Number of Rating'}, inplace=True)

final_rating = rating_with_book.merge(number_rating, on='Title')
final_rating = final_rating[final_rating['Number of Rating'] >= 50]
final_rating.drop_duplicates(['user_id', 'Title'], inplace=True)

book_pivot = final_rating.pivot_table(columns='user_id', index='Title', values='Rating')
book_pivot.fillna(0, inplace=True)

book_sparse = csr_matrix(book_pivot)

model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)  

# music recommendation system
df_music = pd.read_csv('songdata.csv')
df_music = df_music.sample(n=5000).drop('link', axis=1).reset_index(drop=True)

df_music['text'] = df_music['text'].str.lower().replace(r'[^\w\s]', '').replace(r'\n', ' ', regex=True)

stemmer = PorterStemmer()
def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

df_music['text'] = df_music['text'].apply(lambda x: tokenization(x))

# TF-IDF Vectorization
tfidfvector_music = TfidfVectorizer(stop_words='english')
matrix_music = tfidfvector_music.fit_transform(df_music['text'])
similarity_music = cosine_similarity(matrix_music)

# Recommendation function for Music
def recommendation_music(song_df):
    idx = df_music[df_music['song'] == song_df].index[0]
    distances = sorted(list(enumerate(similarity_music[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:21]:
        songs.append(df_music.iloc[m_id[0]]['song'])

    return songs

# Function to perform text summarization
# def summarize_text(text):
#     stopwords = list(STOP_WORDS)
#     nlp = spacy.load('en_core_web_sm')
#     doc = nlp(text)
    
#     # Calculate word frequencies
#     word_frequencies = {}
#     for word in doc:
#         if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
#             if word.text.lower() not in word_frequencies.keys():
#                 word_frequencies[word.text.lower()] = 1
#             else:
#                 word_frequencies[word.text.lower()] += 1
    
#     # Normalize word frequencies
#     max_frequency = max(word_frequencies.values())
#     for word in word_frequencies.keys():
#         word_frequencies[word] = word_frequencies[word] / max_frequency
    
#     # Tokenize sentences
#     sentence_tokens = [sent for sent in doc.sents]
    
#     # Calculate sentence scores
#     sentence_scores = {}
#     for sent in sentence_tokens:
#         for word in sent:
#             if word.text.lower() in word_frequencies.keys():
#                 if str(sent) not in sentence_scores.keys():
#                     sentence_scores[str(sent)] = word_frequencies[word.text.lower()]
#                 else:
#                     sentence_scores[str(sent)] += word_frequencies[word.text.lower()]
    
#     # Get the summary sentences
#     summary_sentences = [sent.text for sent in sentence_tokens if str(sent) in sentence_scores.keys()]
    
#     # Join the summary sentences to create the final summary
#     summary = ' '.join(summary_sentences)
    
#     return summary

# Initialize the TF-IDF vectorizer for Text Similarity Prediction
vectorizer_text_similarity = TfidfVectorizer()

# Train the Text Similarity Prediction model
data_text_similarity = pd.read_csv('Precily_Text_Similarity.csv')
data_text_similarity['text1'] = data_text_similarity['text1'].apply(lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x))
data_text_similarity['text2'] = data_text_similarity['text2'].apply(lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x))

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    return text

data_text_similarity['text1'] = data_text_similarity['text1'].apply(preprocess_text)
data_text_similarity['text2'] = data_text_similarity['text2'].apply(preprocess_text)

# Fit and transform the text columns
tfidf_matrix_text_similarity = vectorizer_text_similarity.fit_transform(data_text_similarity['text1'] + ' ' + data_text_similarity['text2'])

# Calculate cosine similarity
cosine_similarities_text_similarity = cosine_similarity(tfidf_matrix_text_similarity, tfidf_matrix_text_similarity)

# Create a new column "similarity score" with the similarity scores
data_text_similarity['similarity score'] = [similarity_score[1] for similarity_score in cosine_similarities_text_similarity]

# Split the data into training and test sets
X_train_text_similarity, X_test_text_similarity, y_train_text_similarity, y_test_text_similarity = train_test_split(tfidf_matrix_text_similarity, data_text_similarity['similarity score'], test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model_text_similarity = LinearRegression()
model_text_similarity.fit(X_train_text_similarity, y_train_text_similarity)

# Function to predict similarity score
def predict_similarity(text1, text2):
    # Preprocess input text
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Transform input text using the TF-IDF vectorizer
    tfidf_vector_text_similarity = vectorizer_text_similarity.transform([text1 + ' ' + text2])

    # Predict similarity score
    similarity_score_text_similarity = model_text_similarity.predict(tfidf_vector_text_similarity)[0]

    return similarity_score_text_similarity



# Score Prediction System
df = pd.read_csv('ipl.csv')
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)

# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

# Removing the first 5 overs data in every match
df = df[df['overs']>=5.0]

# Converting the column 'date' from string into datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

# Rearranging the columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

# --- Model Building ---
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)    
    

def flashcards():
    st.title("Flashcards")

    # Create flashcards dictionary (you may want to store this data persistently)
    flashcards_data = st.session_state.get('flashcards_data', [])

    # Allow users to add flashcards
    new_question = st.text_input("Enter a new question:")
    new_answer = st.text_input("Enter the answer:")
    if st.button("Add Flashcard"):
        flashcards_data.append({"question": new_question, "answer": new_answer})
        st.session_state.flashcards_data = flashcards_data
        st.success("Flashcard added successfully!")

    # Show flashcards for studying
    if flashcards_data:
        st.subheader("Flashcards for Studying")
        for i, flashcard in enumerate(flashcards_data, start=1):
            st.write(f"**Flashcard {i}:**")
            st.write(f"Question: {flashcard['question']}")
            st.write(f"Answer: {flashcard['answer']}")
            st.write("---")

# Study Notes Page
def study_notes():
    st.title("Study Notes")

    # Create study notes list (you may want to store this data persistently)
    study_notes_data = st.session_state.get('study_notes_data', [])

    # Allow users to add study notes
    new_note = st.text_area("Enter your note:")
    if st.button("Add Note"):
        study_notes_data.append(new_note)
        st.session_state.study_notes_data = study_notes_data
        st.success("Note added successfully!")

    # Show study notes
    if study_notes_data:
        st.subheader("Your Study Notes")
        for i, note in enumerate(study_notes_data, start=1):
            st.write(f"**Note {i}:**")
            st.write(note)
            st.write("---")

def todo_list():
    st.title("Todo List")

    # Create todo list (you may want to store this data persistently)
    todos = st.session_state.get('todos', [])

    # Allow users to add todo
    new_todo = st.text_input("Enter a new task:")
    if st.button("Add Task"):
        todos.append(new_todo)
        st.session_state.todos = todos
        st.success("Task added successfully!")

    # Show todo list
    if todos:
        st.subheader("Your Tasks")
        for i, task in enumerate(todos, start=1):
            st.write(f"**Task {i}:**")
            st.write(task)
            st.write("---")

    # Allow users to delete todo
    delete_todo = st.selectbox("Select a task to delete:", todos, index=0)
    if st.button("Delete Task"):
        todos.remove(delete_todo)
        st.session_state.todos = todos
        st.success("Task deleted successfully!")

# Home Page
def home():
    st.write("Welcome to the App for Study! Explore different features using the navigation bar.")

# Text Similarity Page
def text_similarity():
    st.title("Text Similarity Predictor")
    text1 = st.text_area("Enter text 1:")
    text2 = st.text_area("Enter text 2:")
    if st.button("Predict Similarity"):
        similarity_score = predict_similarity(text1, text2)
        st.success(f"Predicted Similarity Score: {similarity_score:.4f}")

# Function to recommend books
def recommend(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    st.write("Books similar to", book_name, "are:")
    for i in range(len(suggestions[0])):
        st.write(book_pivot.index[suggestions[0][i]])

# Book Recommendation Page
def recommend_page():
    st.title("Book Recommendation App")
    # Streamlit UI
    book_name_input = st.selectbox("Select a book title:", book_pivot.index, index=0)
    if st.button("Recommend"):
        recommend(book_name_input)

# Music Recommendation Page
def music_recommendation():
    st.title("Music Recommendation App")
    selected_song = st.selectbox("Select a song:", df_music['song'].values)  # Use your song list
    if st.button("Get Recommendations"):
        recommendations = recommendation_music(selected_song)
        st.success("Recommended Songs:")
        st.write(recommendations)

def score_prediction():
    st.title("IPL Score Prediction App")

    # Collect user input features
    st.header("User Input Features")

    date = st.date_input('Date', datetime.today())
    batting_team = st.selectbox('Batting Team', consistent_teams)
    bowling_team = st.selectbox('Bowling Team', consistent_teams)
    overs = st.number_input('Overs', min_value=0.0, max_value=20.0, step=0.1, format="%.1f", value=6.0)
    runs = st.number_input('Runs', min_value=0)
    wickets = st.number_input('Wickets', min_value=0)
    runs_last_5 = st.number_input('Runs in Last 5 Overs', min_value=0)
    wickets_last_5 = st.number_input('Wickets in Last 5 Overs', min_value=0)

    data = {'date': date, 'bat_team': batting_team, 'bowl_team': bowling_team,
            'overs': overs, 'runs': runs, 'wickets': wickets,
            'runs_last_5': runs_last_5, 'wickets_last_5': wickets_last_5}

    input_df = pd.DataFrame(data, index=[0])

    # Preprocess user input
    teams = ['bat_team', 'bowl_team']
    input_df_encoded = pd.get_dummies(input_df, columns=teams)
    input_df_encoded = input_df_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Make predictions
    prediction = regressor.predict(input_df_encoded)

    st.subheader("Prediction")
    st.write(f"The predicted score is: {round(prediction[0])}")

def show_intro_text():
    st.title("App for Study")
    st.write("You can explore different features using the navigation bar. The Study App is designed to provide various features for learning and exploration.")
    st.write("Feel free to use the different tools and functionalities available to enhance your study experience.")
    st.write("If you have been tired of studying hard, you can relax also by using the music recommendation system.")
    

# Streamlit UI
def main():
    
    st.sidebar.title("Navigation")
    pages = ["Home", "Text Similarity","IPL Score Prediction","Book Recommendation", "Music Recommendation", "Flashcards", "Study Notes", "Todo List"]
    #choice = st.sidebar.selectbox("Go to", pages)
    choice = st.sidebar.radio("Go to", pages)


    if choice == "Home":
         show_intro_text()
         home()
       
    elif choice == "Text Similarity":
        text_similarity()
    elif choice == "Book Recommendation":
        recommend_page()
    elif choice == "Music Recommendation":
        music_recommendation()
    elif choice == "Flashcards":
        flashcards()
    elif choice == "Todo List":
        todo_list()
    elif choice == "IPLScore Prediction":
        score_prediction()
    elif choice == "Study Notes":
        study_notes()

if __name__ == "__main__":
    main()



