import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load data
book = pd.read_csv("Books.csv")
book = book[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
book.rename(columns={'Book-Title': 'Title', 'Book-Author': 'Author', 'Year-Of-Publication': 'Year'}, inplace=True)

ratings = pd.read_csv('Ratings.csv')
ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'Rating'}, inplace=True)

users = pd.read_csv('Users.csv')
users.rename(columns={'User-ID': 'user_id'}, inplace=True)

# Preprocess data
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
model.fit(book_sparse)  # Fix: Call fit method

# Streamlit app
st.title("Book Recommendation App")

# Function to recommend books
def recommend(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    st.write("Books similar to", book_name, "are:")
    for i in range(len(suggestions[0])):
        st.write(book_pivot.index[suggestions[0][i]])

# Streamlit UI
book_name_input = st.text_input("Enter a book title:", "The Da Vinci Code")
if st.button("Recommend"):
    recommend(book_name_input)
