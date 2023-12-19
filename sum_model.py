import numpy as np
import pandas as pd
import string
import re
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import streamlit as st

# Function to perform text summarization
def summarize_text(text):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    # Calculate word frequencies
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text.lower() not in word_frequencies.keys():
                word_frequencies[word.text.lower()] = 1
            else:
                word_frequencies[word.text.lower()] += 1
    
    # Normalize word frequencies
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    
    # Tokenize sentences
    sentence_tokens = [sent for sent in doc.sents]
    
    # Calculate sentence scores
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if str(sent) not in sentence_scores.keys():
                    sentence_scores[str(sent)] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[str(sent)] += word_frequencies[word.text.lower()]
    
    # Get the summary sentences
    summary_sentences = [sent.text for sent in sentence_tokens if str(sent) in sentence_scores.keys()]
    
    # Join the summary sentences to create the final summary
    summary = ' '.join(summary_sentences)
    
    return summary

# Streamlit app
def main():
    st.title("Text Summarization App")
    text = st.text_area("Enter the text you want to summarize:")
    
    if st.button("Summarize"):
        if text:
            summary = summarize_text(text)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
