import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import streamlit as st

# Load the data
doc = '''In the era of digital transformation, data science has emerged as a transformative force, reshaping industries and decision-making processes across the globe. At its core, data science is the amalgamation of statistical methodologies, computational techniques, and domain expertise aimed at extracting meaningful insights from vast and complex datasets. As organizations increasingly recognize the value of data, the demand for skilled data scientists has surged. Data science encompasses a broad spectrum, from data collection and cleaning to advanced machine learning algorithms and predictive modeling. This article delves into the foundational concepts of data science, the key steps in its lifecycle, the tools and technologies driving its advancements, and its myriad applications across diverse sectors.
The data science lifecycle comprises a series of interconnected stages, each crucial for the effective extraction of knowledge from data. The journey begins with data collection and preprocessing, where raw data is gathered and refined to ensure accuracy and relevance. Exploratory Data Analysis (EDA) follows, a stage characterized by a deep dive into the dataset to identify patterns, trends, and outliers. The subsequent steps involve modeling and algorithm selection, where machine learning techniques are applied to build predictive models. Ensuring the reliability of these models is achieved through rigorous evaluation and validation processes. Feature engineering and selection further optimize model performance. The final stages involve the deployment of models into real-world scenarios and ongoing maintenance to adapt to changing data patterns and external factors.
A robust data science ecosystem is powered by a suite of programming languages, libraries, and frameworks. Python and R stand out as the primary languages, celebrated for their versatility and extensive libraries. Libraries such as NumPy and Pandas facilitate data manipulation, while Scikit-Learn and TensorFlow offer a comprehensive set of tools for machine learning tasks. Data visualization, a critical aspect of data science, is made accessible through tools like Matplotlib and Tableau. In the realm of big data, technologies like Hadoop and Spark provide scalable solutions for handling and processing massive datasets. This dynamic toolkit empowers data scientists to navigate the intricacies of data with efficiency and precision.
The impact of data science reverberates across industries, introducing innovative solutions and catalyzing advancements. In healthcare, data science plays a pivotal role in disease diagnosis, treatment personalization, and public health management. The finance sector leverages data science for fraud detection, risk assessment, and algorithmic trading, enhancing operational efficiency and security. Marketing and e-commerce thrive on data-driven insights, enabling personalized customer experiences and targeted campaigns. Social media analysis, including sentiment analysis, informs brand strategies and public perception. In manufacturing, predictive maintenance powered by data science reduces downtime and optimizes machinery performance. These diverse applications underscore the versatility and transformative potential of data science.
As data science continues to evolve, its trajectory points towards an even more data-centric future. The amalgamation of artificial intelligence and machine learning with data science is poised to unlock unprecedented possibilities. Ethical considerations, privacy concerns, and responsible data usage are becoming central themes in the data science narrative. As industries increasingly integrate data-driven approaches into their operations, the demand for skilled professionals will persist. In conclusion, the journey through the data science landscape—from its foundational concepts to applications and future prospects—reveals a field that not only interprets the language of data but also shapes the future of industries and societies at large. The data-driven era has just begun, and the possibilities it presents are limited only by our imagination and ethical considerations.
'''

# Preprocess the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([doc])
len(tokenizer.word_index)
tokenizer.word_index
tokenizer.word_counts
input_sequences = []
for sentence in doc.split('\n'):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    
    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i+1])

max_len = max([len(x) for x in input_sequences])

padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]

y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# Build the model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length=max_len-1))
model.add(LSTM(150))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100)

# Streamlit app
st.title("Text Generation with LSTM")

# Text input
seed_text = st.text_area("Enter Seed Text:", "In the era of digital")

# Button to trigger text generation
if st.button("Generate Text"):
    generated_text = seed_text
    for _ in range(5):
        token_text = tokenizer.texts_to_sequences([seed_text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_len-1, padding='pre')
        pos = np.argmax(model.predict(padded_token_text))
        
        for word, index in tokenizer.word_index.items():
            if index == pos:
                seed_text += " " + word
                generated_text += " " + word
                break

    # Display generated text
    st.success("Generated Text:")
    st.write(generated_text)
