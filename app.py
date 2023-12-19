from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text_similarity')
def text_similarity():
    # Add logic to handle text similarity model
    return "Text Similarity page"

@app.route('/text_summarizer')
def text_summarizer():
    # Add logic to handle text summarizer model
    return "Text Summarizer page"

@app.route('/book_recommendation')
def music_recommendation():
    # Add logic to handle music recommendation model
    return "Book Recommendation page"

@app.route('/music_recommendation', methods=['GET', 'POST'])
def music_recommendation():
    if request.method == 'POST':
        selected_song = request.form['selected_song']
        recommendations = recommendation(selected_song)
        return render_template('music_recommendation.html', selected_song=selected_song, recommendations=recommendations)

    return render_template('music_recommendation.html', songs=df['song'].values)

@app.route('/next_word_predictor')
def next_word_predictor():
    # Add logic to handle next word predictor model
    return "Next Word Predictor page"

if __name__ == '__main__':
    app.run(debug=True)
