

from flask import Flask, request
import pickle
import re
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load latest model
model = pickle.load(open('model.pkl', 'rb'))

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text)

    text = re.sub(r'http\\S+', '', text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = text.lower()

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)

@app.route('/')

def home():

    return '''

    <html>

    <head>

    <title>AI Fake News Detector</title>

    <style>

    body{
        font-family:Arial;
        background:#f2f2f2;
        text-align:center;
        padding:50px;
    }

    textarea{
        width:70%;
        height:220px;
        padding:15px;
        font-size:16px;
    }

    button{
        padding:12px 25px;
        margin-top:20px;
        font-size:18px;
        background:blue;
        color:white;
        border:none;
        cursor:pointer;
    }

    </style>

    </head>

    <body>

    <h1>📰 AI Fake News Detector</h1>

    <form action="/predict" method="post">

    <textarea
    name="news"
    placeholder="Paste News Here..."
    required>
    </textarea>

    <br><br>

    <button type="submit">
    Detect News
    </button>

    </form>

    </body>

    </html>

    '''

@app.route('/predict', methods=['POST'])

def predict():

    news = request.form['news']

    cleaned_news = clean_text(news)

    vector_input = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector_input)[0]

    probability = model.predict_proba(vector_input)[0]

    fake_prob = probability[0]
    real_prob = probability[1]

    confidence = round(max(fake_prob, real_prob) * 100, 2)

    if confidence < 60:
        result = "🟡 UNCERTAIN NEWS"

    else:
        if prediction == 0:
            result = "🔴 FAKE NEWS"
        else:
            result = "🟢 REAL NEWS"

    return f'''

    <html>

    <body style="font-family:Arial;text-align:center;padding:50px;">

    <h1>{result}</h1>

    <h2>Confidence: {confidence}%</h2>

    <a href="/">Check Another News</a>

    </body>

    </html>

    '''

if __name__ == '__main__':

    app.run()

