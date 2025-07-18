# imdb_sentiment_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load IMDB word index and reverse it
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the trained model
model = load_model('simple_rnn_imdb.h5')

# ------------------ Helper Functions ------------------

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(text):
    preprocessed = preprocess_text(text)
    prediction = model.predict(preprocessed)[0][0]
    label = "Positive" if prediction > 0.9 else "Negative"
    return label, prediction

# ------------------ Streamlit App Layout ------------------

import streamlit as st
import base64

# ========== Custom Page Design ==========
st.markdown("""
    <style>
        .main {
            background-color: #ffdddd;
            padding: 20px;
            border-radius: 10px;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
        }
        .stTextArea textarea {
            font-size: 18px;
        }
        .big-font {
            font-size:28px !important;
            font-weight: bold;
        }
        .emoji {
            font-size: 80px;
            text-align: center;
        }
        .result-box {
            background-color: #ffffffcc;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== App Title ==========
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 style="color:#b30000; text-align:center;">🎬 IMDB Movie Review Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:18px;">Enter a movie review to classify it as <b>Positive</b> or <b>Negative</b>.</p>', unsafe_allow_html=True)

# ========== User Input ==========
user_input = st.text_area('🎥 Movie Review:', height=150)

if st.button('🚀 Classify'):
    if user_input.strip() != "":
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.9 else 'Negative'
        emoji = "😀" if sentiment == 'Positive' else "😠"

        # Display result
        st.markdown(f'<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="emoji">{emoji}</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">Sentiment: {sentiment}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:20px;">Prediction Score: {prediction[0][0]:.4f}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a movie review.")
else:
    st.markdown('<p style="text-align:center; font-size:16px;">👈 Enter your review above and click <b>Classify</b> to begin!</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)