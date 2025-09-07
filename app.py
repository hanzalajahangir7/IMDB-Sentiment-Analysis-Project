import streamlit as st
import pickle
from keras.models import load_model
import numpy as np

@st.cache_data
def load_vectorizer():
    with open("vectorize.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_keras_model():
    return load_model('sentiment_model.h5', compile=False)

vectorizer = load_vectorizer()
model = load_keras_model()

st.title("IMDB Sentiment Analysis")
review = st.text_area("Enter your movie review:")

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review text.")
    else:
        review_vec = vectorizer.transform([review])
        review_vec = review_vec.toarray()
        prediction_prob = model.predict(review_vec)[0][0]
        sentiment = "Positive 😊" if prediction_prob > 0.5 else "Negative 😞"
        st.success(f"Prediction: {sentiment} (Confidence: {prediction_prob:.2f})")
        