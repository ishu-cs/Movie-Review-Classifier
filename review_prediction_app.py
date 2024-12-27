# Step 1 import all the libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model 

# Step 2 load the IMDB dataset word index
word_index = imdb.get_word_index()
revesre_word_index = {value: key for key,value in word_index.items()}

# Load the pre-trained RNN model
model = load_model('simple_rnn_imdb.h5')

# Function to decode the reviews
def decode_review(encoded_review):
    return ' '.join([revesre_word_index.get(i-3,'?') for i in encoded_review])

#Function to encode the review(convert the review to word embeddings)

def  preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word,2) +3  for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#Prediction Function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction >0.5 else 'Negative'
    return prediction[0][0],sentiment

## Create the streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

#use input for review
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    score,sentiment=predict_sentiment(user_input)
    st.write(f"Sntiment of the review is {sentiment} with score of {score}")
