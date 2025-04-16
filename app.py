import streamlit as st
import pickle

# Load the trained model and TF-IDF vectorizer
with open('/content/sentiment_classifier.pkl', 'rb') as file:  # Update the path if your model is saved elsewhere
    model = pickle.load(file)

with open('/content/tfidf_vectorizer.pkl', 'rb') as file:  # Update the path if your vectorizer is saved elsewhere
    vectorizer = pickle.load(file)

# Define a function for prediction
def predict_sentiment(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return prediction

# Streamlit app layout
st.title('Sentiment Analysis App')
st.write('Enter text to analyze:')

# Display "Hello World" once
st.write("Hello World!")  # This will be displayed when the app starts

user_input = st.text_input('Input Text')

if st.button('Predict'):
    if user_input:
        prediction = predict_sentiment(user_input)
        st.write(f'Predicted Sentiment: {prediction}')
    else:
        st.write('Please enter some text.')
