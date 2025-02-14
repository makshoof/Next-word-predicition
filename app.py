import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Ensure the sequence is within the required max length
    token_list = token_list[-(max_sequence_len - 1):]  # Keep last max_sequence_len-1 tokens

    # Fix: Pad sequences to match the expected model input length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

    # Predict the next word
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted)

    # Get the word from the tokenizer
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None  # Return None if word not found

# Streamlit UI
st.title("Next Word Predictor")
st.write("Enter the phrase")

user_input = st.text_input("Enter your text:")

# Get max sequence length from the model
max_sequence_len = model.input_shape[1]  # Ensure correct input shape

if st.button("Predict Next Word"):
    if user_input:
        next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
        if next_word:
            st.success(f"Predicted next word: *{next_word}*")
        else:
            st.warning("Unable to predict, try something different.")
    else:
        st.error("Enter some text to predict the next word.")
