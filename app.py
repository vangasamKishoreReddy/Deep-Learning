import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense

## Load the dataset
def load_data():
    data = pd.read_csv(r"C:\Users\vkr20\Downloads\archive\netflix_reviews.csv")
    return data  # Display first few rows of the dataframe for debugging

## Create and train the model
def create_and_train_model(data):
    texts = data["content"][0:200]  # Use the first 200 reviews for training
    token = Tokenizer()  # Initialize the tokenizer
    token.fit_on_texts(texts)  # Fit tokenizer on texts

    list1 = []
    list2 = []
    for word in texts:  # For each review
        sequences = token.texts_to_sequences([word])[0]  # Convert text to sequences
        for i in range(1, len(sequences)):  # Generate input-output pairs
            list1.append(sequences[:i])
            list2.append(sequences[i])

    fv = pad_sequences(list1)  # Pad the input sequences
    num_classes = len(token.word_index) + 1  # Calculate number of unique words
    cv = to_categorical(list2, num_classes=num_classes)  # Convert outputs to categorical

    model = Sequential()  # Initialize the model
    model.add(Embedding(input_dim=num_classes, output_dim=100, input_length=fv.shape[1]))  # Add Embedding layer
    model.add(LSTM(100))  # Add LSTM layer
    model.add(Dense(num_classes, activation="softmax"))  # Add Dense layer

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # Compile the model
    model.fit(fv, cv, batch_size=30, epochs=20, validation_split=0.2)  # Train the model

    return model, token  # Return the trained model and tokenizer

data = load_data()  # Load the data

model, token = create_and_train_model(data)  # Create and train the model

st.title('🔮 Next Word Prediction 🔮')  # Set the title for the Streamlit app

## Text input for prediction
input_text = st.text_input("Input text")
if st.button('Predict'):  # When the button is clicked
    text = input_text  # Get the input text
    st.write("Initial text:", text)  # Display the initial text
    for i in range(5):  # Predict the next 5 words
        sequences = token.texts_to_sequences([text])  # Convert text to sequences
        padded_sequences = np.array(sequences)  # Convert sequences to numpy array
        v = np.argmax(model.predict(padded_sequences))  # Predict the next word
        if v in token.index_word:  # Check if the predicted word is in the vocabulary
            text += " " + token.index_word[v]  # Append the predicted word to the text
            st.write(text)  # Display the updated text
        else:  # If the predicted word is out of vocabulary
            st.write(f"Predicted index {v} is out of the vocabulary range.")
            break  # Exit the loop
