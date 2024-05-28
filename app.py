import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense


def load_data():
    data=pd.read_csv(r"C:\Users\vkr20\Downloads\archive\netflix_reviews.csv")
    #display first few rows of the dataframe for debugging
    return data


def create_and_train_model(data):
    texts=data["content"][0:200]
    token=Tokenizer()
    token.fit_on_texts(texts)

    list1=[]
    list2=[]
    for word in texts:
        sequences=token.texts_to_sequences([word])[0]
        for i in range(1,len(sequences)):
            list1.append(sequences[:i])
            list2.append(sequences[i])


    fv=pad_sequences(list1)
    num_classes=len(token.word_index) +1
    cv =to_categorical(list2,num_classes=num_classes)


    model=Sequential()
    model.add(Embedding(input_dim=num_classes,output_dim=100,input_length=fv.shape[1]))
    model.add(LSTM(100))
    model.add(Dense(num_classes,activation="softmax"))

    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(fv,cv,batch_size=30,epochs=20,validation_split=0.2)

    return model, token

data = load_data()


model, token = create_and_train_model(data)


st.title('🔮 Next Word Prediction 🔮')

input_text = st.text_input("Input text")
if st.button('Predict'):
    text = input_text
    st.write("Initial text:", text)
    for i in range(5):
        sequences = token.texts_to_sequences([text])
        padded_sequences = np.array(sequences)  # Convert sequences to numpy array
        v = np.argmax(model.predict(padded_sequences))
        if v in token.index_word:
            text += " " + token.index_word[v]
            st.write(text)
        else:
            st.write(f"Predicted index {v} is out of the vocabulary range.")
            break
