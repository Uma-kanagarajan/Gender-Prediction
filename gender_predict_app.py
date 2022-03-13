import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

oov_tok = "<OOV>"

# load the model
model = tf.keras.models.load_model('gender_prediction.h5')

# Function to preprocess the given name
def preprocess_name(names_df):
    names_df['name'] = names_df['name'].str.replace('[^a-zA-Z]', '')  # remove unncessary characters
    names = names_df['name'].values
    tokenizer = Tokenizer(num_words=None, oov_token=oov_tok, char_level=True)  # Character splitting in tokenizer
    tokenizer.fit_on_texts(names)  # Train the corpus
    sequences = tokenizer.texts_to_sequences(names)  # convert character to integer
    name_length = 20
    input_sequences = pad_sequences(sequences, maxlen=name_length, padding='post')  # padding
    return input_sequences

# Function to predict gender
def gender_predict(name):
    # Convert to dataframe
    pred_df = pd.DataFrame({'name': [name]})
    name_array = preprocess_name(pred_df)
    predict_x = model.predict(name_array)
    return predict_x

# Code for web UI
def main():
    st.set_page_config(page_title="Gender Predictor", page_icon=":couple:", layout="wide")

    col1, col2, col3 = st.columns([1, 2, 1])
    col2.write("""
             # Name based Gender Prediction
             """)
    col2.write("This is a web app to predict the gender based on given name")
    name = col2.text_input(label='', max_chars=20, placeholder='Enter Name')
    st.write("")
    predict = col2.button('Predict')
    st.write("")

    if predict:
        predict_value = gender_predict(name)
        # print(predict_value)
        if predict_value > 0.5:
            col2.write('{} is a Male'.format(name))
        else:
            col2.write('{} is a Female'.format(name))


if __name__ == '__main__':
    main()
