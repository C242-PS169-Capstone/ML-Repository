import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langdetect import detect
from googletrans import Translator

# Load tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Load model
def load_model_file(model_path):
    model = load_model(model_path)
    return model

# Translate text to English
def translate_to_english(text):
    lang = detect(text)
    if lang != 'en':
        translator = Translator()
        translated_text = translator.translate(text, src=lang, dest='en').text
        return translated_text
    return text

# Preprocess input text
def preprocess_text(text, tokenizer, maxlen=200):
    text = translate_to_english(text)
    tokenized = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokenized, maxlen=maxlen, padding='post')
    return padded

# Predict text class
def predict_text(model, tokenizer, text):
    padded_input = preprocess_text(text, tokenizer)
    prediction = model.predict(padded_input)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_prob = prediction[0][predicted_class[0]] * 100

    class_names = ['Not Suicidal', 'Suicidal']
    return class_names[predicted_class[0]], predicted_prob
