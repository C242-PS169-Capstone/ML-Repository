import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langdetect import detect
from googletrans import Translator

# Fungsi untuk memuat tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Fungsi untuk memuat model
def load_model_file(model_path):
    model = load_model(model_path)
    return model

# Fungsi untuk menerjemahkan teks ke bahasa Inggris
def translate_to_english(text):
    lang = detect(text)
    if lang != 'en':  # Jika bahasa bukan Inggris, terjemahkan
        translator = Translator()
        translated_text = translator.translate(text, src=lang, dest='en').text
        return translated_text
    return text

# Fungsi untuk preprocessing teks input
def preprocess_text(text, tokenizer, maxlen=100):
    text = translate_to_english(text)  # Menerjemahkan ke bahasa Inggris
    tokenized = tokenizer.texts_to_sequences([text])  # Tokenisasi
    padded = pad_sequences(tokenized, maxlen=maxlen, padding='post')  # Padding
    return padded

# Fungsi untuk prediksi
def predict_text(model, tokenizer, text):
    padded_input = preprocess_text(text, tokenizer)
    prediction = model.predict(padded_input)  # Probabilitas keluaran model
    predicted_class = 1 if prediction[0][0] > 0.5 else 0  # Binary classification
    predicted_prob = prediction[0][0] * 100 if predicted_class == 1 else (1 - prediction[0][0]) * 100

    # Penamaan kelas
    class_names = ['Non-Suicidal', 'Suicidal']
    return class_names[predicted_class], predicted_prob
