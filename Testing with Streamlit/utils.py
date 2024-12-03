import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langdetect import detect
from googletrans import Translator

# load tokenizer
def load_tokenizer(tokenizer_path='tokenizer.pkl'):
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

# load model
def load_model_file(model_path='model.keras'):
    model = load_model(model_path)
    return model

# translate
def translate_to_english(text):
    lang = detect(text)
    if lang != 'en':  
        translator = Translator()
        translated_text = translator.translate(text, src=lang, dest='en').text
        print(f"Translated text: {translated_text}")  # debugging
        return translated_text
    return text

# preprocessing teks input
def preprocess_text(text, tokenizer, maxlen=200):
    
    text = translate_to_english(text)

    new_data_tokenized = tokenizer.texts_to_sequences([text])
    
    new_data_padded = pad_sequences(new_data_tokenized, maxlen=maxlen, padding='post')
    
    print(f"Padded input: {new_data_padded}") 
    return new_data_padded

def predict_text(model, tokenizer, text):
    new_data_padded = preprocess_text(text, tokenizer)
    
    prediction = model.predict(new_data_padded)
    predicted_class = np.argmax(prediction, axis=1)
    
    predicted_prob = prediction[0][predicted_class[0]] * 100
    
    class_names = ['Not Suicidal', 'Suicidal']
    print(f"Predicted class: {class_names[predicted_class[0]]}")
    print(f"Prediction probability: {predicted_prob:.2f}%")
    
    return class_names[predicted_class[0]], predicted_prob
