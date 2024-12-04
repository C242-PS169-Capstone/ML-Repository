import pickle
import tensorflow as tf
import numpy as np
import re
import string
from tensorflow.keras.models import load_model
from langdetect import detect
from googletrans import Translator

# Load vectorizer
def load_vectorizer(vectorizer_path):
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    vectorizer_layer = tf.keras.layers.TextVectorization.from_config(vectorizer['config'])
    vectorizer_layer.set_weights(vectorizer['weights'])
    vectorizer_layer.set_vocabulary(vectorizer['vocabulary'])
    return vectorizer_layer

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

# Eliminate abbreviations
def remove_abbr (text):
    abb = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "dont": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "idk": "i do not know",
    "he'd've": "he would have",
    "he'll": "he will",  "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is", "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",      "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have", "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have", "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"
    }
    abb_re = re.compile('(%s)' % '|'.join(abb.keys()))
    def replace(match):
        return abb[match.group(0)]
    return abb_re.sub(replace, text)

def remove_stopwords(text):
    with open('stopwords.txt', 'r') as file:
        stopwords = set(file.read().splitlines())
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)

def remove_punctuation(text):
    words = text.split()
    filtered_words = [word for word in words if word not in string.punctuation and not word.isdigit()]
    return ' '.join(filtered_words)

def remove_special_characters(text):
    words = text.split()
    filtered_words = [re.sub(r'[^a-zA-Z0-9\s]', '', word) for word in words]
    return ' '.join(filtered_words)

def remove_emoji(text):
    words = text.split()
    filtered_words = [re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001FB00-\U0001FBFF\U0001FE00-\U0001FE0F\U0001F004]+', '', word) for word in words]
    return ' '.join(filtered_words)

# Preprocess input text
def preprocess_text(text, vectorizer):
    text = translate_to_english(text)
    text = remove_abbr(text)
    text = remove_emoji(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    
    preprocessed = vectorizer([text])
    return preprocessed

# Predict text class
def predict_text(model, vectorizer, text):
    input = preprocess_text(text, vectorizer)
    prediction = model.predict(input)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_prob = prediction[0][predicted_class[0]] * 100

    class_names = ['Normal', 'Bipolar', 'Stress', 'Anxiety', 'Depression', 'Personality disorder']
    return class_names[predicted_class[0]], predicted_prob