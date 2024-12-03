import streamlit as st
from utils import load_tokenizer, load_model_file, predict_text

# model load
@st.cache_resource
def load_resources():
    tokenizer = load_tokenizer()
    model = load_model_file()
    return tokenizer, model

tokenizer, model = load_resources()

# Streamlit
st.title("Text Classification App")
st.write("This app predicts if a given text is related to 'suicide' or 'non-suicide'.")

# Input teks
user_input = st.text_area("Enter your text:", placeholder="Type here...")

if st.button("Predict"):
    if user_input.strip():
        predicted_class, predicted_prob = predict_text(model, tokenizer, user_input)
        
        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {predicted_prob:.2f}%")
    else:
        st.error("Please enter some text to classify.")
