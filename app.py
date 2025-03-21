import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned model & tokenizer
model_path = "./finbert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    """Predicts sentiment of the given financial news headline."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_mapping[predicted_class_id]

# Streamlit Web App

st.title("Financial News Sentiment Analyzer")
headline = st.text_input("Enter a financial news headline:")

if st.button("Analyze"):
    if headline:
        sentiment = predict_sentiment(headline)
        st.write("Predicted Sentiment:", sentiment)
    else:
        st.write("Please enter a headline.")
