import streamlit as st
import pickle
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load model dan vectorizer
with open('model_nb.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

stopwords_ind = set(stopwords.words('indonesian'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_ind]
    return ' '.join(tokens)

st.title("Klasifikasi Sentimen Ulasan E-Commerce (Naive Bayes)")
review = st.text_area("Masukkan ulasan e-commerce:")

if st.button("Prediksi Sentimen"):
    clean_review = clean_text(review)
    X_input = tfidf.transform([clean_review])
    pred = model.predict(X_input)[0]
    st.write(f"Sentimen ulasan: **{pred.capitalize()}**")
