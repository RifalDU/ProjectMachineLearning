import streamlit as st
import pickle
import re
import string

# Import NLTK dan download stopwords jika belum ada
import nltk
from nltk.corpus import stopwords

# Cek dan download stopwords hanya jika belum ada
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

# Load model dan vectorizer
with open('model_nb.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

stopwords_ind = set(stopwords.words('indonesian'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra spaces
    text = text.strip()
    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_ind]
    return ' '.join(tokens)

# Streamlit UI
st.title("Klasifikasi Sentimen Ulasan E-Commerce")
st.write("Masukkan ulasan produk dari e-commerce, lalu klik Prediksi untuk mengetahui sentimennya.")

user_input = st.text_area("Tulis ulasan di sini:")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        clean_review = clean_text(user_input)
        X_input = tfidf.transform([clean_review])
        prediction = model.predict(X_input)[0]
        st.success(f"Sentimen ulasan: **{prediction.capitalize()}**")
