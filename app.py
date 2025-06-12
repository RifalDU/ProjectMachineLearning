# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing function
def preprocess_text(text):
    # Case folding
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenization
    tokens = text.split()
    # Stopword removal (contoh sederhana, bisa diganti list lebih lengkap)
    stopwords = set([
        'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'ini', 'itu', 'karena', 'jika', 'ada', 'tidak', 'sudah'
    ])
    tokens = [word for word in tokens if word not in stopwords]
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit UI
st.title('Klasifikasi Sentimen Ulasan E-Commerce (Naive Bayes)')
st.write('Upload file CSV (kolom: text, sentiment) untuk memulai.')

uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Contoh data:")
    st.dataframe(df.head())

    # Preprocessing
    st.subheader("Preprocessing Data")
    df['clean_text'] = df['text'].astype(str).apply(preprocess_text)
    st.dataframe(df[['text', 'clean_text', 'sentiment']].head())

    # Split data
    X = df['clean_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Feature Extraction
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model Training
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    st.subheader("Evaluasi Model")
    st.write(f"**Akurasi:** {acc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif','Positif'], yticklabels=['Negatif','Positif'], ax=ax)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    st.pyplot(fig)

    # Classification Report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=['Negatif', 'Positif']))

    # Visualisasi distribusi label
    st.subheader("Distribusi Sentimen")
    fig2, ax2 = plt.subplots()
    df['sentiment'].value_counts().plot(kind='bar', ax=ax2)
    plt.xticks(rotation=0)
    st.pyplot(fig2)

    # Prediksi real-time
    st.subheader("Prediksi Sentimen Ulasan Baru")
    user_input = st.text_area("Masukkan ulasan e-commerce di sini:")
    if st.button("Prediksi Sentimen"):
        if user_input.strip() != "":
            clean_input = preprocess_text(user_input)
            input_vec = vectorizer.transform([clean_input])
            pred = model.predict(input_vec)[0]
            label = "Positif" if int(pred) == 1 else "Negatif"
            st.success(f"Sentimen prediksi: **{label}**")
        else:
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")

else:
    st.info("Silakan upload file dataset CSV dengan kolom 'text' dan 'sentiment'.")
