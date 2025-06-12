import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt

# --- Styling ---
st.set_page_config(page_title="Sentimen E-Commerce Naive Bayes", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f2f6fc;}
    .stButton>button {background-color: #4f8bf9; color: white;}
    .st-cb {background-color: #e3eafc;}
    .st-bb {background-color: #f9fafb;}
    </style>
""", unsafe_allow_html=True)

# --- Load Model & Vectorizer ---
with open('model_nb.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# --- Stopwords Indo ---
import nltk
from nltk.corpus import stopwords
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')
stopwords_ind = set(stopwords.words('indonesian'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_ind]
    return ' '.join(tokens)

# --- UI Header ---
st.markdown("<h1 style='color:#4f8bf9'>Klasifikasi Sentimen Ulasan E-Commerce</h1>", unsafe_allow_html=True)
st.markdown("**Kelompok 4 | Universitas Nusa Putra | 2025**")
st.write("Analisis sentimen otomatis pada ulasan e-commerce menggunakan algoritma Naive Bayes.")

# --- File Upload ---
st.markdown("### 1. Upload Dataset Ulasan")
uploaded_file = st.file_uploader("Upload file dataset (.csv atau .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='latin1')
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"File `{uploaded_file.name}` berhasil diupload!")

    # --- Data Preview ---
    with st.expander("Lihat Sampel Data"):
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"Jumlah data: {df.shape[0]} baris")
    
    # --- Kolom Review & Label ---
    st.markdown("### 2. Preprocessing Data")
    review_col = st.selectbox("Pilih kolom teks ulasan:", df.columns)
    label_col = st.selectbox("Pilih kolom label sentimen:", df.columns)
    
    df['clean_review'] = df[review_col].astype(str).apply(clean_text)
    st.write("Contoh hasil preprocessing:")
    st.dataframe(df[[review_col, 'clean_review']].head(5), use_container_width=True)
    
    # --- Prediksi Batch ---
    st.markdown("### 3. Prediksi Sentimen Ulasan")
    X = tfidf.transform(df['clean_review'])
    pred = model.predict(X)
    df['prediksi_sentimen'] = pred
    
    label_map = {1: "Positif", 0: "Negatif", "positif": "Positif", "negatif": "Negatif"}
    df['prediksi_sentimen'] = df['prediksi_sentimen'].map(label_map).fillna(df['prediksi_sentimen'])
    
    st.dataframe(df[[review_col, 'clean_review', 'prediksi_sentimen']].head(10), use_container_width=True)
    
    # --- Evaluasi Model (Jika label tersedia) ---
    if set(df[label_col].unique()) & set(["positif", "negatif", 0, 1]):
        st.markdown("### 4. Evaluasi Model")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
        
        # Normalisasi label
        label_true = df[label_col].map(label_map).fillna(df[label_col])
        label_pred = df['prediksi_sentimen']
        
        acc = accuracy_score(label_true, label_pred)
        prec = precision_score(label_true, label_pred, pos_label="Positif")
        rec = recall_score(label_true, label_pred, pos_label="Positif")
        f1 = f1_score(label_true, label_pred, pos_label="Positif")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi", f"{acc*100:.2f}%")
        col2.metric("Presisi", f"{prec*100:.2f}%")
        col3.metric("Recall", f"{rec*100:.2f}%")
        col4.metric("F1-Score", f"{f1*100:.2f}%")
        
        # Confusion Matrix
        st.write("#### Confusion Matrix")
        cm = confusion_matrix(label_true, label_pred, labels=["Positif", "Negatif"])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Positif", "Negatif"], yticklabels=["Positif", "Negatif"], ax=ax)
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        st.pyplot(fig)
    else:
        st.info("Kolom label tidak ditemukan atau format label tidak dikenali. Evaluasi model tidak dapat dilakukan.")
    
    # --- Download Hasil ---
    st.markdown("### 5. Unduh Hasil Prediksi")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Hasil (.csv)", csv, "hasil_prediksi.csv", "text/csv")
    
else:
    st.info("Silakan upload file dataset ulasan e-commerce (.csv atau .xlsx) untuk memulai analisis.")

# --- Single Prediction ---
st.markdown("---")
st.markdown("### Coba Prediksi Ulasan Manual")
user_review = st.text_area("Masukkan satu ulasan produk e-commerce di sini:")
if st.button("Prediksi Sentimen Ulasan"):
    if user_review.strip() == "":
        st.warning("Teks ulasan belum diisi.")
    else:
        clean_user_review = clean_text(user_review)
        X_user = tfidf.transform([clean_user_review])
        user_pred = model.predict(X_user)[0]
        user_sentimen = label_map.get(user_pred, str(user_pred))
        st.markdown(f"<h3 style='color:#4f8bf9'>Sentimen: {user_sentimen}</h3>", unsafe_allow_html=True)
