import streamlit as st
import joblib
import os
import re
import string

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Prediction",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Prediksi Sentimen")
st.markdown("---")

# ============================================================
# CEK MODEL
# ============================================================

MODEL_PATH = "model/model_naive_bayes.pkl"
TFIDF_PATH = "model/tfidf_vectorizer.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

if not (
    os.path.exists(MODEL_PATH)
    and os.path.exists(TFIDF_PATH)
    and os.path.exists(ENCODER_PATH)
):
    st.warning("Silakan lakukan Training Model terlebih dahulu.")
    st.stop()

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
encoder = joblib.load(ENCODER_PATH)

# ============================================================
# NLP
# ============================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(stopwords.words("indonesian"))

custom_stopwords = {
    "aplikasi",
    "caccess",
    "access",
    "commuter",
    "kci"
}

stop_words.update(custom_stopwords)

normalization_dict = {

    "gk":"tidak",
    "ga":"tidak",
    "gak":"tidak",
    "nggak":"tidak",
    "tdk":"tidak",

    "yg":"yang",
    "dgn":"dengan",
    "dr":"dari",
    "krn":"karena",
    "utk":"untuk",

    "bgt":"banget",
    "bgtt":"banget",
    "bngt":"banget",

    "udh":"sudah",
    "udah":"sudah",

    "dpt":"dapat",
    "trs":"terus",
    "aja":"saja"
}

# ============================================================
# FUNCTION
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize(text):

    words = text.split()

    words = [
        normalization_dict[word]
        if word in normalization_dict
        else word
        for word in words
    ]

    return words


def remove_stopword(words):

    return [
        word
        for word in words
        if word not in stop_words
    ]


def stemming(words):

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


def preprocess(text):

    text = clean_text(text)

    words = normalize(text)

    words = remove_stopword(words)

    text = stemming(words)

    return text

# ============================================================
# INPUT
# ============================================================

review = st.text_area(
    "Masukkan Review",
    height=200,
    placeholder="Contoh : Aplikasi ini sangat membantu..."
)

# ============================================================
# BUTTON
# ============================================================

if st.button("🔍 Prediksi", use_container_width=True):

    if review.strip() == "":

        st.warning("Masukkan review terlebih dahulu.")

    else:

        clean_review = preprocess(review)

        vector = tfidf.transform([clean_review])

        prediction = model.predict(vector)

        sentiment = encoder.inverse_transform(prediction)[0]

        st.markdown("---")

        st.subheader("Hasil Prediksi")

        if sentiment == "positif":

            st.success("😊 Sentimen Positif")

        elif sentiment == "negatif":

            st.error("😠 Sentimen Negatif")

        else:

            st.info("😐 Sentimen Netral")

        st.markdown("### Hasil Preprocessing")

        st.code(clean_review)