
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import json
import os

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Analisis Sentimen C-Access")
st.markdown("---")

DATA_PATH = "data/dataset_preprocessing.csv"
EVALUATION_PATH = "model/evaluation.json"

# =========================
# Cek Dataset
# =========================
if not os.path.exists(DATA_PATH):
    st.warning("Dataset belum tersedia.")
    st.info("Silakan lakukan Crawling → Preprocessing → Training terlebih dahulu.")
    st.stop()

dataset = pd.read_csv(DATA_PATH)

dataset["preprocessed_text"] = (
    dataset["preprocessed_text"]
    .fillna("")
    .astype(str)
)

dataset = dataset[
    dataset["preprocessed_text"].str.strip() != ""
].reset_index(drop=True)

# =========================
# Statistik Dataset
# =========================
total_review = len(dataset)
positif = len(dataset[dataset["sentiment"] == "positif"])
negatif = len(dataset[dataset["sentiment"] == "negatif"])
netral = len(dataset[dataset["sentiment"] == "netral"])

accuracy = precision = recall = f1score = 0

if os.path.exists(EVALUATION_PATH):
    with open(EVALUATION_PATH, "r", encoding="utf-8") as f:
        evaluation = json.load(f)

    accuracy = evaluation.get("accuracy", 0)
    precision = evaluation.get("precision", 0)
    recall = evaluation.get("recall", 0)
    f1score = evaluation.get("f1score", 0)

# =========================
# KPI
# =========================
c1,c2,c3,c4,c5 = st.columns(5)

c1.metric("📄 Total Review", total_review)
c2.metric("🎯 Accuracy", f"{accuracy}%")
c3.metric("📌 Precision", f"{precision}%")
c4.metric("📈 Recall", f"{recall}%")
c5.metric("🏆 F1 Score", f"{f1score}%")

st.markdown("---")

left,right = st.columns(2)

with left:
    st.subheader("🥧 Distribusi Sentimen")

    sentiment_df = dataset["sentiment"].value_counts().reset_index()
    sentiment_df.columns=["Sentiment","Jumlah"]

    fig = px.pie(
        sentiment_df,
        names="Sentiment",
        values="Jumlah",
        hole=.45
    )

    st.plotly_chart(fig,use_container_width=True)

with right:
    st.subheader("⭐ Distribusi Rating")

    rating = dataset["score"].value_counts().sort_index()

    rating_df = pd.DataFrame({
        "Rating":rating.index,
        "Jumlah":rating.values
    })

    fig = px.bar(
        rating_df,
        x="Rating",
        y="Jumlah",
        text="Jumlah"
    )

    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

left,right = st.columns(2)

text = " ".join(dataset["preprocessed_text"].tolist())

with left:
    st.subheader("☁️ WordCloud")

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white"
    ).generate(text)

    fig,ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

with right:
    st.subheader("📈 Top 20 Kata")

    counter = Counter(text.split())

    top20 = pd.DataFrame(
        counter.most_common(20),
        columns=["Kata","Jumlah"]
    )

    fig = px.bar(
        top20,
        x="Jumlah",
        y="Kata",
        orientation="h",
        text="Jumlah"
    )

    fig.update_layout(
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

st.subheader("🔥 Confusion Matrix")

if os.path.exists(EVALUATION_PATH):

    cm = evaluation["confusion_matrix"]

    cm_df = pd.DataFrame(
        cm,
        index=["Negatif","Netral","Positif"],
        columns=["Negatif","Netral","Positif"]
    )

    fig, ax = plt.subplots(figsize=(6,4))

    im = ax.imshow(cm_df.values)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))

    ax.set_xticklabels(cm_df.columns)
    ax.set_yticklabels(cm_df.index)

    for i in range(3):
        for j in range(3):
            ax.text(j,i,cm_df.iloc[i,j],
                    ha="center",
                    va="center")

    plt.colorbar(im)

    st.pyplot(fig)

else:

    st.info("Model belum dilatih.")

st.markdown("---")

st.subheader("📄 Preview Dataset")

st.dataframe(
    dataset.head(20),
    use_container_width=True
)
