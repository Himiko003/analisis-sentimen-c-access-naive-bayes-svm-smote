
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import json
import os
import numpy as np

st.set_page_config(page_title="Visualization", page_icon="📊", layout="wide")

st.title("📊 Visualisasi Analisis Sentimen")
st.markdown("---")

DATA_PATH = "data/dataset_preprocessing.csv"
EVAL_PATH = "model/evaluation.json"

if not os.path.exists(DATA_PATH):
    st.warning("Dataset belum tersedia.")
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

# ==============================
# PIE & BAR
# ==============================

c1,c2 = st.columns(2)

with c1:
    st.subheader("🥧 Distribusi Sentimen")
    sentiment = dataset["sentiment"].value_counts().reset_index()
    sentiment.columns=["Sentiment","Jumlah"]
    fig = px.pie(
        sentiment,
        names="Sentiment",
        values="Jumlah",
        hole=.45
    )
    st.plotly_chart(fig,use_container_width=True)

with c2:
    st.subheader("⭐ Distribusi Rating")
    rating = dataset["score"].value_counts().sort_index().reset_index()
    rating.columns=["Rating","Jumlah"]
    fig = px.bar(
        rating,
        x="Rating",
        y="Jumlah",
        text="Jumlah"
    )
    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

# ==============================
# WORDCLOUD & TOP WORD
# ==============================

text = " ".join(dataset["preprocessed_text"].tolist())

c1,c2 = st.columns(2)

with c1:
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

with c2:
    st.subheader("📈 Top 20 Kata")

    counter = Counter(text.split())
    top = pd.DataFrame(counter.most_common(20),columns=["Kata","Jumlah"])

    fig = px.bar(
        top,
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

# ==============================
# HEATMAP CONFUSION MATRIX
# ==============================

st.subheader("🔥 Confusion Matrix")

if os.path.exists(EVAL_PATH):

    with open(EVAL_PATH,"r",encoding="utf-8") as f:
        evaluation = json.load(f)

    cm = np.array(evaluation["confusion_matrix"])

    labels=["Negatif","Netral","Positif"]

    heatmap = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hovertemplate=
            "Actual : %{y}<br>Predicted : %{x}<br>Jumlah : %{z}<extra></extra>"
        )
    )

    heatmap.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=500
    )

    st.plotly_chart(heatmap,use_container_width=True)

else:
    st.info("Model belum tersedia.")

st.markdown("---")

# ==============================
# STATISTIK
# ==============================

st.subheader("📊 Statistik Dataset")

a,b,c,d = st.columns(4)

a.metric("Total Review",len(dataset))
b.metric("Positif",len(dataset[dataset.sentiment=="positif"]))
c.metric("Negatif",len(dataset[dataset.sentiment=="negatif"]))
d.metric("Netral",len(dataset[dataset.sentiment=="netral"]))

st.markdown("---")

st.subheader("📄 Preview Dataset")

st.dataframe(
    dataset.head(20),
    use_container_width=True
)
