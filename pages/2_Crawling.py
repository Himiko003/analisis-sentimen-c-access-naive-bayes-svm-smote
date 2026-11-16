import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Crawling Data (Read-Only)",
    page_icon="📥",
    layout="wide",
)

st.title("📥 Data Sumber Penelitian (Read-Only)")
st.markdown("---")

st.info(
    "Mode **Read-Only**: Fitur crawling dinonaktifkan untuk menjaga integritas "
    "dataset utama penelitian (`data/raw_dataset.csv`) agar tidak tertimpa."
)

RAW_DATA_PATH = "data/raw_dataset.csv"

if not os.path.exists(RAW_DATA_PATH):
    st.error("File `data/raw_dataset.csv` tidak ditemukan.")
    st.stop()

dataset = pd.read_csv(RAW_DATA_PATH)

c1, c2, c3 = st.columns(3)
c1.metric("📄 Total Data Mentah", len(dataset))
c2.metric("🔁 Duplikat Content", int(dataset["content"].duplicated().sum()))
c3.metric("⚠️ Missing Values", int(dataset.isnull().sum().sum()))

if "date" in dataset.columns:
    st.caption(f"Rentang tanggal ulasan: {dataset['date'].min()} s/d {dataset['date'].max()}")

st.markdown("---")
st.subheader("Preview Dataset Utama (`data/raw_dataset.csv`)")
st.dataframe(dataset.head(20), use_container_width=True)
