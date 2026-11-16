import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dataset",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Dataset Analisis Sentimen")
st.markdown("---")

RAW_PATH = "data/raw_dataset.csv"
LABELED_PATH = "data/labeled_dataset.csv"
PREP_PATH = "data/dataset_preprocessing.csv"

dataset_choice = st.radio(
    "Pilih Sumber Dataset",
    [
        "Dataset Hasil Preprocessing (dataset_preprocessing.csv)",
        "Dataset Anotasi Manual (labeled_dataset.csv)",
        "Dataset Mentah Asli (raw_dataset.csv)",
    ],
    horizontal=True,
)

if dataset_choice.startswith("Dataset Hasil Preprocessing"):
    target_path = PREP_PATH
    download_name = "dataset_preprocessing.csv"
elif dataset_choice.startswith("Dataset Anotasi Manual"):
    target_path = LABELED_PATH
    download_name = "labeled_dataset.csv"
else:
    target_path = RAW_PATH
    download_name = "raw_dataset.csv"

if not os.path.exists(target_path):
    st.warning(f"Dataset `{target_path}` belum tersedia.")
    st.stop()

dataset = pd.read_csv(target_path)

if target_path == PREP_PATH and "preprocessed_text" in dataset.columns:
    dataset["preprocessed_text"] = dataset["preprocessed_text"].fillna("").astype(str)
    dataset = dataset[dataset["preprocessed_text"].str.strip() != ""].reset_index(drop=True)

st.success(f"Total Dataset ({download_name}) : {len(dataset)} Review")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    keyword = st.text_input("🔍 Cari Review")

with col2:
    if "sentiment" in dataset.columns:
        sentiment = st.selectbox(
            "Filter Sentimen",
            ["Semua", "positif", "negatif", "netral"],
        )
    else:
        sentiment = "Semua"
        st.selectbox("Filter Sentimen", ["Tidak tersedia pada raw dataset"], disabled=True)

filtered = dataset.copy()

if keyword != "":
    filtered = filtered[
        filtered["content"].str.contains(keyword, case=False, na=False)
    ]

if sentiment != "Semua" and "sentiment" in filtered.columns:
    filtered = filtered[filtered["sentiment"] == sentiment]

st.dataframe(filtered, use_container_width=True, height=550)

st.markdown("---")

csv = filtered.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name=download_name,
    mime="text/csv",
)

if "sentiment" in dataset.columns:
    st.markdown("---")
    st.subheader("📊 Statistik Sentimen")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Review", len(dataset))
    c2.metric("Positif", int((dataset["sentiment"] == "positif").sum()))
    c3.metric("Negatif", int((dataset["sentiment"] == "negatif").sum()))
    c4.metric("Netral", int((dataset["sentiment"] == "netral").sum()))
