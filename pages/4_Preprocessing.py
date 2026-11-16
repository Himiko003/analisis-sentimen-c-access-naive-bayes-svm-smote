import os
import pandas as pd
import streamlit as st

from utils.annotation import LABELED_DATA_PATH, atomic_save_csv, get_annotation_summary
from utils.preprocessing import NEGATION_WORDS, preprocess_dataframe

st.set_page_config(
    page_title="Preprocessing",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 Preprocessing Data")
st.markdown("---")

OUTPUT_PATH = "data/dataset_preprocessing.csv"

if not os.path.exists(LABELED_DATA_PATH):
    st.warning(
        f"File `{LABELED_DATA_PATH}` belum tersedia. "
        "Silakan buka halaman **Annotation** terlebih dahulu untuk menginisialisasi dan melabeli ulasan."
    )
    st.stop()

labeled_df = pd.read_csv(
    LABELED_DATA_PATH,
    dtype={"review_id": str, "sentiment": str, "annotation_status": str, "notes": str},
)
summary = get_annotation_summary(labeled_df)

st.info(
    f"Sumber data: `{LABELED_DATA_PATH}` (**{summary['total']}** ulasan unik). "
    "Hanya ulasan berstatus **`terverifikasi`** yang akan diproses untuk training. "
    "Ulasan **`ambigu`** dan **`belum_dilabeli`** tidak diikutsertakan."
)

m1, m2, m3 = st.columns(3)
m1.metric("✅ Siap Diproses (Terverifikasi)", summary["terverifikasi"])
m2.metric("❓ Dilewati (Ambigu)", summary["ambigu"])
m3.metric("⏳ Dilewati (Belum Dilabeli)", summary["belum_dilabeli"])

with st.expander("Kata Negasi yang Dipertahankan"):
    st.write(", ".join(sorted(NEGATION_WORDS)))

if summary["terverifikasi"] == 0:
    st.warning(
        "Belum ada ulasan berstatus `terverifikasi`. "
        "Silakan lakukan anotasi sentimen pada halaman **Annotation** terlebih dahulu."
    )
    st.stop()

if st.button("🧹 Mulai Preprocessing", use_container_width=True):
    with st.spinner("Memproses data terverifikasi (cleaning, normalisasi, stopword, stemming)..."):
        processed_df, stats = preprocess_dataframe(labeled_df)
        atomic_save_csv(processed_df, OUTPUT_PATH)

    st.success(f"Preprocessing berhasil! Hasil disimpan secara atomik ke `{OUTPUT_PATH}`.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input Terverifikasi", stats["verified_input"])
    c2.metric("Ambigu / Belum Dilabeli Dilewati", stats["skipped_ambigu"] + stats["skipped_unlabeled"])
    c3.metric("Teks Kosong Dihapus", stats["empty_removed"])
    c4.metric("Data Bersih Final", stats["final_count"])

    st.markdown("### Distribusi Sentimen Hasil Preprocessing")
    st.json(stats["sentiment_counts"])

    st.markdown("### Preview Hasil Preprocessing")
    st.dataframe(
        processed_df[["review_id", "content", "sentiment", "label", "clean_text", "preprocessed_text"]].head(15),
        use_container_width=True,
    )
elif os.path.exists(OUTPUT_PATH):
    existing_df = pd.read_csv(OUTPUT_PATH)
    st.caption(f"File `{OUTPUT_PATH}` saat ini tersedia dengan {len(existing_df)} baris.")
