import os
import pandas as pd
import streamlit as st

from utils.experiments import ORDERED_LABELS, SCENARIO_SPECS, run_all_experiments
from utils.pipeline import (
    PREPROCESSED_DATA_PATH,
    load_and_validate_preprocessed_dataset,
    prepare_tfidf_split,
)

st.set_page_config(page_title="Training Model", page_icon="🤖", layout="wide")
st.title("🤖 Training & Evaluasi 4 Skenario Eksperimen")
st.markdown("---")

if not os.path.exists(PREPROCESSED_DATA_PATH):
    st.warning(
        f"File `{PREPROCESSED_DATA_PATH}` belum tersedia. "
        "Silakan lakukan Anotasi → Preprocessing terlebih dahulu."
    )
    st.stop()

try:
    dataset = load_and_validate_preprocessed_dataset(PREPROCESSED_DATA_PATH)
except Exception as exc:
    st.error(f"Validasi dataset gagal: {exc}")
    st.stop()

st.success(f"Dataset terverifikasi siap digunakan ({len(dataset)} data).")

st.markdown(
    """
    **Skenario Eksperimen (Split 80:20 Stratified & TF-IDF Fit pada Training Saja):**
    1. Multinomial Naive Bayes (Tanpa SMOTE)
    2. Multinomial Naive Bayes + SMOTE
    3. Linear Support Vector Machine (`LinearSVC`) (Tanpa SMOTE)
    4. Linear Support Vector Machine (`LinearSVC`) + SMOTE
    """
)

if st.button("🚀 Jalankan Keempat Skenario Eksperimen", use_container_width=True):
    try:
        with st.spinner("Menjalankan split, TF-IDF, SMOTE (train saja), dan training 4 skenario..."):
            prepared = prepare_tfidf_split(PREPROCESSED_DATA_PATH)
            summary = run_all_experiments(prepared, model_dir="model", random_state=42)

        st.success("✅ Keempat skenario berhasil dilatih, dievaluasi, dan disimpan ke folder `model/`.")

        meta = summary["pipeline_metadata"]
        smote_info = summary["smote_info"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data", meta["total_samples"])
        c2.metric("Data Training (80%)", meta["train_samples"])
        c3.metric("Data Testing (20%)", meta["test_samples"])
        c4.metric("SMOTE k_neighbors", smote_info["k_neighbors"])

        st.markdown("### Distribusi Kelas Training (Sebelum vs Sesudah SMOTE) & Testing")
        dist_df = pd.DataFrame(
            {
                "Train (Sebelum SMOTE)": smote_info["before_smote"],
                "Train (Sesudah SMOTE)": smote_info["after_smote"],
                "Testing (Asli, Tanpa SMOTE)": meta["class_distribution"]["test"],
            }
        )
        st.dataframe(dist_df, use_container_width=True)

        st.markdown("### Ringkasan Hasil 4 Skenario")
        summary_df = pd.DataFrame(summary["scenarios"])
        display_cols = [
            "scenario_name",
            "algorithm",
            "use_smote",
            "train_samples_before_smote",
            "train_samples_after_smote",
            "test_samples",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "training_time_sec",
            "evaluation_time_sec",
        ]
        st.dataframe(summary_df[display_cols], use_container_width=True)

        with st.expander("Metadata Reproducibility & Versi Library"):
            st.json(summary["library_versions"])

        st.markdown("### Detail Evaluasi Per Skenario")
        tabs = st.tabs([SCENARIO_SPECS[k]["name"] for k in SCENARIO_SPECS])
        for tab, key in zip(tabs, SCENARIO_SPECS):
            res = summary["results"][key]
            with tab:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{res['accuracy']:.2f}%")
                m2.metric("F1 Macro", f"{res['f1_macro']:.2f}%")
                m3.metric("F1 Weighted", f"{res['f1_weighted']:.2f}%")
                m4.metric("Training", f"{res['training_time_sec']:.4f} s")
                m5.metric("Evaluasi", f"{res['evaluation_time_sec']:.4f} s")

                st.write("**Confusion Matrix** (Baris: Aktual, Kolom: Prediksi)")
                cm_df = pd.DataFrame(res["confusion_matrix"], index=ORDERED_LABELS, columns=ORDERED_LABELS)
                st.dataframe(cm_df, use_container_width=True)

                st.write("**Classification Report**")
                st.dataframe(pd.DataFrame(res["classification_report"]).transpose(), use_container_width=True)

    except Exception as exc:
        st.error(f"Gagal menjalankan eksperimen: {exc}")
