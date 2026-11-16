import pandas as pd
import streamlit as st

from utils.annotation import (
    LABELED_DATA_PATH,
    atomic_save_csv,
    get_annotation_summary,
    load_or_init_labeled_dataset,
    merge_imported_annotations,
    update_single_annotation,
)

st.set_page_config(
    page_title="Annotation",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ Anotasi Sentimen Manual")
st.markdown("---")

st.info(
    "Label sentimen ditetapkan berdasarkan **makna teks ulasan (`content`)**, "
    "bukan berdasarkan rating (`score`). Setiap perubahan disimpan secara atomik ke "
    f"`{LABELED_DATA_PATH}` agar progres tidak hilang."
)

df = load_or_init_labeled_dataset()
summary = get_annotation_summary(df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📄 Total Unik", summary["total"])
c2.metric("✅ Terverifikasi", summary["terverifikasi"])
c3.metric("❓ Ambigu", summary["ambigu"])
c4.metric("⏳ Belum Dilabeli", summary["belum_dilabeli"])
progress_pct = (summary["terverifikasi"] / summary["total"] * 100) if summary["total"] else 0.0
c5.metric("📊 Progres Verifikasi", f"{progress_pct:.1f}%")

if summary["total"] > 0:
    st.progress(summary["terverifikasi"] / summary["total"])

st.caption(
    f"Distribusi label terverifikasi — Negatif: **{summary['negatif']}** | "
    f"Netral: **{summary['netral']}** | Positif: **{summary['positif']}**"
)

st.markdown("---")

tab_focus, tab_batch = st.tabs(["📝 Anotasi Fokus (Satu per Satu)", "📤 Ekspor / Impor & Tabel"])

with tab_focus:
    f1, f2 = st.columns([1, 2])
    with f1:
        status_filter = st.selectbox(
            "Filter Antrian Status",
            ["belum_dilabeli", "ambigu", "terverifikasi", "semua"],
        )
    with f2:
        search_kw = st.text_input("🔍 Cari Teks Ulasan", key="ann_search")

    queue_df = df.copy()
    if status_filter != "semua":
        queue_df = queue_df[queue_df["annotation_status"] == status_filter]
    if search_kw.strip():
        queue_df = queue_df[
            queue_df["content"].str.contains(search_kw.strip(), case=False, na=False)
        ]
    queue_df = queue_df.reset_index(drop=True)

    if queue_df.empty:
        st.success("Tidak ada ulasan pada filter antrian ini.")
    else:
        idx = st.number_input(
            f"Nomor Antrian (1 - {len(queue_df)})",
            min_value=1,
            max_value=len(queue_df),
            value=1,
            step=1,
        )
        row = queue_df.iloc[int(idx) - 1]
        rid = row["review_id"]

        st.markdown(f"**Review ID:** `{rid}` | **Tanggal:** `{row['date']}` | **Status Saat Ini:** `{row['annotation_status']}` | **Label:** `{row['sentiment'] or '-'}`")
        st.caption(f"Metadata rating Play Store: {row['score']} (hanya referensi, tidak menentukan sentimen)")

        st.text_area(
            "Isi Teks Ulasan (`content`)",
            value=row["content"],
            height=150,
            disabled=True,
            key=f"content_view_{rid}",
        )

        notes_input = st.text_input(
            "Catatan Peneliti (opsional, misalnya alasan ulasan ambigu)",
            value=row["notes"],
            key=f"notes_{rid}",
        )

        b1, b2, b3, b4 = st.columns(4)
        action = None
        if b1.button("😠 Negatif", use_container_width=True, key=f"btn_neg_{rid}"):
            action = ("negatif", "terverifikasi")
        if b2.button("😐 Netral", use_container_width=True, key=f"btn_net_{rid}"):
            action = ("netral", "terverifikasi")
        if b3.button("😊 Positif", use_container_width=True, key=f"btn_pos_{rid}"):
            action = ("positif", "terverifikasi")
        if b4.button("❓ Tandai Ambigu", use_container_width=True, key=f"btn_amb_{rid}"):
            action = ("", "ambigu")

        if action is not None:
            new_sent, new_status = action
            updated_df = update_single_annotation(
                df,
                review_id=rid,
                sentiment=new_sent,
                status=new_status,
                notes=notes_input,
            )
            atomic_save_csv(updated_df, LABELED_DATA_PATH)
            st.rerun()

with tab_batch:
    st.subheader("Ekspor File Anotasi")
    export_csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇ Download labeled_dataset.csv",
        data=export_csv,
        file_name="labeled_dataset.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Impor CSV Hasil Anotasi (Aman & Tervalidasi)")
    st.caption(
        "File impor wajib memiliki kolom `review_id`, `sentiment`, dan `annotation_status`. "
        "Baris berstatus `belum_dilabeli` pada file impor tidak akan menghapus anotasi yang sudah ada."
    )
    uploaded_file = st.file_uploader("Unggah CSV Anotasi", type=["csv"])
    if uploaded_file is not None:
        try:
            imp_df = pd.read_csv(
                uploaded_file,
                dtype={"review_id": str, "sentiment": str, "annotation_status": str, "notes": str},
            )
            merged_df, merge_stats = merge_imported_annotations(df, imp_df)
            st.success(
                f"Validasi berhasil! {merge_stats['updated_count']} baris siap diperbarui "
                f"({merge_stats['skipped_unlabeled']} baris belum_dilabeli dilewati)."
            )
            if st.button("💾 Terapkan & Simpan Impor Anotasi", use_container_width=True):
                atomic_save_csv(merged_df, LABELED_DATA_PATH)
                st.success("Hasil impor berhasil disimpan secara atomik.")
                st.rerun()
        except Exception as exc:
            st.error(f"Gagal memvalidasi file impor: {exc}")

    st.markdown("---")
    st.subheader("Preview Tabel Anotasi")
    st.dataframe(df.head(50), use_container_width=True)
