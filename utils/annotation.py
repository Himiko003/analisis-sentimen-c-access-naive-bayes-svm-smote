"""Persistent manual sentiment annotation management and validation."""

import hashlib
import os
import tempfile
from typing import Dict, Tuple

import pandas as pd

RAW_DATA_PATH = "data/raw_dataset.csv"
LABELED_DATA_PATH = "data/labeled_dataset.csv"

VALID_SENTIMENTS = {"negatif", "netral", "positif"}
VALID_STATUSES = {"belum_dilabeli", "terverifikasi", "ambigu"}

ANNOTATION_COLUMNS = [
    "review_id",
    "content",
    "score",
    "date",
    "sentiment",
    "annotation_status",
    "notes",
]


def generate_review_id(content: str) -> str:
    """Generate a deterministic, unique 16-char ID from exact review content."""
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:16]


def atomic_save_csv(df: pd.DataFrame, target_path: str) -> None:
    """Write DataFrame to CSV atomically using a temp file and os.replace."""
    parent_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(parent_dir, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=parent_dir,
            suffix=".tmp",
            encoding="utf-8-sig",
            newline="",
        ) as tmp_file:
            tmp_path = tmp_file.name
            df.to_csv(tmp_file, index=False)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def deduplicate_raw_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Deduplicate raw dataset by exact content, keeping the most recent metadata.

    Aturan pemilihan metadata duplikat:
    Jika terdapat teks ulasan (`content`) yang sama persis dengan `score` atau `date`
    berbeda, baris diurutkan berdasarkan `date` terbaru (descending) lalu entri
    paling baru dipertahankan tanpa mengubah isi teks `content`.
    """
    required = {"content", "score"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Kolom wajib pada raw dataset tidak ditemukan: {sorted(missing)}")

    raw_count = len(raw_df)
    work_df = raw_df.copy()
    work_df = work_df.dropna(subset=["content"]).copy()
    work_df["content"] = work_df["content"].astype(str)
    work_df = work_df[work_df["content"].str.strip() != ""].copy()

    if "date" in work_df.columns:
        work_df["_parsed_date"] = pd.to_datetime(work_df["date"], errors="coerce")
        work_df = work_df.sort_values(
            by="_parsed_date",
            ascending=False,
            na_position="last",
            kind="mergesort",
        )
        work_df = work_df.drop_duplicates(subset="content", keep="first")
        work_df = work_df.drop(columns=["_parsed_date"])
    else:
        work_df["date"] = ""
        work_df = work_df.drop_duplicates(subset="content", keep="first")

    work_df = work_df.reset_index(drop=True)
    work_df["review_id"] = work_df["content"].apply(generate_review_id)

    if work_df["review_id"].duplicated().any():
        raise ValueError("Terjadi duplikasi review_id setelah deduplikasi content.")

    dedup_count = len(work_df)
    stats = {
        "raw_count": int(raw_count),
        "dedup_count": int(dedup_count),
        "duplicates_removed": int(raw_count - dedup_count),
    }
    return work_df, stats


def _normalize_annotation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure annotation DataFrame has canonical columns and clean string values."""
    out = df.copy()
    for col in ANNOTATION_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    out = out[ANNOTATION_COLUMNS].copy()
    out["review_id"] = out["review_id"].fillna("").astype(str).str.strip()
    out["content"] = out["content"].fillna("").astype(str)
    out["date"] = out["date"].fillna("").astype(str)
    out["sentiment"] = out["sentiment"].fillna("").astype(str).str.strip().str.lower()
    out["annotation_status"] = (
        out["annotation_status"].fillna("belum_dilabeli").astype(str).str.strip().str.lower()
    )
    out.loc[~out["annotation_status"].isin(VALID_STATUSES), "annotation_status"] = "belum_dilabeli"
    out.loc[out["annotation_status"] == "belum_dilabeli", "sentiment"] = ""
    out["notes"] = out["notes"].fillna("").astype(str)
    return out


def load_or_init_labeled_dataset(
    raw_path: str = RAW_DATA_PATH,
    labeled_path: str = LABELED_DATA_PATH,
) -> pd.DataFrame:
    """Load persistent labeled dataset, or initialize it from raw_dataset.csv."""
    if os.path.exists(labeled_path):
        df = pd.read_csv(labeled_path, dtype={"review_id": str, "sentiment": str, "annotation_status": str, "notes": str})
        return _normalize_annotation_df(df)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset sumber tidak ditemukan: {raw_path}")

    raw_df = pd.read_csv(raw_path)
    dedup_df, _ = deduplicate_raw_dataset(raw_df)
    dedup_df["sentiment"] = ""
    dedup_df["annotation_status"] = "belum_dilabeli"
    dedup_df["notes"] = ""
    labeled_df = _normalize_annotation_df(dedup_df)
    atomic_save_csv(labeled_df, labeled_path)
    return labeled_df


def validate_annotation_row(sentiment: str, status: str) -> Tuple[str, str]:
    """Validate sentiment and annotation_status consistency."""
    clean_status = str(status).strip().lower()
    clean_sentiment = str(sentiment).strip().lower()

    if clean_status not in VALID_STATUSES:
        raise ValueError(f"Status anotasi tidak valid: '{status}'")

    if clean_status == "terverifikasi":
        if clean_sentiment not in VALID_SENTIMENTS:
            raise ValueError(
                "Status 'terverifikasi' hanya dapat diberikan jika sentiment adalah "
                "'negatif', 'netral', atau 'positif'."
            )
    elif clean_status == "belum_dilabeli":
        clean_sentiment = ""
    elif clean_status == "ambigu":
        if clean_sentiment not in VALID_SENTIMENTS:
            clean_sentiment = ""

    return clean_sentiment, clean_status


def update_single_annotation(
    df: pd.DataFrame,
    review_id: str,
    sentiment: str,
    status: str,
    notes: str = "",
) -> pd.DataFrame:
    """Update a single review's annotation after validation."""
    clean_sentiment, clean_status = validate_annotation_row(sentiment, status)
    mask = df["review_id"] == review_id
    if not mask.any():
        raise KeyError(f"review_id tidak ditemukan: {review_id}")

    updated = df.copy()
    updated.loc[mask, "sentiment"] = clean_sentiment
    updated.loc[mask, "annotation_status"] = clean_status
    updated.loc[mask, "notes"] = str(notes).strip()
    return updated


def merge_imported_annotations(
    existing_df: pd.DataFrame,
    imported_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Safely merge imported CSV annotations into existing_df without data loss."""
    required_cols = {"review_id", "sentiment", "annotation_status"}
    missing = required_cols - set(imported_df.columns)
    if missing:
        raise ValueError(f"Kolom wajib pada file impor tidak ditemukan: {sorted(missing)}")

    imp = imported_df.copy()
    imp["review_id"] = imp["review_id"].fillna("").astype(str).str.strip()
    imp["sentiment"] = imp["sentiment"].fillna("").astype(str).str.strip().str.lower()
    imp["annotation_status"] = imp["annotation_status"].fillna("").astype(str).str.strip().str.lower()
    if "notes" not in imp.columns:
        imp["notes"] = ""
    else:
        imp["notes"] = imp["notes"].fillna("").astype(str)

    if (imp["review_id"] == "").any():
        raise ValueError("Terdapat baris impor dengan review_id kosong.")

    if imp["review_id"].duplicated().any():
        raise ValueError("Terdapat duplikasi review_id pada file impor.")

    existing_ids = set(existing_df["review_id"].astype(str))
    unknown_ids = set(imp["review_id"]) - existing_ids
    if unknown_ids:
        sample_unknown = sorted(unknown_ids)[:5]
        raise ValueError(f"Ditemukan review_id tidak dikenal pada file impor: {sample_unknown}")

    merged = existing_df.copy()
    updated_count = 0
    skipped_unlabeled = 0

    for _, row in imp.iterrows():
        rid = row["review_id"]
        clean_sent, clean_stat = validate_annotation_row(
            row["sentiment"], row["annotation_status"]
        )
        # Jangan menimpa anotasi yang sudah ada dengan status belum_dilabeli
        if clean_stat == "belum_dilabeli":
            skipped_unlabeled += 1
            continue

        mask = merged["review_id"] == rid
        merged.loc[mask, "sentiment"] = clean_sent
        merged.loc[mask, "annotation_status"] = clean_stat
        if row["notes"].strip() != "":
            merged.loc[mask, "notes"] = row["notes"].strip()
        updated_count += 1

    stats = {
        "imported_rows": int(len(imp)),
        "updated_count": int(updated_count),
        "skipped_unlabeled": int(skipped_unlabeled),
    }
    return merged, stats


def get_annotation_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return summary counts of annotation statuses and verified sentiments."""
    verified = df[df["annotation_status"] == "terverifikasi"]
    return {
        "total": int(len(df)),
        "terverifikasi": int((df["annotation_status"] == "terverifikasi").sum()),
        "ambigu": int((df["annotation_status"] == "ambigu").sum()),
        "belum_dilabeli": int((df["annotation_status"] == "belum_dilabeli").sum()),
        "negatif": int((verified["sentiment"] == "negatif").sum()),
        "netral": int((verified["sentiment"] == "netral").sum()),
        "positif": int((verified["sentiment"] == "positif").sum()),
    }
