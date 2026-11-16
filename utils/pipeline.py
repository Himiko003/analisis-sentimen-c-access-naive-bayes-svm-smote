"""Leakage-free dataset loading, stratified splitting, and TF-IDF pipeline."""

from dataclasses import dataclass
import json
import os
import tempfile
from typing import Any, Dict, Tuple

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils.preprocessing import ID_TO_LABEL, LABEL_MAP, VALID_SENTIMENTS

PREPROCESSED_DATA_PATH = "data/dataset_preprocessing.csv"
PIPELINE_CONFIG_PATH = "model/pipeline_config.json"


@dataclass
class PreparedSplitData:
    """Container for leakage-free train/test splits and TF-IDF artifacts."""

    X_train: csr_matrix
    X_test: csr_matrix
    y_train: pd.Series
    y_test: pd.Series
    X_train_text: pd.Series
    X_test_text: pd.Series
    vectorizer: TfidfVectorizer
    metadata: Dict[str, Any]


def atomic_save_json(data: Dict[str, Any], target_path: str) -> None:
    """Write JSON data atomically using a temporary file and os.replace."""
    parent_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(parent_dir, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=parent_dir,
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp_file:
            tmp_path = tmp_file.name
            json.dump(data, tmp_file, indent=4, ensure_ascii=False)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_and_validate_preprocessed_dataset(
    data_path: str = PREPROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Load dataset_preprocessing.csv and validate text and sentiment labels."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset hasil preprocessing tidak ditemukan di `{data_path}`. "
            "Silakan lakukan anotasi dan jalankan Preprocessing terlebih dahulu."
        )

    df = pd.read_csv(data_path)
    required_cols = {"preprocessed_text", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib pada dataset preprocessing tidak ditemukan: {sorted(missing)}"
        )

    work_df = df.copy()
    if "annotation_status" in work_df.columns:
        statuses = work_df["annotation_status"].fillna("").astype(str).str.strip().str.lower()
        if not (statuses == "terverifikasi").all():
            raise ValueError(
                "Dataset preprocessing mengandung data yang belum berstatus 'terverifikasi'."
            )

    work_df["preprocessed_text"] = work_df["preprocessed_text"].fillna("").astype(str)
    work_df["sentiment"] = work_df["sentiment"].fillna("").astype(str).str.strip().str.lower()

    if (work_df["preprocessed_text"].str.strip() == "").any():
        raise ValueError(
            "Ditemukan baris dengan `preprocessed_text` kosong pada dataset preprocessing."
        )

    invalid_sentiments = set(work_df["sentiment"]) - VALID_SENTIMENTS
    if invalid_sentiments:
        raise ValueError(
            f"Ditemukan label sentimen tidak valid: {sorted(invalid_sentiments)}"
        )

    work_df["label"] = work_df["sentiment"].map(LABEL_MAP).astype(int)
    return work_df.reset_index(drop=True)


def validate_stratification_feasibility(
    y: pd.Series,
    test_size: float = 0.2,
) -> None:
    """Ensure every class has enough samples for a stratified train-test split."""
    counts = y.value_counts()
    n_classes = len(counts)
    if n_classes < 2:
        raise ValueError(
            "Jumlah kelas sentimen kurang dari 2. Minimal 2 kelas diperlukan untuk training."
        )

    min_count = int(counts.min())
    if min_count < 2:
        insufficient = {
            ID_TO_LABEL.get(int(k), str(k)): int(v)
            for k, v in counts.items()
            if v < 2
        }
        raise ValueError(
            f"Jumlah data per kelas tidak mencukupi untuk stratified split (minimal 2 per kelas): {insufficient}"
        )

    total_test = int(round(len(y) * test_size))
    if total_test < n_classes:
        raise ValueError(
            f"Jumlah data uji ({total_test}) lebih kecil dari jumlah kelas ({n_classes}). "
            "Tambahkan lebih banyak data terverifikasi sebelum training."
        )


def _format_class_distribution(y: pd.Series) -> Dict[str, int]:
    """Format integer label series into ordered sentiment counts."""
    counts = y.value_counts().to_dict()
    return {
        label_name: int(counts.get(label_id, 0))
        for label_name, label_id in LABEL_MAP.items()
    }


def prepare_tfidf_split(
    data_path: str = PREPROCESSED_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> PreparedSplitData:
    """Split dataset first (80:20 stratified), then fit TF-IDF strictly on X_train."""
    df = load_and_validate_preprocessed_dataset(data_path)
    X_text = df["preprocessed_text"]
    y = df["label"]

    validate_stratification_feasibility(y, test_size=test_size)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Fit TF-IDF HANYA pada data training untuk mencegah data leakage.
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    metadata: Dict[str, Any] = {
        "data_path": data_path,
        "total_samples": int(len(df)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "split_ratio": f"{int((1 - test_size) * 100)}:{int(test_size * 100)}",
        "test_size": test_size,
        "random_state": random_state,
        "stratified": True,
        "label_mapping": LABEL_MAP,
        "class_distribution": {
            "total": _format_class_distribution(y),
            "train": _format_class_distribution(y_train),
            "test": _format_class_distribution(y_test),
        },
        "tfidf_config": {
            "max_features": max_features,
            "ngram_range": list(ngram_range),
            "min_df": min_df,
            "max_df": max_df,
            "fitted_on": "train_only",
        },
        "tfidf_num_features": int(X_train.shape[1]),
    }

    return PreparedSplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        X_train_text=X_train_text.reset_index(drop=True),
        X_test_text=X_test_text.reset_index(drop=True),
        vectorizer=vectorizer,
        metadata=metadata,
    )
