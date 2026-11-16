"""Shared text preprocessing and dataset validation utilities."""

from functools import lru_cache
import re
import string
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import stopwords
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

LABEL_MAP: Dict[str, int] = {
    "negatif": 0,
    "netral": 1,
    "positif": 2,
}

ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

VALID_SENTIMENTS = set(LABEL_MAP.keys())

CUSTOM_STOPWORDS = {
    "aplikasi",
    "caccess",
    "access",
    "commuter",
    "kci",
}

# Kata negasi wajib dipertahankan karena membalikkan polaritas sentimen.
NEGATION_WORDS = {
    "tidak",
    "bukan",
    "jangan",
    "belum",
    "kurang",
    "tak",
    "tanpa",
    "tiada",
    "enggak",
    "nggak",
    "gak",
    "ga",
    "gk",
    "tdk",
    "belom",
    "blm",
    "jgn",
    "bkn",
}

NORMALIZATION_DICT: Dict[str, str] = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "enggak": "tidak",
    "tdk": "tidak",
    "kagak": "tidak",
    "kgk": "tidak",
    "blm": "belum",
    "belom": "belum",
    "jgn": "jangan",
    "bkn": "bukan",
    "gabisa": "tidak bisa",
    "gbisa": "tidak bisa",
    "yg": "yang",
    "dgn": "dengan",
    "dr": "dari",
    "krn": "karena",
    "karna": "karena",
    "utk": "untuk",
    "tuk": "untuk",
    "bgt": "banget",
    "bgtt": "banget",
    "bngt": "banget",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "dpt": "dapat",
    "trs": "terus",
    "aja": "saja",
    "aj": "saja",
    "tp": "tapi",
    "tpi": "tapi",
    "pdhl": "padahal",
    "klo": "kalau",
    "kl": "kalau",
    "kalo": "kalau",
    "bs": "bisa",
    "eror": "error",
    "lemot": "lambat",
    "lelet": "lambat",
    "lola": "lambat",
    "apk": "aplikasi",
    "app": "aplikasi",
    "apps": "aplikasi",
    "min": "admin",
    "mimin": "admin",
    "gmn": "bagaimana",
    "gimana": "bagaimana",
    "knp": "kenapa",
    "bnr": "benar",
    "bener": "benar",
    "pke": "pakai",
    "pake": "pakai",
    "skrg": "sekarang",
    "skr": "sekarang",
    "lg": "lagi",
    "lgi": "lagi",
    "sm": "sama",
    "jd": "jadi",
    "jdi": "jadi",
    "msh": "masih",
    "msih": "masih",
    "sy": "saya",
    "gw": "saya",
    "gue": "saya",
    "km": "kamu",
    "org": "orang",
    "bnyk": "banyak",
    "byk": "banyak",
    "bbrp": "beberapa",
    "skali": "sekali",
    "skli": "sekali",
    "gt": "begitu",
    "gtu": "begitu",
}


@lru_cache(maxsize=1)
def get_stemmer():
    """Create and cache a Sastrawi stemmer instance."""
    factory = StemmerFactory()
    return factory.create_stemmer()


@lru_cache(maxsize=1)
def get_stopwords() -> set:
    """Load Indonesian stopwords while preserving negation words."""
    try:
        base_stopwords = set(stopwords.words("indonesian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        base_stopwords = set(stopwords.words("indonesian"))

    combined = base_stopwords.union(CUSTOM_STOPWORDS)
    return combined - NEGATION_WORDS


def clean_text(text: str) -> str:
    """Lowercase and remove URLs, mentions, digits, punctuation, and extra spaces."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    cleaned = str(text).lower()
    cleaned = re.sub(r"http\S+|www\S+", " ", cleaned)
    cleaned = re.sub(r"@\w+", " ", cleaned)
    cleaned = re.sub(r"#", " ", cleaned)
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = cleaned.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize(text: str) -> List[str]:
    """Tokenize cleaned text and normalize slang/abbreviations."""
    tokens: List[str] = []
    for word in text.split():
        mapped = NORMALIZATION_DICT.get(word, word)
        tokens.extend(mapped.split())
    return tokens


def remove_stopword(words: List[str]) -> List[str]:
    """Remove stopwords while keeping negation words intact."""
    stop_words = get_stopwords()
    return [word for word in words if word not in stop_words]


def stemming(words: List[str]) -> str:
    """Stem tokens using Sastrawi and join back into a single string."""
    stemmer = get_stemmer()
    stemmed = [stemmer.stem(word) for word in words]
    return " ".join(w for w in stemmed if w)


def preprocess_text(text: str) -> str:
    """Run the full preprocessing pipeline on a single review string."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    tokens = normalize(cleaned)
    tokens = remove_stopword(tokens)
    return stemming(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Validate manual annotations and preprocess only verified reviews."""
    required_cols = {"review_id", "content", "sentiment", "annotation_status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib hasil anotasi tidak ditemukan: {sorted(missing)}")

    total_rows = int(len(df))
    work_df = df.copy()
    work_df["content"] = work_df["content"].fillna("").astype(str)
    work_df["sentiment"] = work_df["sentiment"].fillna("").astype(str).str.strip().str.lower()
    work_df["annotation_status"] = (
        work_df["annotation_status"].fillna("").astype(str).str.strip().str.lower()
    )

    ambigu_count = int((work_df["annotation_status"] == "ambigu").sum())
    unlabeled_count = int((work_df["annotation_status"] == "belum_dilabeli").sum())

    verified_mask = (
        (work_df["annotation_status"] == "terverifikasi")
        & (work_df["sentiment"].isin(VALID_SENTIMENTS))
        & (work_df["content"].str.strip() != "")
    )
    work_df = work_df[verified_mask].drop_duplicates(subset="review_id").reset_index(drop=True)

    if work_df.empty:
        raise ValueError(
            "Belum ada ulasan berstatus 'terverifikasi' dengan label valid (negatif/netral/positif)."
        )

    verified_count = int(len(work_df))
    work_df["label"] = work_df["sentiment"].map(LABEL_MAP).astype(int)
    work_df["clean_text"] = work_df["content"].apply(clean_text)
    work_df["normalized"] = work_df["clean_text"].apply(normalize)
    work_df["filtered"] = work_df["normalized"].apply(remove_stopword)
    work_df["preprocessed_text"] = work_df["filtered"].apply(stemming)

    non_empty_mask = work_df["preprocessed_text"].str.strip() != ""
    empty_removed = int((~non_empty_mask).sum())
    work_df = work_df[non_empty_mask].reset_index(drop=True)

    stats: Dict[str, object] = {
        "total_annotated_dataset": total_rows,
        "verified_input": verified_count,
        "skipped_ambigu": ambigu_count,
        "skipped_unlabeled": unlabeled_count,
        "empty_removed": empty_removed,
        "final_count": int(len(work_df)),
        "sentiment_counts": work_df["sentiment"].value_counts().to_dict(),
    }
    return work_df, stats
