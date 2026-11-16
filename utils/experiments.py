"""Experiment runner, evaluation, and artifact storage for the 4 scenarios."""

from datetime import datetime, timezone
import json
import os
import platform
import tempfile
import time
from typing import Any, Dict, List, Tuple

import imblearn
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from utils.pipeline import PreparedSplitData, atomic_save_json
from utils.preprocessing import LABEL_MAP

ORDERED_LABELS: List[str] = ["negatif", "netral", "positif"]
ORDERED_LABEL_IDS: List[int] = [LABEL_MAP[name] for name in ORDERED_LABELS]

SCENARIO_SPECS: Dict[str, Dict[str, Any]] = {
    "nb_no_smote": {
        "name": "Naive Bayes (Tanpa SMOTE)",
        "algorithm": "MultinomialNB",
        "use_smote": False,
        "score_type": "predict_proba",
        "model_file": "model_nb_no_smote.pkl",
        "eval_file": "eval_nb_no_smote.json",
    },
    "nb_smote": {
        "name": "Naive Bayes + SMOTE",
        "algorithm": "MultinomialNB",
        "use_smote": True,
        "score_type": "predict_proba",
        "model_file": "model_nb_smote.pkl",
        "eval_file": "eval_nb_smote.json",
    },
    "svm_no_smote": {
        "name": "SVM (LinearSVC) (Tanpa SMOTE)",
        "algorithm": "LinearSVC",
        "use_smote": False,
        "score_type": "decision_score_not_probability",
        "model_file": "model_svm_no_smote.pkl",
        "eval_file": "eval_svm_no_smote.json",
    },
    "svm_smote": {
        "name": "SVM (LinearSVC) + SMOTE",
        "algorithm": "LinearSVC",
        "use_smote": True,
        "score_type": "decision_score_not_probability",
        "model_file": "model_svm_smote.pkl",
        "eval_file": "eval_svm_smote.json",
    },
}


def get_library_versions() -> Dict[str, str]:
    """Collect core library versions for experiment reproducibility."""
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def atomic_save_joblib(obj: Any, target_path: str) -> None:
    """Save a Python object with joblib atomically using a temp file and os.replace."""
    parent_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(parent_dir, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=parent_dir,
            suffix=".tmp",
        ) as tmp_file:
            tmp_path = tmp_file.name
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _format_counts(y: pd.Series) -> Dict[str, int]:
    """Return ordered class counts for [negatif, netral, positif]."""
    counts = pd.Series(y).value_counts().to_dict()
    return {
        label_name: int(counts.get(LABEL_MAP[label_name], 0))
        for label_name in ORDERED_LABELS
    }


def apply_smote_to_train(
    X_train: Any,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[Any, pd.Series, Dict[str, Any]]:
    """Apply SMOTE strictly on training features with safe k_neighbors and non-negativity check."""
    before_counts = _format_counts(y_train)
    min_class_count = int(pd.Series(y_train).value_counts().min())

    if min_class_count < 2:
        raise ValueError(
            "Jumlah sampel kelas minoritas pada data training kurang dari 2, "
            f"sehingga SMOTE tidak dapat diterapkan (distribusi: {before_counts})."
        )

    safe_k = min(5, min_class_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=safe_k)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    if issparse(X_resampled):
        min_val = float(X_resampled.data.min()) if X_resampled.nnz > 0 else 0.0
    else:
        min_val = float(np.min(X_resampled))

    if min_val < 0.0:
        raise ValueError(
            f"Hasil SMOTE mengandung nilai negatif ({min_val:.6f}), "
            "tidak kompatibel dengan MultinomialNB."
        )

    after_counts = _format_counts(y_resampled)
    smote_info: Dict[str, Any] = {
        "applied_on": "train_only",
        "k_neighbors": int(safe_k),
        "random_state": int(random_state),
        "before_smote": before_counts,
        "after_smote": after_counts,
        "train_samples_before": int(len(y_train)),
        "train_samples_after": int(len(y_resampled)),
        "min_feature_value": min_val,
    }
    return X_resampled, pd.Series(y_resampled), smote_info


def _build_classifier(algorithm: str, random_state: int = 42):
    """Instantiate classifier with explicit, reproducible parameters."""
    if algorithm == "MultinomialNB":
        return MultinomialNB(alpha=1.0, fit_prior=True)
    if algorithm == "LinearSVC":
        return LinearSVC(C=1.0, max_iter=5000, random_state=random_state)
    raise ValueError(f"Algoritma tidak dikenal: {algorithm}")


def evaluate_classifier(
    model: Any,
    X_test: Any,
    y_test: pd.Series,
    scenario_key: str,
    training_time_sec: float,
    train_samples_before_smote: int,
    train_samples_used: int,
    class_distribution_train: Dict[str, int],
    class_distribution_test: Dict[str, int],
    is_simulation: bool = False,
    smote_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute full evaluation metrics on original test set in fixed class order."""
    spec = SCENARIO_SPECS[scenario_key]

    t_eval_start = time.perf_counter()
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, labels=ORDERED_LABEL_IDS, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, labels=ORDERED_LABEL_IDS, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, labels=ORDERED_LABEL_IDS, average="macro", zero_division=0)

    prec_weighted = precision_score(y_test, y_pred, labels=ORDERED_LABEL_IDS, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, labels=ORDERED_LABEL_IDS, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, labels=ORDERED_LABEL_IDS, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=ORDERED_LABEL_IDS)
    report = classification_report(
        y_test,
        y_pred,
        labels=ORDERED_LABEL_IDS,
        target_names=ORDERED_LABELS,
        output_dict=True,
        zero_division=0,
    )
    evaluation_time_sec = time.perf_counter() - t_eval_start

    return {
        "experiment_type": "simulation" if is_simulation else "research",
        "is_simulation": bool(is_simulation),
        "scenario_key": scenario_key,
        "scenario_name": spec["name"],
        "algorithm": spec["algorithm"],
        "use_smote": spec["use_smote"],
        "score_type": spec["score_type"],
        "model_params": model.get_params(),
        "smote_params": (
            {
                "k_neighbors": smote_info["k_neighbors"],
                "random_state": smote_info["random_state"],
                "applied_on": smote_info["applied_on"],
            }
            if smote_info is not None
            else None
        ),
        "labels": ORDERED_LABELS,
        "label_mapping": LABEL_MAP,
        "train_samples_before_smote": int(train_samples_before_smote),
        "train_samples_used": int(train_samples_used),
        "test_samples": int(len(y_test)),
        "class_distribution_train": class_distribution_train,
        "class_distribution_test": class_distribution_test,
        "training_time_sec": round(float(training_time_sec), 4),
        "evaluation_time_sec": round(float(evaluation_time_sec), 4),
        "accuracy": round(float(acc) * 100, 2),
        "precision_macro": round(float(prec_macro) * 100, 2),
        "recall_macro": round(float(rec_macro) * 100, 2),
        "f1_macro": round(float(f1_macro) * 100, 2),
        "precision_weighted": round(float(prec_weighted) * 100, 2),
        "recall_weighted": round(float(rec_weighted) * 100, 2),
        "f1_weighted": round(float(f1_weighted) * 100, 2),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "smote_info": smote_info,
    }


def run_all_experiments(
    prepared: PreparedSplitData,
    model_dir: str = "model",
    random_state: int = 42,
    is_simulation: bool = False,
) -> Dict[str, Any]:
    """Train, evaluate, and atomically save all 4 scenarios and comparison summary."""
    os.makedirs(model_dir, exist_ok=True)
    lib_versions = get_library_versions()

    X_train_smote, y_train_smote, smote_info = apply_smote_to_train(
        prepared.X_train,
        prepared.y_train,
        random_state=random_state,
    )

    pipeline_meta = dict(prepared.metadata)
    pipeline_meta["experiment_type"] = "simulation" if is_simulation else "research"
    pipeline_meta["is_simulation"] = bool(is_simulation)
    pipeline_meta["library_versions"] = lib_versions

    atomic_save_joblib(prepared.vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    atomic_save_json(pipeline_meta, os.path.join(model_dir, "pipeline_config.json"))

    results: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    train_dist_before = prepared.metadata["class_distribution"]["train"]
    test_dist = prepared.metadata["class_distribution"]["test"]

    for scenario_key, spec in SCENARIO_SPECS.items():
        if spec["use_smote"]:
            X_tr, y_tr = X_train_smote, y_train_smote
            sc_smote_info = smote_info
            train_dist_used = smote_info["after_smote"]
        else:
            X_tr, y_tr = prepared.X_train, prepared.y_train
            sc_smote_info = None
            train_dist_used = train_dist_before

        clf = _build_classifier(spec["algorithm"], random_state=random_state)
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0

        eval_data = evaluate_classifier(
            model=clf,
            X_test=prepared.X_test,
            y_test=prepared.y_test,
            scenario_key=scenario_key,
            training_time_sec=elapsed,
            train_samples_before_smote=len(prepared.y_train),
            train_samples_used=len(y_tr),
            class_distribution_train=train_dist_used,
            class_distribution_test=test_dist,
            is_simulation=is_simulation,
            smote_info=sc_smote_info,
        )

        atomic_save_joblib(clf, os.path.join(model_dir, spec["model_file"]))
        atomic_save_json(eval_data, os.path.join(model_dir, spec["eval_file"]))

        results[scenario_key] = eval_data
        summary_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_name": spec["name"],
                "algorithm": spec["algorithm"],
                "use_smote": spec["use_smote"],
                "model_params": eval_data["model_params"],
                "smote_params": eval_data["smote_params"],
                "train_samples_before_smote": eval_data["train_samples_before_smote"],
                "train_samples_after_smote": eval_data["train_samples_used"],
                "test_samples": eval_data["test_samples"],
                "class_distribution_train": train_dist_used,
                "class_distribution_test": test_dist,
                "accuracy": eval_data["accuracy"],
                "precision_macro": eval_data["precision_macro"],
                "recall_macro": eval_data["recall_macro"],
                "f1_macro": eval_data["f1_macro"],
                "precision_weighted": eval_data["precision_weighted"],
                "recall_weighted": eval_data["recall_weighted"],
                "f1_weighted": eval_data["f1_weighted"],
                "training_time_sec": eval_data["training_time_sec"],
                "evaluation_time_sec": eval_data["evaluation_time_sec"],
            }
        )

    comparison_summary: Dict[str, Any] = {
        "experiment_type": "simulation" if is_simulation else "research",
        "is_simulation": bool(is_simulation),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": lib_versions,
        "pipeline_metadata": pipeline_meta,
        "smote_info": smote_info,
        "scenarios": summary_rows,
        "results": results,
    }
    atomic_save_json(comparison_summary, os.path.join(model_dir, "comparison_summary.json"))
    return comparison_summary


def load_experiment_artifacts(
    scenario_key: str,
    model_dir: str = "model",
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Reload vectorizer, scenario classifier, and evaluation JSON from disk."""
    if scenario_key not in SCENARIO_SPECS:
        raise ValueError(f"Skenario tidak dikenal: {scenario_key}")

    spec = SCENARIO_SPECS[scenario_key]
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(model_dir, spec["model_file"])
    eval_path = os.path.join(model_dir, spec["eval_file"])

    for path in (vec_path, model_path, eval_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artefak eksperimen tidak ditemukan: {path}")

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    return vectorizer, model, eval_data


def load_comparison_summary(model_dir: str = "model") -> Dict[str, Any]:
    """Load comparison_summary.json from model_dir."""
    summary_path = os.path.join(model_dir, "comparison_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Ringkasan perbandingan tidak ditemukan: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)
