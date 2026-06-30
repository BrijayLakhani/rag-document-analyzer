"""Model evaluation helpers for the document classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from utils.classifier import DEMO_TRAINING_DATA


def evaluate_classifier(model: Pipeline | None = None) -> dict[str, object]:
    """Evaluate classifier using stratified k-fold cross-validation."""

    texts, labels = zip(*DEMO_TRAINING_DATA)
    texts = list(texts)
    labels = list(labels)

    if model is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
                ("nb", MultinomialNB(alpha=0.1)),
            ]
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, texts, labels, cv=cv)

    unique_labels = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=unique_labels)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            labels, predictions, zero_division=0, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "confusion_labels": unique_labels,
    }


def save_evaluation(results: dict[str, object], output_path: str | Path) -> None:
    """Save evaluation metrics to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v

    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
