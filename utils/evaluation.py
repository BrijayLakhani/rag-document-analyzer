"""Model evaluation helpers for the document classifier."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.classifier import DEMO_TRAINING_DATA


def evaluate_classifier(model: Pipeline | None = None) -> dict[str, object]:
    """Evaluate Naive Bayes classifier on the built-in demo dataset."""

    texts, labels = zip(*DEMO_TRAINING_DATA)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        list(texts),
        list(labels),
        test_size=0.4,
        random_state=42,
    )

    if model is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1500)),
                ("nb", MultinomialNB()),
            ]
        )

    model.fit(train_texts, train_labels)

    predictions = model.predict(test_texts)
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision": float(precision_score(test_labels, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(test_labels, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(test_labels, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            test_labels, predictions, zero_division=0, output_dict=True
        ),
    }


def save_evaluation(results: dict[str, object], output_path: str | Path) -> None:
    """Save evaluation metrics to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
