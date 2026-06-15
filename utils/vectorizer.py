"""TF-IDF feature extraction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    documents: list[str],
    max_features: int = 1000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[TfidfVectorizer, object, pd.DataFrame]:
    """Fit TF-IDF on documents and return vectorizer, sparse matrix, and feature scores."""

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(documents)
    scores = matrix.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()

    features = pd.DataFrame({"feature": feature_names, "score": scores})
    features = features.sort_values("score", ascending=False).reset_index(drop=True)
    return vectorizer, matrix, features


def save_tfidf(vectorizer: TfidfVectorizer, matrix: object, output_path: str | Path) -> None:
    """Save TF-IDF vectorizer and matrix using joblib."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, path)
