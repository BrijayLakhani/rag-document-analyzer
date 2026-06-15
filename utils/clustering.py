"""Document chunk clustering with KMeans."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_documents(chunks: list[str], requested_clusters: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster text chunks and return labels plus distribution."""

    if not chunks:
        return pd.DataFrame(columns=["chunk_id", "cluster", "text"]), pd.DataFrame(columns=["cluster", "count"])

    cluster_count = max(1, min(requested_clusters, len(chunks)))
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    matrix = vectorizer.fit_transform(chunks)

    if cluster_count == 1:
        labels = [0] * len(chunks)
    else:
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(matrix).tolist()

    cluster_rows = pd.DataFrame(
        {
            "chunk_id": list(range(1, len(chunks) + 1)),
            "cluster": labels,
            "text": chunks,
        }
    )

    counts = Counter(labels)
    distribution = pd.DataFrame(
        {"cluster": list(counts.keys()), "count": list(counts.values())}
    ).sort_values("cluster")

    return cluster_rows, distribution


def save_clusters(cluster_rows: pd.DataFrame, output_path: str | Path) -> None:
    """Save clustering output to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster_rows.to_csv(path, index=False)
