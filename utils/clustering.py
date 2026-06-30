"""Document chunk clustering with KMeans and PCA visualization."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_documents(
    chunks: list[str], requested_clusters: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster text chunks and return labels, distribution, and PCA coordinates."""

    if not chunks:
        return (
            pd.DataFrame(columns=["chunk_id", "cluster", "text", "x", "y"]),
            pd.DataFrame(columns=["cluster", "count"]),
        )

    cluster_count = max(1, min(requested_clusters, len(chunks)))
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    matrix = vectorizer.fit_transform(chunks)

    if cluster_count == 1:
        labels = [0] * len(chunks)
    else:
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(matrix).tolist()

    n_components = min(2, matrix.shape[1], matrix.shape[0])
    if n_components >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(matrix.toarray())
        x_coords = coords[:, 0].tolist()
        y_coords = coords[:, 1].tolist()
    else:
        x_coords = [0.0] * len(chunks)
        y_coords = [0.0] * len(chunks)

    cluster_rows = pd.DataFrame(
        {
            "chunk_id": list(range(1, len(chunks) + 1)),
            "cluster": labels,
            "text": chunks,
            "x": x_coords,
            "y": y_coords,
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
