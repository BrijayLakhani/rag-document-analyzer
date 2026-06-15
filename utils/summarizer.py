"""Extractive summarization using TextRank-style sentence ranking."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.tokenizer import sentence_tokenize


def textrank_summary(text: str, nlp, sentence_count: int = 5) -> str:
    """Generate an extractive summary with sentence similarity and PageRank."""

    sentences = sentence_tokenize(text, nlp)
    if not sentences:
        return ""
    if len(sentences) <= sentence_count:
        return " ".join(sentences)

    tfidf = TfidfVectorizer(stop_words="english")
    sentence_matrix = tfidf.fit_transform(sentences)
    similarity_matrix = cosine_similarity(sentence_matrix)
    np.fill_diagonal(similarity_matrix, 0)

    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph, max_iter=200)
    ranked_indices = sorted(scores, key=scores.get, reverse=True)[:sentence_count]
    ranked_indices = sorted(ranked_indices)

    return " ".join(sentences[index] for index in ranked_indices)


def save_summary(summary: str, output_path: str | Path) -> None:
    """Save extractive summary to a text file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")
