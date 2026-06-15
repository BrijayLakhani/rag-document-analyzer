"""FAISS-based retrieval for the RAG stage."""

from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _sentence_split(text: str) -> list[str]:
    """Split text into readable sentence-like units without requiring spaCy."""

    text = " ".join(text.split())
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def chunk_text(text: str, chunk_size: int = 900, overlap_sentences: int = 1) -> list[str]:
    """Create sentence-aware chunks for retrieval.

    Sentence chunks improve answer quality because retrieved context does not
    start or end in the middle of words.
    """

    sentences = _sentence_split(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current and current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current).strip())
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_length = sum(len(item) for item in current)

        current.append(sentence)
        current_length += sentence_length

    if current:
        chunks.append(" ".join(current).strip())

    clean_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size + 250:
            clean_chunks.append(chunk)
            continue
        words = chunk.split()
        window: list[str] = []
        for word in words:
            if sum(len(item) + 1 for item in window) + len(word) > chunk_size:
                clean_chunks.append(" ".join(window))
                window = []
            window.append(word)
        if window:
            clean_chunks.append(" ".join(window))

    return clean_chunks


class FaissRetriever:
    """Create, store, and query a FAISS vector index."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedding_model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[str] = []

    def create_index(self, chunks: list[str]) -> None:
        """Embed chunks and create an inner-product FAISS index."""

        self.chunks = chunks
        if not chunks:
            raise ValueError("Cannot build retrieval index because no chunks were provided.")

        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, question: str, top_k: int = 3) -> list[dict[str, object]]:
        """Retrieve top-k chunks for a user question."""

        if self.index is None:
            raise ValueError("FAISS index has not been created yet.")

        query = self.embedding_model.encode([question], convert_to_numpy=True, show_progress_bar=False)
        query = query.astype("float32")
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, min(top_k, len(self.chunks)))
        results: list[dict[str, object]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            results.append(
                {
                    "rank": rank,
                    "chunk_id": int(idx),
                    "score": float(score),
                    "text": self.chunks[int(idx)],
                }
            )
        return results


def save_retrieved_chunks(chunks: list[dict[str, object]], output_path: str | Path) -> None:
    """Save retrieved chunks to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
