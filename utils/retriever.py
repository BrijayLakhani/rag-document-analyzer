"""FAISS-based retrieval with cross-encoder re-ranking for the RAG stage."""

from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


def _sentence_split(text: str) -> list[str]:
    """Split text into readable sentence-like units without requiring spaCy."""

    text = " ".join(text.split())
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def chunk_text(text: str, chunk_size: int = 900, overlap_sentences: int = 1) -> list[str]:
    """Create sentence-aware chunks for retrieval."""

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
    """Create, store, and query a FAISS vector index with cross-encoder re-ranking."""

    def __init__(
        self,
        bi_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.embedding_model = SentenceTransformer(bi_encoder)
        self.reranker = CrossEncoder(cross_encoder)
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
        """Retrieve top-k chunks with FAISS then re-rank using the cross-encoder."""

        if self.index is None:
            raise ValueError("FAISS index has not been created yet.")

        candidate_k = min(top_k * 3, len(self.chunks))
        query = self.embedding_model.encode([question], convert_to_numpy=True, show_progress_bar=False)
        query = query.astype("float32")
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, candidate_k)

        candidates: list[dict[str, object]] = []
        for idx, bi_score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            candidates.append(
                {
                    "chunk_id": int(idx),
                    "bi_score": float(bi_score),
                    "text": self.chunks[int(idx)],
                }
            )

        if not candidates:
            return []

        pairs = [[question, c["text"]] for c in candidates]
        cross_scores = self.reranker.predict(pairs)

        for candidate, cs in zip(candidates, cross_scores):
            candidate["score"] = float(cs)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        top = candidates[:top_k]

        results: list[dict[str, object]] = []
        for rank, c in enumerate(top, start=1):
            results.append(
                {
                    "rank": rank,
                    "chunk_id": c["chunk_id"],
                    "score": c["score"],
                    "text": c["text"],
                }
            )
        return results


def save_retrieved_chunks(chunks: list[dict[str, object]], output_path: str | Path) -> None:
    """Save retrieved chunks to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
