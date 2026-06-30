"""Keyphrase extraction using KeyBERT."""

from __future__ import annotations

import json
from pathlib import Path

from keybert import KeyBERT


_model: KeyBERT | None = None


def _get_model() -> KeyBERT:
    global _model
    if _model is None:
        _model = KeyBERT(model="sentence-transformers/all-MiniLM-L6-v2")
    return _model


def extract_keywords(
    text: str,
    top_n: int = 20,
    keyphrase_ngram_range: tuple[int, int] = (1, 3),
    use_mmr: bool = True,
    diversity: float = 0.5,
) -> list[dict[str, object]]:
    """Extract keyphrases with scores using KeyBERT."""

    model = _get_model()
    results = model.extract_keywords(
        text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words="english",
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity,
    )
    return [{"keyword": kw, "score": round(float(score), 4)} for kw, score in results]


def save_keywords(keywords: list[dict[str, object]], output_path: str | Path) -> None:
    """Save keywords to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keywords, indent=2), encoding="utf-8")
