"""Part-of-speech tagging utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import spacy
from spacy.language import Language


def extract_pos_tags(text: str, nlp: Language, max_tokens: int = 1000) -> list[dict[str, str]]:
    """Extract word, lemma, POS tag, detailed tag, and POS explanation."""

    doc = nlp(text)
    rows: list[dict[str, str]] = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue
        rows.append(
            {
                "word": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "explanation": spacy.explain(token.tag_) or spacy.explain(token.pos_) or "",
            }
        )
        if len(rows) >= max_tokens:
            break

    return rows


def pos_to_dataframe(pos_rows: list[dict[str, str]]) -> pd.DataFrame:
    """Convert POS results to a DataFrame for Streamlit display."""

    return pd.DataFrame(pos_rows, columns=["word", "lemma", "pos", "tag", "explanation"])


def save_pos_tags(pos_rows: list[dict[str, str]], output_path: str | Path) -> None:
    """Save POS tags to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pos_rows, indent=2), encoding="utf-8")
