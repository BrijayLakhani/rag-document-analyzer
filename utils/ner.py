"""Named Entity Recognition with spaCy."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from spacy.language import Language


DEFAULT_ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT",
    "EVENT", "LAW", "NORP", "WORK_OF_ART", "QUANTITY", "LOC",
}


def extract_entities(
    text: str,
    nlp: Language,
    selected_labels: set[str] | None = None,
) -> list[dict[str, str | int]]:
    """Extract named entities filtered by label set."""

    labels = selected_labels or DEFAULT_ENTITY_LABELS
    doc = nlp(text)
    entities: list[dict[str, str | int]] = []

    for ent in doc.ents:
        if ent.label_ in labels:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

    return entities


def entity_dataframe(entities: list[dict[str, str | int]]) -> pd.DataFrame:
    """Convert entities to a DataFrame for display."""

    return pd.DataFrame(entities, columns=["text", "label", "start_char", "end_char"])


def entity_distribution(entities: list[dict[str, str | int]]) -> pd.DataFrame:
    """Return counts by entity label."""

    counts = Counter(str(entity["label"]) for entity in entities)
    return pd.DataFrame({"entity_type": list(counts.keys()), "count": list(counts.values())})


def save_entities(entities: list[dict[str, str | int]], output_path: str | Path) -> None:
    """Save NER output to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entities, indent=2), encoding="utf-8")
