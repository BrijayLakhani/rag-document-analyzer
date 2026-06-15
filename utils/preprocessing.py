"""Classical text preprocessing with spaCy."""

from __future__ import annotations

import re
from pathlib import Path

import spacy
from spacy.language import Language


def load_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """Load spaCy model with a clear installation error for VS Code users."""

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. Run: "
            f"python -m spacy download {model_name}"
        ) from exc


def clean_text(text: str) -> str:
    """Normalize common PDF extraction artifacts before NLP processing."""

    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,;:!?$%()'\" -]", " ", text)
    return text.strip()


def preprocess_text(text: str, nlp: Language) -> str:
    """Lowercase, remove stopwords/punctuation, and lemmatize text."""

    cleaned = clean_text(text).lower()
    doc = nlp(cleaned)
    lemmas: list[str] = []

    for token in doc:
        if token.is_space or token.is_stop or token.is_punct:
            continue
        if not token.lemma_.strip():
            continue
        lemmas.append(token.lemma_.strip())

    return " ".join(lemmas)


def save_cleaned_text(cleaned_text: str, output_path: str | Path) -> None:
    """Save preprocessed text for later inspection."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cleaned_text, encoding="utf-8")
