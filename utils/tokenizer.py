"""Sentence and word tokenization with spaCy."""

from __future__ import annotations

from spacy.language import Language


def sentence_tokenize(text: str, nlp: Language) -> list[str]:
    """Split text into sentences."""

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def word_tokenize(text: str, nlp: Language) -> list[str]:
    """Split text into non-space word tokens."""

    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]
