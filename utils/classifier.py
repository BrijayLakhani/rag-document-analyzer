"""Document classification with TF-IDF and Naive Bayes."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DEMO_TRAINING_DATA = [
    ("machine learning neural networks data model algorithm classification", "Technology"),
    ("software artificial intelligence database computer cloud system", "Technology"),
    ("hospital patient medicine treatment clinical disease doctor health", "Healthcare"),
    ("vaccine diagnosis medical research therapy healthcare symptoms", "Healthcare"),
    ("stock market investment revenue financial bank inflation economy", "Finance"),
    ("credit risk loan profit shareholder accounting financial report", "Finance"),
    ("court law legal contract judge regulation policy rights", "Legal"),
    ("agreement litigation compliance statute attorney legal document", "Legal"),
    ("student university education learning curriculum exam research", "Education"),
    ("school teaching academic classroom assessment course knowledge", "Education"),
]


def train_demo_classifier() -> Pipeline:
    """Train a small demo classifier so the project runs without external datasets."""

    texts, labels = zip(*DEMO_TRAINING_DATA)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1500)),
            ("nb", MultinomialNB()),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


def predict_document_category(text: str, model: Pipeline) -> tuple[str, dict[str, float]]:
    """Predict document category and return class probabilities."""

    prediction = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    confidence = {str(label): float(prob) for label, prob in zip(classes, probabilities)}
    return prediction, confidence


def save_classifier(model: Pipeline, output_path: str | Path) -> None:
    """Save classifier pipeline."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_or_train_classifier(model_path: str | Path) -> Pipeline:
    """Load an existing classifier or train the demo model."""

    path = Path(model_path)
    if path.exists():
        return joblib.load(path)

    model = train_demo_classifier()
    save_classifier(model, path)
    return model
