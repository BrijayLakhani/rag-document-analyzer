"""Document classification with TF-IDF and Naive Bayes."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DEMO_TRAINING_DATA = [
    # ── Technology ──
    ("machine learning neural networks deep learning model training classification algorithms supervised unsupervised reinforcement", "Technology"),
    ("software development agile methodology version control git repository code review continuous integration deployment", "Technology"),
    ("cloud computing amazon web services azure google cloud infrastructure serverless deployment kubernetes docker", "Technology"),
    ("artificial intelligence natural language processing computer vision robotics automation intelligent systems cognitive", "Technology"),
    ("database management sql nosql mongodb postgresql query optimization data modeling schema design indexing", "Technology"),
    ("cybersecurity encryption firewall network security vulnerability penetration testing threat detection intrusion prevention", "Technology"),
    ("blockchain cryptocurrency decentralized ledger smart contracts ethereum bitcoin distributed systems consensus", "Technology"),
    ("internet of things sensors embedded systems microcontrollers arduino raspberry pi connected devices iot", "Technology"),
    ("mobile application development android ios swift kotlin react native flutter user interface design", "Technology"),
    ("data science analytics big data hadoop spark machine learning pipeline feature engineering preprocessing", "Technology"),
    ("web development html css javascript react angular frontend backend api rest graphql microservices", "Technology"),
    ("devops continuous deployment docker containers microservices monitoring logging infrastructure automation pipeline", "Technology"),

    # ── Healthcare ──
    ("hospital patient medicine treatment clinical disease doctor health diagnosis prescription therapy care", "Healthcare"),
    ("vaccine immunization public health epidemiology infectious disease outbreak prevention control pandemic", "Healthcare"),
    ("surgical procedure operation anesthesia recovery postoperative complications hospital operating room sterile", "Healthcare"),
    ("mental health psychology therapy counseling depression anxiety disorder cognitive behavioral treatment wellness", "Healthcare"),
    ("pharmaceutical drug clinical trial fda approval dosage side effects medication prescription pharmacy", "Healthcare"),
    ("medical imaging radiology mri ct scan ultrasound diagnostic imaging pathology xray screening", "Healthcare"),
    ("nursing patient care wound management vital signs medical records healthcare team bedside monitoring", "Healthcare"),
    ("cardiology heart disease cardiovascular hypertension cholesterol cardiac arrhythmia echocardiogram stent bypass", "Healthcare"),
    ("pediatrics children infant neonatal growth development vaccination childhood diseases immunization schedule", "Healthcare"),
    ("oncology cancer tumor chemotherapy radiation therapy malignant benign prognosis staging remission", "Healthcare"),
    ("emergency medicine trauma triage critical care intensive unit resuscitation stabilization ambulance", "Healthcare"),
    ("telemedicine remote consultation virtual healthcare digital health wearable devices patient monitoring telehealth", "Healthcare"),

    # ── Finance ──
    ("stock market investment portfolio dividend equity trading bull bear market capitalization shares", "Finance"),
    ("banking loan mortgage interest rate credit score debt repayment amortization principal balance", "Finance"),
    ("financial statement balance sheet income statement cash flow revenue expenses profit margin analysis", "Finance"),
    ("cryptocurrency bitcoin ethereum blockchain decentralized finance defi trading exchange wallet staking", "Finance"),
    ("insurance premium policy claim deductible coverage liability underwriting risk assessment actuary", "Finance"),
    ("accounting audit tax compliance gaap ifrs financial reporting bookkeeping ledger reconciliation", "Finance"),
    ("venture capital startup funding series round valuation equity stake investor pitch angel seed", "Finance"),
    ("mutual fund etf index fund portfolio diversification asset allocation rebalancing passive investing", "Finance"),
    ("foreign exchange forex currency trading exchange rate hedging international markets arbitrage spread", "Finance"),
    ("real estate investment property mortgage rental income commercial residential valuation appraisal", "Finance"),
    ("risk management derivatives futures options hedging financial risk market volatility exposure", "Finance"),
    ("central bank monetary policy inflation interest rate quantitative easing economic stimulus fiscal", "Finance"),

    # ── Legal ──
    ("court law legal contract judge regulation policy rights legislation statute jurisdiction ruling", "Legal"),
    ("intellectual property patent trademark copyright infringement licensing royalty protection filing", "Legal"),
    ("criminal law prosecution defense verdict sentencing plea bargain felony misdemeanor trial jury", "Legal"),
    ("corporate law merger acquisition shareholder board director governance compliance fiduciary duty", "Legal"),
    ("employment law workplace discrimination harassment wrongful termination labor rights wage overtime", "Legal"),
    ("constitutional law amendment rights freedom speech press religion due process equal protection", "Legal"),
    ("contract law agreement breach damages remedy obligation terms conditions enforceable consideration", "Legal"),
    ("immigration law visa citizenship deportation asylum refugee status green card naturalization", "Legal"),
    ("environmental law regulation pollution emissions climate change epa compliance sustainability carbon", "Legal"),
    ("family law divorce custody child support alimony prenuptial agreement adoption guardianship", "Legal"),
    ("real estate law property deed title transfer lease landlord tenant zoning easement", "Legal"),
    ("international law treaty sovereignty jurisdiction diplomacy human rights humanitarian law extradition", "Legal"),

    # ── Education ──
    ("student university education learning curriculum exam research academic degree scholarship tuition", "Education"),
    ("teaching pedagogy classroom instruction assessment evaluation grading rubric feedback differentiation", "Education"),
    ("online learning elearning distance education virtual classroom mooc platform digital resources", "Education"),
    ("special education disability learning disorder accommodation individualized program inclusion support", "Education"),
    ("higher education university college graduate undergraduate masters doctoral thesis dissertation research", "Education"),
    ("educational technology edtech interactive learning gamification adaptive platform tools analytics", "Education"),
    ("school administration principal superintendent budget policy board education management leadership", "Education"),
    ("stem education science technology engineering mathematics laboratory experiment research innovation", "Education"),
    ("early childhood education preschool kindergarten developmental play based learning readiness social", "Education"),
    ("curriculum development standards alignment learning objectives outcomes competency based design mapping", "Education"),
    ("student assessment standardized testing formative summative evaluation performance measurement benchmark", "Education"),
    ("literacy reading writing composition comprehension phonics vocabulary language arts fluency", "Education"),
]


def train_demo_classifier() -> Pipeline:
    """Train a classifier on the expanded demo dataset."""

    texts, labels = zip(*DEMO_TRAINING_DATA)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
            ("nb", MultinomialNB(alpha=0.1)),
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
        path.unlink()

    model = train_demo_classifier()
    save_classifier(model, path)
    return model
