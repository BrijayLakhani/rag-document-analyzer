# RAG Document Analyzer

> Upload any PDF and instantly analyze it with a full classical NLP pipeline — named entities, keyword extraction, clustering, extractive summarization, and AI-powered Q&A using RAG with FAISS vector search.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-1.8-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What It Does

RAG Document Analyzer is a Streamlit web application that processes any PDF through a **classical NLP pipeline** and ends with a **Retrieval-Augmented Generation (RAG)** Q&A stage powered by a local LLM. No cloud API required.

---

## Features

- **PDF Upload** — drag and drop any PDF, text is extracted automatically
- **Text Preprocessing** — cleaning, lemmatization, and stopword removal via spaCy
- **POS Tagging** — part-of-speech tags for every token in the document
- **Named Entity Recognition** — detects persons, organizations, locations, dates, money, and products
- **TF-IDF Keyword Extraction** — top weighted terms and bigrams across the document
- **Document Classification** — Naive Bayes classifier predicts the document category with confidence scores
- **KMeans Clustering** — groups document chunks into semantic clusters
- **Extractive Summarization** — TextRank-style summary using sentence similarity and PageRank
- **RAG Question Answering** — FAISS retrieves the most relevant chunks, flan-t5 generates the answer
- **Evaluation Metrics** — accuracy, precision, recall, F1 score, and classification report
- **Auto-save Outputs** — all results saved to `/outputs` as JSON, CSV, and text files

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit |
| NLP | spaCy `en_core_web_sm` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS (CPU) |
| Language Model | `google/flan-t5-small` |
| Classification | scikit-learn Naive Bayes + TF-IDF |
| Clustering | scikit-learn KMeans |
| Summarization | TextRank (NetworkX + cosine similarity) |
| PDF Parsing | PyPDF2 |

---

## Project Structure

```
rag_document_analyzer/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt
├── README.md
├── data/                   # Uploaded PDFs (not tracked in git)
├── models/                 # Trained classifier (auto-generated)
├── outputs/                # Analysis results (auto-generated)
└── utils/
    ├── pdf_loader.py       # PDF text extraction
    ├── preprocessing.py    # Text cleaning and lemmatization
    ├── tokenizer.py        # Sentence and word tokenization
    ├── pos_tagger.py       # Part-of-speech tagging
    ├── ner.py              # Named entity recognition
    ├── vectorizer.py       # TF-IDF feature extraction
    ├── classifier.py       # Document category classification
    ├── clustering.py       # KMeans chunk clustering
    ├── summarizer.py       # TextRank extractive summarization
    ├── retriever.py        # FAISS vector index and retrieval
    ├── rag.py              # RAG question answering with flan-t5
    └── evaluation.py       # Classifier evaluation metrics
```

---

## NLP Pipeline

```
PDF Upload
    │
    ▼
Text Extraction (PyPDF2)
    │
    ▼
Cleaning & Preprocessing (spaCy)
    │
    ├──► POS Tagging
    ├──► Named Entity Recognition
    ├──► TF-IDF Keyword Extraction
    ├──► Document Classification (Naive Bayes)
    ├──► KMeans Clustering
    ├──► TextRank Summarization
    │
    ▼
FAISS Vector Index
    │
    ▼
RAG Question Answering (flan-t5-small)
```

---

## Setup

### Requirements

- Python 3.11 or higher
- Windows / macOS / Linux

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/BrijayLakhani/rag-document-analyzer.git
cd rag-document-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download the spaCy language model
python -m spacy download en_core_web_sm
```

### Run

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Usage

1. **Upload** a PDF using the sidebar file uploader
2. Adjust **KMeans clusters**, **summary sentences**, and **retrieved chunks** using the sliders
3. Click **Analyze Document** — the full pipeline runs automatically
4. Explore results across the 9 tabs: Cleaned Text, POS Tagging, Named Entities, TF-IDF, Classification, Clustering, Summary, RAG Q&A, Evaluation
5. In the **RAG Q&A** tab, type any question about the document and click **Generate Answer**

---

## Output Files

All outputs are saved automatically after each analysis run:

| File | Contents |
|---|---|
| `outputs/cleaned_text.txt` | Preprocessed document text |
| `outputs/pos_tags.json` | Part-of-speech tags |
| `outputs/entities.json` | Named entities |
| `outputs/clusters.csv` | Chunk cluster assignments |
| `outputs/summary.txt` | Extractive summary |
| `outputs/retrieved_chunks.json` | RAG retrieval results |
| `outputs/final_answer.json` | Question, answer, and sources |
| `outputs/evaluation.json` | Classifier evaluation metrics |
| `models/classifier.pkl` | Trained Naive Bayes classifier |

---

## Notes

- The first run downloads the flan-t5-small and MiniLM embedding models (~500 MB total). Subsequent runs use the local cache.
- The classifier uses a small built-in demo dataset for instant startup. For stronger results, replace it with a labeled dataset of your own documents.
- Scanned PDFs (image-only) are not supported — the PDF must contain selectable text.

---

## License

MIT License — free to use, modify, and distribute.
