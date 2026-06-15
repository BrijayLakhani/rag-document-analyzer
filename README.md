


- spaCy preprocessing, tokenization, POS tagging, and named entity recognition
- TF-IDF feature extraction
- Naive Bayes document classification
- KMeans document clustering
- TextRank-style extractive summarization
- Classification evaluation metrics
- FAISS retrieval and LLM-based answering only at the final stage

The classification module includes a small demo dataset so the project runs immediately. For a thesis or final Master's submission, replace it with a larger labeled document dataset for stronger experimental results.

## Folder Structure

text
rag_document_analyzer/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── outputs/
└── utils/
    ├── __init__.py
    ├── pdf_loader.py
    ├── preprocessing.py
    ├── tokenizer.py
    ├── pos_tagger.py
    ├── vectorizer.py
    ├── classifier.py
    ├── clustering.py
    ├── ner.py
    ├── summarizer.py
    ├── retriever.py
    ├── rag.py
    └── evaluation.py


## Setup in VS Code on Windows

Open the `rag_document_analyzer` folder in VS Code.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the Streamlit dashboard:

```powershell
streamlit run app.py
```

The app automatically saves:

- `outputs/cleaned_text.txt`
- `outputs/pos_tags.json`
- `outputs/entities.json`
- `outputs/tfidf_features.pkl`
- `outputs/clusters.csv`
- `outputs/summary.txt`
- `outputs/evaluation.json`
- `outputs/retrieved_chunks.json`
- `outputs/final_answer.json`
- `models/classifier.pkl`

## NLP Pipeline

1. Upload PDF
2. Extract text with PyPDF2
3. Clean and preprocess text using spaCy
4. Tokenize sentences and words
5. Extract POS tags
6. Extract named entities
7. Build TF-IDF features
8. Classify document with Naive Bayes
9. Cluster document chunks with KMeans
10. Summarize with TextRank-style extractive summarization
11. Build FAISS retrieval index
12. Ask questions using RAG
13. Save and display evaluation metrics

## RAG Note

The LLM is used only in `utils/rag.py` after FAISS retrieves relevant chunks. This keeps the main project focused on classical NLP while still demonstrating a modern RAG question-answering stage.

The default RAG model is `google/flan-t5-small` and the embedding model is `sentence-transformers/all-MiniLM-L6-v2`. These models may download during first use.
