"""Streamlit dashboard for RAG Document Analyzer with Classical NLP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.classifier import load_or_train_classifier, predict_document_category
from utils.clustering import cluster_documents, save_clusters
from utils.evaluation import evaluate_classifier, save_evaluation
from utils.ner import entity_dataframe, entity_distribution, extract_entities, save_entities
from utils.pdf_loader import load_pdf, save_uploaded_pdf
from utils.pos_tagger import extract_pos_tags, pos_to_dataframe, save_pos_tags
from utils.preprocessing import clean_text, load_spacy_model, preprocess_text, save_cleaned_text
from utils.rag import RAGQuestionAnswerer, save_final_answer
from utils.retriever import FaissRetriever, chunk_text, save_retrieved_chunks
from utils.summarizer import save_summary, textrank_summary
from utils.tokenizer import sentence_tokenize, word_tokenize
from utils.vectorizer import build_tfidf, save_tfidf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


def ensure_directories() -> None:
    """Create required project directories."""

    for directory in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_nlp():
    return load_spacy_model()


@st.cache_resource
def get_classifier():
    return load_or_train_classifier(MODEL_DIR / "classifier.pkl")


@st.cache_resource
def get_rag_answerer():
    return RAGQuestionAnswerer()


def render_metrics(metrics: dict[str, object]) -> None:
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.2f}")
    cols[1].metric("Precision", f"{metrics['precision']:.2f}")
    cols[2].metric("Recall", f"{metrics['recall']:.2f}")
    cols[3].metric("F1 Score", f"{metrics['f1_score']:.2f}")


def is_overview_question(question: str) -> bool:
    """Detect broad questions that need whole-document context."""

    terms = [
        "what is this document about",
        "what is the document about",
        "summarize",
        "summary",
        "main idea",
        "overview",
        "topic",
    ]
    normalized = question.lower().strip()
    return any(term in normalized for term in terms)


def process_document(uploaded_file, cluster_count: int, summary_sentences: int) -> dict[str, object]:
    """Run the full classical NLP pipeline and save artifacts."""

    nlp = get_nlp()
    save_uploaded_pdf(uploaded_file, DATA_DIR)
    pdf_doc = load_pdf(uploaded_file, uploaded_file.name)

    normalized_text = clean_text(pdf_doc.text)
    cleaned_text = preprocess_text(normalized_text, nlp)
    save_cleaned_text(cleaned_text, OUTPUT_DIR / "cleaned_text.txt")

    sentences = sentence_tokenize(normalized_text, nlp)
    words = word_tokenize(normalized_text, nlp)
    chunks = chunk_text(normalized_text)

    pos_rows = extract_pos_tags(normalized_text, nlp)
    save_pos_tags(pos_rows, OUTPUT_DIR / "pos_tags.json")

    entities = extract_entities(normalized_text, nlp)
    save_entities(entities, OUTPUT_DIR / "entities.json")

    tfidf_docs = chunks if chunks else [cleaned_text]
    tfidf_vectorizer, tfidf_matrix, tfidf_features = build_tfidf(tfidf_docs)
    save_tfidf(tfidf_vectorizer, tfidf_matrix, OUTPUT_DIR / "tfidf_features.pkl")

    classifier = get_classifier()
    predicted_category, confidence = predict_document_category(cleaned_text, classifier)

    cluster_rows, cluster_distribution = cluster_documents(chunks, cluster_count)
    save_clusters(cluster_rows, OUTPUT_DIR / "clusters.csv")

    summary = textrank_summary(normalized_text, nlp, summary_sentences)
    save_summary(summary, OUTPUT_DIR / "summary.txt")

    evaluation = evaluate_classifier()
    save_evaluation(evaluation, OUTPUT_DIR / "evaluation.json")

    retriever = FaissRetriever()
    if chunks:
        retriever.create_index(chunks)

    return {
        "file_name": pdf_doc.file_name,
        "page_count": pdf_doc.page_count,
        "raw_text": normalized_text,
        "cleaned_text": cleaned_text,
        "sentences": sentences,
        "words": words,
        "chunks": chunks,
        "pos_rows": pos_rows,
        "entities": entities,
        "tfidf_features": tfidf_features,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "cluster_rows": cluster_rows,
        "cluster_distribution": cluster_distribution,
        "summary": summary,
        "evaluation": evaluation,
        "retriever": retriever,
    }


def main() -> None:
    ensure_directories()
    st.set_page_config(page_title="RAG Document Analyzer", page_icon="RAG", layout="wide")

    st.title("RAG Document Analyzer")
    st.caption("Classical NLP analysis first.")

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        cluster_count = st.slider("KMeans clusters", min_value=1, max_value=8, value=3)
        summary_sentences = st.slider("Summary sentences", min_value=2, max_value=10, value=5)
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=3)
        analyze = st.button("Analyze Document", type="primary", use_container_width=True)

    if analyze and uploaded_file is not None:
        with st.spinner("Running classical NLP pipeline..."):
            try:
                st.session_state["analysis"] = process_document(
                    uploaded_file, cluster_count, summary_sentences
                )
                st.success("Document analysis complete. Outputs were saved automatically.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Upload a PDF and click Analyze Document to begin.")
        return

    st.subheader("Document Statistics")
    stat_cols = st.columns(5)
    stat_cols[0].metric("File", analysis["file_name"])
    stat_cols[1].metric("Pages", analysis["page_count"])
    stat_cols[2].metric("Sentences", len(analysis["sentences"]))
    stat_cols[3].metric("Words", len(analysis["words"]))
    stat_cols[4].metric("Chunks", len(analysis["chunks"]))

    tab_labels = [
        "Cleaned Text",
        "POS Tagging",
        "Named Entities",
        "TF-IDF",
        "Classification",
        "Clustering",
        "Summary",
        "RAG QA",
        "Evaluation",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.text_area("Cleaned Text", analysis["cleaned_text"], height=320)

    with tabs[1]:
        pos_df = pos_to_dataframe(analysis["pos_rows"])
        st.dataframe(pos_df, use_container_width=True, height=420)

    with tabs[2]:
        entities_df = entity_dataframe(analysis["entities"])
        st.dataframe(entities_df, use_container_width=True, height=320)
        distribution = entity_distribution(analysis["entities"])
        if not distribution.empty:
            st.bar_chart(distribution.set_index("entity_type"))

    with tabs[3]:
        features = analysis["tfidf_features"].head(30)
        st.dataframe(features, use_container_width=True)
        if not features.empty:
            st.bar_chart(features.set_index("feature"))

    with tabs[4]:
        st.metric("Predicted Category", analysis["predicted_category"])
        confidence_df = pd.DataFrame(
            {
                "category": list(analysis["confidence"].keys()),
                "confidence": list(analysis["confidence"].values()),
            }
        ).sort_values("confidence", ascending=False)
        st.dataframe(confidence_df, use_container_width=True)
        st.bar_chart(confidence_df.set_index("category"))

    with tabs[5]:
        st.dataframe(analysis["cluster_rows"], use_container_width=True, height=320)
        if not analysis["cluster_distribution"].empty:
            st.bar_chart(analysis["cluster_distribution"].set_index("cluster"))

    with tabs[6]:
        st.write(analysis["summary"])

    with tabs[7]:
        question = st.text_input("Ask a question about the uploaded document")
        ask = st.button("Generate Answer", type="primary")
        if ask and question:
            with st.spinner("Retrieving chunks and generating final answer..."):
                try:
                    retrieval_query = analysis["summary"] if is_overview_question(question) else question
                    retrieved = analysis["retriever"].retrieve(retrieval_query, top_k=top_k)
                    save_retrieved_chunks(retrieved, OUTPUT_DIR / "retrieved_chunks.json")
                    answer = get_rag_answerer().answer(question, retrieved, analysis["summary"])
                    save_final_answer(question, answer, retrieved, OUTPUT_DIR / "final_answer.json")

                    st.session_state["last_answer"] = answer
                    st.session_state["last_chunks"] = retrieved
                except Exception as exc:
                    st.error(f"RAG question answering failed: {exc}")

        if st.session_state.get("last_answer"):
            st.markdown("#### Answer")
            st.write(st.session_state["last_answer"])
            st.markdown("#### Source Chunks")
            st.dataframe(pd.DataFrame(st.session_state["last_chunks"]), use_container_width=True)

    with tabs[8]:
        render_metrics(analysis["evaluation"])
        report = pd.DataFrame(analysis["evaluation"]["classification_report"]).transpose()
        st.dataframe(report, use_container_width=True)


if __name__ == "__main__":
    main()
