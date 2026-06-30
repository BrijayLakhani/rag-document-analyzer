"""Dash dashboard for RAG Document Analyzer with Classical NLP."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from dash import Input, Output, State, callback, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
from wordcloud import WordCloud

from utils.classifier import load_or_train_classifier, predict_document_category
from utils.clustering import cluster_documents, save_clusters
from utils.evaluation import evaluate_classifier, save_evaluation
from utils.keywords import extract_keywords, save_keywords
from utils.ner import entity_distribution, extract_entities, save_entities
from utils.pdf_loader import load_pdf
from utils.pos_tagger import extract_pos_tags, save_pos_tags
from utils.preprocessing import clean_text, load_spacy_model, preprocess_text, save_cleaned_text
from utils.rag import RAGQuestionAnswerer, save_final_answer
from utils.retriever import FaissRetriever, chunk_text, save_retrieved_chunks
from utils.summarizer import save_summary, textrank_summary
from utils.tokenizer import sentence_tokenize, word_tokenize
from utils.vectorizer import build_tfidf, save_tfidf

# ── Directories ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

for d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Global state (single-user local app) ─────────────────────────────────────

nlp = load_spacy_model()
classifier = load_or_train_classifier(MODEL_DIR / "classifier.pkl")
rag_answerer = None
retriever_instance = None
current_summary = ""

PLOTLY_TEMPLATE = "plotly_dark"
CARD_STYLE = {"backgroundColor": "#1e1e2f", "border": "1px solid #2d2d44"}

# ── App ──────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="RAG Document Analyzer",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

sidebar = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4(
                    [html.I(className="fas fa-file-pdf me-2"), "Controls"],
                    className="text-info mb-4",
                ),
                html.Label("Upload PDF", className="fw-bold mb-1"),
                dcc.Upload(
                    id="upload-pdf",
                    children=dbc.Card(
                        dbc.CardBody(
                            [
                                html.I(className="fas fa-cloud-upload-alt fa-2x text-muted mb-2"),
                                html.Div("Drag & Drop or Click", className="text-muted"),
                            ],
                            className="text-center py-3",
                        ),
                        style={"border": "2px dashed #444", "cursor": "pointer", "backgroundColor": "#161625"},
                    ),
                    multiple=False,
                ),
                html.Div(id="upload-filename", className="text-success small mt-1 mb-3"),
                html.Hr(style={"borderColor": "#333"}),
                html.Label("KMeans Clusters", className="fw-bold mb-1"),
                dcc.Slider(id="cluster-slider", min=1, max=8, value=3, step=1,
                           marks={i: str(i) for i in range(1, 9)},
                           className="mb-3"),
                html.Label("Summary Sentences", className="fw-bold mb-1"),
                dcc.Slider(id="summary-slider", min=2, max=10, value=5, step=1,
                           marks={i: str(i) for i in range(2, 11)},
                           className="mb-3"),
                html.Label("Retrieved Chunks (top-k)", className="fw-bold mb-1"),
                dcc.Slider(id="topk-slider", min=1, max=8, value=3, step=1,
                           marks={i: str(i) for i in range(1, 9)},
                           className="mb-3"),
                html.Label("Chunk Size (chars)", className="fw-bold mb-1"),
                dcc.Slider(id="chunk-size-slider", min=400, max=2000, value=900, step=100,
                           marks={v: str(v) for v in range(400, 2100, 400)},
                           className="mb-4"),
                dbc.Button(
                    [html.I(className="fas fa-play me-2"), "Analyze Document"],
                    id="analyze-btn",
                    color="info",
                    size="lg",
                    className="w-100 fw-bold",
                ),
                dbc.Progress(id="progress-bar", value=0, striped=True, animated=True,
                             className="mt-3", style={"display": "none"}),
                html.Div(id="status-msg", className="mt-2 small"),
            ]
        )
    ],
    style={**CARD_STYLE, "minHeight": "100vh"},
)

# ── Stat Cards ───────────────────────────────────────────────────────────────


def stat_card(icon: str, label: str, value_id: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.I(className=f"fas {icon} fa-lg text-info me-2"),
                            html.Span(label, className="text-muted small"),
                        ]
                    ),
                    html.H3(id=value_id, className="mb-0 mt-1 text-white"),
                ]
            ),
            style=CARD_STYLE,
            className="h-100",
        ),
        md=True,
    )


stats_row = dbc.Row(
    [
        stat_card("fa-file", "File", "stat-file"),
        stat_card("fa-copy", "Pages", "stat-pages"),
        stat_card("fa-align-left", "Sentences", "stat-sentences"),
        stat_card("fa-font", "Words", "stat-words"),
        stat_card("fa-puzzle-piece", "Chunks", "stat-chunks"),
    ],
    className="g-3 mb-4",
)

# ── Tab content placeholders ─────────────────────────────────────────────────

tabs = dbc.Tabs(
    id="main-tabs",
    active_tab="tab-cleaned",
    className="mb-3",
    children=[
        dbc.Tab(label="Cleaned Text", tab_id="tab-cleaned",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
        dbc.Tab(label="POS Tagging", tab_id="tab-pos",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
        dbc.Tab(label="Named Entities", tab_id="tab-ner",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
        dbc.Tab(label="TF-IDF & Word Cloud", tab_id="tab-tfidf",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
        dbc.Tab(label="Keywords", tab_id="tab-keywords",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
        dbc.Tab(label="Classification", tab_id="tab-class",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
        dbc.Tab(label="Clustering", tab_id="tab-cluster",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
        dbc.Tab(label="Summary", tab_id="tab-summary",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
        dbc.Tab(label="RAG Q&A", tab_id="tab-rag",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
        dbc.Tab(label="Evaluation", tab_id="tab-eval",
                label_style={"color": "#888"}, active_label_style={"color": "#17a2b8"}),
    ],
)

tab_content = html.Div(id="tab-content")

# ── Main layout ──────────────────────────────────────────────────────────────

app.layout = dbc.Container(
    [
        dcc.Store(id="chat-store", data=[]),
        dcc.Download(id="download-component"),
        # Header
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H2(
                            [html.I(className="fas fa-brain me-3 text-info"), "RAG Document Analyzer"],
                            className="mb-1 text-white",
                        ),
                        html.P(
                            "Classical NLP pipeline + RAG question answering",
                            className="text-muted mb-0",
                        ),
                    ],
                    className="py-3",
                )
            ),
            className="mb-3",
        ),
        # Body
        dbc.Row(
            [
                dbc.Col(sidebar, md=3, className="pe-0"),
                dbc.Col(
                    dcc.Loading(
                        [
                            dcc.Store(id="analysis-store"),
                            stats_row,
                            tabs,
                            tab_content,
                        ],
                        type="circle",
                        color="#17a2b8",
                        fullscreen=False,
                        style={"minHeight": "300px"},
                    ),
                    md=9,
                ),
            ]
        ),
    ],
    fluid=True,
    className="px-4",
    style={"backgroundColor": "#0f0f1a", "minHeight": "100vh"},
)

# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════


@callback(
    Output("upload-filename", "children"),
    Input("upload-pdf", "filename"),
)
def show_filename(filename):
    if filename:
        return [html.I(className="fas fa-check-circle me-1"), filename]
    return ""


@callback(
    Output("analysis-store", "data"),
    Output("status-msg", "children"),
    Output("progress-bar", "style"),
    Input("analyze-btn", "n_clicks"),
    State("upload-pdf", "contents"),
    State("upload-pdf", "filename"),
    State("cluster-slider", "value"),
    State("summary-slider", "value"),
    State("chunk-size-slider", "value"),
    prevent_initial_call=True,
)
def analyze_document(n_clicks, contents, filename, cluster_count, summary_sentences, chunk_size):
    global retriever_instance, current_summary, rag_answerer

    print(f"[analyze] Button clicked. filename={filename}, has_contents={bool(contents)}")

    if not contents or not filename:
        return no_update, dbc.Alert("Please upload a PDF first.", color="warning"), {"display": "none"}

    try:
        content_string = contents.split(",", 1)[1]
        decoded = base64.b64decode(content_string)
        pdf_bytes = io.BytesIO(decoded)
        print(f"[analyze] Decoded PDF: {len(decoded)} bytes")

        print("[analyze] Loading PDF...")
        pdf_doc = load_pdf(pdf_bytes, filename)
        print(f"[analyze] PDF loaded: {pdf_doc.page_count} pages")

        normalized_text = clean_text(pdf_doc.text)
        cleaned_text = preprocess_text(normalized_text, nlp)
        save_cleaned_text(cleaned_text, OUTPUT_DIR / "cleaned_text.txt")
        print("[analyze] Text cleaned")

        sentences = sentence_tokenize(normalized_text, nlp)
        words = word_tokenize(normalized_text, nlp)
        chunks = chunk_text(normalized_text, chunk_size=chunk_size)
        print(f"[analyze] Tokenized: {len(sentences)} sents, {len(chunks)} chunks")

        pos_rows = extract_pos_tags(normalized_text, nlp)
        save_pos_tags(pos_rows, OUTPUT_DIR / "pos_tags.json")
        print("[analyze] POS done")

        entities = extract_entities(normalized_text, nlp)
        save_entities(entities, OUTPUT_DIR / "entities.json")
        print("[analyze] NER done")

        tfidf_docs = chunks if chunks else [cleaned_text]
        tfidf_vectorizer, tfidf_matrix, tfidf_features = build_tfidf(tfidf_docs)
        save_tfidf(tfidf_vectorizer, tfidf_matrix, OUTPUT_DIR / "tfidf_features.pkl")
        print("[analyze] TF-IDF done")

        keywords = extract_keywords(normalized_text)
        save_keywords(keywords, OUTPUT_DIR / "keywords.json")
        print("[analyze] Keywords done")

        predicted_category, confidence = predict_document_category(cleaned_text, classifier)
        print(f"[analyze] Classification: {predicted_category}")

        cluster_rows, cluster_distribution = cluster_documents(chunks, cluster_count)
        save_clusters(cluster_rows, OUTPUT_DIR / "clusters.csv")
        print("[analyze] Clustering done")

        summary = textrank_summary(normalized_text, nlp, summary_sentences)
        save_summary(summary, OUTPUT_DIR / "summary.txt")
        current_summary = summary
        print("[analyze] Summary done")

        evaluation = evaluate_classifier()
        save_evaluation(evaluation, OUTPUT_DIR / "evaluation.json")
        print("[analyze] Evaluation done")

        retriever_instance = FaissRetriever()
        if chunks:
            retriever_instance.create_index(chunks)
        print("[analyze] FAISS index built")

        if rag_answerer is None:
            rag_answerer = RAGQuestionAnswerer()
        print("[analyze] RAG answerer ready")

        data = {
            "file_name": pdf_doc.file_name,
            "page_count": pdf_doc.page_count,
            "raw_text": normalized_text[:50000],
            "cleaned_text": cleaned_text[:50000],
            "sentences_count": len(sentences),
            "words_count": len(words),
            "chunks_count": len(chunks),
            "chunks": chunks[:200],
            "pos_rows": pos_rows[:500],
            "entities": entities,
            "tfidf_features": tfidf_features.head(50).to_dict("records"),
            "keywords": keywords,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "cluster_rows": cluster_rows.to_dict("records"),
            "cluster_distribution": cluster_distribution.to_dict("records"),
            "summary": summary,
            "evaluation": evaluation,
        }

        print(f"[analyze] SUCCESS - returning data with {len(data)} keys")

        return (
            data,
            dbc.Alert(
                [html.I(className="fas fa-check-circle me-2"), "Analysis complete!"],
                color="success",
                duration=4000,
            ),
            {"display": "none"},
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return (
            no_update,
            dbc.Alert(f"Analysis failed: {exc}", color="danger"),
            {"display": "none"},
        )


# ── Stats ────────────────────────────────────────────────────────────────────

@callback(
    Output("stat-file", "children"),
    Output("stat-pages", "children"),
    Output("stat-sentences", "children"),
    Output("stat-words", "children"),
    Output("stat-chunks", "children"),
    Input("analysis-store", "data"),
)
def update_stats(data):
    if not data:
        return "-", "-", "-", "-", "-"
    return (
        data["file_name"],
        data["page_count"],
        data["sentences_count"],
        data["words_count"],
        data["chunks_count"],
    )


# ── Tab routing ──────────────────────────────────────────────────────────────

@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    Input("analysis-store", "data"),
)
def render_tab(active_tab, data):
    if not data:
        return dbc.Card(
            dbc.CardBody(
                html.Div(
                    [
                        html.I(className="fas fa-upload fa-3x text-muted mb-3"),
                        html.H5("Upload a PDF and click Analyze Document", className="text-muted"),
                    ],
                    className="text-center py-5",
                )
            ),
            style=CARD_STYLE,
        )

    if active_tab == "tab-cleaned":
        return _render_cleaned(data)
    elif active_tab == "tab-pos":
        return _render_pos(data)
    elif active_tab == "tab-ner":
        return _render_ner(data)
    elif active_tab == "tab-tfidf":
        return _render_tfidf(data)
    elif active_tab == "tab-keywords":
        return _render_keywords(data)
    elif active_tab == "tab-class":
        return _render_classification(data)
    elif active_tab == "tab-cluster":
        return _render_clustering(data)
    elif active_tab == "tab-summary":
        return _render_summary(data)
    elif active_tab == "tab-rag":
        return _render_rag(data)
    elif active_tab == "tab-eval":
        return _render_evaluation(data)
    return html.Div()


# ── Tab renderers ────────────────────────────────────────────────────────────

def _render_cleaned(data):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5([html.I(className="fas fa-broom me-2 text-info"), "Cleaned Text"],
                                className="d-inline"),
                        dbc.Button(
                            [html.I(className="fas fa-download me-1"), "Download"],
                            id="btn-dl-cleaned", size="sm", color="outline-info",
                            className="float-end",
                        ),
                    ],
                    className="mb-3",
                ),
                dcc.Textarea(
                    value=data["cleaned_text"],
                    style={
                        "width": "100%", "height": "450px",
                        "backgroundColor": "#161625", "color": "#ccc",
                        "border": "1px solid #333", "borderRadius": "8px",
                        "padding": "12px", "fontFamily": "monospace", "fontSize": "13px",
                    },
                    readOnly=True,
                ),
            ]
        ),
        style=CARD_STYLE,
    )


def _render_pos(data):
    df = pd.DataFrame(data["pos_rows"])
    if df.empty:
        return _empty_card("No POS tags extracted.")

    pos_counts = df["pos"].value_counts().reset_index()
    pos_counts.columns = ["POS", "Count"]

    fig = px.bar(
        pos_counts, x="POS", y="Count", color="POS",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", showlegend=False, margin=dict(t=30, b=30),
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="fas fa-tags me-2 text-info"), "Part-of-Speech Tagging"],
                        className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig, config={"displayModeBar": False}), md=5),
                        dbc.Col(
                            _data_table(df, max_rows=200),
                            md=7,
                        ),
                    ]
                ),
            ]
        ),
        style=CARD_STYLE,
    )


def _render_ner(data):
    entities = data["entities"]
    if not entities:
        return _empty_card("No named entities found.")

    df = pd.DataFrame(entities)
    dist = pd.DataFrame(entity_distribution(entities))

    fig_bar = px.bar(
        dist, x="entity_type", y="count", color="entity_type",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", showlegend=False, margin=dict(t=30, b=30),
    )

    fig_pie = px.pie(
        dist, names="entity_type", values="count",
        template=PLOTLY_TEMPLATE, hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_pie.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", margin=dict(t=30, b=30),
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="fas fa-map-marker-alt me-2 text-info"), "Named Entity Recognition"],
                        className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig_bar, config={"displayModeBar": False}), md=6),
                        dbc.Col(dcc.Graph(figure=fig_pie, config={"displayModeBar": False}), md=6),
                    ],
                    className="mb-3",
                ),
                _data_table(df[["text", "label"]], max_rows=200),
            ]
        ),
        style=CARD_STYLE,
    )


def _render_tfidf(data):
    features = data["tfidf_features"]
    if not features:
        return _empty_card("No TF-IDF features extracted.")

    df = pd.DataFrame(features)

    fig = px.bar(
        df.head(25), x="score", y="feature", orientation="h",
        template=PLOTLY_TEMPLATE,
        color="score", color_continuous_scale="Tealgrn",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", yaxis={"categoryorder": "total ascending"},
        margin=dict(t=30, b=30), coloraxis_showscale=False,
    )

    freq = {row["feature"]: row["score"] for row in features}
    wc = WordCloud(
        width=800, height=350, background_color="#1e1e2f",
        colormap="cool", max_words=80, prefer_horizontal=0.7,
    ).generate_from_frequencies(freq)

    img_bytes = io.BytesIO()
    wc.to_image().save(img_bytes, format="PNG")
    img_bytes.seek(0)
    wc_base64 = base64.b64encode(img_bytes.read()).decode()

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="fas fa-chart-bar me-2 text-info"), "TF-IDF Features & Word Cloud"],
                        className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig, config={"displayModeBar": False}), md=6),
                        dbc.Col(
                            html.Img(
                                src=f"data:image/png;base64,{wc_base64}",
                                style={"width": "100%", "borderRadius": "8px"},
                            ),
                            md=6,
                        ),
                    ]
                ),
            ]
        ),
        style=CARD_STYLE,
    )


def _render_keywords(data):
    keywords = data.get("keywords", [])
    if not keywords:
        return _empty_card("No keywords extracted.")

    df = pd.DataFrame(keywords)

    fig = px.bar(
        df.head(15), x="score", y="keyword", orientation="h",
        template=PLOTLY_TEMPLATE,
        color="score", color_continuous_scale="Viridis",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", yaxis={"categoryorder": "total ascending"},
        margin=dict(t=30, b=30), coloraxis_showscale=False,
    )

    keyword_chips = html.Div(
        [
            dbc.Badge(
                f"{kw['keyword']}  ({kw['score']:.3f})",
                color="info", className="me-2 mb-2 px-3 py-2",
                style={"fontSize": "14px"},
            )
            for kw in keywords
        ],
        className="mt-3",
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="fas fa-key me-2 text-info"), "KeyBERT Keyword Extraction"],
                        className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig, config={"displayModeBar": False}), md=6),
                        dbc.Col(keyword_chips, md=6),
                    ]
                ),
            ]
        ),
        style=CARD_STYLE,
    )


def _render_classification(data):
    conf = data["confidence"]
    categories = list(conf.keys())
    scores = list(conf.values())

    fig_bar = px.bar(
        x=scores, y=categories, orientation="h",
        template=PLOTLY_TEMPLATE,
        color=scores, color_continuous_scale="Tealgrn",
        labels={"x": "Confidence", "y": "Category"},
    )
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", yaxis={"categoryorder": "total ascending"},
        margin=dict(t=30, b=30), coloraxis_showscale=False,
    )

    fig_pie = px.pie(
        names=categories, values=scores,
        template=PLOTLY_TEMPLATE, hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", margin=dict(t=30, b=30),
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="fas fa-layer-group me-2 text-info"), "Document Classification"],
                        className="mb-3"),
                dbc.Alert(
                    [
                        html.Strong("Predicted Category: "),
                        html.Span(data["predicted_category"], className="fs-5"),
                    ],
                    color="info",
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig_bar, config={"displayModeBar": False}), md=6),
                        dbc.Col(dcc.Graph(figure=fig_pie, config={"displayModeBar": False}), md=6),
                    ]
                ),
            ]
        ),
        style=CARD_STYLE,
    )


def _render_clustering(data):
    rows = data["cluster_rows"]
    if not rows:
        return _empty_card("No clusters generated.")

    df = pd.DataFrame(rows)
    df["cluster"] = df["cluster"].astype(str)

    fig_scatter = px.scatter(
        df, x="x", y="y", color="cluster",
        hover_data=["chunk_id", "text"],
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"x": "PCA Component 1", "y": "PCA Component 2"},
    )
    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", margin=dict(t=30, b=30),
    )
    fig_scatter.update_traces(marker=dict(size=10, opacity=0.8))

    dist = pd.DataFrame(data["cluster_distribution"])
    dist["cluster"] = dist["cluster"].astype(str)

    fig_bar = px.bar(
        dist, x="cluster", y="count", color="cluster",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", showlegend=False, margin=dict(t=30, b=30),
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="fas fa-project-diagram me-2 text-info"), "Document Clustering (KMeans + PCA)"],
                        className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}), md=8),
                        dbc.Col(dcc.Graph(figure=fig_bar, config={"displayModeBar": False}), md=4),
                    ]
                ),
            ]
        ),
        style=CARD_STYLE,
    )


def _render_summary(data):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5(
                            [html.I(className="fas fa-file-alt me-2 text-info"), "Extractive Summary (TextRank)"],
                            className="d-inline",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-download me-1"), "Download"],
                            id="btn-dl-summary", size="sm", color="outline-info",
                            className="float-end",
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Card(
                    dbc.CardBody(
                        html.P(data["summary"], className="mb-0", style={"lineHeight": "1.8", "fontSize": "15px"}),
                    ),
                    style={"backgroundColor": "#161625", "border": "1px solid #333"},
                ),
            ]
        ),
        style=CARD_STYLE,
    )


def _render_rag(data):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="fas fa-robot me-2 text-info"), "RAG Question Answering"],
                        className="mb-3"),
                html.Div(id="chat-display", style={
                    "maxHeight": "400px", "overflowY": "auto",
                    "backgroundColor": "#161625", "borderRadius": "8px",
                    "padding": "16px", "marginBottom": "16px",
                    "border": "1px solid #333",
                }),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id="rag-question", type="text",
                            placeholder="Ask a question about the document...",
                            style={"backgroundColor": "#1e1e2f", "color": "#ccc", "border": "1px solid #444"},
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-paper-plane me-1"), "Ask"],
                            id="rag-ask-btn", color="info",
                        ),
                    ]
                ),
            ]
        ),
        style=CARD_STYLE,
    )


def _render_evaluation(data):
    evaluation = data["evaluation"]

    metrics_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(name.upper(), className="text-muted small mb-1"),
                            html.H3(
                                f"{evaluation[key]:.2f}",
                                className="mb-0 text-info",
                            ),
                        ],
                        className="text-center",
                    ),
                    style=CARD_STYLE,
                ),
                md=3,
            )
            for name, key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                              ("Recall", "recall"), ("F1 Score", "f1_score")]
        ],
        className="g-3 mb-4",
    )

    cm = np.array(evaluation["confusion_matrix"])
    labels = evaluation["confusion_labels"]

    fig_cm = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Teal",
        annotation_font={"color": "white", "size": 14},
    )
    fig_cm.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        xaxis_title="Predicted", yaxis_title="Actual",
        margin=dict(t=40, b=40),
    )

    report = evaluation["classification_report"]
    report_rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict) and "precision" in metrics:
            report_rows.append({
                "Class": label,
                "Precision": f"{metrics['precision']:.2f}",
                "Recall": f"{metrics['recall']:.2f}",
                "F1": f"{metrics['f1-score']:.2f}",
                "Support": metrics["support"],
            })
    report_df = pd.DataFrame(report_rows)

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5([html.I(className="fas fa-chart-line me-2 text-info"), "Classifier Evaluation"],
                        className="mb-3"),
                metrics_cards,
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H6("Confusion Matrix", className="text-muted mb-2"),
                                dcc.Graph(figure=fig_cm, config={"displayModeBar": False}),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.H6("Classification Report", className="text-muted mb-2"),
                                _data_table(report_df),
                            ],
                            md=6,
                        ),
                    ]
                ),
            ]
        ),
        style=CARD_STYLE,
    )


# ── RAG Chat ─────────────────────────────────────────────────────────────────

@callback(
    Output("chat-store", "data"),
    Output("rag-question", "value"),
    Input("rag-ask-btn", "n_clicks"),
    State("rag-question", "value"),
    State("chat-store", "data"),
    State("analysis-store", "data"),
    State("topk-slider", "value"),
    prevent_initial_call=True,
)
def rag_answer(n_clicks, question, chat_history, analysis_data, top_k):
    if not question or not question.strip() or not analysis_data:
        raise PreventUpdate

    global retriever_instance, rag_answerer, current_summary

    if retriever_instance is None or rag_answerer is None:
        chat_history.append({"role": "user", "text": question})
        chat_history.append({"role": "bot", "text": "Retriever not ready. Please re-analyze the document."})
        return chat_history, ""

    try:
        overview_terms = ["summary", "summarize", "overview", "topic", "main idea", "about"]
        is_overview = any(t in question.lower() for t in overview_terms)
        retrieval_query = current_summary if is_overview else question

        retrieved = retriever_instance.retrieve(retrieval_query, top_k=top_k)
        save_retrieved_chunks(retrieved, OUTPUT_DIR / "retrieved_chunks.json")

        answer = rag_answerer.answer(question, retrieved, current_summary)
        save_final_answer(question, answer, retrieved, OUTPUT_DIR / "final_answer.json")

        chat_history.append({"role": "user", "text": question})
        chat_history.append({
            "role": "bot",
            "text": answer,
            "sources": [{"rank": c["rank"], "score": round(c["score"], 3)} for c in retrieved],
        })
    except Exception as exc:
        chat_history.append({"role": "user", "text": question})
        chat_history.append({"role": "bot", "text": f"Error: {exc}"})

    return chat_history, ""


@callback(
    Output("chat-display", "children"),
    Input("chat-store", "data"),
)
def render_chat(chat_history):
    if not chat_history:
        return html.Div(
            [
                html.I(className="fas fa-comments fa-2x text-muted mb-2"),
                html.P("Ask a question about the uploaded document", className="text-muted"),
            ],
            className="text-center py-4",
        )

    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(
                html.Div(
                    [
                        html.Div(
                            [html.I(className="fas fa-user me-2"), msg["text"]],
                            className="d-inline-block px-3 py-2",
                            style={
                                "backgroundColor": "#17a2b8", "borderRadius": "16px 16px 4px 16px",
                                "color": "white", "maxWidth": "80%",
                            },
                        ),
                    ],
                    className="text-end mb-2",
                )
            )
        else:
            source_badges = ""
            if msg.get("sources"):
                source_badges = html.Div(
                    [
                        dbc.Badge(f"Chunk #{s['rank']} ({s['score']})", color="dark", className="me-1 mt-1")
                        for s in msg["sources"]
                    ],
                    className="mt-2",
                )
            messages.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div([html.I(className="fas fa-robot me-2"), msg["text"]]),
                                source_badges,
                            ],
                            className="d-inline-block px-3 py-2",
                            style={
                                "backgroundColor": "#2d2d44", "borderRadius": "16px 16px 16px 4px",
                                "color": "#ccc", "maxWidth": "80%",
                            },
                        ),
                    ],
                    className="text-start mb-2",
                )
            )

    return messages


# ── Downloads ────────────────────────────────────────────────────────────────

@callback(
    Output("download-component", "data"),
    Input("btn-dl-cleaned", "n_clicks"),
    Input("btn-dl-summary", "n_clicks"),
    State("analysis-store", "data"),
    prevent_initial_call=True,
)
def download_file(dl_cleaned, dl_summary, data):
    if not data:
        raise PreventUpdate

    triggered = ctx.triggered_id
    if triggered == "btn-dl-cleaned":
        return dict(content=data["cleaned_text"], filename="cleaned_text.txt")
    elif triggered == "btn-dl-summary":
        return dict(content=data["summary"], filename="summary.txt")
    raise PreventUpdate


# ── Helpers ──────────────────────────────────────────────────────────────────

def _data_table(df: pd.DataFrame, max_rows: int = 300) -> html.Div:
    if df.empty:
        return html.P("No data.", className="text-muted")

    header = html.Thead(
        html.Tr([html.Th(col, style={"color": "#17a2b8", "borderBottom": "1px solid #333"}) for col in df.columns])
    )
    body = html.Tbody(
        [
            html.Tr(
                [html.Td(row[col], style={"color": "#ccc", "borderBottom": "1px solid #222"}) for col in df.columns]
            )
            for _, row in df.head(max_rows).iterrows()
        ]
    )
    return html.Div(
        dbc.Table(
            [header, body],
            bordered=False, hover=True, responsive=True,
            style={"backgroundColor": "transparent"},
            className="table-sm",
        ),
        style={"maxHeight": "400px", "overflowY": "auto"},
    )


def _empty_card(message: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            html.P(message, className="text-muted text-center py-5"),
        ),
        style=CARD_STYLE,
    )


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, port=8050)
