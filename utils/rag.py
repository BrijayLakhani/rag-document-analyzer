"""RAG question answering. The LLM is used only in this final stage."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import pipeline


class RAGQuestionAnswerer:
    """Generate answers from retrieved source chunks."""

    def __init__(self, model_name: str = "google/flan-t5-small") -> None:
        self.generator = pipeline("text2text-generation", model=model_name)

    @staticmethod
    def _is_overview_question(question: str) -> bool:
        overview_terms = [
            "what is this document about",
            "what is the document about",
            "summarize",
            "summary",
            "main idea",
            "overview",
            "topic",
        ]
        normalized = question.lower().strip()
        return any(term in normalized for term in overview_terms)

    @staticmethod
    def build_prompt(
        question: str,
        retrieved_chunks: list[dict[str, object]],
        document_summary: str = "",
    ) -> str:
        """Build a grounded prompt from retrieved context."""

        summary_context = ""
        if document_summary:
            summary_context = f"Document summary:\n{document_summary}\n\n"

        context = "\n\n".join(
            f"Source {chunk['rank']}:\n{chunk['text']}" for chunk in retrieved_chunks
        )
        return (
            "You are answering questions about one uploaded PDF document. "
            "Use only the document summary and source chunks below. "
            "Give a clear, specific answer in 2 to 4 sentences. "
            "If the answer is not supported, say the document does not provide enough information.\n\n"
            f"Question: {question}\n\n"
            f"{summary_context}"
            f"Sources:\n{context}\n\n"
            "Answer:"
        )

    def answer(
        self,
        question: str,
        retrieved_chunks: list[dict[str, object]],
        document_summary: str = "",
    ) -> str:
        """Generate the final RAG answer."""

        if not retrieved_chunks:
            return "No relevant source chunks were retrieved."

        summary = document_summary if self._is_overview_question(question) else ""
        prompt = self.build_prompt(question, retrieved_chunks, summary)
        response = self.generator(
            prompt,
            max_new_tokens=220,
            do_sample=False,
            truncation=True,
        )
        return str(response[0]["generated_text"]).strip()


def save_final_answer(
    question: str,
    answer: str,
    source_chunks: list[dict[str, object]],
    output_path: str | Path,
) -> None:
    """Save question, answer, and source chunks."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"question": question, "answer": answer, "source_chunks": source_chunks},
            indent=2,
        ),
        encoding="utf-8",
    )
