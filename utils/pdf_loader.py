"""PDF loading utilities for the RAG Document Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PyPDF2 import PdfReader


@dataclass
class PDFDocument:
    """Structured representation of an uploaded PDF."""

    file_name: str
    text: str
    page_count: int


def _extract_text(reader: PdfReader) -> str:
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            page_text = f"\n[Warning: could not extract text from page {page_number}: {exc}]\n"
        pages.append(page_text)
    return "\n\n".join(pages).strip()


def load_pdf(file: str | Path | BinaryIO, file_name: str | None = None) -> PDFDocument:
    """Load a PDF from disk or Streamlit upload and return extracted text.

    Args:
        file: Path-like object or a binary file object.
        file_name: Optional display name. Streamlit uploads provide this separately.

    Raises:
        ValueError: If the PDF has no extractable text.
    """

    try:
        reader = PdfReader(file)
    except Exception as exc:
        raise ValueError(f"Unable to read PDF file: {exc}") from exc

    text = _extract_text(reader)
    if not text:
        raise ValueError(
            "No extractable text was found. The PDF may be scanned; OCR is needed first."
        )

    resolved_name = file_name or getattr(file, "name", "uploaded_document.pdf")
    return PDFDocument(file_name=Path(str(resolved_name)).name, text=text, page_count=len(reader.pages))


def save_uploaded_pdf(uploaded_file: BinaryIO, destination_dir: str | Path) -> Path:
    """Persist an uploaded PDF to the data directory."""

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    file_name = Path(getattr(uploaded_file, "name", "uploaded_document.pdf")).name
    output_path = destination / file_name

    uploaded_file.seek(0)
    output_path.write_bytes(uploaded_file.read())
    uploaded_file.seek(0)
    return output_path
