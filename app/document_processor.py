
import re
from pathlib import Path
from typing import Any

import fitz

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def extract_pages(pdf_path: str) -> list[dict[str, Any]]:
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be processed.")
    try:
        doc = fitz.open(path)
    except (fitz.FileDataError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc
    try:
        return [
            {"page": number, "text": text, "source": path.name}
            for number, page in enumerate(doc, start=1)
            if (text := clean_text(page.get_text("text")))
        ]
    finally:
        doc.close()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size.")
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = max(0, end - overlap)

    return chunks


def process_pdf(pdf_path: str, company: str, document: str | None = None):
    if not company or not company.strip():
        raise ValueError("Company is required.")
    pages = extract_pages(pdf_path)
    chunks = []
    label = (document or Path(pdf_path).stem).strip()
    company = company.strip()
    safe_company = re.sub(r"[^a-z0-9]+", "_", company.lower()).strip("_") or "company"
    safe_document = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "document"

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for idx, chunk in enumerate(page_chunks, start=1):
            chunks.append({
                "chunk_id": f"{safe_company}_{safe_document}_{page['page']:04d}_{idx:02d}",
                "text": chunk,
                "source": page["source"],
                "company": company,
                "document": label,
                "page": page["page"],
                "chunk": idx,
            })

    return chunks
