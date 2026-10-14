import re
from pathlib import Path
from typing import Any, Callable, Optional

import fitz

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ENABLE_CHART_ANALYSIS,
    ENABLE_TABLE_EXTRACTION,
    MAX_CHARTS_PER_DOCUMENT,
    MIN_IMAGE_PX,
    TABLE_MAX_ROWS,
)
from .multimodal_extractor import extract_elements, rows_to_markdown


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def _check_pdf(pdf_path: str) -> Path:
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be processed.")
    return path


def extract_pages(pdf_path: str) -> list[dict[str, Any]]:
    """Text-only extraction (kept as before; existing tests keep working)."""
    path = _check_pdf(pdf_path)
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


def extract_page_elements(
    pdf_path: str,
    describe_fn: Optional[Callable[[bytes], Optional[str]]] = None,
) -> list[dict[str, Any]]:
    """Text + tables + charts, page by page. Each item has page, chunk_type, method."""
    path = _check_pdf(pdf_path)
    try:
        elements = extract_elements(
            str(path),
            describe_fn,
            min_image_px=MIN_IMAGE_PX,
            max_charts=MAX_CHARTS_PER_DOCUMENT,
        )
    except (fitz.FileDataError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc
    if not ENABLE_TABLE_EXTRACTION:
        elements = [e for e in elements if e["chunk_type"] != "table"]
    for e in elements:
        e["source"] = path.name
        if "text" in e:
            e["text"] = clean_text(e["text"])
    return [e for e in elements if e.get("rows") or e.get("text")]


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


def chunk_table(rows: list[list], page: int, max_rows: int = TABLE_MAX_ROWS) -> list[str]:
    """Whole table = 1 chunk; long tables are split by rows with the header repeated."""
    header, body = rows[0], rows[1:]
    pieces = [body[i:i + max_rows] for i in range(0, len(body), max_rows)] or [[]]
    return [f"Table (page {page}):\n" + rows_to_markdown([header] + piece) for piece in pieces]


def _default_describe_fn() -> Optional[Callable[[bytes], Optional[str]]]:
    if not ENABLE_CHART_ANALYSIS:
        return None
    from .vision import describe_image   # imported lazily: needs google-genai + API key
    return describe_image


def process_pdf(
    pdf_path: str,
    company: str,
    document: str | None = None,
    describe_fn: Optional[Callable[[bytes], Optional[str]]] = None,
):
    if not company or not company.strip():
        raise ValueError("Company is required.")
    if describe_fn is None:
        describe_fn = _default_describe_fn()
    elements = extract_page_elements(pdf_path, describe_fn)

    chunks = []
    label = (document or Path(pdf_path).stem).strip()
    company = company.strip()
    safe_company = re.sub(r"[^a-z0-9]+", "_", company.lower()).strip("_") or "company"
    safe_document = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "document"
    counter: dict[int, int] = {}          # running chunk number per page (text + table + chart)

    for el in elements:
        page = el["page"]
        if el["chunk_type"] == "text":
            parts = chunk_text(el["text"])
        elif el["chunk_type"] == "table":
            parts = chunk_table(el["rows"], page)
        else:                              # chart: one description = one chunk
            parts = [f"Chart (page {page}): {el['text']}"]

        for part in parts:
            counter[page] = counter.get(page, 0) + 1
            idx = counter[page]
            chunks.append({
                "chunk_id": f"{safe_company}_{safe_document}_{page:04d}_{idx:02d}",
                "text": part,
                "source": el["source"],
                "company": company,
                "document": label,
                "page": page,
                "chunk": idx,
                "chunk_type": el["chunk_type"],     # NEW: text | table | chart
                "extraction": el["method"],         # NEW: text_layer | ocr | table_extraction | vision_generated
            })

    return chunks
