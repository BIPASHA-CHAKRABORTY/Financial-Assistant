"""Text + table + chart extraction, page by page, then chunking.

Order of work for every page:
  1. TABLES  -> Markdown (structure kept)
  2. CHARTS  -> image crop -> Gemini Vision -> written description
  3. TEXT    -> only the text OUTSIDE tables/charts (no duplicates)
  4. CHUNKING-> text: overlapping words | table: whole (split by rows) | chart: one chunk
"""
from __future__ import annotations

from typing import Callable, Optional

import fitz  # PyMuPDF

DescribeFn = Callable[[bytes], Optional[str]]


# ------------------------------------------------------------------ tables
def _rows_to_markdown(rows: list[list]) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    head, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def extract_tables(page: "fitz.Page") -> list[dict]:
    out = []
    try:
        found = page.find_tables().tables
    except Exception:
        return out
    for t in found:
        rows = t.extract()
        if len(rows) < 2:           # ignore 1-row "tables" (layout boxes)
            continue
        out.append({"rows": rows, "bbox": fitz.Rect(t.bbox)})
    return out


# ------------------------------------------------------------------ charts
def find_chart_regions(page: "fitz.Page", min_px: int = 150) -> list["fitz.Rect"]:
    rects: list[fitz.Rect] = []
    # (a) embedded raster images
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            if r.width >= min_px and r.height >= min_px:
                rects.append(fitz.Rect(r))
    # (b) vector charts (bars/lines drawn as shapes, not images)
    try:
        for r in page.cluster_drawings():
            if r.width >= min_px and r.height >= min_px:
                rects.append(fitz.Rect(r))
    except Exception:
        pass
    return rects


def render_region(page: "fitz.Page", rect: "fitz.Rect", dpi: int = 200) -> bytes:
    rect = fitz.Rect(rect) + (-10, -10, 10, 10)          # small margin (titles/legends)
    return page.get_pixmap(clip=rect & page.rect, dpi=dpi).tobytes("png")


# ----------------------------------------------------------------- one PDF
def extract_elements(
    pdf_path: str,
    describe_image: Optional[DescribeFn] = None,
    *,
    min_image_px: int = 150,
    max_charts: int = 30,
    ocr_fn: Optional[Callable[[bytes], str]] = None,
    ocr_min_chars: int = 50,
) -> list[dict]:
    """Return [{page, chunk_type, text, source}] in reading order per page."""
    elements: list[dict] = []
    charts_done = 0
    with fitz.open(pdf_path) as doc:
        for page in doc:
            n = page.number + 1
            skip: list[fitz.Rect] = []

            for t in extract_tables(page):                     # 1) tables
                skip.append(t["bbox"])
                elements.append({"page": n, "chunk_type": "table",
                                 "rows": t["rows"], "source": "table_extraction"})

            if describe_image:                                 # 2) charts
                for r in find_chart_regions(page, min_image_px):
                    if charts_done >= max_charts:
                        break
                    if any(r.intersects(s) and (r & s).get_area() > 0.5 * r.get_area() for s in skip):
                        continue                               # it is just a table
                    desc = describe_image(render_region(page, r))
                    if desc:
                        skip.append(r)
                        charts_done += 1
                        elements.append({"page": n, "chunk_type": "chart",
                                         "text": desc, "source": "vision_generated"})

            text = " ".join(                                   # 3) text outside tables/charts
                b[4].strip() for b in page.get_text("blocks")
                if b[6] == 0 and not any(fitz.Rect(b[:4]).intersects(s) for s in skip)
            )
            source = "text_layer"
            if len(text) < ocr_min_chars and ocr_fn:           # scanned page
                text, source = ocr_fn(page.get_pixmap(dpi=200).tobytes("png")), "ocr"
            if text.strip():
                elements.append({"page": n, "chunk_type": "text",
                                 "text": text, "source": source})
    return elements


# ---------------------------------------------------------------- chunking
def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


def chunk_elements(elements: list[dict], *, chunk_words: int = 300,
                   overlap_words: int = 150, table_rows: int = 25) -> list[dict]:
    chunks: list[dict] = []
    for el in elements:
        base = {"page": el["page"], "chunk_type": el["chunk_type"], "source": el["source"]}
        if el["chunk_type"] == "text":
            for part in _split_words(el["text"], chunk_words, overlap_words):
                chunks.append({**base, "text": part})
        elif el["chunk_type"] == "table":
            rows = el["rows"]
            head, body = rows[0], rows[1:]
            for i in range(0, len(body), table_rows):          # header repeated per piece
                md = _rows_to_markdown([head] + body[i:i + table_rows])
                chunks.append({**base, "text": "Table (page %d):\n%s" % (el["page"], md)})
        else:                                                  # chart: one chunk each
            chunks.append({**base, "text": "Chart (page %d): %s" % (el["page"], el["text"])})
    for i, c in enumerate(chunks):
        c["chunk_id"] = i
    return chunks
