import os, subprocess, sys
import pytest

fitz = pytest.importorskip("fitz")
from app.multimodal_extractor import extract_elements, chunk_elements

PDF = "data/documents/Sample_Report_Charts_Tables.pdf"


@pytest.fixture(scope="module")
def pdf_path():
    if not os.path.exists(PDF):
        subprocess.run([sys.executable, "scripts/make_sample_report.py"], check=True)
    return PDF


def fake_vision(_png):                      # no Gemini key needed in tests
    return "Bar chart: revenue rises from 100 to 170 (approx.)."


def test_text_table_chart_all_found(pdf_path):
    types = {e["chunk_type"] for e in extract_elements(pdf_path, fake_vision)}
    assert {"text", "table", "chart"} <= types


def test_chunks_keep_page_and_type(pdf_path):
    chunks = chunk_elements(extract_elements(pdf_path, fake_vision))
    assert chunks and all({"page", "chunk_type", "chunk_id", "text"} <= c.keys() for c in chunks)
    assert any("| Year | Revenue | Debt |" in c["text"] for c in chunks if c["chunk_type"] == "table")


def test_without_vision_charts_skipped(pdf_path):
    assert not [e for e in extract_elements(pdf_path, None) if e["chunk_type"] == "chart"]
