import fitz

from app.document_processor import chunk_text, process_pdf


def test_processing_preserves_page_company_and_chunk_id(tmp_path):
    pdf = tmp_path / "report.pdf"
    doc = fitz.open()
    first = doc.new_page()
    first.insert_text((72, 72), "Debt obligations and cash flow risks are disclosed.")
    second = doc.new_page()
    second.insert_text((72, 72), "Revenue increased during the reporting period.")
    doc.save(pdf)
    doc.close()

    chunks = process_pdf(str(pdf), "TCS", "TCS Annual Report 2025")

    assert [chunk["page"] for chunk in chunks] == [1, 2]
    assert all(chunk["company"] == "TCS" for chunk in chunks)
    assert chunks[0]["chunk_id"].startswith("tcs_tcs_annual_report_2025_0001")


def test_chunk_overlap_and_validation():
    chunks = chunk_text("one two three four five", chunk_size=3, overlap=1)
    assert chunks == ["one two three", "three four five"]
