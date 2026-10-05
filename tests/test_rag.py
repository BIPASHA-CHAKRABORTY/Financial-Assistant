import numpy as np

from app.rag import RAGEngine


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "debt" in text.lower() else [0.0, 1.0])
        return np.array(vectors, dtype="float32")


def test_company_filter_threshold_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr("app.rag.process_pdf", lambda *_: [
        {"chunk_id": "tcs_0001_01", "company": "TCS", "document": "TCS Report", "source": "tcs.pdf", "page": 1, "chunk": 1, "text": "debt increased"},
        {"chunk_id": "infy_0001_01", "company": "Infosys", "document": "Infosys Report", "source": "infy.pdf", "page": 1, "chunk": 1, "text": "revenue increased"},
    ])
    engine = RAGEngine(model=FakeEmbeddingModel(), store_path=tmp_path)
    engine.ingest("unused.pdf", "TCS", "TCS Report")
    assert engine.search("debt", company="TCS", threshold=0.9)[0]["company"] == "TCS"
    assert engine.search("debt", company="Infosys", threshold=0.9) == []
    restored = RAGEngine(model=FakeEmbeddingModel(), store_path=tmp_path)
    assert restored.search("debt", company="TCS", threshold=0.9)[0]["page"] == 1
