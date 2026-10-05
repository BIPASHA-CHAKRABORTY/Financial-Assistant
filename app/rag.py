"""Embedding, persistent FAISS index and company-aware semantic retrieval."""
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD, TOP_K, VECTOR_STORE_PATH
from .document_processor import process_pdf


class RAGEngine:
    def __init__(self, model: Any = None, store_path: Path = VECTOR_STORE_PATH):
        self.model = model or SentenceTransformer(EMBEDDING_MODEL)
        self.store_path = Path(store_path)
        self.index = None
        self.documents: list[dict[str, Any]] = []

    @property
    def index_path(self): return self.store_path / "financial.index"

    @property
    def metadata_path(self): return self.store_path / "documents.json"

    def ingest(self, pdf_path: str, company: str, document: str | None = None) -> int:
        chunks = process_pdf(pdf_path, company, document)
        if not chunks:
            raise ValueError("No readable text was found in the supplied PDF.")
        self._ensure_loaded()
        vectors = np.asarray(self.model.encode([x["text"] for x in chunks], normalize_embeddings=True), dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        if vectors.shape[1] != self.index.d:
            raise ValueError("Embedding dimension does not match the existing vector index.")
        self.index.add(vectors)
        self.documents.extend(chunks)
        self.save()
        return len(chunks)

    def build_index(self, pdf_paths: list[str]) -> int:
        """Backward-compatible builder for the command-line demo."""
        self.index, self.documents = None, []
        total = 0
        for path in pdf_paths:
            total += self.ingest(path, Path(path).stem, Path(path).stem)
        return total

    def search(self, query: str, top_k: int = TOP_K, company: str | None = None,
               threshold: float = SIMILARITY_THRESHOLD) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("A non-empty query is required.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self._ensure_loaded()
        if self.index is None or not self.documents:
            raise ValueError("Vector index is empty. Upload a financial PDF first.")
        vector = np.asarray(self.model.encode([query], normalize_embeddings=True), dtype="float32")
        scores, indices = self.index.search(vector, len(self.documents))
        requested = company.casefold().strip() if company else None
        found = []
        # Searching every vector ensures filtering cannot hide a relevant company result.
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or float(score) < threshold:
                continue
            item = self.documents[int(index)]
            if requested and item["company"].casefold() != requested:
                continue
            result = dict(item)
            result["score"] = float(score)
            found.append(result)
            if len(found) == top_k:
                break
        return found

    def list_documents(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        grouped = {}
        for item in self.documents:
            key = (item["company"], item["document"], item["source"])
            current = grouped.setdefault(key, {"company": key[0], "document": key[1], "file": key[2], "chunks": 0, "pages": set()})
            current["chunks"] += 1
            current["pages"].add(item["page"])
        return [{**entry, "pages": sorted(entry["pages"])} for entry in grouped.values()]

    def save(self):
        self.store_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.documents, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self):
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.documents = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _ensure_loaded(self):
        if self.index is None:
            self.load()
