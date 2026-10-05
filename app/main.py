"""HTTP contract consumed by the central Node.js/Express service."""
from pathlib import Path
import shutil

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import DOCUMENT_STORAGE_PATH, SIMILARITY_THRESHOLD, TOP_K
from .llm import generate_answer
from .rag import RAGEngine

app = FastAPI(title="AI Financial Risk & Investment Copilot — RAG Service", version="2.0.0")
rag = RAGEngine()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    company: str | None = Field(default=None, max_length=200)
    companies: list[str] | None = Field(default=None, max_length=10)
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    similarity_threshold: float = Field(default=SIMILARITY_THRESHOLD, ge=-1, le=1)
    module_context: str | None = Field(default=None, max_length=10000)


class IngestRequest(BaseModel):
    filename: str = Field(min_length=1)
    company: str = Field(min_length=1, max_length=200)
    document: str = Field(min_length=1, max_length=300)


def service_error(exc: Exception) -> HTTPException:
    # Do not expose paths, credentials, or stack traces in public responses.
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="The document service could not complete the request.")


@app.get("/")
def root():
    return {"project": "AI Financial Risk & Investment Copilot", "module": "RAG + LLM Financial Assistant"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "financial-rag-assistant"}


@app.get("/documents")
def documents():
    return {"documents": rag.list_documents()}


@app.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...), company: str = Form(...), document: str = Form(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    filename = Path(file.filename).name  # discard an attempted client-side path
    destination = DOCUMENT_STORAGE_PATH / filename
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        count = rag.ingest(str(destination), company, document)
        return {"message": "Financial document indexed successfully.", "file": filename, "company": company, "document": document, "chunks_indexed": count}
    except Exception as exc:
        # A failed upload is retained for debugging/re-ingestion; it cannot be served publicly.
        raise service_error(exc)
    finally:
        await file.close()


@app.post("/documents/ingest")
def ingest_existing(request: IngestRequest):
    filename = Path(request.filename).name
    path = DOCUMENT_STORAGE_PATH / filename
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="Stored PDF was not found.")
    try:
        return {"message": "Financial document indexed successfully.", "chunks_indexed": rag.ingest(str(path), request.company, request.document)}
    except Exception as exc:
        raise service_error(exc)


@app.post("/query")
def query(request: QueryRequest):
    try:
        if request.company and request.companies:
            raise ValueError("Use either company or companies, not both.")
        if request.companies:
            # Retrieve independently so comparison evidence cannot be accidentally cross-attributed.
            retrieved = []
            for company in request.companies:
                retrieved.extend(rag.search(request.query, request.top_k, company, request.similarity_threshold))
        else:
            retrieved = rag.search(request.query, request.top_k, request.company, request.similarity_threshold)
        answer, mode = generate_answer(request.query, retrieved, request.module_context)
        sources = [{"company": x["company"], "document": x["document"], "page": x["page"], "chunk_id": x["chunk_id"], "retrieval_score": x["score"]} for x in retrieved]
        return {"query": request.query, "answer": answer, "sources": sources, "retrieved_context": retrieved, "grounded": bool(retrieved), "answer_mode": mode}
    except Exception as exc:
        raise service_error(exc)


# Compatibility aliases for the initial service API.
@app.post("/ask")
def ask(request: QueryRequest):
    return query(request)
