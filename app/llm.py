"""Gemini generation is intentionally separate from retrieval."""
from google import genai

from .config import GEMINI_API_KEY, GEMINI_MODEL

INSUFFICIENT_EVIDENCE = "Insufficient evidence was found in the indexed financial documents to answer this question."
SYSTEM_INSTRUCTION = """You are a financial-document assistant for a decision-support system.
Answer only from supplied evidence. Do not add outside knowledge or make investment recommendations.
If evidence is insufficient, say so plainly. Cite [Document, p. N] for important facts."""


def generate_answer(question: str, retrieved_chunks: list[dict], module_context: str | None = None) -> tuple[str, str]:
    if not retrieved_chunks:
        return INSUFFICIENT_EVIDENCE, "insufficient_evidence"
    if not GEMINI_API_KEY:
        sources = "; ".join(f"{x['document']} p. {x['page']}" for x in retrieved_chunks)
        return f"Gemini is not configured, so no generated interpretation is available. Relevant document evidence was retrieved: {sources}.", "retrieval_only"
    evidence = "\n\n".join(
        f"[Evidence {i}: {item['company']} | {item['document']} | p. {item['page']}]\n{item['text']}"
        for i, item in enumerate(retrieved_chunks, 1)
    )
    extra = f"\nStructured module context (not document evidence):\n{module_context}" if module_context else ""
    prompt = f"{SYSTEM_INSTRUCTION}\n\nQuestion: {question}\n\nEvidence:\n{evidence}{extra}\n\nGive a concise grounded answer and Sources section."
    response = genai.Client(api_key=GEMINI_API_KEY).models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return (response.text or INSUFFICIENT_EVIDENCE), "gemini_grounded"
