"""Record retrieval evidence for human evaluation; it never invents quality scores."""
import argparse
import json
from pathlib import Path

from app.rag import RAGEngine


parser = argparse.ArgumentParser(description="Evaluate retrieval against a JSONL question set")
parser.add_argument("questions", help="JSONL: question, expected_pages, optional company")
parser.add_argument("--output", default="data/processed/retrieval_evaluation.jsonl")
args = parser.parse_args()

engine = RAGEngine()
records = []
for line in Path(args.questions).read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    hits = engine.search(item["question"], company=item.get("company"))
    records.append({
        "question": item["question"],
        "expected_pages": item.get("expected_pages", []),
        "retrieved_pages": [hit["page"] for hit in hits],
        "retrieval_scores": [hit["score"] for hit in hits],
        "retrieved_sources": [{"document": hit["document"], "page": hit["page"]} for hit in hits],
        "generated_answer": None,
        "groundedness": None,
        "note": "Answer quality and groundedness require human review.",
    })

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
print(f"Wrote {len(records)} evaluation records to {output}")
