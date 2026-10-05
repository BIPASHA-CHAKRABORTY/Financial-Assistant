"""Convenience command for ingesting a local report without HTTP."""
import argparse

from app.rag import RAGEngine


parser = argparse.ArgumentParser(description="Ingest a financial PDF into the FAISS store")
parser.add_argument("pdf")
parser.add_argument("--company", required=True)
parser.add_argument("--document", required=True)
args = parser.parse_args()

count = RAGEngine().ingest(args.pdf, args.company, args.document)
print(f"Indexed {count} chunks.")
