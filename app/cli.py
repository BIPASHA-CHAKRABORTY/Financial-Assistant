
import argparse
from .rag import RAGEngine
from .llm import generate_answer


def main():
    parser = argparse.ArgumentParser(
        description="RAG + LLM Financial Assistant"
    )
    parser.add_argument("--pdf", required=True, help="Path to annual report PDF")
    parser.add_argument(
        "--question",
        required=True,
        help="Financial question to ask",
    )
    args = parser.parse_args()

    rag = RAGEngine()
    rag.build_index([args.pdf])

    results = rag.search(args.question)
    answer, _ = generate_answer(args.question, results)

    print("\n===== FINANCIAL COPILOT ANSWER =====\n")
    print(answer)

    print("\n===== RETRIEVED EVIDENCE =====\n")
    for item in results:
        print(
            f"- {item['source']} | Page {item['page']} | "
            f"Similarity {item['score']:.3f}"
        )


if __name__ == "__main__":
    main()
