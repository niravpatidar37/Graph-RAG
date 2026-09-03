from __future__ import annotations

import argparse
from pathlib import Path

from .core import GraphRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a small local Graph RAG index.")
    parser.add_argument("directory", type=Path, help="Directory containing .md or .txt documents")
    parser.add_argument("question", nargs="*", help="Question to answer")
    args = parser.parse_args()

    rag = GraphRAG()
    count = rag.add_directory(args.directory)
    if not count:
        parser.error(f"No .md or .txt documents found in {args.directory}")
    question = " ".join(args.question).strip()
    if question:
        print(rag.answer(question))
        return
    print(f"Indexed {count} document(s), {len(rag.nodes)} graph nodes.")
    print("Ask a question, or press Ctrl+C to exit.")
    while True:
        print()
        print(rag.answer(input("> ")))


if __name__ == "__main__":
    main()
