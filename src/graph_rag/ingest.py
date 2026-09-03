from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import CloudGraphRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Markdown and text files into Graph RAG cloud stores.")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    pipeline = CloudGraphRAG.from_env()
    files = sorted(args.directory.glob("*.md")) + sorted(args.directory.glob("*.txt"))
    chunks = sum(pipeline.ingest_file(path) for path in files)
    print(f"Indexed {chunks} chunks from {len(files)} documents.")