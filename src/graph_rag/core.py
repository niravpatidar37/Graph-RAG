from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_ENTITY = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}(?:\s+[A-Z][a-z]+)*)\b")
_RELATION = re.compile(
    r"(?P<source>[A-Z][A-Za-z]*(?:\s+[A-Z][a-z]+)*)\s+"
    r"(?P<predicate>works at|founded|collaborates with|is based in|leads|joined)\s+"
    r"(?P<target>[A-Z][A-Za-z]*(?:\s+[A-Z][a-z]+)*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SearchResult:
    document: str
    text: str
    score: float
    entities: tuple[str, ...]


@dataclass
class GraphRAG:
    """In-memory Graph RAG pipeline: index, expand through a graph, and cite evidence."""

    chunks: list[tuple[str, str]] = field(default_factory=list)
    nodes: set[str] = field(default_factory=set)
    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_documents(self, documents: Iterable[tuple[str, str]]) -> None:
        for name, text in documents:
            for chunk in self._chunk(text):
                self.chunks.append((name, chunk))
                entities = self._entities(chunk)
                self.nodes.update(entities)
                for source, target in self._relations(chunk):
                    self.edges.setdefault(source, set()).add(target)
                    self.edges.setdefault(target, set()).add(source)

    def add_directory(self, directory: str | Path) -> int:
        files = sorted(Path(directory).glob("*.md")) + sorted(Path(directory).glob("*.txt"))
        self.add_documents((path.name, path.read_text(encoding="utf-8")) for path in files)
        return len(files)

    def search(self, question: str, limit: int = 5) -> list[SearchResult]:
        question_words = set(self._words(question))
        question_entities = self._entities(question)
        expanded_entities = set(question_entities)
        for entity in question_entities:
            expanded_entities.update(self.edges.get(entity, set()))

        results: list[SearchResult] = []
        for document, text in self.chunks:
            words = set(self._words(text))
            entities = tuple(sorted(self._entities(text)))
            lexical = len(question_words & words) / max(len(question_words), 1)
            graph_bonus = 0.15 * len(expanded_entities & set(entities))
            if lexical or graph_bonus:
                results.append(SearchResult(document, text, lexical + graph_bonus, entities))
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def answer(self, question: str, limit: int = 4) -> str:
        results = self.search(question, limit)
        if not results:
            return "I could not find supporting evidence in the indexed documents."
        evidence = " ".join(result.text.strip() for result in results)
        citations = ", ".join(sorted({result.document for result in results}))
        return f"Based on the indexed evidence: {evidence}\n\nSources: {citations}"

    @staticmethod
    def _chunk(text: str) -> list[str]:
        return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|\n+", text) if chunk.strip()]

    @staticmethod
    def _words(text: str) -> list[str]:
        return [word.lower() for word in _WORDS.findall(text)]

    @staticmethod
    def _entities(text: str) -> set[str]:
        return {match.group(0).strip() for match in _ENTITY.finditer(text)}

    @staticmethod
    def _relations(text: str) -> set[tuple[str, str]]:
        return {
            (match.group("source").strip(), match.group("target").strip())
            for match in _RELATION.finditer(text)
        }
