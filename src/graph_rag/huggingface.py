from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any

from huggingface_hub import InferenceClient


class HuggingFaceModels:
    def __init__(self, token: str, embedding_model: str, ner_model: str, llm_model: str) -> None:
        self.client = InferenceClient(token=token)
        self.embedding_model = embedding_model
        self.ner_model = ner_model
        self.llm_model = llm_model
        self._embed_cache: dict[str, list[float]] = {}
        self._entity_cache: dict[str, list[str]] = {}

    def embed(self, text: str) -> list[float]:
        if text in self._embed_cache:
            return self._embed_cache[text]
        vector = self._retry(lambda: self.client.feature_extraction(text, model=self.embedding_model))
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        while vector and isinstance(vector[0], list):
            vector = vector[0]
        normalized = [float(value) for value in vector]
        self._embed_cache[text] = normalized
        return normalized

    def entities(self, text: str) -> list[str]:
        if text in self._entity_cache:
            return self._entity_cache[text]
        predictions = self._retry(lambda: self.client.token_classification(text, model=self.ner_model))
        names = set()
        for item in predictions:
            word = item.get("word") if isinstance(item, dict) else getattr(item, "word", "")
            if word:
                names.add(str(word).replace("##", "").strip())
        ordered = sorted(name for name in names if name)
        self._entity_cache[text] = ordered
        return ordered

    def relations(self, text: str) -> list[dict[str, Any]]:
        prompt = (
            "Extract factual relationships from the text. Return only a JSON array of objects "
            'with keys "source", "predicate", "target", and "confidence". '
            f"Text: {text}"
        )
        response = self._retry(lambda: self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.llm_model,
            max_tokens=500,
            temperature=0,
        ))
        content = response.choices[0].message.content or "[]"
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    def answer(self, question: str, context: str) -> str:
        response = self._retry(lambda: self.client.chat_completion(
            messages=[
                {"role": "system", "content": "Answer only from the supplied context. Cite supporting source IDs in square brackets. If the context does not support the answer, say you do not know."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nGive a concise answer and do not add facts not present in the context."},
            ],
            model=self.llm_model,
            max_tokens=250,
            temperature=0.1,
        ))
        return response.choices[0].message.content or "I could not generate an answer."

    @staticmethod
    def _retry(operation, attempts: int = 4):
        for attempt in range(attempts):
            try:
                return operation()
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(2 ** attempt)