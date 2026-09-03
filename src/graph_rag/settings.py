from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    hf_token: str
    hf_embedding_model: str = "BAAI/bge-base-en-v1.5"
    hf_ner_model: str = "dslim/bert-base-NER"
    hf_llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    neo4j_uri: str = ""
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "graph_rag_chunks"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            hf_token=os.environ.get("HF_TOKEN", ""),
            hf_embedding_model=os.environ.get("HF_EMBEDDING_MODEL", cls.hf_embedding_model),
            hf_ner_model=os.environ.get("HF_NER_MODEL", cls.hf_ner_model),
            hf_llm_model=os.environ.get("HF_LLM_MODEL", cls.hf_llm_model),
            neo4j_uri=os.environ.get("NEO4J_URI", ""),
            neo4j_username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", ""),
            qdrant_url=os.environ.get("QDRANT_URL", ""),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY", ""),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", cls.qdrant_collection),
            langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=os.environ.get("LANGFUSE_HOST", cls.langfuse_host),
        )

    def validate_cloud(self) -> None:
        missing = [
            name for name, value in {
                "HF_TOKEN": self.hf_token,
                "NEO4J_URI": self.neo4j_uri,
                "NEO4J_PASSWORD": self.neo4j_password,
                "QDRANT_URL": self.qdrant_url,
                "QDRANT_API_KEY": self.qdrant_api_key,
            }.items() if not value
        ]
        if missing:
            raise RuntimeError(f"Missing cloud configuration: {', '.join(missing)}")