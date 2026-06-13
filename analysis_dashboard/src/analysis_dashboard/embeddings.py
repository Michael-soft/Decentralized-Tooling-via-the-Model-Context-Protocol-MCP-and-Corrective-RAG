"""
embeddings.py (analysis side)
─────────────────────────────
Mirror of the operational client's embedding configuration.

Both the writer (agent_client) and this reader MUST embed with the same
local model, otherwise query vectors will not align with the stored
document vectors and semantic search will silently degrade.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384


class FastEmbedEmbeddings(Embeddings):
    """LangChain-compatible wrapper around a local fastembed ONNX model."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vec)) for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, next(iter(self._model.embed([text])))))


@lru_cache(maxsize=1)
def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name)
