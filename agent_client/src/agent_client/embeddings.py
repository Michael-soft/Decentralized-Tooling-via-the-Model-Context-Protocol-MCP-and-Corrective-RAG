"""
embeddings.py
─────────────
Local, key-free embedding model for the LangGraph vector store.

Groq exposes no embeddings endpoint, so the hierarchical log store is
backed by a CPU-friendly ONNX model (BAAI/bge-small-en-v1.5, 384-dim)
served through `fastembed`. Using a local model guarantees there are no
broken HTTP connection strings and no extra API keys in the pipeline.

The class implements the minimal LangChain `Embeddings` contract
(`embed_documents` / `embed_query`) so it can be handed directly to the
`index={"embed": ...}` config of `SqliteStore` / `AsyncSqliteStore`.

IMPORTANT: the SAME model name must be used by every process that opens
the store (the operational MCP client that writes vectors and the
decoupled analysis agent that queries them), otherwise query embeddings
will not align with the stored document embeddings.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

#: Shared across the writer (agent_client) and reader (analysis_dashboard).
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384


class FastEmbedEmbeddings(Embeddings):
    """LangChain-compatible wrapper around a local fastembed ONNX model."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents for vector indexing."""
        return [list(map(float, vec)) for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for similarity search."""
        return list(map(float, next(iter(self._model.embed([text])))))


@lru_cache(maxsize=1)
def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> FastEmbedEmbeddings:
    """Return a process-wide singleton embedding model (lazy, cached)."""
    return FastEmbedEmbeddings(model_name)
