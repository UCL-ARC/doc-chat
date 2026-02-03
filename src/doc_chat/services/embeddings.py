"""Embedding backends for RAG: Ollama (local, no HF token) and optional SentenceTransformers."""

from __future__ import annotations

import logging
from typing import List, Union

import httpx
import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def is_ollama_embedding_model(model_name: str) -> bool:
    """Return True if the configured embedding model is an Ollama model."""
    return model_name.strip().lower().startswith("ollama/")


def _ollama_model_name(rag_embedding_model: str) -> str:
    """Extract Ollama model name from RAG_EMBEDDING_MODEL (e.g. 'ollama/nomic-embed-text' -> 'nomic-embed-text')."""
    return rag_embedding_model.split("/", 1)[1].strip()


async def embed_with_ollama(
    texts: List[str],
    model_name: str | None = None,
    base_url: str | None = None,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using Ollama's /api/embed endpoint.

    Uses OLLAMA_API_BASE_URL and RAG_EMBEDDING_MODEL from settings if not provided.
    Works with any Ollama model that supports embeddings (e.g. nomic-embed-text,
    mxbai-embed-large, or same LLM as chat like llama3.2 if it supports embeddings).

    Args:
        texts: List of input strings to embed.
        model_name: Ollama model name (e.g. 'nomic-embed-text'). If None, derived from settings.
        base_url: Ollama API base URL. If None, uses settings.OLLAMA_API_BASE_URL.

    Returns:
        numpy array of shape (len(texts), embedding_dim), float32, L2-normalized per row.

    Raises:
        httpx.HTTPError: On request failure.
        ValueError: If response format is invalid.
    """
    base_url = base_url or settings.OLLAMA_API_BASE_URL
    if model_name is None:
        model_name = _ollama_model_name(settings.RAG_EMBEDDING_MODEL)

    if not texts:
        return np.array([], dtype=np.float32).reshape(0, 0)

    base = base_url.rstrip("/")
    # Native Ollama endpoint (preferred)
    url_native = f"{base}/api/embed"
    payload: dict[str, Union[str, List[str]]] = {
        "model": model_name,
        "input": texts if len(texts) > 1 else texts[0],
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url_native, json=payload)

        if response.status_code == 404:
            # Fallback: OpenAI-compatible endpoint (some Ollama versions use only this)
            url_openai = f"{base}/v1/embeddings"
            logger.debug("Trying OpenAI-compatible embeddings endpoint %s", url_openai)
            response = await client.post(url_openai, json=payload)
            if response.status_code == 404:
                raise ValueError(
                    f"Ollama 404: embedding model '{model_name}' not found or Ollama too old. "
                    f"Pull the model with: ollama pull {model_name}. "
                    f"Ensure Ollama is running (e.g. ollama serve)."
                )
            response.raise_for_status()
            data = response.json()
            # OpenAI format: {"data": [{"embedding": [...]}, ...]}
            data_list = data.get("data") or []
            embeddings = [item.get("embedding") for item in data_list if "embedding" in item]
        else:
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")

    if not embeddings:
        raise ValueError("Ollama embed response missing 'embeddings' or 'data'")

    arr = np.array(embeddings, dtype=np.float32)
    # Normalize for cosine similarity (FAISS IndexFlatIP expects normalized vectors)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arr = arr / norms
    return arr
