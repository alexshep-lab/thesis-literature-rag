"""Эмбеддинги multilingual-e5-base (CPU). e5 требует префиксы 'query: ' / 'passage: '."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer

    m =SentenceTransformer(EMBED_MODEL, device="cpu")
    m.max_seq_length = 512
    return m


def embed_passages(texts: list[str], batch_size: int = 16, progress: bool = False) -> np.ndarray:
    return model().encode(
        ["passage: " + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=progress,
        convert_to_numpy=True,
    )


def embed_query(text: str) -> np.ndarray:
    return model().encode(["query: " + text], normalize_embeddings=True, convert_to_numpy=True)[0]
