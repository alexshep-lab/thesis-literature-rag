"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3): CPU или GPU (RERANK_DEVICE)."""
from __future__ import annotations

from functools import lru_cache

from . import config as cfg
from .config import RERANK_MODEL


def device() -> str:
    if cfg.RERANK_DEVICE != "auto":
        return cfg.RERANK_DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import CrossEncoder

    dev = device()
    # на GPU — fp16: вдвое меньше видеопамяти рядом с моделью Ollama
    kw = {"model_kwargs": {"torch_dtype": "float16"}} if dev == "cuda" else {}
    return CrossEncoder(RERANK_MODEL, max_length=512, device=dev, **kw)


def rerank(query: str, hits: list, batch_size: int = 8) -> list:
    if not hits:
        return hits
    scores = model().predict(
        [(query, f"{h.header}\n{h.text}") for h in hits], batch_size=batch_size, show_progress_bar=False
    )
    for h, s in zip(hits, scores):
        h.ranks = {**h.ranks, "rrf_score": round(h.score, 5)}
        h.score = float(s)
    return sorted(hits, key=lambda h: h.score, reverse=True)
