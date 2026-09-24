"""Клиент локальной LLM через HTTP API Ollama (без облачных API)."""
from __future__ import annotations

import json
import os
import time

import requests

from .config import LLM_MODEL, LLM_NUM_PREDICT, LLM_TIMEOUT_S, OLLAMA_URL


def chat(messages: list[dict], schema: dict | None = None, temperature: float = 0.1,
         num_ctx: int = 8192, model: str | None = None) -> tuple[str, dict]:
    # num_predict: без лимита генерация по JSON-схеме изредка зацикливается (Ollama сдвигает контекст и пишет
    # до таймаута, 10 мин). Нормальный ответ — до ~950 токенов (eval 23.09), лимит на них не влияет.
    options = {"temperature": temperature, "num_ctx": num_ctx, "seed": 42, "num_predict": LLM_NUM_PREDICT}
    # LLM_NUM_GPU=0 — принудительно CPU (для замера латентности без видеокарты)
    if os.getenv("LLM_NUM_GPU") is not None:
        options["num_gpu"] = int(os.environ["LLM_NUM_GPU"])
    payload = {"model": model or LLM_MODEL, "messages": messages, "stream": False,
               "options": options, "keep_alive": "30m"}
    if schema is not None:
        payload["format"] = schema
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    meta = {
        "wall_s": round(time.perf_counter() - t0, 2),
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "model": data.get("model"),
    }
    return data["message"]["content"], meta


def chat_json(messages: list[dict], schema: dict, **kw) -> tuple[dict, dict]:
    text, meta = chat(messages, schema=schema, **kw)
    try:
        return json.loads(text), meta
    except json.JSONDecodeError:
        meta["parse_error"] = True
        return {}, meta
