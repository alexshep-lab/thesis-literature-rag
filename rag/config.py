"""Настройки проекта. Всё, что зависит от машины, читается из окружения / .env."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Windows-консоль: печатаем кириллицу без кракозябр
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# По умолчанию корпус ищется в ./corpus (в .gitignore); свои пути — в .env
_DEFAULT_CORPUS = str(ROOT / "corpus")
_DEFAULT_BIB = str(ROOT / "corpus" / "library.bib")

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", _DEFAULT_CORPUS))
PDF_DIR = CORPUS_DIR / "PDFs"
NOTES_DIR = CORPUS_DIR / "Individual_Notes"
MANIFEST_PATH = CORPUS_DIR / "export_manifest.json"
BIB_PATH = Path(os.getenv("BIB_PATH", _DEFAULT_BIB))

# --- модели ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBED_DIM = 768
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "600"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "1500"))  # лимит токенов ответа LLM

# --- чанкинг (в токенах токенизатора e5 / XLM-R) ---
# e5-base обрезает вход на 512 токенах, поэтому тело чанка + заголовок должны влезать в 512.
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "400"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "440"))
CHUNK_OVERLAP_RATIO = float(os.getenv("CHUNK_OVERLAP_RATIO", "0.15"))

# --- поиск ---
VECTOR_K = 50       # кандидатов из векторного поиска
FTS_K = 50          # кандидатов из полнотекстового
RRF_K = 60          # константа RRF
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))  # сколько гибридных кандидатов отдаём reranker'у
CONTEXT_FRAGMENTS = 6   # сколько фрагментов попадает в промпт генерации
# Варианты гибрида (выбор — на dev-части eval, см. README). Читаются в search.py в момент вызова,
# eval может переопределить их флагами.
FTS_WEIGHT = float(os.getenv("FTS_WEIGHT", "1.0"))        # вес полнотекстового списка в RRF (вектор = 1)
FTS_TRANSLATE = os.getenv("FTS_TRANSLATE", "1") == "1"    # английские термины запроса (LLM) в FTS-ветку; 0 — выключить
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "auto")        # auto (cuda, если есть) | cpu | cuda


def pg_conninfo() -> str:
    missing = [v for v in ("PGPASSWORD",) if not os.getenv(v)]
    if missing:
        raise RuntimeError("Не задан PGPASSWORD: скопируйте .env.example в .env и задайте пароль")
    return (
        f"host={os.getenv('PGHOST', 'localhost')} port={os.getenv('PGPORT', '5433')} "
        f"dbname={os.getenv('PGDATABASE', 'rag')} user={os.getenv('PGUSER', 'rag')} "
        f"password={os.getenv('PGPASSWORD')}"
    )
