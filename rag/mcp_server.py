"""MCP-сервер (stdio, SDK mcp 2.x — MCPServer, бывший FastMCP) поверх поиска и графа: инструменты search и ask для агента.

    .venv/Scripts/python.exe -m rag.mcp_server       # запускает MCP-клиент (Claude Code по .mcp.json, Inspector)

stdout занят протоколом MCP: всё диагностическое — только в stderr.
Трейсинг в локальный Phoenix — RAG_TRACE=1 (см. rag/tracing.py).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Literal

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from . import tracing

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("rag.mcp")

Mode = Literal["vector", "fts", "hybrid", "hybrid_rerank"]

MAX_K = 20
SNIPPET_CHARS = 600
QUOTE_CHARS = 500
ANSWER_CHARS = 4000

# mcp 2.x: FastMCP переименован в MCPServer, API декораторов тот же
mcp = MCPServer(
    "thesis-literature-rag",
    instructions=(
        "Поиск и ответы с цитатами по личной научной библиотеке диссертации: гидроэнергетика, водные ресурсы, "
        "прогноз стока, стохастическая оптимизация (SDDP), ~180 PDF на русском и английском. Всё локально "
        "(PostgreSQL + pgvector, Ollama qwen2.5:7b). Вопросы задавай по-русски или по-английски. "
        "search — быстро найти фрагменты и источники; ask — готовый ответ с проверенными дословными цитатами "
        "(медленно, ~15–60 с)."
    ),
)


class Fragment(BaseModel):
    rank: int
    source_key: str = Field(description="ключ Zotero источника")
    file: str
    location: str = Field(description="страница PDF («с. 12») или раздел заметки")
    doc_type: Literal["pdf", "note"]
    score: float = Field(description="оценка в выбранном режиме; шкалы режимов несравнимы")
    header: str = Field(description="автор (год). Название")
    text: str = Field(description=f"начало фрагмента, не длиннее {SNIPPET_CHARS} символов")


class SearchResult(BaseModel):
    mode: str
    query: str
    fts_query: str | None = Field(None, description="запрос FTS-ветки с английскими терминами от LLM")
    timings_s: dict[str, float]
    fragments: list[Fragment]


class Citation(BaseModel):
    fragment: int = Field(description="номер фрагмента в контексте ответа (1-based)")
    quote: str
    source_key: str
    source: str = Field(description="ключ | файл | страница")
    support: float = Field(description="доля токенов цитаты, найденных во фрагменте подряд")


class AskResult(BaseModel):
    answer: str
    self_check: Literal["passed", "failed"] = Field(
        description="passed — подтверждённых цитат не меньше min(3, число фрагментов в контексте); "
                    "failed — меньше даже после повтора поиска. Если context_sources пуст, источники не найдены")
    citations: list[Citation] = Field(description="только цитаты, дословно найденные во фрагментах")
    dropped_citations: int = Field(description="сколько цитат модели отброшено проверкой")
    context_sources: list[str] = Field(description="фрагменты, которые видела модель: ключ | файл | страница")
    retries: int
    search_query: str = Field(description="запрос последней попытки поиска (после повтора — переформулированный); "
                                          "при self_check=failed ответ и цитаты — из лучшей попытки, она может быть первой")
    timings_s: dict[str, float]


# Один тяжёлый вызов за раз: параллельные вызовы на холодном старте грузили бы e5 и reranker дважды
# (lru_cache не защищён от гонки), а вторая копия reranker'а на GPU рядом с Ollama — риск CUDA OOM.
_HEAVY = anyio.CapacityLimiter(1)


async def _run(fn, *args):
    import psycopg
    import requests

    try:
        return await anyio.to_thread.run_sync(fn, *args, limiter=_HEAVY)
    except psycopg.OperationalError as e:
        log.exception("БД недоступна")
        raise ToolError("База PostgreSQL недоступна: запущен ли контейнер rag-pgvector (порт 5433) и задан ли "
                        "PGPASSWORD в .env?") from e
    except requests.RequestException as e:
        log.exception("Ollama недоступна")
        raise ToolError(f"Локальная LLM (Ollama) не ответила: {type(e).__name__}. Запущена ли Ollama с qwen2.5:7b?") from e


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _numeric(d: dict) -> dict[str, float]:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}


def _search_sync(query: str, mode: str, k: int, pdf_only: bool) -> SearchResult:
    from .db import connect
    from .search import search as run_search

    timings: dict = {}
    with connect() as conn:
        hits = run_search(conn, query, mode, k, doc_types=("pdf",) if pdf_only else None, timings=timings)
    return SearchResult(
        mode=mode,
        query=query,
        fts_query=timings.get("fts_query"),
        timings_s=_numeric(timings),
        fragments=[
            Fragment(rank=i, source_key=h.zotero_key, file=h.file_name, location=h.location,
                     doc_type=h.doc_type, score=round(h.score, 4), header=_clip(h.header, 200),
                     text=_clip(h.text, SNIPPET_CHARS))
            for i, h in enumerate(hits, 1)
        ],
    )


def _ask_sync(question: str, mode: str, pdf_only: bool) -> AskResult:
    from .db import connect
    from .graph import answer

    with connect() as conn:
        state = answer(conn, question, mode, ("pdf",) if pdf_only else None)
    cits = state.get("citations", [])
    valid = [c for c in cits if c["valid"]]
    return AskResult(
        answer=_clip(state.get("draft") or "", ANSWER_CHARS),
        self_check=state.get("self_check", "failed"),
        citations=[Citation(fragment=c["fragment"], quote=_clip(c["quote"], QUOTE_CHARS),
                            source_key=c["zotero_key"], source=c["source"], support=c["support"])
                   for c in valid],
        dropped_citations=len(cits) - len(valid),
        context_sources=[h.source_label() for h in state.get("selected", [])],
        retries=state.get("retries", 0),
        search_query=_clip(state.get("search_query", question), 300),
        timings_s=_numeric(state.get("timings", {})),
    )


# только чтение локальной базы и локальной LLM, без побочных эффектов и внешних сервисов
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


@mcp.tool(annotations=_READ_ONLY)
async def search(query: str, mode: Mode = "hybrid_rerank", k: int = 6, pdf_only: bool = True) -> SearchResult:
    """Найти фрагменты научной литературы по запросу, без генерации ответа.

    Возвращает до k фрагментов (k ≤ 20): ключ Zotero, файл, страница, score, начало текста.
    Используй, чтобы найти источники и страницы или проверить, есть ли тема в библиотеке.

    Режимы (mode):
    - hybrid_rerank (по умолчанию) — вектор + полнотекстовый через RRF, затем cross-encoder reranker.
      Лучший по качеству (на eval MRR@10 0,92), но самый медленный: ~4–7 с, первый вызов дольше —
      загрузка моделей.
    - fts — полнотекстовый BM25; русские вопросы автоматически дополняются английскими терминами от LLM.
      Сильнее vector и hybrid на кросс-языковых вопросах (русский вопрос к английской статье) и на точных
      терминах, фамилиях, названиях ГЭС; ~3 с. Хороший выбор, если reranker не нужен.
    - hybrid — RRF вектора и FTS без reranker'а: выше recall@10, но порядок хуже, чем у fts; ~4 с.
    - vector — только эмбеддинги e5; самый быстрый (~0,2 с), но слаб на кросс-языковых вопросах.
    pdf_only=True — искать только в полных текстах PDF; False — ещё и в аналитических заметках автора.
    """
    k = max(1, min(k, MAX_K))
    log.info("search mode=%s k=%s pdf_only=%s", mode, k, pdf_only)
    return await _run(_search_sync, query, mode, k, pdf_only)


@mcp.tool(annotations=_READ_ONLY)
async def ask(question: str, mode: Mode = "hybrid_rerank", pdf_only: bool = True) -> AskResult:
    """Ответить на вопрос по литературе с дословными цитатами (граф LangGraph:
    retrieve → rerank → generate → self_check).

    Модель qwen2.5:7b отвечает только по 6 найденным фрагментам. Каждая цитата проверяется на дословное
    присутствие во фрагменте; неподтверждённые отбрасываются. Если подтверждённых меньше 3, запрос
    переформулируется и поиск повторяется один раз. self_check=failed означает слабую опору на источники —
    так и сообщай пользователю, не выдавай ответ за надёжный.

    Медленно: ~15–60 с (локальная LLM). Для поиска источников без ответа бери search.
    mode — как в search; hybrid_rerank (по умолчанию) даёт лучшие цитаты, но дороже по времени;
    fts — быстрее и тоже сильный на русских вопросах к английским статьям.
    """
    log.info("ask mode=%s pdf_only=%s", mode, pdf_only)
    return await _run(_ask_sync, question, mode, pdf_only)


def main() -> None:
    # Без таймаута psycopg ждёт недоступную БД минутами (на этой машине 4 мин 20 с) — агент висит.
    # Только для MCP-сервера: CLI и eval не меняются.
    os.environ.setdefault("PGCONNECT_TIMEOUT", "10")
    if tracing.setup():
        log.info("трейсинг включён: %s", tracing.endpoint())
    mcp.run("stdio")


if __name__ == "__main__":
    main()
