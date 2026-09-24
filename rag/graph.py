"""Граф LangGraph: retrieve -> [rerank] -> generate -> self_check (-> retrieve, не более 1 повтора).

    python -m rag ask "вопрос"                 # гибрид + reranker (по умолчанию)
    python -m rag ask "вопрос" --mode hybrid   # без reranker'а (быстрее на CPU)
"""
from __future__ import annotations

import argparse
import textwrap
import time
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .citations import verify
from .config import CONTEXT_FRAGMENTS, RERANK_CANDIDATES
from .llm import chat, chat_json
from .search import MODES, rrf_fuse, search

MAX_RETRIES = 1
MIN_VALID_CITATIONS = 3


class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    mode: str                 # vector | fts | hybrid | hybrid_rerank
    doc_types: tuple | None   # ('pdf',) — искать только в PDF
    candidates: list          # кандидаты после retrieve
    selected: list            # фрагменты, попавшие в промпт
    draft: str                # черновик ответа
    citations: list           # проверенные цитаты
    retries: int
    self_check: str           # passed | retry | failed
    timings: dict
    llm_meta: list
    history: list             # что происходило на попытках (для отладки и eval)
    best: dict                # лучшая попытка (если повтор не помог)


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {"fragment": {"type": "integer"}, "quote": {"type": "string"}},
                "required": ["fragment", "quote"],
            },
        },
    },
    "required": ["answer", "citations"],
}

SYSTEM_PROMPT = """Ты — ассистент по научной литературе диссертации (гидроэнергетика, водные ресурсы, прогнозирование, оптимизация).
Правила:
1. Отвечай по-русски и ТОЛЬКО на основе пронумерованных фрагментов. Внешние знания не используй.
2. Если во фрагментах нет ответа, прямо напиши, что в найденных источниках ответа нет.
3. В тексте ответа ставь ссылки на фрагменты в квадратных скобках: [1], [3].
4. Приведи от 3 до 5 цитат в поле citations. Каждая цитата — ДОСЛОВНЫЙ непрерывный отрывок фрагмента
   (8–40 слов), скопированный символ в символ на языке оригинала, без перевода, пересказа и многоточий.
   В поле fragment — номер фрагмента, из которого взят отрывок.
5. Ответ краткий: 3–8 предложений."""

REWRITE_PROMPT = """Поиск по научной библиотеке не дал достаточных подтверждений для ответа на вопрос.
Сформулируй ОДИН улучшенный поисковый запрос: ключевые термины вопроса на русском и их английские эквиваленты.
Используй только понятия, фамилии и названия, которые есть в вопросе; ничего не выдумывай.
Без пояснений, одной строкой, не длиннее 25 слов.

Вопрос: {question}"""


def _fragments_block(hits) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] Источник: {h.zotero_key}, {h.location}. {h.header}\n{h.text}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ узлы

def retrieve(state: RAGState, config: RunnableConfig) -> dict:
    conn = config["configurable"]["conn"]
    timings: dict = {}
    mode = state.get("mode", "hybrid_rerank")
    base_mode = "hybrid" if mode == "hybrid_rerank" else mode
    hits = search(conn, state["search_query"], base_mode, top=RERANK_CANDIDATES,
                  doc_types=state.get("doc_types"), timings=timings)
    if state.get("retries", 0) > 0 and state.get("candidates"):
        # повтор: не теряем кандидатов первой попытки, объединяем списки через RRF
        hits = rrf_fuse({"first": state["candidates"], "retry": hits})[:RERANK_CANDIDATES]
    t = dict(state.get("timings", {}))
    t["retrieve_s"] = round(t.get("retrieve_s", 0) + timings["retrieve_s"], 3)
    return {"candidates": hits, "selected": hits[:CONTEXT_FRAGMENTS], "timings": t}


def rerank_node(state: RAGState) -> dict:
    from .rerank import rerank

    t0 = time.perf_counter()
    ranked = rerank(state["search_query"], list(state["candidates"]))
    t = dict(state.get("timings", {}))
    t["rerank_s"] = round(t.get("rerank_s", 0) + time.perf_counter() - t0, 3)
    return {"candidates": ranked, "selected": ranked[:CONTEXT_FRAGMENTS], "timings": t}


def generate(state: RAGState) -> dict:
    hits = state["selected"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Вопрос: {state['question']}\n\nФрагменты:\n\n{_fragments_block(hits)}"},
    ]
    t0 = time.perf_counter()
    data, meta = chat_json(messages, ANSWER_SCHEMA)
    t = dict(state.get("timings", {}))
    t["generate_s"] = round(t.get("generate_s", 0) + time.perf_counter() - t0, 3)
    citations = verify(data.get("citations", []), hits)
    return {
        "draft": data.get("answer", ""),
        "citations": citations,
        "timings": t,
        "llm_meta": [*state.get("llm_meta", []), meta],
    }


def self_check(state: RAGState) -> dict:
    """Утверждения должны опираться на фрагменты: каждая цитата проверяется на дословное
    присутствие в указанном фрагменте. Мало подтверждённых цитат -> переформулировать запрос
    и повторить поиск (один раз)."""
    cits = state.get("citations", [])
    valid = [c for c in cits if c["valid"]]
    need = min(MIN_VALID_CITATIONS, len(state.get("selected", [])))
    attempt = {
        "search_query": state["search_query"],
        "n_citations": len(cits),
        "n_valid": len(valid),
        "citations_raw": cits,
        "selected_keys": [h.zotero_key for h in state.get("selected", [])],
    }
    history = [*state.get("history", []), attempt]
    best = state.get("best")
    if best is None or len(valid) > best["n_valid"]:
        best = {"n_valid": len(valid), "draft": state.get("draft"), "citations": cits,
                "selected": state.get("selected", [])}
    if len(valid) >= need:
        return {"self_check": "passed", "history": history, "best": best}
    if state.get("retries", 0) >= MAX_RETRIES:
        # повтор не помог: возвращаем лучшую из попыток (по числу подтверждённых цитат)
        return {"self_check": "failed", "history": history, "best": best, "draft": best["draft"],
                "citations": best["citations"], "selected": best["selected"]}
    t0 = time.perf_counter()
    new_q, meta = chat([{"role": "user", "content": REWRITE_PROMPT.format(question=state["question"])}],
                       temperature=0.2, num_ctx=8192)  # как у генерации: иначе Ollama перезагружает модель
    t = dict(state.get("timings", {}))
    t["rewrite_s"] = round(time.perf_counter() - t0, 3)
    new_q = new_q.strip().strip('"«»').splitlines()[0][:300] if new_q.strip() else state["question"]
    return {"self_check": "retry", "search_query": new_q, "retries": state.get("retries", 0) + 1,
            "history": history, "best": best, "timings": t,
            "llm_meta": [*state.get("llm_meta", []), meta]}


def _after_retrieve(state: RAGState) -> str:
    return "rerank" if state.get("mode", "hybrid_rerank") == "hybrid_rerank" else "generate"


def _after_check(state: RAGState) -> str:
    return "retrieve" if state.get("self_check") == "retry" else END


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", generate)
    g.add_node("self_check", self_check)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", _after_retrieve, {"rerank": "rerank", "generate": "generate"})
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "self_check")
    g.add_conditional_edges("self_check", _after_check, {"retrieve": "retrieve", END: END})
    return g.compile()


_GRAPH = None


def answer(conn, question: str, mode: str = "hybrid_rerank", doc_types=None) -> dict[str, Any]:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    t0 = time.perf_counter()
    state = _GRAPH.invoke(
        {"question": question, "search_query": question, "mode": mode, "doc_types": doc_types,
         "retries": 0, "timings": {}, "llm_meta": [], "history": []},
        config={"configurable": {"conn": conn}},
    )
    state["timings"]["total_s"] = round(time.perf_counter() - t0, 3)
    return state


def format_answer(state: dict) -> str:
    lines = [textwrap.fill(state.get("draft") or "(пустой ответ)", 110), "", "Цитаты:"]
    valid = [c for c in state.get("citations", []) if c["valid"]]
    for c in valid:
        lines.append(f"  [{c['fragment']}] «{c['quote']}»")
        lines.append(f"      — {c['source']}")
    dropped = [c for c in state.get("citations", []) if not c["valid"]]
    if dropped:
        lines.append(f"  (отброшено self_check: {len(dropped)} цитат(ы) не найдены дословно во фрагментах)")
    if not valid:
        lines.append("  нет подтверждённых цитат")
    lines += ["", "Фрагменты в контексте:"]
    for i, h in enumerate(state.get("selected", []), 1):
        lines.append(f"  [{i}] {h.source_label()}")
    lines += ["", f"self_check: {state.get('self_check')}, повторов: {state.get('retries', 0)}; "
                  f"запрос поиска: {state.get('search_query')!r}",
              f"время: {state.get('timings')}"]
    return "\n".join(lines)


def main() -> None:
    from .db import connect

    ap = argparse.ArgumentParser(description="Ответ с цитатами по литературе (LangGraph)")
    ap.add_argument("question")
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank",
                    help="hybrid_rerank (по умолчанию) или hybrid — без cross-encoder, быстрее на CPU")
    ap.add_argument("--pdf-only", action="store_true")
    args = ap.parse_args()
    with connect() as conn:
        state = answer(conn, args.question, args.mode, ("pdf",) if args.pdf_only else None)
    print(format_answer(state))


if __name__ == "__main__":
    main()
