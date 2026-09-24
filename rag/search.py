"""Поиск: векторный (HNSW), полнотекстовый (tsvector/GIN), гибрид через RRF, опционально reranker."""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field

from . import config as cfg
from .config import FTS_K, RRF_K, VECTOR_K

FTS_RANKING = os.getenv("FTS_RANKING", "bm25")  # bm25 | ts_rank_cd

MODES = ("vector", "fts", "hybrid", "hybrid_rerank")

# Вопросительные и служебные слова, которые стеммер не выкидывает, а в OR-запросе они дают шум.
_QUESTION_WORDS = {
    "какие", "какой", "какая", "каком", "каких", "каким", "какими", "каково", "каков", "какова",
    "каковы", "что", "чем", "чём", "как", "почему", "зачем", "где", "когда", "сколько", "ли",
    "можно", "нужно", "ли", "это", "этот", "эта", "эти", "который", "которые", "которая",
    "рассматривается", "рассматриваются", "рассматривались", "описывается", "описано",
    "предлагается", "предложено", "используется", "используются", "работе", "работа", "статье",
    "статья", "источник", "источнике", "автор", "авторы", "авторов",
}


@dataclass
class Hit:
    chunk_id: int
    zotero_key: str
    doc_type: str
    header: str
    text: str
    page_start: int | None
    page_end: int | None
    anchor: str | None
    file_name: str
    score: float = 0.0
    ranks: dict = field(default_factory=dict)  # ранги в отдельных списках, для отладки

    @property
    def location(self) -> str:
        if self.doc_type == "pdf" and self.page_start:
            p = f"с. {self.page_start}" if self.page_start == self.page_end else f"с. {self.page_start}–{self.page_end}"
            return p
        return f"раздел «{self.anchor}»" if self.anchor else "заметка"

    def source_label(self) -> str:
        return f"{self.zotero_key} | {self.file_name} | {self.location}"


_COLS = """c.id, c.zotero_key, c.doc_type, c.header, c.text, c.page_start, c.page_end, c.anchor, d.file_name"""


def _rows_to_hits(rows) -> list[Hit]:
    return [Hit(*r[:9], score=float(r[9])) for r in rows]


def _type_filter(doc_types: tuple[str, ...] | None) -> tuple[str, list]:
    if not doc_types:
        return "", []
    return " AND c.doc_type = ANY(%s)", [list(doc_types)]


def vector_search(conn, query: str, k: int = VECTOR_K, doc_types=None) -> list[Hit]:
    from .embed import embed_query

    qv = embed_query(query)
    flt, params = _type_filter(doc_types)
    conn.execute("SET hnsw.ef_search = 200")
    conn.execute("SET hnsw.iterative_scan = relaxed_order")  # корректный top-k при фильтре по doc_type
    rows = conn.execute(
        f"""SELECT {_COLS}, 1 - (c.embedding <=> %s) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE true {flt}
            ORDER BY c.embedding <=> %s LIMIT %s""",
        [qv, *params, qv, k],
    ).fetchall()
    return _rows_to_hits(rows)


def fts_query_text(query: str) -> str:
    words = re.findall(r"[\wёЁ-]+", query.lower())
    return " ".join(w for w in words if w not in _QUESTION_WORDS)


# Первая версия промпта перечисляла области знаний («гидрология, гидроэнергетика, ... машинное обучение»),
# и модель копировала этот список в термины каждого запроса — шум в BM25. Список убран (вариант v3 на dev).
TRANSLATE_PROMPT = """Переведи на английский ключевые термины поискового запроса так, как их пишут
в англоязычных научных статьях. Имена собственные и географические названия — в принятой английской
транслитерации. Только термины, которые есть в запросе; ничего не добавляй от себя.
Термины через запятую, одной строкой, без пояснений.

Запрос: {query}"""

def english_terms(query: str) -> tuple[str, float]:
    """Английские термины запроса от локальной LLM — только для FTS-ветки: русский вопрос к английской
    статье не имеет с ней общих лексем. Возвращает (термины, секунды на вызов LLM). Без кэша: в eval
    каждая конфигурация платит за перевод, как при реальном запросе (temperature=0 — ответ тот же)."""
    from .llm import chat

    t0 = time.perf_counter()
    # num_ctx как у генерации: другое значение заставляет Ollama перезагружать модель
    text, _ = chat([{"role": "user", "content": TRANSLATE_PROMPT.format(query=query)}],
                   temperature=0.0, num_ctx=8192)
    line = " ".join(text.strip().splitlines())[:400]
    return line, round(time.perf_counter() - t0, 3)


def fts_search(conn, query: str, k: int = FTS_K, doc_types=None, timings: dict | None = None) -> list[Hit]:
    """Полнотекстовый поиск по tsvector (GIN). Термины запроса объединяются через OR: plainto_tsquery
    требует присутствия ВСЕХ слов вопроса в чанке и почти ничего не находит.

    Ранжирование по умолчанию — BM25 поверх tsvector: IDF каждой лексемы берётся из ts_stat по корпусу
    (таблица fts_lexeme_stats), tf — число позиций лексемы в tsvector чанка. Встроенный ts_rank_cd
    не знает IDF, и на вопросах из 10–20 слов частые термины («ГЭС», «режим», «водохранилище»)
    вытесняют редкие и значимые. Старое ранжирование: FTS_RANKING=ts_rank_cd."""
    q = fts_query_text(query)
    if cfg.FTS_TRANSLATE:
        terms, dt = english_terms(query)
        q = f"{q} {fts_query_text(terms)}"
        if timings is not None:
            timings["translate_s"] = timings.get("translate_s", 0) + dt
            timings["fts_query"] = q
    if not q.strip():
        return []
    if FTS_RANKING == "ts_rank_cd":
        return _fts_ts_rank_cd(conn, q, k, doc_types)
    return _fts_bm25(conn, q, k, doc_types)


def _fts_ts_rank_cd(conn, q: str, k: int, doc_types) -> list[Hit]:
    flt, params = _type_filter(doc_types)
    rows = conn.execute(
        f"""WITH q AS (
                SELECT NULLIF(replace(plainto_tsquery('russian', %s)::text, ' & ', ' | '), '')::tsquery AS tq
            )
            SELECT {_COLS}, ts_rank_cd(c.tsv, q.tq, 1) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id, q
            WHERE q.tq IS NOT NULL AND c.tsv @@ q.tq {flt}
            ORDER BY score DESC LIMIT %s""",
        [q, *params, k],
    ).fetchall()
    return _rows_to_hits(rows)


BM25_K1, BM25_B = 1.2, 0.75


def refresh_fts_stats(conn) -> None:
    """Статистика для BM25: документная частота лексем и средняя длина tsvector. Пересчитывается
    после индексации (секунды на ~12 тыс. чанков)."""
    conn.execute("DROP TABLE IF EXISTS fts_lexeme_stats; DROP TABLE IF EXISTS fts_corpus_stats;")
    conn.execute("CREATE TABLE fts_lexeme_stats AS SELECT word, ndoc FROM ts_stat('SELECT tsv FROM chunks')")
    conn.execute("ALTER TABLE fts_lexeme_stats ADD PRIMARY KEY (word)")
    conn.execute("CREATE TABLE fts_corpus_stats AS SELECT count(*)::float8 AS n, avg(length(tsv))::float8 AS avgdl FROM chunks")


def _fts_bm25(conn, q: str, k: int, doc_types) -> list[Hit]:
    exists = conn.execute("SELECT to_regclass('fts_corpus_stats') IS NOT NULL").fetchone()[0]
    if not exists:
        refresh_fts_stats(conn)
    n, avgdl = conn.execute("SELECT n, avgdl FROM fts_corpus_stats").fetchone()
    lex = conn.execute(
        """SELECT u.lexeme, coalesce(s.ndoc, 0)
           FROM unnest(to_tsvector('russian', %s)) u LEFT JOIN fts_lexeme_stats s ON s.word = u.lexeme""",
        [q],
    ).fetchall()
    lex = [(w, df) for w, df in lex if df > 0]
    if not lex:
        return []
    words = [w for w, _ in lex]
    idf = [math.log((n - df + 0.5) / (df + 0.5) + 1.0) for _, df in lex]
    tsq = " | ".join("'" + w.replace("'", "''") + "'" for w in words)
    flt, params = _type_filter(doc_types)
    rows = conn.execute(
        f"""WITH q(lex, idf) AS (SELECT * FROM unnest(%s::text[], %s::float8[])),
            cand AS (
                SELECT c.id, c.tsv, length(c.tsv)::float8 AS dl FROM chunks c
                WHERE c.tsv @@ to_tsquery('russian', %s) {flt}
            ),
            scored AS (
                SELECT cand.id,
                       sum(q.idf * tf.n * (%s + 1) / (tf.n + %s * (1 - %s + %s * cand.dl / %s))) AS score
                FROM cand
                CROSS JOIN LATERAL (
                    SELECT u.lexeme, coalesce(array_length(u.positions, 1), 1)::float8 AS n
                    FROM unnest(cand.tsv) u WHERE u.lexeme = ANY(%s::text[])
                ) tf
                JOIN q ON q.lex = tf.lexeme
                GROUP BY cand.id
                ORDER BY score DESC LIMIT %s
            )
            SELECT {_COLS}, scored.score
            FROM scored JOIN chunks c ON c.id = scored.id JOIN documents d ON d.id = c.document_id
            ORDER BY scored.score DESC""",
        [words, idf, tsq, *params, BM25_K1, BM25_K1, BM25_B, BM25_B, avgdl, words, k],
    ).fetchall()
    return _rows_to_hits(rows)


def rrf_fuse(lists: dict[str, list[Hit]], k_rrf: int = RRF_K, weights: dict[str, float] | None = None) -> list[Hit]:
    """Взвешенный RRF: score = sum w_list / (k_rrf + rank). Без весов — классический RRF."""
    fused: dict[int, Hit] = {}
    for name, hits in lists.items():
        w = (weights or {}).get(name, 1.0)
        for rank, h in enumerate(hits, 1):
            if h.chunk_id not in fused:
                fused[h.chunk_id] = Hit(**{**h.__dict__, "score": 0.0, "ranks": {}})
            fused[h.chunk_id].score += w / (k_rrf + rank)
            fused[h.chunk_id].ranks[name] = rank
    return sorted(fused.values(), key=lambda h: h.score, reverse=True)


def hybrid_search(conn, query: str, doc_types=None, timings: dict | None = None) -> list[Hit]:
    return rrf_fuse({
        "vector": vector_search(conn, query, VECTOR_K, doc_types),
        "fts": fts_search(conn, query, FTS_K, doc_types, timings),
    }, weights={"fts": cfg.FTS_WEIGHT})


def search(conn, query: str, mode: str = "hybrid", top: int = 10, doc_types=None,
           timings: dict | None = None) -> list[Hit]:
    t0 = time.perf_counter()
    sub: dict = {}
    if mode == "vector":
        hits = vector_search(conn, query, max(top, VECTOR_K), doc_types)
    elif mode == "fts":
        hits = fts_search(conn, query, max(top, FTS_K), doc_types, sub)
    elif mode in ("hybrid", "hybrid_rerank"):
        hits = hybrid_search(conn, query, doc_types, sub)
    else:
        raise ValueError(f"unknown mode {mode}")
    t1 = time.perf_counter()
    if mode == "hybrid_rerank":
        from .rerank import rerank

        hits = rerank(query, hits[:cfg.RERANK_CANDIDATES])
    t2 = time.perf_counter()
    if timings is not None:
        # retrieve_s включает перевод запроса (translate_s), если он включён
        timings["retrieve_s"] = round(t1 - t0, 3)
        timings["rerank_s"] = round(t2 - t1, 3)
        timings.update(sub)
    return hits[:top]


def main() -> None:
    import argparse
    import textwrap

    from .db import connect

    ap = argparse.ArgumentParser(description="Поиск фрагментов по корпусу")
    ap.add_argument("query")
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    ap.add_argument("-k", "--top", type=int, default=5)
    ap.add_argument("--pdf-only", action="store_true", help="искать только в PDF, без заметок")
    args = ap.parse_args()

    timings: dict = {}
    with connect() as conn:
        hits = search(conn, args.query, args.mode, args.top,
                      doc_types=("pdf",) if args.pdf_only else None, timings=timings)
    print(f"Режим: {args.mode}; найдено {len(hits)}; время: {timings}\n")
    for i, h in enumerate(hits, 1):
        ranks = f" ранги={h.ranks}" if h.ranks else ""
        print(f"[{i}] {h.source_label()}  score={h.score:.4f}{ranks}")
        print(f"    {h.header}")
        print(textwrap.indent(textwrap.fill(h.text[:600] + ("…" if len(h.text) > 600 else ""), 110), "    "))
        print()


if __name__ == "__main__":
    main()
