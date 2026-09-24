"""Индексация корпуса: PDF + заметки -> чанки -> e5-эмбеддинги -> PostgreSQL.

    python -m rag.index            # доиндексировать то, чего нет в базе
    python -m rag.index --reset    # пересоздать таблицы и проиндексировать всё
    python -m rag.index --dry-run  # только парсинг и чанкинг, статистика без БД
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime

from .chunking import count_tokens, pack, parse_markdown, parse_pdf
from .config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_RATIO, CHUNK_TARGET_TOKENS, EMBED_MODEL, ROOT
from .corpus import chapter_map, discover, make_header, parse_bib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["pdf", "note"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    bib = parse_bib()
    chapters = chapter_map()
    files = discover()
    if args.only:
        files = [f for f in files if f.doc_type == args.only]
    if args.limit:
        files = files[: args.limit]

    conn = None
    done: set[tuple[str, str]] = set()
    if not args.dry_run:
        from .db import connect, init_schema

        init_schema(reset=args.reset)
        conn = connect(autocommit=True)  # каждый документ — отдельная транзакция, индексация возобновляема
        done = {tuple(r) for r in conn.execute(
            "SELECT zotero_key, doc_type FROM documents WHERE n_chunks IS NOT NULL")}

    t0 = time.perf_counter()
    t_parse = t_embed = 0.0
    stats = {"docs": 0, "docs_skipped_existing": 0, "docs_no_text": [], "docs_no_bib": [],
             "pages": 0, "empty_pages": 0, "blocks_total": 0, "blocks_repeated": 0,
             "blocks_symbolic": 0, "chunks": {"pdf": 0, "note": 0}, "chunk_tokens": [],
             "embed_input_over_512": 0}

    for i, f in enumerate(files, 1):
        if (f.key, f.doc_type) in done:
            stats["docs_skipped_existing"] += 1
            continue
        tp = time.perf_counter()
        header, meta = make_header(f.key, bib)
        if meta["title"] is None:
            stats["docs_no_bib"].append(f.key)
        n_pages = None
        try:
            if f.doc_type == "pdf":
                paras, ps = parse_pdf(f.path)
                n_pages = ps.pages
                for k in ("pages", "empty_pages", "blocks_total", "blocks_repeated", "blocks_symbolic"):
                    stats[k] += getattr(ps, k)
            else:
                paras = parse_markdown(f.path)
        except Exception as e:  # битый PDF не должен ронять всю индексацию
            print(f"[{i}/{len(files)}] ОШИБКА {f.path.name}: {e}")
            stats["docs_no_text"].append(f.key + f" ({f.doc_type}, error)")
            continue
        chunks = pack(paras)
        t_parse += time.perf_counter() - tp
        if not chunks:
            stats["docs_no_text"].append(f"{f.key} ({f.doc_type})")
            print(f"[{i}/{len(files)}] {f.path.name}: нет текста (скан без OCR?)")
            continue
        stats["docs"] += 1
        stats["chunks"][f.doc_type] += len(chunks)
        stats["chunk_tokens"].extend(c.n_tokens for c in chunks)

        embed_inputs = [f"{header}\n{c.text}" for c in chunks]
        header_tok = count_tokens([header])[0]
        stats["embed_input_over_512"] += sum(1 for c in chunks if c.n_tokens + header_tok + 4 > 512)

        if conn is not None:
            from .embed import embed_passages

            te = time.perf_counter()
            vecs = embed_passages(embed_inputs)
            t_embed += time.perf_counter() - te
            with conn.transaction():
                conn.execute("DELETE FROM documents WHERE zotero_key=%s AND doc_type=%s", (f.key, f.doc_type))
                doc_id = conn.execute(
                    """INSERT INTO documents (zotero_key, doc_type, title, authors, year, chapters, lang,
                                              file_name, n_pages, n_chunks)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                    (f.key, f.doc_type, meta["title"], meta["authors"], meta["year"],
                     chapters.get(f.key, []), meta["lang"], f.path.name, n_pages, len(chunks)),
                ).fetchone()[0]
                with conn.cursor() as cur:
                    cur.executemany(
                        """INSERT INTO chunks (document_id, zotero_key, doc_type, chunk_index, header, text,
                                               page_start, page_end, anchor, n_tokens, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        [(doc_id, f.key, f.doc_type, j, header, c.text, c.page_start, c.page_end,
                          c.anchor, c.n_tokens, v) for j, (c, v) in enumerate(zip(chunks, vecs))],
                    )
        el = time.perf_counter() - t0
        print(f"[{i}/{len(files)}] {f.doc_type:4s} {f.key[:55]:55s} чанков={len(chunks):4d}  t={el:6.0f}s")

    total = time.perf_counter() - t0
    toks = stats.pop("chunk_tokens")
    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "embed_model": EMBED_MODEL,
        "chunking": {"target_tokens": CHUNK_TARGET_TOKENS, "max_tokens": CHUNK_MAX_TOKENS,
                     "overlap_ratio": CHUNK_OVERLAP_RATIO},
        "dry_run": args.dry_run,
        **stats,
        "chunks_total": sum(stats["chunks"].values()),
        "chunk_tokens_mean": round(statistics.mean(toks), 1) if toks else None,
        "chunk_tokens_median": statistics.median(toks) if toks else None,
        "time_total_s": round(total, 1),
        "time_parse_chunk_s": round(t_parse, 1),
        "time_embed_s": round(t_embed, 1),
    }
    if conn is not None:
        from .search import refresh_fts_stats

        refresh_fts_stats(conn)  # IDF для BM25-ранжирования полнотекстового поиска
        summary["db_documents"] = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        summary["db_chunks"] = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        summary["db_size"] = conn.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()[0]
        conn.close()
    out = ROOT / "evals" / "results" / ("index_stats_dry.json" if args.dry_run else "index_stats.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("docs_no_bib",)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
