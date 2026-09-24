"""Прогон eval-набора одной командой и сравнение конфигураций поиска.

    python -m rag eval                    # 4 конфигурации: поиск + генерация ответа
    python -m rag eval --no-generate      # только метрики поиска (быстро)
    python -m rag eval --scope pdf        # искать только в PDF (без заметок)
    python -m rag eval --split dev        # только dev-часть (настройка); test — один раз в конце

Метрики (источник = ключ Zotero; попадание засчитывается, если ключ чанка входит в expect_sources):
    recall@k           доля вопросов, где ожидаемый источник есть среди top-k чанков
    MRR@10             среднее 1/ранг первого чанка ожидаемого источника (0, если его нет в top-10)
    citation_precision доля цитат в ответах, которые дословно найдены в указанном фрагменте
    latency            время поиска и полного ответа графа (p50/p95)

Метрики поиска дополнительно разбиты по группам вопросов:
    same   язык вопроса (русский) совпадает с языком источника
    cross  русский вопрос к англоязычному источнику
    scan   источник — скан без текстового слоя (в scope=pdf найти нельзя)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml

from . import config as cfg
from .config import CONTEXT_FRAGMENTS, LLM_MODEL, ROOT
from .search import MODES, search

EVAL_PATH = ROOT / "evals" / "questions.yaml"
RESULTS_DIR = ROOT / "evals" / "results"


def load_questions(path: Path = EVAL_PATH) -> list[dict]:
    qs = yaml.safe_load(path.read_text(encoding="utf-8"))
    for i, q in enumerate(qs, 1):
        q.setdefault("id", i)
        q.setdefault("must_contain", [])
        q.setdefault("origin", "note")
        q["group"] = "scan" if q.get("scan") else ("cross" if q.get("src_lang") == "en" else "same")
    return qs


def retrieval_metrics(keys: list[str], expected: set[str]) -> dict:
    rank = next((i for i, k in enumerate(keys[:10], 1) if k in expected), None)
    return {
        "hit@1": int(rank == 1),
        "hit@5": int(rank is not None and rank <= 5),
        "hit@10": int(rank is not None),
        "rr": 1.0 / rank if rank else 0.0,
        "rank": rank,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--no-generate", action="store_true", help="только метрики поиска")
    ap.add_argument("--gen-configs", nargs="+", choices=MODES, help="для каких конфигураций генерировать ответ")
    ap.add_argument("--scope", choices=["all", "pdf"], default="all",
                    help="all — PDF + заметки; pdf — только полные тексты")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fts-weight", type=float, help=f"вес FTS в RRF (по умолчанию {cfg.FTS_WEIGHT})")
    ap.add_argument("--fts-translate", choices=["0", "1"], help="английские термины запроса в FTS-ветке")
    ap.add_argument("--rerank-candidates", type=int, help=f"кандидатов reranker'у (по умолчанию {cfg.RERANK_CANDIDATES})")
    ap.add_argument("--rerank-device", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    if args.fts_weight is not None:
        cfg.FTS_WEIGHT = args.fts_weight
    if args.fts_translate is not None:
        cfg.FTS_TRANSLATE = args.fts_translate == "1"
    if args.rerank_candidates is not None:
        cfg.RERANK_CANDIDATES = args.rerank_candidates
    if args.rerank_device is not None:
        cfg.RERANK_DEVICE = args.rerank_device

    from .db import connect

    questions = load_questions()
    if args.split != "all":
        questions = [q for q in questions if q.get("split") == args.split]
    if args.limit:
        questions = questions[: args.limit]
    doc_types = ("pdf",) if args.scope == "pdf" else None
    gen_configs = [] if args.no_generate else (args.gen_configs or args.configs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    split = "" if args.split == "all" else f"_{args.split}"
    name = f"eval_{stamp}_{args.scope}{split}{'_' + args.tag if args.tag else ''}"
    details_path = RESULTS_DIR / f"{name}.jsonl"
    details = details_path.open("w", encoding="utf-8")

    conn = connect()
    # прогрев моделей, чтобы загрузка весов не попала в латентность
    search(conn, "прогрев", "vector", top=1)
    if "hybrid_rerank" in args.configs:
        search(conn, "прогрев", "hybrid_rerank", top=1)
    if gen_configs or cfg.FTS_TRANSLATE:
        from .llm import chat

        chat([{"role": "user", "content": "Ответь одним словом: да"}], num_ctx=8192)

    summary: dict[str, dict] = {}
    for mode in args.configs:
        rows = []
        print(f"\n=== {mode} ({args.scope}) ===")
        for q in questions:
            expected = set(q["expect_sources"])
            timings: dict = {}
            hits = search(conn, q["q"], mode, top=10, doc_types=doc_types, timings=timings)
            keys = [h.zotero_key for h in hits]
            m = retrieval_metrics(keys, expected)
            row = {"id": q["id"], "mode": mode, "scope": args.scope, "q": q["q"],
                   "expect": sorted(expected), "top10_keys": keys,
                   "kind": q.get("kind"), "src_lang": q.get("src_lang"), "group": q["group"],
                   "origin": q["origin"], "split": q.get("split"),
                   "retrieval_s": timings["retrieve_s"] + timings["rerank_s"],
                   "translate_s": timings.get("translate_s"), "fts_query": timings.get("fts_query"), **m}
            if mode in gen_configs:
                from .graph import answer

                try:
                    st = answer(conn, q["q"], mode, doc_types)
                except Exception as exc:  # сбой одного вопроса (таймаут LLM и т. п.) не обрывает прогон
                    st = {"draft": "", "citations": [], "self_check": "error", "error": repr(exc)[:300],
                          "timings": {"total_s": None}}
                cits = st.get("citations", [])
                text = (st.get("draft") or "").lower()
                row.update({
                    "answer": st.get("draft"),
                    "citations": cits,
                    "n_citations": len(cits),
                    "n_valid_citations": sum(c["valid"] for c in cits),
                    "cited_expected_source": int(any(c["valid"] and c["zotero_key"] in expected for c in cits)),
                    "must_contain_ok": int(all(s.lower() in text for s in q["must_contain"])) if q["must_contain"] else None,
                    "self_check": st.get("self_check"),
                    "retries": st.get("retries", 0),
                    "context_keys": [h.zotero_key for h in st.get("selected", [])][:CONTEXT_FRAGMENTS],
                    "answer_s": st["timings"]["total_s"],
                    "timings": st["timings"],
                    "llm_meta": st.get("llm_meta"),
                    "history_valid": [h["n_valid"] for h in st.get("history", [])],
                    "error": st.get("error"),
                })
            rows.append(row)
            details.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            details.flush()
            extra = ""
            if "answer_s" in row:
                extra = (f" цитат={row['n_valid_citations']}/{row['n_citations']} "
                         f"check={row['self_check']} t={row['answer_s'] or 0:.1f}s")
            print(f"  q{q['id']:>2} rank={str(m['rank']):>4}{extra}")

        s = summarize(rows, with_generation=mode in gen_configs)
        summary[mode] = s
        print(json.dumps(s, ensure_ascii=False))
    details.close()
    conn.close()

    from .rerank import device as rerank_device

    meta = {"run_at": datetime.now().isoformat(timespec="seconds"), "scope": args.scope, "split": args.split,
            "n_questions": len(questions), "llm": LLM_MODEL if gen_configs else None,
            "gen_configs": gen_configs, "context_fragments": CONTEXT_FRAGMENTS,
            "fts_weight": cfg.FTS_WEIGHT, "fts_translate": cfg.FTS_TRANSLATE,
            "rerank_candidates": cfg.RERANK_CANDIDATES,
            "rerank_device": rerank_device() if "hybrid_rerank" in args.configs else None}
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    table = render_table(summary)
    (RESULTS_DIR / f"{name}.md").write_text(
        f"# Eval {meta['run_at']} (scope={args.scope}, split={args.split}, вопросов={len(questions)}, LLM={meta['llm']})\n\n"
        f"Настройки: FTS_WEIGHT={cfg.FTS_WEIGHT}, FTS_TRANSLATE={cfg.FTS_TRANSLATE}, "
        f"RERANK_CANDIDATES={cfg.RERANK_CANDIDATES}, reranker на {meta['rerank_device']}\n\n"
        f"{table}\n\n## По группам вопросов\n\n{render_groups(summary)}\n",
        encoding="utf-8")
    print("\n" + table + "\n\n" + render_groups(summary))
    print(f"\nСохранено: {RESULTS_DIR / name}.{{json,md,jsonl}}")


def summarize(rows: list[dict], with_generation: bool) -> dict:
    """Метрики одной конфигурации по строкам .jsonl (используется и для пересчёта сохранённых прогонов)."""
    n = len(rows)
    s = {
        "n_questions": n,
        "recall@1": sum(r["hit@1"] for r in rows) / n,
        "recall@5": sum(r["hit@5"] for r in rows) / n,
        "recall@10": sum(r["hit@10"] for r in rows) / n,
        "MRR@10": sum(r["rr"] for r in rows) / n,
        "retrieval_ms_p50": round(1000 * _pct([r["retrieval_s"] for r in rows], 0.5)),
        "retrieval_ms_p95": round(1000 * _pct([r["retrieval_s"] for r in rows], 0.95)),
        "retrieval_ms_mean": round(1000 * _mean([r["retrieval_s"] for r in rows])),
    }
    if any(r.get("translate_s") for r in rows):
        s["translate_ms_p50"] = round(1000 * _pct([r.get("translate_s") for r in rows], 0.5))
    for field, val in (("split", "dev"), ("split", "test"), ("group", "same"), ("group", "cross"), ("group", "scan"),
                       ("kind", "lex"), ("kind", "sem"), ("origin", "note"), ("origin", "pdf")):
        sub = [r for r in rows if r.get(field) == val]
        if sub:
            s[f"recall@5[{val}]"] = sum(r["hit@5"] for r in sub) / len(sub)
            s[f"MRR@10[{val}]"] = sum(r["rr"] for r in sub) / len(sub)
            s[f"recall@10[{val}]"] = sum(r["hit@10"] for r in sub) / len(sub)
            s[f"n[{val}]"] = len(sub)
    if with_generation:
        total_c = sum(r["n_citations"] for r in rows)
        valid_c = sum(r["n_valid_citations"] for r in rows)
        mc = [r["must_contain_ok"] for r in rows if r["must_contain_ok"] is not None]
        s.update({
            "citation_precision": valid_c / total_c if total_c else None,
            "citations_per_answer": total_c / n,
            "answers_with_3plus_valid": sum(r["n_valid_citations"] >= 3 for r in rows) / n,
            "answers_citing_expected": sum(r["cited_expected_source"] for r in rows) / n,
            "must_contain_rate": sum(mc) / len(mc) if mc else None,
            "self_check_retry_rate": sum(r["retries"] > 0 for r in rows) / n,
            "errors": sum(bool(r.get("error")) for r in rows),
            "answer_s_p50": _pct([r["answer_s"] for r in rows], 0.5),
            "answer_s_p95": _pct([r["answer_s"] for r in rows], 0.95),
            "answer_s_mean": round(_mean([r["answer_s"] for r in rows]), 2),
            "cost_usd": 0.0,
        })
    return s


def render_table(summary: dict) -> str:
    cols = [("recall@5", "recall@5", "{:.2f}"), ("recall@10", "recall@10", "{:.2f}"),
            ("MRR@10", "MRR@10", "{:.3f}"), ("retrieval_ms_p50", "поиск p50, мс", "{}"),
            ("retrieval_ms_p95", "поиск p95, мс", "{}"),
            ("citation_precision", "citation_precision", "{:.2f}"),
            ("answers_citing_expected", "ответ цитирует нужный источник", "{:.2f}"),
            ("answer_s_p50", "ответ p50, с", "{:.1f}"), ("answer_s_p95", "ответ p95, с", "{:.1f}")]
    cols = [c for c in cols if any(c[0] in s for s in summary.values())]
    head = "| конфигурация | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [head, sep]
    for mode, s in summary.items():
        vals = [c[2].format(s[c[0]]) if s.get(c[0]) is not None else "—" for c in cols]
        lines.append(f"| {mode} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def render_groups(summary: dict) -> str:
    groups = ["dev", "test", "same", "cross", "scan", "note", "pdf"]

    def n(g):
        return next((s[f"n[{g}]"] for s in summary.values() if f"n[{g}]" in s), 0)

    groups = [g for g in groups if n(g)]
    head = "| конфигурация | " + " | ".join(f"{g} (n={n(g)}) R@5 / R@10 / MRR" for g in groups) + " |"
    lines = [head, "|---|" + "---|" * len(groups)]
    for mode, s in summary.items():
        cells = []
        for g in groups:
            if f"n[{g}]" in s:
                cells.append(f"{s[f'recall@5[{g}]']:.2f} / {s[f'recall@10[{g}]']:.2f} / {s[f'MRR@10[{g}]']:.3f}")
            else:
                cells.append("—")
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
