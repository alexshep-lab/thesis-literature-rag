"""Smoke-тест MCP-сервера через официальный Python-клиент SDK (stdio): список инструментов, search, ask.

    .venv/Scripts/python.exe scripts/mcp_smoke.py                       # короткий вопрос
    .venv/Scripts/python.exe scripts/mcp_smoke.py --eval 5 7            # вопросы №5 и №7 из evals/questions.yaml
    RAG_TRACE=1 .venv/Scripts/python.exe scripts/mcp_smoke.py ...       # трейсы сервера в локальный Phoenix

Печатает только служебные поля и источники; полные ответы и цитаты из корпуса не сохраняются.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"

SMOKE_Q = "SDDP стохастическое двойственное динамическое программирование"


async def run(questions: list[dict], search_only: bool) -> int:
    params = StdioServerParameters(command=str(PY if PY.exists() else sys.executable),
                                   args=["-m", "rag.mcp_server"], cwd=str(ROOT), env=dict(os.environ))
    fails = 0
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = {t.name: t for t in (await s.list_tools()).tools}
            print("tools:", sorted(tools), "| outputSchema:", {n: bool(t.output_schema) for n, t in tools.items()})
            assert {"search", "ask"} <= set(tools), "нет search/ask"

            for q in questions:
                t0 = time.perf_counter()
                r = await s.call_tool("search", {"query": q["q"], "mode": "hybrid_rerank", "k": 6})
                sc = r.structured_content or {}
                frags = sc.get("fragments", [])
                keys = [f["source_key"] for f in frags]
                hit = any(k in q.get("expect_sources", []) for k in keys) if q.get("expect_sources") else None
                print(f"\nsearch: {q['q'][:80]!r}\n  isError={r.is_error} {time.perf_counter() - t0:.1f} с, "
                      f"фрагментов {len(frags)}, timings={sc.get('timings_s')}")
                for f in frags[:3]:
                    print(f"  [{f['rank']}] {f['source_key']} | {f['location']} | score={f['score']}")
                if hit is not None:
                    print(f"  ожидаемый источник в top-6: {hit}")
                fails += bool(r.is_error or not frags)
                if search_only:
                    continue
                t0 = time.perf_counter()
                r = await s.call_tool("ask", {"question": q["q"]})
                sc = r.structured_content or {}
                cits = sc.get("citations", [])
                ck = {c["source_key"] for c in cits}
                hit = bool(ck & set(q.get("expect_sources", []))) if q.get("expect_sources") else None
                print(f"ask: isError={r.is_error} {time.perf_counter() - t0:.1f} с, self_check={sc.get('self_check')}, "
                      f"цитат {len(cits)} (отброшено {sc.get('dropped_citations')}), повторов {sc.get('retries')}, "
                      f"длина ответа {len(sc.get('answer', ''))} симв., размер JSON {len(r.content[0].text) if r.content else 0} симв.")
                print(f"  источники цитат: {sorted(ck)}; ожидаемый процитирован: {hit}")
                print(f"  timings={sc.get('timings_s')}")
                fails += bool(r.is_error or not sc.get("answer"))
    print("\nSMOKE", "OK" if not fails else f"FAIL ({fails})")
    return 1 if fails else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, nargs="*", help="номера вопросов (1-based) из evals/questions.yaml")
    ap.add_argument("--search-only", action="store_true")
    args = ap.parse_args()
    if args.eval:
        allq = yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text(encoding="utf-8"))
        questions = [allq[i - 1] for i in args.eval]
    else:
        questions = [{"q": SMOKE_Q}]
    sys.exit(asyncio.run(run(questions, args.search_only)))


if __name__ == "__main__":
    main()
