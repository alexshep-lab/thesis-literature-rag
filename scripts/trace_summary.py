"""Латентность по спанам последних трейсов из локального Phoenix (только имена, длительности, токены — без текста).

    .venv/Scripts/python.exe scripts/trace_summary.py [--last 3] [--project thesis-rag]
"""
from __future__ import annotations

import argparse
import sys

from phoenix.client import Client

for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8")

ATTRS = {
    "attributes.llm.token_count.prompt": "tok_in",
    "attributes.llm.token_count.completion": "tok_out",
}
# Phoenix отдаёт неизвестные OpenInference атрибуты вложенным словарём: attributes.ollama = {...}
OLLAMA = {"total_duration_s": "server_s", "load_duration_s": "load_s",
          "prompt_eval_duration_s": "prefill_s", "eval_duration_s": "decode_s"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=3, help="сколько последних трейсов показать")
    ap.add_argument("--project", default="thesis-rag")
    ap.add_argument("--url", default="http://127.0.0.1:6006")
    args = ap.parse_args()

    df = Client(base_url=args.url).spans.get_spans_dataframe(project_identifier=args.project, limit=2000)
    df["dur_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds().round(3)
    df = df.sort_values("start_time")
    roots = df[df["parent_id"].isna()].tail(args.last)
    by_id = df.set_index("context.span_id")
    cols = {k: v for k, v in ATTRS.items() if k in df.columns}

    def depth(span_id: str) -> int:
        d, pid = 0, by_id.at[span_id, "parent_id"]
        while isinstance(pid, str) and pid in by_id.index:
            d, pid = d + 1, by_id.at[pid, "parent_id"]
        return d

    for _, root in roots.iterrows():
        g = df[df["context.trace_id"] == root["context.trace_id"]]
        print(f"\nтрейс {root['context.trace_id'][:8]}  {root['start_time']:%H:%M:%S}  всего {root['dur_s']} с, спанов {len(g)}")
        for _, s in g.iterrows():
            extra = " ".join(f"{v}={s[k]:g}" for k, v in cols.items() if s[k] == s[k] and s[k] is not None)
            ol = s.get("attributes.ollama")
            if isinstance(ol, dict):
                extra += " " + " ".join(f"{v}={ol[k]:g}" for k, v in OLLAMA.items() if k in ol)
            print(f"  {'  ' * depth(s['context.span_id'])}{s['name']:<28} {s['span_kind']:<8} {s['dur_s']:>7.3f} с  {extra}")


if __name__ == "__main__":
    main()
