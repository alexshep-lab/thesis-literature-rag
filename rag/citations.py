"""Проверка цитат: действительно ли цитата из ответа есть в указанном фрагменте.

Сравнение по нормализованным токенам: регистр, ё/е, пунктуация и пробелы не важны.
Цитата засчитывается, если она подстрока фрагмента или >= 90 % её токенов
совпадают с фрагментом одними и теми же блоками по порядку (допуск на пропущенное слово).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.9
MIN_QUOTE_TOKENS = 4


def _tokens(s: str) -> list[str]:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"(\w)[-\u00ad]\s+(\w)", r"\1\2", s)
    return re.findall(r"\w+", s)


def quote_support(quote: str, fragment: str) -> float:
    q, f = _tokens(quote), _tokens(fragment)
    if len(q) < MIN_QUOTE_TOKENS or not f:
        return 0.0
    if " ".join(q) in " ".join(f):
        return 1.0
    sm = SequenceMatcher(None, q, f, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks() if b.size >= 2)
    return matched / len(q)


def verify(citations: list[dict], fragments: list) -> list[dict]:
    """citations: [{'fragment': n (1-based), 'quote': str}], fragments: list[Hit]."""
    out = []
    for c in citations or []:
        try:
            n = int(c.get("fragment"))
        except (TypeError, ValueError):
            n = -1
        quote = str(c.get("quote", "")).strip().strip("«»\"“”")
        ok_index = 1 <= n <= len(fragments)
        support = quote_support(quote, fragments[n - 1].text) if ok_index else 0.0
        out.append({
            "fragment": n,
            "quote": quote,
            "support": round(support, 3),
            "valid": ok_index and support >= FUZZY_THRESHOLD,
            "source": fragments[n - 1].source_label() if ok_index else None,
            "zotero_key": fragments[n - 1].zotero_key if ok_index else None,
        })
    return out
