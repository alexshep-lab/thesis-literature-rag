"""Парсинг PDF / Markdown в абзацы и упаковка в чанки.

Правила упаковки:
* размер считается в токенах токенизатора эмбеддинг-модели (e5 = XLM-R);
* абзац добавляется целиком, если влезает в CHUNK_MAX_TOKENS;
* если не влезает и чанк уже >= MIN — чанк закрывается на границе абзаца;
* иначе абзац режется по предложениям (никогда не посреди предложения,
  кроме «предложений» длиннее лимита — их режем по словам);
* перекрытие — хвостовые предложения закрытого чанка в пределах ~15 % его длины.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pymupdf as fitz

from .config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_RATIO,
    CHUNK_TARGET_TOKENS,
    EMBED_MODEL,
)

CHUNK_MIN_TOKENS = int(CHUNK_TARGET_TOKENS * 0.6)

_SENT_SPLIT = re.compile(r"(?<=[.!?…;])\s+(?=[\"«(\[]?[A-ZА-ЯЁ0-9])")
_HYPHEN_BREAK = re.compile(r"(\w)[-\u00ad]\s*\n\s*(\w)")


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBED_MODEL)


def count_tokens(texts: list[str]) -> list[int]:
    if not texts:
        return []
    enc = tokenizer()(texts, add_special_tokens=False)["input_ids"]
    return [len(x) for x in enc]


@dataclass
class Sent:
    text: str
    ntok: int
    loc: int | str  # номер страницы (PDF) или заголовок раздела (заметка)


@dataclass
class Chunk:
    text: str
    n_tokens: int
    page_start: int | None = None
    page_end: int | None = None
    anchor: str | None = None


@dataclass
class ParseStats:
    pages: int = 0
    empty_pages: int = 0          # страницы без текстового слоя (скан)
    blocks_total: int = 0
    blocks_repeated: int = 0      # колонтитулы
    blocks_symbolic: int = 0      # формулы / таблицы чисел: мало букв
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- PDF

def _letter_ratio(s: str) -> float:
    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return 0.0
    return sum(c.isalpha() for c in non_space) / len(non_space)


def _clean_block(text: str) -> str:
    text = text.replace("\x00", "")                  # NUL-байты в битых PDF — PostgreSQL их не принимает
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # «опти-\nмизация» -> «оптимизация»
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s*\n\s*", " ", text)            # переносы строк внутри абзаца
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return text.strip()


def parse_pdf(path: Path) -> tuple[list[tuple[str, int]], ParseStats]:
    """Возвращает абзацы (блоки pymupdf) с номером страницы (1-based)."""
    stats = ParseStats()
    raw: list[tuple[str, int]] = []
    with fitz.open(path) as doc:
        stats.pages = doc.page_count
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            blocks = page.get_text("blocks")
            page_text = [b[4] for b in blocks if b[6] == 0 and b[4].strip()]
            if sum(len(t.strip()) for t in page_text) < 30:
                stats.empty_pages += 1
                continue
            for t in page_text:
                raw.append((_clean_block(t), pno + 1))
    stats.blocks_total = len(raw)

    # колонтитулы: короткие блоки, повторяющиеся на многих страницах
    norm = lambda s: re.sub(r"\d+", "#", s.lower())[:80]
    freq = Counter(norm(t) for t, _ in raw if len(t) < 150)
    rep_threshold = max(3, int(0.2 * max(stats.pages, 1)))

    paras: list[tuple[str, int]] = []
    for text, page in raw:
        if len(text) < 150 and freq[norm(text)] >= rep_threshold:
            stats.blocks_repeated += 1
            continue
        if len(text) < 3 or re.fullmatch(r"[\d\s.,–-]+", text):
            stats.blocks_symbolic += 1
            continue
        if _letter_ratio(text) < 0.5:
            stats.blocks_symbolic += 1
            continue
        paras.append((text, page))
    return paras, stats


# ---------------------------------------------------------------- Markdown

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")


def _clean_md(text: str) -> str:
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"(?m)^\s*[*-]\s+(\[[ x]\]\s*)?", "• ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


def parse_markdown(path: Path) -> list[tuple[str, str]]:
    """Возвращает абзацы заметки с якорем — текущим заголовком раздела."""
    anchor = ""
    paras: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush():
        if buf:
            t = _clean_md("\n".join(buf))
            if t and t not in ("---", "•"):
                paras.append((t, anchor))
            buf.clear()

    for line in path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").splitlines():
        m = _MD_HEADING.match(line)
        if m:
            flush()
            anchor = _clean_md(m.group(1)).strip(" :") or anchor
            paras.append((anchor, anchor))  # заголовок — отдельный короткий абзац
            continue
        if not line.strip() or line.strip() == "---":
            flush()
            continue
        buf.append(line)
    flush()
    return paras


# ---------------------------------------------------------------- упаковка

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _split_long(sent: str, ntok: int) -> list[str]:
    """«Предложение» длиннее лимита (таблица, список литературы) — режем по словам."""
    words = sent.split()
    n_parts = ntok // CHUNK_TARGET_TOKENS + 1
    size = max(1, len(words) // n_parts + 1)
    return [" ".join(words[i: i + size]) for i in range(0, len(words), size)]


def _to_sentences(paras: list[tuple[str, int | str]]) -> list[list[Sent]]:
    flat: list[tuple[int, str, int | str]] = []
    for pi, (text, loc) in enumerate(paras):
        for s in _split_sentences(text):
            flat.append((pi, s, loc))
    toks = count_tokens([s for _, s, _ in flat])
    out: dict[int, list[Sent]] = {}
    for (pi, s, loc), n in zip(flat, toks):
        if n > CHUNK_MAX_TOKENS:
            pieces = _split_long(s, n)
            for p, pn in zip(pieces, count_tokens(pieces)):
                out.setdefault(pi, []).append(Sent(p, pn, loc))
        else:
            out.setdefault(pi, []).append(Sent(s, n, loc))
    return [out[k] for k in sorted(out)]


def pack(paras: list[tuple[str, int | str]]) -> list[Chunk]:
    para_sents = _to_sentences(paras)
    chunks: list[Chunk] = []
    cur: list[Sent] = []
    n_overlap = 0  # сколько первых предложений cur — перекрытие с прошлым чанком

    def tot() -> int:
        return sum(s.ntok for s in cur)

    def close():
        nonlocal cur, n_overlap
        if len(cur) <= n_overlap:
            return
        chunks.append(_make_chunk(cur))
        budget = CHUNK_OVERLAP_RATIO * tot()
        ov: list[Sent] = []
        acc = 0
        for s in reversed(cur):
            if acc + s.ntok > budget:
                break
            ov.insert(0, s)
            acc += s.ntok
        cur, n_overlap = ov, len(ov)

    for sents in para_sents:
        ptok = sum(s.ntok for s in sents)
        if tot() + ptok <= CHUNK_MAX_TOKENS:
            cur.extend(sents)
            continue
        if tot() - sum(s.ntok for s in cur[:n_overlap]) >= CHUNK_MIN_TOKENS:
            close()  # закрываем на границе абзаца
            if tot() + ptok <= CHUNK_MAX_TOKENS:
                cur.extend(sents)
                continue
        for s in sents:  # абзац не влез целиком — режем по предложениям
            if len(cur) > n_overlap and tot() + s.ntok > CHUNK_MAX_TOKENS:
                close()
            cur.append(s)
    close()
    return chunks


def _make_chunk(sents: list[Sent]) -> Chunk:
    text = " ".join(s.text for s in sents)
    n = sum(s.ntok for s in sents)
    locs = [s.loc for s in sents]
    if isinstance(locs[0], int):
        return Chunk(text, n, page_start=min(locs), page_end=max(locs))
    anchor = next((l for l in locs if l), None)
    return Chunk(text, n, anchor=anchor)
