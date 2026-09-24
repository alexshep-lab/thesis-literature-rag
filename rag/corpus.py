"""Метаданные корпуса: BibTeX из Zotero, привязка к главам из export_manifest.json, список файлов.

Имя файла в «Базе знаний» совпадает с ключом цитирования Zotero:
PDFs/<key>.pdf и Individual_Notes/<key>_note.md.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import BIB_PATH, MANIFEST_PATH, NOTES_DIR, PDF_DIR

CHAPTER_NAMES = {
    "00_Введение.md": "Введение",
    "01_Глава_1_Обзор_методов.md": "Глава 1",
    "02_Глава_2_Моделирование.md": "Глава 2",
    "03_Глава_3_Оптимизация.md": "Глава 3",
    "04_Глава_4_Реализация.md": "Глава 4",
    "05_Без_привязки.md": "Без привязки",
}


@dataclass
class BibEntry:
    key: str
    entry_type: str
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return _clean(self.fields.get("title", ""))

    @property
    def year(self) -> str:
        m = re.search(r"\d{4}", self.fields.get("date", "") or self.fields.get("year", ""))
        return m.group(0) if m else ""

    @property
    def authors_list(self) -> list[str]:
        raw = self.fields.get("author") or self.fields.get("editor") or ""
        return [_clean(a) for a in re.split(r"\s+and\s+", raw) if a.strip()]

    @property
    def authors_short(self) -> str:
        names = self.authors_list
        if not names:
            return ""
        first = names[0].split(",")[0].strip()
        return first + (" и др." if len(names) > 1 else "")

    @property
    def lang(self) -> str:
        return self.fields.get("langid", "")


def _clean(s: str) -> str:
    s = s.replace("{", "").replace("}", "").replace("\\&", "&")
    return re.sub(r"\s+", " ", s).strip()


def parse_bib(path: Path = BIB_PATH) -> dict[str, BibEntry]:
    """Минимальный парсер BibLaTeX-экспорта Zotero (поля вида name = {value})."""
    text = path.read_text(encoding="utf-8")
    entries: dict[str, BibEntry] = {}
    for m in re.finditer(r"@(\w+)\{([^,\s]+),", text):
        etype, key = m.group(1).lower(), m.group(2)
        if etype in ("comment", "string", "preamble"):
            continue
        # тело записи — до парной закрывающей скобки
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[m.end(): i - 1]
        entry = BibEntry(key=key, entry_type=etype)
        pos = 0
        for fm in re.finditer(r"(\w+)\s*=\s*\{", body):
            if fm.start() < pos:
                continue
            depth, j = 1, fm.end()
            while j < len(body) and depth:
                depth += {"{": 1, "}": -1}.get(body[j], 0)
                j += 1
            entry.fields[fm.group(1).lower()] = body[fm.end(): j - 1]
            pos = j
        entries[key] = entry
    return entries


def chapter_map(path: Path = MANIFEST_PATH) -> dict[str, list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for fname, keys in manifest.get("file_mapping", {}).items():
        chapter = CHAPTER_NAMES.get(fname, fname)
        for k in keys:
            out.setdefault(k, []).append(chapter)
    return out


@dataclass
class SourceFile:
    key: str
    doc_type: str  # 'pdf' | 'note'
    path: Path


def discover() -> list[SourceFile]:
    files = [SourceFile(p.stem, "pdf", p) for p in sorted(PDF_DIR.glob("*.pdf"))]
    for p in sorted(NOTES_DIR.glob("*.md")):
        key = p.stem[: -len("_note")] if p.stem.endswith("_note") else p.stem
        files.append(SourceFile(key, "note", p))
    return files


def make_header(key: str, bib: dict[str, BibEntry]) -> tuple[str, dict]:
    """Короткий заголовок чанка «Автор (год). Название» + метаданные документа."""
    e = bib.get(key)
    if e is None:
        return key, {"title": None, "authors": None, "year": None, "lang": None}
    parts = []
    if e.authors_short:
        parts.append(e.authors_short + (f" ({e.year})" if e.year else ""))
    elif e.year:
        parts.append(f"({e.year})")
    title = e.title
    if len(title) > 200:
        title = title[:200] + "…"
    header = ". ".join(p for p in parts + [title] if p)
    return header or key, {
        "title": e.title or None,
        "authors": "; ".join(e.authors_list) or None,
        "year": e.year or None,
        "lang": e.lang or None,
    }
