"""Подключение к PostgreSQL + pgvector и схема БД."""
from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from .config import EMBED_DIM, pg_conninfo

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          serial PRIMARY KEY,
    zotero_key  text NOT NULL,
    doc_type    text NOT NULL CHECK (doc_type IN ('pdf', 'note')),
    title       text,
    authors     text,
    year        text,
    chapters    text[],
    lang        text,
    file_name   text NOT NULL,
    n_pages     int,
    n_chunks    int,
    indexed_at  timestamptz DEFAULT now(),
    UNIQUE (zotero_key, doc_type)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    zotero_key   text NOT NULL,
    doc_type     text NOT NULL,
    chunk_index  int NOT NULL,
    header       text NOT NULL,          -- «Автор (год). Название» — контекст для поиска
    text         text NOT NULL,
    page_start   int,                    -- для PDF: 1-based номер страницы
    page_end     int,
    anchor       text,                   -- для заметок: ближайший заголовок раздела
    n_tokens     int,
    embedding    vector({EMBED_DIM}),
    -- Конфигурация 'russian' в PostgreSQL стеммит кириллицу russian_stem,
    -- а латиницу (asciiword) — english_stem, поэтому одной колонки хватает на RU+EN корпус.
    tsv          tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('russian', header), 'A') ||
                    setweight(to_tsvector('russian', text), 'B')
                 ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_key_idx ON chunks (zotero_key);
CREATE INDEX IF NOT EXISTS chunks_doc_type_idx ON chunks (doc_type);
"""


def connect(autocommit: bool = False) -> psycopg.Connection:
    conn = psycopg.connect(pg_conninfo(), autocommit=autocommit)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(reset: bool = False) -> None:
    with psycopg.connect(pg_conninfo(), autocommit=True) as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS documents;")
        conn.execute(SCHEMA_SQL)
