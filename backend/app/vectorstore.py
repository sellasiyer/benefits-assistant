"""RAG ingestion: load -> chunk -> embed -> store in Chroma (a persistent vector DB).

Interview talking points live in the comments:
  * Markdown-aware splitting keeps each chunk inside one section, and the section heading is
    copied into metadata so citations can say *where* an answer came from.
  * Overlap stops a fact being cut in half at a chunk boundary.
  * Stable chunk IDs (file + index) make re-ingestion idempotent: upsert, not duplicate.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import settings
from .providers import get_embeddings


@lru_cache
def get_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},  # cosine similarity, HNSW index
    )


def chunk_markdown(text: str, source: str) -> list[Document]:
    by_section = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False
    ).split_text(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(by_section)
    for i, c in enumerate(chunks):
        c.metadata.update(source=source, chunk=i)
    return chunks


def ingest_text(text: str, source: str) -> int:
    chunks = chunk_markdown(text, source)
    store = get_store()
    # delete old chunks for this file first so a shorter new version leaves no orphans
    old = store.get(where={"source": source})
    if old["ids"]:
        store.delete(ids=old["ids"])
    store.add_documents(chunks, ids=[f"{source}::{c.metadata['chunk']}" for c in chunks])
    return len(chunks)


def ingest_dir(path: Path | None = None) -> dict[str, int]:
    path = path or settings.docs_dir
    return {f.name: ingest_text(f.read_text(encoding="utf-8"), f.name) for f in sorted(path.glob("*.md"))}


def retrieve(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Return (chunk, relevance 0..1) pairs, best first."""
    return get_store().similarity_search_with_relevance_scores(query, k=k or settings.top_k)


def doc_count() -> int:
    return len(get_store().get()["ids"])
