"""Chroma access: create, list, inspect and delete collections (subjects).

Listing and stats go through the raw chromadb client on purpose. Instantiating
the LangChain wrapper without a collection name creates the default collection
named "langchain" as a side effect, and that is where the phantom subject in
the old menu came from. The raw client only reads.

Reading a collection's stats never touches Ollama either, so the status panel
still works with the model server down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .config import EMBED_MODEL, OLLAMA_URL, PERSIST_DIR, Settings, validate_chunking


class StoreError(RuntimeError):
    """Base class for anything wrong with the vector store."""


class CollectionNotFound(StoreError):
    pass


class CollectionExists(StoreError):
    pass


class InvalidCollectionName(StoreError):
    pass


# Chroma's own rule: 3-63 chars, starts and ends alphanumeric, and only
# letters, digits, dot, dash and underscore in between. Checked here so the
# user gets a readable message instead of a driver-level error.
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")


# ---- CLIENT ----
@lru_cache(maxsize=1)
def client() -> chromadb.ClientAPI:
    """The single PersistentClient for this process.

    Cached because chroma_db is SQLite: one process, one connection. The web
    server must run with a single worker for the same reason.
    """
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(PERSIST_DIR))


def embeddings() -> OllamaEmbeddings:
    """The one embedding model, for ingestion and for queries alike.

    If these ever disagree, retrieval silently returns nonsense instead of
    failing, which is why EMBED_MODEL is a locked constant.
    """
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)


# ---- COLLECTIONS ----
def validate_name(name: str) -> str:
    """Normalise and check a subject name, or raise InvalidCollectionName."""
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise InvalidCollectionName(
            f"Invalid subject name: '{name}'. Use 3 to 63 characters, letters, "
            f"digits, '.', '-' or '_', starting and ending with a letter or digit."
        )
    if ".." in name:
        raise InvalidCollectionName(f"Invalid subject name: '{name}'. It cannot contain '..'.")
    return name


def list_collections() -> list[str]:
    """Names of every existing subject. Creates nothing."""
    return sorted(c.name for c in client().list_collections())


def exists(name: str) -> bool:
    return name in list_collections()


def create(name: str, chunk_size: int, chunk_overlap: int) -> str:
    """Create an empty subject, recording its chunking for good.

    Chunking belongs to the collection, so it has to be chosen before the first
    PDF goes in -- which is why creating a subject is its own action instead of
    a side effect of ingesting.
    """
    name = validate_name(name)
    validate_chunking(chunk_size, chunk_overlap)
    if exists(name):
        raise CollectionExists(f"Subject '{name}' already exists.")

    client().create_collection(
        name=name,
        # None on purpose: LangChain computes the vectors and hands them over,
        # so Chroma must not fall back to its bundled ONNX model.
        embedding_function=None,
        metadata={
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "embed_model": EMBED_MODEL,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return name


def delete(name: str) -> None:
    """Drop a subject and every vector in it. Irreversible."""
    _collection(name)  # raises CollectionNotFound with a clear message
    client().delete_collection(name)


def open_vectorstore(name: str) -> Chroma:
    """LangChain wrapper over an existing collection, for adding and searching.

    create_collection_if_not_exists=False is deliberate: a typo has to fail
    loudly instead of quietly producing a new empty subject.
    """
    _collection(name)
    return Chroma(
        client=client(),
        collection_name=name,
        embedding_function=embeddings(),
        create_collection_if_not_exists=False,
    )


def _collection(name: str) -> chromadb.Collection:
    try:
        return client().get_collection(name=name, embedding_function=None)
    except (NotFoundError, ValueError) as exc:
        raise CollectionNotFound(f"Subject '{name}' does not exist.") from exc


# ---- METADATA AND STATS ----
@dataclass
class Chunking:
    chunk_size: int
    chunk_overlap: int
    embed_model: str
    # True when the collection predates per-collection chunking and carries no
    # metadata at all. Its real chunk size is unknowable; these are the current
    # global defaults, shown as a guess rather than as fact.
    legacy: bool


@dataclass
class CollectionStats:
    name: str
    chunks: int
    pdfs: list[str]
    chunking: Chunking
    created_at: str | None


def chunking_of(name: str, settings: Settings) -> Chunking:
    """Chunking recorded on a collection, falling back for legacy ones.

    Collections created before this refactor have metadata = None. Tolerating
    that is what keeps the existing 'patterns' subject usable.
    """
    meta = _collection(name).metadata or {}
    if "chunk_size" not in meta:
        return Chunking(
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
            embed_model=EMBED_MODEL,
            legacy=True,
        )
    return Chunking(
        chunk_size=int(meta["chunk_size"]),
        chunk_overlap=int(meta.get("chunk_overlap", 0)),
        embed_model=str(meta.get("embed_model", EMBED_MODEL)),
        legacy=False,
    )


def pdfs_in(name: str) -> set[str]:
    """File names already ingested into a subject. Drives the dedup on ingest."""
    data = _collection(name).get(include=["metadatas"])
    return {m["source_pdf"] for m in data["metadatas"] or [] if m and m.get("source_pdf")}


def pages_in(name: str) -> dict[str, set[int]]:
    """Which pages of which PDF actually produced chunks.

    A page with no text layer produces no chunk and can therefore never be
    retrieved. The evaluation harness checks its ground truth against this, so
    an unreachable page is reported as a broken question rather than scored as
    a miss the retriever is blamed for.
    """
    data = _collection(name).get(include=["metadatas"])
    pages: dict[str, set[int]] = {}
    for meta in data["metadatas"] or []:
        if not meta:
            continue
        pdf, page = meta.get("source_pdf"), meta.get("page")
        if pdf is None or page is None:
            continue
        pages.setdefault(pdf, set()).add(int(page))
    return pages


def stats(name: str, settings: Settings) -> CollectionStats:
    collection = _collection(name)
    meta = collection.metadata or {}
    return CollectionStats(
        name=name,
        chunks=collection.count(),
        pdfs=sorted(pdfs_in(name)),
        chunking=chunking_of(name, settings),
        created_at=meta.get("created_at"),
    )


def stats_all(settings: Settings) -> list[CollectionStats]:
    return [stats(name, settings) for name in list_collections()]
