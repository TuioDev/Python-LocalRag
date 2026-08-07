"""PDF -> pages -> chunks -> embeddings, reporting progress through a callback.

The callback is the reason this module exists outside the CLI. The old code
could only report progress by printing, and a browser cannot read stdout; with
on_progress the same function feeds a terminal line and an SSE stream.

Chunking is read from the collection, never from the global settings: two PDFs
in the same subject must be split the same way, or the retriever compares
fragments of different sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import store
from .config import PDF_DIR, Settings


@dataclass
class Progress:
    """One step of an ingestion, for whoever is watching."""

    stage: str  # reading | splitting | embedding | done | skipped
    done: int
    total: int
    message: str


@dataclass
class IngestResult:
    pdf: str
    collection: str
    skipped: bool
    pages: int
    chunks: int


ProgressCallback = Callable[[Progress], None]


def available_pdfs() -> list[Path]:
    """PDFs sitting in the pdfs/ folder, ready to be ingested."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PDF_DIR.glob("*.pdf"))


def ingest_pdf(
    pdf_path: str | Path,
    collection: str,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
) -> IngestResult:
    """Load one PDF into an existing subject, incrementally and in batches.

    Re-ingesting a file already in the subject is skipped rather than
    duplicated; the dedup key is the file name, stored on every chunk as
    metadata["source_pdf"].
    """
    pdf_path = Path(pdf_path)
    file_name = pdf_path.name

    def report(stage: str, done: int, total: int, message: str) -> None:
        if on_progress is not None:
            on_progress(Progress(stage=stage, done=done, total=total, message=message))

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Raises CollectionNotFound if the subject was never created -- collections
    # no longer spring into existence on first ingest.
    if file_name in store.pdfs_in(collection):
        report("skipped", 0, 0, f"'{file_name}' is already in '{collection}'. Skipping.")
        return IngestResult(file_name, collection, skipped=True, pages=0, chunks=0)

    chunking = store.chunking_of(collection, settings)

    report("reading", 0, 0, f"Reading '{file_name}'...")
    documents = PyPDFLoader(str(pdf_path)).load()
    pages = len(documents)

    report("splitting", 0, 0, f"{pages} pages. Splitting into chunks of {chunking.chunk_size}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking.chunk_size,
        chunk_overlap=chunking.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["source_pdf"] = file_name

    total = len(chunks)
    report("embedding", 0, total, f"{total} chunks. Embedding in batches of {settings.batch_size}...")

    vector_store = store.open_vectorstore(collection)
    for done, batch in _batches(chunks, settings.batch_size):
        vector_store.add_documents(batch)
        report("embedding", done, total, f"{done}/{total} chunks embedded...")

    report("done", total, total, f"'{file_name}' added to '{collection}'.")
    return IngestResult(file_name, collection, skipped=False, pages=pages, chunks=total)


def _batches(chunks: list, batch_size: int) -> Iterable[tuple[int, list]]:
    """Yield (chunks_done_so_far, batch).

    Batching is not an optimisation: one add_documents call per book overloads
    the Ollama runner on Windows. Do not collapse this into a single call.
    """
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        yield min(start + batch_size, len(chunks)), batch
