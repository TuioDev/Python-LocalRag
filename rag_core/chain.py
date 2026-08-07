"""Retrieval-augmented answering over one subject.

Two ways in, same prompt and same model:

* build_chain() -- the plain LCEL chain, question in, answer out. Use it when
  the sources do not matter.
* answer() / astream() -- retrieve first, hand back the sources, then generate.
  The web UI needs this order: the retriever finishes before the first token,
  so the page can show which PDFs are being used while the answer is still
  being written.

The prompt is deliberately strict. This app is only useful if it refuses to
answer from the model's own memory, so "not in the context" must produce an
admission, not a guess.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from . import store
from .config import OLLAMA_URL, Settings

PROMPT = ChatPromptTemplate.from_template(
    "You answer ONLY from the context below.\n"
    "If the answer is not in the context, say you could not find it.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)

SNIPPET_CHARS = 300


@dataclass
class Source:
    pdf: str
    page: str | None
    snippet: str


@dataclass
class Answer:
    text: str
    sources: list[Source]


@dataclass
class SourcesEvent:
    """Emitted once, before any token."""

    sources: list[Source]


@dataclass
class TokenEvent:
    text: str


StreamEvent = SourcesEvent | TokenEvent


# ---- PIECES ----
def llm(settings: Settings) -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=OLLAMA_URL,
        temperature=settings.temperature,
    )


def retriever(collection: str, settings: Settings):
    return store.open_vectorstore(collection).as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def to_sources(docs: list[Document]) -> list[Source]:
    """One entry per retrieved chunk: which PDF, which page, and a taste of it."""
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        page = meta.get("page_label")
        if page is None and meta.get("page") is not None:
            page = str(int(meta["page"]) + 1)  # PyPDFLoader counts from zero
        snippet = " ".join(doc.page_content.split())[:SNIPPET_CHARS]
        sources.append(Source(pdf=meta.get("source_pdf", "?"), page=page, snippet=snippet))
    return sources


# ---- ENTRY POINTS ----
def build_chain(collection: str, settings: Settings):
    """question -> answer text, retrieval included."""
    return (
        {"context": retriever(collection, settings) | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm(settings)
        | StrOutputParser()
    )


def retrieve(collection: str, question: str, settings: Settings) -> list[Document]:
    return retriever(collection, settings).invoke(question)


def answer(collection: str, question: str, settings: Settings) -> Answer:
    """Blocking answer, sources included."""
    docs = retrieve(collection, question, settings)
    chain = PROMPT | llm(settings) | StrOutputParser()
    text = chain.invoke({"context": format_docs(docs), "question": question})
    return Answer(text=text, sources=to_sources(docs))


async def astream(collection: str, question: str, settings: Settings) -> AsyncIterator[StreamEvent]:
    """Sources first, then the answer token by token."""
    # Retrieval is synchronous and hits SQLite plus the embedding model; keep it
    # off the event loop so the server stays responsive.
    docs = await asyncio.to_thread(retrieve, collection, question, settings)
    yield SourcesEvent(sources=to_sources(docs))

    chain = PROMPT | llm(settings) | StrOutputParser()
    async for token in chain.astream({"context": format_docs(docs), "question": question}):
        if token:
            yield TokenEvent(text=token)
