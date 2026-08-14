"""Score retrieval against a hand-written golden set.

The question this answers is deliberately narrow: **did the right page come
back?** Nothing here calls the LLM. Retrieval is the ceiling on everything the
model can do -- what the retriever misses, no amount of generation quality can
recover -- and leaving the model out is what lets a whole run finish in seconds
instead of minutes.

Ground truth is (pdf, page), never a chunk id. Chunk ids are reassigned on
every re-ingest, and comparing two chunkings of the same book is exactly what a
later stage does; a page number survives re-chunking, a chunk id does not.

Same rule as the rest of rag_core: nothing here prints. Results come back as
data, progress goes out through a callback, and a broken golden set is reported
as problems the caller can show rather than swallowed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from . import chain, store
from .config import EMBED_MODEL, PROJECT_ROOT, Settings

EVALS_DIR = PROJECT_ROOT / "evals"
QUESTIONS_PATH = EVALS_DIR / "questions.json"
RUNS_DIR = EVALS_DIR / "runs"

# A retrieved chunk counts as a hit when it lands within this many pages of the
# expected one. The window is not sloppiness: chunks straddle page boundaries
# by construction, so demanding an exact page would punish the retriever for
# the splitter's arbitrary cut.
PAGE_WINDOW = 1

# The depths the summary reports. Anything above the run's top_k is dropped,
# since a rank that deep never existed.
RECALL_DEPTHS = (1, 4, 8, 10, 20)

# jargon    -- the book's own vocabulary; dense retrieval should ace these
# paraphrase-- deliberately shares almost no words with the page
# spanning  -- the answer runs across two pages
# specific  -- hinges on a proper noun or a number
# absent    -- not in the corpus at all; scored by nothing, read by a human
KINDS = ("jargon", "paraphrase", "spanning", "specific", "absent")

SNIPPET_CHARS = 160


class EvalError(RuntimeError):
    """The golden set is unreadable, malformed, or does not match the store."""


# ---- THE GOLDEN SET ----
@dataclass(frozen=True)
class Expected:
    """One page that answers a question.

    `page` is the raw 0-based value PyPDFLoader writes into metadata["page"],
    not the number printed on the paper. It is stored rather than page_label
    because it is guaranteed present and guaranteed an integer.
    """

    pdf: str
    page: int


@dataclass
class Question:
    id: str
    collection: str
    kind: str
    text: str
    expected: list[Expected]
    evidence: str

    @property
    def scored(self) -> bool:
        """False for the absent questions: no right answer, so no rank."""
        return bool(self.expected)


@dataclass
class Retrieved:
    rank: int  # 1-based, the order the retriever returned them in
    pdf: str
    page: int | None
    snippet: str


@dataclass
class QuestionResult:
    question: Question
    retrieved: list[Retrieved]
    # Rank of the first retrieved chunk matching any expected page, or None.
    hit_rank: int | None
    seconds: float


@dataclass
class RunResult:
    collection: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    legacy_chunking: bool
    embed_model: str
    started_at: str
    results: list[QuestionResult]

    # ---- metrics ----
    @property
    def scored(self) -> list[QuestionResult]:
        return [r for r in self.results if r.question.scored]

    @property
    def absent(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.question.scored]

    def recall_at(self, k: int) -> float:
        """Share of questions whose expected page came back within the top k.

        A question counts as a hit when *any* of its expected pages is found.
        For the questions whose answer spans two pages that is the honest bar:
        with a +/-1 window, one chunk on page 100 already satisfies both 100
        and 101, so demanding every expected page would measure nothing.
        """
        scored = self.scored
        if not scored:
            return 0.0
        found = sum(1 for r in scored if r.hit_rank is not None and r.hit_rank <= k)
        return found / len(scored)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank: separates 'found it first' from 'found it last'."""
        scored = self.scored
        if not scored:
            return 0.0
        return sum(1.0 / r.hit_rank if r.hit_rank else 0.0 for r in scored) / len(scored)

    @property
    def mean_seconds(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.seconds for r in self.results) / len(self.results)

    @property
    def depths(self) -> tuple[int, ...]:
        return tuple(d for d in RECALL_DEPTHS if d <= self.top_k)


ProgressCallback = Callable[[int, int, Question], None]


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Read the golden set. Raises rather than skipping a malformed entry."""
    if not path.exists():
        raise EvalError(f"No golden set at {path}. Stage 7 needs evals/questions.json.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(raw, list):
        raise EvalError(f"{path.name} must contain a JSON list of questions.")

    questions = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise EvalError(f"Question {index} in {path.name} is not an object.")
        try:
            expected = [
                Expected(pdf=str(e["pdf"]), page=int(e["page"]))
                for e in item.get("expected", [])
            ]
            question = Question(
                id=str(item["id"]),
                collection=str(item["collection"]),
                kind=str(item["kind"]),
                text=str(item["question"]),
                expected=expected,
                evidence=str(item.get("evidence", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvalError(f"Question {index} in {path.name} is malformed: {exc}") from exc
        if question.kind not in KINDS:
            raise EvalError(
                f"Question '{question.id}' has kind '{question.kind}'; "
                f"expected one of {', '.join(KINDS)}."
            )
        questions.append(question)
    return questions


def for_collection(questions: list[Question], collection: str) -> list[Question]:
    return [q for q in questions if q.collection == collection]


def collections_covered(questions: list[Question]) -> list[str]:
    return sorted({q.collection for q in questions})


# ---- CHECKING THE SET AGAINST THE STORE ----
def check(questions: list[Question]) -> list[str]:
    """Report every way the golden set disagrees with what is in the store.

    Returned as data, the way health.check() reports Ollama being down: a
    question pointing at a page that produced no chunks is a mistake in the
    question, and scoring it as a retrieval miss would quietly lower the number
    for a reason that has nothing to do with retrieval.
    """
    problems: list[str] = []

    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            problems.append(f"Duplicate question id '{question.id}'.")
        seen.add(question.id)
        if not question.text.strip():
            problems.append(f"'{question.id}' has an empty question.")

    existing = set(store.list_collections())
    for collection in collections_covered(questions):
        if collection not in existing:
            problems.append(f"Subject '{collection}' does not exist in the store.")
            continue

        pages = store.pages_in(collection)
        for question in for_collection(questions, collection):
            for expected in question.expected:
                if expected.pdf not in pages:
                    problems.append(
                        f"'{question.id}' expects '{expected.pdf}', which is not in "
                        f"'{collection}'."
                    )
                    continue
                window = range(expected.page - PAGE_WINDOW, expected.page + PAGE_WINDOW + 1)
                if not any(page in pages[expected.pdf] for page in window):
                    problems.append(
                        f"'{question.id}' expects page {expected.page} of "
                        f"'{expected.pdf}', which produced no chunks. No retriever "
                        f"can ever return it."
                    )
    return problems


# ---- RUNNING ----
def _to_retrieved(docs: list[Document]) -> list[Retrieved]:
    items = []
    for rank, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        page = meta.get("page")
        items.append(
            Retrieved(
                rank=rank,
                pdf=str(meta.get("source_pdf", "?")),
                page=int(page) if page is not None else None,
                snippet=" ".join((doc.page_content or "").split())[:SNIPPET_CHARS],
            )
        )
    return items


def _hit_rank(retrieved: list[Retrieved], expected: list[Expected]) -> int | None:
    for item in retrieved:
        if item.page is None:
            continue
        for want in expected:
            if item.pdf == want.pdf and abs(item.page - want.page) <= PAGE_WINDOW:
                return item.rank
    return None


def run(
    collection: str,
    questions: list[Question],
    settings: Settings,
    on_progress: ProgressCallback | None = None,
) -> RunResult:
    """Retrieve for every question and score the result.

    Goes through chain.retrieve(), the same call the app makes, on purpose: a
    harness that measured its own private code path would report a number the
    app never earns.
    """
    if not questions:
        raise EvalError(f"No questions for subject '{collection}'.")

    chunking = store.chunking_of(collection, settings)

    # One throwaway retrieval first. The embedding model is loaded lazily by
    # Ollama, so without this the first question absorbs several seconds of
    # model load and the latency column lies about it.
    chain.retrieve(collection, "warmup", settings)

    results = []
    for index, question in enumerate(questions, start=1):
        if on_progress is not None:
            on_progress(index, len(questions), question)
        start = time.perf_counter()
        docs = chain.retrieve(collection, question.text, settings)
        seconds = time.perf_counter() - start
        retrieved = _to_retrieved(docs)
        results.append(
            QuestionResult(
                question=question,
                retrieved=retrieved,
                hit_rank=_hit_rank(retrieved, question.expected) if question.scored else None,
                seconds=seconds,
            )
        )

    return RunResult(
        collection=collection,
        top_k=settings.top_k,
        chunk_size=chunking.chunk_size,
        chunk_overlap=chunking.chunk_overlap,
        legacy_chunking=chunking.legacy,
        embed_model=EMBED_MODEL,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        results=results,
    )


# ---- RECORDING ----
def to_dict(result: RunResult) -> dict:
    """The shape written to evals/runs/. A number is only interesting next to
    the one before it, so everything needed to compare two runs goes in here:
    the retrieval knobs, the chunking, and the embedding model."""
    return {
        "started_at": result.started_at,
        "collection": result.collection,
        "config": {
            "top_k": result.top_k,
            "chunk_size": result.chunk_size,
            "chunk_overlap": result.chunk_overlap,
            "legacy_chunking": result.legacy_chunking,
            "embed_model": result.embed_model,
            "page_window": PAGE_WINDOW,
        },
        "metrics": {
            "questions_scored": len(result.scored),
            "questions_absent": len(result.absent),
            **{f"recall@{d}": round(result.recall_at(d), 4) for d in result.depths},
            "mrr": round(result.mrr, 4),
            "mean_seconds": round(result.mean_seconds, 4),
        },
        "questions": [
            {
                "id": item.question.id,
                "kind": item.question.kind,
                "question": item.question.text,
                "expected": [{"pdf": e.pdf, "page": e.page} for e in item.question.expected],
                "hit_rank": item.hit_rank,
                "seconds": round(item.seconds, 4),
                # Snippets stay out: they would dominate the diff between two
                # runs, and (pdf, page) is enough to see what came back instead.
                "retrieved": [
                    {"rank": r.rank, "pdf": r.pdf, "page": r.page} for r in item.retrieved
                ],
            }
            for item in result.results
        ],
    }


def save_run(result: RunResult, runs_dir: Path = RUNS_DIR) -> Path:
    """Write one run, named so the directory sorts chronologically."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = runs_dir / f"{stamp}-{result.collection}.json"
    path.write_text(json.dumps(to_dict(result), indent=2) + "\n", encoding="utf-8")
    return path
