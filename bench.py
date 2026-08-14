"""
Retrieval benchmark -- how good is the retrieval, in numbers
===========================================================

A thin client over rag_core.evaluate, the way rag.py is a thin client over the
rest of the engine: this file only prints. Every decision about what counts as
a hit lives in rag_core/evaluate.py.

    .venv\\Scripts\\python.exe bench.py                 # every subject with questions
    .venv\\Scripts\\python.exe bench.py patterns        # one subject
    .venv\\Scripts\\python.exe bench.py --check         # validate the set, retrieve nothing

Ollama has to be running -- the question is embedded before anything can be
searched -- but the answering model is never called, so a run takes seconds.

Stop the web server first. chroma_db is SQLite and takes one process at a time.
"""

from __future__ import annotations

import argparse
import sys

from rag_core import evaluate, health, store
from rag_core.config import EMBED_MODEL, ConfigError, load_settings

MISS = "-"


# ---- TABLE ----
def table(headers: list[str], rows: list[list[str]], right: tuple[int, ...] = ()) -> str:
    """A plain ASCII table. No box-drawing characters: the Windows console
    mangles anything outside ASCII, which is why all output here stays in it."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(f" {cell:>{widths[i]}} " if i in right else f" {cell:<{widths[i]}} ")
        return "|" + "|".join(out) + "|"

    return "\n".join([rule, line(headers), rule, *(line(r) for r in rows), rule])


# ---- REPORTING ----
def show_subject_header(result: evaluate.RunResult, stats: store.CollectionStats) -> None:
    chunking = f"chunk {result.chunk_size}/{result.chunk_overlap}"
    if result.legacy_chunking:
        chunking += " (not recorded, assumed)"
    print(f"\nSubject '{result.collection}'")
    print(f"  top_k {result.top_k} | {chunking} | {stats.chunks} chunks | {len(stats.pdfs)} pdfs")
    print(f"  {len(result.scored)} scored questions, {len(result.absent)} absent")


def show_questions(result: evaluate.RunResult) -> None:
    rows = []
    for item in result.results:
        if item.question.scored:
            rank = str(item.hit_rank) if item.hit_rank else MISS
        else:
            rank = "n/a"
        rows.append([item.question.id, item.question.kind, rank, f"{item.seconds:.2f}"])
    print()
    print(table(["question", "kind", "rank", "secs"], rows, right=(2, 3)))


def show_by_kind(result: evaluate.RunResult) -> None:
    """Where the average hides the lesson. The kinds are the axes the set was
    built along, so a technique that helps one and hurts another shows up here
    and nowhere else."""
    k = result.top_k
    rows = []
    for kind in evaluate.KINDS:
        items = [r for r in result.scored if r.question.kind == kind]
        if not items:
            continue
        hits = sum(1 for r in items if r.hit_rank is not None and r.hit_rank <= k)
        mrr = sum(1.0 / r.hit_rank if r.hit_rank else 0.0 for r in items) / len(items)
        rows.append([kind, str(len(items)), f"{hits / len(items):.2f}", f"{mrr:.2f}"])
    if rows:
        print()
        print(table(["kind", "n", f"recall@{k}", "MRR"], rows, right=(1, 2, 3)))


def show_metrics(result: evaluate.RunResult) -> None:
    print()
    for depth in result.depths:
        print(f"  recall@{depth:<3} {result.recall_at(depth):.2f}")
    print(f"  MRR       {result.mrr:.2f}")
    print(f"  mean      {result.mean_seconds:.2f} s per query")


def show_absent(result: evaluate.RunResult) -> None:
    """The two questions with no answer in the corpus. There is nothing to
    score -- the retriever always returns top_k chunks, whether or not any of
    them are relevant -- so this is read, not averaged."""
    absent = result.absent
    if not absent:
        return
    print("\n  Absent questions (nothing in the corpus should answer these):")
    for item in absent:
        print(f"    {item.question.id}: {item.question.text}")
        for got in item.retrieved[:3]:
            page = got.page if got.page is not None else "?"
            print(f"      {got.rank}. {got.pdf[:44]} p.{page}")


# ---- MAIN ----
def main() -> int:
    parser = argparse.ArgumentParser(description="Score retrieval against the golden set.")
    parser.add_argument("collection", nargs="*", help="subjects to run; default is all covered")
    parser.add_argument(
        "--check", action="store_true", help="validate the golden set and stop"
    )
    parser.add_argument("--no-write", action="store_true", help="do not record the run")
    args = parser.parse_args()

    print("Local RAG -- retrieval benchmark")
    print("================================")

    # A benchmark that silently fell back to default settings would report a
    # number for a configuration nobody chose, so this aborts where the CLI
    # warns and carries on.
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"\n  settings.json is unusable: {exc}")
        return 1

    try:
        questions = evaluate.load_questions()
    except evaluate.EvalError as exc:
        print(f"\n  {exc}")
        return 1

    problems = evaluate.check(questions)
    if problems:
        print(f"\n  The golden set does not match the store ({len(problems)} problems):")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print(f"\n  Golden set: {len(questions)} questions, checked against the store.")

    if args.check:
        return 0

    status = health.check()
    if not status.up:
        print(f"\n  Ollama is NOT reachable at {status.url}. The question has to be")
        print("  embedded before anything can be retrieved. Start it and try again.")
        return 1
    if not status.has(EMBED_MODEL):
        print(f"\n  Ollama is up but '{EMBED_MODEL}' is missing. Run: ollama pull {EMBED_MODEL}")
        return 1
    print(f"  Ollama up. Embeddings by '{EMBED_MODEL}'. The answering model is never called.")

    wanted = args.collection or evaluate.collections_covered(questions)
    unknown = [c for c in wanted if not evaluate.for_collection(questions, c)]
    if unknown:
        print(f"\n  No questions for: {', '.join(unknown)}")
        return 1

    for collection in wanted:
        subset = evaluate.for_collection(questions, collection)
        try:
            result = evaluate.run(collection, subset, settings)
            stats = store.stats(collection, settings)
        except (evaluate.EvalError, store.StoreError) as exc:
            print(f"\n  Subject '{collection}' failed: {exc}")
            return 1

        show_subject_header(result, stats)
        show_questions(result)
        show_by_kind(result)
        show_metrics(result)
        show_absent(result)

        if not args.no_write:
            path = evaluate.save_run(result)
            print(f"\n  Recorded: {path.relative_to(evaluate.PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
