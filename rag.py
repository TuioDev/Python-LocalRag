"""
Local RAG over PDFs -- Ollama + Chroma, arrow-key terminal menu
===============================================================

A thin client over rag_core: this file only asks questions and prints answers.
Every decision about the store, the chunking and the chain lives in the engine,
which the web interface drives the same way.

One-off setup:
    ollama pull nomic-embed-text
    ollama pull qwen3:14b

Ollama has to be running. Navigate with the up/down arrows, Enter to choose,
Ctrl+C to leave.
"""

import questionary

from rag_core import chain, health, ingest, store
from rag_core.config import EMBED_MODEL, ConfigError, Settings, load_settings

CANCEL = "<< cancel >>"
BACK = "<< back >>"
NEW_SUBJECT = "+ Create a new subject"


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def show_header(settings):
    """One line about the engine, so a dead Ollama is obvious before you ask."""
    status = health.check()
    if not status.up:
        print(f"\n  Ollama is NOT reachable at {status.url} -- start it before asking anything.")
        return
    missing = health.missing_models(status, settings)
    if missing:
        print(f"\n  Ollama is up, but these models are missing: {', '.join(missing)}")
        print(f"  Run: ollama pull {missing[0]}")
    else:
        print(f"\n  Ollama up. Answering with '{settings.llm_model}', embeddings by '{EMBED_MODEL}'.")


def pick_subject(message, extra_choices=()):
    """Arrow-key subject picker. Returns None when the user backs out."""
    subjects = store.list_collections()
    if not subjects and not extra_choices:
        print("\nNo subjects yet. Create one first.\n")
        return None
    choice = questionary.select(
        message, choices=subjects + list(extra_choices) + [CANCEL]
    ).ask()
    # ask() returns None on Ctrl+C.
    if choice is None or choice == CANCEL:
        return None
    return choice


def ask_int(message, default):
    """questionary.text that insists on a whole number."""
    raw = questionary.text(message, default=str(default)).ask()
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        print(f"'{raw}' is not a whole number.")
        return None


def describe(stats):
    """One line summarising a subject for the menus."""
    chunking = stats.chunking
    size = f"chunks of {chunking.chunk_size}/{chunking.chunk_overlap}"
    if chunking.legacy:
        size = "chunking not recorded (created before it was tracked)"
    return f"{stats.name}: {stats.chunks} chunks, {len(stats.pdfs)} PDF(s), {size}"


# ----------------------------------------------------------------------
# FLOW: ASK QUESTIONS
# ----------------------------------------------------------------------
def flow_ask(settings):
    subject = pick_subject("Which subject do you want to ask about?")
    if subject is None:
        return

    if not store.pdfs_in(subject):
        print(f"\n  Subject '{subject}' is empty. Add a PDF before asking.\n")
        return

    print(f"\nSubject '{subject}'. Type your question (empty to go back).\n")
    while True:
        question = questionary.text("Question>").ask()
        if question is None or not question.strip():
            break
        result = chain.answer(subject, question.strip(), settings)
        print(f"\n{result.text}\n")
        if result.sources:
            print("  Sources:")
            for source in result.sources:
                page = f" p.{source.page}" if source.page else ""
                print(f"    - {source.pdf}{page}")
            print()


# ----------------------------------------------------------------------
# FLOW: CREATE SUBJECT
# ----------------------------------------------------------------------
def flow_create_subject(settings):
    """Chunking belongs to the subject, so it is chosen here -- before the
    first PDF, because every PDF in the subject has to be split the same way."""
    name = questionary.text("Name of the new subject:").ask()
    if name is None or not name.strip():
        print("Cancelled.\n")
        return None

    print("\n  Chunk size is the trade-off: smaller chunks retrieve more precisely,")
    print("  larger ones keep more context around each answer. This cannot be")
    print("  changed later without re-ingesting the PDFs.\n")

    chunk_size = ask_int("Chunk size:", settings.default_chunk_size)
    if chunk_size is None:
        print("Cancelled.\n")
        return None
    chunk_overlap = ask_int("Chunk overlap:", settings.default_chunk_overlap)
    if chunk_overlap is None:
        print("Cancelled.\n")
        return None

    created = store.create(name, chunk_size, chunk_overlap)
    print(f"\nSubject '{created}' created ({chunk_size}/{chunk_overlap}). It is empty; add a PDF next.\n")
    return created


# ----------------------------------------------------------------------
# FLOW: ADD PDF
# ----------------------------------------------------------------------
def flow_add_pdf(settings):
    pdfs = ingest.available_pdfs()
    if not pdfs:
        print("\nNo PDFs found in the pdfs/ folder. Drop your files there and try again.\n")
        return

    chosen = questionary.select(
        "Which PDF do you want to add?", choices=[p.name for p in pdfs] + [CANCEL]
    ).ask()
    if chosen is None or chosen == CANCEL:
        print("Cancelled.\n")
        return
    pdf_path = next(p for p in pdfs if p.name == chosen)

    subject = pick_subject("Add it to which subject?", extra_choices=[NEW_SUBJECT])
    if subject is None:
        print("Cancelled.\n")
        return
    if subject == NEW_SUBJECT:
        subject = flow_create_subject(settings)
        if subject is None:
            return

    ingest.ingest_pdf(pdf_path, subject, settings, on_progress=lambda p: print(f"  {p.message}"))
    print()


# ----------------------------------------------------------------------
# FLOW: DELETE SUBJECT
# ----------------------------------------------------------------------
def flow_delete_subject(settings):
    subject = pick_subject("Which subject do you want to DELETE (everything in it)?")
    if subject is None:
        print("Cancelled.\n")
        return

    # Show what is about to be lost, then default the confirmation to No.
    info = store.stats(subject, settings)
    print(f"\n  {describe(info)}")
    if info.pdfs:
        print(f"  Contains: {', '.join(info.pdfs)}")

    confirmed = questionary.confirm(
        f"Delete subject '{subject}' for good? This cannot be undone.", default=False
    ).ask()
    if confirmed:
        store.delete(subject)
        print(f"Subject '{subject}' deleted.\n")
    else:
        print("Nothing was deleted.\n")


# ----------------------------------------------------------------------
# MAIN MENU
# ----------------------------------------------------------------------
ACTIONS = {
    "Ask questions about a subject": flow_ask,
    "Create a subject": flow_create_subject,
    "Add a PDF to a subject": flow_add_pdf,
    "Delete a subject": flow_delete_subject,
}


def main_menu():
    print("=" * 62)
    print("  Local RAG -- Ollama + Chroma")
    print("  (arrows to navigate, Enter to choose)")
    print("=" * 62)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"\n  settings.json is unusable ({exc}). Falling back to defaults.")
        settings = Settings()

    show_header(settings)

    while True:
        subjects = store.stats_all(settings)
        if subjects:
            print("\nSubjects:")
            for info in subjects:
                print(f"  - {describe(info)}")
        else:
            print("\nNo subjects yet.")

        option = questionary.select(
            "What do you want to do?", choices=list(ACTIONS) + ["Exit"]
        ).ask()

        if option is None or option == "Exit":
            print("See you.")
            break

        try:
            ACTIONS[option](settings)
        except (store.StoreError, ConfigError, FileNotFoundError) as exc:
            print(f"\n  {exc}\n")
        except Exception as exc:  # keep the menu alive; the engine raises, the CLI reports
            print(f"\n  Something went wrong: {exc}\n")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nStopped.")
