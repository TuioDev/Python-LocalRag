# Local RAG over PDFs

Ask questions about your own PDFs, in natural language, on your own machine.
Nothing leaves it: no API key, no cloud, no network beyond a local Ollama server.

PDFs are grouped into **subjects**. Each subject is its own vector collection, so a
question about one book is not answered with text from another. Answers are grounded
in what was actually retrieved -- if it is not in the documents, the app says so
instead of guessing.

## Status

Both interfaces work.

**Web** (`run_web.py`) -- opens a tab on `127.0.0.1:8000`: ask a question and watch
the answer being written, with the sources it used listed before it starts; a panel
showing whether Ollama is up, which models are installed and what each subject
holds; a model picker; subject creation with its own chunk size; and PDF upload
with a progress bar. If Ollama is not running, one button starts it -- and stops it
again, though only ever the server the app started itself. Nothing is served beyond
your own machine.

**Terminal** (`rag.py`) -- the same engine behind an arrow-key menu. Kept as a
fallback. Only run one of the two at a time: the vector store is SQLite.

## How it is put together

- `rag_core/` -- the engine: config, vector store, ingestion, chain, health checks.
  No terminal output; both interfaces drive it.
- `web/` -- FastAPI routes, one HTML page, one JS file, one CSS file. No build step.
- `rag.py` -- the terminal client.
- `run_web.py` -- starts the server and opens the browser.

## Tools

| | |
|---|---|
| **Ollama** | runs the models locally: `nomic-embed-text` for embeddings, `qwen3:14b` for answers |
| **Chroma** | vector store, persisted on disk |
| **LangChain** | retrieval chain and the PDF/text splitting pipeline |
| **pypdf** | reads the PDFs |
| **questionary** | the arrow-key terminal menu |
| **FastAPI + Jinja2 + SSE** | the web interface |

Python 3.12, dependencies managed with `uv`.

## Running

The models have to be pulled. The Ollama server itself does not have to be running
-- the web app can start it for you:

```powershell
ollama pull nomic-embed-text
ollama pull qwen3:14b
.venv\Scripts\python.exe run_web.py
```

That opens the app in your browser. If the panel says Ollama is unreachable, click
**Start Ollama**; it stops again with the same button, or on its own when you close
the app. Create a subject, upload a PDF into it, wait for the bar, then ask away.
`--port` moves it; `--no-browser` leaves the tab alone.

For the terminal version instead, run `.venv\Scripts\python.exe rag.py`. Either way
you can also just drop PDFs straight into `pdfs/` and they show up.
