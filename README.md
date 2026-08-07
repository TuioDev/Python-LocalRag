# Local RAG over PDFs

Ask questions about your own PDFs, in natural language, on your own machine.
Nothing leaves it: no API key, no cloud, no network beyond a local Ollama server.

PDFs are grouped into **subjects**. Each subject is its own vector collection, so a
question about one book is not answered with text from another. Answers are grounded
in what was actually retrieved -- if it is not in the documents, the app says so
instead of guessing.

## Status

Working today: a terminal app (`rag.py`) to create subjects, load PDFs into them and
ask questions.

Being built: a local web interface -- same engine, opened in the browser, with live
streamed answers, the sources on screen, and a status panel for models and
collections.

## How it is put together

- `rag_core/` -- the engine: config, vector store, ingestion, chain, health checks.
  No terminal output; the CLI and the future web server both drive it.
- `rag.py` -- the terminal client.

## Tools

| | |
|---|---|
| **Ollama** | runs the models locally: `nomic-embed-text` for embeddings, `qwen3:14b` for answers |
| **Chroma** | vector store, persisted on disk |
| **LangChain** | retrieval chain and the PDF/text splitting pipeline |
| **pypdf** | reads the PDFs |
| **questionary** | the arrow-key terminal menu |
| **FastAPI + Jinja2 + SSE** | the web interface (in progress) |

Python 3.12, dependencies managed with `uv`.

## Running

Ollama has to be running, with the models pulled:

```powershell
ollama pull nomic-embed-text
ollama pull qwen3:14b
.venv\Scripts\python.exe rag.py
```

Drop your PDFs in `pdfs/`, create a subject, add a PDF to it, then ask away.
