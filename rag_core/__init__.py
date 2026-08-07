"""Engine of the local RAG app: config, store, ingestion, chain, health.

No terminal I/O lives here. Nothing in this package prints, prompts or exits;
progress is reported through callbacks and failures are raised. That is what
lets the CLI and the web server drive the exact same code.
"""

from .config import EMBED_MODEL, OLLAMA_URL, PDF_DIR, PERSIST_DIR, Settings, load_settings, save_settings

__all__ = [
    "EMBED_MODEL",
    "OLLAMA_URL",
    "PDF_DIR",
    "PERSIST_DIR",
    "Settings",
    "load_settings",
    "save_settings",
]
