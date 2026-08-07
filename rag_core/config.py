"""Configuration in three layers.

Where a value lives decides who is allowed to change it:

* Locked -- EMBED_MODEL, OLLAMA_URL. Plain constants here. Changing the
  embedding model would invalidate every collection already indexed, because
  vectors written by one model are meaningless to another, so the interface
  shows it read-only.
* Global editable -- the Settings dataclass, persisted in settings.json.
  Editable at runtime; a change takes effect on the next question.
* Per collection -- chunk_size / chunk_overlap / embed_model, written into the
  Chroma collection metadata when the collection is created and never touched
  again. The two default_chunk_* fields below are only the values the creation
  form starts from; they do not affect a collection that already exists.

Paths resolve from the project root rather than the working directory, so the
CLI and the web server find the same store no matter where they are started.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "pdfs"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
SETTINGS_PATH = PROJECT_ROOT / "settings.json"

# ---- LOCKED LAYER ----
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://127.0.0.1:11434"


class ConfigError(RuntimeError):
    """settings.json is unreadable, or a value in it is out of range."""


# ---- GLOBAL EDITABLE LAYER ----
@dataclass
class Settings:
    """Everything the interface may change, persisted in settings.json."""

    llm_model: str = "qwen3:14b"
    temperature: float = 0.2
    top_k: int = 4
    # Chunks embedded per call. Sending a whole book at once overloads the
    # Ollama runner on Windows, so keep the batching.
    batch_size: int = 50
    # Suggestions for the "create collection" form only.
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200

    def validate(self) -> None:
        """Raise ConfigError if any value would break the app downstream."""
        if not self.llm_model.strip():
            raise ConfigError("llm_model cannot be empty.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(f"temperature must be between 0.0 and 2.0, got {self.temperature}.")
        if self.top_k < 1:
            raise ConfigError(f"top_k must be at least 1, got {self.top_k}.")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be at least 1, got {self.batch_size}.")
        validate_chunking(self.default_chunk_size, self.default_chunk_overlap)

    def replace(self, **changes: object) -> "Settings":
        """Return a validated copy with some fields changed.

        Used by the web PATCH route: build the candidate, validate it, and only
        then persist -- a rejected change never reaches settings.json.
        """
        known = {f.name for f in fields(Settings)}
        unknown = set(changes) - known
        if unknown:
            raise ConfigError(f"Unknown setting(s): {', '.join(sorted(unknown))}.")
        candidate = Settings(**{**asdict(self), **changes})
        candidate.validate()
        return candidate


def validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    """Shared by the global defaults and by collection creation."""
    if chunk_size < 100:
        raise ConfigError(f"chunk_size must be at least 100, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ConfigError(f"chunk_overlap cannot be negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ConfigError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})."
        )


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    """Read settings.json, falling back to the defaults when it does not exist.

    Unknown keys are ignored so an old file survives a new field. A corrupt or
    out-of-range file raises instead of being silently discarded -- silently
    reverting to defaults would look like the app ignoring the user's choice.
    """
    if not path.exists():
        return Settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name} must contain a JSON object.")

    known = {f.name for f in fields(Settings)}
    try:
        settings = Settings(**{k: v for k, v in raw.items() if k in known})
    except TypeError as exc:
        raise ConfigError(f"Invalid value in {path.name}: {exc}") from exc
    settings.validate()
    return settings


def save_settings(settings: Settings, path: Path = SETTINGS_PATH) -> None:
    """Persist the global layer. Validates first: never write a broken file."""
    settings.validate()
    path.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")
