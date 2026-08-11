"""Is Ollama up, and which models does it have?

Feeds the status panel and the model picker. It also answers the question the
old CLI could not: when nothing works, is the server down or is the store
empty? Those looked identical before, because every error was swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from .config import EMBED_MODEL, OLLAMA_URL, Settings

TIMEOUT = 3.0


def same_model(installed: str, wanted: str) -> bool:
    """Compare two model names. Ollama omits the ':latest' suffix about as often
    as it includes it, so compare both ways."""
    return installed == wanted or installed == (wanted if ":" in wanted else f"{wanted}:latest")


@dataclass
class Health:
    url: str
    up: bool
    models: list[str] = field(default_factory=list)
    # The subset of models that can generate text. An embedding model is
    # installed and perfectly healthy and still cannot answer a question, so
    # offering one in the answering-model picker only invites a broken setting.
    chat_models: list[str] = field(default_factory=list)
    error: str | None = None

    def has(self, model: str) -> bool:
        """True if the model is installed."""
        return any(same_model(m, model) for m in self.models)

    def can_chat(self, model: str) -> bool:
        """True if the model can generate an answer rather than only vectors."""
        return any(same_model(m, model) for m in self.chat_models)


def _read_models(payload: dict) -> tuple[list[str], list[str]]:
    """Split /api/tags into everything installed and everything that can answer.

    Ollama reports a capability list per model: nomic-embed-text says
    ["embedding"] and cannot emit a single token, while a chat model says
    ["completion", ...]. Older servers leave the field out entirely, and there
    the only thing known for certain is that EMBED_MODEL is not a chat model.
    """
    installed: list[str] = []
    chat: list[str] = []
    for entry in payload.get("models", []):
        name = entry.get("name")
        if not name:
            continue
        installed.append(name)
        capabilities = entry.get("capabilities")
        if capabilities is None:
            answers = not same_model(name, EMBED_MODEL)
        else:
            answers = "completion" in capabilities
        if answers:
            chat.append(name)
    return sorted(installed), sorted(chat)


def check(timeout: float = TIMEOUT) -> Health:
    """Ping /api/tags. Never raises: being down is an answer, not a failure."""
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return Health(url=OLLAMA_URL, up=False, error=str(exc))
    models, chat_models = _read_models(payload)
    return Health(url=OLLAMA_URL, up=True, models=models, chat_models=chat_models)


async def acheck(timeout: float = TIMEOUT) -> Health:
    """Same check, for the web server's event loop."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return Health(url=OLLAMA_URL, up=False, error=str(exc))
    models, chat_models = _read_models(payload)
    return Health(url=OLLAMA_URL, up=True, models=models, chat_models=chat_models)


def missing_models(health: Health, settings: Settings) -> list[str]:
    """Models the app needs but Ollama does not have. Empty when it is down --
    there is nothing to report until we can see the list."""
    if not health.up:
        return []
    return [m for m in (EMBED_MODEL, settings.llm_model) if not health.has(m)]
