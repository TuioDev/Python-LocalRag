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


@dataclass
class Health:
    url: str
    up: bool
    models: list[str] = field(default_factory=list)
    error: str | None = None

    def has(self, model: str) -> bool:
        """True if the model is installed. Ollama omits the ':latest' suffix
        about as often as it includes it, so compare both ways."""
        wanted = model if ":" in model else f"{model}:latest"
        return any(m == model or m == wanted for m in self.models)


def check(timeout: float = TIMEOUT) -> Health:
    """Ping /api/tags. Never raises: being down is an answer, not a failure."""
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return Health(url=OLLAMA_URL, up=False, error=str(exc))
    models = sorted(m["name"] for m in payload.get("models", []) if m.get("name"))
    return Health(url=OLLAMA_URL, up=True, models=models)


async def acheck(timeout: float = TIMEOUT) -> Health:
    """Same check, for the web server's event loop."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return Health(url=OLLAMA_URL, up=False, error=str(exc))
    models = sorted(m["name"] for m in payload.get("models", []) if m.get("name"))
    return Health(url=OLLAMA_URL, up=True, models=models)


def missing_models(health: Health, settings: Settings) -> list[str]:
    """Models the app needs but Ollama does not have. Empty when it is down --
    there is nothing to report until we can see the list."""
    if not health.up:
        return []
    return [m for m in (EMBED_MODEL, settings.llm_model) if not health.has(m)]
