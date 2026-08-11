"""Start and stop the local Ollama server.

health.py answers "is it answering?". This module answers the next question:
"can I make it answer?" -- because being told the server is down is only useful
if there is something you can do about it without leaving the app.

Two rules keep this honest:

* **Only what we started.** A server launched from the tray icon, or from a
  terminal running `ollama serve`, belongs to whoever started it; this module
  refuses to kill it and the interface says so instead of pretending the button
  is broken. The one process it will stop is the child it spawned itself.
* **What we start, we stop.** An `ollama serve` with no console window and no
  tray icon is a process the user has no obvious way to find, so the web server
  shuts its child down on the way out (see the lifespan hook in web/server.py).

The command is never taken from a request: the executable is resolved here,
from PATH or from the known install directories, and the only argument is
`serve`. Nothing a browser sends can turn this into "run anything".
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT

# Ollama's own output, truncated on every start. When a start fails, the reason
# is in here and nowhere else: the port already taken, a missing runtime, a GPU
# driver refusing to load.
LOG_PATH = PROJECT_ROOT / "ollama-serve.log"
LOG_TAIL_LINES = 12
STOP_TIMEOUT = 10.0

_WINDOWS = os.name == "nt"

# Where the Windows installer puts it when the shell has not been restarted and
# PATH is still the one from before the install.
_INSTALL_DIRS = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama",
    Path(os.environ.get("PROGRAMFILES", "")) / "Ollama",
)
_BINARY = "ollama.exe" if _WINDOWS else "ollama"


class OllamaControlError(RuntimeError):
    """Base class for anything wrong with starting or stopping the server."""


class ExecutableNotFound(OllamaControlError):
    pass


class AlreadyRunning(OllamaControlError):
    pass


class NotManaged(OllamaControlError):
    """Asked to stop a server this process did not start."""


class StartFailed(OllamaControlError):
    pass


# The server this process started, if any. Module state for the same reason
# store.client() is cached: there is one of these per process, and the web
# server runs a single worker.
_process: subprocess.Popen | None = None


# ---- WHAT IS THERE ----
def executable() -> Path | None:
    """The ollama program, or None if this machine has no visible copy."""
    found = shutil.which("ollama")
    if found:
        return Path(found)
    for folder in _INSTALL_DIRS:
        candidate = folder / _BINARY
        if candidate.is_file():
            return candidate
    return None


def managed() -> subprocess.Popen | None:
    """The server we started, if it is still alive.

    The poll() is what keeps this honest: a server that crashed on its own must
    stop counting as ours, or Stop would go on offering to kill a pid that is
    already gone.
    """
    global _process
    if _process is not None and _process.poll() is not None:
        _process = None
    return _process


@dataclass
class Control:
    """Everything the interface needs to decide what the button says."""

    executable: str | None
    managed: bool
    pid: int | None


def control() -> Control:
    process = managed()
    found = executable()
    return Control(
        executable=str(found) if found else None,
        managed=process is not None,
        pid=process.pid if process is not None else None,
    )


def log_tail(lines: int = LOG_TAIL_LINES) -> str:
    """The last few lines Ollama wrote, for when a start fails silently."""
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


# ---- STARTING AND STOPPING ----
def _spawn_options() -> dict:
    """Keep the child out of our console and out of our process group.

    Without CREATE_NO_WINDOW a console flashes up on every start, and without
    its own process group a Ctrl+C aimed at this server would travel to Ollama
    as well -- which would stop the model server every time you stop the app,
    whether or not that was the intention.
    """
    if _WINDOWS:
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def start() -> subprocess.Popen:
    """Spawn `ollama serve` and return the child immediately.

    It does not wait for the server to answer: readiness is a poll against
    /api/tags, and the caller that owns an event loop is the one that can do
    that without blocking everything else. Raises rather than returning a dead
    process, so a caller cannot mistake "nothing to run" for "starting".
    """
    if managed() is not None:
        raise AlreadyRunning("Ollama is already running here; this app started it.")

    binary = executable()
    if binary is None:
        raise ExecutableNotFound(
            "Could not find the 'ollama' program on PATH or in the usual install "
            "folder. Install Ollama, then refresh."
        )

    # A log we cannot open is not a reason to refuse to start; it only costs the
    # explanation if the start then fails.
    try:
        log = LOG_PATH.open("wb")
    except OSError:
        log = None

    try:
        process = subprocess.Popen(
            [str(binary), "serve"],
            stdin=subprocess.DEVNULL,
            stdout=log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            **_spawn_options(),
        )
    except OSError as exc:
        raise StartFailed(f"Could not start '{binary}': {exc}") from exc
    finally:
        # The child inherited its own handle; ours is of no further use.
        if log is not None:
            log.close()

    global _process
    _process = process
    return process


def stop(timeout: float = STOP_TIMEOUT) -> int:
    """Stop the server this app started, and return the pid it stopped.

    Refuses anything else. Stopping a server somebody else started would mean
    reaching outside this app to kill a process it knows nothing about, and the
    tray icon that launched it would be left claiming to be running.
    """
    process = managed()
    if process is None:
        raise NotManaged(
            "This app did not start Ollama, so it will not stop it. Stop it where "
            "you started it: the tray icon, or the terminal running 'ollama serve'."
        )

    pid = process.pid
    _kill_tree(process, timeout)
    global _process
    _process = None
    return pid


def _kill_tree(process: subprocess.Popen, timeout: float) -> None:
    """Take the runners down with the server.

    `ollama serve` spawns one runner process per loaded model. Ending the parent
    on its own would leave those behind holding several gigabytes of VRAM with
    nothing left to talk to them.
    """
    if _WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        process.terminate()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
