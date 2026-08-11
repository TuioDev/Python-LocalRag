"""FastAPI over rag_core: the same engine the CLI drives, reachable from a tab.

Every route here is a thin translation between HTTP and the engine. The rules
themselves are not restated: a bad subject name raises InvalidCollectionName in
store.py, and this module only decides that it means 400. Keeping the line in
that place is what stops the browser and the terminal from slowly growing two
different sets of behaviour.

Two constraints shape the whole file:

* chroma_db is SQLite, so exactly one process may write it. Hence a single
  uvicorn worker, and ingestions serialised behind one lock. Running the CLI at
  the same time is what corrupts the store.
* There is no authentication, so run_web.py binds 127.0.0.1 and nothing else.

Everything the engine does is blocking -- SQLite reads, PDF parsing, embedding
calls -- so the routes hand it to a thread instead of stalling the event loop,
which would freeze the stream of whatever answer is being written at the time.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import AsyncIterator

from chromadb.errors import ChromaError
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict
from sse_starlette.sse import EventSourceResponse

from rag_core import chain, health, ingest, ollama_server, store
from rag_core.config import (
    EMBED_MODEL,
    OLLAMA_URL,
    PDF_DIR,
    ConfigError,
    Settings,
    load_settings,
    save_settings,
)

WEB_DIR = Path(__file__).resolve().parent

EMBED_LOCK_REASON = (
    "Vectors written by one embedding model mean nothing to another. Changing "
    "this would silently turn every existing subject into noise, so it can only "
    "change in code, with a re-ingest."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """On the way out, stop the Ollama we started.

    Only that one. An `ollama serve` spawned by the Start button has no console
    window and no tray icon, so leaving it behind would leave the user with a
    model server they have no obvious way to find, let alone stop. A server that
    was already running when the app opened is untouched: it was never ours.
    """
    yield
    if ollama_server.managed() is not None:
        await run_blocking(ollama_server.stop)


# "no-cache" is badly named: it does not mean "keep no copy", it means "ask me
# before you use the copy you have". Without it the browser is left to guess how
# long app.js stays good for, and it guesses generously -- which is how a page
# ends up running the JavaScript from before the last change while the server
# happily serves the new one. With it, every load revalidates: unchanged files
# come back as an empty 304 and the browser reuses what it already had.
NO_CACHE = {"cache-control": "no-cache"}


class RevalidatedStatic(StaticFiles):
    """StaticFiles that asks the browser to check back every time.

    The header goes on whatever came out, 200 or 304, because the not-modified
    reply is built inside the parent before this can reach it -- and a 304
    without the header would teach the browser to stop asking again.
    """

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE)
        return response


# docs_url is off on purpose: the Swagger page pulls its assets from a CDN, and
# an app whose whole point is running offline should not ship a route that only
# works online.
app = FastAPI(title="Local RAG", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", RevalidatedStatic(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


# ---- STATE ----
@dataclass
class ServerState:
    """What the process keeps between requests.

    Only the global editable layer lives here. Collections are read from Chroma
    every time, because the CLI may have changed them and a cached list would
    start lying.
    """

    settings: Settings
    settings_error: str | None = None


def _initial_state() -> ServerState:
    """Load settings.json the way the CLI does: report it, do not hide it.

    A corrupt file must not stop the server -- you would have no interface left
    to fix it with -- but it must stay visible in the panel instead of being
    quietly replaced by defaults.
    """
    try:
        return ServerState(settings=load_settings())
    except ConfigError as exc:
        return ServerState(settings=Settings(), settings_error=str(exc))


state = _initial_state()


# ---- ERROR TRANSLATION ----
def _status_for(exc: Exception) -> int:
    if isinstance(exc, store.CollectionNotFound):
        return 404
    if isinstance(exc, store.CollectionExists):
        return 409
    if isinstance(exc, (store.InvalidCollectionName, ConfigError)):
        return 400
    # Ollama already running, or running under somebody else: a conflict with
    # the state of the machine, not a bad request.
    if isinstance(exc, (ollama_server.AlreadyRunning, ollama_server.NotManaged)):
        return 409
    if isinstance(exc, ollama_server.ExecutableNotFound):
        return 503
    return 500


@app.exception_handler(store.StoreError)
@app.exception_handler(ConfigError)
@app.exception_handler(ollama_server.OllamaControlError)
async def _engine_error(request: Request, exc: Exception) -> JSONResponse:
    """The engine raises, the interface reports -- here, as JSON."""
    return JSONResponse({"error": str(exc)}, status_code=_status_for(exc))


@app.exception_handler(ChromaError)
@app.exception_handler(sqlite3.OperationalError)
async def _store_busy(request: Request, exc: Exception) -> JSONResponse:
    """A locked store is the one failure with an obvious cause worth naming."""
    if "lock" in str(exc).lower():
        return JSONResponse(
            {
                "error": (
                    "The vector store is locked. Chroma is SQLite and takes one "
                    "process at a time -- close the CLI (rag.py) and try again."
                )
            },
            status_code=503,
        )
    return JSONResponse({"error": f"The vector store failed: {exc}"}, status_code=500)


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
    """Every failure leaves through the same door, so the page has one place to
    look for the message."""
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}" for err in exc.errors()
    )
    return JSONResponse({"error": f"Invalid request. {problems}"}, status_code=422)


# ---- PAYLOADS ----
async def run_blocking(func, *args):
    """Run engine work off the event loop.

    Everything in rag_core blocks: SQLite, pypdf, the embedding calls. Left on
    the loop they would stall every other request, including the tokens of an
    answer already being streamed to the page.
    """
    return await asyncio.to_thread(func, *args)



def settings_payload() -> dict:
    """The three config layers, told apart so the page can grey out layer one."""
    return {
        "editable": asdict(state.settings),
        "locked": {
            "embed_model": EMBED_MODEL,
            "ollama_url": OLLAMA_URL,
            "reason": EMBED_LOCK_REASON,
        },
        "error": state.settings_error,
    }


def collections_payload() -> list[dict]:
    """Blocking: reads SQLite and every chunk's metadata. Call it in a thread."""
    return [asdict(stats) for stats in store.stats_all(state.settings)]


# ---- PAGE ----
@app.get("/")
async def index(request: Request):
    # The page carries the ids app.js reaches for, so a stale copy of it breaks
    # the same way a stale script does.
    return templates.TemplateResponse(request, "index.html", headers=NO_CACHE)


# ---- STATUS ----
@app.get("/api/status")
async def api_status(request: Request) -> dict:
    """Everything the panel needs in one round trip.

    Ollama being down is reported, not raised: the subjects and their sizes are
    read straight from Chroma and stay perfectly legible with no model server
    running at all.
    """
    ollama = await health.acheck()
    return {
        "ollama": asdict(ollama),
        "control": asdict(ollama_server.control()),
        "missing_models": health.missing_models(ollama, state.settings),
        "settings": settings_payload(),
        "collections": await run_blocking(collections_payload),
    }


@app.get("/api/collections")
async def api_collections() -> list[dict]:
    return await run_blocking(collections_payload)


# ---- OLLAMA PROCESS ----
# Being told the server is down is only half an answer, so the panel can start
# it. What it will not do is stop a server it did not start -- see the two rules
# in rag_core/ollama_server.py.
OLLAMA_LOCK = asyncio.Lock()
OLLAMA_READY_TIMEOUT = 60.0
OLLAMA_POLL_SECONDS = 0.4


def ollama_payload(ollama: health.Health) -> dict:
    """What the page needs to redraw the pill and the button together."""
    return {"ollama": asdict(ollama), "control": asdict(ollama_server.control())}


@app.post("/api/ollama/start")
async def api_ollama_start() -> dict:
    """Start the model server and wait until it actually answers.

    Waiting is the point. Returning as soon as the process exists would report
    success for a server that is still binding its port, or that is about to die
    on one already taken -- and the page would then blame the first question.
    """
    if OLLAMA_LOCK.locked():
        raise HTTPException(409, "Ollama is already being started or stopped. Give it a moment.")

    async with OLLAMA_LOCK:
        current = await health.acheck()
        if current.up:
            raise HTTPException(409, f"Ollama is already answering at {current.url}.")
        process = await run_blocking(ollama_server.start)
        return ollama_payload(await wait_until_up(process))


async def wait_until_up(process: subprocess.Popen) -> health.Health:
    """Poll /api/tags until the new server answers, dies, or runs out of time.

    The poll() is the fast path: a server that cannot bind its port is gone in
    well under a second, and there is no reason to sit through the whole timeout
    when the child is already dead and its log says why.
    """
    deadline = time.monotonic() + OLLAMA_READY_TIMEOUT
    while True:
        current = await health.acheck(timeout=2.0)
        if current.up:
            return current
        if process.poll() is not None:
            tail = await run_blocking(ollama_server.log_tail)
            detail = f" It said:\n{tail}" if tail else ""
            raise HTTPException(500, f"Ollama started and stopped immediately.{detail}")
        if time.monotonic() >= deadline:
            raise HTTPException(
                504,
                f"Ollama did not answer within {int(OLLAMA_READY_TIMEOUT)} seconds. "
                "It may still be starting -- hit Refresh in a moment.",
            )
        await asyncio.sleep(OLLAMA_POLL_SECONDS)


@app.post("/api/ollama/stop")
async def api_ollama_stop() -> dict:
    """Stop the server this app started. NotManaged -> 409, and the page says
    where the running one actually came from."""
    if OLLAMA_LOCK.locked():
        raise HTTPException(409, "Ollama is already being started or stopped. Give it a moment.")

    async with OLLAMA_LOCK:
        await run_blocking(ollama_server.stop)
        return ollama_payload(await health.acheck())


# ---- SETTINGS ----
class SettingsPatch(BaseModel):
    """Any subset of the editable layer; whatever is left out stays as it was."""

    model_config = ConfigDict(extra="forbid")

    llm_model: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    batch_size: int | None = None
    default_chunk_size: int | None = None
    default_chunk_overlap: int | None = None


@app.patch("/api/settings")
async def api_patch_settings(patch: SettingsPatch) -> dict:
    """Change the global layer, taking effect on the next question.

    Settings.replace() validates a candidate before anything is written, so a
    rejected value never reaches settings.json.
    """
    changes = patch.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(400, "Nothing to change.")

    wanted_model = changes.get("llm_model")
    if wanted_model is not None and wanted_model != state.settings.llm_model:
        await require_installed(wanted_model)

    candidate = state.settings.replace(**changes)
    await run_blocking(save_settings, candidate)
    state.settings = candidate
    state.settings_error = None
    return settings_payload()


async def require_installed(model: str) -> None:
    """Refuse a model Ollama does not have, or that cannot answer at all.

    A typo saved now would look like it worked and only fail at the next
    question, pointing at the chat instead of at the setting that broke it. An
    embedding model is worse still: it is installed, so nothing looks wrong
    until every answer comes back empty.
    """
    ollama = await health.acheck()
    if not ollama.up:
        raise HTTPException(
            503, f"Ollama is not reachable at {ollama.url}, so '{model}' cannot be verified."
        )
    if not ollama.has(model):
        installed = ", ".join(ollama.models) or "none"
        raise HTTPException(400, f"Ollama does not have '{model}'. Installed: {installed}.")
    if not ollama.can_chat(model):
        answering = ", ".join(ollama.chat_models) or "none"
        raise HTTPException(
            400,
            f"'{model}' only produces embeddings, so it cannot answer questions. "
            f"Models that can: {answering}.",
        )


# ---- CHAT ----
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str
    question: str


@app.post("/api/chat/stream")
async def api_chat_stream(payload: ChatRequest) -> EventSourceResponse:
    """One question, one answer, streamed. No history: see PLANNING section 7.

    What can be checked before the response starts is checked here, because a
    404 delivered as an SSE event is a much worse way to say "no such subject"
    than a 404. Whatever goes wrong after that -- Ollama dying mid-answer --
    can only leave as an error event.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "Ask something first.")
    if not await run_blocking(store.exists, payload.collection):
        raise HTTPException(404, f"Subject '{payload.collection}' does not exist.")

    # Read the settings once, so a model swap mid-answer cannot change the model
    # halfway through the reply.
    return EventSourceResponse(chat_events(payload.collection, question, state.settings))


async def chat_events(collection: str, question: str, settings: Settings) -> AsyncIterator[dict]:
    """chain.astream() as SSE: one sources event, then tokens, then done.

    Every payload is JSON, including the tokens. A raw token would be fine until
    the model emitted a newline, which is the one character SSE uses as a field
    separator -- encoding sidesteps the whole question.
    """
    try:
        async for event in chain.astream(collection, question, settings):
            if isinstance(event, chain.SourcesEvent):
                yield {"event": "sources", "data": json.dumps([asdict(s) for s in event.sources])}
            else:
                yield {"event": "token", "data": json.dumps({"text": event.text})}
    except asyncio.CancelledError:
        # The tab was closed or the user hit Stop. Do not send "done": nobody is
        # listening, and the answer genuinely did not finish.
        raise
    except Exception as exc:
        yield {"event": "error", "data": json.dumps({"error": readable_failure(exc)})}
    yield {"event": "done", "data": "{}"}


def readable_failure(exc: Exception) -> str:
    """Turn a driver-level exception into something worth reading in a bubble."""
    text = str(exc).strip() or exc.__class__.__name__
    if "connect" in text.lower():
        return f"Ollama stopped answering at {OLLAMA_URL}. Start it and ask again."
    return text


# ---- COLLECTIONS ----
class NewCollection(BaseModel):
    """Chunking is optional here only because the globals are its default; it is
    still written to the collection and still permanent."""

    model_config = ConfigDict(extra="forbid")

    name: str
    chunk_size: int | None = None
    chunk_overlap: int | None = None


@app.post("/api/collections", status_code=201)
async def api_create_collection(payload: NewCollection) -> dict:
    """Create an empty subject.

    This exists as its own action because chunking belongs to the collection:
    somebody has to choose it before the first PDF, so a subject can no longer
    be born as a side effect of ingesting one.
    """
    settings = state.settings
    chunk_size = settings.default_chunk_size if payload.chunk_size is None else payload.chunk_size
    chunk_overlap = (
        settings.default_chunk_overlap if payload.chunk_overlap is None else payload.chunk_overlap
    )
    name = await run_blocking(store.create, payload.name, chunk_size, chunk_overlap)
    return asdict(await run_blocking(store.stats, name, settings))


@app.delete("/api/collections/{name}")
async def api_delete_collection(name: str) -> dict:
    """Irreversible, so it reports what it destroyed.

    The confirmation itself is the page's job, the same way it is the CLI's --
    the engine has no opinion about who is allowed to say yes.
    """
    doomed = await run_blocking(store.stats, name, state.settings)
    await run_blocking(store.delete, name)
    return {"deleted": name, "chunks": doomed.chunks, "pdfs": doomed.pdfs}


# ---- PDF LIBRARY ----
MAX_UPLOAD_MB = 200
UPLOAD_CHUNK = 1024 * 1024


def pdfs_payload() -> list[dict]:
    """Every PDF in pdfs/, with the subjects that already contain it.

    "Already contains" is decided by file name, because that is exactly what
    ingest_pdf() dedups on -- so what the page shows and what actually causes a
    skip cannot drift apart.
    """
    holders: dict[str, list[str]] = {}
    for name in store.list_collections():
        for pdf in store.pdfs_in(name):
            holders.setdefault(pdf, []).append(name)

    return [
        {
            "name": path.name,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            "in_collections": sorted(holders.get(path.name, [])),
        }
        for path in ingest.available_pdfs()
    ]


@app.get("/api/pdfs")
async def api_pdfs() -> list[dict]:
    return await run_blocking(pdfs_payload)


def safe_pdf_name(raw: str | None) -> str:
    """Reduce whatever the browser sent to a bare file name inside pdfs/.

    Only the last component survives, so '../../rag.py' cannot reach out of the
    folder, and the extension is checked because everything downstream assumes
    PyPDFLoader can open it.
    """
    name = Path(raw or "").name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(400, "That file has no usable name.")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, f"'{name}' is not a PDF.")
    if any(char in name for char in '<>:"/\\|?*'):
        raise HTTPException(400, f"'{name}' contains characters that cannot be stored on Windows.")
    return name


@app.post("/api/pdfs", status_code=201)
async def api_upload_pdf(file: UploadFile = File(...)) -> dict:
    """Write an upload into pdfs/, streaming it so a big book never sits in RAM.

    An existing name is a conflict rather than an overwrite: the file name is
    the dedup key for ingestion, so replacing the bytes under it would leave
    every subject claiming to contain a book it no longer has.
    """
    name = safe_pdf_name(file.filename)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    target = PDF_DIR / name
    if target.exists():
        raise HTTPException(409, f"'{name}' is already in pdfs/. Rename it, or delete the old one.")

    size = 0
    try:
        with target.open("wb") as out:
            while data := await file.read(UPLOAD_CHUNK):
                if not size and not data.startswith(b"%PDF-"):
                    # A renamed .docx would otherwise only fail deep inside
                    # ingestion, pointing at the wrong thing entirely.
                    raise HTTPException(400, f"'{name}' does not look like a PDF inside.")
                size += len(data)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"'{name}' is larger than {MAX_UPLOAD_MB} MB.")
                out.write(data)
    except BaseException:
        # Never leave a truncated PDF behind to be ingested later.
        target.unlink(missing_ok=True)
        raise

    return {"name": name, "size_mb": round(size / (1024 * 1024), 1)}


# ---- INGESTION JOBS ----
# One at a time. Chroma is SQLite and ingestion is the only heavy writer, so a
# second one is queued rather than allowed to race the first.
INGEST_LOCK = asyncio.Lock()
JOBS: dict[str, "Job"] = {}
JOB_HISTORY = 40


class Job:
    """One ingestion, watchable by a stream that connects after it started.

    Events are kept, not merely forwarded. The POST returns a job_id and the
    browser opens the stream a moment later, by which time 'reading' and
    'splitting' have often already gone past -- replaying the backlog is what
    stops the bar from appearing to start at forty percent.
    """

    def __init__(self, pdf: str, collection: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.pdf = pdf
        self.collection = collection
        self.status = "queued"  # queued | running | done | skipped | error
        self.events: list[dict] = []
        self.ended = False
        self.changed = asyncio.Event()
        self.task: asyncio.Task | None = None

    def emit(self, event: str, payload: dict) -> None:
        """Append an event. Must run on the event loop, never in the worker."""
        self.events.append({"event": event, "data": json.dumps(payload)})
        self.changed.set()

    async def follow(self) -> AsyncIterator[dict]:
        """Replay what has happened, then follow along until the job ends."""
        seen = 0
        while True:
            while seen < len(self.events):
                yield self.events[seen]
                seen += 1
            if self.ended:
                return
            await self.changed.wait()
            self.changed.clear()


async def run_job(job: Job) -> None:
    loop = asyncio.get_running_loop()

    def on_progress(progress: ingest.Progress) -> None:
        # Runs in the worker thread. Hop back to the loop before touching
        # anything the SSE stream reads.
        loop.call_soon_threadsafe(job.emit, "progress", asdict(progress))

    async with INGEST_LOCK:
        job.status = "running"
        try:
            result = await asyncio.to_thread(
                ingest.ingest_pdf, PDF_DIR / job.pdf, job.collection, state.settings, on_progress
            )
        except Exception as exc:
            job.status = "error"
            job.emit("error", {"error": readable_failure(exc)})
        else:
            job.status = "skipped" if result.skipped else "done"
            job.emit("result", asdict(result))
        finally:
            job.emit("done", {"status": job.status})
            job.ended = True
            job.changed.set()


def prune_jobs() -> None:
    """Keep the recent ones; a finished job is only a few kilobytes of text."""
    finished = [job_id for job_id, job in JOBS.items() if job.ended]
    for job_id in finished[: max(0, len(JOBS) - JOB_HISTORY)]:
        del JOBS[job_id]


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf: str
    collection: str


@app.post("/api/ingest", status_code=202)
async def api_ingest(payload: IngestRequest) -> dict:
    """Start an ingestion and hand back its job_id.

    Never blocking: 656 chunks took minutes on this machine, which is several
    times any sane HTTP timeout and would leave the page with nothing to show
    in the meantime.
    """
    pdf = safe_pdf_name(payload.pdf)
    if not (PDF_DIR / pdf).is_file():
        raise HTTPException(404, f"'{pdf}' is not in pdfs/.")
    if not await run_blocking(store.exists, payload.collection):
        raise HTTPException(404, f"Subject '{payload.collection}' does not exist.")

    job = Job(pdf, payload.collection)
    prune_jobs()
    JOBS[job.id] = job
    queued = INGEST_LOCK.locked()
    if queued:
        job.emit("progress", {"stage": "queued", "done": 0, "total": 0,
                              "message": "Waiting for the current ingestion to finish..."})
    job.task = asyncio.create_task(run_job(job))
    return {"job_id": job.id, "pdf": job.pdf, "collection": job.collection, "queued": queued}


@app.get("/api/ingest/{job_id}/stream")
async def api_ingest_stream(job_id: str) -> EventSourceResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "No such ingestion; it may have been forgotten after finishing.")
    return EventSourceResponse(job.follow())
