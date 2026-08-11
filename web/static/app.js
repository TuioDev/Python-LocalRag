"use strict";

/* Local RAG -- the whole client.
 *
 * No framework and no build step, so the rule that keeps this honest is: the
 * server owns the truth, the page owns nothing. Every mutation re-reads what
 * the server says instead of patching a local copy, because the CLI can change
 * the same store behind our back.
 *
 * All text from the store (PDF names, answers, error messages) goes in through
 * textContent. Nothing here builds HTML out of data. */

// ---- HTTP ----
async function request(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    // Every route reports failures as {"error": ...}; the fallback is for the
    // ones that never reach FastAPI, like a server that just died.
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

const getJSON = (path) => request(path);

const sendJSON = (path, method, body) =>
  request(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

// ---- DOM HELPERS ----
const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function fill(node, children) {
  node.replaceChildren(...children);
}

function setNote(node, message, kind) {
  node.className = kind ? `note ${kind}` : "note";
  node.textContent = message || "";
  node.hidden = !message;
}

// ---- STATE ----
const SELECTED_KEY = "localrag.subject";

const state = {
  ollama: null,
  // Whether the Ollama that is running (or missing) is one this app can act on.
  control: null,
  settings: null,
  missing: [],
  collections: [],
  pdfs: [],
  selected: localStorage.getItem(SELECTED_KEY) || null,
  // One transcript per subject; see TRANSCRIPTS for why the page keeps them
  // when the server keeps nothing.
  transcripts: loadTranscripts(),
  // What the last ingestion of each PDF into each subject ended up saying. Kept
  // out here because the row that reported it is destroyed by the refresh that
  // follows.
  ingestNotes: {},
};

// Ingestions in flight, as {subject, row, ui}, reused across re-renders so a
// second job finishing cannot wipe the first one's progress bar.
const liveRows = new Map();

const collection = (name) => state.collections.find((c) => c.name === name) || null;

function selectSubject(name) {
  state.selected = name;
  if (name) localStorage.setItem(SELECTED_KEY, name);
  else localStorage.removeItem(SELECTED_KEY);
  renderSubjects();
  renderChat();
  renderPdfs();
}

// ---- WORDING ----
function plural(count, word) {
  return `${count.toLocaleString()} ${word}${count === 1 ? "" : "s"}`;
}

function describeChunking(chunking) {
  if (chunking.legacy) {
    // The value is a guess, and saying so is the whole point: this collection
    // predates per-collection chunking, so its real size is unknowable.
    return `chunking not recorded (today's defaults are ${chunking.chunk_size}/${chunking.chunk_overlap})`;
  }
  return `chunks of ${chunking.chunk_size} overlapping ${chunking.chunk_overlap}`;
}

function describe(stats) {
  return `${plural(stats.chunks, "chunk")}, ${plural(stats.pdfs.length, "PDF")}, ${describeChunking(stats.chunking)}`;
}

// ---- BANNER ----
function showBanner(message) {
  const banner = $("banner");
  banner.textContent = message;
  banner.hidden = false;
}

function hideBanner() {
  $("banner").hidden = true;
}

// ---- ENGINE PANEL ----
function renderEngine() {
  const ollama = state.ollama;
  const settings = state.settings;

  const pill = $("ollama-pill");
  pill.className = `pill ${ollama.up ? "pill-ok" : "pill-bad"}`;
  pill.textContent = ollama.up ? "Ollama up" : "Ollama unreachable";

  $("ollama-url").textContent = ollama.url;
  // Only what went wrong. What to do about it is the control's line to say,
  // and it depends on whether there is anything here to start.
  setNote($("ollama-error"), ollama.up ? "" : `No answer: ${ollama.error || "no reply"}.`, "bad");
  renderOllamaControl();
  setNote(
    $("missing-models"),
    state.missing.length ? `Not installed: ${state.missing.join(", ")}. Run: ollama pull ${state.missing[0]}` : "",
    "warn"
  );

  renderModelPicker();

  $("temperature").value = settings.editable.temperature;
  $("temperature").disabled = false;
  $("top-k").value = settings.editable.top_k;
  $("top-k").disabled = false;

  $("embed-model").textContent = settings.locked.embed_model;
  $("embed-reason").textContent = settings.locked.reason;

  setNote(
    $("settings-note"),
    settings.error ? `settings.json is unusable (${settings.error}). Showing defaults; saving repairs it.` : "",
    "bad"
  );
}

// Ollama omits the ':latest' suffix about as often as it includes it, the same
// tolerance health.same_model() has on the other side.
const sameModel = (installed, wanted) =>
  installed === wanted || installed === (wanted.includes(":") ? wanted : `${wanted}:latest`);

function renderModelPicker() {
  const select = $("llm-model");
  const current = state.settings.editable.llm_model;
  // Only models that can generate text. Ollama lists the embedding model too --
  // it is installed and healthy and still cannot produce one word of an answer,
  // so offering it here would only be a way to break the app from the UI.
  const usable = state.ollama.chat_models;

  const options = usable.map((name) => new Option(name, name, false, sameModel(name, current)));
  if (!usable.some((name) => sameModel(name, current))) {
    // Keep the saved model visible even when Ollama does not have it, or the
    // select would quietly claim the app is answering with something else.
    options.unshift(new Option(`${current} (not installed)`, current, false, true));
  }
  fill(select, options);
  select.disabled = !state.ollama.up;
}

async function saveSettings(changes) {
  const note = $("settings-note");
  setNote(note, "Saving...");
  let message = ["Saved. It applies to your next question.", "ok"];
  try {
    state.settings = await sendJSON("/api/settings", "PATCH", changes);
  } catch (error) {
    message = [error.message, "bad"];
  }
  // Re-render either way: on success to show what was stored, on failure to put
  // the widget back to the value that is actually saved. It runs before the
  // note because renderEngine() owns that element too.
  renderEngine();
  setNote(note, message[0], message[1]);
}

// ---- STARTING AND STOPPING OLLAMA ----
/* Being told the model server is down is only half an answer, so the panel can
 * start it. The other half is what the button refuses to do: a server this app
 * did not start belongs to whoever did start it -- the tray icon, another
 * terminal -- so the button disappears and the note says where to go instead of
 * offering a Stop that would kill somebody else's process. */
let ollamaBusy = null; // the label to show while a start or stop is in flight
let ollamaNote = null; // what the last attempt had to say, when it failed

function renderOllamaControl() {
  if (!state.ollama) return;
  const button = $("ollama-toggle");
  const note = $("ollama-note");
  const control = state.control || {};
  const managed = Boolean(control.managed);

  button.classList.toggle("stop", managed);
  button.title = control.executable || "";

  if (ollamaBusy) {
    button.hidden = false;
    button.disabled = true;
    button.textContent = ollamaBusy;
    setNote(note, "A first start takes a few seconds.", "muted");
    return;
  }

  // Nothing useful to offer for a server somebody else started, and nothing to
  // start with if the program is not on this machine.
  const answering = managed && Boolean(streaming);
  button.hidden = state.ollama.up && !managed;
  button.disabled = answering || (!state.ollama.up && !control.executable);
  button.textContent = managed ? "Stop Ollama" : "Start Ollama";

  let message;
  let kind = "muted";
  if (answering) {
    message = "An answer is being written. Stopping Ollama now would cut it off.";
    kind = "warn";
  } else if (managed) {
    message = `Started here, pid ${control.pid}. It stops when this app stops.`;
  } else if (state.ollama.up) {
    message =
      "Running, but this app did not start it. Stop it where you started it: " +
      "the tray icon, or the terminal running 'ollama serve'.";
  } else if (control.executable) {
    message = "Starts 'ollama serve' on this machine, in the background.";
  } else {
    message = "No 'ollama' program found on PATH. Install Ollama, then hit Refresh.";
    kind = "bad";
  }
  // A failure outranks the description: it is the newer news, and it is the
  // only place the reason will ever be shown.
  if (ollamaNote) [message, kind] = [ollamaNote.text, ollamaNote.kind];
  setNote(note, message, kind);
}

async function toggleOllama() {
  const stopping = Boolean(state.control && state.control.managed);
  ollamaBusy = stopping ? "Stopping Ollama..." : "Starting Ollama...";
  ollamaNote = null;
  renderOllamaControl();
  try {
    await request(stopping ? "/api/ollama/stop" : "/api/ollama/start", { method: "POST" });
  } catch (error) {
    // Held in a variable because the refresh below redraws this note from the
    // server's state, and the server's state has nothing to say about a start
    // that never happened.
    ollamaNote = { text: error.message, kind: "bad" };
  } finally {
    ollamaBusy = null;
    // Everything the panel shows just changed: the pill, the installed models,
    // the picker. Read it all back rather than guess at it.
    await refresh();
    // refresh() gives up quietly when it is the local server that has gone
    // away, and the button must not be left saying "Starting Ollama..." for
    // the rest of the session.
    renderOllamaControl();
  }
}

// ---- SUBJECT LIST ----
function renderSubjects() {
  const list = $("subjects");
  if (!state.collections.length) {
    fill(list, [el("p", "empty", "No subjects yet. Create one, then add a PDF to it.")]);
    return;
  }

  const items = state.collections.map((stats) => {
    const item = el("li", "subject-row");

    const button = el("button", "subject");
    button.type = "button";
    button.setAttribute("aria-current", String(stats.name === state.selected));
    button.append(el("span", "subject-name", stats.name), el("span", "subject-meta", describe(stats)));
    button.addEventListener("click", () => selectSubject(stats.name));

    const remove = el("button", "danger", "x");
    remove.type = "button";
    remove.title = `Delete '${stats.name}'`;
    remove.setAttribute("aria-label", `Delete subject ${stats.name}`);
    remove.addEventListener("click", () => confirmDelete(item, stats));

    item.append(button, remove);
    return item;
  });
  fill(list, items);
}

/* Deleting is irreversible, so it says what will be lost and the safe answer
 * comes first -- the same contract the CLI has, where the confirmation defaults
 * to No. */
function confirmDelete(item, stats) {
  const box = el("div", "inset");
  box.append(
    el("p", "subject-name", `Delete '${stats.name}'?`),
    el("p", "note", `${describe(stats)}. This cannot be undone.`)
  );
  if (stats.pdfs.length) {
    box.append(el("p", "note muted", `Contains: ${stats.pdfs.join(", ")}`));
  }

  const keep = el("button", "ghost", "Keep it");
  keep.type = "button";
  keep.addEventListener("click", renderSubjects);

  const remove = el("button", "danger", "Delete");
  remove.type = "button";
  remove.addEventListener("click", async () => {
    keep.disabled = remove.disabled = true;
    try {
      await request(`/api/collections/${encodeURIComponent(stats.name)}`, { method: "DELETE" });
      // The subject is gone; a transcript about books nobody can search any
      // more is just clutter waiting to reappear under a reused name.
      delete state.transcripts[stats.name];
      saveTranscripts();
      if (state.selected === stats.name) selectSubject(null);
      await refresh();
    } catch (error) {
      showBanner(error.message);
      renderSubjects();
    }
  });

  const actions = el("div", "actions");
  actions.append(keep, remove);
  box.append(actions);
  fill(item, [box]);
}

// ---- CREATING A SUBJECT ----
function showCreateForm(show) {
  const form = $("create-form");
  form.hidden = !show;
  if (!show) return;

  // The globals are suggestions for this form and nothing else: once written,
  // the chunking belongs to the collection.
  $("create-name").value = "";
  $("create-size").value = state.settings.editable.default_chunk_size;
  $("create-overlap").value = state.settings.editable.default_chunk_overlap;
  setNote($("create-note"), "");
  $("create-name").focus();
}

async function createSubject(event) {
  event.preventDefault();
  setNote($("create-note"), "Creating...");
  try {
    const stats = await sendJSON("/api/collections", "POST", {
      name: $("create-name").value.trim(),
      chunk_size: parseInt($("create-size").value, 10),
      chunk_overlap: parseInt($("create-overlap").value, 10),
    });
    showCreateForm(false);
    await refresh();
    selectSubject(stats.name);
  } catch (error) {
    setNote($("create-note"), error.message, "bad");
  }
}

// ---- CHAT PANEL ----
function renderChat() {
  const select = $("chat-subject");
  fill(
    select,
    state.collections.map((stats) => new Option(stats.name, stats.name, false, stats.name === state.selected))
  );
  select.disabled = !state.collections.length || Boolean(streaming);

  const stats = collection(state.selected);
  $("chat-subject-info").textContent = stats ? describe(stats) : "";
  $("question").disabled = !stats;
  $("ask").disabled = !stats && !streaming;
  disarmClear();

  renderTranscript();
}

function emptyState(stats) {
  const box = el("div", "empty-state");
  if (!stats) {
    box.append(
      el("h3", null, "No subject selected"),
      el("p", null, "A subject is one Chroma collection: its own PDFs, its own chunking, its own answers.")
    );
    return box;
  }

  box.append(el("h3", null, stats.name), el("p", null, describe(stats)));

  const facts = el("dl");
  const add = (term, value) => facts.append(el("dt", null, term), el("dd", null, value));
  add("Embeddings", stats.chunking.embed_model);
  add("Chunking", describeChunking(stats.chunking));
  add("Created", stats.created_at ? stats.created_at.replace("T", " ") : "before this was recorded");
  add("PDFs", stats.pdfs.length ? stats.pdfs.join("\n") : "none yet");
  facts.querySelectorAll("dd").forEach((dd) => (dd.style.whiteSpace = "pre-line"));
  box.append(facts);
  return box;
}

// ---- SERVER-SENT EVENTS ----
/* Read by hand instead of with EventSource, which only speaks GET -- and the
 * question belongs in a body, not in a URL. The delimiter is a regex because
 * sse-starlette separates events with CRLF while the spec allows three forms;
 * an incomplete one simply does not match yet, which is what makes this safe
 * across chunk boundaries. */
const EVENT_BREAK = /\r\n\r\n|\n\n|\r\r/;

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    for (;;) {
      const split = EVENT_BREAK.exec(buffer);
      if (!split) break;
      const event = parseEvent(buffer.slice(0, split.index));
      buffer = buffer.slice(split.index + split[0].length);
      if (event) yield event;
    }
  }
}

function parseEvent(block) {
  let name = "message";
  const data = [];
  for (const line of block.split(/\r\n|\n|\r/)) {
    if (!line || line.startsWith(":")) continue; // blank line, or a keep-alive comment
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") name = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { name, data: data.join("\n") } : null;
}

// ---- TRANSCRIPTS ----
/* Chat is stateless on the server: one question, one answer, nothing of the
 * conversation is ever sent back to the model. That is a retrieval decision and
 * it stands -- but it is no reason to throw the answers away, because the page
 * is the only place they exist. So each subject keeps its own transcript here,
 * saved in localStorage: switching subjects puts one away and takes another
 * out, and a reload finds them where they were left.
 *
 * A turn is plain data ({question, raw, sources, error}) rather than DOM. The
 * nodes are built from it, which is what lets the same record be shown, hidden
 * while another subject is up, written to by a stream in flight, and stored. */
const TRANSCRIPTS_KEY = "localrag.transcripts";
const MAX_TURNS = 40; // per subject; older exchanges fall off the top

function loadTranscripts() {
  let saved;
  try {
    saved = JSON.parse(localStorage.getItem(TRANSCRIPTS_KEY) || "{}");
  } catch (error) {
    return {}; // written by hand, or by a version that stored something else
  }
  // Rebuild every turn to a known shape. This is the one input the page cannot
  // check as it arrives, and a half-written record must not be able to take the
  // whole transcript down when it is drawn.
  const clean = {};
  for (const [name, turns] of Object.entries(saved || {})) {
    if (!Array.isArray(turns)) continue;
    clean[name] = turns.map((turn) => ({
      question: String((turn && turn.question) || ""),
      raw: String((turn && turn.raw) || ""),
      sources: Array.isArray(turn && turn.sources) ? turn.sources : [],
      error: turn && turn.error ? String(turn.error) : null,
    }));
  }
  return clean;
}

function saveTranscripts() {
  try {
    localStorage.setItem(TRANSCRIPTS_KEY, JSON.stringify(state.transcripts));
  } catch (error) {
    // Out of quota, and the oldest exchanges are the ones worth losing. Keep a
    // few of each and try once more; if even that will not fit, the page goes
    // on working from memory and only the saved copy falls behind.
    for (const name of Object.keys(state.transcripts)) {
      state.transcripts[name] = state.transcripts[name].slice(-3);
    }
    try {
      localStorage.setItem(TRANSCRIPTS_KEY, JSON.stringify(state.transcripts));
    } catch (again) {
      /* nothing else to try */
    }
  }
}

const transcript = (name) => (name && state.transcripts[name]) || [];

function recordTurn(name, turn) {
  const turns = state.transcripts[name] || (state.transcripts[name] = []);
  turns.push(turn);
  if (turns.length > MAX_TURNS) turns.splice(0, turns.length - MAX_TURNS);
}

// The nodes of the turns currently on screen. Rebuilt by every render, so an
// answer still streaming repaints into whatever is showing it now -- or into
// nothing at all, if the user has moved to another subject meanwhile.
const turnNodes = new Map();
let renderedSubject = null;

function scrollToLatest() {
  const box = $("messages");
  box.scrollTop = box.scrollHeight;
}

function renderTranscript() {
  const box = $("messages");
  // Coming back to the same subject keeps your place; arriving at a different
  // one starts at the newest exchange.
  const keepAt = renderedSubject === state.selected ? box.scrollTop : null;

  turnNodes.clear();
  const turns = transcript(state.selected);
  fill(box, turns.length ? turns.flatMap(buildTurn) : [emptyState(collection(state.selected))]);

  renderedSubject = state.selected;
  if (keepAt === null) scrollToLatest();
  else box.scrollTop = keepAt;
}

function buildTurn(turn) {
  const question = el("div", "msg msg-user", turn.question);

  const sources = el("details", "block");
  const sourcesSummary = el("summary", null, "Sources");
  const sourcesBody = el("div", "body");
  sources.append(sourcesSummary, sourcesBody);

  const reasoning = el("details", "block");
  const reasoningBody = el("div", "body reasoning");
  reasoning.append(el("summary", null, "Reasoning"), reasoningBody);

  const answer = el("div", "answer");
  const failure = el("p", "note bad");
  const bubble = el("div", "msg msg-bot");
  bubble.append(sources, reasoning, answer, failure);

  turnNodes.set(turn, { bubble, sources, sourcesSummary, sourcesBody, reasoning, reasoningBody, answer, failure });
  paintTurn(turn);
  return [question, bubble];
}

function paintTurn(turn) {
  const nodes = turnNodes.get(turn);
  if (!nodes) return; // this turn belongs to a subject that is not on screen

  // Sources arrive before the first token, which is the point of the ordering in
  // chain.astream(): you can see what is being read while it is answered.
  nodes.sourcesSummary.textContent = `Sources (${turn.sources.length})`;
  fill(nodes.sourcesBody, turn.sources.map(sourceRow));
  nodes.sources.hidden = !turn.sources.length;

  const split = splitReasoning(turn.raw);
  nodes.reasoningBody.textContent = split.reasoning;
  nodes.reasoning.hidden = !split.reasoning;
  nodes.answer.textContent = split.answer;

  nodes.failure.textContent = turn.error || "";
  nodes.failure.hidden = !turn.error;
  nodes.bubble.classList.toggle("msg-error", Boolean(turn.error) && !turn.raw);
  // Only the answer being written now may say "thinking..."; a restored one that
  // came back empty must not sit there pretending.
  nodes.bubble.classList.toggle("settled", turn !== live);
}

/* Clearing is the one way to lose answers for good, so it asks first -- in
 * place, the same bargain the delete flow makes, minus a dialog for two
 * sentences of consequence. */
let clearArmed = null;

function disarmClear() {
  const button = $("clear-chat");
  clearTimeout(clearArmed);
  clearArmed = null;
  button.textContent = "Clear chat";
  button.classList.remove("danger");
  button.disabled = !transcript(state.selected).length;
}

function clearChat() {
  if (clearArmed === null) {
    const button = $("clear-chat");
    button.textContent = "Clear it? Click again";
    button.classList.add("danger");
    clearArmed = setTimeout(disarmClear, 5000);
    return;
  }
  delete state.transcripts[state.selected];
  saveTranscripts();
  disarmClear();
  renderTranscript();
}

function sourceRow(source) {
  const row = el("div", "source");
  const head = el("div", "source-head", source.pdf);
  if (source.page) head.append(el("span", "source-page", ` -- page ${source.page}`));
  row.append(head, el("p", "source-snippet", source.snippet));
  return row;
}

/* Reasoning models stream their scratchpad inside <think> tags, and dumping
 * that raw into the answer makes every reply look like a mess. The whole text
 * is re-split on each token rather than parsed incrementally, so a tag arriving
 * in two pieces resolves itself on the next one. */
function splitReasoning(raw) {
  const OPEN = "<think>";
  const CLOSE = "</think>";
  let reasoning = "";
  let answer = "";
  let at = 0;
  while (at < raw.length) {
    const opened = raw.indexOf(OPEN, at);
    if (opened < 0) {
      answer += raw.slice(at);
      break;
    }
    answer += raw.slice(at, opened);
    const closed = raw.indexOf(CLOSE, opened + OPEN.length);
    if (closed < 0) {
      reasoning += raw.slice(opened + OPEN.length); // still being written
      break;
    }
    reasoning += raw.slice(opened + OPEN.length, closed);
    at = closed + CLOSE.length;
  }
  return { reasoning: reasoning.trim(), answer: answer.trim() };
}

// ---- ASKING ----
let streaming = null; // the AbortController of the answer in flight, if any
let live = null; // the turn being written right now, if any

function setStreaming(controller) {
  streaming = controller;
  $("ask").textContent = controller ? "Stop" : "Ask";
  $("ask").disabled = !controller && !collection(state.selected);
  $("chat-subject").disabled = Boolean(controller) || !state.collections.length;
  // The Stop Ollama button stands down while an answer is being written: it
  // would cut off the very answer you are reading.
  renderOllamaControl();
}

function follow(turn) {
  // Keep the newest text in view, but only while its own subject is the one
  // being looked at.
  if (turnNodes.has(turn)) scrollToLatest();
}

async function ask(question) {
  const subject = state.selected;
  const turn = { question, raw: "", sources: [], error: null };
  recordTurn(subject, turn);
  live = turn;

  const box = $("messages");
  const placeholder = box.querySelector(".empty-state");
  if (placeholder) placeholder.remove();
  box.append(...buildTurn(turn));
  scrollToLatest();
  $("clear-chat").disabled = false;

  const controller = new AbortController();
  setStreaming(controller);
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collection: subject, question }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `${response.status} ${response.statusText}`);
    }
    for await (const event of readEvents(response)) {
      if (event.name === "sources") turn.sources = JSON.parse(event.data);
      else if (event.name === "token") turn.raw += JSON.parse(event.data).text;
      else if (event.name === "error") turn.error = JSON.parse(event.data).error;
      else if (event.name === "done") break;
      paintTurn(turn);
      follow(turn);
    }
  } catch (error) {
    turn.error = error.name === "AbortError" ? "Stopped." : error.message;
  } finally {
    live = null;
    paintTurn(turn);
    setStreaming(null);
    // Saved once, at the end: a partial answer is worth keeping, but not worth
    // re-serialising the whole transcript for on every token.
    saveTranscripts();
  }
}

// ---- PDF LIBRARY ----
// A subject name cannot contain '/' and neither can a file name, so the pair
// keys an ingestion note without ambiguity.
const noteKey = (subject, pdf) => `${subject}/${pdf}`;

/* The panel shows the books of the subject you are asking about, not the whole
 * folder. A file sitting in pdfs/ that this subject has never read has nothing
 * to do with the answers you are getting, so it waits in the picker at the
 * bottom until you ask for it. */
function renderPdfs() {
  const stats = collection(state.selected);
  $("pdf-title").textContent = stats ? `PDFs in ${stats.name}` : "PDFs";
  $("drop-text").textContent = stats
    ? `Drop a PDF here to add it to '${stats.name}'`
    : "Drop a PDF here, or click to choose one";

  // What the subject contains, plus anything being added to it right now.
  const names = new Set(stats ? stats.pdfs : []);
  for (const [name, job] of liveRows) {
    if (job.subject === state.selected) names.add(name);
  }
  const shown = [...names].sort();
  $("pdf-count").textContent = shown.length ? String(shown.length) : "";

  const box = $("pdfs");
  if (!shown.length) {
    const message = !state.collections.length
      ? "No subjects yet. Create one, then drop a PDF into it."
      : !stats
        ? "No subject selected."
        : "Nothing in this subject yet. Drop a PDF here to add the first one.";
    fill(box, [el("p", "empty", message)]);
  } else {
    // A row mid-ingestion is kept as it is, bar and all.
    fill(
      box,
      shown.map((name) => {
        const job = liveRows.get(name);
        return (job && job.subject === state.selected ? job : buildPdfRow(name, state.selected)).row;
      })
    );
  }

  renderAddExisting(names);
}

function buildPdfRow(name, subject) {
  const file = state.pdfs.find((pdf) => pdf.name === name) || null;
  const row = el("div", "pdf");
  // A book can be in a subject and gone from pdfs/: the chunks stay searchable,
  // but saying "12 MB" about a file that is not there would be a lie.
  row.append(
    el("div", "pdf-name", name),
    el("div", "pdf-meta", file ? `${file.size_mb} MB` : "no longer in pdfs/")
  );

  const status = el("p", "note");
  status.hidden = true;
  const bar = el("div", "bar");
  const filled = el("span");
  bar.append(filled);
  bar.hidden = true;
  row.append(status, bar);

  const previous = state.ingestNotes[noteKey(subject, name)];
  if (previous) setNote(status, previous.text, previous.kind);

  return { row, ui: { status, bar, filled } };
}

function renderAddExisting(shown) {
  const box = $("add-existing");
  const others = state.pdfs.filter((pdf) => !shown.has(pdf.name));
  box.hidden = !collection(state.selected) || !others.length;
  if (box.hidden) return;
  fill($("existing-pdf"), others.map((pdf) => new Option(`${pdf.name} (${pdf.size_mb} MB)`, pdf.name)));
}

/* Adding a book takes minutes, so the row goes up first and the progress
 * arrives on its own stream. */
async function addPdf(name, subject) {
  delete state.ingestNotes[noteKey(subject, name)];
  const built = buildPdfRow(name, subject);
  liveRows.set(name, { subject, ...built });
  renderPdfs();
  await ingestPdf(name, subject, built.ui);
}

async function ingestPdf(pdf, subject, ui) {
  ui.bar.hidden = false;
  ui.filled.style.width = "0%";
  setNote(ui.status, "Starting...");

  let ending = { text: "Finished.", kind: "ok" };
  try {
    const job = await sendJSON("/api/ingest", "POST", { pdf, collection: subject });
    const response = await fetch(`/api/ingest/${encodeURIComponent(job.job_id)}/stream`);
    if (!response.ok) throw new Error(`Cannot follow the ingestion (${response.status}).`);

    for await (const event of readEvents(response)) {
      const payload = JSON.parse(event.data);
      if (event.name === "progress") {
        setNote(ui.status, payload.message);
        ui.filled.style.width = payload.total ? `${Math.round((100 * payload.done) / payload.total)}%` : "0%";
      } else if (event.name === "result") {
        if (payload.skipped) {
          ending = { text: `Already in '${subject}'. Nothing to do.`, kind: "warn" };
        } else if (!payload.chunks) {
          // A scan with no text layer reads perfectly well and extracts to
          // nothing. It is worth saying so, because the ingestion itself
          // succeeded and the subject is none the wiser.
          ending = {
            text: `No text could be read from '${pdf}', so nothing was added. A scanned PDF with no text layer looks like this.`,
            kind: "warn",
          };
        } else {
          ending = {
            text: `Added ${payload.chunks} chunks from ${payload.pages} pages to '${subject}'.`,
            kind: "ok",
          };
        }
      } else if (event.name === "error") {
        ending = { text: payload.error, kind: "bad" };
      } else if (event.name === "done") {
        break;
      }
    }
  } catch (error) {
    ending = { text: error.message, kind: "bad" };
  }

  state.ingestNotes[noteKey(subject, pdf)] = ending;
  liveRows.delete(pdf);
  // The chunk count and the subject's list of books both just changed.
  await refresh();
}

/* Dropping a PDF means "put this in the subject I am reading", so it is saved
 * and then ingested in one gesture. A name already in pdfs/ is not an error
 * here: the file name is the dedup key everywhere else, so the same name is
 * treated as the same book and goes straight to the subject. */
async function upload(file) {
  if (!file) return;
  const note = $("upload-note");
  const subject = state.selected;
  let name = file.name;

  if (!state.pdfs.some((pdf) => pdf.name === name)) {
    setNote(note, `Uploading '${name}'...`);
    const body = new FormData();
    body.append("file", file);
    try {
      // No Content-Type here on purpose: the browser has to set the multipart
      // boundary itself.
      const saved = await request("/api/pdfs", { method: "POST", body });
      name = saved.name;
    } catch (error) {
      setNote(note, error.message, "bad");
      return;
    }
    await loadPdfs();
  }

  if (!subject) {
    setNote(note, `'${name}' is in pdfs/. Pick a subject to add it to.`, "warn");
    return;
  }
  // Said here rather than left to the row, because the row is only on screen
  // while that subject is the one selected.
  setNote(note, `Adding '${name}' to '${subject}'...`);
  await addPdf(name, subject);
  setNote(note, "");
}

async function loadPdfs() {
  try {
    state.pdfs = await getJSON("/api/pdfs");
  } catch (error) {
    showBanner(error.message);
    return;
  }
  renderPdfs();
}

// ---- LOADING ----
async function refresh() {
  try {
    const status = await getJSON("/api/status");
    state.ollama = status.ollama;
    state.control = status.control;
    state.settings = status.settings;
    state.missing = status.missing_models;
    state.collections = status.collections;
    hideBanner();
  } catch (error) {
    showBanner(`Cannot reach the local server: ${error.message}`);
    return;
  }

  if (!collection(state.selected)) {
    // The subject remembered from last time may have been deleted since.
    state.selected = state.collections.length ? state.collections[0].name : null;
  }

  renderEngine();
  renderSubjects();
  renderChat();
  await loadPdfs();
}

// ---- WIRING ----
function wire() {
  $("refresh").addEventListener("click", () => {
    // Asking for the truth again drops what the last failed start had to say.
    ollamaNote = null;
    refresh();
  });

  $("ollama-toggle").addEventListener("click", toggleOllama);

  $("llm-model").addEventListener("change", (event) => saveSettings({ llm_model: event.target.value }));

  $("temperature").addEventListener("change", (event) => {
    const value = parseFloat(event.target.value);
    if (Number.isNaN(value)) return renderEngine();
    saveSettings({ temperature: value });
  });

  $("top-k").addEventListener("change", (event) => {
    const value = parseInt(event.target.value, 10);
    if (Number.isNaN(value)) return renderEngine();
    saveSettings({ top_k: value });
  });

  $("chat-subject").addEventListener("change", (event) => selectSubject(event.target.value));
  $("clear-chat").addEventListener("click", clearChat);

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    if (streaming) {
      // The button is the Stop button while an answer is being written.
      streaming.abort();
      return;
    }
    const box = $("question");
    const question = box.value.trim();
    if (!question || !collection(state.selected)) return;
    box.value = "";
    ask(question);
  });

  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });

  $("new-subject").addEventListener("click", () => showCreateForm($("create-form").hidden));
  $("create-cancel").addEventListener("click", () => showCreateForm(false));
  $("create-form").addEventListener("submit", createSubject);

  $("add-existing-go").addEventListener("click", () => {
    const name = $("existing-pdf").value;
    if (name && collection(state.selected)) addPdf(name, state.selected);
  });

  $("file").addEventListener("change", (event) => {
    upload(event.target.files[0]);
    event.target.value = ""; // so the same file can be picked twice running
  });

  const drop = $("drop");
  for (const name of ["dragenter", "dragover"]) {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("over");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.remove("over");
    });
  }
  drop.addEventListener("drop", (event) => upload(event.dataTransfer.files[0]));
}

wire();
refresh();
