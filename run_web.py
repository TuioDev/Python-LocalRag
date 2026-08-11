"""Start the local web interface and open it in a tab.

One command, one window. The host is fixed at 127.0.0.1 and there is no flag to
change it: the app has no authentication of any kind, and everything it can
reach -- your PDFs, your collections, a model that will happily answer -- would
be reachable by anyone on the network the moment it bound 0.0.0.0.

One worker, no reload. chroma_db is SQLite, so a second process writing it
corrupts it; that includes a second copy of this server and the CLI.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from web.server import app

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
STARTUP_TIMEOUT = 20.0


def port_is_taken(port: int) -> bool:
    """Fail with a sentence instead of a stack trace when the port is in use."""
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((HOST, port)) == 0


def open_when_ready(server: uvicorn.Server, url: str) -> None:
    """Open the tab once uvicorn is really listening.

    Opening it straight away races the server and lands on a connection error
    often enough to be annoying.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if server.started:
            webbrowser.open(url)
            return
        time.sleep(0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local RAG web interface.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    parser.add_argument("--no-browser", action="store_true", help="start the server without opening a tab")
    args = parser.parse_args()

    url = f"http://{HOST}:{args.port}"
    if port_is_taken(args.port):
        print(f"Something is already listening on {HOST}:{args.port}.")
        print(f"If it is this app, open {url}. Otherwise start it with --port <other>.")
        return 1

    print("=" * 62)
    print("  Local RAG -- web interface")
    print(f"  {url}")
    print("  Ctrl+C to stop. Do not run rag.py at the same time.")
    print("=" * 62)

    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=args.port, log_level="info"))
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(server, url), daemon=True).start()
    server.run()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
