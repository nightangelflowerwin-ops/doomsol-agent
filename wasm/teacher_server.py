from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import os
import threading


class TeacherHandler(SimpleHTTPRequestHandler):
    output_path: Path
    output_lock = threading.Lock()
    server_token: str

    def end_headers(self):
        # Rebuilds reuse index.js/index.wasm names; stale browser caches can
        # otherwise run an old engine against a new shell and break the API.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path == "/health":
            payload = json.dumps({
                "ok": True,
                "pid": os.getpid(),
                "token": self.server_token,
                "output": str(self.output_path),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/teacher-output":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Every POST is a durable NDJSON chunk. Append under a lock so an hour
        # run survives refreshes and concurrent beacon/fetch completion.
        with self.output_lock:
            with self.output_path.open("ab") as stream:
                stream.write(payload)
                stream.flush()
        self.send_response(204)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8127)
    parser.add_argument("--token", default="manual")
    parser.add_argument("--resume", action="store_true", help="append to an existing interrupted run")
    args = parser.parse_args()
    TeacherHandler.output_path = Path(args.output).resolve()
    TeacherHandler.server_token = args.token
    if TeacherHandler.output_path.exists() and not args.resume:
        TeacherHandler.output_path.unlink()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), lambda *a, **kw: TeacherHandler(*a, directory=args.directory, **kw))
    print(f"Teacher server: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
