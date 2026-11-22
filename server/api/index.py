import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathochat import core

ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "https://mehdi-abidi.github.io,http://127.0.0.1:8124,http://localhost:8124").split(",")]


class handler(BaseHTTPRequestHandler):
    def route(self):
        url = urlparse(self.path)
        named = parse_qs(url.query).get("route", [""])[0]
        return (named or url.path).strip("/").split("/")[-1]

    def who(self):
        return (self.headers.get("x-forwarded-for") or self.client_address[0]).split(",")[0].strip()

    def reply(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        origin = self.headers.get("origin", "")
        if origin in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        length = int(self.headers.get("content-length") or 0)
        if length > core.MAX_AUDIO + 1024:
            raise core.Problem(413, "That is too large.")
        return self.rfile.read(length) if length else b""

    def run(self, work):
        try:
            self.reply(200, work())
        except core.Problem as p:
            self.reply(p.status, {"detail": p.message})
        except Exception:
            self.reply(500, {"detail": "Something went wrong on the server."})

    def do_OPTIONS(self):
        self.reply(200, {})

    def do_GET(self):
        if self.route() == "health":
            self.run(core.health)
        else:
            self.reply(404, {"detail": "Not found."})

    def do_POST(self):
        route = self.route()
        if route == "ask":
            def work():
                try:
                    question = json.loads(self.body() or b"{}").get("question", "")
                except ValueError:
                    raise core.Problem(400, "Send the question as JSON.")
                return core.ask(question, self.who())
            self.run(work)
        elif route == "transcribe":
            self.run(lambda: core.transcribe(self.body(), self.headers.get("content-type"), self.who()))
        else:
            self.reply(404, {"detail": "Not found."})

    def log_message(self, *args):
        pass
