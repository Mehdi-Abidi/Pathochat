import os
import sys
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))

from index import handler

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"PathoChat API on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
