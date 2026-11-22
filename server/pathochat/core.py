import json
import os
import time
from collections import defaultdict, deque

import numpy as np
import requests

from .encoder import Encoder

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "openai/gpt-oss-120b")
SPEECH_MODEL = os.getenv("SPEECH_MODEL", "whisper-large-v3-turbo")
PASSAGES = 5
MAX_QUESTION = 600
MAX_AUDIO = 4 * 1024 * 1024
WINDOW = 600
PER_WINDOW = int(os.getenv("ASK_PER_10_MIN", "12"))

state = {}
recent = defaultdict(deque)


class Problem(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def load():
    if not state:
        state["encoder"] = Encoder(DATA)
        state["embeddings"] = np.load(os.path.join(DATA, "embeddings.npy"))
        with open(os.path.join(DATA, "passages.json"), encoding="utf-8") as f:
            state["passages"] = json.load(f)
        with open(os.path.join(DATA, "prompt.txt"), encoding="utf-8") as f:
            state["prompt"] = f.read()
    return state


def allow(who):
    now = time.time()
    q = recent[who]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= PER_WINDOW:
        raise Problem(429, "Too many questions in a short time. Wait a few minutes.")
    q.append(now)


def key():
    value = os.getenv("LLM_API_KEY", "")
    if not value:
        raise Problem(503, "The server has no model key configured.")
    return value


def health():
    s = load()
    return {"ok": True, "model": MODEL, "passages": PASSAGES, "chunks": int(s["embeddings"].shape[0])}


def retrieve(question):
    s = load()
    scores = s["embeddings"] @ s["encoder"].encode(question)
    top = np.argpartition(-scores, PASSAGES)[:PASSAGES]
    return [s["passages"][int(i)] for i in top[np.argsort(-scores[top])]]


def complete(prompt):
    busy = False
    for model in [MODEL, FALLBACK_MODEL]:
        for attempt in range(2):
            r = requests.post(BASE_URL + "/chat/completions", headers={"Authorization": "Bearer " + key()},
                              json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1500}, timeout=50)
            if r.status_code == 200:
                text = (r.json()["choices"][0]["message"].get("content") or "").strip()
                if text:
                    return text, model
                break
            if r.status_code == 429:
                busy = True
                wait = float(r.headers.get("retry-after", 30))
                if attempt == 0 and wait <= 6:
                    time.sleep(wait + 0.5)
                    continue
            break
    if busy:
        raise Problem(429, "The model is busy. Try again in a minute.")
    raise Problem(502, "The model did not answer. Try again.")


def ask(question, who):
    question = " ".join(str(question or "").split())
    if not question:
        raise Problem(400, "Ask a question first.")
    if len(question) > MAX_QUESTION:
        raise Problem(400, f"Keep the question under {MAX_QUESTION} characters.")
    allow(who)
    docs = retrieve(question)
    answer, model = complete(load()["prompt"].format(context="\n\n".join(d["text"] for d in docs), question=question))
    return {"answer": answer, "model": model,
            "sources": [{"book": d["book"], "page": d["page"], "excerpt": " ".join(d["text"].split())[:300]} for d in docs]}


def transcribe(audio, content_type, who):
    if not audio:
        raise Problem(400, "The recording was empty.")
    if len(audio) > MAX_AUDIO:
        raise Problem(400, "The recording is too long. Keep it under a minute.")
    allow(who)
    kind = (content_type or "audio/webm").split(";")[0].strip()
    ext = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav"}.get(kind, "webm")
    r = requests.post(BASE_URL + "/audio/transcriptions", headers={"Authorization": "Bearer " + key()},
                      files={"file": ("question." + ext, audio, kind)},
                      data={"model": SPEECH_MODEL, "language": "en", "prompt": "A question about pathology."}, timeout=50)
    if r.status_code != 200:
        raise Problem(502, "The recording could not be transcribed. Try again.")
    return {"text": (r.json().get("text") or "").strip()}
