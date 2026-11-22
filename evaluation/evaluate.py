import argparse
import json
import math
import os
import pickle
import re
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
MEDMCQA = "https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/main/data/validation-00000-of-00001.parquet"

PROMPTS = {
    "original": """
You are a focused assistant. Your task is to generate accurate answers using only the context provided.
- Do not include external knowledge.
- Do not speculate. If the context lacks the answer, respond with: "I don't know."
- No fluff, greetings, or commentary — jump straight to the answer.

[Context]
{context}

[Question]
{question}
Reply with the letter of the correct option only.

[Answer]
""",
    "grounded": """
You are a focused pathology assistant. Use the context below as your main evidence.
- Read the question and every option before answering.
- If the context does not settle it, use standard pathology knowledge.
- No fluff, greetings, or commentary.

[Context]
{context}

[Question]
{question}
Reply with the letter of the correct option only.

[Answer]
""",
    "direct": """Answer the multiple-choice question.

[Question]
{question}
Reply with the letter of the correct option only.

[Answer]
""",
    "tutor": """You are PathoChat, a pathology tutor. Answer the single-best-answer question below.

Reference passages from Robbins & Cotran and Pathoma follow. Use them when they are relevant. When they do not settle the question, rely on standard pathology knowledge.

[Passages]
{context}

[Question]
{question}

Reply in exactly two lines:
Reason: <one sentence>
Answer: <A, B, C or D>
""",
    "tutor_no_retrieval": """You are a pathology tutor. Answer the single-best-answer question below.

[Question]
{question}

Reply in exactly two lines:
Reason: <one sentence>
Answer: <A, B, C or D>
""",
}


def questions(split):
    path = os.path.join(DATA, "medmcqa_validation.parquet")
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        r = requests.get(MEDMCQA, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    df = pd.read_parquet(path)
    df = df[(df["subject_name"] == "Pathology") & (df["choice_type"] == "single")]
    test = df.sample(n=100, random_state=0)
    return (test if split == "test" else df.drop(test.index)).reset_index(drop=True)


def options(row):
    return "\n".join(f"{l}. {row[c]}" for l, c in zip("ABCD", ["opa", "opb", "opc", "opd"]))


class Retriever:
    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer
        folder = os.path.join(ROOT, "faiss_database")
        with open(os.path.join(folder, "index.pkl"), "rb") as f:
            self.store, self.ids = pickle.load(f)
        self.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query, k):
        import numpy as np
        hits = self.index.search(np.asarray(self.model.encode([query]), dtype="float32"), k)[1][0]
        return [self.store._dict[self.ids[int(i)]].page_content for i in hits]


def parse(text):
    text = re.sub(r"(?s)<think>.*?</think>", "", text or "")
    found = re.findall(r"(?i)answer\s*[:\-]\s*\**\s*\(?([ABCD])\b", text)
    if found:
        return found[-1].upper()
    if re.search(r"(?i)i don.?t", text):
        return "IDK"
    found = re.search(r"\b([ABCD])\b", text.upper())
    return found.group(1) if found else "NONE"


def ask(base_url, api_key, model, prompt, max_tokens):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens}
    for _ in range(40):
        r = requests.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=body, timeout=900)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"].get("content") or ""
        if r.status_code in (429, 500, 502, 503):
            time.sleep(min(float(r.headers.get("retry-after", 8)) + 1, 120))
            continue
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    raise RuntimeError("the model endpoint kept failing")


def run(args):
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, args.name + ".jsonl")
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    qs = questions(args.split)
    if args.limit:
        qs = qs.iloc[: args.limit]
    template = PROMPTS[args.prompt]
    retriever = Retriever() if "{context}" in template else None
    for _, row in qs.iterrows():
        if row["id"] in done:
            continue
        question = f"{row['question']}\n{options(row)}"
        context = "\n\n".join(retriever.search(question, args.k)) if retriever else ""
        text = ask(args.base_url, args.api_key, args.model, template.format(context=context, question=question), args.max_tokens)
        gold = "ABCD"[int(row["cop"])]
        pred = parse(text)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": row["id"], "gold": gold, "pred": pred, "correct": pred == gold}) + "\n")
        print(gold, pred, flush=True)
    report([args.name])


def load(name):
    with open(os.path.join(RESULTS, name + ".jsonl"), encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def report(names):
    runs = {name: load(name) for name in names}
    for name, rows in runs.items():
        n = len(rows)
        k = sum(r["correct"] for r in rows.values())
        lo, hi = wilson(k, n)
        refused = sum(r["pred"] in ("IDK", "NONE") for r in rows.values())
        print(f"{name}: {k}/{n} = {100 * k / n:.1f}% (95% CI {lo} to {hi}), no answer on {refused}")
    for a in names:
        for b in names:
            if a < b:
                common = set(runs[a]) & set(runs[b])
                only_a = sum(runs[a][i]["correct"] and not runs[b][i]["correct"] for i in common)
                only_b = sum(runs[b][i]["correct"] and not runs[a][i]["correct"] for i in common)
                print(f"{a} vs {b}: {len(common)} shared questions, only {a} right on {only_a}, only {b} right on {only_b}, McNemar p = {mcnemar(only_a, only_b)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--prompt", choices=list(PROMPTS), required=True)
    r.add_argument("--split", choices=["dev", "test"], default="test")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--max-tokens", type=int, default=1500)
    r.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1"))
    r.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    s = sub.add_parser("report")
    s.add_argument("names", nargs="+")
    a = p.parse_args()
    run(a) if a.cmd == "run" else report(a.names)
