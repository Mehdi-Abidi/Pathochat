import ast
import json
import os
import pickle
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
K = 5

QUESTIONS = [
    "What is the difference between apoptosis and necrosis?",
    "Which cell is characteristic of Hodgkin lymphoma, and which markers does it express?",
    "How does a myocardial infarct change under the microscope during its first week?",
    "How does nephritic syndrome differ from nephrotic syndrome?",
    "What are the steps by which a neutrophil leaves a blood vessel in acute inflammation?",
    "Which genetic change drives chronic myeloid leukemia?",
    "What distinguishes Crohn disease from ulcerative colitis?",
    "Why does vitamin B12 deficiency cause megaloblastic anemia?",
]


def app_prompt():
    with open(os.path.join(ROOT, "Pathochat.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PROMPT_TEMPLATE":
            return ast.literal_eval(node.value)
    raise RuntimeError("PROMPT_TEMPLATE was not found in Pathochat.py")


def book(source):
    name = os.path.basename(str(source).replace("\\", "/"))
    if name.lower().startswith("robbins"):
        return "Robbins & Cotran Pathologic Basis of Disease, 10th ed."
    if name.lower().startswith("pathoma"):
        return "Pathoma: Fundamentals of Pathology (2021)"
    return name


def ask(prompt):
    for _ in range(30):
        r = requests.post(BASE_URL.rstrip("/") + "/chat/completions",
                          headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"]},
                          json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1500},
                          timeout=120)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(min(float(r.headers.get("retry-after", 8)) + 1, 90))
            continue
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    raise RuntimeError("the model endpoint kept failing")


def main():
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    folder = os.path.join(ROOT, "faiss_database")
    with open(os.path.join(folder, "index.pkl"), "rb") as f:
        store, ids = pickle.load(f)
    index = faiss.read_index(os.path.join(folder, "index.faiss"))
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    template = app_prompt()
    out = []
    for q in QUESTIONS:
        hits = index.search(np.asarray(encoder.encode([q]), dtype="float32"), K)[1][0]
        docs = [store._dict[ids[int(i)]] for i in hits]
        answer = ask(template.format(context="\n\n".join(d.page_content for d in docs), question=q))
        out.append({
            "question": q,
            "answer": answer,
            "sources": [{"book": book(d.metadata.get("source", "")), "page": int(d.metadata.get("page", 0)) + 1,
                         "excerpt": " ".join(d.page_content.split())[:300]} for d in docs],
        })
        print(q, "->", len(answer), "chars", flush=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "examples.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "passages": K, "generated": time.strftime("%Y-%m-%d"), "examples": out}, f, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
