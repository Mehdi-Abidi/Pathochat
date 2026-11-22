import ast
import json
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "data")


def book(source):
    name = os.path.basename(str(source).replace("\\", "/")).lower()
    if name.startswith("robbins"):
        return "Robbins & Cotran Pathologic Basis of Disease, 10th ed."
    if name.startswith("pathoma"):
        return "Pathoma: Fundamentals of Pathology (2021)"
    return name


def app_prompt():
    with open(os.path.join(ROOT, "Pathochat.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PROMPT_TEMPLATE":
            return ast.literal_eval(node.value)
    raise RuntimeError("PROMPT_TEMPLATE was not found in Pathochat.py")


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "prompt.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(app_prompt())
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    state = model[0].auto_model.state_dict()
    np.savez(os.path.join(OUT, "weights.npz"), **{k: v.numpy().astype(np.float16) for k, v in state.items() if "position_ids" not in k})
    vocab = model[0].tokenizer.get_vocab()
    with open(os.path.join(OUT, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])) + "\n")
    index = faiss.read_index(os.path.join(ROOT, "faiss_database", "index.faiss"))
    np.save(os.path.join(OUT, "embeddings.npy"), index.reconstruct_n(0, index.ntotal).astype(np.float32))
    with open(os.path.join(ROOT, "faiss_database", "index.pkl"), "rb") as f:
        store, ids = pickle.load(f)
    docs = [store._dict[ids[i]] for i in range(index.ntotal)]
    with open(os.path.join(OUT, "passages.json"), "w", encoding="utf-8") as f:
        json.dump([{"text": d.page_content, "book": book(d.metadata.get("source", "")), "page": int(d.metadata.get("page", 0)) + 1} for d in docs], f, ensure_ascii=False)
    for name in sorted(os.listdir(OUT)):
        print(name, round(os.path.getsize(os.path.join(OUT, name)) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
