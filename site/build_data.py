import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))

import evaluate as ev


def rate(rows):
    n = len(rows)
    k = sum(r["correct"] for r in rows.values())
    lo, hi = ev.wilson(k, n)
    return {"correct": k, "n": n, "low": lo, "high": hi}


def main():
    old = ev.load("mistral7b_original")
    old_plain = ev.load("mistral7b_direct")
    new = ev.load("qwen38_tutor")
    new_plain = ev.load("qwen38_tutor_no_retrieval")
    reasons = {r["id"]: r["reason"] for r in ev.load("qwen38_tutor_reasons").values()}
    qs = ev.questions("test")
    items = []
    for _, row in qs.iterrows():
        i = row["id"]
        items.append({
            "question": row["question"],
            "options": [row[c] for c in ["opa", "opb", "opc", "opd"]],
            "gold": new[i]["gold"],
            "old": old[i]["pred"],
            "new": new[i]["pred"],
            "plain": new_plain[i]["pred"],
            "reason": reasons.get(i, ""),
        })
    only_new = sum(new[i]["correct"] and not old[i]["correct"] for i in new)
    only_old = sum(old[i]["correct"] and not new[i]["correct"] for i in new)
    out = {
        "setups": [
            dict(rate(old_plain), name="Mistral-7B alone", note="no retrieval"),
            dict(rate(old), name="PathoChat as first built", note="Mistral-7B, retrieval, context-only prompt"),
            dict(rate(new), name="PathoChat now", note="Qwen3.8-27B, retrieval, new prompt"),
            dict(rate(new_plain), name="Qwen3.8-27B alone", note="no retrieval"),
        ],
        "paired": {"only_new": only_new, "only_old": only_old, "p": ev.mcnemar(only_new, only_old)},
        "questions": items,
    }
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print([(s["name"], s["correct"]) for s in out["setups"]], out["paired"])


if __name__ == "__main__":
    main()
