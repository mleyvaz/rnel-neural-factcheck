"""Experiment 23: encode every (evidence, claim) pair with a frozen NLI cross-encoder.

Output, one file per dataset in features/: for every pair the 3 NLI logits (order: contradiction,
entailment, neutral), the [CLS] vector of the last layer (768), the claim index and the claim label.
Run: python features.py [averitec|climate|chaos]   (CPU; resumable per dataset)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
OUT = HERE / "features"
MODEL = "cross-encoder/nli-deberta-v3-base"
TYPES = ["T", "F", "C", "N", "U"]  # T, F, glut, gap, ambiguous

AVERITEC = {"Supported": "T", "Refuted": "F", "Conflicting Evidence/Cherrypicking": "C",
            "Not Enough Evidence": "N"}
CLIMATE = {"SUPPORTS": "T", "REFUTES": "F", "DISPUTED": "C", "NOT_ENOUGH_INFO": "N"}
CHAOS = {"e": "T", "c": "F", "n": "N", "entailment": "T", "contradiction": "F", "neutral": "N"}


def averitec():
    claims = []
    for split in ("train", "dev"):
        for x in json.load(open(RAW / f"averitec_{split}.json", encoding="utf-8")):
            ev = []
            for q in x["questions"]:
                for a in q["answers"]:
                    txt = f"Q: {q['question']} A: {a['answer']}"
                    if a.get("boolean_explanation"):
                        txt += f" {a['boolean_explanation']}"
                    ev.append(txt)
            claims.append({"claim": x["claim"], "evidence": ev, "type": AVERITEC[x["label"]],
                           "split": split})
    return claims


def climate():
    claims = []
    for line in open(RAW / "climate-fever.jsonl", encoding="utf-8"):
        x = json.loads(line)
        claims.append({"claim": x["claim"], "evidence": [e["evidence"] for e in x["evidences"]],
                       "type": CLIMATE[x["claim_label"]], "split": "all"})
    return claims


def chaos():
    claims = []
    for line in open(RAW / "chaos_mnli.jsonl", encoding="utf-8"):
        x = json.loads(line)
        counts = x["label_count"]  # order e, n, c in ChaosNLI
        top = max(counts) / sum(counts)
        typ = "U" if top < 0.60 else CHAOS[x["majority_label"]]
        claims.append({"claim": x["hypothesis"], "evidence": [x["premise"]], "type": typ,
                       "split": "all", "entropy": x["entropy"], "label_dist": x["label_dist"],
                       "majority": CHAOS[x["majority_label"]]})
    return claims


LOADERS = {"averitec": averitec, "climate": climate, "chaos": chaos}


@torch.no_grad()
def encode(name: str, batch: int = 16) -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.npz"
    if path.exists():
        print(name, "exists, skipped")
        return
    claims = LOADERS[name]()
    pairs, owner = [], []
    for i, c in enumerate(claims):
        for e in c["evidence"]:
            pairs.append((e, c["claim"]))
            owner.append(i)
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    logits, cls = [], []
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        enc = tok([p[0] for p in chunk], [p[1] for p in chunk], truncation=True, max_length=256,
                  padding=True, return_tensors="pt")
        out = model(**enc, output_hidden_states=True)
        logits.append(out.logits.float().numpy())
        cls.append(out.hidden_states[-1][:, 0].float().numpy())
        if (s // batch) % 50 == 0:
            print(name, s, "/", len(pairs), flush=True)
    meta = [{k: v for k, v in c.items() if k != "evidence"} for c in claims]
    np.savez_compressed(path, logits=np.concatenate(logits), cls=np.concatenate(cls).astype(np.float16),
                        owner=np.array(owner), meta=json.dumps(meta))
    print(name, "done", len(pairs), "pairs", len(claims), "claims")


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["chaos", "climate", "averitec"]):
        encode(n)
