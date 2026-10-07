"""Theory ablation for Paper A (Smarandache & Leyva-Vazquez, SL as a retract of neutrosophic logic).

Question: after SL cumulative fusion, is information about the conflict class lost that the
individual evidence opinions still carry (Proposition "SL measures are constant on fibres")?

Design (version 2, after an adversarial re-check of version 1):
  * same claims, same grouped folds as run.py (classes encoded in run.py's T,F,C,N order);
  * binary task: conflict class C against the rest; threshold-free metrics (AUROC and average
    precision, whose chance level is the prevalence of C), from out-of-fold probabilities of a
    logistic regression with balanced class weights, scaler fitted on training data only;
  * every feature set contains the number of evidence pieces, a confound (conflicting claims
    have more pieces in AVeriTeC);
  * paired cluster bootstrap over claim groups (2,000 draws) for differences.
Feature sets
  F0  number of pieces
  A1  F0 + fused SL opinion (b, d, u) + its dissonance 2 min(b, d)
  A2  A1 + fused contradiction C = largest pairwise SL degree of conflict between pieces (a=0.5)
  A3  A1 + largest per-piece belief and disbelief (max b_i, max d_i)
  A4  A1 + C + max b_i + max d_i
Run:  python theory_ablation.py   -> results/theory_ablation.json
"""
import json
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import run  # reuses load(), folds(), sl_cumulative() of the experiment

A = 0.5
B_DRAWS = 2000


def features(P):
    """P rows: (contradiction, entailment, neutral) probabilities of each evidence piece."""
    b, d, u = run.sl_cumulative(P)
    C = 0.0
    for i, j in combinations(range(len(P)), 2):
        bi, di, ui = P[i, 1], P[i, 0], P[i, 2]
        bj, dj, uj = P[j, 1], P[j, 0], P[j, 2]
        C = max(C, abs((bi + A * ui) - (bj + A * uj)) * (1 - ui) * (1 - uj))
    return {"n": [float(len(P))], "fused": [b, d, u, 2 * min(b, d)], "C": [C],
            "ext": [float(P[:, 1].max()), float(P[:, 0].max())]}


SETS = {"F0": ["n"], "A1": ["n", "fused"], "A2": ["n", "fused", "C"],
        "A3": ["n", "fused", "ext"], "A4": ["n", "fused", "C", "ext"]}


def boot(y, s1, s2, groups, rng):
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    da, dp = [], []
    for _ in range(B_DRAWS):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        da.append(roc_auc_score(y[idx], s2[idx]) - roc_auc_score(y[idx], s1[idx]))
        dp.append(average_precision_score(y[idx], s2[idx]) - average_precision_score(y[idx], s1[idx]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"dAUROC": float(np.mean(da)), "dAUROC_ci": q(da), "dAP": float(np.mean(dp)), "dAP_ci": q(dp)}


def main():
    out = {}
    rng = np.random.default_rng(0)
    for name in ["averitec", "climate"]:
        meta, X, Xnli, pair_probs, groups = run.load(name)
        ystr = np.array([m["type"] for m in meta])
        classes = [c for c in "TFCNU" if c in set(ystr)]  # run.py's encoding -> same folds
        y_flat = np.array([classes.index(c) for c in ystr])
        y = (ystr == "C").astype(int)
        feats = [features(P) for P in pair_probs]
        mats = {k: np.array([sum((f[g] for g in groups_), []) for f in feats])
                for k, groups_ in SETS.items()}
        scores = {k: np.zeros(len(y)) for k in SETS}
        for f, tr, va, te in run.folds(y_flat, groups):
            trv = np.concatenate([tr, va])
            for k, M in mats.items():
                sc = StandardScaler().fit(M[trv])
                lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(M[trv]), y[trv])
                scores[k][te] = lr.predict_proba(sc.transform(M[te]))[:, 1]
        res = {"prevalence_C": float(y.mean()),
               "mean_pieces_C_vs_other": [float(mats["F0"][y == 1].mean()), float(mats["F0"][y == 0].mean())],
               "mean_C_C_vs_other": [float(mats["A2"][y == 1, -1].mean()), float(mats["A2"][y == 0, -1].mean())]}
        for k, s in scores.items():
            res[k] = {"AUROC": float(roc_auc_score(y, s)), "AP": float(average_precision_score(y, s))}
        for a, b in [("A1", "A2"), ("A1", "A3"), ("A1", "A4"), ("A3", "A4"), ("F0", "A1")]:
            res[f"{b}-{a}"] = boot(y, scores[a], scores[b], groups, rng)
        out[name] = res
        print(name, json.dumps(res, indent=1))
    (run.RES / "theory_ablation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
