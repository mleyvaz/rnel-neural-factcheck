"""Experiment 23 runner (see PREREGISTRATION.md, v2 after the adversarial review). CPU, no API.

  python run.py [averitec climate chaos]

Grouped 5-fold cross-validation (groups = normalised claim text, so duplicate claims never cross
folds). In each fold 15% of the training groups are held out for validation (early stopping and
threshold tuning). Hyperparameters are chosen once per dataset and method on fold 0's validation
part and reused for the other folds. Out-of-fold predictions of every method and seed are stored
in results/preds_<dataset>.npz.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from nel.neutro_edl import NeutroEvidentialHead, TypeLayout, nel_components  # noqa: E402

FEAT = HERE / "features"
RES = HERE / "results"
NESTED = "--nested" in sys.argv  # corrected analysis: hyperparameters tuned inside every outer fold
SEEDS = [0, 1, 2, 3, 4]
FOLDS = 5
GRID = {"lr": [1e-3, 3e-4], "wd": [1e-4, 1e-2], "W": [2.0, 5.0], "reg": [0.0, 0.1]}
LAYOUT = TypeLayout(2)  # type index: 0 T, 1 F, 2 glut (C), 3 amb (U), 4 gap (N)
TYPE_OF = np.array(["T", "F", "C", "U", "N"])
RNEL_IDX = {"T": 0, "F": 1, "C": 2, "U": 3, "N": 4}
EPOCHS, PATIENCE, HIDDEN, DROPOUT, BATCH = 80, 10, 256, 0.2, 64
torch.set_num_threads(4)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ----------------------------------------------------------------------------- data
def load(name):
    z = np.load(FEAT / f"{name}.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    logits, cls, owner = z["logits"], z["cls"].astype(np.float32), z["owner"]
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)  # order: contradiction, entailment, neutral
    per = np.concatenate([cls, logits, probs], 1)
    n = len(meta)
    X = np.zeros((n, 2 * per.shape[1]), np.float32)
    Xnli = np.zeros((n, 6), np.float32)  # mean and max of the NLI probabilities
    pair_probs = []
    for i in range(n):
        rows = per[owner == i]
        X[i] = np.concatenate([rows.mean(0), rows.max(0)])
        pp = probs[owner == i]
        Xnli[i] = np.concatenate([pp.mean(0), pp.max(0)])
        pair_probs.append(pp)
    groups = np.array([norm(m["claim"]) if name != "chaos" else str(i) for i, m in enumerate(meta)])
    return meta, X, Xnli, pair_probs, groups


def folds(y, groups):
    sgk = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=0)
    out = []
    for f, (tr_all, te) in enumerate(sgk.split(np.zeros(len(y)), y, groups)):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=f)
        a, b = next(gss.split(tr_all, y[tr_all], groups[tr_all]))
        out.append((f, tr_all[a], tr_all[b], te))
    return out


# ----------------------------------------------------------------------------- models
class Net(nn.Module):
    def __init__(self, d, out, rnel=False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                                  nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT))
        self.head = NeutroEvidentialHead(HIDDEN, LAYOUT) if rnel else nn.Linear(HIDDEN, out)

    def forward(self, x):
        return self.head(self.body(x))


def dirichlet_nll(e, y, W, reg, w):
    """Per-sample form of nel.neutro_edl.neutro_edl_loss, with class weights w (same formula)."""
    M = e.shape[-1]
    alpha = e + W / M
    S = alpha.sum(-1)
    nll = torch.log(S) - torch.log(alpha.gather(1, y.unsqueeze(1)).squeeze(1))
    if reg > 0:
        mask = torch.ones_like(e).scatter_(1, y.unsqueeze(1), 0.0)
        nll = nll + reg * (e * mask).sum(-1) / M
    return (w[y] * nll).sum() / w[y].sum()


def class_weights(ytr, n_out):
    counts = np.bincount(ytr, minlength=n_out).astype(np.float32)
    present = counts > 0
    w = np.where(present, counts.sum() / np.maximum(counts, 1) / present.sum(), 0.0)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(net, method, X, hp, allowed):
    """Primary prediction: argmax of the (typed) posterior alpha/S, or softmax, over allowed labels."""
    net.eval()
    out = net(torch.as_tensor(X))
    mask = torch.full((out.shape[-1],), -1e9)
    mask[allowed] = 0.0
    if method == "softmax":
        p = torch.softmax(out, -1)
        return {"pred": (torch.log(p) + mask).argmax(-1).numpy(), "prob": p.numpy()}
    if method == "edl":
        e = nn.functional.softplus(out)
        alpha = e + 1.0
        p = alpha / alpha.sum(-1, keepdim=True)
        return {"pred": (torch.log(p) + mask).argmax(-1).numpy(), "prob": p.numpy(),
                "vac": (e.shape[-1] / alpha.sum(-1)).numpy()}
    e = out
    comp = nel_components(e, LAYOUT, hp["W"])
    alpha = e + hp["W"] / LAYOUT.M
    p = alpha / alpha.sum(-1, keepdim=True)
    return {"pred": (torch.log(p) + mask).argmax(-1).numpy(), "prob": p.numpy(), "T": comp["T"].numpy(),
            "I_vac": comp["I_vac"].numpy(), "I_glut": comp["I_glut"].numpy(), "I_amb": comp["I_amb"].numpy(),
            "I_gap": comp["I_gap"].numpy(), "N": comp["N"].numpy()}


def train(method, Xtr, ytr, Xva, yva, n_out, hp, seed, allowed):
    torch.manual_seed(seed)
    net = Net(Xtr.shape[1], n_out, rnel=(method == "rnel"))
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    w = class_weights(ytr, n_out)
    Xt, yt, Xv = torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xva)
    best, best_state, wait = -1.0, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        for b in range(0, len(Xt), BATCH):
            ib = perm[b:b + BATCH]
            out = net(Xt[ib])
            if method == "softmax":
                loss = nn.functional.cross_entropy(out, yt[ib], weight=w)
            elif method == "edl":
                loss = dirichlet_nll(nn.functional.softplus(out), yt[ib], float(n_out), hp["reg"], w)
            else:
                loss = dirichlet_nll(out, yt[ib], hp["W"], hp["reg"], w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = predict(net, method, Xv, hp, allowed)["pred"]
        f = f1_score(yva, pv, average="macro", labels=np.unique(yva))
        if f > best + 1e-9:  # ties keep the earlier epoch
            best, best_state, wait = f, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net, best


# ----------------------------------------------------------------------------- non-neural baselines
def rule_predict(pair_probs, te, tc):
    out = []
    for P in pair_probs:
        s, r = (P[:, 1] > te).any(), (P[:, 0] > tc).any()
        out.append("C" if s and r else "T" if s else "F" if r else "N")
    return np.array(out)


def sl_cumulative(P):
    b, d, u = P[0, 1], P[0, 0], max(P[0, 2], 1e-9)
    for k in range(1, len(P)):
        b2, d2, u2 = P[k, 1], P[k, 0], max(P[k, 2], 1e-9)
        den = u + u2 - u * u2
        b, d, u = (b * u2 + b2 * u) / den, (d * u2 + d2 * u) / den, (u * u2) / den
    return b, d, u


def sl_features(P, averaging=False):
    b, d, u = (P[:, 1].mean(), P[:, 0].mean(), P[:, 2].mean()) if averaging else sl_cumulative(P)
    conf = max((P[i, 1] * P[j, 0] + P[i, 0] * P[j, 1]) for i in range(len(P)) for j in range(len(P))) \
        if len(P) > 1 else 0.0
    return b, d, u, conf


def sl_predict(feats, tconf, tu):
    return np.array(["C" if c > tconf else "N" if u > tu else ("T" if b >= d else "F") for b, d, u, c in feats])


def tune(fn, grid, idx, y):
    best = (-1, None)
    for g in grid:
        f = f1_score(y[idx], fn(g, idx), average="macro", labels=sorted(set(y[idx])))
        if f > best[0]:
            best = (f, g)
    return best[1]


# ----------------------------------------------------------------------------- main
def run_dataset(name):
    meta, X, Xnli, pair_probs, groups = load(name)
    y_str = np.array([m["type"] for m in meta])
    classes = [c for c in "TFCNU" if c in set(y_str)]
    flat_idx = {c: i for i, c in enumerate(classes)}
    y_flat = np.array([flat_idx[c] for c in y_str])
    y_rnel = np.array([RNEL_IDX[c] for c in y_str])
    y_rnel_neeU = np.array([RNEL_IDX["U" if c == "N" else c] for c in y_str])
    allowed_rnel = sorted({RNEL_IDX[c] for c in classes})
    allowed_rnel_neeU = sorted({RNEL_IDX["U" if c == "N" else c] for c in classes})
    allowed_flat = list(range(len(classes)))
    n = len(meta)
    fs = folds(y_flat, groups)
    store = {"y_str": y_str, "classes": np.array(classes), "groups": groups,
             "n_evidence": np.array([len(p) for p in pair_probs]), "fold": np.zeros(n, int)}
    if name == "chaos":
        store["entropy"] = np.array([m["entropy"] for m in meta])
        store["label_dist"] = np.array([m["label_dist"] for m in meta])
    runs = []
    variants = [("softmax", y_flat, len(classes), allowed_flat), ("edl", y_flat, len(classes), allowed_flat),
                ("rnel", y_rnel, LAYOUT.M, allowed_rnel)]
    if name != "chaos":
        variants.append(("rnel_neeU", y_rnel_neeU, LAYOUT.M, allowed_rnel_neeU))
    sl_cum = [sl_features(P) for P in pair_probs]
    sl_avg = [sl_features(P, averaging=True) for P in pair_probs]
    chosen = {}
    oof = {}
    for f, tr, va, te in fs:
        if NESTED:
            chosen = {}  # tune again on this fold's own validation split only
        store["fold"][te] = f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Z = (X - mu) / sd
        # capacity-light learned baseline: logistic regression on pooled NLI probabilities
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xnli[tr], y_flat[tr])
        oof.setdefault("logreg_pred", np.empty(n, object))[te] = np.array(classes)[lr.predict(Xnli[te])]
        if name != "chaos":
            grid = [(a, b) for a in np.linspace(0.3, 0.95, 14) for b in np.linspace(0.3, 0.95, 14)]
            g = tune(lambda g, i: rule_predict([pair_probs[k] for k in i], *g), grid, va, y_str)
            oof.setdefault("rule_pred", np.empty(n, object))[te] = rule_predict([pair_probs[k] for k in te], *g)
            grid2 = [(a, b) for a in np.linspace(0.05, 0.9, 18) for b in np.linspace(0.05, 0.95, 19)]
            slth = {}
            for tag, feats in (("sl", sl_cum), ("slavg", sl_avg)):
                g2 = tune(lambda g, i: sl_predict([feats[k] for k in i], *g), grid2, va, y_str)
                slth[tag] = list(map(float, g2))
                oof.setdefault(f"{tag}_pred", np.empty(n, object))[te] = sl_predict([feats[k] for k in te], *g2)
            runs.append({"dataset": name, "fold": f, "rule": list(map(float, g)), "sl_thresholds": slth})
        for method, yy, n_out, allowed in variants:
            base = "rnel" if method.startswith("rnel") else method
            if method not in chosen:  # hyperparameters chosen on fold 0 validation, reused
                keys = {"softmax": ["lr", "wd"], "edl": ["lr", "wd", "reg"]}.get(base, list(GRID))
                best = (-1, None)
                for vals in itertools.product(*[GRID[k] for k in keys]):
                    hp = {"W": 2.0, "reg": 0.0, **dict(zip(keys, vals))}
                    _, sc = train(base, Z[tr], yy[tr], Z[va], yy[va], n_out, hp, 0, allowed)
                    if sc > best[0]:
                        best = (sc, hp)
                chosen[method] = best[1]
                runs.append({"dataset": name, "fold": f, "method": method, "hp": best[1], "val_macro_f1": best[0]})
            hp = chosen[method]
            for seed in SEEDS:
                net, _ = train(base, Z[tr], yy[tr], Z[va], yy[va], n_out, hp, seed, allowed)
                out = predict(net, base, Z[te], hp, allowed)
                for k, v in out.items():
                    key = f"{method}_seed{seed}_{k}"
                    if key not in oof:
                        oof[key] = np.zeros((n,) + v.shape[1:], np.float32 if k != "pred" else np.int64)
                    oof[key][te] = v
            print(name, "fold", f, method, "done", flush=True)
    for k, v in oof.items():
        store[k] = v.astype(str) if v.dtype == object else v
    RES.mkdir(exist_ok=True)
    np.savez_compressed(RES / (f"preds_{name}_nested.npz" if NESTED else f"preds_{name}.npz"), **store)
    return runs


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or ["chaos", "climate", "averitec"]
    p = RES / ("runs_nested.json" if NESTED else "runs.json")
    allruns = [r for r in json.loads(p.read_text()) if r["dataset"] not in names] if p.exists() else []
    for nm in names:
        allruns += run_dataset(nm)
        RES.mkdir(exist_ok=True)
        p.write_text(json.dumps(allruns, indent=1))
