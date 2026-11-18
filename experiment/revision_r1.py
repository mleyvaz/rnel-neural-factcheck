"""Additions requested in the first review round of Paper A (NMNC-D-26-00874). Not preregistered.

1. Standard deviation over the 5 training seeds of the out-of-fold macro-F1 and F1 on C
   (softmax, flat EDL, RNEL head; nested-tuning predictions).
2. A TRAINED Subjective Logic baseline: multinomial logistic regression (balanced class weights,
   scaler fitted on the training part of each outer fold, no hyperparameter tuned) on
     SLt1: fused SL opinion (b, d, u) + dissonance 2 min(b, d)
     SLt2: SLt1 + the pairwise conflict score used by the threshold rule of run.py
     SLt3: SLt2 + number of evidence pieces
   with the same grouped folds as run.py. Paired hierarchical bootstrap (claim groups and seeds,
   2,000 draws), as in analyze.py, for RNEL minus each trained SL baseline.

Run:  python revision_r1.py   -> results/revision_r1.json
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import analyze
import run

SEEDS = [0, 1, 2, 3, 4]
B = 2000
f1c = lambda y, p: analyze.f1(y, p, "C")  # noqa: E731


def main():
    out = {}
    rng = np.random.default_rng(20261117)
    for name in ("averitec", "climate"):
        z = dict(np.load(run.RES / f"preds_{name}_nested.npz", allow_pickle=False))
        y = z["y_str"]
        classes = z["classes"]
        res = {}
        # 1. spread over seeds
        for m in ("softmax", "edl", "rnel"):
            mac = [analyze.macro(y, analyze.labels_of(_Z(z), m, s, classes)) for s in SEEDS]
            fc = [f1c(y, analyze.labels_of(_Z(z), m, s, classes)) for s in SEEDS]
            res[f"seed_spread_{m}"] = {"macro_mean": float(np.mean(mac)), "macro_sd": float(np.std(mac, ddof=1)),
                                       "f1c_mean": float(np.mean(fc)), "f1c_sd": float(np.std(fc, ddof=1))}
        # 2. trained SL baselines on the same folds
        meta, X, Xnli, pair_probs, groups = run.load(name)
        ystr = np.array([m["type"] for m in meta])
        assert (ystr == y).all() and (groups == z["groups"]).all()
        cl = [c for c in "TFCNU" if c in set(ystr)]
        y_flat = np.array([cl.index(c) for c in ystr])
        feats = np.array([run.sl_features(P) for P in pair_probs], float)  # b, d, u, conf
        diss = 2 * np.minimum(feats[:, 0], feats[:, 1])
        npieces = np.array([len(P) for P in pair_probs], float)
        mats = {"SLt1": np.column_stack([feats[:, :3], diss]),
                "SLt2": np.column_stack([feats[:, :3], diss, feats[:, 3]]),
                "SLt3": np.column_stack([feats[:, :3], diss, feats[:, 3], npieces])}
        for k, M in mats.items():
            pred = np.empty(len(y), dtype="<U1")
            for f, tr, va, te in run.folds(y_flat, groups):
                assert (z["fold"][te] == f).all()
                trv = np.concatenate([tr, va])
                sc = StandardScaler().fit(M[trv])
                lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(M[trv]), y_flat[trv])
                pred[te] = np.array(cl)[lr.predict(sc.transform(M[te]))]
            z[f"{k}_pred"] = pred
            res[k] = {"macro_f1": analyze.macro(y, pred), "f1c": f1c(y, pred)}
        bt = analyze.Boot(_Z(z), rng)
        analyze.B = B
        for k in mats:
            res[f"rnel_minus_{k}_F1C"] = bt.diff("rnel", k, f1c)
            res[f"rnel_minus_{k}_macro"] = bt.diff("rnel", k, analyze.macro)
        res["SLt1_minus_sl_F1C"] = bt.diff("SLt1", "sl", f1c)
        res["SLt1_minus_sl_macro"] = bt.diff("SLt1", "sl", analyze.macro)
        out[name] = res
        print(name, json.dumps(res, indent=1))
    (run.RES / "revision_r1.json").write_text(json.dumps(out, indent=1))


class _Z(dict):
    """dict with the .files attribute that analyze.labels_of expects from an NpzFile."""

    @property
    def files(self):
        return list(self.keys())


if __name__ == "__main__":
    main()
