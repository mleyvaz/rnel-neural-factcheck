"""Experiment 23 analysis (PREREGISTRATION.md v2): R1, R2 (confirmatory) and descriptive D1-D4.

Out-of-fold predictions over the whole dataset. Bootstrap (2,000 draws): claim groups (normalised
claim text) are resampled with replacement, and training seeds are resampled with replacement
inside each draw (hierarchical); the statistic for a method is the mean over the drawn seeds.
Differences between methods are paired (same groups, same seed draw).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
RES = HERE / "results"
SUF = "_nested" if "--nested" in sys.argv else ""  # corrected (nested-tuning) analysis
B = 2000
SEEDS = [0, 1, 2, 3, 4]
TYPE_OF = np.array(["T", "F", "C", "U", "N"])
LABELS = ["T", "F", "C", "N"]


def f1(y, p, c):
    tp = np.sum((p == c) & (y == c))
    fp = np.sum((p == c) & (y != c))
    fn = np.sum((p != c) & (y == c))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def macro(y, p):
    return float(np.mean([f1(y, p, c) for c in LABELS if c in set(y)]))


def labels_of(z, method, seed, classes):
    if f"{method}_pred" in z.files:
        return z[f"{method}_pred"]
    p = z[f"{method}_seed{seed}_pred"]
    if method == "rnel_neeU":  # NEE was trained as the ambiguous type: an amb prediction means N
        return np.where(TYPE_OF[p] == "U", "N", TYPE_OF[p])
    return TYPE_OF[p] if method.startswith("rnel") else classes[p]


def seeded(method):
    return method in ("softmax", "edl", "rnel", "rnel_neeU")


class Boot:
    def __init__(self, z, rng):
        self.z, self.rng = z, rng
        self.y = z["y_str"]
        self.classes = z["classes"]
        g = z["groups"]
        self.ug, self.ginv = np.unique(g, return_inverse=True)
        self.members = [np.where(self.ginv == k)[0] for k in range(len(self.ug))]
        self.cache = {}

    def pred(self, m, s):
        key = (m, s if seeded(m) else 0)
        if key not in self.cache:
            self.cache[key] = labels_of(self.z, m, key[1], self.classes)
        return self.cache[key]

    def value(self, m, stat, idx, seeds):
        ss = seeds if seeded(m) else [0]
        return float(np.mean([stat(self.y[idx], self.pred(m, s)[idx]) for s in ss]))

    def draw(self):
        gs = self.rng.integers(0, len(self.ug), len(self.ug))
        idx = np.concatenate([self.members[k] for k in gs])
        seeds = list(self.rng.choice(SEEDS, len(SEEDS)))
        return idx, seeds

    def diff(self, m1, m2, stat):
        allidx = np.arange(len(self.y))
        point = self.value(m1, stat, allidx, SEEDS) - self.value(m2, stat, allidx, SEEDS)
        bs = []
        for _ in range(B):
            idx, seeds = self.draw()
            bs.append(self.value(m1, stat, idx, seeds) - self.value(m2, stat, idx, seeds))
        return float(point), tuple(np.percentile(bs, [2.5, 97.5]).tolist())


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return 0.0 if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])


def main():
    rng = np.random.default_rng(20260926)
    summary = {}
    f1c = lambda y, p: f1(y, p, "C")  # noqa: E731
    for name in ("averitec", "climate"):
        z = np.load(RES / f"preds_{name}{SUF}.npz", allow_pickle=False)
        bt = Boot(z, rng)
        y = bt.y
        methods = ["softmax", "edl", "rnel", "rnel_neeU", "logreg", "rule", "sl", "slavg"]
        d = {"n": int(len(y)), "n_groups": int(len(bt.ug)), "class_counts": {c: int((y == c).sum()) for c in LABELS}}
        allidx = np.arange(len(y))
        d["macro_f1"] = {m: bt.value(m, macro, allidx, SEEDS) for m in methods}
        d["f1_per_class"] = {m: {c: bt.value(m, lambda yy, pp, c=c: f1(yy, pp, c), allidx, SEEDS) for c in LABELS}
                             for m in methods}
        d["R1_rnel_minus_edl_macro"] = bt.diff("rnel", "edl", macro)
        d["R1_neeU_minus_edl_macro"] = bt.diff("rnel_neeU", "edl", macro)
        d["R1_holds"] = d["R1_rnel_minus_edl_macro"][1][0] > -0.02
        d["R1_neeU_holds"] = d["R1_neeU_minus_edl_macro"][1][0] > -0.02
        for other in ("softmax", "rule"):
            d[f"R2_rnel_minus_{other}_F1C"] = bt.diff("rnel", other, f1c)
            d[f"R2_neeU_minus_{other}_F1C"] = bt.diff("rnel_neeU", other, f1c)
        d["R2_holds_on_dataset"] = all(d[f"R2_rnel_minus_{o}_F1C"][1][0] > 0 for o in ("softmax", "rule"))
        d["R2_neeU_holds_on_dataset"] = all(d[f"R2_neeU_minus_{o}_F1C"][1][0] > 0 for o in ("softmax", "rule"))
        for a, b in (("rnel", "edl"), ("rnel", "sl"), ("rnel", "slavg"), ("rnel", "logreg")):
            d[f"desc_{a}_minus_{b}_F1C"] = bt.diff(a, b, f1c)
        # PR-AUC for C from the posterior probability of C (seed mean)
        yC = (y == "C").astype(int)
        cidx = list(z["classes"]).index("C")
        d["prauc_C"] = {
            "softmax": float(np.mean([average_precision_score(yC, z[f"softmax_seed{s}_prob"][:, cidx]) for s in SEEDS])),
            "edl": float(np.mean([average_precision_score(yC, z[f"edl_seed{s}_prob"][:, cidx]) for s in SEEDS])),
            "rnel": float(np.mean([average_precision_score(yC, z[f"rnel_seed{s}_prob"][:, 2]) for s in SEEDS])),
            "prevalence": float(yC.mean())}
        p = labels_of(z, "rnel", 0, z["classes"])
        d["confusion_rnel_seed0"] = {t: {q: int(np.sum((y == t) & (p == q))) for q in ["T", "F", "C", "N", "U"]} for t in LABELS}
        comp = {}
        for c in LABELS:
            mk = y == c
            comp[c] = {k: float(np.mean([z[f"rnel_seed{s}_{k}"][mk].mean() for s in SEEDS]))
                       for k in ("I_vac", "I_glut", "I_amb", "I_gap")}
            T = np.stack([z[f"rnel_seed{s}_T"][mk] for s in SEEDS])
            comp[c]["T_true"], comp[c]["T_false"] = float(T[..., 0].mean()), float(T[..., 1].mean())
            comp[c]["share_T_plus_F_gt_1"] = float((T.sum(-1) > 1).mean())
        d["D2_components_by_class_rnel"] = comp
        # D3: each score ranks the errors of its OWN method (seed 0); evidence count against RNEL's errors
        def own_err(m):
            return (labels_of(z, m, 0, z["classes"]) != y).astype(int)
        sm = z["softmax_seed0_prob"]
        d["D3_error_detection_auroc"] = {
            "rnel_I_vac_on_rnel_errors": float(roc_auc_score(own_err("rnel"), z["rnel_seed0_I_vac"])),
            "edl_vac_on_edl_errors": float(roc_auc_score(own_err("edl"), z["edl_seed0_vac"])),
            "softmax_1_minus_msp_on_softmax_errors": float(roc_auc_score(own_err("softmax"), 1 - sm.max(1))),
            "minus_n_evidence_on_rnel_errors": float(roc_auc_score(own_err("rnel"), -z["n_evidence"]))}
        d["I_amb_range_by_class"] = [float(min(v["I_amb"] for v in comp.values())), float(max(v["I_amb"] for v in comp.values()))]
        summary[name] = d
    summary["R2_holds_both"] = summary["averitec"]["R2_holds_on_dataset"] and summary["climate"]["R2_holds_on_dataset"]

    # D1 (descriptive): ChaosNLI ambiguity against human entropy, fair comparators
    z = np.load(RES / f"preds_chaos{SUF}.npz", allow_pickle=False)
    H = z["entropy"]
    cls = list(z["classes"])
    u = cls.index("U")
    r = {"rnel_I_amb": [], "rnel_p_amb": [], "rnel_I_glut": [], "softmax_pU": [], "softmax_entropy": [], "edl_pU": []}
    for s in SEEDS:
        r["rnel_I_amb"].append(spearman(z[f"rnel_seed{s}_I_amb"], H))
        r["rnel_p_amb"].append(spearman(z[f"rnel_seed{s}_prob"][:, 3], H))
        r["rnel_I_glut"].append(spearman(z[f"rnel_seed{s}_I_glut"], H))
        ps = z[f"softmax_seed{s}_prob"]
        r["softmax_pU"].append(spearman(ps[:, u], H))
        r["softmax_entropy"].append(spearman(-(ps * np.log(np.clip(ps, 1e-12, 1))).sum(1), H))
        r["edl_pU"].append(spearman(z[f"edl_seed{s}_prob"][:, u], H))
    summary["D1_chaos_spearman_with_human_entropy"] = {k: float(np.mean(v)) for k, v in r.items()}
    yc = z["y_str"]
    summary["D1_chaos_macro_f1"] = {m: float(np.mean([macro(yc, labels_of(z, m, s, z["classes"])) for s in SEEDS]))
                                    for m in ("softmax", "edl", "rnel")}

    # D4: reduction check moved to implementation validation (tests/test_neutro_edl.py); rerun here
    import torch
    from nel.neutro_edl import NeutroEvidentialHead, TypeLayout, nel_components
    torch.manual_seed(0)
    lay = TypeLayout(4, gluts=False, ambiguity=False, gap=False)
    e = NeutroEvidentialHead(3, lay)(torch.randn(500, 3))
    c = nel_components(e, lay, W=4.0)
    S = (e + 1).sum(-1)
    summary["D4_reduction_max_abs_error"] = max(float((c["T"] - e / S.unsqueeze(-1)).abs().max()),
                                                float((c["I_vac"] - 4 / S).abs().max()))
    (RES / f"summary{SUF}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
