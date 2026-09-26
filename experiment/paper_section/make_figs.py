"""Figures and numbers for the empirical subsection (8.6) of the Smarandache-Leyva SL/RNEL paper.

Reads the nested-tuning out-of-fold predictions (results/preds_*_nested.npz) and summary_nested.json.
Definition 8.1 components are recovered exactly from the saved posterior: with alpha = e + W/M and
S = sum(e) + W, e_X / S = alpha_X / S - G / M, where G = W / S is the saved I_vac.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
SEEDS = [0, 1, 2, 3, 4]
TYPES = ["T", "F", "C", "U", "N"]  # RNEL type order in the head (glut = C, amb = U, gap = N)
INK, MUTED = "#1f2937", "#6b7280"
COL = {"T": "#2a78b5", "F": "#c2542d", "C": "#7b4ea3", "U": "#3f9f6b", "N": "#b5892a", "G": "#9aa3ad"}


def architecture():
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    boxes = [
        (0.1, 1.9, "Claim + k evidence\npieces", "one (evidence, claim)\npair per piece", "#f8fafc"),
        (2.3, 1.9, "Frozen NLI encoder", "DeBERTa-v3 cross-encoder\n[CLS] + 3 NLI logits", "#eef3fa"),
        (4.5, 1.9, "Pooling over pieces", "mean and max\n(1,548 features)", "#eef3fa"),
        (6.7, 1.9, "MLP", "256 - 256, ReLU,\ndropout 0.2", "#f3eef8"),
        (8.9, 1.9, "RNEL head", "softplus evidence\ne_T, e_F, e_C, e_U, e_N", "#f3eef8"),
    ]
    for x, w, title, sub, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.25), w, 1.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=INK, lw=1.2))
        ax.text(x + w / 2, 2.45, title, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
        ax.text(x + w / 2, 1.75, sub, ha="center", va="center", fontsize=8.8, color=INK)
    for x in (2.0, 4.2, 6.4, 8.6):
        ax.annotate("", xy=(x + 0.3, 2.05), xytext=(x, 2.05), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.add_patch(FancyBboxPatch((2.3, 0.1), 8.5, 0.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#eef7f1", ec=INK, lw=1.2))
    ax.text(6.55, 0.5, "Definition 8.1:  T = e_T/S,  F = e_F/S,  C = e_C/S,  U = e_U/S,  N = e_N/S,  G = W/S,"
            "   S = e_T + e_F + e_C + e_U + e_N + W", ha="center", va="center", fontsize=8.6, color=INK)
    ax.annotate("", xy=(9.85, 0.92), xytext=(9.85, 1.23), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    fig.tight_layout()
    fig.savefig(HERE / "fig_rnel_architecture.png", dpi=220)
    plt.close(fig)


def def81_components(z):
    """Mean over seeds of the Definition 8.1 tuple (T, F, C, U, N, G) for every claim."""
    out = []
    for s in SEEDS:
        p = z[f"rnel_seed{s}_prob"]
        G = z[f"rnel_seed{s}_I_vac"]
        comp = p - G[:, None] / p.shape[1]
        out.append(np.concatenate([comp, G[:, None]], 1))
    return np.mean(out, 0)  # columns T, F, C, U, N, G


def components_fig(numbers):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, (name, title) in zip(axes, (("averitec", "AVeriTeC"), ("climate", "Climate-FEVER"))):
        z = np.load(RES / f"preds_{name}_nested.npz", allow_pickle=False)
        y = z["y_str"]
        comp = def81_components(z)
        labels = ["T", "F", "C", "N"]
        names = {"T": "supported", "F": "refuted", "C": "conflicting", "N": "not enough\nevidence"}
        bottom = np.zeros(4)
        means = {}
        for j, k in enumerate(["T", "F", "C", "U", "N", "G"]):
            v = np.array([comp[y == c, j].mean() for c in labels])
            means[k] = v
            ax.bar(range(4), v, bottom=bottom, color=COL[k], edgecolor="white", linewidth=1.5, width=0.62,
                   label={"T": "T", "F": "F", "C": "C (contradiction)", "U": "U (undetermined)",
                          "N": "N (neither)", "G": "G (no evidence)"}[k])
            bottom += v
        ax.set_xticks(range(4), [names[c] for c in labels], fontsize=9)
        ax.set_title(f"{title}: mean RNEL tuple by true label", fontsize=10, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
        numbers[f"{name}_def81_by_class"] = {c: {k: float(means[k][i]) for k in means} for i, c in enumerate(labels)}
    axes[0].set_ylabel("share of the evaluation", color=INK)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(HERE / "fig_rnel_components.png", dpi=220)
    plt.close(fig)


def results_fig(summary, numbers):
    methods = [("rnel", "RNEL head"), ("edl", "flat EDL"), ("softmax", "flat softmax"), ("logreg", "logistic (NLI)"),
               ("sl", "SL cumulative"), ("slavg", "SL averaging"), ("rule", "NLI rule")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    for ax, (name, title) in zip(axes, (("averitec", "AVeriTeC (C prevalence 6.5%)"),
                                         ("climate", "Climate-FEVER (C prevalence 10.0%)"))):
        d = summary[name]
        v = [d["f1_per_class"][m]["C"] for m, _ in methods]
        colors = [COL["C"] if m == "rnel" else ("#b9a6cf" if m in ("edl", "softmax", "logreg") else "#cfd4da")
                  for m, _ in methods]
        ax.barh(range(len(methods))[::-1], v, color=colors, height=0.62)
        for i, val in zip(range(len(methods))[::-1], v):
            ax.text(val + 0.004, i, f"{val:.3f}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(methods))[::-1], [lab for _, lab in methods], fontsize=9)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("F1 on the conflicting class (C)", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.set_xlim(0, 0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_rnel_results.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    summary = json.loads((RES / "summary_nested.json").read_text())
    numbers = {}
    architecture()
    components_fig(numbers)
    results_fig(summary, numbers)
    (HERE / "section_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))
