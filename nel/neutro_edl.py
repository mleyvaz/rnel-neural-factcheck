"""Neutrosophic evidential deep learning: a multinomial NEL head for neural networks.

The head outputs non-negative evidence for every observation type of the multinomial
NEL frame (nel/multinomial.py) with K classes:

    K crisp types      "it is class i"
    P glut types       "it is classes i and j at once", one per pair (optional)
    P ambiguous types  "it is class i or class j", one per pair
    1 gap type         "it is none of the classes" (e.g. an input from outside the task)

with P = K(K-1)/2. With evidence e (M types), prior weight W spread uniformly over the
types (alpha = e + W/M) and S = sum(alpha), the NEL components are those of the
multinomial NEL value with counts e:

    T_i    = (e_i + sum of glut evidence on pairs containing i) / (N + W),   N = sum(e)
    I_vac  = W / (N + W)
    I_glut, I_amb, I_gap = shares of glut, ambiguous and gap evidence in N

Loss: negative log-likelihood of the observed typed label under the expected
categorical distribution alpha / S (the Dirichlet-multinomial likelihood of a single
observation), plus the evidence regulariser of Sensoy et al. (2018) on the types that
were not observed.

Reduction (tested): with only the K crisp types and W = K, alpha_i = e_i + 1, and
(T_i, I_vac) = (e_i / S, K / S), which is the belief and uncertainty of evidential deep
learning (Sensoy et al., 2018).
"""
from __future__ import annotations

from itertools import combinations

import torch
from torch import nn


class TypeLayout:
    """Index layout of the observation types for K classes."""

    def __init__(self, K: int, gluts: bool = True, ambiguity: bool = True, gap: bool = True):
        self.K = K
        self.pairs = list(combinations(range(K), 2))
        self.glut_idx = list(range(K, K + len(self.pairs))) if gluts else []
        start = K + len(self.glut_idx)
        self.amb_idx = list(range(start, start + len(self.pairs))) if ambiguity else []
        self.gap_idx = [start + len(self.amb_idx)] if gap else []
        self.M = K + len(self.glut_idx) + len(self.amb_idx) + len(self.gap_idx)
        # membership of each glut type in each class, shape (n_glut, K)
        member = torch.zeros(len(self.glut_idx), K)
        for g, (i, j) in enumerate(self.pairs[:len(self.glut_idx)]):
            member[g, i] = member[g, j] = 1.0
        self.glut_member = member

    def glut_type(self, i: int, j: int) -> int:
        return self.glut_idx[self.pairs.index(tuple(sorted((i, j))))]

    def amb_type(self, i: int, j: int) -> int:
        return self.amb_idx[self.pairs.index(tuple(sorted((i, j))))]


class NeutroEvidentialHead(nn.Module):
    """Linear map from features to non-negative typed evidence (softplus)."""

    def __init__(self, in_dim: int, layout: TypeLayout):
        super().__init__()
        self.layout = layout
        self.linear = nn.Linear(in_dim, layout.M)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.linear(h))


def nel_components(e: torch.Tensor, layout: TypeLayout, W: float) -> dict[str, torch.Tensor]:
    """Multinomial NEL components from typed evidence e of shape (batch, M)."""
    K = layout.K
    N = e.sum(-1)
    denom = N + W
    T = e[:, :K].clone()
    if layout.glut_idx:
        T = T + e[:, layout.glut_idx] @ layout.glut_member.to(e)
    T = T / denom.unsqueeze(-1)
    safe_N = N.clamp_min(1e-12)
    share = lambda idx: (e[:, idx].sum(-1) / safe_N) if idx else torch.zeros_like(N)  # noqa: E731
    return {"T": T, "I_vac": W / denom, "I_glut": share(layout.glut_idx),
            "I_amb": share(layout.amb_idx), "I_gap": share(layout.gap_idx), "N": N}


def neutro_edl_loss(e: torch.Tensor, y: torch.Tensor, W: float, reg: float = 0.0) -> torch.Tensor:
    """Mean negative log-likelihood of the observed type y (long, shape (batch,)) under
    alpha / S with alpha = e + W / M, plus ``reg`` times the evidence on the other types."""
    M = e.shape[-1]
    alpha = e + W / M
    S = alpha.sum(-1)
    nll = torch.log(S) - torch.log(alpha.gather(1, y.unsqueeze(1)).squeeze(1))
    if reg > 0:
        mask = torch.ones_like(e).scatter_(1, y.unsqueeze(1), 0.0)
        nll = nll + reg * (e * mask).sum(-1) / M
    return nll.mean()
