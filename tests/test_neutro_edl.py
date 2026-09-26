"""Tests of the RNEL neural head: reduction to evidential deep learning and component identities."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nel.neutro_edl import NeutroEvidentialHead, TypeLayout, nel_components, neutro_edl_loss  # noqa: E402


def test_reduces_to_edl():
    torch.manual_seed(0)
    lay = TypeLayout(4, gluts=False, ambiguity=False, gap=False)
    e = NeutroEvidentialHead(3, lay)(torch.randn(200, 3))
    c = nel_components(e, lay, W=4.0)
    S = (e + 1).sum(-1)
    assert torch.allclose(c["T"], e / S.unsqueeze(-1), atol=1e-6)
    assert torch.allclose(c["I_vac"], 4 / S, atol=1e-6)


def test_typed_layout_and_shares():
    lay = TypeLayout(2)
    assert lay.M == 5 and lay.glut_idx == [2] and lay.amb_idx == [3] and lay.gap_idx == [4]
    e = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    c = nel_components(e, lay, W=2.0)
    N = e.sum()
    assert torch.allclose(c["I_glut"], torch.tensor([3.0]) / N)
    assert torch.allclose(c["I_amb"], torch.tensor([4.0]) / N)
    assert torch.allclose(c["I_gap"], torch.tensor([5.0]) / N)
    assert torch.allclose(c["T"][0], torch.tensor([1.0 + 3.0, 2.0 + 3.0]) / (N + 2.0))


def test_loss_is_finite_and_decreases_for_observed_type():
    lay = TypeLayout(2)
    y = torch.tensor([2])
    low = neutro_edl_loss(torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]), y, W=2.0)
    high = neutro_edl_loss(torch.tensor([[1.0, 1.0, 20.0, 1.0, 1.0]]), y, W=2.0)
    assert torch.isfinite(low) and high < low
