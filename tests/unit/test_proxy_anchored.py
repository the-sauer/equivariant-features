"""The proxy-anchored objectives: `ProxyAnchoredSupCon` and `ProxyAnchoredFPR95`.

Both restrict their pair set to pairs with one proxy endpoint (the board's own
rendering, flagged `is_pdf`) and one image patch — the loss as a rows/columns split of
the logit matrix, the metric as a cross-set mask over the distance matrix.

The restriction is easy to implement *almost* right, and every near-miss is silent:

* dropping the PDF columns from the positives but leaving them in the log-sum-exp
  denominator still lets the loss be lowered by separating two renderings — a term the
  outer sum no longer asks for;
* keeping the non-PDF rows in the final mean divides by the wrong count, so the reported
  number moves with the batch's PDF fraction rather than with the descriptor;
* a PDF patch whose positives all fall outside the batch has an empty numerator, and the
  upstream edge-case path turns that into a 0 term rather than dropping the row.

These tests pin the restricted loss against a straight-from-the-formula reference, and
pin that `is_proxy=None` still reproduces upstream SupCon exactly.
"""

import pytest
import torch
from omegaconf import OmegaConf

from aef.train.descriptor import process_batch_blobs
from aef.evaluate import fpr, fpr_from_distances
from aef.train.losses import (SupConLoss, ProxyAnchoredFPR95, ProxyAnchoredSupCon)

TAU = 0.07


def _reference(f, labels, is_pdf=None):
    """The loss straight from the paper, written as loops over the index sets."""
    n = f.size(0)
    sim = f @ f.T / TAU
    rows = range(n) if is_pdf is None else [i for i in range(n) if is_pdf[i]]
    terms = []
    for i in rows:
        if is_pdf is None:
            a_set = [a for a in range(n) if a != i]
        else:
            a_set = [a for a in range(n) if not is_pdf[a] and a != i]
        p_set = [p for p in a_set if labels[p] == labels[i]]
        if not p_set:
            # Upstream counts an empty numerator as a 0 term; the restricted loss drops
            # the row instead (see the module docstring).
            if is_pdf is None:
                terms.append(0.0)
            continue
        lse = torch.logsumexp(torch.stack([sim[i, a] for a in a_set]), 0)
        terms.append(-sum(float(sim[i, p] - lse) for p in p_set) / len(p_set))
    return sum(terms) / len(terms)


def _batch(seed=0, n=9, d=8):
    g = torch.Generator().manual_seed(seed)
    f = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1)
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3])
    #                      D        D     D        D  <- the last one has no partner
    is_pdf = torch.tensor([1, 0, 0, 1, 0, 1, 0, 0, 1])
    return f, labels, is_pdf


def _loss(**kw):
    return SupConLoss(temperature=TAU, base_temperature=TAU, **kw)


def test_restricted_loss_matches_the_reference():
    f, labels, is_pdf = _batch()
    got = _loss()(f.unsqueeze(1), labels=labels, is_proxy=is_pdf)
    assert float(got) == pytest.approx(_reference(f, labels, is_pdf), abs=1e-5)


def test_without_the_flag_it_is_still_upstream_supcon():
    f, labels, _ = _batch()
    got = _loss()(f.unsqueeze(1), labels=labels)
    assert float(got) == pytest.approx(_reference(f, labels), abs=1e-5)


def test_pdf_columns_leave_the_denominator_too():
    # Move ONE PDF patch that is nobody's positive. It appears in no numerator either
    # way, so if the loss changes, it is still sitting in some row's log-sum-exp.
    f, labels, is_pdf = _batch()
    before = _loss()(f.unsqueeze(1), labels=labels, is_proxy=is_pdf)
    f2 = f.clone()
    f2[8] = torch.nn.functional.normalize(-f[8] + 0.3 * f[0], dim=0)   # label 3, is_pdf
    after = _loss()(f2.unsqueeze(1), labels=labels, is_proxy=is_pdf)
    assert float(before) == pytest.approx(float(after), abs=1e-6)


def test_image_image_pairs_leave_the_loss():
    # Two image patches of the same track (indices 1 and 2): under plain SupCon they
    # form a positive pair, under the restriction they only ever appear as columns.
    f, labels, is_pdf = _batch()
    idx = [0, 1, 2]                       # one PDF patch + its two observations
    f, labels, is_pdf = f[idx], labels[idx], is_pdf[idx]
    restricted = float(_loss()(f.unsqueeze(1), labels=labels, is_proxy=is_pdf))
    # The value must equal a single-row loss: the PDF row against the two image columns.
    sim = f @ f.T / TAU
    lse = torch.logsumexp(torch.stack([sim[0, 1], sim[0, 2]]), 0)
    expect = -float(sim[0, 1] - lse + sim[0, 2] - lse) / 2
    assert restricted == pytest.approx(expect, abs=1e-5)


def test_rows_without_a_positive_are_dropped_not_counted_as_zero():
    # Index 8 is a PDF patch of a track with no observation in the batch. Removing it
    # must not change the loss — if it were folded in as a 0 term, it would.
    f, labels, is_pdf = _batch()
    full = _loss()(f.unsqueeze(1), labels=labels, is_proxy=is_pdf)
    keep = torch.arange(8)
    trimmed = _loss()(f[keep].unsqueeze(1), labels=labels[keep], is_proxy=is_pdf[keep])
    assert float(full) == pytest.approx(float(trimmed), abs=1e-6)


def test_a_batch_without_image_patches_is_zero_and_differentiable():
    f, labels, _ = _batch()
    f = f.clone().requires_grad_(True)
    loss = _loss()(f.unsqueeze(1), labels=labels, is_proxy=torch.ones(9, dtype=torch.long))
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.isfinite(f.grad).all()


def test_gradients_are_finite_and_reach_the_image_patches():
    f, labels, is_pdf = _batch()
    f = f.clone().requires_grad_(True)
    _loss()(f.unsqueeze(1), labels=labels, is_proxy=is_pdf).backward()
    assert torch.isfinite(f.grad).all()
    assert float(f.grad[is_pdf == 0].abs().sum()) > 0


def test_mismatched_flag_length_raises():
    f, labels, _ = _batch()
    with pytest.raises(ValueError, match="is_proxy"):
        _loss()(f.unsqueeze(1), labels=labels, is_proxy=torch.ones(3, dtype=torch.long))


# --------------------------------------------------------------- the wrapper + plumbing


def test_wrapper_matches_the_underlying_loss():
    f, labels, is_pdf = _batch()
    got = ProxyAnchoredSupCon(temperature=TAU, base_temperature=TAU)(
        {"features": f, "indices": labels, "is_pdf": is_pdf})
    assert float(got) == pytest.approx(_reference(f, labels, is_pdf), abs=1e-5)


def test_wrapper_refuses_a_batch_without_the_flag():
    f, labels, _ = _batch()
    with pytest.raises(ValueError, match="is_pdf"):
        ProxyAnchoredSupCon()({"features": f, "indices": labels, "is_pdf": None})


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 4)

    def forward(self, patches):
        return self.lin(patches.reshape(patches.size(0), -1))


def _run(data, validation=False):
    seen = {}

    def criterion(x):
        seen["is_pdf"] = x["is_pdf"]
        return x["features"].square().mean()

    process_batch_blobs(_Model(), data, {"c": (criterion, 1.0, True)}, None, "cpu",
                        OmegaConf.create({"training": {}}), validation=validation)
    return seen["is_pdf"]


def _patch_batch():
    return {
        "keypoints": torch.stack([torch.zeros(4, dtype=torch.long),
                                  torch.arange(4) // 2], dim=1),
        # Third patch is out of frame -> dropped from features, and from `is_pdf` too.
        "keypoint_coords": torch.tensor([[0.5, 0.5], [0.5, 0.5], [9.0, 9.0], [0.5, 0.5]]),
        "image_size": torch.tensor([1.0, 1.0]),
        "patches": torch.rand(4, 1, 4, 4),
        "is_pdf": torch.tensor([1, 0, 1, 0]),
    }


def test_process_batch_hands_the_flag_to_the_loss_filtered_like_the_features():
    got = _run(_patch_batch())
    assert torch.equal(got, torch.tensor([1, 0, 0]))


def test_the_flag_survives_validation():
    # Unlike the GT mask (withheld, because test time has none), the flag is a label:
    # withholding it would silently make the validation loss a different loss.
    got = _run(_patch_batch(), validation=True)
    assert torch.equal(got, torch.tensor([1, 0, 0]))


# ------------------------------------------------------------------ ProxyAnchoredFPR95


def _fpr_batch():
    """1-D embeddings, so a distance is just |a - b| and the ranking is readable.

    Laid out so the two pair populations disagree: every observation sits within 1.0 of
    its own proxy and at least 2.05 from the other, so the proxy<->image pairs are
    perfectly separated — while the two observations of a track sit on opposite sides of
    it, putting an observation of the *other* track (1.15 away) closer than its own
    partner (1.9). Image<->image confusion, none of it visible to the deployment task.
    """
    pos = [0.0, 3.0, -1.0, 0.9, 2.05, 3.9]
    f = torch.tensor(pos, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor([1, 2, 1, 1, 2, 2])
    is_pdf = torch.tensor([1, 1, 0, 0, 0, 0])
    return f, labels, is_pdf


def _cross_pair_reference(f, labels, is_pdf, target_recall=0.95):
    d, y = [], []
    for i in range(f.size(0)):
        for j in range(f.size(0)):
            if i != j and bool(is_pdf[i]) != bool(is_pdf[j]):
                d.append(float((f[i] - f[j]).abs()))
                y.append(int(labels[i] == labels[j]))
    return fpr_from_distances(torch.tensor(d), torch.tensor(y), target_recall)


def test_restricted_fpr_matches_the_cross_pair_reference():
    f, labels, is_pdf = _fpr_batch()
    got = fpr()(f, labels, is_proxy=is_pdf)
    assert float(got) == pytest.approx(float(_cross_pair_reference(f, labels, is_pdf)))


def test_image_image_confusion_does_not_reach_the_restricted_metric():
    # The whole point: the batch is perfect at the deployment task and messy at a task
    # nobody asks about. Plain FPR95 reports the mess, the restricted one does not.
    f, labels, is_pdf = _fpr_batch()
    assert float(fpr()(f, labels, is_proxy=is_pdf)) == 0.0
    assert float(fpr()(f, labels)) > 0.0


def test_restricted_fpr_is_nan_without_a_cross_pair():
    # An all-proxy batch has no proxy<->image pair at all, hence no positive: NaN, which
    # the validation loop skips (as it does for a positive-free batch of plain FPR95).
    f, labels, _ = _fpr_batch()
    got = fpr()(f, labels, is_proxy=torch.ones(6, dtype=torch.long))
    assert torch.isnan(got).all()


def test_restricted_fpr_rejects_a_mismatched_flag():
    f, labels, _ = _fpr_batch()
    with pytest.raises(ValueError, match="is_proxy"):
        fpr()(f, labels, is_proxy=torch.ones(3, dtype=torch.long))


def test_fpr_wrapper_reads_the_batch_flag():
    f, labels, is_pdf = _fpr_batch()
    got = ProxyAnchoredFPR95()({"features": f, "indices": labels, "is_pdf": is_pdf})
    assert float(got) == pytest.approx(float(fpr()(f, labels, is_proxy=is_pdf)))


def test_fpr_wrapper_refuses_a_batch_without_the_flag():
    f, labels, _ = _fpr_batch()
    with pytest.raises(ValueError, match="is_pdf"):
        ProxyAnchoredFPR95()({"features": f, "indices": labels, "is_pdf": None})
