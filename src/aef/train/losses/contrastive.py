# Affine Equivariant Features, the main implementation of my master thesis.
# Copyright (C) 2026 Hendrik Sauer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pytorch_metric_learning.losses as pml_losses
import torch

from ...evaluate import fpr
from .SupConLoss import SupConLoss


class Contrastive(torch.nn.Module):
    """Metric-learning loss over already-extracted descriptor features.

    Callers pass ``{"features": (N, D), "indices": (N,)}``; entries sharing an
    index are positives (see ``process_batch_blobs`` / ``process_batch_canonicalize``).
    """

    def __init__(
        self,
        contrastive_loss: str = "NPairsLoss",
        contrastive_loss_kwargs: dict = None,
        **_
    ):
        super().__init__()
        if contrastive_loss == "fpr":
            self.contrastive_loss = fpr()
        else:
            try:
                self.contrastive_loss = getattr(pml_losses, contrastive_loss)(**(contrastive_loss_kwargs or {}))
            except AttributeError:
                raise ValueError(f"Unsupported distance metric: {contrastive_loss}")

    def forward(self, x) -> torch.Tensor:
        return self.contrastive_loss(x["features"], x["indices"])


class FPR95(Contrastive):
    def __init__(self, **kwargs):
        super().__init__(contrastive_loss="fpr", **kwargs)


def _proxy_flag(x, who):
    """The batch's per-patch proxy flag, or a pointed error.

    Falling back to the unrestricted objective would swap in a *different* quantity
    under the same name — silently, and only on datasets without the flag — so a
    missing flag is an error.
    """
    is_pdf = x.get("is_pdf")
    if is_pdf is None:
        raise ValueError(
            f"{who} needs a per-patch `is_pdf` flag in the batch — it marks the "
            "proxies, and the dataset in use provides none (BlobTrackData always does; "
            "HomographyData whenever its patches are precomputed). Use the unrestricted "
            "variant (`SupCon` / `FPR95`) for such a dataset."
        )
    return is_pdf


class ProxyAnchoredFPR95(FPR95):
    """FPR95 over proxy<->image pairs only — the metric counterpart of
    ``ProxyAnchoredSupCon``.

    Plain ``FPR95`` ranks every patch pair in the batch, so its number is dominated by
    image<->image pairs: how well two observations of one blob agree, against how well
    two observations of different blobs are told apart. At test time neither question is
    asked — a detection is matched against the board's rendering — so this variant keeps
    only the pairs with exactly one proxy endpoint. Same threshold logic, smaller pair
    set, and a number that answers the deployment question.

    Note it is **not** comparable with `FPR95` (different pair population), so a run
    switched over to it starts a fresh series of validation numbers.
    """

    def forward(self, x) -> torch.Tensor:
        return self.contrastive_loss(x["features"], x["indices"],
                                     is_proxy=_proxy_flag(x, "ProxyAnchoredFPR95"))


class SupCon(Contrastive):
    def __init__(self, **kwargs):
        super().__init__(contrastive_loss="SupConLoss", **kwargs)


class ProxyAnchoredSupCon(torch.nn.Module):
    """SupCon with the board's own rendering as the **proxy anchor** of its blob.

    Plain ``SupCon`` contrasts every patch against every other, so most of the loss is
    made of image<->image terms: two oblique observations of the same blob pulled
    together, two observations of different blobs pushed apart. That is not the
    deployment task — matching happens against the board's own rendering — so this
    variant keeps only the terms that involve it: the outer sum runs over the ``is_pdf``
    patches and both ``A(i)`` and ``P(i)`` hold image patches exclusively (see
    ``SupConLoss.forward``). Everything else about the loss is upstream SupCon.

    The structure is Proxy-Anchor Loss (Kim et al., CVPR 2020) — proxies as anchors,
    associated with the whole batch — with one difference: the proxy is not a learned
    per-class vector but the embedded rendering itself, so it moves with the encoder and
    is defined for a blob the model has never been trained on.

    Needs ``x["is_pdf"]`` in the batch (``process_batch_blobs`` supplies it whenever the
    dataset carries the flag); without it this degenerates to plain SupCon, which would
    quietly train a different objective, so its absence is an error rather than a
    fallback. Features are L2-normalized first — upstream SupCon assumes unit vectors,
    while ``pytorch_metric_learning``'s implementation (used by ``SupCon``) normalizes
    internally, so this keeps the two comparable for descriptors that do not.
    """

    def __init__(self, temperature=0.07, base_temperature=0.07,
                 contrast_mode="all", normalize=True, **_):
        super().__init__()
        self.loss = SupConLoss(temperature=temperature, contrast_mode=contrast_mode,
                               base_temperature=base_temperature)
        self.normalize = normalize

    def forward(self, x) -> torch.Tensor:
        is_pdf = _proxy_flag(x, "ProxyAnchoredSupCon")
        f = x["features"]
        if self.normalize:
            f = torch.nn.functional.normalize(f, p=2, dim=1)
        return self.loss(f.unsqueeze(1), labels=x["indices"], is_proxy=is_pdf)


class Recall1(torch.nn.Module):
    """Report-only diagnostic: top-1 nearest-neighbour retrieval accuracy — the
    per-batch **true-positive rate**.

    For every patch that has at least one same-``track_id`` partner in the batch,
    check whether its nearest neighbour (in descriptor space, excluding itself) is a
    true match. This is the quantity that most directly says "is the descriptor
    telling tracks apart?" — SupCon going down while this stays near chance means the
    embedding isn't actually separating identities. Returns NaN for a batch with no
    positive pair (so it's skipped, like FPR95).
    """

    def __init__(self, **_):
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        f, y = x["features"], x["indices"]
        n = f.size(0)
        if n < 2:
            return torch.full((1,), float("nan"), device=f.device)
        same = y.unsqueeze(0) == y.unsqueeze(1)
        has_pos = same.sum(1) > 1  # > 1 because the diagonal (self) always counts
        if not bool(has_pos.any()):
            return torch.full((1,), float("nan"), device=f.device)
        d = torch.cdist(f, f)
        d.fill_diagonal_(float("inf"))
        nn = d.argmin(dim=1)
        correct = same[torch.arange(n, device=f.device), nn]
        return correct[has_pos].float().mean().view(1)


class PosCoverage(torch.nn.Module):
    """Report-only data-health diagnostic: the fraction of patches in the batch that
    have >= 1 same-``track_id`` partner present (i.e. something to be pulled toward).

    With the balanced sampler this should be high; a low value means the batches are
    starved of positives and the contrastive signal is weak regardless of the model.
    """

    def __init__(self, **_):
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        y = x["indices"]
        n = y.numel()
        if n == 0:
            return torch.full((1,), float("nan"), device=y.device)
        same = y.unsqueeze(0) == y.unsqueeze(1)
        return (same.sum(1) > 1).float().mean().view(1)
