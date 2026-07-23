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


class SupCon(Contrastive):
    def __init__(self, **kwargs):
        super().__init__(contrastive_loss="SupConLoss", **kwargs)


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
