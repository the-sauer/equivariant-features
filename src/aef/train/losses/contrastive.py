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
