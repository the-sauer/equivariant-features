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

import torch


def fpr(preds, labels, target_recall=0.95):
    # Sort by descending confidence
    sorted_indices = torch.argsort(preds, dim=0)
    sorted_labels = labels[sorted_indices]

    # Calculate cumulative sums of positives and negatives
    # cum_positives = torch.cumsum(sorted_labels, dim=0)
    cum_negatives = torch.cumsum(1 - sorted_labels, dim=0)

    # Calculate false positive rates at each threshold
    fpr = cum_negatives / (cum_negatives[-1] + 1e-8)  # Avoid division by zero

    # Find the index where FPR is closest to the target recall
    idx = torch.argmax((torch.cumsum(sorted_labels, dim=0) >= target_recall * torch.sum(sorted_labels)).float(), dim=0)

    return fpr[idx].item() if idx < len(fpr) else 1.0
