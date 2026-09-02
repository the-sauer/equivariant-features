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


def fpr(**_):
    def fpr_from_features(features, labels, target_recall=0.95, is_proxy=None):
        """FPR at ``target_recall`` over the batch's patch pairs.

        ``is_proxy`` (0/1 per patch) restricts the pair set to the pairs with exactly
        one proxy endpoint — the metric counterpart of ``ProxyAnchoredSupCon``, scoring
        only proxy<->data matches. Unlike the loss the pair set is unordered (the
        distance matrix is symmetric and there is no per-row normaliser), so the
        restriction is a cross-set mask rather than a rows/columns split; each pair
        still appears in both orders, as in the unrestricted metric.
        """
        lx, ly = torch.meshgrid(labels, labels, indexing="ij")
        # Exclude the diagonal: pairing a patch with itself is a trivial positive at
        # distance 0. Those always rank first, which inflates recall and biases the
        # FPR low — and in a batch whose labels are all singletons (e.g. only
        # background garbage) they are the *only* positives, which made such batches
        # report a meaningless FPR of 0.
        pairs = ~torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
        if is_proxy is not None:
            p = is_proxy.reshape(-1).bool().to(pairs.device)
            if p.numel() != labels.numel():
                raise ValueError("Num of `is_proxy` flags does not match num of labels")
            pairs = pairs & (p.unsqueeze(1) != p.unsqueeze(0))
        return fpr_from_distances(
            torch.cdist(features, features, p=2)[pairs],
            (lx == ly)[pairs].int(),
            target_recall=target_recall
        )
    return fpr_from_features


def fpr_from_distances(preds, labels, target_recall=0.95):
    if labels.sum() == 0:
        # No positive pairs -> FPR@recall is undefined. Report NaN rather than a
        # misleading 0; the validation loop skips non-finite batches.
        return torch.full((1,), float("nan"), device=preds.device)
    # Sort by descending confidence
    sorted_indices = torch.argsort(preds, dim=0)
    sorted_labels = labels[sorted_indices]

    # Calculate cumulative sums of positives and negatives
    # cum_positives = torch.cumsum(sorted_labels, dim=0)
    cum_negatives = torch.cumsum(1 - sorted_labels, dim=0)

    # Calculate false positive rates at each threshold
    fpr_val = cum_negatives / (cum_negatives[-1] + 1e-8)  # Avoid division by zero

    # Find the index where FPR is closest to the target recall
    idx = torch.argmax((torch.cumsum(sorted_labels, dim=0) >= target_recall * torch.sum(sorted_labels)).float(), dim=0)
    return fpr_val[idx:idx+1] if idx < len(fpr_val) else torch.tensor(1.0)
