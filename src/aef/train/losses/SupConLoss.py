# BSD 2-Clause License
#
# Copyright (c) 2020, Yonglong Tian
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, is_proxy=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            is_proxy: optional 0/1 (or bool) tensor of shape [bsz] marking the
                samples that act as **proxies**. When given, the loss becomes
                *proxy-anchored*: every scored pair runs between a proxy and a
                non-proxy sample, i.e.

                  * the outer sum runs over the proxy rows only,
                  * A(i) and P(i) hold non-proxy samples only,

                so no proxy-proxy and no data-data term ever enters the loss —
                neither in the numerator nor in the log-sum-exp denominator. The
                flag is per *sample* and is shared by all of that sample's views. A
                proxy with no positive in the batch is dropped from the outer mean
                (rather than contributing a 0 term, as the upstream edge-case
                handling below does), so the loss stays the mean over the rows that
                actually carry signal.

                This is the structure of Proxy-Anchor Loss (Kim et al., CVPR 2020) —
                proxies as anchors, associated with the whole batch — except that
                here a proxy is an *embedded sample* (in this repo the board's own
                rendering, flagged ``is_pdf``) rather than a learned parameter.

                Note the two senses of "anchor": ``anchor_feature`` /
                ``anchor_count`` below are SupCon's own term for the *rows* of the
                logit matrix, which is why this flag is not called ``is_anchor``.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        if is_proxy is not None:
            is_proxy = is_proxy.reshape(-1).to(device).bool()
            if is_proxy.shape[0] != batch_size:
                raise ValueError('Num of `is_proxy` flags does not match num of features')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # Proxy-anchored: drop every proxy column, so A(i) (and hence P(i) <= A(i))
        # holds non-proxy samples exclusively. `contrast_feature` is view-major (cat
        # over unbound views), so the per-sample flag tiles the same way `mask` does
        # above. The rows are restricted at the reduction step below.
        if is_proxy is not None:
            data_cols = (~is_proxy).repeat(contrast_count).to(logits_mask.dtype)
            logits_mask = logits_mask * data_cols.view(1, -1)

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # clamp: a row whose columns are all masked out (a proxy in a batch with no
        # non-proxy sample) would otherwise give log(0) = -inf and a NaN gradient,
        # even though the row is dropped from the mean below.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        has_pos = mask_pos_pairs >= 1e-6
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if is_proxy is None:
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            # outer sum over the proxy rows only, and only those that actually have a
            # positive in the batch.
            keep = is_proxy.repeat(anchor_count) & has_pos
            loss = loss[keep].mean() if bool(keep.any()) else (features.sum() * 0.0)

        return loss