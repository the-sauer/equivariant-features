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

import abc

import escnn
import torch
import torch.nn.functional as F

from .utils import L2Norm, input_norm


class AbstractBlobDescriptor(torch.nn.Module, abc.ABC):
    net: torch.nn.Module
    field_type: escnn.nn.FieldType

    def __init__(self, n_rotations=8, **_):
        super().__init__()
        self.s = escnn.gspaces.rot2dOnR2(n_rotations)
        self.field_type = escnn.nn.FieldType(self.s, 1 * [self.s.trivial_repr])


# --- Learned board-validity masking (shared by the steerable descriptors) ---------
#
# Same contract as ``HardNetLogPolar(learned_mask=True)`` (see models/hardnet.py), so
# both families are configured and supervised identically:
#
#   * on the ANCHOR view the board-coverage mask is GT (the identity view's off-board
#     region is known), so it is used directly;
#   * on the TARGET (warped) views it is unavailable at test time, so a small predictor
#     head emits ``m_pred`` in [0, 1] from the backbone features and that is used instead;
#   * ``forward`` returns ``(descriptor, m_pred)`` so ``process_batch_blobs`` can add the
#     BCE that trains the predictor against the targets' true coverage.
#
# Everything stays exactly rotation-equivariant: the mask (GT or predicted) is a SCALAR
# field — the predictor maps the regular-representation features to a single trivial
# representation — and a pointwise product with a scalar field commutes with the group
# action, exactly like the existing radial ``inner_mask``/``MaskModule`` weighting.


def _mask_predictor(field_type, gspace):
    """1x1 steerable conv reading a per-cell validity scalar off the feature field."""
    return escnn.nn.R2Conv(
        field_type, escnn.nn.FieldType(gspace, 1 * [gspace.trivial_repr]), 1
    )


def _fill_invalid_with_mean(x, mask):
    """Replace invalid pixels by the patch's valid mean (a neutral, edge-free fill).

    The extractors fill off-board/out-of-frame area with white, which is a strong
    (and fake) structure. Zeroing it instead would just trade it for a fake black
    edge; substituting the mean of the valid pixels removes the contrast entirely.
    This is the un-normalized analogue of ``hardnet.input_norm(x, mask)``, which
    sets invalid pixels to the normalized mean (i.e. 0).
    """
    m = mask.expand_as(x)
    n = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    mean = (x * m).sum(dim=(1, 2, 3), keepdim=True) / n
    return x * m + mean * (1.0 - m)


def _check_oracle(learned_mask, oracle_mask):
    if oracle_mask and not learned_mask:
        raise ValueError("oracle_mask=True needs learned_mask=True — it replaces the "
                         "predictor's output by the GT, it does not add a mask path")


def _gate_input(x, mask, is_pdf, oracle=False):
    """Neutralize the known off-board region — on ANCHORS only (elsewhere unknown).

    ``oracle`` extends it to every view: the ceiling ablation hands the model the true
    mask on the targets too (see ``HardNetLogPolar``'s ``oracle_mask``).
    """
    if oracle:
        return _fill_invalid_with_mean(x, mask)
    a = is_pdf.view(-1, 1, 1, 1).to(x.dtype)
    return _fill_invalid_with_mean(x, a * mask + (1.0 - a) * torch.ones_like(mask))


def _validity_weight(size, mask, m_pred, is_pdf, oracle=False):
    """Per-cell validity at feature resolution: GT on PDF patches, ``m_pred`` elsewhere.

    ``oracle`` uses the GT on every view, so the prediction is bypassed entirely.
    """
    if mask is None:
        return m_pred                       # no GT routing available (e.g. validation)
    gt = F.adaptive_avg_pool2d(mask, size)   # (B, 1, H, W) board coverage
    if oracle:
        return gt
    if is_pdf is None:
        return m_pred
    a = is_pdf.view(-1, 1, 1, 1).to(m_pred.dtype)
    return a * gt + (1.0 - a) * m_pred


class BlobDescriptorEfficient(AbstractBlobDescriptor):
    """Memory/compute-efficient steerable blob descriptor.

    Keeps ``NoStride``'s equivariant inductive bias but fixes its two cost drivers:

    1. **Antialiased downsampling** — instead of running every layer at 24-62 px, a
       ``PointwiseAvgPoolAntialiased2D`` (Gaussian blur + subsample, equivariance-
       preserving) halves the resolution after each stage (64 -> 32 -> 16 -> 8), so
       the wide deep layers run cheaply.
    2. **Small dense head** — replaces the ``R2Conv(kernel=24)`` readout (whose escnn
       basis-expanded filter alone is ~2.25 GB) with an adaptive pool to a small grid
       followed by a small dense ``R2Conv`` (spatial layout preserved, ~34x cheaper),
       or an attention-weighted pool (``head="attention"``).

    **Centre masking** (``inner_mask``): the anchor blob within
    ``sigma_cutoff * (patch_size/2) / scale_factor`` px of the centre is the same
    normalized structure for every keypoint, so it carries no discriminative signal.
    A radially symmetric mask commutes with the rotation group, so zeroing it keeps
    equivariance intact and forces the descriptor onto the informative surround.

    **Board-validity masking** (``learned_mask``, default off): the *data-dependent*
    counterpart of the fixed centre mask — it removes the off-board part of the patch,
    which is real background rather than a fixed geometric region. GT mask on PDF patches,
    predicted ``m_pred`` on targets, ``(descriptor, m_pred)`` returned; see the
    module-level notes. Both heads consume the weight where it matters: ``attention``
    multiplies it into the saliency it pools with, ``dense`` weights the field before
    the readout conv.
    """

    def __init__(
        self,
        out_dim=128,
        widths=(32, 64, 128),
        head="attention",          # "attention" (equivariance-robust) | "dense" (layout, but sensitive)
        l2_normalize=True,          # NoStride returns normalized features; match by default
        pool_sigma=0.6,            # Gaussian sigma of the antialiased downsampling
        frequencies_cutoff=None,   # bandlimit the steerable conv basis (None = escnn default)
        dropout=0.1,
        inner_mask=True,           # zero the uninformative anchor blob at the patch centre
        sigma_cutoff=2.0,
        scale_factors=(64.0,),     # the patch scale factor(s); sets the centre-mask radius
        patch_size=64,
        learned_mask=False,        # board-validity aware pooling + mask predictor
        oracle_mask=False,         # ceiling ablation: the TRUE mask on every view
        **kwargs,
    ):
        super().__init__(**kwargs)
        _check_oracle(learned_mask, oracle_mask)
        self.oracle_mask = oracle_mask
        s = self.s

        # --- Centre (donut) mask ------------------------------------------------
        if inner_mask:
            sf = float(min(scale_factors))
            r_pixel = sigma_cutoff * (patch_size / 2.0) / sf
            yy, xx = torch.meshgrid(
                torch.arange(patch_size), torch.arange(patch_size), indexing="ij"
            )
            c = (patch_size - 1) / 2.0
            dist = torch.sqrt((xx - c) ** 2 + (yy - c) ** 2)
            mask = (dist >= r_pixel).to(torch.float32).view(1, 1, patch_size, patch_size)
            self.register_buffer("inner_mask", mask)
        else:
            self.inner_mask = None

        c_in = escnn.nn.FieldType(s, 1 * [s.trivial_repr])
        w1, w2, w3 = widths
        c1 = escnn.nn.FieldType(s, w1 * [s.regular_repr])
        c2 = escnn.nn.FieldType(s, w2 * [s.regular_repr])
        c3 = escnn.nn.FieldType(s, out_dim * [s.regular_repr])

        # Mask the boundary ring after every conv: padded convs leak zero-padding
        # artifacts at the border that break rotation equiightly equivariant.
        def block(cin, cout, k, size):
            return [
                escnn.nn.R2Conv(
                    cin, cout, k, padding=k // 2, bias=False,
                    frequencies_cutoff=frequencies_cutoff,
                ),
                escnn.nn.MaskModule(cout, size, margin=1),
                escnn.nn.InnerBatchNorm(cout),
                escnn.nn.ReLU(cout),
                escnn.nn.FieldDropout(cout, p=dropout),
            ]

        s1, s2, s3 = patch_size, patch_size // 2, patch_size // 4  # 64, 32, 16
        self.net = escnn.nn.SequentialModule(
            escnn.nn.MaskModule(c_in, patch_size, margin=1),
            *block(c_in, c1, 5, s1),
            *block(c1, c1, 5, s1),
            escnn.nn.PointwiseAvgPoolAntialiased2D(c1, sigma=pool_sigma, stride=2),  # 64 -> 32
            *block(c1, c2, 5, s2),
            *block(c2, c2, 5, s2),
            escnn.nn.PointwiseAvgPoolAntialiased2D(c2, sigma=pool_sigma, stride=2),  # 32 -> 16
            *block(c2, c3, 5, s3),
            *block(c3, c3, 5, s3),
            escnn.nn.PointwiseAvgPoolAntialiased2D(c3, sigma=pool_sigma, stride=2),  # 16 -> 8
        )

        self.head_type = head
        if head == "dense":
            # A single dense R2Conv over the *whole* backbone grid reads it out to
            # 1x1: it keeps a distinct learned weight per spatial cell (full layout)
            # AND stays exactly equivariant. (Adaptive-pooling to a non-divisor grid
            # is NOT rotation-symmetric and wrecks invariance; a valid conv over the
            # full grid is the equivariant analogue of flatten+FC.) The three stride-2
            # antialiased pools take patch_size -> patch_size/8, which is the grid the
            # readout consumes.
            grid = patch_size // 8
            self.readout = escnn.nn.R2Conv(
                c3, c3, grid, bias=False, frequencies_cutoff=frequencies_cutoff
            )
            self.group_pool = escnn.nn.GroupPooling(c3)
        elif head == "attention":
            # Learned non-uniform spatial weighting instead of a uniform average.
            self.group_pool = escnn.nn.GroupPooling(c3)
            self.att = escnn.nn.R2Conv(
                c3, escnn.nn.FieldType(s, 1 * [s.trivial_repr]), 1
            )
        else:
            raise ValueError(f"head must be 'dense' or 'attention', got {head!r}")
        self.l2_normalize = l2_normalize
        self.learned_mask = learned_mask
        if learned_mask:
            self.mask_head = _mask_predictor(c3, s)

    def forward(self, x, mask=None, is_pdf=None):
        x = input_norm(x)
        if self.learned_mask and mask is not None and (is_pdf is not None or self.oracle_mask):
            x = _gate_input(x, mask, is_pdf, self.oracle_mask)
        if self.inner_mask is not None:
            x = x * self.inner_mask
        x = escnn.nn.GeometricTensor(x, self.field_type)
        x = self.net(x)

        m_pred = w = None
        if self.learned_mask:
            m_pred = torch.sigmoid(self.mask_head(x).tensor)   # [B, 1, H, W] validity
            w = _validity_weight(m_pred.shape[-2:], mask, m_pred, is_pdf, self.oracle_mask)

        if self.head_type == "dense":
            if w is not None:
                x = escnn.nn.GeometricTensor(x.tensor * w, x.type)
            x = self.readout(x)                 # whole grid -> 1x1
            x = self.group_pool(x).tensor       # [B, out_dim, 1, 1]
            x = x.view(x.size(0), -1)
        else:  # attention
            a = torch.sigmoid(self.att(x).tensor)          # [B, 1, H, W] saliency
            if w is not None:
                # Fold validity into the pooling weights: an off-board cell can no
                # longer win the attention, and — because the SAME weight divides the
                # normalizer — the descriptor is the average over the valid region
                # rather than a downscaled average over everything.
                a = a * w
            f = self.group_pool(x).tensor                  # [B, out_dim, H, W] invariant
            x = (f * a).sum(dim=(2, 3)) / (a.sum(dim=(2, 3)) + 1e-6)

        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=1)
        return (x, m_pred) if self.learned_mask else x
