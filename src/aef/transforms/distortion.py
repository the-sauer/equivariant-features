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

"""Radial (fisheye) lens distortion, post-composed onto a view's homography.

A view of the board is no longer ``H`` alone but ``D ∘ H``: the plane-to-image
homography followed by the lens. ``D`` is not projective, so nothing downstream can
keep treating the view map as a 3x3 matrix — see :class:`RadialDistortion` for what
the rest of the pipeline needs from it (forward map, undistortion, Jacobian) and
:func:`render_view` for the image warp that replaces ``warp_perspective``.
"""

import math

import numpy as np
import torch


_EPS = 1e-12


def _as_lambdas(lambdas):
    # Dtype is preserved rather than forced to float32: a float64 lens is what makes a
    # finite-difference check of the Jacobians meaningful.
    t = torch.as_tensor(lambdas)
    if not t.is_floating_point():
        t = t.to(torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2 or t.size(-1) != 2:
        raise ValueError(f"lambdas must be (2,) or (N, 2), got {tuple(t.shape)}")
    return t


class RadialDistortion:
    r"""Two-parameter radial lens distortion about a principal point, in pixels.

    The **undistortion** — observed view pixel ``q`` to the ideal pinhole point it
    would have had — is the closed form

        ``U(q) = c + (q - c) · g(ρ)``,  ``g(ρ) = 1 + λ₁ρ + λ₂ρ²``,  ``ρ = |q - c| / R``

    with ``c`` the principal point and ``R`` half the frame diagonal, so ``ρ = 1`` at
    the corners and the ``λ`` are dimensionless (a lens with ``λ₁ = 0.1`` moves the
    corner by ~10% of the half-diagonal). ``λ ≥ 0`` means undistortion *expands* the
    periphery, i.e. the lens compressed it: the rendered view is **barrel** distorted
    (fisheye). ``λ ≤ 0`` gives pincushion.

    Why the closed form sits in the undistortion direction and not the forward one:
    rendering a view fills each output pixel ``q`` from the source point
    ``H⁻¹(U(q))``, so a full frame is one closed-form evaluation per pixel — several
    million of them — whereas the forward map ``D = U⁻¹`` is only ever needed for
    keypoints (a few hundred per board) and is solved for by Newton on the radius.

    The Jacobian of the undistortion is closed-form too, and that is what the shape
    normalization consumes: the view Jacobian at a source point ``p`` is
    ``J_D(u)·J_H(p)`` with ``u = H(p)``, and ``J_D(u) = J_U(q)⁻¹`` at ``q = D(u)``.

    ``λ`` may be per-view: ``lambdas`` is ``(N, 2)`` and every method broadcasts it
    against a leading batch dimension of the points.
    """

    def __init__(self, lambdas, size=None, center=None, radius=None,
                 validate=True, max_rho=1.5):
        self.lambdas = _as_lambdas(lambdas)
        device = self.lambdas.device
        if size is None and (center is None or radius is None):
            raise ValueError("give `size` (H, W), or both `center` and `radius`")
        if size is not None:
            frame_h, frame_w = float(size[0]), float(size[1])
        if center is None:
            center = ((frame_w - 1.0) / 2.0, (frame_h - 1.0) / 2.0)
        if radius is None:
            radius = 0.5 * math.hypot(frame_w - 1.0, frame_h - 1.0)
        self.center = torch.as_tensor(
            center, dtype=self.lambdas.dtype, device=device
        )
        self.radius = float(radius)
        # Points can leave the frame (a warped keypoint may land outside it and is
        # only dropped further downstream), so invertibility is checked out to
        # `max_rho` times the corner radius, not just to the corner.
        self.max_rho = float(max_rho)
        if validate:
            self._check_invertible()

    def _check_invertible(self):
        """The radial map ``ψ(ρ) = ρ·g(ρ)`` must be strictly increasing.

        Otherwise the lens folds the image onto itself: ``U`` stops being injective,
        ``distort``'s Newton iteration has no unique root to find, and ``J_U`` is
        singular somewhere in the frame. Sampled on a grid rather than solved
        analytically — ``ψ'`` is only a quadratic, but the closed-form case analysis
        buys nothing over 65 evaluations.
        """
        rho = torch.linspace(0.0, self.max_rho, 65, device=self.lambdas.device)
        derivative = (
            1.0
            + 2.0 * self.lambdas[:, :1] * rho
            + 3.0 * self.lambdas[:, 1:] * rho * rho
        )
        bad = derivative.min(dim=-1).values <= 0.0
        if bool(bad.any()):
            raise ValueError(
                "non-invertible radial distortion: the radial map folds within "
                f"rho <= {self.max_rho} for lambdas "
                f"{self.lambdas[bad][:4].tolist()} (of {int(bad.sum())} bad views)"
            )

    @property
    def is_identity(self):
        return not bool(self.lambdas.any())

    def to(self, device):
        if self.lambdas.device == torch.device(device):
            return self
        return RadialDistortion(
            self.lambdas.to(device), center=self.center.to(device),
            radius=self.radius, validate=False, max_rho=self.max_rho,
        )

    def _coeffs(self, points):
        """``λ₁, λ₂`` shaped to broadcast against ``points[..., 0]``.

        ``points`` is ``(N, ..., 2)`` with ``N`` the lambda batch (or 1, broadcast).
        """
        shape = (self.lambdas.size(0),) + (1,) * (points.dim() - 2)
        return self.lambdas[:, 0].view(shape), self.lambdas[:, 1].view(shape)

    def _radial(self, points):
        offset = points - self.center
        rho = offset.norm(dim=-1) / self.radius
        lambda_1, lambda_2 = self._coeffs(points)
        return offset, rho, lambda_1, lambda_2

    def undistort(self, points):
        """Observed view pixels -> ideal pinhole pixels. ``(N, ..., 2)`` in and out."""
        offset, rho, lambda_1, lambda_2 = self._radial(points)
        # Written as a *displacement* from the input rather than `c + d·g(ρ)`: the two
        # agree mathematically, but this form leaves a zero lens bit-for-bit exact (the
        # displacement is identically 0) instead of off by the rounding of `c + d`. Batches
        # mix lens-free identity views with distorted ones, so those rows must not drift.
        displacement = lambda_1 * rho + lambda_2 * rho * rho
        return points + offset * displacement.unsqueeze(-1)

    def jacobian_undistort(self, points):
        """``J_U`` at each observed point, ``(N, ..., 2, 2)``.

        ``U(q) = c + d·g(ρ)`` with ``d = q - c`` and ``ρ = |d|/R`` gives
        ``J_U = g(ρ)·I + ρ·g'(ρ)·d̂d̂ᵀ``: the tangential stretch is ``g`` and the
        radial one ``g + ρg' = ψ'(ρ)``. At ``d = 0`` the ``d̂d̂ᵀ`` term is killed by
        ``ρ → 0``, so the ill-defined direction never reaches the result.
        """
        offset, rho, lambda_1, lambda_2 = self._radial(points)
        g = 1.0 + lambda_1 * rho + lambda_2 * rho * rho
        g_prime = lambda_1 + 2.0 * lambda_2 * rho
        unit = offset / offset.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        outer = unit.unsqueeze(-1) * unit.unsqueeze(-2)
        eye = torch.eye(2, dtype=points.dtype, device=points.device)
        return g[..., None, None] * eye + (rho * g_prime)[..., None, None] * outer

    def det_jacobian_undistort(self, points):
        """``det J_U`` — the product of the tangential and radial stretches."""
        _, rho, lambda_1, lambda_2 = self._radial(points)
        g = 1.0 + lambda_1 * rho + lambda_2 * rho * rho
        g_prime = lambda_1 + 2.0 * lambda_2 * rho
        return g * (g + rho * g_prime)

    def distort(self, points, iterations=25):
        """Ideal pinhole pixels -> observed view pixels: the inverse of ``undistort``.

        Only the radius has to be solved for (the direction is unchanged): find ``ρ``
        with ``ψ(ρ) = ρ·g(ρ) = σ``, the normalized target radius. ``ψ`` is strictly
        increasing (enforced in ``__init__``), so Newton from ``ρ₀ = σ`` converges
        quadratically; with ``λ = 0`` it is exact after the first step.
        """
        offset = points - self.center
        target = offset.norm(dim=-1)
        sigma = target / self.radius
        lambda_1, lambda_2 = self._coeffs(points)

        rho = sigma.clone()
        for _ in range(iterations):
            psi = rho * (1.0 + lambda_1 * rho + lambda_2 * rho * rho)
            psi_prime = 1.0 + 2.0 * lambda_1 * rho + 3.0 * lambda_2 * rho * rho
            rho = (rho - (psi - sigma) / psi_prime).clamp_min(0.0)

        # As in `undistort`, applied as a displacement so a zero lens is exact — which
        # is also why the ratio is taken between the *normalized* radii: with lambda 0
        # Newton returns rho == sigma untouched, so `rho / sigma - 1` is identically 0,
        # while the equivalent `rho * R / |d| - 1` would round.
        ratio = torch.where(
            sigma > _EPS, rho / sigma.clamp_min(_EPS) - 1.0, torch.zeros_like(sigma)
        )
        return points + offset * ratio.unsqueeze(-1)


def sample_radial_distortion(spec=None, rng=None):
    """Draw one ``(λ₁, λ₂)`` from a config spec.

    ``spec`` is ``{"lambda1": v, "lambda2": v}`` where each ``v`` is a fixed number
    or a ``[lo, hi]`` range drawn uniformly; a missing or ``None`` entry is 0. An
    empty/absent spec yields ``(0, 0)`` — no lens.
    """
    if not spec:
        return torch.zeros(2, dtype=torch.float32)
    rng = np.random if rng is None else rng

    def _draw(value):
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        low, high = float(value[0]), float(value[1])
        return float(rng.uniform(low, high)) if high > low else low

    return torch.tensor(
        [_draw(spec.get("lambda1")), _draw(spec.get("lambda2"))], dtype=torch.float32
    )


def render_view(imgs, homographies, distortion, size, fill=1.0):
    """Render ``out(q) = imgs(H⁻¹(U(q)))`` — the distorted counterpart of
    ``warp_perspective``.

    Once a lens is in the chain the view-pixel-to-source map is no longer projective,
    so kornia cannot do this: the grid is built explicitly, undistorted, and only then
    pushed through ``H⁻¹``. ``imgs`` is ``(N, C, H, W)``, ``homographies`` ``(N, 3, 3)``
    mapping source pixels to view pixels, ``size`` the ``(H, W)`` of the output frame.
    Everything the grid misses is filled with ``fill`` (white, matching the extractors'
    out-of-frame convention).

    Pixel convention matches ``warp_perspective(..., align_corners=True)``: output
    pixel centres are the integers ``0..W-1``, normalized as ``2x/(W-1) - 1``.
    """
    device = imgs.device
    n = imgs.size(0)
    out_h, out_w = int(size[0]), int(size[1])
    src_h, src_w = imgs.shape[-2:]

    ys, xs = torch.meshgrid(
        torch.arange(out_h, device=device, dtype=torch.float32),
        torch.arange(out_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    q = torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
    ideal = q if distortion is None or distortion.is_identity else distortion.undistort(q)

    ideal = torch.cat([ideal, torch.ones_like(ideal[..., :1])], dim=-1)
    source = torch.einsum("nij,nhwj->nhwi", torch.linalg.inv(homographies), ideal)
    source = source[..., :2] / source[..., 2:3]

    grid = torch.stack(
        [
            source[..., 0] / (src_w - 1) * 2.0 - 1.0,
            source[..., 1] / (src_h - 1) * 2.0 - 1.0,
        ],
        dim=-1,
    )
    out = torch.nn.functional.grid_sample(
        imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    oob = (grid.abs() > 1.0).any(dim=-1, keepdim=True).permute(0, 3, 1, 2)
    return torch.where(oob, torch.full_like(out, fill), out)
