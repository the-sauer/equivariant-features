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

import functools
import hashlib
import inspect
import json
import math
import os
from typing import Iterable, Union

import kornia
import numpy as np
import omegaconf
import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

from ..geometry import homogenize, linearize_homography
from ..transforms.homography import sample_homography


# Bump when a pipeline change makes previously written caches invalid. The key covers
# constructor params, not this module's code, so an algorithm change needs a bump here.
#   3: blob_normalizations forces a proper (det>0) factor — patch content changed.
#   4: scale normalization fixed — blob_normalizations is now det-1 (size is left to
#      `scales`, which no longer double-corrects) and every Jacobian is evaluated at
#      the source point. Both change the patch footprint of every warped view.
#   5: optional per-patch log-polar validity mask (`precompute_masks`) added to the
#      cached state; bumped so pre-v5 caches (which lack the tensor) are rebuilt.
# `precompute_masks` also covers CARTESIAN patches now (for the steerable learned-mask
# heads). No bump: it was silently forced off for cartesian before, so every affected
# cache key is new — existing caches stay valid.
CACHE_VERSION = 5

# Params that don't change a dataset's *contents* (so they must not split the cache).
# Everything else a constructor accepts is hashed — including params left at their
# default (see `effective_params`) — so anything that realistically changes the
# boards, keypoints or extracted patches (patch_size, supersample, patch_type, the
# scale factors, compositing/garbage settings, augmentation, ...) is covered.
_CACHE_KEY_EXCLUDE = {
    "data_dir", "cache_dir", "cache_path", "extraction_batch_size", "sift_batch_size",
}


def effective_params(fns, passed: dict) -> dict:
    """``passed`` overlaid on the constructor defaults of ``fns`` (later fns win).

    Hashing the *effective* values rather than only what the caller happened to pass
    means a param left at its default still participates in the cache key. Two
    consequences: a config that omits e.g. ``supersample`` still keys on the value it
    actually ran with, and changing a default in code changes every key — invalidating
    stale caches instead of silently reusing them.
    """
    merged = {}
    for fn in fns:
        for name, p in inspect.signature(fn).parameters.items():
            if p.default is not inspect.Parameter.empty:
                merged[name] = p.default
    merged.update(passed)
    return merged


def dataset_cache_key(params: dict) -> str:
    """Stable short hash over the params that determine a dataset's contents.

    OmegaConf nodes are resolved to plain containers so a config-driven run and a
    hand-constructed one with the same values hit the same cache entry.
    """
    def _norm(v):
        if isinstance(v, (omegaconf.DictConfig, omegaconf.ListConfig)):
            return omegaconf.OmegaConf.to_container(v, resolve=True)
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        if isinstance(v, dict):
            return {k: _norm(x) for k, x in v.items()}
        return v

    payload = {"__cache_version__": CACHE_VERSION}
    payload.update(
        {k: _norm(v) for k, v in params.items() if k not in _CACHE_KEY_EXCLUDE}
    )
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def curry(f):
    def function(x):
        def inner(y):
            return f(x, y)

        return inner

    return function


def fchain(*args):
    def function(x):
        for f in reversed(args):
            x = f(x)
        return x

    return function


def find_images(
    dir, extensions=[".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
) -> Iterable[str]:
    return fchain(
        curry(filter)(lambda file: any(map(lambda e: file.endswith(e), extensions))),
        curry(functools.reduce)(list.__add__),
        curry(map)(lambda x: list(map(lambda f: os.path.join(x[0], f), x[2]))),
    )(os.walk(dir))


def resolve_background_dir(background, kaggle_slug):
    """Resolve a directory of background scene images.

    Prefers a local ``background`` folder (any non-empty directory of images);
    falls back to downloading ``kaggle_slug`` via kagglehub. Returns ``None`` if
    neither yields any image, so the caller can disable compositing gracefully.
    """
    if background is not None and os.path.isdir(background):
        if any(True for _ in find_images(background)):
            return background
    if kaggle_slug:
        try:
            import kagglehub

            path = kagglehub.dataset_download(kaggle_slug)
            if any(True for _ in find_images(path)):
                return path
        except Exception as e:  # noqa: BLE001 — background is optional; never hard-fail training
            print(f"Could not download kagglehub background {kaggle_slug!r}: {e}")
    return None


def load_background(path, size):
    """Decode one background image as a single grayscale channel in ``[0, 1]``,
    covering ``size`` (H, W) at its original aspect ratio.

    Resize-to-fit-the-short-side then centre crop, rather than resizing straight to
    ``size``: the latter takes a 2-element size, which torchvision honours exactly,
    stretching a landscape photo onto a square canvas. Backgrounds are what the
    garbage keypoints are detected on — they are the dataset's pure negatives — so
    distorting their image statistics makes those negatives unrepresentative.
    """
    img = torchvision.io.decode_image(path, torchvision.io.ImageReadMode.GRAY)
    h, w = img.shape[-2:]
    target_h, target_w = int(size[0]), int(size[1])
    # Scale so both axes cover the target, then trim the overhang.
    scale = max(target_h / h, target_w / w)
    img = torchvision.transforms.functional.resize(
        img, [max(target_h, int(round(h * scale))), max(target_w, int(round(w * scale)))]
    )
    img = torchvision.transforms.functional.center_crop(img, [target_h, target_w])
    return img.to(torch.float32) / 255.0


def sample_placement_similarity(size, scale_range, generator):
    """Sample a placement mapping a board raster (size (H, W)) into an axis-aligned
    sub-rectangle of a same-size frame: uniform scale + a random translation that
    keeps the scaled board fully inside the frame.

    No rotation is applied here — the per-view homography already rotates the board,
    so adding it at placement would just compound it (and the axis-aligned box keeps
    the identity view a clean, upright reference). Returns ``(A, s)`` where ``A`` is a
    ``(3, 3)`` matrix mapping board-raster pixels (x, y) to frame pixels and ``s`` is
    the uniform scale. Being a pure similarity (uniform scale, no shear), circular
    blobs stay circular — the identity-view shape normalization assumes isotropy.
    """
    H, W = float(size[0]), float(size[1])

    def _uniform(lo, hi):
        return lo + (hi - lo) * torch.rand((), generator=generator).item()

    s = _uniform(float(scale_range[0]), float(scale_range[1]))

    # Axis-aligned scaled board bbox, so translation keeps it in-frame.
    bbox_w, bbox_h = s * W, s * H
    cx = _uniform(bbox_w / 2, max(bbox_w / 2, W - bbox_w / 2))
    cy = _uniform(bbox_h / 2, max(bbox_h / 2, H - bbox_h / 2))

    # A = T(new_center) @ S(s) @ T(-board_center)
    A = torch.eye(3, dtype=torch.float32)
    A[0, 0] = s
    A[1, 1] = s
    A[0, 2] = cx - s * W / 2
    A[1, 2] = cy - s * H / 2
    return A, s


def composite_board(board, background, A, size, lighting=True, shading_strength=0.3):
    """Place a single-channel ``board`` (1, H, W) onto ``background`` (1, H, W) using
    the placement similarity ``A``, optionally light-matching the board to the scene.

    Returns ``(composite, mask)`` both (1, H, W). ``mask`` is the warped board
    coverage (soft edges from bilinear sampling). Lighting is *multiplicative* — a
    gentle brightness match to the local background plus a low-frequency shading
    field — so the board's internal blob/paper contrast is preserved (a full
    mean/std match would flatten faint blobs and hurt the descriptor).
    """
    H, W = int(size[0]), int(size[1])
    A_b = A.unsqueeze(0)
    board_on_frame = kornia.geometry.transform.warp_perspective(
        board.unsqueeze(0), A_b, dsize=(H, W), padding_mode="zeros"
    )[0]
    mask = kornia.geometry.transform.warp_perspective(
        torch.ones_like(board).unsqueeze(0), A_b, dsize=(H, W), padding_mode="zeros"
    )[0]

    board_lit = board_on_frame
    if lighting:
        inside = mask > 0.5
        if inside.any():
            bg_mean = background[inside].mean()
            board_mean = board_on_frame[inside].mean().clamp_min(1e-4)
            # Gentle brightness match, clamped so paper/blobs keep their contrast.
            brightness = (bg_mean / board_mean).clamp(0.5, 1.5)
            board_lit = board_on_frame * brightness
        if shading_strength > 0:
            # Low-frequency shading field from the background, normalized to ~1 mean
            # so it modulates (not recolours) the board. Computed on a small
            # downsample + blur + upsample so the cost is independent of the (large)
            # board raster resolution.
            small = torch.nn.functional.interpolate(
                background.unsqueeze(0), size=(64, 64), mode="area"
            )
            small = kornia.filters.gaussian_blur2d(small, (9, 9), (2.0, 2.0))
            shading = torch.nn.functional.interpolate(
                small, size=(H, W), mode="bilinear", align_corners=False
            )[0]
            shading = shading / shading.mean().clamp_min(1e-4)
            board_lit = board_lit * (1.0 - shading_strength + shading_strength * shading)
        board_lit = board_lit.clamp(0.0, 1.0)

    composite = mask * board_lit + (1.0 - mask) * background
    return composite, mask


def blob_normalizations(homographies, coords, device):
    """Affine *shape* normalization for each keypoint.

    Whitens the local Jacobian of ``homographies`` (dropping the rotational part
    via SVD) so that the elliptical blob maps to an isotropic one. Shared by the
    cartesian and log-polar patch extractors.

    ``coords`` are the keypoint's coordinates in the *warped* frame — the codomain
    of ``homographies``. ``linearize_homography`` differentiates at a point of its
    map's *domain*, so it must never be handed ``coords`` together with
    ``homographies``. What is wanted, ``inv(J_H(p))`` at the source point ``p``, is
    by the chain rule the Jacobian of the inverse homography at the warped point —
    hence linearizing ``inv(homographies)`` here. These coincide only for a purely
    affine warp.

    The factor is normalized to ``det == 1``: shape and orientation only, never
    size. ``scales`` already carries the warp's ``sqrt(det J)`` and the extractors
    divide by it, so a factor that also removed the size would remove it twice and
    leave the same blob at a view-dependent size in its patch.
    """
    normalizations = linearize_homography(
        torch.linalg.inv(homographies),
        coords=torch.cat(
            [coords, torch.ones((1, 1)).expand(coords.size(0), 1).to(device)],
            dim=-1,
        ),
    )
    _, S, Vh = torch.linalg.svd(normalizations)
    # Force a proper (det>0) factor. torch's SVD may return a reflection pair
    # (det(U)=det(Vh)=-1) — the factorization is only unique up to that sign, and for an
    # isotropic warp the singular values are degenerate so it is fully ambiguous. The
    # warp itself has det>0, so whatever is left after dropping U must be a *rotation*;
    # if Vh carries a reflection the patch comes out MIRRORED relative to the identity
    # view, which no rotation (nor log-polar angular pooling) can undo. Negating a row
    # of Vh flips both dets, leaving U@Σ@Vh unchanged.
    #
    # The residual *rotation* this leaves is intentional: the detector yields blobs at
    # arbitrary orientation downstream, so the training pairs should exercise exactly
    # that. Do not "fix" it with a polar decomposition (P = Vh^T Σ Vh) — that is
    # rotation-free by construction and would train on a distribution the deployed
    # detector never produces. Only the reflection was wrong.
    Vh = Vh.clone()
    flip = torch.linalg.det(Vh) < 0
    Vh[flip, -1, :] = -Vh[flip, -1, :]
    Σ = torch.zeros((S.size(0), 2, 2), dtype=torch.float32, device=device)
    Σ[:, 0, 0] = S[..., 0]
    Σ[:, 1, 1] = S[..., 1]
    normalization = Σ @ Vh
    # Shape only — hand the size back to `scales` (see the docstring).
    return normalization / torch.linalg.det(normalization).abs().sqrt().view(-1, 1, 1)


def extract_multiscale_patches(
    imgs, homographies, coords, scales, patch_size=64, scale_factors=[16.0, 64.0, 128.0],
    supersample=1, return_mask=False, mask_imgs=None,
):
    """Extract cartesian patches, one channel per entry in ``scale_factors``.

    ``return_mask`` additionally samples a per-patch board-validity mask (1 = on the
    board) on the SAME warp, for the learned-mask descriptor heads — the cartesian
    counterpart of :func:`extract_logpolar_patches`'s. ``mask_imgs`` is a board-coverage
    image in the same frame as ``imgs``; without one, validity is "inside the frame"
    (correct for the identity/anchor view, whose board raster fills the frame).

    The mask is single-channel and therefore only defined for a SINGLE scale factor:
    with several, each channel samples a different physical extent onto the same pixel
    grid, so one mask cannot describe them all.
    """
    device = imgs.device
    blob_normalizations_ = blob_normalizations(homographies, coords, device)

    # Warp at `supersample` taps per output pixel along each axis and
    # area-average back down (in addition to the per-scale pre-blur below), so
    # each output pixel integrates its full source footprint. ``supersample=1``
    # recovers the plain single-tap behaviour.
    ss = max(1, int(supersample))
    P = patch_size * ss

    if return_mask and len(scale_factors) != 1:
        raise ValueError(
            "cartesian validity masks are single-channel and need exactly one "
            f"scale factor, got {list(scale_factors)}"
        )

    multiscale_patches = []
    valid = None

    # Wir nehmen an, dass der kleinste Scale-Faktor die "schärfste" Basis ist
    base_scale = min(scale_factors)

    for sf in scale_factors:
        # --- Anti-Aliasing (Pre-Blur) für große Skalen ---
        downsample_ratio = sf / base_scale
        if downsample_ratio >= 2.0:
            sigma = float(downsample_ratio / 2.0)
            k_size = int(2 * math.ceil(2 * sigma) + 1)  # Garantiert ungerade
            blurred_imgs = kornia.filters.gaussian_blur2d(
                imgs, kernel_size=(k_size, k_size), sigma=(sigma, sigma)
            )
        else:
            blurred_imgs = imgs

        # --- Affine Patch-Warping Matrix ---
        # Output spans the supersampled grid (P = patch_size * ss).
        M = (
            torch.diag(torch.tensor([P, P, 1.0], dtype=torch.float32))
            .to(device)
            .unsqueeze(0)
            @ homogenize(torch.eye(2), b=torch.tensor([0.5, 0.5]))
            .unsqueeze(0)
            .to(device)
            @ homogenize(
                torch.eye(2).to(device).unsqueeze(0) / scales.view(-1, 1, 1) / sf
            )
            @ homogenize(blob_normalizations_)
            @ homogenize(
                torch.eye(2).to(device).unsqueeze(0).expand(coords.size(0), -1, -1),
                b=-coords,
            )
        )

        patches = kornia.geometry.transform.warp_perspective(
            blurred_imgs.expand(-1, 3, -1, -1),
            M,
            dsize=(P, P),
            padding_mode="fill",
            fill_value=torch.tensor([1.0, 1.0, 1.0], device=device),
        )[
            :, :1
        ]  # Zurück auf 1 Kanal pro Skala

        # Board validity on the very same warp: 0 outside the frame (the `fill_value`),
        # and — with a coverage image — 0 on the background *inside* the frame too.
        if return_mask:
            msrc = (mask_imgs if mask_imgs is not None
                    else torch.ones_like(imgs[:, :1]))
            valid = kornia.geometry.transform.warp_perspective(
                msrc.expand(-1, 3, -1, -1),
                M,
                dsize=(P, P),
                padding_mode="fill",
                fill_value=torch.tensor([0.0, 0.0, 0.0], device=device),
            )[:, :1]

        # Area-average each output pixel's supersampled footprint.
        if ss > 1:
            patches = torch.nn.functional.avg_pool2d(patches, kernel_size=ss, stride=ss)
            if valid is not None:
                valid = torch.nn.functional.avg_pool2d(valid, kernel_size=ss, stride=ss)

        multiscale_patches.append(patches)

    # Stapeln entlang der Kanal-Dimension -> Shape: [Batch, len(scale_factors), patch_size, patch_size]
    patches = torch.cat(multiscale_patches, dim=1)
    if return_mask:
        return patches, valid
    return patches


def extract_logpolar_patches(
    imgs,
    homographies,
    coords,
    scales,
    patch_size=64,
    inner_factor=2.0,
    outer_factor=96.0,
    supersample=3,
    return_mask=False,
    mask_imgs=None,
):
    """Extract a log-polar patch around each keypoint.

    The sampled annulus has inner radius ``inner_factor * scale`` and outer
    radius ``outer_factor * scale`` (measured in the shape-normalized frame),
    where ``scale`` is the per-feature scale in ``scales``. The radial axis
    (dim ``-1``) is log-spaced from the inner to the outer radius; the angular
    axis (dim ``-2``) spans a full turn. Shape normalization is identical to
    :func:`extract_multiscale_patches`.

    For anti-aliased sampling, the lattice is built at ``supersample`` sub-taps
    per output pixel along each axis and area-averaged back down. Because the
    sub-taps live in log-polar coordinates, they automatically span each output
    pixel's true source footprint — which grows with radius — so this
    approximates area interpolation and suppresses the aliasing that a single
    bilinear tap produces in the outer (condensed) radii. ``supersample=1``
    recovers the plain single-tap behaviour.

    Returns a tensor of shape ``(N, 1, patch_size, patch_size)`` (one channel,
    since the radial axis already encodes scale).
    """
    device = imgs.device
    N = coords.size(0)
    H, W = imgs.shape[-2:]

    normalizations = blob_normalizations(homographies, coords, device)
    # Linear map from the shape-normalized frame back into image pixels.
    inv_normalizations = torch.linalg.inv(normalizations)

    ss = max(1, int(supersample))
    P = patch_size * ss  # supersampled lattice resolution per axis

    # Log-polar sampling lattice, in units of the feature scale.
    radii = torch.exp(
        torch.linspace(
            math.log(inner_factor), math.log(outer_factor), P, device=device
        )
    )  # (P,) radial, log-spaced
    angles = torch.linspace(0.0, 2 * math.pi, P + 1, device=device)[
        :-1
    ]  # (P,) angular, uniform full turn
    aa, rr = torch.meshgrid(angles, radii, indexing="ij")  # (P, P): dim0 angular, dim1 radial
    lattice = torch.stack(
        [rr * torch.cos(aa), rr * torch.sin(aa)], dim=-1
    )  # (P, P, 2)

    # Scale to shape-normalized pixels, then map into the source image.
    offsets = lattice.view(1, P, P, 2) * scales.view(-1, 1, 1, 1)
    offsets = torch.einsum("nij,nhwj->nhwi", inv_normalizations, offsets)
    sample_coords = coords.view(N, 1, 1, 2) + offsets  # (x, y) image pixels

    # Normalize to grid_sample's [-1, 1] range (align_corners=True).
    grid = torch.stack(
        [
            sample_coords[..., 0] / (W - 1) * 2.0 - 1.0,
            sample_coords[..., 1] / (H - 1) * 2.0 - 1.0,
        ],
        dim=-1,
    )

    patches = torch.nn.functional.grid_sample(
        imgs,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # White fill outside the image, matching the cartesian extractor.
    oob = (grid.abs() > 1.0).any(dim=-1, keepdim=True).permute(0, 3, 1, 2)
    patches = torch.where(oob, torch.ones_like(patches), patches)

    # Per-sample board validity (1 = on the board, 0 = off-board). When a board-coverage
    # image is given (``mask_imgs``, same frame as ``imgs``) it is sampled on the *same*
    # lattice — this is the true validity for a warped/composited view, where off-board
    # is real background *inside* the frame. Without one, fall back to ``~oob`` (correct
    # for the identity/anchor view, whose board raster fills the frame).
    valid = None
    if return_mask:
        if mask_imgs is not None:
            valid = torch.nn.functional.grid_sample(
                mask_imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
            )
        else:
            valid = (~oob).to(patches.dtype)

    # Area-average each output pixel's supersampled footprint.
    if ss > 1:
        patches = torch.nn.functional.avg_pool2d(patches, kernel_size=ss, stride=ss)
        if valid is not None:
            valid = torch.nn.functional.avg_pool2d(valid, kernel_size=ss, stride=ss)

    if return_mask:
        return patches, valid
    return patches


class HomographyData(torch.utils.data.Dataset):
    images: torch.Tensor
    transforms: torch.Tensor
    transforms_inv: torch.Tensor
    size: tuple[int, int]
    c: int

    def __init__(
        self,
        images: Union[str, torch.Tensor],
        image_size: tuple[int, int] = (2000, 2000),
        in_memory=True,
        transform_params=None,
        transforms_per_image=1,
        homography_seed=None,  # seed the per-view warp draw; None = numpy's global RNG, i.e. a different draw every build

        augmentation=None,
        sift_batch_size=1000,
        sift_min_response_threshold=0.03,
        features_per_image=500,
        gt_keypoint_coords=None,
        gt_keypoint_scales=None,
        gt_keypoint_mask=None,
        patch_scale_factors=[16.0, 64.0, 128.0],  # NEU: Scale Space Faktoren
        patch_size=64,
        extraction_batch_size=512,
        patch_type="cartesian",  # "cartesian" oder "logpolar"
        logpolar_inner_factor=2.0,
        logpolar_outer_factor=32.0,
        supersample=3,  # sub-taps per output pixel per axis, area-averaged (both patch types)
        precompute_masks=False,  # logpolar only: also cache a per-patch board-validity mask (GT for the anchor view, for the learned-mask descriptor head)
        scale_quantile_range=None,  # (lo, hi) in [0, 1]: keep keypoints whose intrinsic blob scale falls in this quantile band
        scale_range=None,  # (lo, hi) absolute blob-scale bounds; alternative to scale_quantile_range
        max_keypoints=None,  # cap the kept keypoints to a fixed count (deterministic subsample) so different splits are exactly the same size
        subsample_seed=0,  # RNG seed for the max_keypoints subsample; keep fixed so a split's members are stable across runs
        background=None,  # local folder of scene images to place boards in front of; None disables all compositing/garbage
        background_kaggle_slug="arnaud58/landscape-pictures",  # kagglehub fallback when `background` is unset or missing on disk
        board_scale_range=(0.4, 0.7),  # board extent as a fraction of the frame (min, max); the placement's uniform scale
        background_lighting=True,  # brightness-match the board to the local background + apply a low-frequency shading gradient
        shading_strength=0.3,  # weight of the multiplicative shading field sampled from the background (0 disables the gradient)
        garbage_fraction=0.0,  # number of background distractor keypoints per board, as a fraction of that board's surviving blobs
        garbage_source="sift",  # "sift" (detect on the background) or "random" (uniform background points)
        background_seed=0,  # base RNG seed for placement/background choice/garbage; per-board seed derives from this + board index
        shuffle_keypoints=True,  # shuffle keypoint order once so appended garbage isn't clustered in the last batches
        shuffle_seed=0,  # RNG seed for that shuffle; fixed so a split's batch composition is stable across runs
        keypoint_jitter=0.0,  # std of the simulated detector position error, in px of the warped frame
        scale_jitter=0.0,  # std of the simulated detector scale error, log-normal relative (0.05 = +/-5%)
        jitter_seed=0,  # RNG seed for the jitter; per-view draw derives from this + the flat index
        cache_path=None,  # if set: load the prepared dataset from here when it exists, else build and save it
        **_,
    ):
        super().__init__()
        if transform_params is None:
            transform_params = {}
        if patch_type not in ("cartesian", "logpolar"):
            raise ValueError(
                f"patch_type must be 'cartesian' or 'logpolar', got {patch_type!r}"
            )
        self.size = image_size
        self.in_memory = in_memory
        self.patch_scale_factors = patch_scale_factors
        self.patch_size = patch_size
        self.patch_type = patch_type
        self.logpolar_inner_factor = logpolar_inner_factor
        self.logpolar_outer_factor = logpolar_outer_factor
        self.supersample = supersample
        # Board-validity masks are available for both patch types now (the cartesian
        # extractor samples them on the same warp). Cartesian masks are single-channel,
        # so they need a single patch scale factor — see extract_multiscale_patches.
        self.precompute_masks = bool(precompute_masks)
        if self.precompute_masks and patch_type == "cartesian" and len(patch_scale_factors) != 1:
            raise ValueError(
                "precompute_masks with patch_type='cartesian' needs exactly one "
                f"patch_scale_factor, got {list(patch_scale_factors)}"
            )
        self.precomputed_masks = None  # (N, 1, patch_size, patch_size) validity, if enabled
        self.keypoint_jitter = float(keypoint_jitter)
        self.scale_jitter = float(scale_jitter)
        self.jitter_seed = int(jitter_seed)
        self.precomputed_patches = None  # Platzhalter für RAM-Modus

        if augmentation is not None:
        # TODO: Check for all innner augmentations
            self.augmentation = v2.Compose([
                v2.ColorJitter(**augmentation.color_jitter),
                v2.GaussianBlur(
                    kernel_size=getattr(augmentation.gaussian_blur, 'kernel_size'),
                    # sigma=getattr(augmentation.gaussian_blur, 'sigma')
                ),
                v2.GaussianNoise(**augmentation.gaussian_noise),
            ])
        else:
            self.augmentation = lambda x: x
        print(f"Augmentation: {self.augmentation}")

        # Cache hit: restore the fully prepared dataset and skip the whole pipeline
        # (board rendering, compositing, SIFT, garbage, patch extraction). `images`
        # is ignored here — the caller (e.g. BlobBoardHomographyData) checks the cache
        # before generating boards, so nothing expensive has run yet.
        if cache_path is not None and os.path.exists(cache_path):
            self._load_cache(cache_path)
            return

        self.images = images
        if tuple(self.images.shape[-2:]) != tuple(self.size):
            # `image_size` is a claim about the rasters, not an instruction to resize
            # them: the sampled warps, the placement similarity (which centres the board
            # on the frame) and the collate's warp all build frames of this size and
            # sample the raster as if it filled them. A mismatch misplaces the board
            # silently rather than raising anywhere downstream.
            raise ValueError(
                f"image_size {tuple(self.size)} must equal the raster shape "
                f"{tuple(self.images.shape[-2:])}"
            )
        self.c = self.images.size(1)
        self.transform_params = transform_params
        # One generator threaded across every call, not one per call: the stream has to
        # advance or every board/view would get the identical warp.
        homography_rng = (
            None if homography_seed is None else np.random.default_rng(homography_seed)
        )
        self.transforms = torch.stack(
            [
                torch.stack(
                    [
                        torch.Tensor(
                            sample_homography(
                                image_size, rng=homography_rng, **transform_params
                            )
                        )
                        for _ in range(transforms_per_image)
                    ]
                )
                for _ in range(len(self.images))
            ]
        )
        self.transforms_inv = torch.linalg.inv(self.transforms)

        # --- Background compositing (blob-board / GT-keypoint path only) ---
        # Place each board as a shrunk sub-rectangle onto a real background scene so
        # the warped views look like the board photographed in a scene. The identity
        # (un-warped) view stays a clean reference: we keep the raw board-frame GT
        # coords/scales in the ``*_clean`` arrays and the original board rasters in
        # ``self.images_clean``, while ``self.images`` / ``keypoint_coords`` /
        # ``keypoint_scales`` carry the composite-frame values used by warped views.
        self.keypoint_is_garbage = None  # populated in the garbage block below
        self.images_clean = self.images
        self._board_masks = None
        gt_keypoint_coords_clean = None
        gt_keypoint_scales_clean = None
        self._garbage_fraction = float(garbage_fraction)
        self._garbage_source = garbage_source
        self._background_seed = int(background_seed)
        compositing = (
            isinstance(self.images, torch.Tensor)
            and gt_keypoint_coords is not None
            and gt_keypoint_scales is not None
            and (background is not None or background_kaggle_slug)
        )
        if compositing:
            bg_dir = resolve_background_dir(background, background_kaggle_slug)
            if bg_dir is None:
                print("No background images found; skipping compositing/garbage.")
                compositing = False
        if compositing:
            bg_paths = sorted(find_images(bg_dir))
            print(f"Compositing {self.images.size(0)} boards onto backgrounds from {bg_dir!r} ({len(bg_paths)} images)")
            gt_keypoint_coords = gt_keypoint_coords.clone()
            gt_keypoint_scales = gt_keypoint_scales.clone()
            gt_keypoint_coords_clean = gt_keypoint_coords.clone()
            gt_keypoint_scales_clean = gt_keypoint_scales.clone()
            clean_boards = self.images
            composites, masks = [], []
            for i in range(clean_boards.size(0)):
                gen = torch.Generator().manual_seed(int(background_seed) + i)
                A, s = sample_placement_similarity(self.size, board_scale_range, gen)
                bg_idx = torch.randint(len(bg_paths), (), generator=gen).item()
                bg = load_background(bg_paths[bg_idx], self.size)
                composite, mask = composite_board(
                    clean_boards[i], bg, A, self.size,
                    lighting=background_lighting, shading_strength=shading_strength,
                )
                composites.append(composite)
                masks.append(mask)
                # Map this board's GT blobs into the composite frame (warped path).
                coords = gt_keypoint_coords[i]  # (F, 2), board-raster px
                ones = torch.ones((coords.size(0), 1), dtype=coords.dtype)
                warped = (A @ torch.cat([coords, ones], dim=-1).T).T  # (F, 3)
                gt_keypoint_coords[i] = warped[:, :2] / warped[:, 2:3]
                gt_keypoint_scales[i] = gt_keypoint_scales[i] * s
            self.images = torch.stack(composites)
            self.images_clean = clean_boards
            self._board_masks = torch.stack(masks)

        with torch.no_grad():
            detector = kornia.feature.ScaleSpaceDetector(
                num_features=features_per_image,
                minima_are_also_good=True,
            )
            keypoints = []
            keypoint_coord_list = []
            keypoint_scale_list = []
            keypoint_coord_clean_list = []
            keypoint_scale_clean_list = []
            loop = tqdm(range(0, len(self.images), sift_batch_size))
            loop.set_description("Obtaining SIFT features")

            for i in loop:
                loop.set_description(
                    f"Obtaining SIFT features [{i}/{len(self.images)}]"
                )
                if i + sift_batch_size > len(self.images):
                    actual_sift_batch_size = len(self.images) - i
                else:
                    actual_sift_batch_size = sift_batch_size

                img = self.images[i : i + actual_sift_batch_size].cuda()
                if (
                    gt_keypoint_coords is not None
                    and gt_keypoint_scales is not None
                    and gt_keypoint_mask is not None
                ):
                    keypoint_coords = gt_keypoint_coords[i : i + actual_sift_batch_size]
                    keypoint_scales = gt_keypoint_scales[i : i + actual_sift_batch_size]
                    keypoint_mask = gt_keypoint_mask[i : i + actual_sift_batch_size]
                    features_per_image = gt_keypoint_coords.size(1)
                    if gt_keypoint_coords_clean is not None:
                        keypoint_coords_clean = gt_keypoint_coords_clean[i : i + actual_sift_batch_size]
                        keypoint_scales_clean = gt_keypoint_scales_clean[i : i + actual_sift_batch_size]
                    else:
                        keypoint_coords_clean = keypoint_coords
                        keypoint_scales_clean = keypoint_scales
                else:
                    img = torchvision.transforms.functional.rgb_to_grayscale(img)
                    lafs, responses = detector(img)

                    keypoint_coords = kornia.feature.get_laf_center(lafs)
                    keypoint_coords = torch.stack(
                        [keypoint_coords[..., 1], keypoint_coords[..., 0]], dim=-1
                    )
                    keypoint_scales = kornia.feature.get_laf_scale(lafs).squeeze()
                    keypoint_mask = responses > sift_min_response_threshold
                    # No compositing on the natural-image path: the identity view
                    # already uses the same image, so clean == detected.
                    keypoint_coords_clean = keypoint_coords
                    keypoint_scales_clean = keypoint_scales

                img_ids = torch.stack(
                    [
                        torch.full(
                            (features_per_image,),
                            (i + j) * (transforms_per_image + 1),
                            dtype=torch.int64,
                        ).cuda()
                        for j in range(actual_sift_batch_size)
                    ],
                    dim=0,
                )
                feature_ids = torch.stack(
                    [
                        torch.arange(features_per_image, dtype=torch.int64).cuda()
                        for _ in range(actual_sift_batch_size)
                    ],
                    dim=0,
                ) + (img_ids << 32)

                keypoints.append(
                    torch.stack([img_ids, feature_ids], dim=-1)[keypoint_mask].cpu()
                )
                keypoint_coord_list.append(keypoint_coords[keypoint_mask].cpu())
                keypoint_scale_list.append(keypoint_scales[keypoint_mask].cpu())
                keypoint_coord_clean_list.append(keypoint_coords_clean[keypoint_mask].cpu())
                keypoint_scale_clean_list.append(keypoint_scales_clean[keypoint_mask].cpu())

            self.keypoints = torch.cat(keypoints)
            self.keypoint_coords = torch.cat(keypoint_coord_list)
            self.keypoint_scales = torch.cat(keypoint_scale_list)
            # Raw board-frame coords/scales for the clean identity view (== the
            # composite-frame arrays when compositing is off).
            self.keypoint_coords_clean = torch.cat(keypoint_coord_clean_list)
            self.keypoint_scales_clean = torch.cat(keypoint_scale_clean_list)

        # Optionally restrict the dataset to a band of the (intrinsic) blob
        # scale distribution. Filtering happens on the per-keypoint arrays here,
        # before the view expansion in ``__getitem__``, so every view of a kept
        # keypoint is kept together (positives for contrastive/FPR metrics stay
        # intact). This lets a single board be split into e.g. "small" and
        # "large" blob validation sets whose FPRs are reported separately.
        if scale_quantile_range is not None or scale_range is not None:
            # Band on the raw *intrinsic* blob scale (clean, pre-placement) so the
            # small/medium/large splits reflect true blob size, not the random
            # per-board placement scale baked into ``keypoint_scales``.
            scales = self.keypoint_scales_clean
            if scale_quantile_range is not None:
                lo_q, hi_q = float(scale_quantile_range[0]), float(scale_quantile_range[1])
                lo = torch.quantile(scales, lo_q).item()
                hi = torch.quantile(scales, hi_q).item()
                include_hi = hi_q >= 1.0
            else:
                lo = -float("inf") if scale_range[0] is None else float(scale_range[0])
                hi = float("inf") if scale_range[1] is None else float(scale_range[1])
                include_hi = True
            # Half-open on the upper edge (except the top-most band) so adjacent
            # quantile bands partition the keypoints without overlap or gaps.
            mask = (scales >= lo) & ((scales <= hi) if include_hi else (scales < hi))
            n_before = self.keypoints.size(0)
            self.keypoints = self.keypoints[mask]
            self.keypoint_coords = self.keypoint_coords[mask]
            self.keypoint_scales = self.keypoint_scales[mask]
            self.keypoint_coords_clean = self.keypoint_coords_clean[mask]
            self.keypoint_scales_clean = self.keypoint_scales_clean[mask]
            print(
                f"Scale filter {scale_quantile_range or scale_range}: kept "
                f"{self.keypoints.size(0)}/{n_before} keypoints "
                f"(intrinsic scale in [{lo:.4f}, {hi:.4f}])"
            )

        # Cap to a fixed number of keypoints so sibling splits (e.g. the
        # small/medium/large blob-scale bands, which naturally hold different
        # counts) end up exactly the same size. The subsample is deterministic
        # (seeded generator) so a split's members are stable across runs, and we
        # keep the original ordering afterwards so downstream indexing is
        # unaffected. Applied after scale filtering, so the cap counts only the
        # keypoints that survived the band selection.
        if max_keypoints is not None and self.keypoints.size(0) > max_keypoints:
            n_before = self.keypoints.size(0)
            generator = torch.Generator().manual_seed(int(subsample_seed))
            perm = torch.randperm(n_before, generator=generator)[:max_keypoints]
            perm = perm.sort().values
            self.keypoints = self.keypoints[perm]
            self.keypoint_coords = self.keypoint_coords[perm]
            self.keypoint_scales = self.keypoint_scales[perm]
            self.keypoint_coords_clean = self.keypoint_coords_clean[perm]
            self.keypoint_scales_clean = self.keypoint_scales_clean[perm]
            print(
                f"Keypoint cap: subsampled {self.keypoints.size(0)}/{n_before} "
                f"keypoints (max_keypoints={max_keypoints}, seed={subsample_seed})"
            )
        elif max_keypoints is not None:
            print(
                f"Keypoint cap: only {self.keypoints.size(0)} keypoints available "
                f"(< max_keypoints={max_keypoints}); split will be smaller than the cap"
            )

        # --- Background "garbage" keypoints (pure-negative distractors) ---
        # Detect points on the composited background and add them as keypoints that
        # never form a positive pair (unique-per-view negative labels, see
        # __getitem__). Added AFTER the cap so every split shares the same constant
        # garbage set and stays the same size. Their identity-view patch is a bland
        # clean-board sample; only their warped views carry real background content.
        views = self.transforms.size(1) + 1
        self.keypoint_is_garbage = torch.zeros(self.keypoints.size(0), dtype=torch.bool)
        if self._board_masks is not None and self._garbage_fraction > 0:
            with torch.no_grad():
                garbage_detector = kornia.feature.ScaleSpaceDetector(
                    num_features=2000, minima_are_also_good=True,
                )
                # Exact target: since every split shares the same (capped) blob
                # count, keying the garbage count off it makes all splits the same
                # total size. We pool SIFT background detections (preferred) and
                # random background points (top-up), then pick exactly this many.
                target_total = int(round(self._garbage_fraction * self.keypoints.size(0)))
                sift_c, sift_s, sift_b = [], [], []
                rand_c, rand_s, rand_b = [], [], []
                for b in range(self.images.size(0)):
                    gen = torch.Generator().manual_seed(self._background_seed + 100003 + b)
                    bg_region = self._board_masks[b, 0] < 0.05  # strictly background
                    ys_all, xs_all = torch.nonzero(bg_region, as_tuple=True)
                    if ys_all.numel() == 0:
                        continue
                    if self._garbage_source == "sift":
                        lafs, _ = garbage_detector(self.images[b:b + 1].cuda())
                        # kornia get_laf_center returns (x, y), matching the (x, y)
                        # convention of the blob-board GT coords.
                        coords = kornia.feature.get_laf_center(lafs)[0].cpu()
                        scales = kornia.feature.get_laf_scale(lafs).reshape(-1).cpu()
                        xs = coords[:, 0].round().long().clamp(0, self.size[1] - 1)
                        ys = coords[:, 1].round().long().clamp(0, self.size[0] - 1)
                        on_bg = bg_region[ys, xs]
                        sift_c.append(coords[on_bg]); sift_s.append(scales[on_bg])
                        sift_b.append(torch.full((int(on_bg.sum()),), b, dtype=torch.int64))
                    # Random background candidates (used verbatim for the "random"
                    # source, and as a top-up when SIFT is sparse).
                    ridx = torch.randperm(ys_all.numel(), generator=gen)[: max(target_total, 256)]
                    rc = torch.stack([xs_all[ridx].float(), ys_all[ridx].float()], dim=-1)
                    rand_c.append(rc)
                    rand_s.append(torch.full((rc.size(0),), 2.0))
                    rand_b.append(torch.full((rc.size(0),), b, dtype=torch.int64))

                def _shuffled_cat(cs, ss, bs, seed):
                    if not cs:
                        return (torch.empty((0, 2)), torch.empty((0,)), torch.empty((0,), dtype=torch.int64))
                    c, s, bd = torch.cat(cs), torch.cat(ss), torch.cat(bs)
                    perm = torch.randperm(c.size(0), generator=torch.Generator().manual_seed(seed))
                    return c[perm], s[perm], bd[perm]

                sc, ss_, sb = _shuffled_cat(sift_c, sift_s, sift_b, self._background_seed + 11)
                rc, rs, rb = _shuffled_cat(rand_c, rand_s, rand_b, self._background_seed + 22)
                # Prefer SIFT points; top up the remainder with random background.
                take_sift = min(target_total, sc.size(0))
                rem = target_total - take_sift
                g_coords = torch.cat([sc[:take_sift], rc[:rem]])
                g_scales = torch.cat([ss_[:take_sift], rs[:rem]])
                g_board = torch.cat([sb[:take_sift], rb[:rem]])
                if g_coords.size(0) < target_total:
                    print(f"Only {g_coords.size(0)} garbage keypoints available (< target {target_total}); splits may differ in size")
                if g_coords.size(0) > 0:
                    # Unique negative feature ids (for the training sampler); the
                    # per-view FPR label is further made unique in __getitem__.
                    fids = -(1 + torch.arange(g_coords.size(0), dtype=torch.int64))
                    g_kp = torch.stack([g_board * views, fids], dim=-1)
                    self.keypoints = torch.cat([self.keypoints, g_kp])
                    self.keypoint_coords = torch.cat([self.keypoint_coords, g_coords])
                    self.keypoint_scales = torch.cat([self.keypoint_scales, g_scales])
                    # Garbage identity view mirrors the composite values.
                    self.keypoint_coords_clean = torch.cat([self.keypoint_coords_clean, g_coords])
                    self.keypoint_scales_clean = torch.cat([self.keypoint_scales_clean, g_scales])
                    self.keypoint_is_garbage = torch.cat(
                        [self.keypoint_is_garbage, torch.ones(g_coords.size(0), dtype=torch.bool)]
                    )
                    print(f"Added {g_coords.size(0)} garbage keypoints ({take_sift} sift + {min(rem, rc.size(0))} random) across {self.images.size(0)} boards")

        # Shuffle the keypoint order once, deterministically. Garbage is appended at
        # the end, so without this a `shuffle=False` validation loader hands whole
        # trailing batches of only-garbage keypoints to the metric — those contain no
        # real positive pair and report a meaningless FPR of 0, dragging the average
        # down. A keypoint's views stay contiguous (index = keypoint_i * views + j),
        # so its views still land in the same batch and positives are preserved.
        if shuffle_keypoints and self.keypoints.size(0) > 1:
            generator = torch.Generator().manual_seed(int(shuffle_seed))
            perm = torch.randperm(self.keypoints.size(0), generator=generator)
            self.keypoints = self.keypoints[perm]
            self.keypoint_coords = self.keypoint_coords[perm]
            self.keypoint_scales = self.keypoint_scales[perm]
            self.keypoint_coords_clean = self.keypoint_coords_clean[perm]
            self.keypoint_scales_clean = self.keypoint_scales_clean[perm]
            self.keypoint_is_garbage = self.keypoint_is_garbage[perm]

        avg_keypoints_per_image = self.keypoints.size(0) / len(self.images)
        print(f"{avg_keypoints_per_image=}")
        self.extraction_batch_size = extraction_batch_size
        self.compute_patches()

        if cache_path is not None:
            self._save_cache(cache_path)

    # Everything needed to reconstruct a prepared dataset without re-running the
    # pipeline. `_board_masks` is deliberately omitted (only used while generating
    # garbage) and `augmentation` is rebuilt from the constructor params.
    _CACHE_TENSORS = (
        "transforms", "transforms_inv",
        "keypoints", "keypoint_coords", "keypoint_scales",
        "keypoint_coords_clean", "keypoint_scales_clean", "keypoint_is_garbage",
        "images", "images_clean", "precomputed_patches", "precomputed_masks",
    )
    _CACHE_META = (
        "size", "c", "in_memory", "patch_type", "patch_size", "patch_scale_factors",
        "logpolar_inner_factor", "logpolar_outer_factor", "supersample", "precompute_masks",
        "transform_params", "patches_available", "extraction_batch_size",
        # __getitem__ reads these on every access, including after a cache load
        # (which returns from __init__ before they would otherwise be set).
        "keypoint_jitter", "scale_jitter", "jitter_seed",
    )

    def _save_cache(self, path):
        def _plain(v):
            if isinstance(v, (omegaconf.DictConfig, omegaconf.ListConfig)):
                return omegaconf.OmegaConf.to_container(v, resolve=True)
            return v

        state = {
            "tensors": {k: getattr(self, k, None) for k in self._CACHE_TENSORS},
            "meta": {k: _plain(getattr(self, k, None)) for k in self._CACHE_META},
        }
        # Write + atomic rename: concurrent jobs (e.g. a sweep) can never read a
        # half-written cache. Two cold jobs may both build it; last one wins, which
        # is wasteful but correct. Saving is best-effort — an unwritable cache dir
        # must never take down a training run.
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}"
            torch.save(state, tmp)
            os.replace(tmp, path)
            print(f"Cached prepared dataset -> {path}")
        except OSError as e:
            print(f"Could not write dataset cache to {path!r}: {e}")

    def _load_cache(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        for k, v in state["meta"].items():
            setattr(self, k, v)
        for k, v in state["tensors"].items():
            setattr(self, k, v)
        self._board_masks = None
        n_g = int(self.keypoint_is_garbage.sum()) if self.keypoint_is_garbage is not None else 0
        print(
            f"Loaded cached dataset from {path} "
            f"({self.keypoints.size(0)} keypoints, {n_g} garbage, len={len(self)})"
        )

    def compute_patches(self):
        self.patches_available = False
        if self.in_memory:
            # Log-polar erzeugt einen einzelnen Kanal (die Radialachse kodiert
            # bereits die Skala), Cartesian einen Kanal pro Scale-Space-Faktor.
            n_channels = (
                1 if self.patch_type == "logpolar" else len(self.patch_scale_factors)
            )
            print(f"Pre-extracting {self.patch_type} patches into memory...")
            # Allokiere Speicher für alle Patches (N, Channels, H, W)
            self.precomputed_patches = torch.empty(
                (len(self), n_channels, self.patch_size, self.patch_size),
                dtype=torch.float32,
            )
            if self.precompute_masks:
                self.precomputed_masks = torch.empty(
                    (len(self), 1, self.patch_size, self.patch_size),
                    dtype=torch.float32,
                )

            # Wir nutzen einen temporären DataLoader, um die bestehende Logik wiederzuverwenden
            extraction_loader = torch.utils.data.DataLoader(
                self,
                batch_size=self.extraction_batch_size,
                collate_fn=self.get_collate_func(),
                shuffle=False,
                num_workers=0,
            )

            idx = 0
            with torch.no_grad():
                for batch in tqdm(extraction_loader, desc="Warping Patches"):
                    # Rekonstruiere den Batched-Image-Tensor aus dem Dictionary
                    img_ids = batch["keypoints"][:, 0].cpu().numpy()
                    # The identity (un-warped) view is the clean reference: never
                    # augment it. Warped views (img_id % views < views-1) get the aug.
                    views = self.transforms.size(1) + 1
                    images = {
                        img_id: (img if img_id % views == views - 1 else self.augmentation(img))
                        for img_id, img in batch["images"].items()
                    }
                    imgs_tensor = torch.stack(
                        [images[img_id] for img_id in img_ids], dim=0
                    ).cuda()

                    # Board-coverage images (un-augmented — augmentation is photometric
                    # and would corrupt the 0/1 mask; the geometry is already baked in).
                    mask_tensor = None
                    if self.precompute_masks:
                        mask_images = batch["mask_images"]
                        mask_tensor = torch.stack(
                            [mask_images[img_id] for img_id in img_ids], dim=0
                        ).cuda()

                    if self.patch_type == "logpolar":
                        patches = extract_logpolar_patches(
                            imgs_tensor,
                            batch["homographies"].cuda(),
                            batch["keypoint_coords"].cuda(),
                            batch["scales"].cuda(),
                            patch_size=self.patch_size,
                            inner_factor=self.logpolar_inner_factor,
                            outer_factor=self.logpolar_outer_factor,
                            supersample=self.supersample,
                            return_mask=self.precompute_masks,
                            mask_imgs=mask_tensor,
                        )
                        if self.precompute_masks:
                            patches, valid = patches
                            self.precomputed_masks[idx : idx + valid.size(0)] = valid.cpu()
                    else:
                        patches = extract_multiscale_patches(
                            imgs_tensor,
                            batch["homographies"].cuda(),
                            batch["keypoint_coords"].cuda(),
                            batch["scales"].cuda(),
                            patch_size=self.patch_size,
                            scale_factors=self.patch_scale_factors,
                            supersample=self.supersample,
                            return_mask=self.precompute_masks,
                            mask_imgs=mask_tensor,
                        )
                        if self.precompute_masks:
                            patches, valid = patches
                            self.precomputed_masks[idx : idx + valid.size(0)] = valid.cpu()

                    self.precomputed_patches[idx : idx + patches.size(0)] = (
                        patches.cpu()
                    )
                    idx += patches.size(0)
            self.patches_available = True


    def __getitem__(self, index):
        keypoint_i = index // (self.transforms.size(1) + 1)
        homography_j = index % (self.transforms.size(1) + 1)
        homography_i = self.keypoints[keypoint_i, 0] // (self.transforms.size(1) + 1)

        if homography_j == self.transforms.size(1):
            # Identity view = clean reference: raw board-frame coords/scale, no warp
            # and (in compute_patches) no augmentation. Uses the clean board image.
            keypoint_coords = self.keypoint_coords_clean[keypoint_i]
            base_scale = self.keypoint_scales_clean[keypoint_i]
            scale_factor = torch.tensor(1.0)
        else:
            transform = self.transforms[homography_i, homography_j]
            source_coords = torch.cat(
                [self.keypoint_coords[keypoint_i], torch.ones((1,))], dim=-1
            )
            keypoint_coords = transform @ source_coords.unsqueeze(-1)
            # Linearize at the source point: `transform`'s domain is the composite
            # frame. Its warped image is the wrong point, and the un-normalized
            # homogeneous form doubly so — linearize_homography is not invariant to
            # that vector's overall scale.
            scale_factor = (
                linearize_homography(transform.unsqueeze(0), coords=source_coords.view(1, 3))
                .view(2, 2)
                .det()
                .abs()
                .sqrt()
            )
            keypoint_coords = (keypoint_coords[:2] / keypoint_coords[2:]).squeeze(-1)
            base_scale = self.keypoint_scales[keypoint_i]

        scale = base_scale * scale_factor
        # Simulated detector error, on the warped views only — the identity view is
        # the clean reference. Coords and scale here come from exact GT, so without
        # this the descriptor never sees the localization error a real detector makes.
        # The draw is keyed on the flat index, so it is fixed per (keypoint, view):
        # with in_memory=True the patch is extracted once, so this perturbs the
        # training pair rather than resampling per epoch.
        if homography_j != self.transforms.size(1) and (
            self.keypoint_jitter > 0 or self.scale_jitter > 0
        ):
            generator = torch.Generator().manual_seed(self.jitter_seed + index)
            if self.keypoint_jitter > 0:
                # In absolute px, not units of the blob's scale: the detector's
                # localization error is roughly constant (~1 px) across scales, so a
                # scale-relative jitter would be an order of magnitude too large on
                # the biggest blobs — which are the most jitter-sensitive. Expressing
                # it in px reproduces the real effect, which falls hardest on the
                # small blobs because their patch is normalized by a small sigma.
                keypoint_coords = keypoint_coords + torch.randn(
                    2, generator=generator
                ) * self.keypoint_jitter
            if self.scale_jitter > 0:
                # Log-normal: scale error is relative, and must stay positive.
                scale = scale * torch.exp(
                    torch.randn((), generator=generator) * self.scale_jitter
                )

        label = self.keypoints[keypoint_i, 1]
        if getattr(self, "keypoint_is_garbage", None) is not None and bool(
            self.keypoint_is_garbage[keypoint_i]
        ):
            # Untracked singleton: a globally-unique, always-negative label per
            # emitted patch, so background garbage never forms a positive pair.
            label = torch.tensor(-(1 + index), dtype=self.keypoints.dtype)

        res = {
            "keypoint": torch.stack(
                [
                    self.keypoints[keypoint_i, 0] + homography_j,
                    label,
                ]
            ),
            "keypoint_coords": keypoint_coords,
            "scales": scale,
            "homographies": (
                self.transforms[homography_i, homography_j]
                if homography_j < self.transforms.size(1)
                else torch.eye(3)
            ),
        }

        # NEU: Falls fertig berechnet, geben wir den Patch direkt hier mit raus
        if self.patches_available:
            res["patch"] = self.precomputed_patches[index]
            if getattr(self, "precomputed_masks", None) is not None:
                # GT board-validity mask + anchor flag for the learned-mask head. The
                # identity view (homography_j == last) is the anchor whose mask is GT;
                # warped views carry a mask too but the head predicts theirs instead.
                res["mask"] = self.precomputed_masks[index]
                res["is_anchor"] = torch.tensor(
                    homography_j == self.transforms.size(1), dtype=torch.bool
                )

        return res

    def __len__(self):
        return len(self.keypoints) * (self.transforms.size(1) + 1)

    def get_sampler(self, batch_size, m=4):
        """Class-balanced sampler so each batch holds ``m`` views per keypoint.

        Contrastive losses (SupCon/FPR95) need multiple views of the same
        physical keypoint in a batch to form positive pairs; plain shuffling
        scatters the ``transforms_per_image + 1`` views so most anchors end up
        with no positive. The per-index label is the keypoint's feature id,
        which repeats across its views (see ``__getitem__``/``__len__``).
        """
        from pytorch_metric_learning.samplers import MPerClassSampler

        views = self.transforms.size(1) + 1
        labels = self.keypoints[:, 1].repeat_interleave(views)
        return MPerClassSampler(
            labels, m=m, batch_size=batch_size, length_before_new_iter=len(labels)
        )


    def get_collate_func(self):
        def collate_homography(batch):
            # Wir sparen uns das Image-Loading, wenn wir die Patches schon haben!
            needs_images = "patch" not in batch[0]

            res = {
                "keypoints": torch.stack([item["keypoint"] for item in batch]),
                "keypoint_coords": torch.stack(
                    [item["keypoint_coords"] for item in batch]
                ),
                "scales": torch.stack([item["scales"] for item in batch]),
                "homographies": torch.stack([item["homographies"] for item in batch]),
                # Frame size (H, W) so downstream can bound warped coords correctly.
                "image_size": torch.tensor(self.size, dtype=torch.float32),
            }

            if not needs_images:
                # Patches sind da, packe sie in den Batch
                res["patches"] = torch.stack([item["patch"] for item in batch])
                if "mask" in batch[0]:
                    res["masks"] = torch.stack([item["mask"] for item in batch])
                    res["is_anchor"] = torch.stack([item["is_anchor"] for item in batch])
            else:
                # Fallback Logik für dynamische Extraktion (in_memory=False)
                img_ids = {item["keypoint"][0].item() for item in batch}
                views = self.transforms.size(1) + 1
                imgs = {}
                mask_imgs = {} if self.precompute_masks else None
                for img_id in img_ids:
                    board = img_id // views
                    is_identity = img_id % views == self.transforms.size(1)
                    # Identity (un-warped) view reads the clean board; warped views
                    # read the composite (== clean board when compositing is off).
                    source = self.images_clean if is_identity else self.images
                    img = source[board]
                    # Board-coverage source in the SAME frame: full validity on the clean
                    # raster (identity), else the composite board mask. Warped by the
                    # identical transform below so it stays aligned with `img`.
                    if self.precompute_masks:
                        if is_identity or self._board_masks is None:
                            msrc = torch.ones((1, *self.size), device=img.device)
                        else:
                            msrc = self._board_masks[board].to(img.device)
                    if img_id % (self.transforms.size(1) + 1) < self.transforms.size(1):
                        transform = self.transforms[
                            img_id // (self.transforms.size(1) + 1),
                            img_id % (self.transforms.size(1) + 1) - 1,
                        ].unsqueeze(0)
                        img = kornia.geometry.transform.warp_perspective(
                            img.unsqueeze(0).expand(-1, 3, -1, -1),
                            transform,
                            self.size,
                            padding_mode="fill",
                            fill_value=torch.tensor([1.0, 1.0, 1.0], device=img.device),
                        ).squeeze(0)[:1]
                        if self.precompute_masks:
                            # Off-board (outside the warped board) -> 0 = invalid.
                            msrc = kornia.geometry.transform.warp_perspective(
                                msrc.unsqueeze(0).expand(-1, 3, -1, -1),
                                transform,
                                self.size,
                                padding_mode="fill",
                                fill_value=torch.tensor([0.0, 0.0, 0.0], device=img.device),
                            ).squeeze(0)[:1]
                    assert img.size(0) == 1
                    imgs[img_id] = img
                    if self.precompute_masks:
                        mask_imgs[img_id] = msrc
                res["images"] = imgs
                if self.precompute_masks:
                    res["mask_images"] = mask_imgs

            return res

        return collate_homography
