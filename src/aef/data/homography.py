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
import math
import os
from typing import Iterable, Union

import kornia
import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

from ..train.detector import homogenize, linearize_homography
from ..transforms.homography import sample_homography


def flip(f):
    def flipped_f(y, x):
        return f(x, y)

    return flipped_f


def curry(f):
    def function(x):
        def inner(y):
            return f(x, y)

        return inner

    return function


def uncurry(f):
    def function(x, y):
        return f(y)(x)

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


def load_images(
    dir, size, extensions=[".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
) -> torch.Tensor:
    resize = torchvision.transforms.Resize(size)
    return fchain(
        torch.stack,
        list,
        curry(map)(resize),
        curry(map)(torchvision.io.decode_image),
    )(find_images(dir, extensions))


def blob_normalizations(homographies, coords, device):
    """Affine shape normalization for each keypoint.

    Whitens the local Jacobian of ``homographies`` at ``coords`` (dropping the
    rotational part via SVD) so that the elliptical blob maps to an isotropic
    one. Shared by the cartesian and log-polar patch extractors.
    """
    normalizations = torch.linalg.inv(
        linearize_homography(
            homographies,
            coords=torch.cat(
                [coords, torch.ones((1, 1)).expand(coords.size(0), 1).to(device)],
                dim=-1,
            ),
        )
    )
    _, S, Vh = torch.linalg.svd(normalizations)
    Σ = torch.zeros((S.size(0), 2, 2), dtype=torch.float32, device=device)
    Σ[:, 0, 0] = S[..., 0]
    Σ[:, 1, 1] = S[..., 1]
    return Σ @ Vh


def extract_multiscale_patches(
    imgs, homographies, coords, scales, patch_size=64, scale_factors=[16.0, 64.0, 128.0],
    supersample=1,
):
    device = imgs.device
    blob_normalizations_ = blob_normalizations(homographies, coords, device)

    # Warp at `supersample` taps per output pixel along each axis and
    # area-average back down (in addition to the per-scale pre-blur below), so
    # each output pixel integrates its full source footprint. ``supersample=1``
    # recovers the plain single-tap behaviour.
    ss = max(1, int(supersample))
    P = patch_size * ss

    multiscale_patches = []

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

        # Area-average each output pixel's supersampled footprint.
        if ss > 1:
            patches = torch.nn.functional.avg_pool2d(patches, kernel_size=ss, stride=ss)

        multiscale_patches.append(patches)

    # Stapeln entlang der Kanal-Dimension -> Shape: [Batch, len(scale_factors), patch_size, patch_size]
    return torch.cat(multiscale_patches, dim=1)


def extract_logpolar_patches(
    imgs,
    homographies,
    coords,
    scales,
    patch_size=64,
    inner_factor=2.0,
    outer_factor=32.0,
    supersample=3,
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

    # Area-average each output pixel's supersampled footprint.
    if ss > 1:
        patches = torch.nn.functional.avg_pool2d(patches, kernel_size=ss, stride=ss)

    return patches


class HomographyData(torch.utils.data.Dataset):
    images: torch.Tensor | list[str]
    transforms: torch.Tensor
    transforms_inv: torch.Tensor
    size: tuple[int, int]
    c: int

    def __init__(
        self,
        images: Union[str, torch.Tensor],
        image_size: tuple[int, int] = (512, 512),
        in_memory=True,
        transform_params=None,
        transforms_per_image=1,
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
        scale_quantile_range=None,  # (lo, hi) in [0, 1]: keep keypoints whose intrinsic blob scale falls in this quantile band
        scale_range=None,  # (lo, hi) absolute blob-scale bounds; alternative to scale_quantile_range
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
        self.precomputed_patches = None  # Platzhalter für RAM-Modus

        if augmentation is not None:
        # TODO: Check for all innner augmentations
            self.augmentation = v2.Compose([
                v2.ColorJitter(**augmentation.color_jitter),
                v2.GaussianBlur(
                    kernel_size=getattr(augmentation.gaussian_blur, 'kernel_size'),
                    sigma=getattr(augmentation.gaussian_blur, 'sigma')
                ),
                v2.GaussianNoise(**augmentation.gaussian_noise),
            ])
        else:
            self.augmentation = lambda x: x
        print(f"Augmentation: {self.augmentation}")

        if isinstance(images, torch.Tensor):
            self.images = images
            self.c = self.images.size(1)
        else:
            if in_memory:
                self.images = (
                    load_images(images, size=image_size).to(torch.float32) / 255
                )
                self.c = self.images.size(1)
                self.resize = torchvision.transforms.Identity()  # Keine weitere Größenänderung erforderlich
            else:
                self.images = list(find_images(images))
                self.resize = torchvision.transforms.Resize(image_size)
                self.c = torchvision.io.decode_image(self.images[0]).size(0)
        self.transform_params = transform_params
        self.transforms = torch.stack(
            [
                torch.stack(
                    [
                        torch.Tensor(sample_homography(image_size, **transform_params))
                        for _ in range(transforms_per_image)
                    ]
                )
                for _ in range(len(self.images))
            ]
        )
        self.transforms_inv = torch.linalg.inv(self.transforms)

        with torch.no_grad():
            detector = kornia.feature.ScaleSpaceDetector(
                num_features=features_per_image,
                minima_are_also_good=True,
            )
            keypoints = []
            keypoint_coord_list = []
            keypoint_scale_list = []
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

                if hasattr(self, "images"):
                    img = self.images[i : i + actual_sift_batch_size].cuda()
                else:
                    img = torch.stack(
                        [
                            self.load_and_resize(p)
                            for p in self.images[i : i + sift_batch_size]
                        ],
                        dim=0,
                    ).cuda()
                    img = augmentation(img)
                if (
                    gt_keypoint_coords is not None
                    and gt_keypoint_scales is not None
                    and gt_keypoint_mask is not None
                ):
                    keypoint_coords = gt_keypoint_coords[i : i + actual_sift_batch_size]
                    keypoint_scales = gt_keypoint_scales[i : i + actual_sift_batch_size]
                    keypoint_mask = gt_keypoint_mask[i : i + actual_sift_batch_size]
                    features_per_image = gt_keypoint_coords.size(1)
                else:
                    img = torchvision.transforms.functional.rgb_to_grayscale(img)
                    lafs, responses = detector(img)

                    keypoint_coords = kornia.feature.get_laf_center(lafs)
                    keypoint_coords = torch.stack(
                        [keypoint_coords[..., 1], keypoint_coords[..., 0]], dim=-1
                    )
                    keypoint_scales = kornia.feature.get_laf_scale(lafs).squeeze()
                    keypoint_mask = responses > sift_min_response_threshold

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

            self.keypoints = torch.cat(keypoints)
            self.keypoint_coords = torch.cat(keypoint_coord_list)
            self.keypoint_scales = torch.cat(keypoint_scale_list)

        # Optionally restrict the dataset to a band of the (intrinsic) blob
        # scale distribution. Filtering happens on the per-keypoint arrays here,
        # before the view expansion in ``__getitem__``, so every view of a kept
        # keypoint is kept together (positives for contrastive/FPR metrics stay
        # intact). This lets a single board be split into e.g. "small" and
        # "large" blob validation sets whose FPRs are reported separately.
        if scale_quantile_range is not None or scale_range is not None:
            scales = self.keypoint_scales
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
            print(
                f"Scale filter {scale_quantile_range or scale_range}: kept "
                f"{self.keypoints.size(0)}/{n_before} keypoints "
                f"(intrinsic scale in [{lo:.4f}, {hi:.4f}])"
            )

        avg_keypoints_per_image = self.keypoints.size(0) / len(self.images)
        print(f"{avg_keypoints_per_image=}")
        self.extraction_batch_size = extraction_batch_size
        self.compute_patches()

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
                    images = {img_id: self.augmentation(img) for img_id, img in batch["images"].items()}
                    # TODO: Consider using no augmention for the non-warped images
                    imgs_tensor = torch.stack(
                        [images[img_id] for img_id in img_ids], dim=0
                    ).cuda()
                    # for i, img in enumerate(images.values()):
                    #     torchvision.utils.save_image(img, f"./debug_imgs/img_{i}.png")

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
                        )
                    else:
                        patches = extract_multiscale_patches(
                            imgs_tensor,
                            batch["homographies"].cuda(),
                            batch["keypoint_coords"].cuda(),
                            batch["scales"].cuda(),
                            patch_size=self.patch_size,
                            scale_factors=self.patch_scale_factors,
                            supersample=self.supersample,
                        )

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
            keypoint_coords = self.keypoint_coords[keypoint_i]
            scale_factor = torch.tensor(1.0)
        else:
            keypoint_coords = self.transforms[homography_i, homography_j] @ torch.cat(
                [self.keypoint_coords[keypoint_i], torch.ones((1,))], dim=-1
            ).unsqueeze(-1)
            scale_factor = (
                linearize_homography(
                    self.transforms[homography_i, homography_j].unsqueeze(0),
                    coords=keypoint_coords.view(1, 1, 1, 3),
                )
                .view(2, 2)
                .det()
                .abs()
                .sqrt()
            )
            keypoint_coords = (keypoint_coords[:2] / keypoint_coords[2:]).squeeze(-1)

        res = {
            "keypoint": torch.stack(
                [
                    self.keypoints[keypoint_i, 0] + homography_j,
                    self.keypoints[keypoint_i, 1],
                ]
            ),
            "keypoint_coords": keypoint_coords,
            "scales": self.keypoint_scales[keypoint_i] * scale_factor,
            "homographies": (
                self.transforms[homography_i, homography_j]
                if homography_j < self.transforms.size(1)
                else torch.eye(3)
            ),
        }

        # NEU: Falls fertig berechnet, geben wir den Patch direkt hier mit raus
        if self.patches_available:
            res["patch"] = self.precomputed_patches[index]

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

    def load_and_resize(self, img_path):
        return (
            self.resize(
                torchvision.io.decode_image(img_path, torchvision.io.ImageReadMode.GRAY)
            ).to(torch.float32)
            / 255
        )

    def resample_homographies(self):
        self.transforms = torch.stack(
            [
                torch.stack(
                    [
                        torch.Tensor(sample_homography(self.size, **self.transform_params))
                        for _ in range(self.transforms_per_image)
                    ]
                )
                for _ in range(len(self.images))
            ]
        )
        self.transforms_inv = torch.linalg.inv(self.transforms)
        self.compute_patches()


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
            else:
                # Fallback Logik für dynamische Extraktion (in_memory=False)
                img_ids = {item["keypoint"][0].item() for item in batch}
                imgs = {}
                for img_id in img_ids:
                    try:
                        img = self.images[img_id // (self.transforms.size(1) + 1)]
                    except TypeError:
                        img = (
                            self.load_and_resize(
                                self.images[img_id // (self.transforms.size(1) + 1)]
                            )
                            if not self.in_memory
                            else self.images[img_id // (self.transforms.size(1) + 1)]
                        )
                    if img_id % (self.transforms.size(1) + 1) < self.transforms.size(1):
                        img = kornia.geometry.transform.warp_perspective(
                            img.unsqueeze(0).expand(-1, 3, -1, -1),
                            (
                                self.transforms[
                                    img_id // (self.transforms.size(1) + 1),
                                    img_id % (self.transforms.size(1) + 1) - 1,
                                ]
                            ).unsqueeze(0),
                            self.size,
                            padding_mode="fill",
                            fill_value=torch.tensor([1.0, 1.0, 1.0], device=img.device),
                        ).squeeze(0)[:1]
                    assert img.size(0) == 1
                    imgs[img_id] = img
                res["images"] = imgs

            return res

        return collate_homography
