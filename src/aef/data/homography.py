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


def extract_multiscale_patches(
    imgs, homographies, coords, scales, patch_size=64, scale_factors=[16.0, 64.0, 128.0]
):
    device = imgs.device
    blob_normalizations = torch.linalg.inv(
        linearize_homography(
            homographies,
            coords=torch.cat(
                [coords, torch.ones((1, 1)).expand(coords.size(0), 1).to(device)],
                dim=-1,
            ),
        )
    )
    _, S, Vh = torch.linalg.svd(blob_normalizations)
    Σ = torch.zeros((S.size(0), 2, 2), dtype=torch.float32, device=device)
    Σ[:, 0, 0] = S[..., 0]
    Σ[:, 1, 1] = S[..., 1]
    blob_normalizations = Σ @ Vh

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
        M = (
            torch.diag(torch.tensor([patch_size, patch_size, 1.0], dtype=torch.float32))
            .to(device)
            .unsqueeze(0)
            @ homogenize(torch.eye(2), b=torch.tensor([0.5, 0.5]))
            .unsqueeze(0)
            .to(device)
            @ homogenize(
                torch.eye(2).to(device).unsqueeze(0) / scales.view(-1, 1, 1) / sf
            )
            @ homogenize(blob_normalizations)
            @ homogenize(
                torch.eye(2).to(device).unsqueeze(0).expand(coords.size(0), -1, -1),
                b=-coords,
            )
        )

        patches = kornia.geometry.transform.warp_perspective(
            blurred_imgs.expand(-1, 3, -1, -1),
            M,
            dsize=(patch_size, patch_size),
            padding_mode="fill",
            fill_value=torch.tensor([1.0, 1.0, 1.0], device=device),
        )[
            :, :1
        ]  # Zurück auf 1 Kanal pro Skala

        multiscale_patches.append(patches)

    # Stapeln entlang der Kanal-Dimension -> Shape: [Batch, len(scale_factors), patch_size, patch_size]
    return torch.cat(multiscale_patches, dim=1)


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
        sift_batch_size=1000,
        sift_min_response_threshold=0.03,
        features_per_image=500,
        gt_keypoint_coords=None,
        gt_keypoint_scales=None,
        gt_keypoint_mask=None,
        patch_scale_factors=[16.0, 64.0, 128.0],  # NEU: Scale Space Faktoren
        **_,
    ):
        super().__init__()
        if transform_params is None:
            transform_params = {}
        self.size = image_size
        self.in_memory = in_memory
        self.patch_scale_factors = patch_scale_factors
        self.precomputed_patches = None  # Platzhalter für RAM-Modus

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

                if in_memory:
                    img = self.images[i : i + actual_sift_batch_size].cuda()
                else:
                    img = torch.stack(
                        [
                            self.load_and_resize(p)
                            for p in self.images[i : i + sift_batch_size]
                        ],
                        dim=0,
                    ).cuda()
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

        avg_keypoints_per_image = self.keypoints.size(0) / len(self.images)
        print(f"{avg_keypoints_per_image=}")

        # --- NEU: Pre-Extraktion des Scale Space ---
        self.patches_available = False
        if self.in_memory:
            print("Pre-extracting multiscale patches into memory...")
            # Allokiere Speicher für alle Patches (N, Channels, H, W)
            self.precomputed_patches = torch.empty(
                (len(self), len(self.patch_scale_factors), 64, 64), dtype=torch.float32
            )

            # Wir nutzen einen temporären DataLoader, um die bestehende Logik wiederzuverwenden
            extraction_loader = torch.utils.data.DataLoader(
                self,
                batch_size=512,
                collate_fn=self.get_collate_func(),
                shuffle=False,
                num_workers=0,
            )

            idx = 0
            with torch.no_grad():
                for batch in tqdm(extraction_loader, desc="Warping Patches"):
                    # Rekonstruiere den Batched-Image-Tensor aus dem Dictionary
                    img_ids = batch["keypoints"][:, 0].cpu().numpy()
                    imgs_tensor = torch.stack(
                        [batch["images"][img_id] for img_id in img_ids], dim=0
                    ).cuda()

                    patches = extract_multiscale_patches(
                        imgs_tensor,
                        batch["homographies"].cuda(),
                        batch["keypoint_coords"].cuda(),
                        batch["scales"].cuda(),
                        patch_size=64,
                        scale_factors=self.patch_scale_factors,
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

    def load_and_resize(self, img_path):
        return (
            self.resize(
                torchvision.io.decode_image(img_path, torchvision.io.ImageReadMode.GRAY)
            ).to(torch.float32)
            / 255
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
            }

            if not needs_images:
                # Patches sind da, packe sie in den Batch
                res["patches"] = torch.stack([item["patch"] for item in batch])
            else:
                # Fallback Logik für dynamische Extraktion (in_memory=False)
                img_ids = {item["keypoint"][0].item() for item in batch}
                imgs = {}
                for img_id in img_ids:
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
