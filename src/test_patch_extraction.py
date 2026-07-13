from typing import NamedTuple

import torchvision

from aef.configuration import AugmentationConfig
from aef.data import BlobBoardHomographyData


class BlurConfig(NamedTuple):
    kernel_size: int
    sigma: float


augmentation_config = AugmentationConfig(
    color_jitter={
        "brightness": 0.5,
        "contrast": 1,
        "saturation": 1,
    },
    gaussian_blur=BlurConfig(31, 5.0),
    gaussian_noise={
        "mean": 0.0,
        "sigma": 0.1
    }
)

dataset = BlobBoardHomographyData(
    1,
    blobboard_params={
        "board_size": (142, 142),
        "cutoff": 2,
        "alpha": 1.6,
        "max_diameter_fraction": 0.4,
        "min_scale": 0.2,
        "frame_width": 10
    },
    transform_params={
        "fit_to_frame": True,
    },
    transform_per_image=8,
    augmentation=augmentation_config,
    image_size=(150, 150),
    suffix="train",
    polarity="dark",
    in_memory=True,
    patch_type="logpolar",
    logpolar_inner_factor=2.0,
    logpolar_outer_factor=96.0,
    extraction_batch_size=4096
)

for i in range(1000):
    patch = dataset[i]["patch"]
    for j in range(patch.shape[0]):
        torchvision.utils.save_image(patch[j].unsqueeze(0), f"/raid/data/hsa/debug_imgs/patch_log_polar_{i}_{j}.png")
