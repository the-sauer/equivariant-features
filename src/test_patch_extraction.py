import torchvision

from aef.data import BlobBoardHomographyData

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
    image_size=(150, 150),
    suffix="train",
    polarity="dark",
    in_memory=True,
    patch_type="logpolar",
    logpolar_inner_factor=2.0,
    logpolar_outer_factor=96.0,
)

for i in range(1000):
    patch = dataset[i]["patch"]
    for j in range(patch.shape[0]):
        torchvision.utils.save_image(patch[j].unsqueeze(0), f"/raid/data/hsa/debug_imgs/patch_log_polar_{i}_{j}.png")
