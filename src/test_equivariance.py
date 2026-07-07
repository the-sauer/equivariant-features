import torch

from aef.data.blobboards import BlobBoardHomographyData
from aef.models.blob_descriptor import *
from aef.train.descriptor import sanity_check


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    in_memory=True
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

model = BlobDescriptorHardNet()

batch = next(iter(dataloader))

img = batch["image"]
img_size = next(iter(img.values())).shape[-2:]
keypoints = batch["keypoints"].to(device)
keypoint_coords = batch["keypoint_coords"].to(device)
scales = batch["scales"].to(device)
coordinate_in_bound_mask = ((keypoint_coords >= 0).all(dim=1) & (keypoint_coords[:, 0] < img_size[1]) & (keypoint_coords[:, 1] < img_size[0])).cpu()

patches = sanity_check(
    torch.stack([img[img_id.item()].to(device) for img_id in keypoints[..., 0]], dim=0),
    batch["homographies"][coordinate_in_bound_mask].to(device),
    keypoint_coords,
    scales,
    scale_factor=16
).to(device)

out = model(patches)
print(torch.cdist(out[:1], out[1:2], p=2.0))
print(torch.cdist(out[1:2], out[2:3], p=2.0))
print(torch.cdist(out[2:3], out[:1], p=2.0))
