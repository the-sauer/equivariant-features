import os
import random

import dotenv
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch

from aef.data.colmap import ColmapData

def visualize_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=15, name=None):
    """
    Shows image pairs with annotated matches and epipolar lines.
    """
    _, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Convert tensor (C, H, W) to numpy (H, W, C)
    img1_np = img1.permute(1, 2, 0).numpy()
    img2_np = img2.permute(1, 2, 0).numpy()

    axes[0].imshow(img1_np)
    axes[0].set_title("Image 1 with Epipolar Lines")
    axes[1].imshow(img2_np)
    axes[1].set_title("Image 2 with Epipolar Lines")

    num_pts = len(pts1)
    if num_pts == 0:
        plt.show()
        return

    indices = np.random.default_rng().choice(num_pts, min(num_pts, num_lines), replace=False)
    cmap = plt.get_cmap('hsv', len(indices))

    h, w = img1_np.shape[:2]

    for idx_c, i in enumerate(indices):
        p1 = pts1[i].numpy()
        p2 = pts2[i].numpy()
        color = cmap(idx_c)

        axes[0].scatter(p1[0], p1[1], color=color, s=30, marker='x', zorder=5)
        axes[1].scatter(p2[0], p2[1], color=color, s=30, marker='x', zorder=5)

        # Epipolar line in Image 2 corresponding to point in Image 1: l2 = F * p1
        p1_h = np.array([p1[0], p1[1], 1.0])
        l2 = F.numpy() @ p1_h

        x_vals = np.array([0, w])
        y_vals_2 = -(l2[0] * x_vals + l2[2]) / l2[1]
        axes[1].plot(x_vals, y_vals_2, color=color, linewidth=1, zorder=4)

        # Epipolar line in Image 1 corresponding to point in Image 2: l1 = F.T * p2
        p2_h = np.array([p2[0], p2[1], 1.0])
        l1 = F.numpy().T @ p2_h

        y_vals_1 = -(l1[0] * x_vals + l1[2]) / l1[1]
        axes[0].plot(x_vals, y_vals_1, color=color, linewidth=1, zorder=4)

    # Clean limits to ignore lines extending out of bounds
    axes[0].set_xlim(0, w)
    axes[0].set_ylim(h, 0)
    axes[1].set_xlim(0, w)
    axes[1].set_ylim(h, 0)

    plt.tight_layout()
    plt.savefig(f"/data/debug_imgs/epipolar_lines_{name}.png")
    plt.close()

if __name__ == "__main__":
    dotenv.load_dotenv()
    path = "/data/datasets"

    scene_dir = os.path.join(path, "south-building")
    dataset = ColmapData(scene_dir, image_size=(512, 512))

    sample_idx = random.randint(0, len(dataset) - 1)
    sample = dataset.get_collate_func()([dataset[i] for i in range(len(dataset))])

    feature_ids = sample["keypoints"][:, 1].unique().tolist()

    for feature_id in feature_ids:
        feature_mask = sample["keypoints"][:, 1] == feature_id
        if feature_mask.sum() < 2:
            continue

        img_ids = sample["keypoints"][feature_mask][:, 0].unique().tolist()
        for indices in torch.triu_indices(len(img_ids), len(img_ids), offset=1).T:
            img_1_id = img_ids[indices[0]]
            img_2_id = img_ids[indices[1]]
            pt_1 = sample["keypoint_coords"][feature_mask][sample["keypoints"][feature_mask][:, 0] == img_1_id][0]
            pt_2 = sample["keypoint_coords"][feature_mask][sample["keypoints"][feature_mask][:, 0] == img_2_id][0]
            F = sample["fundamental"][img_1_id, img_2_id]
            e = torch.abs(torch.tensor([pt_1[0], pt_1[1], 1]).unsqueeze(0) @ F @ torch.tensor([pt_2[0], pt_2[1], 1]).unsqueeze(1)).item()
            if e < 2:
                continue
            print(f"{img_1_id=}, {img_2_id=}, {feature_id=}, {e=}")
            visualize_epipolar_lines(
                sample["images"][img_1_id],
                sample["images"][img_2_id],
                pt_1.unsqueeze(0),
                pt_2.unsqueeze(0),
                F,
                name=f"{img_1_id}_{img_2_id}_{feature_id}"
            )
