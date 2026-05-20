import os
import random

import dotenv
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch

from aef.data.colmap import ColmapData

def visualize_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=15):
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
    # p1 = pts1[indices]
    # axes[0].scatter(p1[:,0], p1[:,1], color=cmap, s=30, marker='x', zorder=5)

    # l2 = F.unsqueeze(0) @ torch.stack([p1[:,0], p1[:,1], torch.ones(len(p1))], dim=-1).unsqueeze(-1)
    # x_vals = torch.tensor([0, w]).unsqueeze(0)
    # y_vals_2 = -(l2[:,0] * x_vals + l2[:,2]) / l2[:,1]
    # axes[1].plot(x_vals.expand(len(p1), -1), y_vals_2, cmap=cmap, linewidth=1)

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
    plt.show()

if __name__ == "__main__":
    dotenv.load_dotenv()
    path = "./data/datasets"
    if not os.path.exists(path):
        kagglehub.competition_download('image-matching-challenge-2025', output_dir=path, force_download=True)

    scene_dir = os.path.join(path, "south-building")
    dataset = ColmapData(scene_dir, image_size=(512, 512))

    print(f"Dataset has {len(dataset)} valid pairs with >20 common 3D points.")

    # Display N samples
    N = 3
    for _ in range(min(N, len(dataset))):
        sample_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[sample_idx]
        visualize_epipolar_lines(
            sample["image1"],
            sample["image2"],
            sample["pts1"],
            sample["pts2"],
            sample["fundamental_matrix"]
        )
