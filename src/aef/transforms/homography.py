# MIT License
#
# Copyright (c) 2018 Paul-Edouard Sarlin & Rémi Pautrat
# Copyright (c) 2025 Hendrik Sauer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging

import numpy as np
import torch


def sample_homography(
        shape: tuple[int | float, int | float] | list[int | float],
        perspective: bool = True,
        scaling: bool = True,
        rotation: bool = True,
        translation: bool = True,
        n_scales: int = 5,
        n_angles: int = 25,
        scaling_amplitude: float = 0.1,
        perspective_amplitude_x: float = 0.1,
        perspective_amplitude_y: float = 0.1,
        min_perspective_amplitude_x: float = 0.0,
        min_perspective_amplitude_y: float = 0.0,
        patch_ratio: float = 0.5,
        max_angle: float | str = np.pi,
        allow_artifacts: bool = False,
        translation_overflow: float = 0.0,
        fit_to_frame: bool = False,
        min_fit_scale: float = 0.0,
        rng=None,
) -> torch.Tensor:
    """Sample a random valid homography.

    **Note:** This function is an adapted version from [SuperPoints](github.com/rpautrat/SuperPoint) homography
    sampling.

    Computes the homography transformation between a random patch in the original image and a warped projection
    with the same image size. It maps the output point (warped patch) to a transformed input point (original
    patch). The original patch, which is initialized with a simple half-size centered crop, is iteratively
    projected, scaled, rotated and translated.

    Arguments:
        original_shape: A rank-2 `Tensor` specifying the height and width of the original image.
        patch_shape: A rank-2 `Tensor` specifying the height and width of the patch image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.
        fit_to_frame: If True, post-compose a uniform scale+translation so the entire original
            image maps inside the output frame (letterboxed), guaranteeing nothing is cut off.
        min_fit_scale: Lower bound on the fit_to_frame shrink factor, disabled (0.0) by default
            since even the worst-case fit still covers ~24% of the frame. Set >0 to floor the
            shrink at the cost of re-introducing minor clipping for extreme transforms. Only
            used when ``fit_to_frame`` is True.
        rng: A numpy ``Generator`` to draw from, or None for numpy's global RNG (the
            historical behaviour — a different draw on every call, unreproducible).
            Thread *one* generator across calls rather than passing a fresh
            ``default_rng(seed)`` each time: the stream must advance, or every view of
            every board gets the identical homography.

    Returns:
        An `array` of shape `(3,3)` corresponding to the homography.
    """
    # Both the module and a Generator expose .normal/.uniform with these signatures.
    rng = np.random if rng is None else rng

    def _truncated_normal(loc, scale, shape):
        result = rng.normal(loc, scale, shape)
        while any(result < loc - 2 * scale) or any(result > loc + 2 * scale):
            result = rng.normal(loc, scale, shape)
            logging.debug("Recalculated truncated normal")
        return result
    
    pi = np.pi
    π = np.pi
    
    if isinstance(max_angle, str):
        max_angle = eval(max_angle)

    # Corners of the output image
    margin = (1 - patch_ratio) / 2
    pts1 = margin + np.array([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]], np.float32)
    pts1.flags.writeable = False
    # Corners of the input patch
    pts2 = pts1.copy()

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = _truncated_normal(0.0, (perspective_amplitude_y - min_perspective_amplitude_y) / 2, (1,)) + min_perspective_amplitude_y
        h_displacement_left = _truncated_normal(0.0, (perspective_amplitude_x - min_perspective_amplitude_x) / 2, (1,)) + min_perspective_amplitude_x
        h_displacement_right = _truncated_normal(0.0, (perspective_amplitude_x - min_perspective_amplitude_x) / 2, (1,)) + min_perspective_amplitude_x
        pts2 += np.stack([
            np.concatenate([h_displacement_left, perspective_displacement], axis=0),
            np.concatenate([h_displacement_left, -perspective_displacement], axis=0),
            np.concatenate([h_displacement_right, perspective_displacement], axis=0),
            np.concatenate([h_displacement_right, -perspective_displacement], axis=0),
        ])

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = np.concatenate([[1.0], _truncated_normal(1, scaling_amplitude / 2, (n_scales,))], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center
        if allow_artifacts:
            valid = np.arange(start=1, stop=n_scales + 1)  # all scales are valid except scale=1
        else:
            valid = np.where(np.all((scaled >= 0.0) & (scaled <= 1.0), axis=(1, 2)))[0]
        idx = valid[int(rng.uniform(low=0, high=valid.shape[0]))]
        pts2 = scaled[idx]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.expand_dims(
            np.stack([
                rng.uniform(low=-t_min[0], high=t_max[0]),
                rng.uniform(low=-t_min[1], high=t_max[1])
            ]),
            axis=0,
        )

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        angles = np.concatenate([[0.0], angles], axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(
            np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], axis=1),
            (-1, 2, 2)
        )
        rotated = np.tile(np.expand_dims(pts2 - center, axis=0), (n_angles + 1, 1, 1)) @ rot_mat + center
        if allow_artifacts:
            valid = np.arange(1, n_angles + 1)  # all angles are valid, except angle=0
        else:
            valid = np.where(np.all((rotated >= 0.0) & (rotated <= 1.0), axis=(1, 2)))[0]
        idx = valid[int(rng.uniform(low=0, high=valid.shape[0]))]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = tuple(map(float, shape[::-1]))  # different convention [y, x]
    pts1 = pts1 * np.expand_dims(shape, axis=0)
    pts2 = pts2 * np.expand_dims(shape, axis=0)

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0).T
    flat_homography = np.linalg.lstsq(a_mat, p_mat)[0].T
    homography = np.ones(shape=(3, 3))
    homography[0, :] = flat_homography[0][:3]
    homography[1, :] = flat_homography[0][3:6]
    homography[2, :2] = flat_homography[0][6:]

    if fit_to_frame:
        # Post-compose a uniform scale + translation so the *entire* original
        # image maps inside the output frame (letterbox), guaranteeing nothing
        # is cut off. `shape` is (W, H) here; the homography maps original -> warped.
        w, h = shape
        corners = np.array(
            [[0.0, 0.0, 1.0], [0.0, h, 1.0], [w, h, 1.0], [w, 0.0, 1.0]]
        ).T  # (3, 4): homogeneous (x, y, 1) corners of the full original image
        warped = homography @ corners
        warped = warped[:2] / warped[2]  # (2, 4): (x, y) of the warped corners
        (xmin, ymin), (xmax, ymax) = warped.min(axis=1), warped.max(axis=1)
        s = min(w / (xmax - xmin), h / (ymax - ymin))  # shrink to fit, keep aspect
        s = max(s, min_fit_scale)  # cap the shrink; extreme warps overflow slightly instead
        tx = (w - s * (xmax - xmin)) / 2 - s * xmin  # center the letterboxed content
        ty = (h - s * (ymax - ymin)) / 2 - s * ymin
        fit = np.array([[s, 0.0, tx], [0.0, s, ty], [0.0, 0.0, 1.0]])
        homography = fit @ homography

    return torch.tensor(homography, dtype=torch.float32)
