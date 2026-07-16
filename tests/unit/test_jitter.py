import torch

from aef.data.homography import HomographyData


def _dataset(keypoint_jitter=0.0, scale_jitter=0.0, jitter_seed=0, transforms_per_image=1):
    """A HomographyData with just the state __getitem__ touches.

    The real constructor runs SIFT on CUDA and the Julia board bridge, neither of
    which the jitter logic depends on.
    """
    data = object.__new__(HomographyData)
    n_keypoints = 6
    data.transforms = torch.eye(3).repeat(2, transforms_per_image, 1, 1)
    # a non-identity warp so the warped branch is genuinely exercised
    data.transforms[:, :, 0, 0] = 0.5
    data.transforms[:, :, 1, 1] = 0.5
    data.keypoints = torch.stack([
        torch.zeros(n_keypoints, dtype=torch.int64),
        torch.arange(n_keypoints, dtype=torch.int64),
    ], dim=-1)
    data.keypoint_coords = torch.full((n_keypoints, 2), 100.0)
    data.keypoint_scales = torch.full((n_keypoints,), 8.0)
    data.keypoint_coords_clean = torch.full((n_keypoints, 2), 100.0)
    data.keypoint_scales_clean = torch.full((n_keypoints,), 8.0)
    data.keypoint_is_garbage = torch.zeros(n_keypoints, dtype=torch.bool)
    data.keypoint_jitter = keypoint_jitter
    data.scale_jitter = scale_jitter
    data.jitter_seed = jitter_seed
    data.patches_available = False
    return data


def _views(data):
    return data.transforms.size(1) + 1


def test_identity_view_is_never_jittered():
    """The identity view is the clean reference: it must stay on exact GT however
    hard the warped views are jittered."""
    plain = _dataset()
    jittered = _dataset(keypoint_jitter=5.0, scale_jitter=0.5)
    views = _views(plain)

    for keypoint_i in range(4):
        index = keypoint_i * views + (views - 1)  # the identity view
        a, b = plain[index], jittered[index]
        torch.testing.assert_close(a["keypoint_coords"], b["keypoint_coords"])
        torch.testing.assert_close(a["scales"], b["scales"])


def test_warped_views_are_jittered():
    plain = _dataset()
    jittered = _dataset(keypoint_jitter=2.0, scale_jitter=0.2)
    views = _views(plain)

    moved = scaled = 0
    for keypoint_i in range(6):
        index = keypoint_i * views + 0  # a warped view
        a, b = plain[index], jittered[index]
        moved += not torch.allclose(a["keypoint_coords"], b["keypoint_coords"])
        scaled += not torch.allclose(a["scales"], b["scales"])
    assert moved == 6
    assert scaled == 6


def test_jitter_is_deterministic_per_view():
    """Patches are pre-extracted once, so a re-read must reproduce the same draw —
    otherwise the cached patch and the batch's coords/scales disagree."""
    first = _dataset(keypoint_jitter=2.0, scale_jitter=0.2)
    second = _dataset(keypoint_jitter=2.0, scale_jitter=0.2)
    views = _views(first)

    for keypoint_i in range(4):
        index = keypoint_i * views
        torch.testing.assert_close(
            first[index]["keypoint_coords"], second[index]["keypoint_coords"]
        )
        torch.testing.assert_close(first[index]["scales"], second[index]["scales"])
        # and stable across repeated reads of the same instance
        torch.testing.assert_close(
            first[index]["keypoint_coords"], first[index]["keypoint_coords"]
        )


def test_jitter_seed_changes_the_draw():
    a = _dataset(keypoint_jitter=2.0, scale_jitter=0.2, jitter_seed=0)
    b = _dataset(keypoint_jitter=2.0, scale_jitter=0.2, jitter_seed=1000)
    views = _views(a)

    differ = sum(
        not torch.allclose(a[k * views]["keypoint_coords"], b[k * views]["keypoint_coords"])
        for k in range(6)
    )
    assert differ == 6


def test_zero_jitter_is_exactly_the_unjittered_path():
    plain = _dataset()
    zero = _dataset(keypoint_jitter=0.0, scale_jitter=0.0)
    views = _views(plain)

    for index in range(6 * views):
        a, b = plain[index], zero[index]
        torch.testing.assert_close(a["keypoint_coords"], b["keypoint_coords"])
        torch.testing.assert_close(a["scales"], b["scales"])


def test_keypoint_jitter_is_in_pixels_not_units_of_scale():
    """Two keypoints of very different scale must get the same positional spread:
    detector localization error is ~constant in px, and scaling it by sigma would
    over-jitter the large blobs by an order of magnitude."""
    data = _dataset(keypoint_jitter=3.0)
    data.keypoint_scales = torch.tensor([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
    views = _views(data)

    base = torch.full((2,), 100.0)
    offsets = []
    for keypoint_i in range(6):
        item = data[keypoint_i * views + 0]
        # undo the warp's 0.5 scaling to recover the offset in warped px
        offsets.append((item["keypoint_coords"] - base * 0.5).norm())
    small, large = torch.stack(offsets[:3]), torch.stack(offsets[3:])

    # scale-relative jitter would make these differ by ~100x
    assert large.mean() < 10 * small.mean().clamp_min(1e-6)
