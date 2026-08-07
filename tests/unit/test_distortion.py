import math

import pytest
import torch

from aef.data.homography import (
    HomographyData,
    blob_normalizations,
    dataset_cache_key,
)
from aef.transforms.distortion import (
    RadialDistortion,
    render_view,
    sample_radial_distortion,
)


SIZE = (480, 640)  # (H, W)


def _lens(lambda_1=0.12, lambda_2=-0.03, n=1, dtype=torch.float32):
    lambdas = torch.tensor([[lambda_1, lambda_2]], dtype=dtype).expand(n, 2)
    return RadialDistortion(lambdas, size=SIZE)


def _points(n, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.stack(
        [
            torch.rand(n, generator=g) * SIZE[1],
            torch.rand(n, generator=g) * SIZE[0],
        ],
        dim=-1,
    ).to(dtype)


def test_zero_lambdas_are_exactly_the_identity():
    """The pinhole case must be bit-clean, not merely close.

    Everything upstream stays on the old code path when no lens is configured; a lens
    that only *nearly* reduces to the identity would make "distortion off" a different
    dataset from the one every existing cache holds.
    """
    lens = RadialDistortion(torch.zeros(1, 2), size=SIZE)
    points = _points(64)

    assert lens.is_identity
    torch.testing.assert_close(lens.undistort(points), points, rtol=0, atol=0)
    torch.testing.assert_close(lens.distort(points), points, rtol=0, atol=0)
    torch.testing.assert_close(
        lens.jacobian_undistort(points), torch.eye(2).expand(64, 2, 2), rtol=0, atol=0
    )
    torch.testing.assert_close(
        lens.det_jacobian_undistort(points), torch.ones(64), rtol=0, atol=0
    )


def test_distort_inverts_undistort():
    """Newton on the radius really solves it — in both composition orders."""
    lens = _lens(n=256, dtype=torch.float64)
    points = _points(256, dtype=torch.float64)

    torch.testing.assert_close(
        lens.distort(lens.undistort(points)), points, rtol=1e-9, atol=1e-9
    )
    torch.testing.assert_close(
        lens.undistort(lens.distort(points)), points, rtol=1e-9, atol=1e-9
    )


def test_distort_is_exact_at_the_principal_point():
    """The centre is the one place the direction ``(q - c)/|q - c|`` is undefined."""
    lens = _lens()
    center = lens.center.view(1, 2)

    torch.testing.assert_close(lens.distort(center), center)
    torch.testing.assert_close(lens.undistort(center), center)
    assert torch.isfinite(lens.jacobian_undistort(center)).all()


def test_positive_lambdas_are_barrel():
    """Sign convention: λ ≥ 0 means the *rendered* view compresses its periphery.

    The λ are stated on the undistortion, so getting this backwards would silently
    train on pincushion views while the config says fisheye.
    """
    lens = _lens(lambda_1=0.2, lambda_2=0.05)
    corner = torch.tensor([[float(SIZE[1] - 1), float(SIZE[0] - 1)]])

    ideal_radius = (lens.undistort(corner) - lens.center).norm()
    observed_radius = (lens.distort(corner) - lens.center).norm()
    corner_radius = (corner - lens.center).norm()

    assert observed_radius < corner_radius < ideal_radius


def test_jacobian_matches_finite_differences():
    lens = _lens(n=32, dtype=torch.float64)
    points = _points(32, seed=3, dtype=torch.float64)
    step = 1e-5

    analytic = lens.jacobian_undistort(points)
    for axis in range(2):
        offset = torch.zeros(2, dtype=torch.float64)
        offset[axis] = step
        numeric = (lens.undistort(points + offset) - lens.undistort(points - offset)) / (
            2 * step
        )
        torch.testing.assert_close(analytic[..., axis], numeric, rtol=1e-6, atol=1e-6)


def test_det_jacobian_matches_the_full_jacobian():
    lens = _lens(n=32, dtype=torch.float64)
    points = _points(32, seed=4, dtype=torch.float64)

    torch.testing.assert_close(
        lens.det_jacobian_undistort(points),
        torch.linalg.det(lens.jacobian_undistort(points)),
    )


def test_folding_lambdas_are_rejected():
    """A radial map that stops increasing folds the image onto itself."""
    with pytest.raises(ValueError, match="non-invertible"):
        RadialDistortion(torch.tensor([[-2.0, 0.0]]), size=SIZE)
    # Fine within the frame, folds just outside it — still rejected, because warped
    # keypoints legitimately land outside the frame before they are filtered.
    with pytest.raises(ValueError, match="non-invertible"):
        RadialDistortion(torch.tensor([[0.0, -0.2]]), size=SIZE)
    RadialDistortion(torch.tensor([[0.0, -0.2]]), size=SIZE, max_rho=0.9)


def test_sample_radial_distortion():
    rng_a = __import__("numpy").random.default_rng(0)
    spec = {"lambda1": [0.05, 0.2], "lambda2": 0.01}

    draws = torch.stack([sample_radial_distortion(spec, rng_a) for _ in range(32)])
    assert ((draws[:, 0] >= 0.05) & (draws[:, 0] <= 0.2)).all()
    assert (draws[:, 1] == 0.01).all()
    assert draws[:, 0].std() > 0  # the generator is threaded, not re-seeded per draw

    torch.testing.assert_close(sample_radial_distortion(None), torch.zeros(2))
    torch.testing.assert_close(sample_radial_distortion({}), torch.zeros(2))


# --- Rendering ---------------------------------------------------------------


def _ramp_image(height, width):
    """An image whose value at (x, y) is a known smooth function, so a rendered pixel
    can be checked against the source point it is supposed to have come from."""
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return (0.3 * xs / width + 0.7 * ys / height).view(1, 1, height, width)


def test_render_view_without_a_lens_is_the_identity_warp():
    image = _ramp_image(64, 96)
    out = render_view(image, torch.eye(3).unsqueeze(0), None, (64, 96))
    torch.testing.assert_close(out, image, rtol=1e-5, atol=1e-5)


def test_render_view_samples_the_undistorted_source_point():
    """out(q) == source(H⁻¹(U(q))) — the defining property of the renderer."""
    height, width = 64, 96
    image = _ramp_image(height, width)
    lens = RadialDistortion(torch.tensor([[0.15, 0.02]]), size=(height, width))
    homography = torch.tensor(
        [[1.1, 0.05, -3.0], [-0.04, 0.95, 2.0], [1e-5, -2e-5, 1.0]]
    ).unsqueeze(0)

    out = render_view(image, homography, lens, (height, width))

    # Probe well inside the frame, where nothing is out of bounds.
    q = torch.tensor([[[float(x), float(y)] for x in range(20, 70, 7)] for y in range(15, 50, 5)])
    q = q.reshape(1, -1, 2)
    ideal = lens.undistort(q)
    source = torch.einsum(
        "nij,nkj->nki",
        torch.linalg.inv(homography),
        torch.cat([ideal, torch.ones_like(ideal[..., :1])], dim=-1),
    )
    source = source[..., :2] / source[..., 2:3]
    assert (source[..., 0] > 1).all() and (source[..., 0] < width - 2).all()
    assert (source[..., 1] > 1).all() and (source[..., 1] < height - 2).all()

    expected = 0.3 * source[0, :, 0] / width + 0.7 * source[0, :, 1] / height
    got = out[0, 0][q[0, :, 1].long(), q[0, :, 0].long()]
    # Bilinear sampling of a bilinear ramp is exact up to float noise.
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


# --- Shape normalization with a lens in the chain ----------------------------


def _homography(seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    h = torch.eye(3, dtype=torch.float64)
    h[:2, :2] += 0.4 * (torch.rand((2, 2), generator=g, dtype=torch.float64) - 0.5)
    h[:2, 2] = 30.0 * (torch.rand(2, generator=g, dtype=torch.float64) - 0.5)
    h[2, :2] = 1e-4 * (torch.rand(2, generator=g, dtype=torch.float64) - 0.5)
    return h.to(dtype)


def _view_map(homography, lens, point):
    """The true view map ``D(H(p))`` of a single source point."""
    projected = homography @ torch.cat([point, torch.ones(1, dtype=point.dtype)])
    ideal = (projected[:2] / projected[2]).view(1, 2)
    return lens.distort(ideal).view(2)


def _view_jacobian(homography, lens, point, step=1e-4):
    """``J[i, j] = ∂out_i/∂in_j`` of the composed view map, by central differences."""
    columns = []
    for axis in range(2):
        offset = torch.zeros(2, dtype=point.dtype)
        offset[axis] = step
        columns.append(
            (
                _view_map(homography, lens, point + offset)
                - _view_map(homography, lens, point - offset)
            )
            / (2 * step)
        )
    return torch.stack(columns, dim=-1)


def test_blob_normalizations_ignores_a_zero_lens():
    """Passing an all-zero lens must not perturb the pinhole result at all."""
    homography = _homography(dtype=torch.float32).unsqueeze(0).expand(8, 3, 3)
    coords = _points(8, seed=5)
    lens = RadialDistortion(torch.zeros(8, 2), size=SIZE)

    torch.testing.assert_close(
        blob_normalizations(homography, coords, coords.device, lens),
        blob_normalizations(homography, coords, coords.device),
        rtol=0,
        atol=0,
    )


def test_blob_normalizations_isotropizes_the_lensed_view():
    """``N @ J_(D∘H)(p)`` must still be exactly ``sqrt(det J)`` times a rotation.

    Same contract as the pinhole case, now over the composed view map. It fails if the
    lens Jacobian is composed on the wrong side, or if the homography is linearized at
    the observed point instead of the undistorted one — both of which still produce a
    plausible-looking det-1 matrix, just not one that circularizes the blob.
    """
    n = 12
    lens64 = _lens(n=1, dtype=torch.float64)
    lens32 = _lens(n=n, dtype=torch.float32)

    for i in range(n):
        homography64 = _homography(seed=i)
        source = torch.tensor([120.0 + 17 * i, 90.0 + 23 * i], dtype=torch.float64)

        observed = _view_map(homography64, lens64, source)
        jacobian = _view_jacobian(homography64, lens64, source)

        normalization = blob_normalizations(
            homography64.to(torch.float32).unsqueeze(0),
            observed.to(torch.float32).view(1, 2),
            observed.device,
            RadialDistortion(lens32.lambdas[:1], size=SIZE, validate=False),
        )[0].to(torch.float64)

        # det == 1: shape and orientation only, size stays with `scales`.
        torch.testing.assert_close(
            torch.linalg.det(normalization), torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-5, atol=1e-5,
        )

        composed = normalization @ jacobian
        expected_scale = torch.linalg.det(jacobian).abs()
        torch.testing.assert_close(
            composed @ composed.T,
            expected_scale * torch.eye(2, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-4,
        )


def test_lens_changes_the_blob_size():
    """``scales`` has to pick up the lens' own ``|det J|``.

    The pipeline divides the patch lattice by ``scales``, so a lens whose determinant
    is dropped leaves the same physical blob at a view-dependent size in its patch —
    the exact failure the det-1 normalization exists to avoid.
    """
    lens = _lens(lambda_1=0.25, lambda_2=0.0)
    corner = torch.tensor([[float(SIZE[1] - 1), float(SIZE[0] - 1)]])
    observed = lens.distort(corner)

    # Barrel: the periphery is compressed, so a blob out there shrinks.
    assert lens.det_jacobian_undistort(observed).item() > 1.0
    # ... and the principal point is untouched.
    torch.testing.assert_close(
        lens.det_jacobian_undistort(lens.center.view(1, 2)), torch.ones(1)
    )


# --- Dataset plumbing --------------------------------------------------------


def _bare_dataset(lambdas, homography, coords, scale=3.0, size=SIZE, images=None):
    """A ``HomographyData`` with only the fields ``__getitem__`` reads.

    Built without ``__init__`` on purpose: the real constructor renders boards and runs
    a detector, neither of which this is about. What is under test is that the item a
    warped view yields is the *lensed* one — coords through ``D``, scale through its
    determinant — and that the lens rides along to the extractors.
    """
    data = HomographyData.__new__(HomographyData)
    data.size = size
    data.transforms = homography.view(1, 1, 3, 3)
    data.distortions = torch.tensor(lambdas, dtype=torch.float32).view(1, 1, 2)
    data.keypoints = torch.tensor([[0, 7]], dtype=torch.int64)
    data.keypoint_coords = coords.view(1, 2)
    data.keypoint_scales = torch.tensor([scale])
    data.keypoint_coords_clean = coords.view(1, 2)
    data.keypoint_scales_clean = torch.tensor([scale])
    data.keypoint_is_garbage = torch.zeros(1, dtype=torch.bool)
    data.keypoint_jitter = 0.0
    data.scale_jitter = 0.0
    data.jitter_seed = 0
    data.precompute_masks = False
    data.patches_available = images is None
    if images is None:
        data.precomputed_patches = torch.zeros(2, 1, 4, 4)
        data.precomputed_masks = None
    else:
        data.images = images
        data.images_clean = images
        data._board_masks = None
    return data


def test_getitem_yields_the_lensed_view():
    homography = _homography(seed=2, dtype=torch.float32)
    source = torch.tensor([180.0, 140.0])
    lambdas = (0.18, -0.02)
    data = _bare_dataset(lambdas, homography, source)
    lens = RadialDistortion(torch.tensor([lambdas]), size=SIZE)

    warped = data[0]  # view 0 is the warped one; view 1 is the identity reference
    identity = data[1]

    projected = homography @ torch.cat([source, torch.ones(1)])
    ideal = (projected[:2] / projected[2]).view(1, 2)
    torch.testing.assert_close(warped["keypoint_coords"], lens.distort(ideal).view(2))
    torch.testing.assert_close(warped["distortions"], torch.tensor(lambdas))

    # scale == base * sqrt|det J_view|, and det J_D = 1 / det J_U at the observed point.
    pinhole_scale = _bare_dataset((0.0, 0.0), homography, source)[0]["scales"]
    observed = lens.distort(ideal)
    torch.testing.assert_close(
        warped["scales"],
        pinhole_scale / lens.det_jacobian_undistort(observed).abs().sqrt().view(()),
    )

    # The identity reference view never gets a lens.
    torch.testing.assert_close(identity["keypoint_coords"], source)
    torch.testing.assert_close(identity["distortions"], torch.zeros(2))


def test_collate_carries_the_lens():
    homography = _homography(seed=6, dtype=torch.float32)
    data = _bare_dataset((0.1, 0.0), homography, torch.tensor([200.0, 160.0]))

    batch = data.get_collate_func()([data[0], data[1]])

    assert batch["distortions"].shape == (2, 2)
    torch.testing.assert_close(batch["distortions"][0], torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(batch["distortions"][1], torch.zeros(2))


def test_collate_renders_the_warped_view_through_the_lens():
    """The ``in_memory=False`` path has to leave ``warp_perspective`` behind."""
    size = (32, 40)
    images = _ramp_image(*size)  # (boards, C, H, W), as HomographyData stores them
    homography = torch.tensor(
        [[1.05, 0.02, -1.0], [-0.01, 0.98, 1.5], [0.0, 0.0, 1.0]]
    )
    lambdas = (0.12, 0.01)
    data = _bare_dataset(
        lambdas, homography, torch.tensor([16.0, 12.0]), size=size, images=images
    )

    batch = data.get_collate_func()([data[0]])
    rendered = batch["images"][0]

    expected = render_view(
        images,
        homography.view(1, 3, 3),
        RadialDistortion(torch.tensor([lambdas]), size=size),
        size,
    ).squeeze(0)
    torch.testing.assert_close(rendered, expected)
    # Not a no-op: the stubbed warp_perspective would have handed the source back.
    assert not torch.allclose(rendered, images[0], atol=1e-3)


def test_disabled_lens_does_not_change_the_dataset_cache_key():
    """A pinhole run must land on the entry it already had on disk.

    `effective_params` hashes defaults, so without `_CACHE_KEY_OPTIONAL` merely adding
    the parameter would have invalidated every cached dataset in the repo — a full
    rebuild (Julia boards, SIFT, patch extraction) for data it cannot have changed.
    """
    historical = dataset_cache_key({"num_boards": 8, "patch_size": 64})

    assert dataset_cache_key(
        {"num_boards": 8, "patch_size": 64, "distortion_params": None}
    ) == historical
    assert dataset_cache_key(
        {"num_boards": 8, "patch_size": 64, "distortion_params": {}}
    ) == historical
    assert dataset_cache_key(
        {"num_boards": 8, "patch_size": 64, "distortion_params": {"lambda1": 0.1}}
    ) != historical


def test_frame_diagonal_normalizes_the_parameters():
    """ρ = 1 at the corner, so λ is dimensionless and portable across frame sizes."""
    lens = _lens()
    corner = torch.tensor([[float(SIZE[1] - 1), float(SIZE[0] - 1)]])
    rho = (corner - lens.center).norm() / lens.radius
    assert math.isclose(rho.item(), 1.0, rel_tol=1e-6)
