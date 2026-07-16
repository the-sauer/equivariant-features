import torch

from aef.data.homography import blob_normalizations
from aef.geometry import linearize_homography


def _random_homographies(n, seed=0, perspective=1e-4):
    """Homographies with a genuine perspective component, so a Jacobian evaluated
    at the wrong point is actually a different matrix."""
    g = torch.Generator().manual_seed(seed)
    h = torch.eye(3).repeat(n, 1, 1)
    h[:, :2, :2] += 0.4 * (torch.rand((n, 2, 2), generator=g) - 0.5)
    h[:, :2, 2] = 30.0 * (torch.rand((n, 2), generator=g) - 0.5)
    h[:, 2, :2] = perspective * (torch.rand((n, 2), generator=g) - 0.5)
    return h


def _source_points(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    pts = 200.0 * torch.rand((n, 2), generator=g) + 50.0
    return torch.cat([pts, torch.ones((n, 1))], dim=-1)


def _warp(homographies, source):
    projected = torch.einsum("nij,nj->ni", homographies, source)
    return projected[:, :2] / projected[:, 2:3]


def test_blob_normalizations_is_shape_only():
    """det == 1: the factor carries shape/orientation but never overall size.

    Size normalization is `scales`' job (it already carries sqrt(det J)); a factor
    that also removed the size would take it out twice.
    """
    n = 16
    homographies = _random_homographies(n)
    source = _source_points(n)
    coords = _warp(homographies, source)

    normalization = blob_normalizations(homographies, coords, coords.device)

    torch.testing.assert_close(
        torch.linalg.det(normalization), torch.ones(n), rtol=1e-4, atol=1e-4
    )


def test_blob_normalizations_isotropizes_with_residual_scaled_rotation():
    """N @ J_H(p) must be exactly sqrt(det J) times a rotation.

    That is the whole contract: it kills the anisotropy (so an elliptical blob maps
    to a circular one), leaves a rotation (intentional — the deployed detector yields
    arbitrary orientations), and leaves the size at exactly the sqrt(det J) that the
    extractors then divide out via `scales`.

    Catches both historical bugs: a non-det-1 factor makes the residual size 1
    instead of sqrt(det J), and a Jacobian evaluated at the warped point makes the
    composition not a scaled rotation at all.
    """
    n = 16
    homographies = _random_homographies(n)
    source = _source_points(n)
    coords = _warp(homographies, source)

    normalization = blob_normalizations(homographies, coords, coords.device)
    jacobian = linearize_homography(homographies, coords=source)

    composed = normalization @ jacobian
    det_j = torch.linalg.det(jacobian).abs()

    # composed @ composed.T == det(J) * I  <=>  composed is sqrt(det J) * rotation
    gram = composed @ composed.transpose(-1, -2)
    expected = det_j.view(-1, 1, 1) * torch.eye(2).expand(n, -1, -1)
    torch.testing.assert_close(gram, expected, rtol=1e-3, atol=1e-3)


def test_blob_normalizations_identity_homography_is_identity():
    n = 4
    homographies = torch.eye(3).repeat(n, 1, 1)
    coords = _source_points(n)[:, :2]

    normalization = blob_normalizations(homographies, coords, coords.device)

    torch.testing.assert_close(
        normalization, torch.eye(2).expand(n, -1, -1), rtol=1e-5, atol=1e-5
    )


def test_blob_normalizations_never_mirrors():
    """The factor must not carry a reflection: a mirrored patch cannot be undone by
    any rotation (nor by log-polar angular pooling), so it would never match its
    own identity view."""
    n = 32
    homographies = _random_homographies(n, seed=7)
    source = _source_points(n, seed=8)
    coords = _warp(homographies, source)

    normalization = blob_normalizations(homographies, coords, coords.device)

    assert bool((torch.linalg.det(normalization) > 0).all())
