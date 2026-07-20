import torch

from aef.data.homography import extract_logpolar_patches


def _args(n=2, h=128, w=128):
    imgs = torch.rand(n, 1, h, w)
    coords = torch.tensor([[w / 2, h / 2]]).repeat(n, 1)
    scales = torch.full((n,), 2.0)   # outer radius = 16*2 = 32 px, well inside a 128px frame
    homs = torch.eye(3).repeat(n, 1, 1)
    return imgs, homs, coords, scales


def test_mask_imgs_is_sampled_as_validity_not_oob():
    """With a board-coverage image supplied, the validity is sampled from it — so an
    all-invalid coverage yields ~0 even though every sample is in-bounds (oob would 1)."""
    imgs, homs, coords, scales = _args()
    zero_cov = torch.zeros(2, 1, 128, 128)
    _, valid = extract_logpolar_patches(
        imgs, homs, coords, scales, patch_size=32, inner_factor=2.0, outer_factor=16.0,
        supersample=1, return_mask=True, mask_imgs=zero_cov,
    )
    assert valid.shape == (2, 1, 32, 32)
    assert valid.max().item() < 1e-4                      # off-board everywhere


def test_mask_imgs_none_falls_back_to_in_bounds():
    """No coverage image -> ~oob fallback: fully in-bounds patch is valid everywhere."""
    imgs, homs, coords, scales = _args()
    _, valid = extract_logpolar_patches(
        imgs, homs, coords, scales, patch_size=32, inner_factor=2.0, outer_factor=16.0,
        supersample=1, return_mask=True,
    )
    assert valid.min().item() > 0.99


def test_mask_imgs_half_coverage_splits_validity():
    """A coverage image valid on the left half only must produce both valid and
    invalid cells (the annulus straddles the boundary)."""
    imgs, homs, coords, scales = _args()
    cov = torch.zeros(2, 1, 128, 128)
    cov[..., : 128 // 2] = 1.0                            # left half on-board
    _, valid = extract_logpolar_patches(
        imgs, homs, coords, scales, patch_size=32, inner_factor=2.0, outer_factor=16.0,
        supersample=1, return_mask=True, mask_imgs=cov,
    )
    assert valid.max().item() > 0.9 and valid.min().item() < 0.1
