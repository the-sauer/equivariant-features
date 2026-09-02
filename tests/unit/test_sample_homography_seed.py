import numpy as np
import torch

from aef.transforms.homography import sample_homography


SHAPE = (200, 200)


def test_same_seed_reproduces_the_draw():
    a = sample_homography(SHAPE, rng=np.random.default_rng(0))
    b = sample_homography(SHAPE, rng=np.random.default_rng(0))

    torch.testing.assert_close(a, b)


def test_different_seeds_give_different_warps():
    a = sample_homography(SHAPE, rng=np.random.default_rng(0))
    b = sample_homography(SHAPE, rng=np.random.default_rng(1))

    assert not torch.allclose(a, b)


def test_one_generator_advances_across_calls():
    """The generator must be threaded, not recreated per call.

    A fresh ``default_rng(seed)`` per call would hand every board and every view the
    identical homography — reproducible, and useless.
    """
    rng = np.random.default_rng(0)
    draws = [sample_homography(SHAPE, rng=rng) for _ in range(5)]

    for i in range(1, len(draws)):
        assert not torch.allclose(draws[0], draws[i])


def test_seeded_sequence_is_reproducible_as_a_whole():
    def sequence(seed):
        rng = np.random.default_rng(seed)
        return torch.stack([sample_homography(SHAPE, rng=rng) for _ in range(4)])

    torch.testing.assert_close(sequence(7), sequence(7))
    assert not torch.allclose(sequence(7), sequence(8))


def test_rng_none_uses_the_global_generator():
    """Default stays on numpy's global RNG, so behaviour is unchanged when unseeded —
    and it still responds to np.random.seed, which existing scripts rely on."""
    np.random.seed(123)
    a = sample_homography(SHAPE)
    np.random.seed(123)
    b = sample_homography(SHAPE)

    torch.testing.assert_close(a, b)


def test_generator_draw_is_independent_of_global_seed():
    """A passed generator must not be perturbed by global numpy state."""
    np.random.seed(1)
    a = sample_homography(SHAPE, rng=np.random.default_rng(42))
    np.random.seed(999)
    b = sample_homography(SHAPE, rng=np.random.default_rng(42))

    torch.testing.assert_close(a, b)


def test_seeding_works_with_fit_to_frame():
    """fit_to_frame post-composes a scale/translation but draws nothing itself; the
    seeded path must stay reproducible through it."""
    a = sample_homography(SHAPE, fit_to_frame=True, rng=np.random.default_rng(3))
    b = sample_homography(SHAPE, fit_to_frame=True, rng=np.random.default_rng(3))

    torch.testing.assert_close(a, b)
    assert torch.isfinite(a).all()
