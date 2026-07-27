"""Phase-keeping angular heads: relative phase and the low-order bispectrum.

Both exist to fix one thing the magnitude head cannot: ``|X_k|`` is blind to where each
ripple sits relative to the others, so structurally different angular profiles collapse
to the same descriptor. So each head is pinned on both counts — it must stay exactly
invariant to an angular roll (a rotation), AND it must separate two profiles that have
identical magnitude spectra and differ only in relative phase. A head that passes the
first test alone is just a more expensive ``AngularRFFTMag``.
"""

import os
import re

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aef.models.hardnet import AngularBispectrum, AngularRelPhase, HardNetLogPolar


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _two_bump_profile(offset_bins, length=16):
    """1-cycle + 2-cycle ripple, the second shifted by ``offset_bins``.

    Shifting only the 2-cycle component leaves every ``|X_k|`` untouched (a shift is a
    pure phase change) but changes ``phi_2 - 2*phi_1`` — the textbook case the magnitude
    head cannot see. Returns (1, 1, A, 1).
    """
    n = torch.arange(length, dtype=torch.float32)
    one = torch.cos(2 * torch.pi * n / length)
    two = torch.cos(2 * torch.pi * 2 * (n - offset_bins) / length)
    return (one + two).view(1, 1, length, 1)


def _roll_invariant(head, x, rolls=(1, 3, 8, 15), atol=1e-5):
    base = head(x)
    return all(torch.allclose(base, head(torch.roll(x, shifts=k, dims=-2)), atol=atol)
               for k in rolls)


def test_relphase_is_cyclic_shift_invariant():
    x = torch.randn(2, 4, 16, 8)
    assert _roll_invariant(AngularRelPhase(n_harmonics=5), x)


def test_bispectrum_is_cyclic_shift_invariant():
    x = torch.randn(2, 4, 16, 8)
    assert _roll_invariant(AngularBispectrum(n_harmonics=5), x)
    assert _roll_invariant(AngularBispectrum(n_harmonics=5, normalize=False), x, atol=1e-4)


def test_magnitudes_alone_cannot_separate_the_two_profiles():
    """The premise of both heads: the fft head sees these two profiles as identical."""
    a, b = _two_bump_profile(0), _two_bump_profile(2)
    spec_a = torch.fft.rfft(a, dim=-2).abs()
    spec_b = torch.fft.rfft(b, dim=-2).abs()
    assert torch.allclose(spec_a, spec_b, atol=1e-5)


def test_relphase_separates_what_magnitude_cannot():
    head = AngularRelPhase(n_harmonics=5)
    a, b = head(_two_bump_profile(0)), head(_two_bump_profile(2))
    assert (a - b).abs().max() > 0.1


def test_bispectrum_separates_what_magnitude_cannot():
    head = AngularBispectrum(n_harmonics=5)
    a, b = head(_two_bump_profile(0)), head(_two_bump_profile(2))
    assert (a - b).abs().max() > 0.1


def test_relphase_rows_are_magnitudes_plus_two_per_harmonic_above_one():
    x = torch.randn(1, 3, 16, 8)
    # F magnitudes + Re/Im for k = 2..F-1 (c_0, c_1 are real, so they add nothing).
    assert AngularRelPhase(n_harmonics=5)(x).shape == (1, 3, 5 + 2 * 3, 8)
    assert AngularRelPhase.n_rows(5) == 11
    # Degenerate: with < 3 harmonics there is no k >= 2, so it degrades to magnitudes.
    assert AngularRelPhase(n_harmonics=2)(x).shape == (1, 3, 2, 8)


def test_bispectrum_rows_follow_the_low_order_triples():
    x = torch.randn(1, 3, 16, 8)
    # F = 5 -> (1,1), (1,2), (1,3), (2,2): k1 <= k2 and k1 + k2 <= 4.
    assert AngularBispectrum.pairs(5) == [(1, 1), (1, 2), (1, 3), (2, 2)]
    assert AngularBispectrum(n_harmonics=5)(x).shape == (1, 3, 5 + 2 * 4, 8)
    assert AngularBispectrum.n_rows(5) == 13
    assert AngularBispectrum(n_harmonics=2)(x).shape == (1, 3, 2, 8)   # no valid triple


def test_normalized_bispectrum_stays_on_the_unit_circle():
    """Normalizing keeps the phase coupling only — bounded, unlike the cubic product."""
    x = torch.randn(2, 3, 16, 8) * 50.0
    out = AngularBispectrum(n_harmonics=5, normalize=True)(x)
    phase_rows = out[:, :, 5:, :]
    assert phase_rows.abs().max() <= 1.0 + 1e-4


def test_relphase_reference_degeneracy_stays_finite():
    """|X_1| ~ 0 is the head's known weak spot; it must still produce finite numbers."""
    n = torch.arange(16, dtype=torch.float32)
    x = torch.cos(2 * torch.pi * 2 * n / 16).view(1, 1, 16, 1)   # pure 2-cycle: X_1 = 0
    assert torch.isfinite(AngularRelPhase(n_harmonics=5)(x)).all()


def test_heads_produce_a_128d_descriptor_through_the_net():
    for head in ("relphase", "bispectrum"):
        model = HardNetLogPolar(patch_size=64, head=head, n_harmonics=5).eval()
        out = model(torch.randn(2, 1, 64, 64))
        assert isinstance(out, torch.Tensor) and out.shape == (2, 128), head


def test_phase_heads_compose_with_the_learned_mask():
    model = HardNetLogPolar(patch_size=64, head="bispectrum", n_harmonics=5,
                            learned_mask=True).eval()
    d, m_pred = model(torch.rand(4, 1, 64, 64), mask=torch.ones(4, 1, 64, 64),
                      is_anchor=torch.tensor([True, False, True, False]))
    assert d.shape == (4, 128) and m_pred.shape[:2] == (4, 1)


def _net_extra(script, net):
    """The launcher's hydra overrides for one network, as a list."""
    with open(os.path.join(REPO, script), encoding="utf-8") as handle:
        text = handle.read()
    # Scope to the NET_EXTRA block: other tables (SCALES, NET_MEM, ...) are keyed by the
    # same network names, and the first match would be the wrong one.
    block = re.search(r"declare -A NET_EXTRA=\((.*?)^\)", text, re.M | re.S)
    assert block, f"no NET_EXTRA block in {script}"
    match = re.search(rf'^\s*\[{re.escape(net)}\]="([^"]*)"\s*$', block[1], re.M)
    assert match, f"{net} has no NET_EXTRA entry in {script}"
    return match[1].split()


@pytest.mark.parametrize("net", ["logpolar_relphase", "logpolar_bispectrum"])
def test_launcher_overrides_build_the_intended_head(net):
    """The matrix scripts pass the head as a raw override string, so a typo there is
    invisible until a job dies on the cluster. Compose the real config with the real
    override string and build the model."""
    overrides = ["scale=96"] + _net_extra("launch_training_matrix.sh", net)
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(REPO, "src", "conf")):
        cfg = compose(config_name="blob_descriptor_logpolar", overrides=overrides)
    assert cfg.model.name == "HardNetLogPolar"
    params = OmegaConf.to_container(cfg.model.params, resolve=True)
    assert params["head"] == net.rsplit("_", 1)[1]
    model = HardNetLogPolar(**params).eval()
    size = params["patch_size"]
    out = model(torch.randn(2, 1, size, size))
    assert out.shape == (2, 128)
    # The head's angular rows must match what the final conv was sized for.
    rows = (AngularRelPhase if params["head"] == "relphase" else AngularBispectrum)
    assert model.head[1].weight.shape[2] == rows.n_rows(params["n_harmonics"])


def test_track_launcher_offers_the_same_heads():
    for net in ("logpolar_relphase", "logpolar_bispectrum"):
        extra = _net_extra("launch_track_matrix.sh", net)
        assert f"++model.params.head={net.rsplit('_', 1)[1]}" in extra
        assert "model.name=HardNetLogPolar" in extra


def test_net_stays_rotation_invariant_with_a_phase_head():
    """End-to-end: an angular roll of the patch must barely move the descriptor."""
    torch.manual_seed(0)
    patches = torch.randn(2, 1, 64, 64)
    for head in ("relphase", "bispectrum"):
        model = HardNetLogPolar(patch_size=64, head=head, n_harmonics=5).eval()
        with torch.no_grad():
            base = model(patches)
            rolled = model(torch.roll(patches, shifts=16, dims=-2))
        drift = (base - rolled).norm(dim=-1) / base.norm(dim=-1)
        assert drift.max() < 1e-3, f"{head} drifted {drift.max():.2e} under a 90 deg roll"
