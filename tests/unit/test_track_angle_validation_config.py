"""The per-viewing-angle-band validation config (`conf/track_angle_validation.yaml`).

The bands are built by interpolating the leaf's own `validation.dataset` into
`validation.shared_params` and deep-merging each band's `params` on top. Two things
can silently break that: Hydra composes in struct mode (a band adding the new key
`view_angle_range` must not be rejected), and the interpolation must resolve against
the real config root, or every band would inherit an unresolved `${track_path}`. So
compose the config for real rather than reading the YAML.
"""

import os

import pytest
from hydra import compose, initialize_config_dir

import aef.data as aef_data
from aef.data import get_validation_specs


CONF_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "conf")
)
TRACK_PATH = "/tmp/does-not-need-to-exist.tracks"


@pytest.fixture(name="composed")
def _composed():
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        return compose(
            config_name="track_descriptor_logpolar_angles",
            overrides=[f"track_path={TRACK_PATH}"],
        )


@pytest.fixture(name="specs")
def _specs(monkeypatch, composed):
    """(label, dataset_cfg, loss) per band — datasets are not actually constructed."""
    monkeypatch.setattr(aef_data, "get_dataset", lambda dataset_cfg: dataset_cfg)
    return get_validation_specs(composed)


def _bands(specs):
    """(label, [lo, hi]) for the angle-filtered bands, in config order."""
    return [(label, list(cfg.params.view_angle_range))
            for label, cfg, _loss in specs if label != "all"]


def test_the_unfiltered_reference_comes_first(specs):
    assert specs[0][0] == "all"
    assert len(specs) > 1


def test_band_labels_match_their_window(specs):
    # The label is what shows up as `FPR95@<label>`; a mislabelled band would report
    # the wrong viewpoint. Derived from the range, so widening the bands stays green.
    for label, (lo, hi) in _bands(specs):
        assert label == f"deg{int(lo):02d}_{int(hi):02d}", (label, lo, hi)


def test_bands_tile_the_whole_angle_range(specs):
    bands = [b for _label, b in _bands(specs)]
    assert bands[0][0] == 0 and bands[-1][1] == 90
    # Half-open windows: each band starts where the previous one ended, no gaps.
    assert all(prev[1] == nxt[0] for prev, nxt in zip(bands, bands[1:]))


def test_every_band_shares_one_keypoint_cap(specs):
    # Bands differ by an order of magnitude in population (footage is fronto-parallel-
    # biased), so a per-band cap is what makes their FPR95 comparable — it only works
    # if it is the SAME cap everywhere (BlobTrackData spends it per kind: positives
    # and confusers each get `max_keypoints`).
    caps = {label: cfg.params.get("max_keypoints")
            for label, cfg, _loss in specs if label != "all"}
    assert all(c is not None for c in caps.values()), caps
    assert len(set(caps.values())) == 1, caps


def test_unfiltered_reference_band_has_no_filter(specs):
    label, cfg, _loss = specs[0]
    assert label == "all"
    assert cfg.params.view_angle_range is None


def test_bands_inherit_the_leafs_dataset_settings(specs, composed):
    boards = list(composed.validation.dataset.params.sequences)
    assert boards, "the leaf pins a board list; the bands must inherit it"
    for label, cfg, _loss in specs:
        assert cfg.name == "BlobTrackData", label
        # Resolved, not a dangling ${track_path}.
        assert cfg.params.h5_path == TRACK_PATH, label
        # patch_type comes from the log-polar leaf, the boards from the base config.
        assert cfg.params.patch_type == "logpolar", label
        assert list(cfg.params.sequences) == boards, label
        assert cfg.params.include_untracked is True, label


def test_only_the_unfiltered_band_drives_checkpoint_selection(specs):
    weights = {label: [loss.get("weight", 1) for loss in losses] for label, _cfg, losses in specs}
    assert weights["all"] == [1]
    assert all(w == [0] for label, w in weights.items() if label != "all")


def test_every_band_reports_fpr95(specs):
    for label, _cfg, losses in specs:
        assert [loss.name for loss in losses] == ["FPR95"], label
