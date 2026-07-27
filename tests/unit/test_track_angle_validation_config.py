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


@pytest.fixture(name="specs")
def _specs(monkeypatch):
    """(label, dataset_cfg, loss) per band — datasets are not actually constructed."""
    monkeypatch.setattr(aef_data, "get_dataset", lambda dataset_cfg: dataset_cfg)
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(
            config_name="track_descriptor_logpolar_angles",
            overrides=[f"track_path={TRACK_PATH}"],
        )
    return get_validation_specs(cfg)


def test_one_spec_per_band(specs):
    assert [label for label, _cfg, _loss in specs] == [
        "all", "deg00_10", "deg10_20", "deg20_30", "deg30_40", "deg40_50", "deg50_90",
    ]


def test_bands_are_contiguous_ten_degree_windows(specs):
    bands = [cfg.params.view_angle_range for label, cfg, _loss in specs if label != "all"]
    bands = [list(b) for b in bands]
    assert bands == [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 90]]
    # Half-open windows: each band starts where the previous one ended, no gaps.
    assert all(prev[1] == nxt[0] for prev, nxt in zip(bands, bands[1:]))


def test_unfiltered_reference_band_has_no_filter(specs):
    label, cfg, _loss = specs[0]
    assert label == "all"
    assert cfg.params.view_angle_range is None


def test_bands_inherit_the_leafs_dataset_settings(specs):
    for label, cfg, _loss in specs:
        assert cfg.name == "BlobTrackData", label
        # Resolved, not a dangling ${track_path}.
        assert cfg.params.h5_path == TRACK_PATH, label
        # patch_type comes from the log-polar leaf, the boards from the base config.
        assert cfg.params.patch_type == "logpolar", label
        assert "25d" in list(cfg.params.sequences), label
        assert cfg.params.include_untracked is True, label


def test_only_the_unfiltered_band_drives_checkpoint_selection(specs):
    weights = {label: [loss.get("weight", 1) for loss in losses] for label, _cfg, losses in specs}
    assert weights["all"] == [1]
    assert all(w == [0] for label, w in weights.items() if label != "all")


def test_every_band_reports_fpr95(specs):
    for label, _cfg, losses in specs:
        assert [loss.name for loss in losses] == ["FPR95"], label
