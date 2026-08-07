"""Viewing-angle band balancing for `.tracks` TRAINING batches (aef.data.track).

Real footage is heavily fronto-parallel-biased, so packing whole track groups feeds
the contrastive loss that same bias. `balance_view_angles` redraws the positive half
of every batch equally across obliquity bands. What has to hold for that to be worth
anything: the balance is per BATCH (SupCon only ever sees one batch at a time), no
patch is duplicated inside a batch (a duplicate is a distance-0 fake positive), every
batch still carries real positive pairs, and a band too small to fill its quota
under-fills rather than padding itself with repeats.
"""

import collections
import random

import numpy as np
import pytest

from aef.data.track import (
    _band_capacity,
    _band_edges,
    _band_pools,
    _band_target,
    _bands_of,
    _pack_band_balanced,
)


# ------------------------------------------------------------------ band binning --

def test_default_edges_are_ten_degree_bands_to_ninety():
    edges = _band_edges(None)
    assert edges[0] == 0.0 and edges[-1] == 90.0
    assert len(edges) - 1 == 9


@pytest.mark.parametrize("bad", [[10], [0, 10, 10], [30, 20], "0-10", [0, "x"]])
def test_invalid_edges_are_rejected(bad):
    with pytest.raises(ValueError):
        _band_edges(bad)


def test_bands_are_half_open_and_match_the_validation_grid():
    edges = _band_edges(None)
    bands = _bands_of(np.radians([0.0, 9.99, 10.0, 25.0, 89.99]), edges)
    assert bands.tolist() == [0, 0, 1, 2, 8]


def test_unknown_and_out_of_range_angles_are_unbanded():
    edges = _band_edges([0, 10, 20])
    bands = _bands_of(np.radians([np.nan, -1.0, 20.0, 5.0]), edges)
    assert bands.tolist() == [-1, -1, -1, 0]


@pytest.mark.parametrize(
    "spec, expected",
    [("min", 5), ("max", 500), ("median", 50), ("mean", 185), (123, 123)],
)
def test_band_target_statistics(spec, expected):
    assert _band_target([500, 50, 5], spec) == expected


def test_band_target_ignores_empty_bands():
    assert _band_target([0, 0, 7], "min") == 7


@pytest.mark.parametrize("bad", ["biggest", 0, -1, True])
def test_invalid_band_target_is_rejected(bad):
    with pytest.raises(ValueError):
        _band_target([10], bad)


# ------------------------------------------------------------------- pool layout --

def test_pools_index_observations_by_band_and_keep_the_anchor_apart():
    # Track 0: anchor (unbanded) + one 5 deg + one 45 deg observation.
    # Track 1: anchor + two 5 deg observations.
    pos_groups = [[0, 1, 2], [3, 4, 5]]
    bands = np.asarray([-1, 0, 4, -1, 0, 0])
    pools, groups = _band_pools(pos_groups, bands)
    assert sorted(pools) == [0, 4]
    assert sorted(pools[0]) == [(0, 1), (1, 4), (1, 5)]
    assert pools[4] == [(0, 2)]
    assert groups[0] == ([0, 1, 2], [0])     # the anchor is the preferred partner
    assert groups[1] == ([3, 4, 5], [3])


def test_group_with_no_banded_member_offers_no_draw():
    pools, groups = _band_pools([[0, 1]], np.asarray([-1, -1]))
    assert pools == {}
    assert groups[0] == ([0, 1], [0, 1])


# ---------------------------------------------------------------------- packing --

def _skewed(n_per_band, anchors=True):
    """Build (pools, groups, n_tracks) for bands of the given sizes.

    Each track is an anchor + exactly one observation in one band, so a band's pool
    size equals its track count and the band composition of a batch is readable
    straight off the patch indices.
    """
    pos_groups, bands, band_of_track = [], [], []
    for band, n in enumerate(n_per_band):
        for _ in range(n):
            base = len(bands)
            if anchors:
                pos_groups.append([base, base + 1])
                bands.extend([-1, band])
            else:
                pos_groups.append([base, base + 1])
                bands.extend([band, band])
            band_of_track.append(band)
    pools, groups = _band_pools(pos_groups, np.asarray(bands))
    return pools, groups, np.asarray(bands), band_of_track


def _band_histogram(batch, bands):
    """Band counts over a batch's positive half (confusers index past `bands`)."""
    return collections.Counter(
        int(bands[i]) for i in batch if i < len(bands) and bands[i] >= 0)


def test_every_batch_draws_equally_from_every_band():
    # 1000x more tracks near-frontal than edge-on: the unbalanced packer would put
    # ~1 oblique pair in a batch of 200; here every band contributes the same quota.
    pools, groups, bands, _ = _skewed([1000, 500, 100, 20])
    batches = _pack_band_balanced(pools, groups, confusers=list(range(6000, 6200)),
                                  batch_size=200, confuser_fraction=0.5,
                                  n_batches=5, rng=random.Random(0))
    assert len(batches) == 5
    quota = (200 - 100) // (2 * 4)
    for batch in batches:
        hist = _band_histogram(batch, bands)
        assert set(hist) == {0, 1, 2, 3}
        assert set(hist.values()) == {quota}


def test_batches_never_repeat_an_index():
    pools, groups, _bands, _ = _skewed([200, 50, 3])
    batches = _pack_band_balanced(pools, groups, confusers=list(range(9000, 9100)),
                                  batch_size=120, confuser_fraction=0.5,
                                  n_batches=8, rng=random.Random(1))
    for batch in batches:
        assert len(batch) == len(set(batch)), "a repeated index is a distance-0 positive"


def test_every_batch_carries_positive_pairs():
    pools, groups, _bands, _ = _skewed([100, 40, 5])
    batches = _pack_band_balanced(pools, groups, confusers=[], batch_size=60,
                                  confuser_fraction=0.5, n_batches=4,
                                  rng=random.Random(2))
    # Every drawn track contributes exactly its (observation, anchor) pair.
    member_of = {i: gi for gi, (members, _) in enumerate(groups) for i in members}
    for batch in batches:
        sizes = collections.Counter(member_of[i] for i in batch)
        assert sizes and set(sizes.values()) == {2}


def test_short_band_underfills_instead_of_duplicating():
    # Band 2 holds 3 tracks of one observation each but the quota is 10: it
    # contributes 3 in-band patches, never 10 copies of them.
    pools, groups, bands, _ = _skewed([100, 100, 3])
    batches = _pack_band_balanced(pools, groups, confusers=[], batch_size=120,
                                  confuser_fraction=0.5, n_batches=3,
                                  rng=random.Random(3))
    quota = (120 - 60) // (2 * 3)
    assert _band_capacity(pools[2], groups) == 3 < quota
    for batch in batches:
        hist = _band_histogram(batch, bands)
        assert hist[0] == hist[1] == quota
        assert hist[2] == 3
        assert len(batch) == len(set(batch))


@pytest.mark.parametrize(
    "n_obs, anchor, expected",
    [(1, True, 1), (2, True, 1), (3, True, 3), (4, True, 3), (6, True, 5),
     (1, False, 0), (2, False, 2), (3, False, 2), (4, False, 4)],
)
def test_band_capacity_counts_disjoint_pairs(n_obs, anchor, expected):
    # One track: the anchor buys a 1-patch pair, the rest couple in twos, and an odd
    # observation left over at the end has nothing to pair with.
    members = ([0] if anchor else []) + list(range(1, n_obs + 1))
    groups = [(members, [0] if anchor else [])]
    assert _band_capacity([(0, i) for i in range(1, n_obs + 1)], groups) == expected


def test_quota_counts_in_band_patches_not_pairs():
    # Band 1's tracks have no anchor left over, so its instances are
    # observation<->observation and put TWO in-band patches in each. Counting pairs
    # instead of patches would let it overshoot the anchor-paired band 0.
    pos_groups = [[2 * k, 2 * k + 1] for k in range(60)] + \
                 [[120 + 4 * k + j for j in range(4)] for k in range(30)]
    bands = [v for _ in range(60) for v in (-1, 0)] + [1] * 120
    pools, groups = _band_pools(pos_groups, np.asarray(bands))
    batches = _pack_band_balanced(pools, groups, confusers=[], batch_size=80,
                                  confuser_fraction=0.0, n_batches=3,
                                  rng=random.Random(8))
    quota = 80 // (2 * 2)
    for batch in batches:
        hist = _band_histogram(batch, np.asarray(bands))
        # The pair that fills a quota can put two in-band patches in, hence the +1.
        assert hist[0] == quota
        assert quota <= hist[1] <= quota + 1


def test_a_track_drawn_by_two_bands_never_reuses_its_anchor():
    # One track observed at 5 deg AND 45 deg (indices 1 and 2, anchor 0): both bands
    # can draw it, but only one instance can take the anchor — the other has to pair
    # the two observations or drop out. Either way index 0 appears at most once.
    pos_groups = [[0, 1, 2]] + [[3 + 2 * k, 4 + 2 * k] for k in range(20)]
    bands = [-1, 0, 3] + [v for _ in range(20) for v in (-1, 0)]
    pools, groups = _band_pools(pos_groups, np.asarray(bands))
    batches = _pack_band_balanced(pools, groups, confusers=[], batch_size=40,
                                  confuser_fraction=0.0, n_batches=6,
                                  rng=random.Random(4))
    for batch in batches:
        assert batch.count(0) <= 1
        assert len(batch) == len(set(batch))


def test_a_sparse_band_fills_its_quota_from_disjoint_pairs():
    # A band with ONE track but five observations of it: the anchor pair plus two
    # observation<->observation pairs are three disjoint positives, so the band is
    # not stuck at a single pair per batch.
    pos_groups = [[0, 1, 2, 3, 4, 5]] + [[6 + 2 * k, 7 + 2 * k] for k in range(40)]
    bands = [-1, 1, 1, 1, 1, 1] + [v for _ in range(40) for v in (-1, 0)]
    pools, groups = _band_pools(pos_groups, np.asarray(bands))
    batches = _pack_band_balanced(pools, groups, confusers=[], batch_size=40,
                                  confuser_fraction=0.0, n_batches=4,
                                  rng=random.Random(7))
    quota = 40 // (2 * 2)
    for batch in batches:
        assert len(batch) == len(set(batch))
        assert sum(1 for i in batch if i <= 5) == min(6, 2 * quota)


def test_confusers_fill_the_negative_half_without_repeating():
    pools, groups, _bands, _ = _skewed([50, 50])
    confusers = list(range(7000, 7080))
    batches = _pack_band_balanced(pools, groups, confusers=confusers, batch_size=100,
                                  confuser_fraction=0.5, n_batches=4,
                                  rng=random.Random(5))
    for batch in batches:
        used = [i for i in batch if i >= 7000]
        assert len(used) == 50 == len(set(used))


def test_tracked_only_groups_pair_two_observations():
    # No anchors: the partner has to be the track's other observation.
    pools, groups, _bands, _ = _skewed([30, 30], anchors=False)
    batches = _pack_band_balanced(pools, groups, confusers=[], batch_size=40,
                                  confuser_fraction=0.0, n_batches=3,
                                  rng=random.Random(6))
    member_of = {i: gi for gi, (members, _) in enumerate(groups) for i in members}
    for batch in batches:
        assert set(collections.Counter(member_of[i] for i in batch).values()) == {2}


def test_packing_is_seed_deterministic():
    pools, groups, _bands, _ = _skewed([80, 40, 10])
    def run(seed):
        return _pack_band_balanced(pools, groups, confusers=list(range(8000, 8050)),
                                   batch_size=80, confuser_fraction=0.5, n_batches=4,
                                   rng=random.Random(seed))
    assert run(0) == run(0)
    assert run(0) != run(1)


# -------------------------------------------------------- end to end on the dataset --

class _Group(dict):
    """dict + `.attrs` + `"a/b"` path lookup — the h5py API the loader uses."""

    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})

    def __getitem__(self, key):
        if "/" not in key:
            return super().__getitem__(key)
        node = self
        for part in key.split("/"):
            node = node[part]
        return node


class _FakeFile(dict):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _fake_tracks_file():
    """One board: 12 tracks, each an anchor + one observation, over three frames.

    Frame angles put 6 tracks in the [0, 10) band, 3 in [20, 30) and 3 in [50, 60) —
    the same lopsided shape real footage has, small enough to assert on exactly.
    """
    track_ids = list(range(1, 13))
    frame_ids = [0] * 6 + [1] * 3 + [2] * 3

    def patches(n):
        return np.zeros((n, 2, 2), dtype=np.float32)

    return _FakeFile(sequences={
        "media_a1": _Group(
            {
                "tracks": {
                    "patches": patches(12),
                    "affine_shapes": np.stack([np.eye(2, dtype=np.float32)] * 12),
                    "track_id": np.asarray(track_ids, dtype=np.int32),
                    "track_lengths": np.asarray([12], dtype=np.int32),
                    "frame_id": np.asarray(frame_ids, dtype=np.int32),
                    "homography_frame_ids": np.asarray([0, 1, 2], dtype=np.int32),
                    "view_angles": np.radians(np.asarray([5.0, 25.0, 55.0])),
                },
                "untracked": {"patches": patches(8)},
            },
            attrs={"is_anchor": 0},
        ),
        "a1": _Group(
            {
                "tracks": {
                    "patches": patches(12),
                    "affine_shapes": np.stack([np.eye(2, dtype=np.float32)] * 12),
                    "track_id": np.asarray(track_ids, dtype=np.int32),
                    "track_lengths": np.asarray([12], dtype=np.int32),
                }
            },
            attrs={"is_anchor": 1},
        ),
    })


@pytest.fixture(name="dataset")
def _dataset(monkeypatch):
    from aef.data import track as track_mod

    monkeypatch.setattr(track_mod.h5py, "File",
                        lambda *_a, **_k: _fake_tracks_file(), raising=False)
    return track_mod.BlobTrackData(
        h5_path="fake.tracks", include_untracked=True, unique_track_ids=False,
        balance_view_angles=True, view_angle_band_target="median",
    )


def test_dataset_carries_one_view_angle_per_patch(dataset):
    assert len(dataset.view_angles) == len(dataset.labels) == 12 + 12 + 8
    tracked = dataset.view_angles[:12]     # the tracked sequence loads first
    assert np.degrees(np.sort(tracked)).tolist() == [5.0] * 6 + [25.0] * 3 + [55.0] * 3
    # Anchors and confusers have no pose, hence no band.
    assert np.isnan(dataset.view_angles[12:]).all()


def test_balanced_sampler_gives_every_band_the_same_share(dataset):
    sampler = dataset.get_sampler(batch_size=12, confuser_fraction=0.0)
    bands = _bands_of(dataset.view_angles, dataset.view_angle_band_edges)
    batches = list(sampler)
    assert batches and len(batches) == len(sampler)
    for batch in batches:
        # 6/3/3 tracks in the three populated bands, but each contributes the same
        # quota of pairs — that is the whole point of the mechanism.
        assert set(_band_histogram(batch, bands).values()) == {2}
        assert len(batch) == len(set(batch))


def test_unbalanced_sampler_is_untouched(monkeypatch):
    from aef.data import track as track_mod

    monkeypatch.setattr(track_mod.h5py, "File",
                        lambda *_a, **_k: _fake_tracks_file(), raising=False)
    d = track_mod.BlobTrackData(h5_path="fake.tracks", include_untracked=True,
                                unique_track_ids=False)
    bands = _bands_of(d.view_angles, d.view_angle_band_edges)
    # Whole track groups, packed in file order: the near-frontal band dominates.
    counts = collections.Counter()
    for batch in d.get_sampler(batch_size=12, confuser_fraction=0.0):
        counts.update(_band_histogram(batch, bands))
    assert counts[0] == 6 and counts[2] == 3 and counts[5] == 3


def test_balancing_without_any_angles_fails_loudly(monkeypatch):
    from aef.data import track as track_mod

    def _no_angles(*_a, **_k):
        f = _fake_tracks_file()
        del f["sequences"]["media_a1"]["tracks"]["view_angles"]
        return f

    monkeypatch.setattr(track_mod.h5py, "File", _no_angles, raising=False)
    d = track_mod.BlobTrackData(h5_path="fake.tracks", unique_track_ids=False,
                                balance_view_angles=True)
    with pytest.raises(ValueError, match="viewing angle"):
        d.get_sampler(batch_size=8)


def test_track_config_enables_balancing_with_keys_the_dataset_accepts():
    # The training dataset params are splatted straight into the constructor, so a
    # renamed/typo'd key would only surface at run time — and `**_` would swallow it.
    import inspect
    import os

    from hydra import compose, initialize_config_dir

    from aef.data.track import BlobTrackData

    conf_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "conf"))
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="track_descriptor_logpolar",
                      overrides=["track_path=/tmp/does-not-need-to-exist.tracks"])

    params = cfg.training.dataset.params
    assert params.balance_view_angles is True
    named = set(inspect.signature(BlobTrackData).parameters) - {"_"}
    assert set(params) <= named, f"unknown dataset params: {set(params) - named}"


def test_empty_pools_pack_nothing():
    assert _pack_band_balanced({}, [], confusers=[1, 2], batch_size=10,
                               confuser_fraction=0.5, n_batches=3,
                               rng=random.Random(0)) == []
