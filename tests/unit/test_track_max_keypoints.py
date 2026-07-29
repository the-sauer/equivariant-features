"""`max_keypoints` on `.tracks` data (aef.data.track).

The per-viewing-angle-band validation config caps every band so that bands holding
wildly different numbers of observations (real footage is fronto-parallel-biased)
still evaluate comparably sized splits. What has to hold for that cap to mean
anything: positive track groups are kept WHOLE (a group trimmed to one member would
silently become a confuser, changing the very class balance the cap equalizes), the
confuser budget covers singletons AND untracked patches, and the selection is
deterministic so a band's members do not drift between runs.
"""

import numpy as np
import pytest

from aef.data.track import _cap_tracked, _untracked_budget


def _labels(*groups):
    """Flat label array from group sizes, e.g. _labels(3, 1, 2) -> [0,0,0,1,2,2]."""
    return np.asarray([i for i, n in enumerate(groups) for _ in range(n)], dtype=np.int64)


def _sizes(labels, keep):
    """Group sizes among the kept indices, as a sorted list."""
    _, counts = np.unique(labels[keep], return_counts=True)
    return sorted(counts.tolist())


def test_cap_keeps_positive_groups_whole():
    labels = _labels(*([3] * 10))   # 10 tracks x 3 patches
    cap = _cap_tracked(labels, max_keypoints=7)
    # 7 // 3 = 2 whole groups fit; the third would overflow.
    assert cap.n_pos == 6
    assert _sizes(labels, cap.keep) == [3, 3]
    assert cap.n_pos_available == 30


def test_cap_budget_is_per_kind_not_shared():
    labels = _labels(*([2] * 10 + [1] * 10))   # 10 pairs + 10 singletons
    cap = _cap_tracked(labels, max_keypoints=4)
    assert cap.n_pos == 4               # 2 pairs
    assert cap.n_singletons == 4        # singletons get their own budget
    assert len(cap.keep) == 8
    assert cap.n_singletons_available == 10


def test_cap_below_budget_keeps_everything():
    labels = _labels(2, 2, 1)
    cap = _cap_tracked(labels, max_keypoints=100)
    assert np.array_equal(cap.keep, np.arange(len(labels)))
    assert (cap.n_pos, cap.n_singletons) == (4, 1)


def test_cap_indices_are_sorted_and_unique():
    labels = _labels(*([2] * 50 + [1] * 50))
    keep = _cap_tracked(labels, max_keypoints=20, seed=3).keep
    assert np.array_equal(keep, np.unique(keep))   # sorted, no duplicates
    assert keep.dtype == np.int64


def test_cap_is_deterministic_and_seed_dependent():
    labels = _labels(*([2] * 100))
    a = _cap_tracked(labels, max_keypoints=20, seed=0).keep
    b = _cap_tracked(labels, max_keypoints=20, seed=0).keep
    c = _cap_tracked(labels, max_keypoints=20, seed=1).keep
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cap_on_empty_split():
    cap = _cap_tracked(np.zeros(0, dtype=np.int64), max_keypoints=10)
    assert cap.keep.size == 0
    assert (cap.n_pos, cap.n_singletons) == (0, 0)


@pytest.mark.parametrize(
    "n_tracked, ratio, n_singletons, cap, expected",
    [
        (100, 1.0, 0, None, 100),     # uncapped: unchanged ratio behaviour
        (100, 2.0, 0, None, 200),
        (100, 1.0, 30, 50, 20),       # singletons already spent 30 of the 50 budget
        (100, 1.0, 60, 50, 0),        # budget exhausted by singletons
        (10, 1.0, 0, 50, 10),         # ratio still binds when it is the smaller of the two
    ],
)
def test_untracked_budget(n_tracked, ratio, n_singletons, cap, expected):
    assert _untracked_budget(n_tracked, ratio, n_singletons, cap) == expected
