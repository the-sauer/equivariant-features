"""Viewing-angle band filter on `.tracks` data (aef.data.track).

`.tracks` stores the detection geometry per FRAME (`view_angles` +
`homography_frame_ids`) and the patches per OBSERVATION (`frame_id`), so the filter
is only correct if that join is. These tests pin the join, the band bounds, and the
alignment of everything that rides along with the patches (ids, masks, affine shapes,
is_anchor) — a filter that silently mismatched them would still "work" and quietly
evaluate the wrong patches against the wrong labels.
"""

import numpy as np
import pytest

from aef.data.track import _load_all_sequences, _view_angle_bounds


class _Group(dict):
    """dict + `.attrs`, the only part of the h5py API the loader uses."""

    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})


def _tracked(name_patch_value, track_ids, frame_ids, hom_frame_ids, angles_deg):
    """A tracked sequence whose patches are constant-valued per observation."""
    n = len(track_ids)
    patches = np.stack([np.full((2, 2), v, dtype=np.float32) for v in name_patch_value])
    assert patches.shape[0] == n
    return _Group(
        {
            "tracks": {
                "patches": patches,
                "masks": np.stack([np.full((2, 2), v, dtype=np.float32)
                                   for v in name_patch_value]),
                "affine_shapes": np.stack([np.eye(2, dtype=np.float32) * v
                                           for v in name_patch_value]),
                "track_id": np.asarray(track_ids, dtype=np.int32),
                "track_lengths": np.asarray([n], dtype=np.int32),
                "frame_id": np.asarray(frame_ids, dtype=np.int32),
                "homography_frame_ids": np.asarray(hom_frame_ids, dtype=np.int32),
                "view_angles": np.radians(np.asarray(angles_deg, dtype=np.float64)),
            }
        },
        attrs={"is_anchor": 0},
    )


def _anchor(patch_values, track_ids):
    """A GT anchor sequence: no frame_id / view_angles, one observation per track."""
    n = len(track_ids)
    return _Group(
        {
            "tracks": {
                "patches": np.stack([np.full((2, 2), v, dtype=np.float32)
                                     for v in patch_values]),
                "masks": np.ones((n, 2, 2), dtype=np.float32),
                "affine_shapes": np.stack([np.eye(2, dtype=np.float32)] * n),
                "track_id": np.asarray(track_ids, dtype=np.int32),
                "track_lengths": np.asarray([n], dtype=np.int32),
            }
        },
        attrs={"is_anchor": 1},
    )


@pytest.fixture(name="tracks_file")
def _tracks_file():
    """One board (uid `a1`): 4 tracked observations over 3 frames + its anchors.

    Frame 7 -> 5 deg, frame 3 -> 25 deg, frame 5 -> 55 deg. The pose table is
    deliberately NOT in frame order, and observations are not grouped by frame, so a
    positional (rather than keyed) join would visibly pick the wrong angle.
    """
    return {
        "sequences": {
            "media_a1": _tracked(
                name_patch_value=[10, 20, 30, 40],
                track_ids=[1, 2, 1, 3],
                frame_ids=[7, 3, 5, 7],
                hom_frame_ids=[3, 7, 5],
                angles_deg=[25.0, 5.0, 55.0],
            ),
            "a1": _anchor(patch_values=[1, 2, 3], track_ids=[1, 2, 3]),
        }
    }


def _values(patches):
    return sorted(int(p[0, 0]) for p in patches)


def test_no_band_loads_everything(tracks_file):
    patches, _masks, ids, _lengths, _affine, is_anchor, _ang = _load_all_sequences(
        tracks_file, with_mask=True)
    assert _values(patches) == [1, 2, 3, 10, 20, 30, 40]
    assert len(ids) == 7
    assert is_anchor.sum() == 3


def test_band_keeps_only_in_band_observations_plus_anchors(tracks_file):
    # [0, 10) deg -> frame 7 only: observations 10 and 40; all 3 anchors ride along.
    patches, _masks, _ids, _lengths, _affine, is_anchor, _ang = _load_all_sequences(
        tracks_file, view_angle_range=[0, 10])
    assert _values(patches) == [1, 2, 3, 10, 40]
    assert is_anchor.sum() == 3


def test_band_boundaries_are_half_open(tracks_file):
    # 25 deg belongs to [20, 30) and not to [10, 20) or [30, 40).
    def tracked_values(band):
        patches, _m, _i, _l, _a, is_anchor, _ang = _load_all_sequences(
            tracks_file, view_angle_range=band)
        return sorted(int(p[0, 0]) for p, a in zip(patches, is_anchor) if not a)

    assert tracked_values([20, 30]) == [20]
    assert tracked_values([10, 20]) == []
    assert tracked_values([30, 40]) == []
    assert tracked_values([50, 90]) == [30]


def test_band_keeps_ids_masks_and_affine_aligned(tracks_file):
    patches, masks, ids, _lengths, affine, is_anchor, _ang = _load_all_sequences(
        tracks_file, with_mask=True, view_angle_range=[0, 10], unique_track_ids=False)
    tracked = ~is_anchor.astype(bool)
    values = patches[tracked][:, 0, 0]
    assert sorted(values.tolist()) == [10, 40]
    # masks / affine_shapes were built from the same per-observation value, and the
    # track ids of observations 10 and 40 are tracks 1 and 3.
    assert np.array_equal(masks[tracked][:, 0, 0], values)
    assert np.array_equal(affine[tracked][:, 0, 0], values)
    assert sorted(ids[tracked].tolist()) == [1, 3]


def test_dropping_anchors_leaves_only_the_band(tracks_file):
    patches, _m, _ids, _l, _a, is_anchor, _ang = _load_all_sequences(
        tracks_file, view_angle_range=[0, 10], view_angle_keep_anchors=False)
    assert _values(patches) == [10, 40]
    assert is_anchor.sum() == 0


def test_empty_band_yields_an_empty_dataset(tracks_file):
    patches, _m, ids, _l, _a, _is_anchor, _ang = _load_all_sequences(
        tracks_file, view_angle_range=[60, 70], view_angle_keep_anchors=False)
    assert patches.shape[0] == 0
    assert ids.shape[0] == 0


def test_observation_without_a_pose_entry_is_dropped(tracks_file):
    # Point one observation at a frame that is missing from the pose table: it must be
    # dropped, never silently bucketed into a neighbouring frame's angle.
    tracks_file["sequences"]["media_a1"]["tracks"]["frame_id"] = np.asarray(
        [7, 3, 5, 99], dtype=np.int32)
    patches, _m, _i, _l, _a, is_anchor, _ang = _load_all_sequences(
        tracks_file, view_angle_range=[0, 10], view_angle_keep_anchors=False)
    del is_anchor
    assert _values(patches) == [10]


def test_view_angles_ride_along_with_the_patches(tracks_file):
    # The per-patch angle is what the band balancer bins, so it has to follow the same
    # frame join as the filter: anchors NaN, tracked observations their frame's angle.
    patches, _m, _i, _l, _a, is_anchor, angles = _load_all_sequences(tracks_file)
    by_value = dict(zip((int(p[0, 0]) for p in patches), np.degrees(angles)))
    assert by_value[10] == pytest.approx(5.0)    # frame 7
    assert by_value[20] == pytest.approx(25.0)   # frame 3
    assert by_value[30] == pytest.approx(55.0)   # frame 5
    assert by_value[40] == pytest.approx(5.0)    # frame 7 again
    assert np.isnan(angles[is_anchor.astype(bool)]).all()


def test_view_angles_are_nan_without_a_pose_entry(tracks_file):
    tracks_file["sequences"]["media_a1"]["tracks"]["frame_id"] = np.asarray(
        [7, 3, 5, 99], dtype=np.int32)
    patches, _m, _i, _l, _a, _anchor, angles = _load_all_sequences(tracks_file)
    unmatched = [a for p, a in zip(patches, angles) if int(p[0, 0]) == 40]
    assert np.isnan(unmatched).all()


def test_view_angles_are_nan_for_files_without_the_export(tracks_file):
    # No band filter -> an old file still loads, it just has no angles to balance on.
    del tracks_file["sequences"]["media_a1"]["tracks"]["view_angles"]
    _p, _m, _i, _l, _a, _anchor, angles = _load_all_sequences(tracks_file)
    assert np.isnan(angles).all()


def test_missing_view_angles_raises(tracks_file):
    del tracks_file["sequences"]["media_a1"]["tracks"]["view_angles"]
    with pytest.raises(ValueError, match="view_angles"):
        _load_all_sequences(tracks_file, view_angle_range=[0, 10])


def test_missing_view_angles_is_fine_without_a_band(tracks_file):
    del tracks_file["sequences"]["media_a1"]["tracks"]["view_angles"]
    patches, _m, _i, _l, _a, _anchor, _ang = _load_all_sequences(tracks_file)
    assert patches.shape[0] == 7


@pytest.mark.parametrize("bad", [[10, 10], [30, 20], [0], "0-10"])
def test_invalid_bands_are_rejected(bad):
    with pytest.raises(ValueError):
        _view_angle_bounds(bad)


def test_bounds_convert_degrees_to_radians():
    assert _view_angle_bounds(None) is None
    lo, hi = _view_angle_bounds([0, 90])
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(np.pi / 2)
