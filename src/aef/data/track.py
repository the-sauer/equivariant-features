# Affine Equivariant Features, the main implementation of my master thesis.
# Copyright (C) 2026 Hendrik Sauer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import collections
import math
import random
import re

import h5py
import numpy as np
import torch


class _ListBatchSampler:
    """A precomputed, deterministic batch sampler: yields the same list of index
    lists every epoch (so validation FPR is comparable across epochs)."""

    is_batch_sampler = True   # marks this for the batch_sampler= DataLoader path

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _ReshufflingBatchSampler:
    """Balanced batch sampler that repacks with a fresh shuffle every epoch.

    Used for TRAINING: positive track groups and confusers are fixed, but their
    order (and thus the batch composition) is re-randomized each epoch for training
    diversity, while ``__len__`` stays stable for the progress bar.
    """

    is_batch_sampler = True

    def __init__(self, pack, pos_groups, confusers, batch_size, confuser_fraction, seed):
        self._pack = pack
        self._pos = pos_groups
        self._conf = confusers
        self._bs = batch_size
        self._cf = confuser_fraction
        self._seed = seed
        self._epoch = 0
        self._len = len(pack(pos_groups, confusers, batch_size, confuser_fraction,
                             random.Random(seed)))

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)
        self._epoch += 1
        return iter(self._pack(self._pos, self._conf, self._bs, self._cf, rng))

    def __len__(self):
        return self._len


# Single-mode `.tracks` layout (BlobBoards >= single-type/single-scale format):
# one file holds exactly ONE patch type at ONE scale, so `tracks/patches` is a
# single (N, P, P) dataset (no per-mode `patches/<type>` group). Each patch has a
# companion board-in-frame mask under `tracks/masks`, and every sequence carries
# an `is_anchor` group attribute (1 = GT anchor sequence, 0 = tracked). The single
# patch type / scale are recorded as attributes on the `tracks` group.
# Blob-index field width for the collision-free remap (see `_load_all_sequences`).
# Boards observed with up to ~1330 blobs (< 2**16); 16 bits leaves room to grow while
# keeping the packed id in 32 bits (board_id in the high bits, blob index in the low).
_BLOB_BITS = 16


def _uid_of(name):
    # Sequence names are either `<media>_<uid>` (tracked) or `<uid>` (anchor); return
    # the trailing hex board uid (works for both forms), or None if it doesn't match.
    m = re.fullmatch(r"(?:.*_)?([A-Fa-f0-9]+)", name)
    return m[1] if m else None


def _select_sequences(seqs, sequences):
    # The `sequences` filter is a set of board uids; keep sequences whose uid matches.
    if sequences is None:
        return list(seqs.keys())
    return [s for s in seqs.keys() if _uid_of(s) in sequences]


def _view_angle_bounds(view_angle_range):
    """Validate a ``[lo, hi]`` viewing-angle band in DEGREES -> bounds in radians.

    ``None`` (no band) passes through as ``None``. Degrees in the config, radians in
    the file: `.tracks` stores `view_angles` in radians, but a band is something you
    reason about in degrees, so that is the config unit.
    """
    if view_angle_range is None:
        return None
    try:
        lo, hi = (float(v) for v in view_angle_range)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"view_angle_range must be a [lo_deg, hi_deg] pair, got {view_angle_range!r}"
        ) from e
    if not lo < hi:
        raise ValueError(f"view_angle_range must satisfy lo < hi, got [{lo}, {hi}]")
    return math.radians(lo), math.radians(hi)


def _read_rows(ds, sel):
    """Read all rows of an HDF5 dataset, or just ``sel`` (an increasing index array)."""
    return ds[:] if sel is None else ds[sel]


def _observation_view_angles(g, name):
    """Per-observation viewing angle (radians) for a tracked sequence.

    `.tracks` stores the detection geometry **per frame, not per blob**:
    ``view_angles`` and ``homography_frame_ids`` have one entry per frame the board
    was detected in, while ``frame_id`` gives each observation's source frame. This
    joins the two, returning ``(angles, matched)`` — both per observation, with
    ``matched`` False wherever an observation's frame is absent from the pose table
    (should not happen; such observations are dropped rather than mis-binned).
    """
    missing = [k for k in ("frame_id", "homography_frame_ids", "view_angles") if k not in g]
    if missing:
        raise ValueError(
            f"sequence {name!r} has no {', '.join(missing)} — it predates the viewing-angle "
            "export in BlobBoards' build_tracks_for_sequences. Rebuild the `.tracks` file "
            "or drop `view_angle_range`."
        )
    fids = np.asarray(g["frame_id"][:]).astype(np.int64)
    hfids = np.asarray(g["homography_frame_ids"][:]).astype(np.int64)
    angles = np.asarray(g["view_angles"][:]).astype(np.float64)
    if hfids.size == 0:
        return np.zeros(fids.shape, dtype=np.float64), np.zeros(fids.shape, dtype=bool)
    # Join frame_id -> pose table without assuming the table is sorted.
    order = np.argsort(hfids)
    pos = np.clip(np.searchsorted(hfids[order], fids), 0, hfids.size - 1)
    idx = order[pos]
    matched = hfids[idx] == fids
    return angles[idx], matched


def _load_all_sequences(f, sequences=None, with_mask=False, with_affine=True,
                        unique_track_ids=True, view_angle_range=None,
                        view_angle_keep_anchors=True):
    """Load and concatenate track data from per-sequence HDF5 groups.

    Returns (patches, masks, track_ids, track_lengths, affine_shapes, is_anchor).
    `masks` is None unless `with_mask`; `affine_shapes` None unless `with_affine`.
    `is_anchor` is a per-patch uint8 array (broadcast from the per-sequence flag).
    `track_lengths` is the sequence's stored (PRE-filter) histogram and is not
    recomputed when a viewing-angle band drops observations; nothing downstream uses
    it (group structure is rebuilt from the labels in ``_grouped_indices``).

    ``view_angle_range`` — ``[lo_deg, hi_deg]``, keeps only observations whose frame
    was viewed at an obliquity in ``[lo, hi)`` (0 = fronto-parallel, 90 = edge-on).
    Anchor sequences carry no pose (they are rendered off the board image, i.e.
    fronto-parallel by construction) and are kept whole unless
    ``view_angle_keep_anchors`` is False — keeping them is what preserves the
    anchor<->tracked positive pairs that make a narrow band evaluable at all.

    ``unique_track_ids`` (default True) remaps track ids to be COLLISION-FREE across
    boards. The stored ids are ``board_hash + blob_index`` packed into one ~16-bit
    field, so different boards whose hashes are closer than their blob count share ids
    — a measured ~34% of ids span >1 board, pairing unrelated blobs as false positives.
    Here each board (identified by its sequence uid) gets a dense index, and the id is
    repacked as ``(board_index << _BLOB_BITS) | (raw_id - board_base)`` — i.e. a clean
    ``board_id | blob_index``. This preserves within-board matches (a board's anchor and
    tracked media share the same raw ids, hence the same blob_index) while making every
    board's id range disjoint. Assumes distinct uid => distinct board (holds for the
    current files; the durable fix is a full-hash registry at export time).
    """
    seqs = f["sequences"]
    seq_names = _select_sequences(seqs, sequences)

    board_index, board_base = {}, {}
    if unique_track_ids:
        board_uids = sorted({u for u in map(_uid_of, seq_names) if u is not None})
        board_index = {u: i for i, u in enumerate(board_uids)}
        # Board base = min raw id over ALL of a board's sequences, so anchor and tracked
        # media of the same board map to the same blob_index.
        for name in seq_names:
            u = _uid_of(name)
            if u is None:
                continue
            tid = seqs[name]["tracks"]["track_id"]
            if tid.shape[0]:
                mn = int(tid[:].min())
                board_base[u] = min(board_base.get(u, mn), mn)

    def remap(name, tid):
        u = _uid_of(name)
        if not unique_track_ids or u is None:
            return tid.astype(np.int64)
        blob_index = tid.astype(np.int64) - board_base[u]
        if blob_index.size and (blob_index.min() < 0 or blob_index.max() >= (1 << _BLOB_BITS)):
            raise ValueError(
                f"blob_index out of range for board {u!r} (max {int(blob_index.max())}); "
                f"raise _BLOB_BITS above {_BLOB_BITS}"
            )
        return (np.int64(board_index[u]) << _BLOB_BITS) | blob_index

    bounds = _view_angle_bounds(view_angle_range)

    all_patches, all_masks = [], []
    all_track_ids, all_track_lengths = [], []
    all_affine_shapes, all_is_anchor = [], []

    for name in seq_names:
        sg = seqs[name]
        g = sg["tracks"]
        is_anchor = int(sg.attrs["is_anchor"]) if "is_anchor" in sg.attrs else 0

        # Viewing-angle band: decide WHICH observations to read before reading them,
        # so a narrow band does not pull a whole (large) sequence through memory.
        sel = None
        if bounds is not None:
            if is_anchor:
                if not view_angle_keep_anchors:
                    continue
            else:
                angles, matched = _observation_view_angles(g, name)
                lo, hi = bounds
                sel = np.flatnonzero(matched & (angles >= lo) & (angles < hi))
                if sel.size == 0:
                    continue
                if sel.size == g["patches"].shape[0]:
                    sel = None            # whole sequence: plain slice reads faster

        patches = _read_rows(g["patches"], sel)
        n = patches.shape[0]
        all_patches.append(patches)
        all_track_ids.append(remap(name, _read_rows(g["track_id"], sel)))
        all_track_lengths.append(g["track_lengths"][:])
        if with_mask:
            all_masks.append(_read_rows(g["masks"], sel) if "masks" in g
                             else np.ones_like(patches))
        if with_affine:
            all_affine_shapes.append(_read_rows(g["affine_shapes"], sel))
        all_is_anchor.append(np.full(n, is_anchor, dtype=np.uint8))

    def cat(parts, empty):
        return np.concatenate(parts) if parts else empty

    patches = cat(all_patches, np.zeros((0, 64, 64), dtype=np.float32))
    track_ids = cat(all_track_ids, np.zeros(0, dtype=np.int64))
    track_lengths = cat(all_track_lengths, np.zeros(0, dtype=np.int32))
    masks = cat(all_masks, np.zeros((0, 64, 64), dtype=np.float32)) if with_mask else None
    affine_shapes = cat(all_affine_shapes, np.zeros((0, 2, 2), dtype=np.float32)) if with_affine else None
    is_anchor = cat(all_is_anchor, np.zeros(0, dtype=np.uint8))

    return patches, masks, track_ids, track_lengths, affine_shapes, is_anchor


def load_untracked_patches(f, sequences=None, with_mask=False):
    """Load confuser (untracked) patches, optionally with their in-frame masks.

    Returns `patches` (or `(patches, masks)` when `with_mask`).
    """
    seqs = f["sequences"]
    seq_names = _select_sequences(seqs, sequences)
    parts, mask_parts = [], []
    for name in seq_names:
        sg = seqs[name]
        if "untracked" in sg:
            parts.append(sg["untracked/patches"][:])
            if with_mask:
                ug = sg["untracked"]
                mask_parts.append(ug["masks"][:] if "masks" in ug
                                  else np.ones_like(parts[-1]))
    patches = np.concatenate(parts) if parts else np.empty((0, 64, 64), dtype=np.float32)
    if not with_mask:
        return patches
    masks = np.concatenate(mask_parts) if mask_parts else np.empty((0, 64, 64), dtype=np.float32)
    return patches, masks


class BlobTrackData(torch.utils.data.Dataset):
    """Dataset of canonicalized blob patches labeled by track identity.

    Each item returns (patch, label) where:
      - patch: (1, P, P) float32 tensor (grayscale, values in [0, 1])
      - label: int, track identity (contiguous 0-based)

    Only tracks with >= `min_track_length` observations are included.
    Loads all sequences by default; pass `sequence="name"` for one.

    `view_angle_range=[lo_deg, hi_deg]` restricts the tracked observations to a band of
    viewing obliquity (0 = fronto-parallel, 90 = edge-on), which is what the per-band
    validation splits in `conf/track_angle_validation.yaml` use to report FPR95 as a
    function of viewpoint. The band applies to tracked observations only: anchors have
    no pose and are kept (they supply the positive partner), and confusers carry no
    frame id at all, so they stay a plain negative pool — capped, as always, at
    `max_untracked_to_tracked_ratio x` the *surviving* patch count, which keeps the
    positive/negative balance (and hence FPR95) comparable across bands.
    """

    def __init__(
        self,
        h5_path=None,
        min_track_length=2,
        load_into_memory=True,
        sequences=None,
        split=None,   # "train"/"val"/"test": pick sequences by residue class (see _split_sequences)
        include_untracked=False,
        max_untracked_to_tracked_ratio=1.0,
        patch_type="cartesian",   # informational: one file = one patch type
        with_mask=False,
        unique_track_ids=True,   # remap ids to be collision-free across boards (see _load_all_sequences)
        view_angle_range=None,   # [lo_deg, hi_deg): keep only observations viewed at this obliquity
        view_angle_keep_anchors=True,   # anchors have no pose; keep them so bands still have positives
        log_stats=False,   # print a per-sequence / positive-structure breakdown on load
        **_,   # tolerate stray config params (mirrors HomographyData's liberal kwargs)
    ):
        if h5_path is None:
            raise ValueError("h5_path must be provided")

        self.with_mask = with_mask
        # Validate the band up front so a bad config fails before the (slow) load.
        self.view_angle_range = list(view_angle_range) if view_angle_range is not None else None
        _view_angle_bounds(view_angle_range)
        if not load_into_memory:
            raise NotImplementedError("BlobTrackData only supports load_into_memory=True")

        # Explicit `sequences` wins; otherwise a `split` name selects a residue class
        # of the file's sequences so one `.tracks` file backs train/val/test.
        if sequences is None and split is not None:
            sequences = self._split_sequences(h5_path, split)

        with h5py.File(h5_path, "r") as f:
            if log_stats:
                self._log_sequence_stats(f, sequences, split=split,
                                         unique_track_ids=unique_track_ids)

            (all_patches, all_masks, track_ids, _track_lengths,
             affine_shapes, is_anchor) = _load_all_sequences(
                f, sequences, with_mask=with_mask, unique_track_ids=unique_track_ids,
                view_angle_range=view_angle_range,
                view_angle_keep_anchors=view_angle_keep_anchors)

            self.patches = all_patches
            # int64: remapped ids (board_id << 16 | blob_index) and the confuser band
            # below can exceed uint32's comfortable range once boards accumulate.
            self.labels = track_ids.astype(np.int64)
            self.affine_shapes = affine_shapes
            self.masks = all_masks
            self.is_anchor = is_anchor

            if include_untracked:
                ut = load_untracked_patches(f, sequences, with_mask=with_mask)
                ut_patches, ut_masks = ut if with_mask else (ut, None)
                keep = int(len(self.patches) * max_untracked_to_tracked_ratio)
                self.untracked_patches = ut_patches[:keep]
                self.untracked_masks = ut_masks[:keep] if with_mask else None
            else:
                self.untracked_patches = None
                self.untracked_masks = None

        if include_untracked and len(self.untracked_patches):
            # Confusers get fresh singleton labels; they are negatives only (their
            # is_anchor is 0, they carry no board mask). Those labels MUST be disjoint
            # from every real track_id — a confuser sharing an id with a real track
            # tells SupCon to pull an unmatchable background patch toward that track,
            # corrupting the embedding. The old scheme ((labels[0] & 0xFFFF0000) +
            # 0x8000 + arange) assumed confusers fit a 0x8000-wide band above one
            # board's base; with n_ut >> 32768 that band overflowed into other boards'
            # id ranges (measured: ~20k confusers colliding with real tracks). Start
            # strictly above the global max instead, in int64 to preclude wraparound.
            n_ut = len(self.untracked_patches)
            self.labels = self.labels.astype(np.int64)
            start = int(self.labels.max()) + 1 if len(self.labels) else 0
            self.labels = np.concatenate([
                self.labels,
                np.arange(start, start + n_ut, dtype=np.int64),
            ])
            self.is_anchor = np.concatenate([self.is_anchor, np.zeros(n_ut, dtype=np.uint8)])

        band = ""
        if self.view_angle_range is not None:
            lo, hi = self.view_angle_range
            band = f", view_angle=[{lo:g}, {hi:g}) deg"
        print(f"BlobTrackDataset: {len(self.labels)} patches (with_mask={with_mask}{band})")

        if self.view_angle_range is not None:
            # A band drops tracked observations, so it can silently starve the batch of
            # positive pairs (FPR95 is NaN without one). Report what survived — this is
            # the number to watch when deciding whether the bands need widening.
            pos_groups, confusers = self._grouped_indices()
            n_tracked = int(len(self.patches))
            print(f"  band [{lo:g}, {hi:g}) deg: {n_tracked} in-band patches "
                  f"(+{len(self.labels) - n_tracked} confusers), "
                  f"{len(pos_groups)} matchable tracks, {len(confusers)} singletons")
            if len(pos_groups) < 50:
                print(f"  !! WARNING: only {len(pos_groups)} matchable tracks in this band — "
                      "FPR95 will be noisy or NaN; widen the band.")

    def __len__(self):
        return len(self.labels)

    def _get_patch_mask(self, idx):
        """Return (patch, mask) numpy (P, P) for a global index, spanning the
        tracked block and the appended untracked (confuser) block."""
        n_tracked = self.patches.shape[0]
        if idx >= n_tracked:
            j = idx - n_tracked
            patch = self.untracked_patches[j]
            mask = self.untracked_masks[j] if self.with_mask else None
        else:
            patch = self.patches[idx]
            mask = self.masks[idx] if self.with_mask else None
        return patch, mask

    def __getitem__(self, idx):
        patch, mask = self._get_patch_mask(idx)
        # (P, P) -> (1, P, P), transposed to undo Julia column-major storage.
        patch = torch.from_numpy(patch).unsqueeze(0).transpose(1, 2)
        item = {"patch": patch, "label": self.labels[idx]}
        if self.affine_shapes is not None:
            item["affine_shape"] = torch.from_numpy(self.affine_shapes[idx]) if idx < self.patches.shape[0] \
                else torch.eye(2)
        if self.with_mask:
            item["mask"] = torch.from_numpy(mask).unsqueeze(0).transpose(1, 2)
            item["is_anchor"] = int(self.is_anchor[idx])
        return item

    @staticmethod
    def _log_sequence_stats(f, sequences, split=None, unique_track_ids=True):
        """Print a per-sequence + positive-structure breakdown of the loaded data.

        The whole point of a track descriptor is separating identities, so the
        diagnostics centre on where positive pairs actually come from. Track ids are
        counted the way the dataset uses them: when ``unique_track_ids`` is set, ids are
        namespaced by board (``(uid, raw_id)``) so a ``track_id`` shared by two DIFFERENT
        boards is NOT counted as matchable — those are cross-board collisions, reported
        separately. So "matchable" here means a genuine within-board recurrence (a
        board's anchor + a tracked media, or two media), which is the only kind that is
        a true positive. If matchable/anchored is tiny, the data is why training stalls.
        """
        seqs = f["sequences"]
        names = _select_sequences(seqs, sequences)

        # Namespaced key: (board uid, raw id) when de-colliding, else the raw id alone
        # (reproduces the pre-fix view). Also track raw-id -> set of boards for the
        # cross-board collision report.
        def key(u, t):
            return (u, t) if unique_track_ids else t
        track_total = collections.Counter()      # namespaced key -> patches
        track_in_anchor = collections.defaultdict(bool)
        raw_boards = collections.defaultdict(set)  # raw id -> set of board uids
        per_seq = []   # (name, uid, is_anchor, n_patches, unique_keys:set, n_untracked)
        obs_angles = []   # per-observation viewing angle (radians), tracked sequences only
        n_no_angle = 0
        for name in names:
            sg = seqs[name]
            u = _uid_of(name)
            isa = int(sg.attrs.get("is_anchor", 0))
            tid = sg["tracks"]["track_id"][:]
            if not isa:
                try:
                    ang, matched = _observation_view_angles(sg["tracks"], name)
                    obs_angles.append(ang[matched])
                except ValueError:
                    n_no_angle += 1
            n_ut = sg["untracked/patches"].shape[0] if "untracked" in sg else 0
            keys = {key(u, int(t)) for t in tid.tolist()}
            per_seq.append((name, u, isa, len(tid), keys, n_ut))
            for t, c in collections.Counter(tid.tolist()).items():
                track_total[key(u, int(t))] += c
                raw_boards[int(t)].add(u)
                if isa:
                    track_in_anchor[key(u, int(t))] = True

        matchable = {k for k, c in track_total.items() if c >= 2}
        anchored = sum(1 for k in matchable if track_in_anchor[k])
        collisions = sum(1 for _t, bs in raw_boards.items() if len(bs) > 1)

        tag = f" [{split}]" if split else ""
        print(f"\n===== BlobTrackData sequence stats{tag}: {len(names)} sequences "
              f"(unique_track_ids={unique_track_ids}) =====")
        print(f"{'sequence':<30}{'kind':>8}{'patches':>9}{'tracks':>8}{'matchable':>10}{'confusers':>10}")
        n_anchor_seq = n_tracked_seq = tot_patch = tot_conf = 0
        for name, _u, isa, npat, keys, n_ut in per_seq:
            kind = "anchor" if isa else "tracked"
            n_match = len(keys & matchable)
            print(f"{name[:30]:<30}{kind:>8}{npat:>9}{len(keys):>8}{n_match:>10}{n_ut:>10}")
            n_anchor_seq += isa
            n_tracked_seq += (not isa)
            tot_patch += npat
            tot_conf += n_ut

        # Track-multiplicity histogram (how many patches share each namespaced id).
        hist = collections.Counter()
        for c in track_total.values():
            bucket = "1" if c == 1 else "2" if c == 2 else "3-4" if c <= 4 else "5-9" if c <= 9 else "10+"
            hist[bucket] += 1
        order = ["1", "2", "3-4", "5-9", "10+"]
        histstr = "  ".join(f"{b}:{hist.get(b, 0)}" for b in order)

        n_tracks = len(track_total)
        print(f"----- summary{tag}: {n_anchor_seq} anchor + {n_tracked_seq} tracked seqs, "
              f"{tot_patch} track patches + {tot_conf} confusers -----")
        print(f"  distinct track_ids: {n_tracks}  |  matchable (>=2 patches): {len(matchable)} "
              f"({100*len(matchable)/max(1,n_tracks):.1f}%)  |  singleton: {n_tracks-len(matchable)}")
        print(f"  of matchable: anchored (anchor<->tracked): {anchored} "
              f"({100*anchored/max(1,len(matchable)):.1f}%)  |  tracked-only: {len(matchable)-anchored}")
        print(f"  raw ids shared across >1 board (cross-board collisions): {collisions}"
              f"{'  (de-collided by remap)' if unique_track_ids else '  !! NOT de-collided'}")
        print(f"  track multiplicity histogram (patches per track): {histstr}")

        # Viewing-angle distribution over tracked observations, in the 10-degree bands
        # the validation splits use — this is how you see whether a band is populated
        # enough to be worth evaluating (see `view_angle_range`).
        if obs_angles:
            deg = np.degrees(np.concatenate(obs_angles)) if len(obs_angles) > 1 \
                else np.degrees(obs_angles[0])
            edges = np.arange(0.0, 100.0, 10.0)
            counts, _ = np.histogram(deg, bins=edges)
            bandstr = "  ".join(f"{int(e)}-{int(e)+10}:{c}" for e, c in zip(edges[:-1], counts))
            print(f"  view angle (deg) over {deg.size} tracked obs: "
                  f"min {deg.min():.1f} median {np.median(deg):.1f} max {deg.max():.1f}")
            print(f"  view-angle 10deg bands: {bandstr}")
        if n_no_angle:
            print(f"  note: {n_no_angle} tracked sequences carry no view_angles "
                  "(pre-dating the BlobBoards viewing-angle export) — view_angle_range "
                  "would reject them")
        if len(matchable) == 0:
            print("  !! WARNING: NO matchable tracks — every batch is positive-free; "
                  "check that track_ids are shared across a board's sequences.")
        elif anchored == 0:
            print("  !! WARNING: matchable tracks exist but NONE are anchored — positives are "
                  "tracked<->tracked only; verify the anchor sequences share track_ids.")
        print()

    @staticmethod
    def _split_sequences(h5_path, split):
        """Return the set of board uids belonging to a train/val/test split.

        A board's anchor and tracked sequences SHARE a uid (names are ``<uid>`` or
        ``<media>_<uid>``), so the split must be decided per *uid*, assigned once on
        first sight — otherwise the same board could land in val via one sequence and
        train via another, leaking data across splits. Same residue rule as the
        ``__main__`` bootstrap: on a uid's first occurrence, enumeration index
        ``% 10 == 0`` is test, ``== 1`` is validation, the rest is training.
        """
        assert split in ("train", "val", "test"), f"Unknown split {split!r}"
        assigned = {}
        with h5py.File(h5_path, "r") as f:
            for i, seq in enumerate(f["sequences"].keys()):
                m = re.fullmatch(r"(?:.*_)?([A-Fa-f0-9]+)", seq)
                if m is None:
                    continue
                uid = m[1]
                if uid in assigned:
                    continue
                assigned[uid] = "test" if i % 10 == 0 else ("val" if i % 10 == 1 else "train")
        return {uid for uid, grp in assigned.items() if grp == split}

    def _grouped_indices(self):
        """Split the dataset's global indices into positive track groups (a
        ``track_id`` with >= 2 patches, so it can form a positive pair — its anchor
        patch is one of them) and confusers (singleton track_ids + the untracked
        block, one representative index each — they can only ever be negatives)."""
        groups = collections.defaultdict(list)
        for i, lab in enumerate(self.labels):
            groups[int(lab)].append(i)
        pos_groups = [inds for inds in groups.values() if len(inds) >= 2]
        confusers = [inds[0] for inds in groups.values() if len(inds) == 1]
        return pos_groups, confusers

    @staticmethod
    def _pack_balanced(pos_groups, confusers, batch_size, confuser_fraction, rng):
        """Pack whole positive groups + distinct confusers into batches.

        Each batch is filled with whole track groups up to ``1 - confuser_fraction``
        of ``batch_size`` (guaranteeing >= 1 positive pair, since a group has >= 2
        members), then topped up with that many distinct confusers as negatives.
        No index is ever repeated within a batch, so there are no distance-0
        "positives" — the trap that MPerClassSampler falls into when it samples a
        singleton class ``m`` times with replacement. ``rng`` fixes the shuffle.
        """
        pos_groups = list(pos_groups)
        confusers = list(confusers)
        rng.shuffle(pos_groups)
        rng.shuffle(confusers)
        n_conf = max(0, round(batch_size * confuser_fraction))
        n_pos = max(1, batch_size - n_conf)

        def take_confusers(cursor):
            if not (n_conf and confusers):
                return [], cursor
            k = min(n_conf, len(confusers))
            take = [confusers[(cursor + j) % len(confusers)] for j in range(k)]
            return take, (cursor + k) % len(confusers)

        batches, cur, cursor = [], [], 0
        for grp in pos_groups:
            # Flush before this group would push the positive block past its budget,
            # but always keep at least one group per batch (>= 1 positive pair).
            if cur and len(cur) + len(grp) > n_pos:
                conf, cursor = take_confusers(cursor)
                batches.append(cur + conf)
                cur = []
            cur.extend(grp)
        if cur:
            conf, cursor = take_confusers(cursor)
            batches.append(cur + conf)
        return batches

    def get_sampler(self, batch_size, m=None, confuser_fraction=0.5, seed=0):
        """Balanced TRAINING batch sampler, re-randomized each epoch.

        Contrastive losses (SupCon) need positive pairs in a batch. The obvious
        ``MPerClassSampler`` is wrong for track data: every confuser is its own
        singleton class, so it would sample that one patch ``m`` times with
        replacement — duplicate rows with the same label, i.e. distance-0 fake
        positives that pull unmatchable background blobs together. Instead this
        packs whole track groups as positives (no duplication) and adds distinct
        confusers as negatives, exactly like the eval sampler but reshuffled each
        epoch. ``m`` is accepted for the loop's ``m_per_class`` hook but unused —
        whole groups already provide multiple views per track.
        """
        pos_groups, confusers = self._grouped_indices()
        n_conf = max(0, round(batch_size * confuser_fraction))
        print(f"BlobTrackData train sampler: {len(pos_groups)} matchable tracks + "
              f"{len(confusers)} confusers (~{batch_size - n_conf} pos / {n_conf} conf "
              f"per batch, reshuffled each epoch)")
        return _ReshufflingBatchSampler(
            self._pack_balanced, pos_groups, confusers, batch_size, confuser_fraction, seed
        )

    def get_collate_func(self):
        """Collate track items into a ``process_batch_blobs``-compatible batch.

        The blob process-batch keys are reused verbatim so track training runs
        through the exact same loop/loss as the synthetic descriptor: ``keypoints``
        carries the track id in column 1 (the contrastive ``indices``). The patches
        are already canonicalized, so there is no frame to fall out of — zero
        ``keypoint_coords`` inside a ``(1, 1)`` ``image_size`` keep every sample
        in-bounds for the frame filter in ``process_batch_blobs``.
        """
        def collate_track(batch):
            n = len(batch)
            labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
            res = {
                # (N, 2): column 0 unused (feature/image id), column 1 = contrastive index.
                "keypoints": torch.stack([torch.zeros(n, dtype=torch.long), labels], dim=1),
                "keypoint_coords": torch.zeros(n, 2),
                "image_size": torch.tensor([1.0, 1.0]),
                "patches": torch.stack([item["patch"] for item in batch]),
            }
            if "mask" in batch[0]:
                res["masks"] = torch.stack([item["mask"] for item in batch])
                res["is_anchor"] = torch.tensor(
                    [int(item["is_anchor"]) for item in batch], dtype=torch.long
                )
            return res

        return collate_track

    def get_eval_batch_sampler(self, batch_size, confuser_fraction=0.5, seed=0):
        """Balanced validation batches, each guaranteed to contain positive pairs.

        FPR95 is NaN for any batch with no same-``track_id`` pair (see
        ``evaluate.fpr_from_distances``). In a ``.tracks`` file each ``track_id``
        appears at most once per sequence, so a track's matching observations live
        in *other* sequences (its anchor, or another media of the board) — a
        contiguous ``shuffle=False`` slice almost never captures two of them, hence
        the all-NaN validation. This sampler instead groups patches by ``track_id``
        and packs each batch as:

          * whole track groups (ids with >= 2 patches) — each already carries its
            anchor patch, so anchor<->tracked positives are guaranteed — filling
            ``1 - confuser_fraction`` of the batch, then
          * that many distinct confusers (singleton ids + the untracked block),
            each used at most once so they only ever act as negatives (no
            distance-0 duplicate "positives").

        One pass over the positive tracks defines an epoch; the packing is seeded,
        so every epoch sees the same batches (comparable FPR curves). Batch sizes
        are approximate (whole groups), which the loss/metric handle fine.
        """
        pos_groups, confusers = self._grouped_indices()
        batches = self._pack_balanced(
            pos_groups, confusers, batch_size, confuser_fraction, random.Random(seed)
        )
        n_conf = max(0, round(batch_size * confuser_fraction))
        print(f"BlobTrackData eval sampler: {len(batches)} batches from {len(pos_groups)} "
              f"matchable tracks + {len(confusers)} confusers "
              f"(~{batch_size - n_conf} pos / {n_conf} conf per batch)")
        return _ListBatchSampler(batches)


if __name__ == "__main__":
    track_file = "../BlobBoards.jl/out/only_absolute.tracks"
    train_sequences = []
    test_sequences = []
    val_sequences = []
    with h5py.File(track_file, "r") as f:
        for i, seq in enumerate(f["sequences"].keys()):
            # Names are `<media>_<uid>` (tracked) or `<uid>` (anchor); take the
            # trailing hex uid either way.
            m = re.fullmatch(r"(?:.*_)?([A-Fa-f0-9]+)", seq)
            if m is None:
                continue
            board_id = m[1]

            if board_id not in test_sequences and board_id not in val_sequences and board_id not in train_sequences:
                if i % 10 == 0:
                    test_sequences.append(board_id)
                elif i % 10 == 1:
                    val_sequences.append(board_id)
                else:
                    train_sequences.append(board_id)

    print("TRAINING.BOARDS =", train_sequences)
    print("TEST.BOARDS =", test_sequences)
    print("VALIDATION.BOARDS =", val_sequences)

    from blob_matcher.configs.defaults import _C as cfg

    test_dataset = BlobTrackData(track_file, sequences=cfg.TEST.BOARDS, include_untracked=True)
    # for i, data in zip(range(100), test_dataset):
    #     _, l = data
    #     print(f"{i=}, {l=}")

    # with h5py.File(track_file, "r") as f:
    #     patches, labels, lengths = _load_all_sequences(f)
    # print(labels)
    # print(lengths)
    val_dataset = BlobTrackData(track_file, sequences=cfg.VALIDATION.BOARDS, include_untracked=True)
    train_dataset = BlobTrackData(track_file, sequences=cfg.TRAINING.BOARDS, include_untracked=True)