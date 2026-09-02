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
# an `is_anchor` group attribute (1 = the GT sequence rendered from the board PDF,
# 0 = tracked in real footage). In-tree that flag is called `is_pdf` — "anchor" is
# SupCon's word for the rows of its logit matrix and means something else — but the
# ON-DISK attribute name stays `is_anchor`, so existing `.tracks` files still read.
# The single patch type / scale are recorded as attributes on the `tracks` group.
# Blob-index field width for the collision-free remap (see `_load_all_sequences`).
# Boards observed with up to ~1330 blobs (< 2**16); 16 bits leaves room to grow while
# keeping the packed id in 32 bits (board_id in the high bits, blob index in the low).
_BLOB_BITS = 16


def _uid_of(name):
    # Sequence names are either `<media>_<uid>` (tracked) or `<uid>` (PDF render); return
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


def _observation_view_angles(g, name, required=True):
    """Per-observation viewing angle (radians) for a tracked sequence.

    `.tracks` stores the detection geometry **per frame, not per blob**:
    ``view_angles`` and ``homography_frame_ids`` have one entry per frame the board
    was detected in, while ``frame_id`` gives each observation's source frame. This
    joins the two, returning ``(angles, matched)`` — both per observation, with
    ``matched`` False wherever an observation's frame is absent from the pose table
    (should not happen; such observations are dropped rather than mis-binned).

    ``required=False`` returns ``(None, None)`` instead of raising when the sequence
    predates the viewing-angle export — used when the angles are merely recorded
    alongside the patches (for band balancing) rather than used to filter them.
    """
    missing = [k for k in ("frame_id", "homography_frame_ids", "view_angles") if k not in g]
    if missing:
        if not required:
            return None, None
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
                        view_angle_keep_pdf=True):
    """Load and concatenate track data from per-sequence HDF5 groups.

    Returns (patches, masks, track_ids, track_lengths, affine_shapes, is_pdf,
    view_angles).
    `masks` is None unless `with_mask`; `affine_shapes` None unless `with_affine`.
    `is_pdf` is a per-patch uint8 array (broadcast from the per-sequence flag).
    `view_angles` is a per-patch float64 array of viewing obliquity in RADIANS, NaN
    wherever the patch has no pose: PDF patches (rendered fronto-parallel off the board
    image), observations whose frame is missing from the pose table, and every patch
    of a file predating the viewing-angle export. It is what
    ``BlobTrackData(balance_view_angles=True)`` bins into bands.
    `track_lengths` is the sequence's stored (PRE-filter) histogram and is not
    recomputed when a viewing-angle band drops observations; nothing downstream uses
    it (group structure is rebuilt from the labels in ``_grouped_indices``).

    ``view_angle_range`` — ``[lo_deg, hi_deg]``, keeps only observations whose frame
    was viewed at an obliquity in ``[lo, hi)`` (0 = fronto-parallel, 90 = edge-on).
    Anchor sequences carry no pose (they are rendered off the board image, i.e.
    fronto-parallel by construction) and are kept whole unless
    ``view_angle_keep_pdf`` is False — keeping them is what preserves the
    pdf<->tracked positive pairs that make a narrow band evaluable at all.

    ``unique_track_ids`` (default True) remaps track ids to be COLLISION-FREE across
    boards. The stored ids are ``board_hash + blob_index`` packed into one ~16-bit
    field, so different boards whose hashes are closer than their blob count share ids
    — a measured ~34% of ids span >1 board, pairing unrelated blobs as false positives.
    Here each board (identified by its sequence uid) gets a dense index, and the id is
    repacked as ``(board_index << _BLOB_BITS) | (raw_id - board_base)`` — i.e. a clean
    ``board_id | blob_index``. This preserves within-board matches (a board's PDF render and
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
        # Board base = min raw id over ALL of a board's sequences, so PDF and tracked
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
    all_affine_shapes, all_is_pdf, all_view_angles = [], [], []

    for name in seq_names:
        sg = seqs[name]
        g = sg["tracks"]
        # On-disk attribute name (see the layout note above); `is_pdf` in-tree.
        is_pdf = int(sg.attrs["is_anchor"]) if "is_anchor" in sg.attrs else 0

        # Join the per-frame pose table to the observations up front, so the angles can
        # be sliced by the same `sel` as the patches. Only a band filter *requires* them
        # (an older file without the export is loadable, its angles just stay NaN).
        angles = matched = None
        if not is_pdf:
            angles, matched = _observation_view_angles(g, name, required=bounds is not None)

        # Viewing-angle band: decide WHICH observations to read before reading them,
        # so a narrow band does not pull a whole (large) sequence through memory.
        sel = None
        if bounds is not None:
            if is_pdf:
                if not view_angle_keep_pdf:
                    continue
            else:
                lo, hi = bounds
                sel = np.flatnonzero(matched & (angles >= lo) & (angles < hi))
                if sel.size == 0:
                    continue
                if sel.size == g["patches"].shape[0]:
                    sel = None            # whole sequence: plain slice reads faster

        patches = _read_rows(g["patches"], sel)
        n = patches.shape[0]
        all_patches.append(patches)
        if angles is None:
            all_view_angles.append(np.full(n, np.nan))
        else:
            # Unmatched observations have no pose -> NaN (unbanded), never a stale angle.
            seq_angles = np.where(matched, angles, np.nan)
            all_view_angles.append(seq_angles if sel is None else seq_angles[sel])
        all_track_ids.append(remap(name, _read_rows(g["track_id"], sel)))
        all_track_lengths.append(g["track_lengths"][:])
        if with_mask:
            all_masks.append(_read_rows(g["masks"], sel) if "masks" in g
                             else np.ones_like(patches))
        if with_affine:
            all_affine_shapes.append(_read_rows(g["affine_shapes"], sel))
        all_is_pdf.append(np.full(n, is_pdf, dtype=np.uint8))

    def cat(parts, empty):
        return np.concatenate(parts) if parts else empty

    patches = cat(all_patches, np.zeros((0, 64, 64), dtype=np.float32))
    track_ids = cat(all_track_ids, np.zeros(0, dtype=np.int64))
    track_lengths = cat(all_track_lengths, np.zeros(0, dtype=np.int32))
    masks = cat(all_masks, np.zeros((0, 64, 64), dtype=np.float32)) if with_mask else None
    affine_shapes = cat(all_affine_shapes, np.zeros((0, 2, 2), dtype=np.float32)) if with_affine else None
    is_pdf = cat(all_is_pdf, np.zeros(0, dtype=np.uint8))
    view_angles = cat(all_view_angles, np.zeros(0, dtype=np.float64))

    return patches, masks, track_ids, track_lengths, affine_shapes, is_pdf, view_angles


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


# Default viewing-obliquity bands for training balance: 0-90 deg in 10 deg steps, the
# same grid `conf/track_angle_validation.yaml` reports FPR95 over, so "balanced in
# training" and "measured per band" mean the same partition of viewpoint space.
_DEFAULT_BAND_EDGES = tuple(float(d) for d in range(0, 100, 10))


def _band_edges(spec):
    """Validate viewing-angle band edges (DEGREES, strictly increasing) -> ndarray.

    ``None`` -> `_DEFAULT_BAND_EDGES`. ``n`` edges define ``n - 1`` half-open bands
    ``[e[i], e[i+1])``; angles outside ``[e[0], e[-1])`` are unbanded, exactly like
    the half-open `view_angle_range` bands.
    """
    if spec is None:
        spec = _DEFAULT_BAND_EDGES
    try:
        edges = np.asarray([float(v) for v in spec], dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"view_angle_band_edges must be a list of degrees, got {spec!r}") from e
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(
            "view_angle_band_edges must be >= 2 strictly increasing degrees, "
            f"got {list(spec)!r}"
        )
    return edges


def _bands_of(angles_rad, edges_deg):
    """Bin per-patch viewing angles (radians) into band indices.

    Returns an int64 array; ``-1`` marks a patch with no band — an unknown angle
    (NaN: PDF patches and files predating the pose export) or one outside the edges.
    Those are never *drawn* by the balancer; they only ride along as partners.
    """
    deg = np.degrees(np.asarray(angles_rad, dtype=np.float64))
    with np.errstate(invalid="ignore"):
        bands = np.digitize(deg, edges_deg) - 1
        outside = ~np.isfinite(deg) | (deg < edges_deg[0]) | (deg >= edges_deg[-1])
    bands = bands.astype(np.int64)
    bands[outside] = -1
    return bands


def _band_target(pool_sizes, spec):
    """How many in-band patches one band should be drawn for per EPOCH.

    ``spec`` is ``"min" | "median" | "mean" | "max"`` over the populated bands' pool
    sizes, or an explicit integer. This only sets the epoch LENGTH (batches per epoch
    = target / per-batch quota); the per-batch balance is unaffected. ``"min"``
    makes an epoch one pass over the rarest band (shortest, least repetition),
    ``"max"`` one pass over the most crowded one (longest, heaviest repetition of the
    rare bands); ``"median"`` sits in between and is the default.
    """
    sizes = [int(s) for s in pool_sizes if s > 0]
    if not sizes:
        return 0
    if isinstance(spec, bool):
        raise ValueError(f"view_angle_band_target must be a name or an int, got {spec!r}")
    if isinstance(spec, str):
        stat = {"min": min, "max": max,
                "mean": lambda v: round(sum(v) / len(v)),
                "median": lambda v: round(float(np.median(v)))}.get(spec)
        if stat is None:
            raise ValueError(
                "view_angle_band_target must be 'min', 'median', 'mean', 'max' or an "
                f"int, got {spec!r}"
            )
        return max(1, int(stat(sizes)))
    target = int(spec)
    if target < 1:
        raise ValueError(f"view_angle_band_target must be >= 1, got {spec}")
    return target


def _band_quotas(bands, edges_deg, budget, bias):
    """Split one batch's in-band-patch ``budget`` over the populated ``bands``.

    ``bias == 0`` is the plain balance: every band gets the same quota. A positive
    bias tilts the batch toward obliquity — a band's weight is ``exp(bias * u)`` with
    ``u`` its band-center angle rescaled so the most frontal POPULATED band sits at 0
    and the most oblique at 1, which makes ``exp(bias)`` exactly the ratio between the
    two ends' quotas (``bias = ln 10 ~ 2.303`` -> the edge-on band draws 10x what the
    near-frontal one does, the bands in between interpolating geometrically). A
    negative bias tilts back toward fronto-parallel, i.e. part of the way to the
    file's own distribution, without giving up the balance's floor.

    Every populated band keeps at least one draw, so no bias can silence a band
    outright; the rounding leftovers go by largest fractional remainder. Bands are
    weighted by their center ANGLE, not by their rank, so custom
    ``view_angle_band_edges`` (say a wide catch-all top band) tilt by how oblique a
    band actually is.
    """
    bands = sorted(bands)
    n = len(bands)
    if n == 0:
        return {}
    budget = max(int(budget), n)
    centers = np.asarray([(edges_deg[b] + edges_deg[b + 1]) / 2.0 for b in bands])
    span = centers[-1] - centers[0]
    u = (centers - centers[0]) / span if span > 0 else np.zeros(n)
    weights = np.exp(float(bias) * u)
    exact = weights / weights.sum() * budget
    floors = np.floor(exact)
    quotas = np.maximum(1, floors).astype(np.int64)
    # Hand the floored-away remainder to the largest fractions, skipping the bands the
    # >= 1 floor already lifted above their share (they have been paid).
    left = budget - int(quotas.sum())
    if left > 0:
        cand = [j for j in np.argsort(-(exact - floors)) if floors[j] >= 1] or list(range(n))
        for k in range(left):
            quotas[cand[k % len(cand)]] += 1
    while left < 0 and quotas.max() > 1:   # the >= 1 floor overshot a tiny budget
        quotas[int(np.argmax(quotas))] -= 1
        left += 1
    return {int(b): int(q) for b, q in zip(bands, quotas)}


class _BandCursor:
    """Cycles a band's pool in shuffled order, reshuffling on every wrap.

    A cursor persisting across an epoch's batches (rather than an independent draw per
    batch) is what makes a crowded band's patches get *covered* instead of resampled:
    batch ``k + 1`` continues where batch ``k`` stopped. It is rebuilt per epoch, so
    which slice of an over-large band an epoch sees varies with the epoch's seed.
    """

    def __init__(self, pool, rng):
        self._pool = list(pool)
        self._order = list(range(len(self._pool)))
        rng.shuffle(self._order)
        self._i = 0

    def __len__(self):
        return len(self._pool)

    def next(self, rng):
        if self._i >= len(self._order):
            rng.shuffle(self._order)
            self._i = 0
        item = self._pool[self._order[self._i]]
        self._i += 1
        return item


_Cap = collections.namedtuple(
    "_Cap", "keep n_pos n_singletons n_pos_available n_singletons_available")


def _cap_tracked(labels, max_keypoints, seed=0):
    """Deterministic cap on the loaded tracked block -> indices to keep.

    ``max_keypoints`` is a budget PER KIND, applied to the two things a batch is
    packed from (see ``_pack_balanced``):

      * **positives** — patches of a ``track_id`` seen >= 2 times, kept as WHOLE
        groups. Never split a group: taking a subset of size 1 would silently
        demote a positive track to a confuser, changing the class balance the cap
        exists to equalize.
      * **confusers** — singleton ``track_id``s. The untracked block shares this
        same budget (see ``_untracked_budget``), so ``max_keypoints`` bounds the
        total negative-only pool, not just its tracked half.

    Groups are shuffled with a seeded RNG and taken until the next one would
    overflow, so a split's members are stable across runs; the returned indices are
    sorted, leaving the on-disk ordering (and hence patch/label alignment) intact.

    Returns a ``_Cap``: the sorted index array, how much of each budget it spends,
    and how much of each kind was on offer (so the caller can tell a split that hit
    the cap from one the file was simply too small to fill).
    """
    groups = collections.defaultdict(list)
    for i, lab in enumerate(labels):
        groups[int(lab)].append(i)
    pos_groups = [inds for inds in groups.values() if len(inds) >= 2]
    singletons = [inds[0] for inds in groups.values() if len(inds) == 1]
    n_pos_available = sum(len(g) for g in pos_groups)

    rng = random.Random(seed)
    rng.shuffle(pos_groups)
    rng.shuffle(singletons)

    keep, n_pos = [], 0
    for grp in pos_groups:
        if n_pos + len(grp) > max_keypoints:
            break
        keep.extend(grp)
        n_pos += len(grp)
    kept_singletons = singletons[:max_keypoints]
    keep.extend(kept_singletons)
    keep.sort()
    return _Cap(np.asarray(keep, dtype=np.int64), n_pos, len(kept_singletons),
                n_pos_available, len(singletons))


def _untracked_budget(n_tracked, ratio, n_singletons, max_keypoints):
    """How many untracked (confuser) patches to keep.

    Always ``ratio x`` the surviving tracked count. With a cap, the untracked block
    additionally shares the confuser budget with the tracked singletons — both are
    negative-only, so ``max_keypoints`` bounds their sum.
    """
    keep = int(n_tracked * ratio)
    if max_keypoints is not None:
        keep = min(keep, max(0, max_keypoints - n_singletons))
    return keep


def _band_pools(pos_groups, bands):
    """Index positive track groups by the viewing-angle band of their observations.

    Returns ``(pools, groups)``. ``pools[band]`` lists every draw that band can offer
    as ``(group, observation)`` — one entry per in-band observation, so a track seen
    ten times at 5 deg is ten draws in that band and none in any other.
    ``groups[i] = (members, unbanded)`` carries group ``i``'s patch indices and the
    pose-less subset of them (in practice its PDF render), which is the preferred
    positive partner: matching an oblique observation against the fronto-parallel
    render is the task the descriptor is trained for.
    """
    pools = collections.defaultdict(list)
    groups = []
    for gi, members in enumerate(pos_groups):
        members = list(members)
        groups.append((members, [i for i in members if bands[i] < 0]))
        for i in members:
            band = int(bands[i])
            if band >= 0:
                pools[band].append((gi, i))
    return dict(pools), groups


def _instance(groups, gi, seed, used, rng):
    """One positive instance: the drawn observation plus a partner from its own track.

    The partner is a pose-less member (the PDF render) when the track has an unused
    one, else another of its observations — matching against the fronto-parallel
    render is the deployment task, but an oblique<->oblique pair is a valid positive
    too and is what lets a sparsely observed band still fill its quota.

    ``used`` is the set of patch indices already in the batch; neither end of the
    instance may come from it, so a track may contribute SEVERAL pairs to one batch
    as long as they are disjoint. Returns None when no such pair is left.
    """
    members, unbanded = groups[gi]
    if seed in used:
        return None
    pool = [i for i in unbanded if i not in used]
    if not pool:
        pool = [i for i in members if i != seed and i not in used]
    if not pool:
        return None
    return [seed, pool[rng.randrange(len(pool))]]


def _band_capacity(pool, groups):
    """Ceiling on the in-band patches one band can put into a SINGLE batch.

    A batch takes disjoint pairs only, so a track with ``m`` in-band observations
    contributes at most ``1 + 2*floor((m-1)/2)`` of them when it has a PDF patch to
    spend on the first pair, and ``2*floor(m/2)`` when it does not. Bands sharing a
    track compete for that PDF patch, so the real number can be lower — this is what the
    band could reach at best, which is the honest thing to compare a quota against.
    """
    total = 0
    for gi, m in collections.Counter(gi for gi, _ in pool).items():
        total += (1 + 2 * ((m - 1) // 2)) if groups[gi][1] else 2 * (m // 2)
    return total


def _pack_band_balanced(pools, groups, confusers, batch_size, confuser_fraction,
                        n_batches, rng, quotas=None):
    """Pack batches that draw EQUALLY from every populated viewing-angle band.

    Balance is enforced per BATCH, which is the granularity that matters: SupCon sees
    exactly the positives and negatives that share a batch, so an epoch-level quota
    that still let the fronto-parallel band dominate each individual batch would not
    change the loss. Every batch therefore asks each band for the same ``quota`` of
    in-band patches, drawn as positive pairs (see ``_instance``), and then tops up
    with distinct confusers, the same negatives ``_pack_balanced`` uses. A band can
    land one patch over its quota (the pair that fills it may put two in-band patches
    in), so batch sizes vary slightly — as they already do under ``_pack_balanced``.

    ``quotas`` (band -> in-band patches per batch, from ``_band_quotas``) overrides
    the equal split when the caller wants the batch tilted toward the oblique bands;
    ``None`` splits the positive half evenly, which is what "balanced" means.

    Two invariants carry over from ``_pack_balanced``: no index appears twice in a
    batch — a duplicate is a distance-0 fake positive — and every batch holds at least
    one positive pair by construction. A band whose observations run out before its
    quota is filled contributes all the disjoint pairs it can rather than being padded
    with duplicates; ``BlobTrackData.get_sampler`` reports that shortfall instead of
    hiding it.
    """
    bands = sorted(pools)
    if not bands:
        return []
    n_conf = max(0, round(batch_size * confuser_fraction))
    # The quota counts IN-BAND patches, not instances: a pdf-paired instance
    # contributes one, an oblique<->oblique pair two, so counting instances would let
    # the sparse bands (the ones that run out of PDF patches) overshoot the crowded ones.
    # Each in-band patch drags in at most one partner, so a batch stays within
    # 2 * quota * n_bands positive slots.
    if quotas is None:
        quotas = {b: max(1, (batch_size - n_conf) // (2 * len(bands))) for b in bands}
    cursors = {b: _BandCursor(pools[b], rng) for b in bands}
    band_of = {i: b for b, pool in pools.items() for _gi, i in pool}
    confusers = list(confusers)
    rng.shuffle(confusers)

    batches, cursor = [], 0
    for _ in range(n_batches):
        used, picks = set(), []
        order = list(bands)
        rng.shuffle(order)   # no band is systematically first to claim a shared track
        for band in order:
            cur, quota = cursors[band], quotas[band]
            taken, attempts, budget = 0, 0, len(cur) + quota
            while taken < quota and attempts < budget:
                gi, seed = cur.next(rng)
                attempts += 1
                inst = _instance(groups, gi, seed, used, rng)
                if inst is None:
                    continue      # nothing of this track left that is not in the batch
                used.update(inst)
                picks.extend(inst)
                taken += sum(1 for i in inst if band_of.get(i) == band)
        if not picks:
            continue
        if n_conf and confusers:
            k = min(n_conf, len(confusers))
            picks.extend(confusers[(cursor + j) % len(confusers)] for j in range(k))
            cursor = (cursor + k) % len(confusers)
        batches.append(picks)
    return batches


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
    function of viewpoint. The band applies to tracked observations only: PDF patches have
    no pose and are kept (they supply the positive partner), and confusers carry no
    frame id at all, so they stay a plain negative pool — capped, as always, at
    `max_untracked_to_tracked_ratio x` the *surviving* patch count, which keeps the
    positive/negative balance (and hence FPR95) comparable across bands.

    `balance_view_angles=True` makes the TRAINING sampler draw equally from every
    viewing-angle band instead of following the file's own (heavily fronto-parallel)
    distribution — see `get_sampler`. `view_angle_band_edges` (degrees, default 0-90 in
    10 deg steps, the grid `conf/track_angle_validation.yaml` measures on) sets the
    partition and `view_angle_band_target` the epoch length. `view_angle_band_bias`
    tilts that equal draw: `0.0` (default) is the plain balance, `b > 0` gives the
    most oblique populated band `exp(b)x` the quota of the most frontal one (so
    `ln 10 ~ 2.3` is a 10:1 tilt), `b < 0` leans back toward fronto-parallel. It is
    orthogonal to
    `view_angle_range`: that one restricts which observations exist at all (used to
    build the per-band validation splits), this one only reweights how the surviving
    ones are batched.

    `max_keypoints` caps the split's SIZE the same way it does for `HomographyData`,
    so bands that differ by an order of magnitude in population (real footage is
    heavily fronto-parallel-biased) still evaluate the same number of patches and
    their FPR95 stays comparable. It is a budget per kind, applied after the band
    filter: at most `max_keypoints` positive-track patches (whole track groups, never
    split) and at most `max_keypoints` confusers (tracked singletons + untracked,
    which share the budget). `subsample_seed` fixes which ones survive.
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
        view_angle_keep_pdf=True,   # PDF patches have no pose; keep them so bands still have positives
        balance_view_angles=False,   # TRAINING: draw each obliquity band equally (see get_sampler)
        view_angle_band_edges=None,   # band edges in degrees; None -> 0..90 in 10 deg steps
        view_angle_band_target="median",   # in-band patches per band per epoch: min|median|mean|max or an int
        view_angle_band_bias=0.0,   # >0 tilts the per-batch quotas toward the oblique bands (see _band_quotas)
        max_keypoints=None,   # cap per kind: <= this many positive-track patches AND confusers (see _cap_tracked)
        subsample_seed=0,   # RNG seed for the max_keypoints subsample; keep fixed so a split's members are stable
        log_stats=False,   # print a per-sequence / positive-structure breakdown on load
        **_,   # tolerate stray config params (mirrors HomographyData's liberal kwargs)
    ):
        if h5_path is None:
            raise ValueError("h5_path must be provided")

        self.with_mask = with_mask
        # Validate the band up front so a bad config fails before the (slow) load.
        self.view_angle_range = list(view_angle_range) if view_angle_range is not None else None
        _view_angle_bounds(view_angle_range)
        self.balance_view_angles = bool(balance_view_angles)
        self.view_angle_band_edges = _band_edges(view_angle_band_edges)
        self.view_angle_band_target = view_angle_band_target
        _band_target([1], view_angle_band_target)   # same: fail on a bad spec, not at epoch 0
        try:
            self.view_angle_band_bias = float(view_angle_band_bias)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"view_angle_band_bias must be a number, got {view_angle_band_bias!r}") from e
        if not np.isfinite(self.view_angle_band_bias):
            raise ValueError(
                f"view_angle_band_bias must be finite, got {view_angle_band_bias!r}")
        if max_keypoints is not None:
            max_keypoints = int(max_keypoints)
            if max_keypoints < 1:
                raise ValueError(f"max_keypoints must be >= 1 or None, got {max_keypoints}")
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
             affine_shapes, is_pdf, view_angles) = _load_all_sequences(
                f, sequences, with_mask=with_mask, unique_track_ids=unique_track_ids,
                view_angle_range=view_angle_range,
                view_angle_keep_pdf=view_angle_keep_pdf)

            self.patches = all_patches
            # Per-patch obliquity in radians, NaN where there is no pose (PDF patches).
            self.view_angles = view_angles
            # int64: remapped ids (board_id << 16 | blob_index) and the confuser band
            # below can exceed uint32's comfortable range once boards accumulate.
            self.labels = track_ids.astype(np.int64)
            self.affine_shapes = affine_shapes
            self.masks = all_masks
            self.is_pdf = is_pdf

            # Cap the split's size AFTER the band filter, so the budget counts only
            # what the band kept (an uncapped band is as big as the file allows,
            # which is what makes bands incomparable in the first place).
            n_singletons = 0
            if max_keypoints is not None:
                n_singletons = self._apply_keypoint_cap(max_keypoints, subsample_seed)

            if include_untracked:
                ut = load_untracked_patches(f, sequences, with_mask=with_mask)
                ut_patches, ut_masks = ut if with_mask else (ut, None)
                keep = _untracked_budget(len(self.patches), max_untracked_to_tracked_ratio,
                                         n_singletons, max_keypoints)
                self.untracked_patches = ut_patches[:keep]
                self.untracked_masks = ut_masks[:keep] if with_mask else None
            else:
                self.untracked_patches = None
                self.untracked_masks = None

        if include_untracked and len(self.untracked_patches):
            # Confusers get fresh singleton labels; they are negatives only (their
            # is_pdf is 0, they carry no board mask). Those labels MUST be disjoint
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
            self.is_pdf = np.concatenate([self.is_pdf, np.zeros(n_ut, dtype=np.uint8)])
            # Confusers carry no frame id, hence no angle: unbanded, negatives only.
            self.view_angles = np.concatenate([self.view_angles, np.full(n_ut, np.nan)])

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

    def _apply_keypoint_cap(self, max_keypoints, subsample_seed):
        """Subsample the tracked block in place to the `max_keypoints` budget.

        Runs before the untracked block is appended, so `self.labels` still holds
        track ids only. Returns how many singleton (confuser) patches survived —
        the untracked block gets the rest of the confuser budget.
        """
        n_before = len(self.labels)
        cap = _cap_tracked(self.labels, max_keypoints, subsample_seed)
        keep = cap.keep
        self.patches = self.patches[keep]
        self.labels = self.labels[keep]
        self.is_pdf = self.is_pdf[keep]
        self.view_angles = self.view_angles[keep]
        if self.affine_shapes is not None:
            self.affine_shapes = self.affine_shapes[keep]
        if self.masks is not None:
            self.masks = self.masks[keep]
        print(f"Keypoint cap: kept {len(keep)}/{n_before} tracked patches "
              f"({cap.n_pos}/{cap.n_pos_available} in positive groups, "
              f"{cap.n_singletons}/{cap.n_singletons_available} singleton confusers; "
              f"max_keypoints={max_keypoints}, seed={subsample_seed})")
        if cap.n_pos_available < max_keypoints:
            print(f"  note: only {cap.n_pos_available} positive-track patches available "
                  f"(< max_keypoints={max_keypoints}); this split is smaller than the cap")
        return cap.n_singletons

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
        # Always emitted, mask path or not: `is_pdf` is also a LOSS-side label
        # (`ProxyAnchoredSupCon` contrasts PDF patches against image patches only).
        item["is_pdf"] = int(self.is_pdf[idx])
        if self.with_mask:
            item["mask"] = torch.from_numpy(mask).unsqueeze(0).transpose(1, 2)
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
        board's PDF render + a tracked media, or two media), which is the only kind that
        is a true positive. If matchable/pdf-backed is tiny, the data is why training
        stalls.
        """
        seqs = f["sequences"]
        names = _select_sequences(seqs, sequences)

        # Namespaced key: (board uid, raw id) when de-colliding, else the raw id alone
        # (reproduces the pre-fix view). Also track raw-id -> set of boards for the
        # cross-board collision report.
        def key(u, t):
            return (u, t) if unique_track_ids else t
        track_total = collections.Counter()      # namespaced key -> patches
        track_in_pdf = collections.defaultdict(bool)
        raw_boards = collections.defaultdict(set)  # raw id -> set of board uids
        per_seq = []   # (name, uid, is_pdf, n_patches, unique_keys:set, n_untracked)
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
                    track_in_pdf[key(u, int(t))] = True

        matchable = {k for k, c in track_total.items() if c >= 2}
        pdf_backed = sum(1 for k in matchable if track_in_pdf[k])
        collisions = sum(1 for _t, bs in raw_boards.items() if len(bs) > 1)

        tag = f" [{split}]" if split else ""
        print(f"\n===== BlobTrackData sequence stats{tag}: {len(names)} sequences "
              f"(unique_track_ids={unique_track_ids}) =====")
        print(f"{'sequence':<30}{'kind':>8}{'patches':>9}{'tracks':>8}{'matchable':>10}{'confusers':>10}")
        n_pdf_seq = n_tracked_seq = tot_patch = tot_conf = 0
        for name, _u, isa, npat, keys, n_ut in per_seq:
            kind = "pdf" if isa else "tracked"
            n_match = len(keys & matchable)
            print(f"{name[:30]:<30}{kind:>8}{npat:>9}{len(keys):>8}{n_match:>10}{n_ut:>10}")
            n_pdf_seq += isa
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
        print(f"----- summary{tag}: {n_pdf_seq} pdf + {n_tracked_seq} tracked seqs, "
              f"{tot_patch} track patches + {tot_conf} confusers -----")
        print(f"  distinct track_ids: {n_tracks}  |  matchable (>=2 patches): {len(matchable)} "
              f"({100*len(matchable)/max(1,n_tracks):.1f}%)  |  singleton: {n_tracks-len(matchable)}")
        print(f"  of matchable: pdf-backed (pdf<->tracked): {pdf_backed} "
              f"({100*pdf_backed/max(1,len(matchable)):.1f}%)  |  "
              f"tracked-only: {len(matchable)-pdf_backed}")
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
        elif pdf_backed == 0:
            print("  !! WARNING: matchable tracks exist but NONE are pdf-backed — positives are "
                  "tracked<->tracked only; verify the PDF sequences share track_ids.")
        print()

    @staticmethod
    def _split_sequences(h5_path, split):
        """Return the set of board uids belonging to a train/val/test split.

        A board's PDF and tracked sequences SHARE a uid (names are ``<uid>`` or
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
        ``track_id`` with >= 2 patches, so it can form a positive pair — its PDF
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

        With ``balance_view_angles`` the positive half of every batch is instead drawn
        EQUALLY from each viewing-obliquity band (``_pack_band_balanced``). Real
        footage is heavily fronto-parallel-biased — the 10 deg histogram
        ``log_stats=True`` prints typically falls off by orders of magnitude — so the
        default packing feeds the loss that same bias: near-frontal pairs dominate
        every batch and the oblique tail, which is exactly what an affine-equivariant
        descriptor has to get right, contributes a handful of pairs per epoch. A band
        too sparse to fill its quota out of disjoint pairs stops at what it has —
        padding it with duplicates would create distance-0 fake positives — and its
        ceiling (``_band_capacity``) is printed alongside the quota below.

        ``view_angle_band_bias`` tilts those per-band quotas further, up the obliquity
        axis (``_band_quotas``): equal treatment of the bands is already a large
        upweighting of the oblique tail, but the tail is also where the descriptor is
        hardest, so it can be worth spending more of the batch there than parity
        gives. The tilt redistributes the same positive budget — the batch size, the
        number of batches per epoch and the confuser half are unchanged.
        """
        pos_groups, confusers = self._grouped_indices()
        n_conf = max(0, round(batch_size * confuser_fraction))
        if not self.balance_view_angles:
            print(f"BlobTrackData train sampler: {len(pos_groups)} matchable tracks + "
                  f"{len(confusers)} confusers (~{batch_size - n_conf} pos / {n_conf} conf "
                  f"per batch, reshuffled each epoch)")
            return _ReshufflingBatchSampler(
                self._pack_balanced, pos_groups, confusers, batch_size, confuser_fraction, seed
            )

        bands = _bands_of(self.view_angles, self.view_angle_band_edges)
        pools, groups = _band_pools(pos_groups, bands)
        if not pools:
            raise ValueError(
                "balance_view_angles=True but no observation of a matchable track "
                "carries a viewing angle inside "
                f"[{self.view_angle_band_edges[0]:g}, {self.view_angle_band_edges[-1]:g}) deg "
                "— either the `.tracks` file predates the viewing-angle export (rebuild "
                "it) or view_angle_band_edges does not cover the data."
            )

        target = _band_target([len(p) for p in pools.values()], self.view_angle_band_target)
        # `base` is the equal share; the bias only redistributes it between the bands,
        # so the epoch length (and the total number of positives per batch) does not
        # move when the bias is turned up.
        base = max(1, (batch_size - n_conf) // (2 * len(pools)))
        bias = self.view_angle_band_bias
        quotas = _band_quotas(pools, self.view_angle_band_edges, base * len(pools), bias)
        n_batches = max(1, -(-target // base))
        edges = self.view_angle_band_edges
        tilt = (f", bias={bias:g} -> quotas {min(quotas.values())}..{max(quotas.values())}"
                if bias else "")
        print(f"BlobTrackData train sampler (view-angle balanced): {len(pools)}/"
              f"{len(edges) - 1} populated bands x ~{base} in-band patches per batch "
              f"+ {n_conf} confusers, {n_batches} batches/epoch "
              f"(target={self.view_angle_band_target!r} -> {target} draws/band/epoch{tilt})")
        # `draws` is what a band is asked for per epoch; against its pool size that is
        # the resampling factor — >1 means the band repeats within an epoch (the rare,
        # oblique end), <1 means only that fraction of it is seen per epoch (the
        # crowded near-frontal end; each epoch reshuffles, so the rest comes up later).
        for band in sorted(pools):
            n_obs = len(pools[band])
            n_tracks = len({gi for gi, _ in pools[band]})
            quota = quotas[band]
            capacity = _band_capacity(pools[band], groups)
            short = (f"  !! caps at {capacity} < quota {quota}: too few disjoint pairs, "
                     "widen this band") if capacity < quota else ""
            print(f"  band [{edges[band]:g}, {edges[band + 1]:g}) deg: {n_obs} obs, "
                  f"{n_tracks} tracks, quota {quota}/batch, "
                  f"{quota * n_batches / n_obs:.2f} draws/obs per epoch{short}")

        def pack(_pos_groups, conf, bs, conf_frac, rng):
            # `pos_groups` is already folded into `pools`/`groups`; the confusers and
            # the per-epoch RNG are what the sampler still supplies.
            return _pack_band_balanced(pools, groups, conf, bs, conf_frac, n_batches, rng,
                                       quotas=quotas)

        return _ReshufflingBatchSampler(
            pack, pos_groups, confusers, batch_size, confuser_fraction, seed
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
            res["is_pdf"] = torch.tensor(
                [int(item["is_pdf"]) for item in batch], dtype=torch.long
            )
            if "mask" in batch[0]:
                res["masks"] = torch.stack([item["mask"] for item in batch])
            return res

        return collate_track

    def get_eval_batch_sampler(self, batch_size, confuser_fraction=0.5, seed=0):
        """Balanced validation batches, each guaranteed to contain positive pairs.

        FPR95 is NaN for any batch with no same-``track_id`` pair (see
        ``evaluate.fpr_from_distances``). In a ``.tracks`` file each ``track_id``
        appears at most once per sequence, so a track's matching observations live
        in *other* sequences (its PDF render, or another media of the board) — a
        contiguous ``shuffle=False`` slice almost never captures two of them, hence
        the all-NaN validation. This sampler instead groups patches by ``track_id``
        and packs each batch as:

          * whole track groups (ids with >= 2 patches) — each already carries its
            PDF patch, so pdf<->tracked positives are guaranteed — filling
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
            # Names are `<media>_<uid>` (tracked) or `<uid>` (PDF render); take the
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