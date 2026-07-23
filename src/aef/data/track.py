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
def _select_sequences(seqs, sequences):
    # Sequence names are either `<media>_<uid>` (tracked) or `<uid>` (anchor). The
    # `sequences` filter is a set of board uids; match the trailing hex uid of each
    # name (works for both forms).
    if sequences is None:
        return list(seqs.keys())
    def uid_of(s):
        m = re.fullmatch(r"(?:.*_)?([A-Fa-f0-9]+)", s)
        return m[1] if m else None
    return [s for s in seqs.keys() if uid_of(s) in sequences]


def _load_all_sequences(f, sequences=None, with_mask=False, with_affine=True):
    """Load and concatenate track data from per-sequence HDF5 groups.

    Returns (patches, masks, track_ids, track_lengths, affine_shapes, is_anchor).
    `masks` is None unless `with_mask`; `affine_shapes` None unless `with_affine`.
    `is_anchor` is a per-patch uint8 array (broadcast from the per-sequence flag).
    Track ids keep their stored (board-offset) values — not remapped.
    """
    seqs = f["sequences"]
    seq_names = _select_sequences(seqs, sequences)

    all_patches, all_masks = [], []
    all_track_ids, all_track_lengths = [], []
    all_affine_shapes, all_is_anchor = [], []

    for name in seq_names:
        sg = seqs[name]
        g = sg["tracks"]
        patches = g["patches"][:]
        n = patches.shape[0]
        all_patches.append(patches)
        all_track_ids.append(g["track_id"][:])
        all_track_lengths.append(g["track_lengths"][:])
        if with_mask:
            all_masks.append(g["masks"][:] if "masks" in g else np.ones_like(patches))
        if with_affine:
            all_affine_shapes.append(g["affine_shapes"][:])
        is_anchor = int(sg.attrs["is_anchor"]) if "is_anchor" in sg.attrs else 0
        all_is_anchor.append(np.full(n, is_anchor, dtype=np.uint8))

    def cat(parts, empty):
        return np.concatenate(parts) if parts else empty

    patches = cat(all_patches, np.zeros((0, 64, 64), dtype=np.float32))
    track_ids = cat(all_track_ids, np.zeros(0, dtype=np.int32))
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
    """

    def __init__(
        self,
        h5_path,
        min_track_length=2,
        load_into_memory=True,
        sequences=None,
        split=None,   # "train"/"val"/"test": pick sequences by residue class (see _split_sequences)
        include_untracked=False,
        max_untracked_to_tracked_ratio=1.0,
        patch_type="cartesian",   # informational: one file = one patch type
        with_mask=False,
        **_,   # tolerate stray config params (mirrors HomographyData's liberal kwargs)
    ):
        self.with_mask = with_mask
        if not load_into_memory:
            raise NotImplementedError("BlobTrackData only supports load_into_memory=True")

        # Explicit `sequences` wins; otherwise a `split` name selects a residue class
        # of the file's sequences so one `.tracks` file backs train/val/test.
        if sequences is None and split is not None:
            sequences = self._split_sequences(h5_path, split)

        with h5py.File(h5_path, "r") as f:
            (all_patches, all_masks, track_ids, _track_lengths,
             affine_shapes, is_anchor) = _load_all_sequences(f, sequences, with_mask=with_mask)

            self.patches = all_patches
            self.labels = track_ids.astype(np.uint32)
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
            # Confusers get fresh singleton labels in a reserved band; they are
            # negatives only (their is_anchor is 0, they carry no board mask).
            n_ut = len(self.untracked_patches)
            untracked_start_label = (self.labels[0] & 0xFFFF0000) + 0x00008000
            self.labels = np.concatenate([
                self.labels,
                np.arange(untracked_start_label, untracked_start_label + n_ut, dtype=self.labels.dtype),
            ])
            self.is_anchor = np.concatenate([self.is_anchor, np.zeros(n_ut, dtype=np.uint8)])

        print(f"BlobTrackDataset: {len(self.labels)} patches (with_mask={with_mask})")

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