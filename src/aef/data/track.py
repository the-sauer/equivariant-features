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

import re

import h5py
import numpy as np
import torch


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

        Same residue rule as the ``__main__`` bootstrap below and BlobBoards'
        ``get_seeds``: enumerating the file's sequences, index ``% 10 == 0`` is
        test, ``== 1`` is validation, and the rest is training. Selecting by uid
        (not sequence name) keeps a board's anchor and tracked sequences together.
        """
        assert split in ("train", "val", "test"), f"Unknown split {split!r}"
        selected = set()
        with h5py.File(h5_path, "r") as f:
            for i, seq in enumerate(f["sequences"].keys()):
                m = re.fullmatch(r"(?:.*_)?([A-Fa-f0-9]+)", seq)
                if m is None:
                    continue
                grp = "test" if i % 10 == 0 else ("val" if i % 10 == 1 else "train")
                if grp == split:
                    selected.add(m[1])
        return selected

    def get_sampler(self, batch_size, m=4):
        """Class-balanced sampler: ``m`` patches per track id per batch.

        Contrastive losses (SupCon/FPR95) need multiple views of the same track
        in a batch to form positive pairs; plain shuffling scatters them. The
        per-sample label is the track id. Mirrors ``HomographyData.get_sampler``.
        """
        from pytorch_metric_learning.samplers import MPerClassSampler

        return MPerClassSampler(
            self.labels, m=m, batch_size=batch_size, length_before_new_iter=len(self.labels)
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