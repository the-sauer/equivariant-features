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


def _load_all_sequences(f, sequences=None, patch_type="cartesian"):
    """Load and concatenate track data from per-sequence HDF5 groups.

    Returns (patches, track_ids, track_lengths) with track_ids offset so
    they are globally unique across sequences.
    """
    seqs = f["sequences"]
    seq_names = list(filter(lambda s: re.fullmatch("\\w+_([A-Fa-f0-9]+)\\)?", s)[1] in sequences, seqs.keys())) if sequences is not None else list(seqs.keys())

    all_patches = []
    all_track_ids = []
    all_track_lengths = []
    tid_offset = 0
    if patch_type == "original":
        all_affine_shapes = []

    for name in seq_names:
        sg = seqs[name]
        g = sg["tracks"]
        patches = g[f"patches/{patch_type}"][:]
        track_ids = g["track_id"][:]# + tid_offset
        track_lengths = g["track_lengths"][:]
        n_tracks = int(g["n_tracks"][()])
        if patch_type == "original":
            all_affine_shapes.append(g["affine_shapes"][:])
        all_patches.append(patches)
        all_track_ids.append(track_ids)
        all_track_lengths.append(track_lengths)
        tid_offset += n_tracks

    patches = np.concatenate(all_patches) if all_patches else np.zeros((0, 64, 64), dtype=np.float32)
    track_ids = np.concatenate(all_track_ids) if all_track_ids else np.zeros(0, dtype=np.int32)
    track_lengths = np.concatenate(all_track_lengths) if all_track_lengths else np.zeros(0, dtype=np.int32)
    if patch_type == "original":
        affine_shapes = np.concatenate(all_affine_shapes)
    else:
        affine_shapes = None

    return patches, track_ids, track_lengths, affine_shapes


def load_untracked_patches(f, sequences=None):
    seqs = f["sequences"]
    seq_names = list(filter(lambda s: re.fullmatch("\\w+_([A-Fa-f0-9]+)\\)?", s)[1] in sequences, seqs.keys())) if sequences is not None else list(seqs.keys())
    parts = []
    for name in seq_names:
        sg = seqs[name]
        if "untracked" in sg:
            parts.append(sg["untracked/patches"][:])
    if parts:
        patches = np.concatenate(parts)
    else:
        patches = np.empty((0, 64, 64), dtype=np.float32)
    return patches


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
        include_untracked=False,
        max_untracked_to_tracked_ratio=1.0,
        patch_type="cartesian"
    ):
        with h5py.File(h5_path, "r") as f:
            all_patches, track_ids, track_lengths, affine_shapes = _load_all_sequences(f, sequences, patch_type)

            # # Identify valid tracks (long enough for positive pairs)
            # valid_set = set(
            #     (np.where(track_lengths >= min_track_length)[0] + 1).tolist()
            # )  # track_ids are 1-based per sequence (offset applied)

            # # Build mask of patches belonging to valid tracks
            # mask = np.isin(track_ids, list(valid_set))

            if load_into_memory:
                self.patches = all_patches
                self.labels = track_ids.astype(np.uint32)
                self.affine_shapes = affine_shapes
                
                if include_untracked:
                    self.untracked_patches = load_untracked_patches(f, sequences)

                self.untracked_patches = self.untracked_patches[:int(len(self.patches) * max_untracked_to_tracked_ratio)] if include_untracked else None
            else:
                self._h5_path = h5_path
                self._sequences = sequences
                # self._indices = np.where(mask)[0]
                self.patches = None

            # self._raw_labels = track_ids[mask]

        # Remap to contiguous 0-based labels
        # unique, inverse = np.unique(self._raw_labels, return_inverse=True)
        # self.labels = inverse.astype(np.int32)
        # self.n_classes = len(unique)
        if include_untracked:
            untracked_start_label = (self.labels[0] & 0xFFFF0000) + 0x00008000
            self.labels = np.concatenate([
                self.labels,
                np.arange(untracked_start_label, untracked_start_label + len(self.untracked_patches), dtype=self.labels.dtype)
            ])

        print(
            f"BlobTrackDataset: {len(self.labels)} patches"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.patches is not None:
            if idx >= self.patches.shape[0]:
                patch = self.untracked_patches[idx - self.patches.shape[0]]
            else:
                patch = self.patches[idx]  # (P, P) float32
        else:
            with h5py.File(self._h5_path, "r") as f:
                patches, _, _, affine_shape = _load_all_sequences(f, self._sequences)
                patch = patches[self._indices[idx]]

        patch = torch.from_numpy(patch).unsqueeze(0).transpose(1, 2)  # (1, P, P)
        label = self.labels[idx]
        if self.affine_shapes:
            affine_shape = torch.from_numpy(patch).unsqueeze(0).transpose(1, 2)
            return {
                "patch": patch,
                "label": label,
                "affine_shape": affine_shape
            }
        else:
            return {
                "patch": patch,
                "label": label,
            }


if __name__ == "__main__":
    track_file = "../BlobBoards.jl/out/only_absolute.tracks"
    train_sequences = []
    test_sequences = []
    val_sequences = []
    with h5py.File(track_file, "r") as f:
        for i, seq in enumerate(f["sequences"].keys()):
            board_id = re.fullmatch("\\w+_([A-Fa-f0-9]+)", seq)[1]

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