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

import os

import omegaconf

try:
    from .blobboards import *
except ImportError as e:
    print("Could not import blobboards dataset. Make sure to install the BlobBoards.jl python bindings and to have the Julia package installed. Error was:", e)

try:
    # Track dataset (real-image track patches from a `.tracks` HDF5 file); needs only
    # h5py, no Julia bridge. Exported so `dataset.name: BlobTrackData` resolves here.
    from .track import BlobTrackData
except ImportError as e:
    print("Could not import track dataset:", e)


def get_dataset(dataset_cfg):
    if isinstance(dataset_cfg, omegaconf.ListConfig):
        return [get_dataset(c) for c in dataset_cfg]
    elif "foreach" in dataset_cfg:
        return [globals[dataset_cfg.name](**dataset_cfg.params, path=os.path.join(dataset_cfg.foreach, d)) for d in os.listdir(dataset_cfg.foreach)]
    else:
        return [eval(f"{dataset_cfg.name}(**dataset_cfg.params)")]


def get_validation_specs(cfg):
    """Resolve the validation config into ``(label, datasets, loss_cfgs)`` specs.

    Three config forms are supported:

    * **Shared-params** — ``cfg.validation.shared_params`` holds a single
      ``{name, params}`` dataset template and each ``cfg.validation.datasets``
      entry supplies only what differs (``params`` overrides + its own ``loss``).
      The template is deep-merged with the per-entry overrides, so the common
      board/patch settings live in one place instead of being repeated (or
      threaded through YAML anchors) across every split. An entry may still
      carry an explicit ``dataset`` block to opt out of the template entirely.
    * **Multi-dataset** — ``cfg.validation.datasets`` is a list of entries, each
      carrying its own full ``{name, dataset, loss}``. Every entry is paired with
      its own criterion and reported separately (metrics keyed ``<loss>@<name>``).
    * **Single-dataset (legacy)** — ``cfg.validation.dataset`` + ``cfg.validation.loss``.
      Returned as a single unlabelled spec, so existing configs are unchanged.
    """
    val = cfg.validation
    if getattr(val, "datasets", None) is not None:
        shared = getattr(val, "shared_params", None)
        specs = []
        for entry in val.datasets:
            label = entry.get("name", "")
            if getattr(entry, "dataset", None) is not None:
                dataset_cfg = entry.dataset
            elif shared is not None:
                # Deep-merge the shared template with this split's overrides.
                # ``entry.name`` is the split *label*, not the dataset class, so
                # only ``entry.params`` participates in the merge; the class name
                # comes from ``shared.name``.
                #
                # Resolve the template against the real config root first (baking
                # in interpolations like ${scale}) and rebuild it as a plain,
                # non-struct config: Hydra composes in struct mode, which would
                # otherwise reject a split's extra keys (e.g. scale_quantile_range).
                base = omegaconf.OmegaConf.create(
                    omegaconf.OmegaConf.to_container(shared, resolve=True)
                )
                entry_params = entry.get("params", None)
                override = (
                    omegaconf.OmegaConf.to_container(entry_params, resolve=True)
                    if entry_params is not None
                    else {}
                )
                dataset_cfg = omegaconf.OmegaConf.merge(
                    base, omegaconf.OmegaConf.create({"params": override})
                )
            else:
                raise ValueError(
                    f"validation.datasets entry {label!r} has no 'dataset' block "
                    "and no 'validation.shared_params' template to inherit from"
                )
            specs.append((label, get_dataset(dataset_cfg), entry.loss))
        return specs
    return [("", get_dataset(val.dataset), getattr(val, "loss", []))]
