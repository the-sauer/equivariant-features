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

import juliacall    # Must be imported before pytorch  # noqa: F401

import dotenv
import hydra

from aef.data import get_dataset, get_validation_specs


@hydra.main(version_base=None, config_path="conf")
def main(cfg) -> None:
    """Build the dataset(s) a training run needs — thereby populating `cache_dir` —
    then exit without training.

    Sweep jobs share datasets (the cache key covers the dataset params, *not* the
    model), so building up front stops N jobs cold-building the same boards
    concurrently: the cache only turns warm *after* a build finishes, so jobs launched
    in parallel would each pay the full board-render + patch-extraction cost.

    `+prebuild_target=` selects what to build, so each dataset can be its own job:
      `all` (default) | `train` | a validation split name (e.g. `overall`, `small`).

    Needs a GPU: SIFT detection and patch extraction run on CUDA.
    """
    dotenv.load_dotenv()

    target = cfg.get("prebuild_target", "all") if "prebuild_target" in cfg else "all"

    if not cfg.training.dataset.params.get("cache_dir"):
        print("WARNING: training.dataset.params.cache_dir is unset — this run caches nothing")

    if target in ("all", "train"):
        train = get_dataset(cfg.training.dataset)
        print(f"training dataset(s): sizes={[len(d) for d in train]}")

    if target != "train":
        if target != "all":
            # Keep only the requested split so get_validation_specs builds just that
            # one (assigning an existing key is allowed under Hydra's struct mode).
            keep = [d for d in cfg.validation.datasets if d.get("name") == target]
            if not keep:
                names = [d.get("name") for d in cfg.validation.datasets]
                raise ValueError(f"unknown prebuild_target {target!r}; expected 'train'/'all' or one of {names}")
            cfg.validation.datasets = keep
        for label, datasets, _ in get_validation_specs(cfg):
            print(f"validation '{label or '-'}': sizes={[len(d) for d in datasets]}")

    print(f"prebuild done (target={target})")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
