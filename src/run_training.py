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

from datetime import datetime
import os
from typing import Any

import dotenv
import hydra
import kagglehub
import omegaconf

from aef.configuration import Config
from aef.data import HomographyData
from aef.data.blobboards import BlobBoardAbsoluteScaleData, BlobBoardHomographyData
from aef.data.colmap import ColmapDataset
from aef.data.constant import ConstantDataset
from aef.models import MODELS


def experiment_name_from_cfg(cfg: Config) -> str:
    # print(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), sort_keys=True))
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    # config_hash = hash(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), sort_keys=True))
    if isinstance(cfg.training.loss, str):
        loss_name = cfg.training.loss
    elif isinstance(cfg.training.loss, omegaconf.ListConfig):
        loss_name: str = "_".join(loss_cfg if isinstance(loss_cfg, str) else loss_cfg.name for loss_cfg in cfg.training.loss)
    else:   # isinstance(cfg.training.loss, omegaconf.DictConfig):
        loss_name = cfg.training.loss.name
    return f"{cfg.model.name}_{cfg.training.dataset.name}_{loss_name}_{date}"

def get_dataset(dataset_cfg):
    if dataset_cfg.name == "blobboards_homographic":
        return BlobBoardHomographyData(**dataset_cfg.params)
    elif dataset_cfg.name == "blobboards_absolute_scale":
        return BlobBoardAbsoluteScaleData(**dataset_cfg.params)
    elif dataset_cfg.name == "constant":
        return ConstantDataset(**dataset_cfg.params)
    else:
        data_dir = os.path.join(dataset_cfg.data_dir, dataset_cfg.name)
        if not os.path.exists(data_dir) or os.path.exists(os.path.join(data_dir, f"{dataset_cfg.name}.archive")):
            dataset_path = kagglehub.competition_download(dataset_cfg.name, output_dir=data_dir, force_download=True)
            return HomographyData(os.path.join(dataset_path, dataset_cfg.suffix), **dataset_cfg.params)
        elif os.path.exists(os.path.join(data_dir, "database.db")):
            return ColmapDataset(data_dir, **dataset_cfg.params)
        else:
            dataset_path = data_dir
            return HomographyData(os.path.join(dataset_path, dataset_cfg.suffix), **dataset_cfg.params)


def train(cfg) -> None:
    train_dataset = get_dataset(dataset_cfg=cfg.training.dataset)
    validation_dataset = get_dataset(dataset_cfg=cfg.validation.dataset)
    model_kwargs: dict[str, Any] = omegaconf.OmegaConf.to_container(cfg.model.params, resolve=True) if "params" in cfg.model else {}
    model_kwargs["in_channels"] = train_dataset.c
    Model, train_func = MODELS[cfg.model.name]
    model = Model(**model_kwargs)
    train_func(model=model, train_dataset=train_dataset, validation_dataset=validation_dataset, cfg=cfg, experiment_name=cfg.experiment_name if hasattr(cfg, "experiment_name") else experiment_name_from_cfg(cfg))


@hydra.main(version_base=None, config_path="conf", config_name="scale")
def main(cfg: Config):
    dotenv.load_dotenv()

    experiment_name = cfg.experiment_name if hasattr(cfg, "experiment_name") else experiment_name_from_cfg(cfg)
    print(f"Running experiment {experiment_name}")
    train(cfg)


if __name__ == "__main__":
    main() # pyright: ignore[reportCallIssue]
