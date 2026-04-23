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

import juliacall

from datetime import datetime
import os
from typing import Any

import dotenv
import hydra
import kagglehub
import omegaconf 

from aef.data import HomographyData
from aef.data.blobboards import BlobBoardData
from aef.models.approach_one import AffineFeatureNetOne
from aef.models.scale import NeuralScaleSpace
from aef.train.descriptor import train as train_descriptor
from aef.train.scale import train_scale


MODELS = {
    "scale_space_sesn": (NeuralScaleSpace, train_scale),
    "affine_feature_net_one": (AffineFeatureNetOne, train_descriptor),
}


def experiment_name_from_cfg(cfg):
    # print(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), sort_keys=True))
    date = datetime.today().strftime('%Y_%m_%d')
    # config_hash = hash(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), sort_keys=True))
    return f"{date}_{cfg.model.name}_{cfg.training.dataset.name}_{cfg.training.loss}"


def get_dataset(dataset_cfg):
    if dataset_cfg.name == "blobboards":
        return BlobBoardData(dataset_cfg.num_boards)
    else:
        data_dir = os.path.join("/home/hendrik/affine-equivariant-features/data", dataset_cfg.name)
        if not os.path.exists(data_dir):
            dataset_path = kagglehub.competition_download(dataset_cfg.name, output_dir=data_dir)
        else:
            dataset_path = data_dir
        return HomographyData(os.path.join(dataset_path, dataset_cfg.suffix), **dataset_cfg.params)


def train(cfg):
    train_dataset = get_dataset(dataset_cfg=cfg.training.dataset)
    validation_dataset = get_dataset(dataset_cfg=cfg.validation.dataset)
    model_kwargs: dict[str, Any] = omegaconf.OmegaConf.to_container(cfg.model.params, resolve=True)     # type: ignore
    model_kwargs["in_channels"] = train_dataset.c
    Model, train_func = MODELS[cfg.model.name]
    model = Model(**model_kwargs)
    train_func(model, train_dataset, validation_dataset, cfg, experiment_name=experiment_name_from_cfg(cfg))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    dotenv.load_dotenv()

    experiment_name = experiment_name_from_cfg(cfg)
    print(f"Running experiment {experiment_name}")
    train(cfg)


if __name__ == "__main__":
    main()
