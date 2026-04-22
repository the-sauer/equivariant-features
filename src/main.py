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

from datetime import datetime
import json
import logging
import os

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
    "scale_space_sesn": NeuralScaleSpace,
    "affine_feature_net_one": AffineFeatureNetOne,
}


def experiment_name_from_cfg(cfg):
    print(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), sort_keys=True))
    date = datetime.today().strftime('%Y-%m-%d')
    # config_hash = hash(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), sort_keys=True))
    return f"{date}_{cfg.model.name}_{cfg.training.dataset.name}_{cfg.training.loss}"


def train(cfg):
    if cfg.training.dataset.name == "blobboards":
        dataset = BlobBoardData(cfg.training.num_boards)
    else:
        data_dir = os.path.join("/home/hendrik/affine-equivariant-features/data", cfg.training.dataset.name)
        if not os.path.exists(data_dir):
            dataset_path = kagglehub.competition_download(cfg.training.dataset.name, output_dir=data_dir)
        else:
            dataset_path = data_dir
        dataset = HomographyData(dataset_path, in_memory=False)
    model_kwargs = omegaconf.OmegaConf.to_container(cfg.model.params, resolve=True)
    model_kwargs["in_channels"] = dataset.c
    model = MODELS[cfg.model.name](**model_kwargs)
    if cfg.model.name.startswith("scale"):
        train_scale(model, dataset, cfg, experiment_name=experiment_name_from_cfg(cfg))
    else:
        train_descriptor(model, dataset, cfg, experiment_name=experiment_name_from_cfg(cfg))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO)

    experiment_name = experiment_name_from_cfg(cfg)
    print(f"Running experiment {experiment_name}")
    train(cfg)


if __name__ == "__main__":
    main()
