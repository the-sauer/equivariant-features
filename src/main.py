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

import argparse
import logging
import os

import dotenv
import kagglehub

from aef.models.approach_one import AffineFeatureNetOne
from aef.data import HomographyData
from aef.models.scale import NeuralScaleSpace
from aef.train.descriptor import train as train_descriptor
from aef.train.scale import train_scale
from aef.data.blobboards import BlobBoardData

model_dict = {
    "scale_space_sesn": NeuralScaleSpace,
    "affine_feature_net_one": AffineFeatureNetOne,
}


def train(args, model_kwargs=None):
    data_dir = os.path.join("/home/hendrik/affine-equivariant-features/data", args.dataset)
    if args.dataset == "blobboards":
        dataset = BlobBoardData(args.num_boards)
    else:
        if not os.path.exists(data_dir):
            dataset_path = kagglehub.competition_download(args.dataset, output_dir=data_dir)
        else:
            dataset_path = data_dir
        dataset = HomographyData(dataset_path, in_memory=False)
    if model_kwargs is None:
        model_kwargs = {"in_channels": dataset.c}
    model = model_dict[args.model](**model_kwargs)
    if args.model.startswith("scale"):
        train_scale(model, dataset, batch_size=args.batch_size)
    else:
        train_descriptor(model, dataset, batch_size=args.batch_size)


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    trainparser = subparsers.add_parser("train", help="run training")
    trainparser.add_argument("model")
    trainparser.add_argument("dataset")
    trainparser.add_argument("--batch-size", default=100, type=int)
    trainparser.add_argument("--num-boards", default=100, type=int)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
