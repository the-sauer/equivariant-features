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

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TransformParams:
    scale: bool
    translate: bool
    rotate: bool


@dataclass
class DatasetParams:
    image_size: tuple[int, int]
    in_memory: bool
    transform_params: TransformParams


@dataclass
class FeatureSampling:
    num_features: Optional[int]
    stride: Optional[int]
    detector: Optional[str]


@dataclass
class DatasetConfig:
    name: str
    data_dir: str
    suffix: str
    params: DatasetParams


@dataclass
class TrainValConfig:
    batch_size: int
    dataset: DatasetConfig
    feature_sampling: FeatureSampling


@dataclass
class AugmentationConfig:
    color_jitter: dict
    gaussian_blur: dict
    gaussian_noise: dict


@dataclass
class ComponentConfig:
    name: str
    params: dict[str, Any]


@dataclass
class LossConfig:
    name: str
    params: Optional[dict[str, Any]]
    weight: Optional[float]


@dataclass
class TrainingConfig(TrainValConfig):
    augmentation: Optional[AugmentationConfig]
    loss: str | LossConfig | list[str | LossConfig]
    optimizer: ComponentConfig


@dataclass
class ValidationConfig:
    pass    # As of now everything is defined in TrainValConfig


@dataclass
class LoggingConfig:
    dir: str
    interval: int


@dataclass
class Config:
    training: TrainingConfig
    validation: ValidationConfig
    model: ComponentConfig
    logging: LoggingConfig
