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
class OptimizerConfig:
    name: str
    params: dict[str, Any]
    scheduler: Optional[ComponentConfig]


@dataclass
class TrainingConfig(TrainValConfig):
    augmentation: Optional[AugmentationConfig]
    loss: str | LossConfig | list[str | LossConfig]
    optimizer: str | OptimizerConfig | dict[str, OptimizerConfig]
    # Weight of the learned-mask supervision loss (process_batch_blobs); only used
    # when the model is mask-aware. Read via getattr with a 1.0 default, so omitting
    # it changes nothing for existing configs.
    mask_loss_weight: float = 1.0
    # Force the mask path off for a mask-aware model (``learned_mask=True``): it is
    # then called as ``model(patches)`` and gets no GT mask / `is_pdf` flag. The ablation
    # switch, and what lets a mask-aware model run on a dataset without masks. Models
    # that are not mask-aware never see the mask kwargs regardless.
    ignore_mask: bool = False


@dataclass
class ValidationConfig:
    # Everything else is defined in TrainValConfig.
    #
    # Run the full validation suite every N *training* batches, in addition to the
    # end-of-epoch run, so the metrics can be watched at sub-epoch resolution. Those
    # extra points are plotted at the fractional epoch they were measured at
    # (`epoch + batches_done / batches_per_epoch`). `null`/0 (the default) keeps the
    # historical behaviour: one validation per epoch, plotted at integer epochs.
    # Checkpoint selection is unaffected — `best.pth` still compares epochs only.
    validate_every_n_batches: Optional[int] = None


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
