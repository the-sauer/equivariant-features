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

import abc

import escnn
import torch

from .utils import L2Norm


class AbstractBlobDescriptor(torch.nn.Module, abc.ABC):
    net: torch.nn.Module
    field_type: escnn.nn.FieldType

    def __init__(self, n_rotations=8, **_):
        super().__init__()
        self.s = escnn.gspaces.rot2dOnR2(n_rotations)
        self.field_type = escnn.nn.FieldType(self.s, 1*[self.s.trivial_repr])


class BlobDescriptorHardNet(AbstractBlobDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c_in = escnn.nn.FieldType(self.s, 1*[self.s.trivial_repr])
        c_hid1 = escnn.nn.FieldType(self.s, 32*[self.s.regular_repr])
        c_hid2 = escnn.nn.FieldType(self.s, 64*[self.s.regular_repr])
        c_out = escnn.nn.FieldType(self.s, 128*[self.s.trivial_repr])
        self.net = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(c_in, c_hid1, 5, padding=2, bias=False),     
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),
            escnn.nn.R2Conv(c_hid1, c_hid1, 5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),
            escnn.nn.R2Conv(c_hid1, c_hid2, 5, padding=2, bias=False, stride=2),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),

            escnn.nn.R2Conv(c_hid2, c_hid2, 5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),
            escnn.nn.R2Conv(c_hid2, c_out, 5, padding=2, bias=False, stride=2),
            escnn.nn.InnerBatchNorm(c_out),
            escnn.nn.ReLU(c_out),

            escnn.nn.R2Conv(c_out, c_out, 16, bias=False),
            escnn.nn.GroupPooling(c_out),
        )

    def forward(self, x):
        x = escnn.nn.GeometricTensor(x, self.field_type)
        x = self.net(x)
        x = x.tensor
        x = L2Norm()(x)
        return x


class BlobDescriptorDeep(AbstractBlobDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c_in = escnn.nn.FieldType(self.s, 1*[self.s.trivial_repr])
        c_hid1 = escnn.nn.FieldType(self.s, 32*[self.s.regular_repr])
        c_hid2 = escnn.nn.FieldType(self.s, 64*[self.s.regular_repr])
        c_out = escnn.nn.FieldType(self.s, 128*[self.s.trivial_repr])
        self.net = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(c_in, c_hid1, 5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),
            escnn.nn.R2Conv(c_hid1, c_hid1, 5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),
            escnn.nn.R2Conv(c_hid1, c_hid1, 5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),
            escnn.nn.R2Conv(c_hid1, c_hid2, 5, padding=2, bias=False, stride=2),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),

            escnn.nn.R2Conv(c_hid2, c_hid2, 5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),
            escnn.nn.R2Conv(c_hid2, c_hid2, 5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),
            escnn.nn.R2Conv(c_hid2, c_out, 5, padding=2, bias=False, stride=2),
            escnn.nn.InnerBatchNorm(c_out),
            escnn.nn.ReLU(c_out),

            escnn.nn.R2Conv(c_out, c_out, 16, bias=False),
            escnn.nn.GroupPooling(c_out),
        )

    def forward(self, x):
        x = escnn.nn.GeometricTensor(x, self.field_type)
        x = self.net(x)
        x = x.tensor
        x = L2Norm()(x)
        return x


class BlobDescriptorNoStride(AbstractBlobDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c_in = escnn.nn.FieldType(self.s, 1*[self.s.trivial_repr])
        c_hid1 = escnn.nn.FieldType(self.s, 32*[self.s.regular_repr])
        c_hid2 = escnn.nn.FieldType(self.s, 64*[self.s.regular_repr])
        c_out = escnn.nn.FieldType(self.s, 128*[self.s.trivial_repr])
        self.net = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(c_in, c_hid1, 5),                   # 64x64 => 60x60
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),

            escnn.nn.R2Conv(c_hid1, c_hid1, 5, bias=False),     # 60x60 => 56x56
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),

            escnn.nn.R2Conv(c_hid1, c_hid1, 5, bias=False),     # 56x56 => 52x52
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),

            escnn.nn.R2Conv(c_hid1, c_hid2, 5, bias=False),     # 52x52 => 48x48
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),

            escnn.nn.R2Conv(c_hid2, c_hid2, 9, bias=False),     # 48x48 => 40x40
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),

            escnn.nn.R2Conv(c_hid2, c_hid2, 9, bias=False),     # 40x40 => 32x32
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),

            escnn.nn.R2Conv(c_hid2, c_out, 9, bias=False),      # 32x32 => 24x24
            escnn.nn.InnerBatchNorm(c_out),
            escnn.nn.ReLU(c_out),

            escnn.nn.R2Conv(c_out, c_out, 24, bias=False),
            escnn.nn.GroupPooling(c_out),
        )

    def forward(self, x):
        x = escnn.nn.GeometricTensor(x, self.field_type)
        x = self.net(x)
        x = x.tensor
        x = L2Norm()(x)
        return x
