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
import escnn.nn as nn
import torch
import torch.nn.functional as F

from .utils import L2Norm


class AbstractBlobDescriptor(torch.nn.Module, abc.ABC):
    net: torch.nn.Module
    field_type: escnn.nn.FieldType

    def __init__(self, n_rotations=8, **_):
        super().__init__()
        self.s = escnn.gspaces.rot2dOnR2(n_rotations)
        self.field_type = escnn.nn.FieldType(self.s, 1 * [self.s.trivial_repr])


class BlobDescriptorHardNet(AbstractBlobDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c_in = escnn.nn.FieldType(self.s, 1 * [self.s.trivial_repr])
        c_hid1 = escnn.nn.FieldType(self.s, 32 * [self.s.regular_repr])
        c_hid2 = escnn.nn.FieldType(self.s, 64 * [self.s.regular_repr])
        c_out = escnn.nn.FieldType(self.s, 128 * [self.s.trivial_repr])
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
        c_in = escnn.nn.FieldType(self.s, 1 * [self.s.trivial_repr])
        c_hid1 = escnn.nn.FieldType(self.s, 32 * [self.s.regular_repr])
        c_hid2 = escnn.nn.FieldType(self.s, 64 * [self.s.regular_repr])
        c_out = escnn.nn.FieldType(self.s, 128 * [self.s.trivial_repr])
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

        c_in = escnn.nn.FieldType(self.s, 1 * [self.s.trivial_repr])
        c_hid1 = escnn.nn.FieldType(self.s, 32 * [self.s.regular_repr])
        c_hid2 = escnn.nn.FieldType(self.s, 64 * [self.s.regular_repr])
        c_hid3 = escnn.nn.FieldType(self.s, 128 * [self.s.regular_repr])
        # Keep the final features in the regular representation!
        c_out = escnn.nn.FieldType(self.s, 128 * [self.s.regular_repr])

        self.net = escnn.nn.SequentialModule(
            # Apply a mask to zero-out the corners of the square grid
            escnn.nn.MaskModule(c_in, 64, margin=1),
            escnn.nn.R2Conv(c_in, c_hid1, 3),  # 64x64 => 62x62
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),
            escnn.nn.FieldDropout(c_hid1, p=0.1),
            escnn.nn.R2Conv(c_hid1, c_hid1, 3, bias=False),  # 62x62 => 60x60
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),
            escnn.nn.FieldDropout(c_hid1, p=0.1),
            escnn.nn.R2Conv(c_hid1, c_hid1, 5, bias=False),  # 60x60 => 56x56
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1),
            escnn.nn.FieldDropout(c_hid1, p=0.1),
            escnn.nn.R2Conv(c_hid1, c_hid2, 5, bias=False),  # 56x56 => 52x52
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),
            escnn.nn.FieldDropout(c_hid2, p=0.1),
            escnn.nn.R2Conv(c_hid2, c_hid2, 7, bias=False),  # 52x52 => 46x46
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),
            escnn.nn.FieldDropout(c_hid2, p=0.1),
            escnn.nn.R2Conv(c_hid2, c_hid2, 7, bias=False),  # 46x46 => 40x40
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2),
            escnn.nn.FieldDropout(c_hid2, p=0.1),
            escnn.nn.R2Conv(c_hid2, c_hid3, 9, bias=False),  # 40x40 => 32x32
            escnn.nn.InnerBatchNorm(c_hid3),
            escnn.nn.ReLU(c_hid3),
            escnn.nn.FieldDropout(c_hid3, p=0.1),
            escnn.nn.R2Conv(c_hid3, c_hid3, 9, bias=False),  # 32x32 => 24x24
            escnn.nn.InnerBatchNorm(c_hid3),
            escnn.nn.ReLU(c_hid3),
            escnn.nn.FieldDropout(c_hid3, p=0.1),
            escnn.nn.R2Conv(c_hid3, c_out, 24, bias=False),
            escnn.nn.GroupPooling(c_out),
        )

    def forward(self, x):
        x = escnn.nn.GeometricTensor(x, self.field_type)
        x = self.net(x)
        x = x.tensor
        return x

        # 3. Spatial collapse (24x24 => 1x1)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # 4. Flatten to 1D vector (Batch, 128)
        x = x.view(x.size(0), -1)

        # 5. Normalize
        x = F.normalize(x, p=2, dim=1)

        return x


class BlobDescriptorRobust(torch.nn.Module):
    def __init__(self, in_channels=1, out_dim=128, **_):
        super().__init__()

        # C8 symmetry (rotations by 45 degrees)
        self.s = escnn.gspaces.rot2dOnR2(N=8)

        # Representations
        c_in = escnn.nn.FieldType(self.s, in_channels * [self.s.trivial_repr])
        c_hid1 = escnn.nn.FieldType(self.s, 32 * [self.s.regular_repr])
        c_hid2 = escnn.nn.FieldType(self.s, 64 * [self.s.regular_repr])
        c_hid3 = escnn.nn.FieldType(self.s, out_dim * [self.s.regular_repr])

        # NEU: Wir definieren, dass die Filterringe nur bis zum maximalen Kreis-Radius (kernel_size//2) gehen dürfen.
        # Das eliminiert die Ecken des 5x5 oder 9x9 Grids aus den Gewichten.

        self.block1 = escnn.nn.SequentialModule(
            # Verwende rings=['U'] (nur Uniforme Ringe) oder explizit Radien, um eckige Filter zu verbieten.
            # escnn generiert standardmäßig auch Filter für die "Ecken" der Bounding Box.
            # Durch 'rings' oder 'maximum_frequency' begrenzen wir das auf reine Kreise.
            escnn.nn.R2Conv(
                c_in, c_hid1, kernel_size=5, bias=False, rings=[0.0, 1.0, 2.0]
            ),
            escnn.nn.MaskModule(c_hid1, 60, margin=1),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1, inplace=True),
        )

        self.block2 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid1, c_hid1, kernel_size=5, bias=False, rings=[0.0, 1.0, 2.0]
            ),
            escnn.nn.MaskModule(c_hid1, 56, margin=1),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1, inplace=True),
        )

        self.block3 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid1, c_hid1, kernel_size=5, bias=False, rings=[0.0, 1.0, 2.0]
            ),
            escnn.nn.MaskModule(c_hid1, 52, margin=1),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1, inplace=True),
        )

        self.block4 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid1, c_hid2, kernel_size=5, bias=False, rings=[0.0, 1.0, 2.0]
            ),
            escnn.nn.MaskModule(c_hid2, 48, margin=1),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2, inplace=True),
        )

        self.block5 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid2,
                c_hid2,
                kernel_size=9,
                bias=False,
                rings=[0.0, 1.0, 2.0, 3.0, 4.0],
            ),
            escnn.nn.MaskModule(c_hid2, 40, margin=1),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2, inplace=True),
        )

        self.block6 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid2,
                c_hid2,
                kernel_size=9,
                bias=False,
                rings=[0.0, 1.0, 2.0, 3.0, 4.0],
            ),
            escnn.nn.MaskModule(c_hid2, 32, margin=1),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2, inplace=True),
        )

        self.block7 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid2,
                c_hid3,
                kernel_size=9,
                bias=False,
                rings=[0.0, 1.0, 2.0, 3.0, 4.0],
            ),
            escnn.nn.MaskModule(c_hid3, 24, margin=1),
            escnn.nn.InnerBatchNorm(c_hid3),
            escnn.nn.ReLU(c_hid3, inplace=True),
        )

        # We do NOT group pool yet. We let the PyTorch unwrap happen first.
        self.in_type = c_in

    def forward(self, x):
        # 1. Equivariant convolutions with strict masking
        x = escnn.nn.GeometricTensor(x, self.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        # 2. Extract standard tensor
        # Shape: [Batch, Channels * Group_Size, 24, 24]
        # For C8 and out_dim=128, shape is [Batch, 128 * 8, 24, 24]
        x = x.tensor

        # 3. Spatial Pooling FIRST
        # Shape becomes [Batch, 1024, 1, 1] -> flatten to [Batch, 1024]
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

        # 4. Rotational Group Pooling (Invariant collapse)
        # We manually reshape to separate Channels and Rotations
        # Shape: [Batch, Channels, Group_Size]
        x = x.view(x.size(0), 128, 8)

        # Max pool across the rotation dimension (dim=2)
        # This gives us rotation invariance.
        # Shape becomes [Batch, 128]
        x, _ = torch.max(x, dim=2)

        # 5. Normalize for metric learning (SupCon/Triplet)
        x = F.normalize(x, p=2, dim=1)

        return x


class BlobDescriptorHierarchical(torch.nn.Module):
    # Ändere einfach den Default-Wert auf 3, oder übergib in_channels=3 beim Instanziieren
    def __init__(
        self,
        in_channels=7,
        out_dim=128,
        sigma_cutoff=2.0,
        scale_factors=[4.0, 8.0, 16.0, 32.0, 64.0, 96.0, 128.0],
        patch_size=64,
        **_
    ):
        super().__init__()

        # --- NEU: Inner Masking (Donut Mask) ---
        # Wir berechnen für jeden Kanal (jede Skala) den Pixel-Radius,
        # der 'sigma_cutoff' entspricht, und nullen diesen Bereich.
        y, x = torch.meshgrid(
            torch.arange(patch_size), torch.arange(patch_size), indexing="ij"
        )
        center = (patch_size - 1) / 2.0
        dist_sq = (x - center) ** 2 + (y - center) ** 2

        inner_mask = torch.ones((1, in_channels, patch_size, patch_size))
        for i, sf in enumerate(scale_factors):
            # sf entspricht dem Gesamtradius des Patches (patch_size / 2)
            # 1 sigma = (patch_size / 2) / sf (in Pixeln)
            r_pixel = sigma_cutoff * (patch_size / 2.0) / sf

            # Alles innerhalb dieses Radius wird auf 0 gesetzt
            inner_mask[0, i, dist_sq <= r_pixel**2] = 0.0

        # register_buffer sorgt dafür, dass die Maske automatisch auf die GPU
        # geschoben wird, ohne dass sie trainierbare Parameter (Gradients) bekommt
        self.register_buffer("inner_mask", inner_mask)

        # C8 symmetry (rotations by 45 degrees)
        self.s = escnn.gspaces.rot2dOnR2(N=8)

        # Representations
        # Hier passiert die Magie: escnn behandelt die 3 Kanäle automatisch als
        # 3 unabhängige "Trivial Representations" (also skalare Felder, die mitsamt
        # dem Grid rotieren, aber ihre Farbwerte nicht mischen).
        c_in = escnn.nn.FieldType(self.s, in_channels * [self.s.trivial_repr])

        # We assign specific channel depths for our 3 extraction points
        # To keep the final concatenated descriptor size manageable, we use 32, 48, and 48 (Total = 128)
        c_hid1 = escnn.nn.FieldType(self.s, 32 * [self.s.regular_repr])
        c_hid2 = escnn.nn.FieldType(self.s, 48 * [self.s.regular_repr])
        c_hid3 = escnn.nn.FieldType(self.s, 48 * [self.s.regular_repr])

        self.mask_in = escnn.nn.MaskModule(c_in, 64, margin=1)

        self.block1 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_in, c_hid1, kernel_size=5, padding=0, rings=[0.0, 1.0, 2.0]
            ),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1, inplace=True),
            escnn.nn.MaskModule(c_hid1, 60, margin=1),
        )

        self.block2 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid1, c_hid1, kernel_size=5, bias=False, rings=[0.0, 1.0, 2.0]
            ),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1, inplace=True),
            escnn.nn.MaskModule(c_hid1, 56, margin=1),
        )

        self.block3 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid1, c_hid1, kernel_size=5, bias=False, rings=[0.0, 1.0, 2.0]
            ),
            escnn.nn.InnerBatchNorm(c_hid1),
            escnn.nn.ReLU(c_hid1, inplace=True),
            escnn.nn.MaskModule(c_hid1, 52, margin=1),
        )

        self.block4 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid1, c_hid2, kernel_size=5, bias=False, rings=[0.0, 1.0, 2.0]
            ),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2, inplace=True),
            escnn.nn.MaskModule(c_hid2, 48, margin=1),
        )

        self.block5 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid2,
                c_hid2,
                kernel_size=9,
                bias=False,
                rings=[0.0, 1.0, 2.0, 3.0, 4.0],
            ),
            escnn.nn.InnerBatchNorm(c_hid2),
            escnn.nn.ReLU(c_hid2, inplace=True),
            escnn.nn.MaskModule(c_hid2, 40, margin=1),
        )

        self.block6 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid2,
                c_hid3,
                kernel_size=9,
                bias=False,
                rings=[0.0, 1.0, 2.0, 3.0, 4.0],
            ),
            escnn.nn.InnerBatchNorm(c_hid3),
            escnn.nn.ReLU(c_hid3, inplace=True),
            escnn.nn.MaskModule(c_hid3, 32, margin=1),
        )

        self.block7 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(
                c_hid3,
                c_hid3,
                kernel_size=9,
                bias=False,
                rings=[0.0, 1.0, 2.0, 3.0, 4.0],
            ),
            escnn.nn.InnerBatchNorm(c_hid3),
            escnn.nn.ReLU(c_hid3, inplace=True),
            escnn.nn.MaskModule(c_hid3, 24, margin=1),
        )
        self.dropout1 = escnn.nn.FieldDropout(c_hid1, p=0.1)
        self.dropout2 = escnn.nn.FieldDropout(c_hid2, p=0.1)
        self.dropout3 = escnn.nn.FieldDropout(c_hid3, p=0.1)

        # NEU: Attention-Layer für jede Skalen-Ebene.
        # Output ist eine "Triviale Repräsentation" (ein skalarer Wert pro Pixel, der sich bei Rotation nicht ändert)
        c_att = escnn.nn.FieldType(self.s, 1 * [self.s.trivial_repr])
        self.att1 = escnn.nn.R2Conv(c_hid1, c_att, kernel_size=1)
        self.att2 = escnn.nn.R2Conv(c_hid2, c_att, kernel_size=1)
        self.att3 = escnn.nn.R2Conv(c_hid3, c_att, kernel_size=1)

        self.pool1 = escnn.nn.GroupPooling(c_hid1)  # For shallow features
        self.pool2 = escnn.nn.GroupPooling(c_hid2)  # For middle features
        self.pool3 = escnn.nn.GroupPooling(c_hid3)  # For deep features

        self.in_type = c_in

    def forward(self, x):
        # --- NEU: Zentrums-Maskierung anwenden ---
        # Blendet den nutzlosen Anchor-Blob aus (Element-weise Multiplikation)
        x = x * self.inner_mask

        x = escnn.nn.GeometricTensor(x, self.in_type)
        x = self.mask_in(x)

        # 1. Shallow Stage (Extract local neighborhood features)
        x = self.block1(x)
        x = self.block2(x)
        x_shallow = self.block3(x)

        # 2. Middle Stage (Extract mid-range features)
        x = self.block4(x_shallow)
        x_mid = self.block5(x)

        # 3. Deep Stage (Extract global contextual features)
        x = self.block6(x_mid)
        x_deep = self.block7(x)

        x_shallow = self.dropout1(x_shallow)
        x_mid = self.dropout2(x_mid)
        x_deep = self.dropout3(x_deep)

        # --- NEU: Attention-Weighted Pooling ---
        # 1. Generiere räumliche Attention-Maps (1 Kanal, Werte zwischen 0 und 1)
        # tensor shape: [Batch, 1, H, W]
        a_shallow = torch.sigmoid(self.att1(x_shallow).tensor)
        a_mid = torch.sigmoid(self.att2(x_mid).tensor)
        a_deep = torch.sigmoid(self.att3(x_deep).tensor)

        # 2. Rotiere die Features in den invarianten Raum (Group Pooling wie zuvor)
        out_shallow = self.pool1(x_shallow).tensor
        out_mid = self.pool2(x_mid).tensor
        out_deep = self.pool3(x_deep).tensor

        # 3. Wende die Spatial Attention an (Gewichtetes Global Average Pooling)
        # Statt eines dummen Durchschnitts gewichten wir Pixel mit hoher Salienz stärker.
        out_shallow = (out_shallow * a_shallow).sum(dim=(2, 3)) / (
            a_shallow.sum(dim=(2, 3)) + 1e-6
        )
        out_mid = (out_mid * a_mid).sum(dim=(2, 3)) / (a_mid.sum(dim=(2, 3)) + 1e-6)
        out_deep = (out_deep * a_deep).sum(dim=(2, 3)) / (a_deep.sum(dim=(2, 3)) + 1e-6)

        # Concatenate: [Batch, 32] + [Batch, 48] + [Batch, 48] = [Batch, 128]
        final_descriptor = torch.cat([out_shallow, out_mid, out_deep], dim=1)

        # Normalize the combined multi-scale descriptor to the unit sphere
        final_descriptor = F.normalize(final_descriptor, p=2, dim=1)

        return final_descriptor
