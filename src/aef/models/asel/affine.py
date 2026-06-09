from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import _weights_init, normalize, Diff, LambdaLayer


# Computing invariants of the affine group
def compute_affine_invariants(u, scales=None):
    if u.dim() == 4:
        u = u.unsqueeze(1)  # Add scale dimension of size 1 for uniform handling in Diff
    if scales is not None and u.shape[1] == 1:
        # Expand scale dimension if scales are provided but input has only one scale, TODO: Is this correct? Shouldn't
        # u also be scaled, consider adding a scaled u as an output of Diff
        u = u.expand(-1, len(scales), -1, -1, -1)
    diff = Diff(scales=scales).to(u.device)
    ux, uy, uxx, uyy, uxy = diff(u)

    uxx_uy = uxx * uy
    uxy_ux = uxy * ux
    uxy_uy = uxy * uy
    uyy_ux = uyy * ux

    inv2 = uxx * uyy - uxy * uxy
    inv3 = uxx_uy * uy - 2 * uxy_ux * uy + uyy_ux * ux
    inv4 = (
        uxx_uy[..., :-1, :, :] * uy[..., 1:, :, :]
        - uxy_ux[..., :-1, :, :] * uy[..., 1:, :, :]
        - uxy_uy[..., :-1, :, :] * ux[..., 1:, :, :]
        + uyy_ux[..., :-1, :, :] * ux[..., 1:, :, :]
    )
    inv5 = (
        uxx_uy[..., 1:, :, :] * uy[..., :-1, :, :]
        - uxy_ux[..., 1:, :, :] * uy[..., :-1, :, :]
        - uxy_uy[..., 1:, :, :] * ux[..., :-1, :, :]
        + uyy_ux[..., 1:, :, :] * ux[..., :-1, :, :]
    )
    inv6 = (
        ux[..., :-1, :, :] * uy[..., 1:, :, :] - uy[..., :-1, :, :] * ux[..., 1:, :, :]
    )

    inv1 = normalize(u)
    inv2 = normalize(inv2)
    inv345 = normalize(torch.cat((inv3, inv4, inv5), -3))
    if u.shape[-3] > 1:
        inv6 = normalize(inv6)

    return torch.cat((inv1, inv2, inv345, inv6), -3)


# EquivarLayer for the affine group
class EquivarLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hid_channels=None,
        type=["0", "0"],
        stride=1,
        scales=None,
        conv_layer: type[torch.nn.Module] = torch.nn.Conv2d,
        kernel_size=1,
        up_stride=1,
    ):
        super(EquivarLayer, self).__init__()
        self.in_channels = in_channels
        self.scales = scales
        num_inv = 6 * in_channels - 3
        if hid_channels is None:
            hid_channels = max(in_channels, out_channels)
        self.conv1 = conv_layer(num_inv, hid_channels, kernel_size=1)
        self.conv2 = conv_layer(
            hid_channels, out_channels, kernel_size=kernel_size, stride=up_stride
        )
        self.stride = stride
        self.type = type
        if self.type[1] == "c":
            self.eq_matrix_layer = EqMatrixLayer(
                in_channels, out_channels, scales=scales
            )

    def forward(self, x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]):
        if isinstance(x, tuple):
            x, s = x
        else:
            s = None

        assert torch.is_tensor(x), (
            "`x` must be a torch.Tensor or a tuple with tensor first element."
        )
        assert x.dim() in (4, 5), "`x` must be 4D (B,C,H,W) or 5D (B,S,C,H,W)."
        assert s is None or self.scales is None
        assert x.shape[-3] == self.in_channels, (
            f"Expected {self.in_channels} input channels, got {x.shape[-3]}."
        )

        inv = compute_affine_invariants(x, scales=self.scales)
        b, s_dim, num_inv, h, w = inv.shape

        inv = inv.view(b * s_dim, num_inv, h, w)
        inv = self.conv1(inv)
        inv = F.leaky_relu(inv)
        if self.stride == 2:
            inv = F.max_pool2d(inv, (2, 2))
        inv = self.conv2(inv)
        inv = inv.view(
            b, s_dim, -1, inv.shape[-2], inv.shape[-1]
        )  # Reshape back to (B, S, out_channels, H', W')

        if self.type[1] == "0":
            if s is not None:
                return inv, s
            else:
                return inv
        elif self.type[1] == "c":
            b, s_dim, _, h, w = inv.shape
            # inv_matrix = inv.view(b, s_dim, 2, 2, h, w)
            eq_matrix = self.eq_matrix_layer((x, s) if s is not None else x).view(
                b, s_dim, 2, 2, h, w
            )
            out_matrix = torch.einsum(
                "bsijkl,bsjnkl->bsinkl", eq_matrix, inv.view(b, s_dim, 2, 2, h, w)
            )

            if s is not None:
                return out_matrix, s
            else:
                return out_matrix


# Computing an equivariant matrix for the affine group
class EqMatrixLayer(nn.Module):
    def __init__(self, in_channels, out_channels=4, scales=None):
        super(EqMatrixLayer, self).__init__()
        self.in_channels = in_channels
        self.scales = scales
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels // 4, kernel_size=1, bias=False
        )

    def forward(self, x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]):
        if isinstance(x, tuple):
            x, Su = x
        else:
            Su = None

        b, s, _, h, w = x.shape

        diff = Diff(scales=self.scales).to(x.device)
        ux, uy, uxx, uyy, uxy = diff(x)

        relative_inv = uxx * uyy - uxy * uxy

        if Su is None and self.scales is None:
            Su = relative_inv.abs().view(x.shape[0], -1).max(dim=-1)[0]
            Su[Su == 0] = 1
            Su = Su.view(x.shape[0], 1, 1, 1, 1) ** 0.5

        if Su is None:
            eq_matrix11 = self.conv1((uxx * uy - uxy * ux).view(b * s, -1, h, w)).view(
                b, s, -1, h, w
            )
            eq_matrix12 = self.conv1(ux.view(b * s, -1, h, w)).view(b, s, -1, h, w)
            eq_matrix22 = self.conv1((uxy * uy - uyy * ux).view(b * s, -1, h, w)).view(
                b, s, -1, h, w
            )
            eq_matrix21 = self.conv1(uy.view(b * s, -1, h, w)).view(b, s, -1, h, w)
        else:
            eq_matrix11 = self.conv1(
                ((uxx * uy - uxy * ux) / Su).view(b * s, -1, h, w)
            ).view(b, s, -1, h, w)
            eq_matrix12 = self.conv1(ux.view(b * s, -1, h, w)).view(b, s, -1, h, w)
            eq_matrix22 = self.conv1(
                ((uxy * uy - uyy * ux) / Su).view(b * s, -1, h, w)
            ).view(b, s, -1, h, w)
            eq_matrix21 = self.conv1(uy.view(b * s, -1, h, w)).view(b, s, -1, h, w)

        return torch.cat((eq_matrix11, eq_matrix12, eq_matrix21, eq_matrix22), dim=2)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A", scales=None):
        super(BasicBlock, self).__init__()
        self.scales = scales
        self.conv1 = EquivarLayer(
            in_planes, planes, type=["0", "0"], stride=stride, scales=scales
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = EquivarLayer(
            planes, planes, type=["0", "0"], stride=1, scales=scales
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]):
        if isinstance(x, tuple):
            x, s = x
        else:
            s = None

        assert torch.is_tensor(x), "`x` must be a torch.Tensor."
        assert x.dim() == 5, "`x` must be 4D (B,C,H,W) or 5D (B,S,C,H,W)."

        # Handle batch norm for both 4D and 5D inputs
        is_5d = x.dim() == 5
        if is_5d:
            b, s_dim, c, h, w = x.shape
            out = self.conv1(x)
            # out shape: (B, S, C_out, H', W')
            b, s_dim, c_out, h, w = out.shape
            # Reshape for batch norm: (B, S, C, H, W) -> (B*S, C, H, W)
            out = out.view(b * s_dim, c_out, h, w)
            out = F.leaky_relu(self.bn1(out))
            out = out.view(b, s_dim, c_out, h, w)
            out = self.conv2(out)
            # out shape after conv2: (B, S, C_out2, H', W')
            b, s_dim, c_out2, h, w = out.shape
            out = out.view(b * s_dim, c_out2, h, w)
            out = self.bn2(out)
            out = out.view(b, s_dim, c_out2, h, w)
        else:
            out = F.leaky_relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

        if is_5d:
            b, s_dim, c_in, h_in, w_in = x.shape
            x_short = x.view(b * s_dim, c_in, h_in, w_in)
            short = self.shortcut(x_short)
            _, c_short, h_short, w_short = short.shape
            short = short.view(b, s_dim, c_short, h_short, w_short)
            out += short
        else:
            short = self.shortcut(x)
            out += short
        out = F.leaky_relu(out)
        if s is None:
            return out
        else:
            return out, s


class ScaleProjectionLayer(nn.Module):
    def __init__(self, scales, beta: float = 10):
        super().__init__()
        self.scales = torch.nn.Parameter(
            torch.Tensor(scales).view(1, -1, 1, 1, 1), requires_grad=False
        )
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6:
            x = x.view(x.shape[0], x.shape[1], 4, x.shape[4], x.shape[5])
        b, s, _, h, w = x.shape
        det = (
            x[:, :, 0, :, :] * x[:, :, 3, :, :] - x[:, :, 1, :, :] * x[:, :, 2, :, :]
        ).abs()
        det_onehot = (
            torch.nn.functional.one_hot(torch.argmax(det, dim=1), s)
            .float()
            .permute(0, 3, 1, 2)
        )
        det_soft = torch.nn.functional.softmax(self.beta * det, dim=1)
        det_weight = det_onehot + det_soft - det_soft.detach()
        pool_x = torch.sum(det_weight.unsqueeze(2) * x * self.scales, dim=1)
        result = pool_x.view(b, 2, 2, w, h)
        return result


class LearnedSaliencyLayer(nn.Module):
    def __init__(self, scales, beta: float = 10):
        super().__init__()
        self.conv = torch.nn.Conv2d(4 * len(scales), len(scales), kernel_size=1)
        self.scales = torch.nn.Parameter(
            torch.Tensor(scales).view(1, -1, 1, 1, 1), requires_grad=False
        )
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6:
            x = x.view(x.shape[0], x.shape[1], 4, x.shape[4], x.shape[5])
        b, s, c, h, w = x.shape
        saliency_map = self.conv(x.reshape(b, s * 4, h, w)).reshape(b, s, h, w)
        saliency_onehot = (
            torch.nn.functional.one_hot(torch.argmax(saliency_map, dim=1), s)
            .float()
            .permute(0, 3, 1, 2)
        )
        saliency_soft = torch.nn.functional.softmax(self.beta * saliency_map, dim=1)
        saliency_weight = saliency_onehot + saliency_soft - saliency_soft.detach()
        pool_x = torch.sum(saliency_weight.unsqueeze(2) * x * self.scales, dim=1)
        result = pool_x.view(b, 2, 2, w, h)
        return result


class EquivarLayer_affine_resnet32(nn.Module):
    def __init__(
        self,
        in_shape,
        beta=10,
        block=BasicBlock,
        num_blocks=[5, 5, 5],
        scales=None,
    ):
        super(EquivarLayer_affine_resnet32, self).__init__()
        self.beta = beta
        self.scales = scales
        self.in_planes = 16
        self.conv1 = EquivarLayer(
            in_shape[0], 16, type=["0", "0"], stride=1, scales=scales
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.outlayer = EquivarLayer(64, 4, type=["0", "c"], scales=scales)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, scales=self.scales))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def det_pool(self, x, s_dim=None):
        # x is always 4D: (B*S, 4, H, W) or (B, 4, H, W)
        assert x.dim() == 4, (
            "`det_pool` expects a 4D tensor shaped (B,4,H,W) or (B*S,4,H,W)."
        )
        assert x.shape[1] == 4, (
            "`det_pool` expects channel dimension to be 4 (flattened 2x2 matrix)."
        )
        b, _, h, w = x.shape
        det = (x[:, 0, :, :] * x[:, 3, :, :] - x[:, 1, :, :] * x[:, 2, :, :]).abs()
        det_onehot = (
            torch.nn.functional.one_hot(torch.argmax(det.view(b, -1), dim=-1), h * w)
            .view(b, h, w)
            .float()
        )
        det_soft = torch.nn.functional.softmax(
            self.beta * det.view(b, -1), dim=-1
        ).view(b, h, w)
        det_weight = det_onehot + det_soft - det_soft.detach()
        pool_x = torch.sum(det_weight.unsqueeze(1) * x, dim=(-2, -1))
        result = pool_x.view(b, 2, 2)
        if s_dim is not None:
            assert s_dim > 0, "`s_dim` must be >= 1 when provided."
            assert b % s_dim == 0, "`b` must be divisible by `s_dim` in det_pool."
            result = result.view(b // s_dim, s_dim, 2, 2)
        return result

    def forward(self, x):
        assert torch.is_tensor(x), "`x` must be a torch.Tensor."
        assert x.dim() in (4, 5), "Model input must be 4D (B,C,H,W) or 5D (B,S,C,H,W)."
        if x.dim() == 4:
            assert x.shape[1] == self.conv1.in_channels, (
                f"Expected input channels {self.conv1.in_channels}, got {x.shape[1]}."
            )
        else:
            assert x.shape[2] == self.conv1.in_channels, (
                f"Expected input channels {self.conv1.in_channels}, got {x.shape[2]}."
            )
        is_5d = x.dim() == 5
        s_dim = None
        if is_5d:
            b, s_dim, c, h, w = x.shape

        out = self.conv1(x)
        if is_5d:
            b, s_dim, c, h, w = out.shape
            out = out.view(b * s_dim, c, h, w)
            out = F.relu(self.bn1(out))
            out = out.view(b, s_dim, c, h, w)
        else:
            out = F.relu(self.bn1(out))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out_matrix = self.outlayer(out)

        # outlayer returns (B, 2, 2, H, W) when projected (S=1 for type='c'),
        # or (B, S, 2, 2, H, W) when no projection is applied.
        if out_matrix.dim() == 6:
            b, s_dim, _, _, h, w = out_matrix.shape
            out_matrix = out_matrix.view(b * s_dim, 4, h, w)
        else:
            b, _, _, h, w = out_matrix.shape
            out_matrix = out_matrix.view(b, 4, h, w)

        out_matrix = self.det_pool(out_matrix, s_dim=s_dim)

        return out_matrix.transpose(-2, -1)
