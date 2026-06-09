from importlib import resources

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
try:
    from rbf.basis import get_rbf  # If you have trouble installing rbf, you can comment out this line and load precomputed kernels from local files.
    rbf_available = True
except ImportError:
    rbf_available = False


def normalize(inv):
    if inv.dim() == 4:
        batch = inv.shape[0]
        max_values = inv.abs().view(batch, -1).max(dim=-1)[0]
        max_values[max_values == 0] = 1
        inv = inv / max_values.view(batch, 1, 1, 1)
        return inv
    else:
        assert inv.dim() == 5
        batch = inv.shape[0]
        scales = inv.shape[1]
        max_values = inv.abs().view(batch, scales, -1).max(dim=-1)[0]
        max_values[max_values == 0] = 1
        inv = inv / max_values.view(batch, scales, 1, 1, 1)
        return inv


class Diff(nn.Module):
    def __init__(self, ord=2, use_rbf=True, scales=None):
        super(Diff, self).__init__()
        self.ord = ord

        # Build derivative kernels with an explicit scale axis S.
        if scales is not None:
            assert rbf_available, "RBF is required to generate multi-scale kernels. Please install rbf."
            kernel_size = int(max(scales) * 7)  # TODO: Verify
            if kernel_size % 2 == 0:
                kernel_size += 1
            # kernel1: (S, 2, kH, kW), kernel2: (S, 3, kH, kW)
            kernel1 = torch.stack([self.make_gauss(1, kernel_size, scale=scale) for scale in scales], dim=1)
            kernel2 = torch.stack([self.make_gauss(2, kernel_size, scale=scale) for scale in scales], dim=1)
        elif use_rbf and rbf_available:
            # By default, generate Gaussian derivative kernels using rbf.
            kernel1 = self.make_gauss(1, 7).unsqueeze(1)
            kernel2 = self.make_gauss(2, 7).unsqueeze(1)
        else:
            # If rbf is not avaibable or not desired load precomputed kernels.
            with resources.files("asel").joinpath("data/kernels/kernel1.pt").open("rb") as f:
                kernel1 = torch.load(f, weights_only=True).unsqueeze(1)
            with resources.files("asel").joinpath("data/kernels/kernel2.pt").open("rb") as f:
                kernel2 = torch.load(f, weights_only=True).unsqueeze(1)
        conv_weights = torch.stack([-kernel1[1], kernel1[0], kernel2[2], kernel2[0]], dim=0)
        self.conv_weights = nn.Parameter(conv_weights, requires_grad=False)

    def make_coord(self, kernel_size):
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        coord = torch.meshgrid([-x, x])
        coord = torch.stack([coord[1], coord[0]], -1)
        return coord.reshape(kernel_size ** 2, 2)

    def make_gauss(self, order, kernel_size, scale=1.0):
        assert rbf_available
        diff = []
        coord = self.make_coord(kernel_size).float()
        # Apply the requested scale by rescaling coordinates. A larger
        # `scale` produces a wider Gaussian (we divide coordinates by scale).
        if scale != 1.0:
            coord = coord / float(scale)
        gauss = get_rbf('ga')
        center = torch.zeros(1, 2, dtype=coord.dtype)
        for i in range(order + 1):
            w = gauss(coord, center, eps=0.99, diff=[i, order - i]).reshape(kernel_size, kernel_size)
            w = torch.as_tensor(w, dtype=torch.float32)
            diff.append(w)
        tensor = torch.stack(diff, 0)
        return tensor

    def dx(self, u):
        assert u.dim() == 5
        b, _, c, h, w = u.shape
        kernel = self.conv_weights[0].unsqueeze(1).unsqueeze(2).expand(-1, c, -1, -1, -1)
        s = kernel.shape[0]
        u = u.expand(-1, s, -1, -1, -1)
        kH, kW = kernel.shape[-2], kernel.shape[-1]
        pad = (kW // 2, kW // 2, kH // 2, kH // 2)
        u = F.pad(u.reshape(b*s, c, h, w), pad, mode='replicate').view(b, s, c, h+pad[2]+pad[3], w+pad[0]+pad[1])
        out = torch.stack([F.conv2d(u[:, i], kernel[i], bias=None, stride=1, groups=c) for i in range(s)], dim=1)
        # Reshape from (B, S*C, H, W) to (B, S, C, H, W)
        # _, _, _, h, w = out.shape
        return out.view(b, s, c, h, w)

    def dy(self, u):
        assert u.dim() == 5
        b, _, c, h, w = u.shape
        kernel = self.conv_weights[1].unsqueeze(1).unsqueeze(2).expand(-1, c, -1, -1, -1)
        s = kernel.shape[0]
        u = u.expand(-1, s, -1, -1, -1)
        kH, kW = kernel.shape[-2], kernel.shape[-1]
        pad = (kW // 2, kW // 2, kH // 2, kH // 2)
        u = F.pad(u.reshape(b*s, c, h, w), pad, mode='replicate').view(b, s, c, h+pad[2]+pad[3], w+pad[0]+pad[1])
        out = torch.stack([F.conv2d(u[:, i], kernel[i], bias=None, stride=1, groups=c) for i in range(s)], dim=1)
        # Reshape from (B, S*C, H, W) to (B, S, C, H, W)
        # _, _, _, h, w = out.shape
        return out.view(b, s, c, h, w)

    def dxx(self, u):
        assert u.dim() == 5
        b, _, c, h, w = u.shape
        kernel = self.conv_weights[2].unsqueeze(1).unsqueeze(2).expand(-1, c, -1, -1, -1)
        s = kernel.shape[0]
        u = u.expand(-1, s, -1, -1, -1)
        kH, kW = kernel.shape[-2], kernel.shape[-1]
        pad = (kW // 2, kW // 2, kH // 2, kH // 2)
        u = F.pad(u.reshape(b*s, c, h, w), pad, mode='replicate').view(b, s, c, h+pad[2]+pad[3], w+pad[0]+pad[1])
        out = torch.stack([F.conv2d(u[:, i], kernel[i], bias=None, stride=1, groups=c) for i in range(s)], dim=1)
        # Reshape from (B, S*C, H, W) to (B, S, C, H, W)
        # _, _, _, h, w = out.shape
        return out.view(b, s, c, h, w)

    def dyy(self, u):
        assert u.dim() == 5
        b, _, c, h, w = u.shape
        kernel = self.conv_weights[3].unsqueeze(1).unsqueeze(2).expand(-1, c, -1, -1, -1)
        s = kernel.shape[0]
        u = u.expand(-1, s, -1, -1, -1)
        kH, kW = kernel.shape[-2], kernel.shape[-1]
        pad = (kW // 2, kW // 2, kH // 2, kH // 2)
        u = F.pad(u.reshape(b*s, c, h, w), pad, mode='replicate').view(b, s, c, h+pad[2]+pad[3], w+pad[0]+pad[1])
        out = torch.stack([F.conv2d(u[:, i], kernel[i], bias=None, stride=1, groups=c) for i in range(s)], dim=1)
        # Reshape from (B, S*C, H, W) to (B, S, C, H, W)
        # _, _, _, h, w = out.shape
        return out.view(b, s, c, h, w)

    def forward(self, u):
        ux = self.dx(u)
        uy = self.dy(u)
        if self.ord == 1:
            return ux, uy
        elif self.ord == 2:
            uxx = self.dxx(u)
            uyy = self.dyy(u)
            uxy = self.dx(uy)
            return ux, uy, uxx, uyy, uxy


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
