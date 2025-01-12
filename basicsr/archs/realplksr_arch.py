# type: ignore  # noqa: PGH003
from functools import partial

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from basicsr.archs.arch_util import DySample
from basicsr.utils.registry import ARCH_REGISTRY


class DCCM(nn.Sequential):
    """Doubled Convolutional Channel Mixer"""

    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class PLKConv2d(nn.Module):
    """Partial Large Kernel Convolutional Layer"""

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x


class EA(nn.Module):
    """Element-wise Attention"""

    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.f(x)


class PLKBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        norm_groups: int,
        use_ea: bool = True,
    ):
        super().__init__()

        # Local Texture
        self.channel_mixer = DCCM(dim)

        # Long-range Dependency
        pdim = int(dim * split_ratio)

        # Conv Layer
        self.lk = PLKConv2d(pdim, kernel_size)

        # Instance-dependent modulation
        if use_ea:
            self.attn = EA(dim)
        else:
            self.attn = nn.Identity()

        # Refinement
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)

        # Group Normalization
        self.norm = nn.GroupNorm(norm_groups, dim)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)

        return x + x_skip


@ARCH_REGISTRY.register()
class realplksr(nn.Module):
    """Partial Large Kernel CNNs for Efficient Super-Resolution:
    https://arxiv.org/abs/2404.11848

    Making PLKSR stable for real-world SISR by. neosr-project
    https://github.com/dslisleedh/PLKSR/issues/4
    
    Args:
        upscaling_factor (int): Upscaling factor for resolution enhancement.
        in_ch (int): Number of input channels (default: 3, for RGB images).
        out_ch (int): Number of output channels (default: 3, for RGB images).
        dim (int): Number of features in convolutional layers (default: 64).
        n_blocks (int): Number of PLK blocks (default: 28).
        kernel_size (int): Kernel size for PLK convolutional layers (default: 17).
        split_ratio (float): Ratio of feature channels for PLK processing (default: 0.25).
        use_ea (bool): Whether to use element-wise attention (default: True).
        norm_groups (int): Number of groups for group normalization (default: 4).
        dropout (float): Dropout rate for regularization (default: 0).
        dysample (bool): Whether to use DySample for upscaling (default: False).
    """

    def __init__(
        self,
        upscaling_factor: int,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 28,
        kernel_size: int = 17,
        split_ratio: float = 0.25,
        use_ea: bool = True,
        norm_groups: int = 4,
        dropout: float = 0,
        dysample: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.upscale = upscaling_factor
        self.dysample = dysample
        if not self.training:
            dropout = 0

        self.feats = nn.Sequential(
            *[nn.Conv2d(in_ch, dim, 3, 1, 1)]
            + [
                PLKBlock(dim, kernel_size, split_ratio, norm_groups, use_ea)
                for _ in range(n_blocks)
            ]
            + [nn.Dropout2d(dropout)]
            + [nn.Conv2d(dim, out_ch * upscaling_factor**2, 3, 1, 1)]
        )
        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)

        self.repeat_op = partial(
            torch.repeat_interleave, repeats=upscaling_factor**2, dim=1
        )

        if dysample and upscaling_factor != 1:
            groups = out_ch if upscaling_factor % 2 != 0 else 4
            self.to_img = DySample(
                in_ch * upscaling_factor**2,
                out_ch,
                upscaling_factor,
                groups=groups,
                end_convolution=True if upscaling_factor != 1 else False,
            )
        else:
            self.to_img = nn.PixelShuffle(upscaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feats(x) + self.repeat_op(x)
        if not self.dysample or (self.dysample and self.upscale != 1):
            x = self.to_img(x)
        return x


@ARCH_REGISTRY.register()
def realplksr_s(**kwargs):
    return realplksr(n_blocks=12, kernel_size=13, use_ea=False, **kwargs)