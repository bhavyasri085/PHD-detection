import torch
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, groups=1, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = SiLU() if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1   = BaseConv(in_channels, hidden, 1, 1)
        self.conv2   = BaseConv(hidden, out_channels, 3, 1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden     = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden, 1, 1)
        self.conv3 = BaseConv(2 * hidden, out_channels, 1, 1)
        self.m     = nn.Sequential(*[Bottleneck(hidden, hidden, shortcut=shortcut, expansion=1.0) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat([self.m(self.conv1(x)), self.conv2(x)], dim=1))


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden     = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.m     = nn.ModuleList([nn.MaxPool2d(k, stride=1, padding=k//2) for k in kernel_sizes])
        self.conv2 = BaseConv(hidden * (len(kernel_sizes) + 1), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, 1)

    def forward(self, x):
        return self.conv(torch.cat([
            x[..., ::2,  ::2],
            x[..., 1::2, ::2],
            x[..., ::2,  1::2],
            x[..., 1::2, 1::2],
        ], dim=1))


BACKBONE_CONFIGS = {
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
}


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul=1.0, wid_mul=1.0, out_features=("dark3","dark4","dark5")):
        super().__init__()
        self.out_features = out_features
        C = int(wid_mul * 64)
        D = max(round(dep_mul * 3), 1)

        self.stem  = Focus(3, C, ksize=3)
        self.dark2 = nn.Sequential(BaseConv(C, C*2, 3, 2),   CSPLayer(C*2,  C*2,  n=D))
        self.dark3 = nn.Sequential(BaseConv(C*2, C*4, 3, 2), CSPLayer(C*4,  C*4,  n=D*3))
        self.dark4 = nn.Sequential(BaseConv(C*4, C*8, 3, 2), CSPLayer(C*8,  C*8,  n=D*3))
        self.dark5 = nn.Sequential(
            BaseConv(C*8, C*16, 3, 2),
            SPPBottleneck(C*16, C*16),
            CSPLayer(C*16, C*16, n=D, shortcut=False),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        d3 = self.dark3(x)
        d4 = self.dark4(d3)
        d5 = self.dark5(d4)
        return {"dark3": d3, "dark4": d4, "dark5": d5}


def build_backbone(size="s"):
    dep_mul, wid_mul = BACKBONE_CONFIGS[size]
    return CSPDarknet(dep_mul=dep_mul, wid_mul=wid_mul)