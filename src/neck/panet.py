import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.cspdarknet import BaseConv, CSPLayer, BACKBONE_CONFIGS


class YoloPAFPN(nn.Module):
    def __init__(self, size="s"):
        super().__init__()
        dep_mul, wid_mul = BACKBONE_CONFIGS[size]
        D = max(round(dep_mul * 3), 1)
        C = int(wid_mul * 64)
        C3, C4, C5 = C*4, C*8, C*16

        self.lateral_conv_p5 = BaseConv(C5, C4, 1, 1)
        self.c3_p4           = CSPLayer(C4*2, C4, n=D, shortcut=False)
        self.reduce_conv_p4  = BaseConv(C4, C3, 1, 1)
        self.c3_p3           = CSPLayer(C3*2, C3, n=D, shortcut=False)
        self.bu_conv_p3      = BaseConv(C3, C3, 3, 2)
        self.c3_n4           = CSPLayer(C3+C4, C4, n=D, shortcut=False)
        self.bu_conv_p4      = BaseConv(C4, C4, 3, 2)
        self.c3_n5           = CSPLayer(C4+C5, C5, n=D, shortcut=False)
        self.out_channels    = [C3, C4, C5]

    def forward(self, features):
        p3 = features["dark3"]
        p4 = features["dark4"]
        p5 = features["dark5"]

        p5_td  = self.lateral_conv_p5(p5)
        p4_td  = self.c3_p4(torch.cat([F.interpolate(p5_td, size=p4.shape[2:], mode="nearest"), p4], dim=1))
        p4_td  = self.reduce_conv_p4(p4_td)
        p3_out = self.c3_p3(torch.cat([F.interpolate(p4_td, size=p3.shape[2:], mode="nearest"), p3], dim=1))
        p4_out = self.c3_n4(torch.cat([self.bu_conv_p3(p3_out), p4_td], dim=1))
        p5_out = self.c3_n5(torch.cat([self.bu_conv_p4(p4_out), p5],    dim=1))
        return [p3_out, p4_out, p5_out]