import torch
import torch.nn as nn
from ..backbone.cspdarknet import BaseConv


class YoloHead(nn.Module):
    def __init__(self, num_classes=1, in_channels=(128, 256, 512)):
        super().__init__()
        self.num_classes = num_classes
        self.stems, self.cls_convs, self.reg_convs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.cls_preds, self.reg_preds, self.obj_preds = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for ch in in_channels:
            self.stems.append(BaseConv(ch, 256, 1, 1))
            self.cls_convs.append(nn.Sequential(BaseConv(256,256,3,1), BaseConv(256,256,3,1)))
            self.reg_convs.append(nn.Sequential(BaseConv(256,256,3,1), BaseConv(256,256,3,1)))
            self.cls_preds.append(nn.Conv2d(256, num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(256, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(256, 1, 1, 1, 0))

        self._init_weights()

    def _init_weights(self):
        import math
        bias = -math.log((1 - 0.01) / 0.01)
        for cp, op in zip(self.cls_preds, self.obj_preds):
            nn.init.constant_(cp.bias, bias)
            nn.init.constant_(op.bias, bias)

    def forward(self, features):
        outputs = []
        for feat, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred in zip(
            features, self.stems, self.cls_convs, self.reg_convs,
            self.cls_preds, self.reg_preds, self.obj_preds
        ):
            x        = stem(feat)
            cls_out  = cls_pred(cls_conv(x))
            reg_feat = reg_conv(x)
            reg_out  = reg_pred(reg_feat)
            obj_out  = obj_pred(reg_feat)
            B, _, H, W = cls_out.shape
            out = torch.cat([reg_out, obj_out, cls_out], dim=1).permute(0,2,3,1).reshape(B, H*W, -1)
            outputs.append(out)
        return outputs


def decode_outputs(raw_outputs, strides=(8,16,32), img_size=(480,640)):
    H, W = img_size
    all_outputs = []
    for out, stride in zip(raw_outputs, strides):
        B, n_anchors, n_ch = out.shape
        fH, fW = H // stride, W // stride
        gy, gx = torch.meshgrid(
            torch.arange(fH, device=out.device, dtype=out.dtype),
            torch.arange(fW, device=out.device, dtype=out.dtype),
            indexing="ij"
        )
        grid = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2)
        out  = out.clone()
        out[..., :2] = (torch.sigmoid(out[..., :2]) + grid) * stride
        out[..., 2:4] = torch.exp(out[..., 2:4]) * stride
        out[..., 4:]  = torch.sigmoid(out[..., 4:])
        all_outputs.append(out)
    return torch.cat(all_outputs, dim=1)