import torch
import torch.nn as nn
from ..backbone.cspdarknet import build_backbone, BACKBONE_CONFIGS
from ..neck.panet import YoloPAFPN
from ..head.yolo_head import YoloHead, decode_outputs
from ..utils.boxes import cxcywh2xyxy, nms


class YoloX(nn.Module):
    STRIDES = (8, 16, 32)

    def __init__(self, size="s", num_classes=1, img_size=(480,640), conf_thresh=0.25, iou_thresh=0.45, neck_type="panet"):
        super().__init__()
        self.num_classes = num_classes
        self.img_size    = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh

        self.backbone = build_backbone(size)

        if neck_type == "panet":
            self.neck = YoloPAFPN(size)
        elif neck_type == "cfe":
            from ..neck.cfe import CFENeck
            self.neck = CFENeck(size)
        else:
            raise ValueError(f"Unknown neck: {neck_type}")

        self.head = YoloHead(num_classes=num_classes, in_channels=self.neck.out_channels)

    def forward(self, x):
        feats    = self.backbone(x)
        feats    = self.neck(feats)
        raw_outs = self.head(feats)

        if self.training:
            return raw_outs

        decoded = decode_outputs(raw_outs, strides=self.STRIDES, img_size=self.img_size)
        return self._postprocess(decoded)

    @torch.no_grad()
    def _postprocess(self, decoded):
        results = []
        for pred in decoded:
            obj    = pred[:, 4]
            cls    = pred[:, 5:]
            scores = obj * cls[:, 0] if self.num_classes == 1 else (obj.unsqueeze(1) * cls).max(1).values
            class_ids = torch.zeros_like(scores) if self.num_classes == 1 else (obj.unsqueeze(1) * cls).max(1).indices

            mask = scores >= self.conf_thresh
            if mask.sum() == 0:
                results.append(torch.zeros((0,6), device=decoded.device))
                continue

            boxes_xyxy = cxcywh2xyxy(pred[mask, :4])
            scores_f   = scores[mask]
            class_ids_f= class_ids[mask]
            keep       = nms(boxes_xyxy, scores_f, self.iou_thresh)
            results.append(torch.cat([boxes_xyxy[keep], scores_f[keep].unsqueeze(1), class_ids_f[keep].unsqueeze(1).float()], dim=1))
        return results


def build_model(size="s", num_classes=1, img_size=(480,640), neck_type="panet"):
    return YoloX(size=size, num_classes=num_classes, img_size=img_size, neck_type=neck_type)