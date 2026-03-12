import numpy as np
import torch


def xyxy2cxcywh(boxes):
    if isinstance(boxes, torch.Tensor):
        out = boxes.clone()
        out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
        out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
        out[..., 2] = boxes[..., 2] - boxes[..., 0]
        out[..., 3] = boxes[..., 3] - boxes[..., 1]
    else:
        out = boxes.copy().astype(np.float32)
        out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
        out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
        out[..., 2] = boxes[..., 2] - boxes[..., 0]
        out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def cxcywh2xyxy(boxes):
    if isinstance(boxes, torch.Tensor):
        out = boxes.clone()
        out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
        out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
        out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
        out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    else:
        out = boxes.copy().astype(np.float32)
        out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
        out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
        out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
        out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def xywh2xyxy(boxes):
    if isinstance(boxes, torch.Tensor):
        out = boxes.clone()
        out[..., 2] = boxes[..., 0] + boxes[..., 2]
        out[..., 3] = boxes[..., 1] + boxes[..., 3]
    else:
        out = boxes.copy().astype(np.float32)
        out[..., 2] = boxes[..., 0] + boxes[..., 2]
        out[..., 3] = boxes[..., 1] + boxes[..., 3]
    return out


def box_iou(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[i:i+1], boxes[rest]).squeeze(0)
        order = rest[ious < iou_threshold]
    return torch.tensor(keep, dtype=torch.long)


def clip_boxes(boxes, img_h, img_w):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, img_w)
        boxes[..., 1].clamp_(0, img_h)
        boxes[..., 2].clamp_(0, img_w)
        boxes[..., 3].clamp_(0, img_h)
    else:
        boxes[..., 0] = np.clip(boxes[..., 0], 0, img_w)
        boxes[..., 1] = np.clip(boxes[..., 1], 0, img_h)
        boxes[..., 2] = np.clip(boxes[..., 2], 0, img_w)
        boxes[..., 3] = np.clip(boxes[..., 3], 0, img_h)
    return boxes