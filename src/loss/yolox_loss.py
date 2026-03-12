import torch
import torch.nn as nn
import torch.nn.functional as F


def iou_loss(pred, gt, eps=1e-7):
    px1 = pred[:,0] - pred[:,2]/2; py1 = pred[:,1] - pred[:,3]/2
    px2 = pred[:,0] + pred[:,2]/2; py2 = pred[:,1] + pred[:,3]/2
    gx1 = gt[:,0]   - gt[:,2]/2;   gy1 = gt[:,1]   - gt[:,3]/2
    gx2 = gt[:,0]   + gt[:,2]/2;   gy2 = gt[:,1]   + gt[:,3]/2
    inter = (torch.min(px2,gx2)-torch.max(px1,gx1)).clamp(0) * (torch.min(py2,gy2)-torch.max(py1,gy1)).clamp(0)
    union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
    iou   = inter / (union + eps)
    cx_d  = (pred[:,0]-gt[:,0])**2 + (pred[:,1]-gt[:,1])**2
    ox1   = torch.min(px1,gx1); oy1 = torch.min(py1,gy1)
    ox2   = torch.max(px2,gx2); oy2 = torch.max(py2,gy2)
    diag  = (ox2-ox1)**2 + (oy2-oy1)**2 + eps
    v     = (4/torch.pi**2) * (torch.atan(gt[:,2]/(gt[:,3]+eps)) - torch.atan(pred[:,2]/(pred[:,3]+eps)))**2
    alpha = v / (1 - iou + v + eps)
    return 1 - iou + cx_d/diag + alpha*v


class YoloXLoss(nn.Module):
    def __init__(self, num_classes=1, strides=(8,16,32), img_size=(480,640)):
        super().__init__()
        self.num_classes = num_classes
        self.strides     = strides
        self.img_size    = img_size
        self.bce         = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, raw_outputs, targets):
        device = raw_outputs[0].device
        H, W   = self.img_size

        all_anchors, all_strides_t = [], []
        for stride, out in zip(self.strides, raw_outputs):
            _, n_anchors, _ = out.shape
            fH, fW = H//stride, W//stride
            gy, gx = torch.meshgrid(
                torch.arange(fH, device=device, dtype=torch.float32),
                torch.arange(fW, device=device, dtype=torch.float32),
                indexing="ij"
            )
            grid = torch.stack([gx,gy], dim=-1).reshape(-1,2)
            all_anchors.append((grid + 0.5) * stride)
            all_strides_t.append(torch.full((n_anchors,), stride, device=device))

        anchors   = torch.cat(all_anchors, dim=0)
        strides_t = torch.cat(all_strides_t, dim=0)
        all_out   = torch.cat(raw_outputs, dim=1)
        B         = all_out.shape[0]

        pred_boxes = torch.cat([
            (torch.sigmoid(all_out[...,:2]) + anchors.unsqueeze(0)/strides_t.unsqueeze(0).unsqueeze(-1)) * strides_t.unsqueeze(0).unsqueeze(-1),
            torch.exp(all_out[...,2:4].clamp(max=4)) * strides_t.unsqueeze(0).unsqueeze(-1),
        ], dim=-1)

        loss_obj = loss_cls = loss_box = torch.zeros(1, device=device)
        n_fg_total = 0

        for b in range(B):
            bt = targets[targets[:,0]==b]
            pred_b = all_out[b]
            pbox_b = pred_boxes[b]

            if len(bt) == 0:
                loss_obj = loss_obj + self.bce(pred_b[:,4], torch.zeros(len(pred_b), device=device)).mean()
                continue

            gt_boxes = bt[:,2:6].clone()
            gt_boxes[:,[0,2]] *= W
            gt_boxes[:,[1,3]] *= H
            gt_cls = bt[:,1].long()

            # Simple center-based assignment
            gt_cx = gt_boxes[:,0].unsqueeze(1)
            gt_cy = gt_boxes[:,1].unsqueeze(1)
            ac_x  = anchors[:,0].unsqueeze(0)
            ac_y  = anchors[:,1].unsqueeze(0)
            dist  = ((ac_x - gt_cx)**2 + (ac_y - gt_cy)**2).sqrt()

            fg_mask = torch.zeros(len(anchors), dtype=torch.bool, device=device)
            matched_gt_box = torch.zeros((len(anchors),4), device=device)

            for g in range(len(gt_boxes)):
                topk_idx = torch.topk(-dist[g], k=min(10, len(anchors))).indices
                fg_mask[topk_idx] = True
                matched_gt_box[topk_idx] = gt_boxes[g]

            n_fg = fg_mask.sum().item()
            n_fg_total += n_fg

            obj_t = fg_mask.float()
            loss_obj = loss_obj + self.bce(pred_b[:,4], obj_t).mean()

            if n_fg > 0:
                loss_box = loss_box + iou_loss(pbox_b[fg_mask], matched_gt_box[fg_mask]).mean()
                if self.num_classes > 1:
                    cls_t = F.one_hot(gt_cls[0:1].expand(n_fg), self.num_classes).float()
                    loss_cls = loss_cls + self.bce(pred_b[fg_mask, 5:], cls_t).mean()

        loss_obj = loss_obj / B
        loss_box = loss_box / max(1, n_fg_total/B)
        loss_cls = loss_cls / max(1, n_fg_total/B)
        total    = loss_obj + loss_cls + 5.0 * loss_box

        return total, {
            "loss_total": total.item(),
            "loss_obj":   loss_obj.item(),
            "loss_cls":   loss_cls.item(),
            "loss_box":   loss_box.item(),
            "n_fg":       n_fg_total / B,
        }