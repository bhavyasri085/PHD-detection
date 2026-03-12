import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PHDDataset(Dataset):
    CLASS_NAMES = ["head"]

    def __init__(self, data_root, split="train", img_size=(480, 640), transforms=None, return_raw=False):
        assert split in ("train", "val", "test")
        self.data_root  = Path(data_root)
        self.split      = split
        self.img_size   = img_size
        self.transforms = transforms
        self.return_raw = return_raw

        self.img_dir  = self.data_root / "JPEGImages"
        self.ann_dir  = self.data_root / "Annotations"
        self.sets_dir = self.data_root / "ImageSets" / "Main"

        split_file = self.sets_dir / f"{split}.txt"
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        print(f"[PHDDataset] {split}: {len(self.ids)} images")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_path = self.img_dir / f"{img_id}.jpg"
        img      = cv2.imread(str(img_path))
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        ann_path = self.ann_dir / f"{img_id}.xml"
        boxes    = self._parse_xml(ann_path, orig_w, orig_h)

        target_h, target_w = self.img_size
        img_resized = cv2.resize(img, (target_w, target_h))

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= target_w / orig_w
            boxes[:, [1, 3]] *= target_h / orig_h

        boxes_norm = self._xyxy_to_cxcywh_norm(boxes, target_w, target_h)

        if not self.return_raw:
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
        else:
            img_tensor = img_resized

        labels  = torch.zeros((len(boxes_norm), 1), dtype=torch.float32)
        targets = torch.cat([labels, torch.from_numpy(boxes_norm)], dim=1) if len(boxes_norm) > 0 else torch.zeros((0, 5))

        return {
            "img":       img_tensor,
            "targets":   targets,
            "img_id":    img_id,
            "orig_size": (orig_h, orig_w),
            "img_size":  self.img_size,
        }

    def _parse_xml(self, xml_path, img_w, img_h):
        if not xml_path.exists():
            return np.zeros((0, 4), dtype=np.float32)
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            b  = obj.find("bndbox")
            x1 = float(b.find("xmin").text)
            y1 = float(b.find("ymin").text)
            x2 = float(b.find("xmax").text)
            y2 = float(b.find("ymax").text)
            x1 = max(0.0, min(x1, img_w - 1))
            y1 = max(0.0, min(y1, img_h - 1))
            x2 = max(0.0, min(x2, float(img_w)))
            y2 = max(0.0, min(y2, float(img_h)))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

    def _xyxy_to_cxcywh_norm(self, boxes, w, h):
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        out = np.empty_like(boxes)
        out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0 / w
        out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0 / h
        out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w
        out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h
        return out.astype(np.float32)


def collate_fn(batch):
    imgs     = torch.stack([b["img"] for b in batch], dim=0)
    img_ids  = [b["img_id"] for b in batch]
    orig_sizes = [b["orig_size"] for b in batch]

    targets_list = []
    for i, b in enumerate(batch):
        t = b["targets"]
        if len(t) > 0:
            batch_idx = torch.full((len(t), 1), i, dtype=torch.float32)
            targets_list.append(torch.cat([batch_idx, t], dim=1))

    targets = torch.cat(targets_list, dim=0) if targets_list else torch.zeros((0, 6))

    return {
        "imgs":      imgs,
        "targets":   targets,
        "img_ids":   img_ids,
        "orig_sizes": orig_sizes,
    }