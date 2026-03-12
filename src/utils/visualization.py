import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def draw_boxes(img, boxes, scores=None, color=(255,0,0), thickness=2, score_thresh=0.3):
    img = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        score = float(scores[i]) if scores is not None else None
        if score is not None and score < score_thresh:
            continue
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
        if score is not None:
            label = f"{score:.2f}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1-th-4), (x1+tw+2, y1), color, -1)
            cv2.putText(img, label, (x1+1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return img


def plot_training_curves(train_losses, val_maps=None, out_path="experiments/curves.png"):
    n = 1 + (val_maps is not None)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4))
    if n == 1:
        axes = [axes]
    axes[0].plot(train_losses, color="blue", label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if val_maps:
        axes[1].plot(val_maps, color="green", label="mAP@0.5")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP")
        axes[1].set_title("Validation mAP@0.5")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()