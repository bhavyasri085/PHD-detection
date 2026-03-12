import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/PHD/VOC2007")
    p.add_argument("--out_dir",   default="data/samples")
    return p.parse_args()


def parse_xml(xml_path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append((float(b.find("xmin").text), float(b.find("ymin").text),
                       float(b.find("xmax").text), float(b.find("ymax").text)))
    return boxes, w, h


def collect(data_root, splits=("train","val","test")):
    data_root = Path(data_root)
    stats = {}
    for split in splits:
        sf = data_root / "ImageSets" / "Main" / f"{split}.txt"
        if not sf.exists(): continue
        ids = [l.strip() for l in open(sf) if l.strip()]
        cx,cy,density,area_r,asp_r = [],[],[],[],[]
        total_ann = 0
        for img_id in ids:
            xp = data_root / "Annotations" / f"{img_id}.xml"
            if not xp.exists(): continue
            boxes, iw, ih = parse_xml(xp)
            density.append(len(boxes))
            total_ann += len(boxes)
            for x1,y1,x2,y2 in boxes:
                bw,bh = x2-x1, y2-y1
                if bw<=0 or bh<=0: continue
                cx.append((x1+x2)/2/iw)
                cy.append((y1+y2)/2/ih)
                area_r.append(bw*bh/(iw*ih))
                asp_r.append(bw/bh)
        stats[split] = dict(ids=ids, cx=np.array(cx), cy=np.array(cy),
                            density=np.array(density), area_ratio=np.array(area_r),
                            aspect_ratio=np.array(asp_r), total=total_ann)
        print(f"[{split}] images={len(ids)} annotations={total_ann} avg_heads={np.mean(density):.1f}")
    return stats


def plot_positions(stats, out_dir):
    splits = list(stats.keys())
    fig, axes = plt.subplots(1, len(splits), figsize=(5*len(splits), 5))
    if len(splits)==1: axes=[axes]
    colors = {"train":"blue","val":"green","test":"red"}
    for ax, sp in zip(axes, splits):
        s = stats[sp]
        ax.scatter(s["cx"]*640, s["cy"]*480, s=1, alpha=0.3, c=colors.get(sp,"blue"))
        ax.set_xlim(0,640); ax.set_ylim(0,480); ax.invert_yaxis()
        ax.set_title(f"Position ({sp})")
    plt.suptitle("Fig 3: Position Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/"fig3_position.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3")


def plot_density(stats, out_dir):
    splits = list(stats.keys())
    fig, axes = plt.subplots(1, len(splits), figsize=(5*len(splits), 4))
    if len(splits)==1: axes=[axes]
    colors = {"train":"blue","val":"green","test":"red"}
    for ax, sp in zip(axes, splits):
        s = stats[sp]
        ax.hist(s["density"], bins=range(0, int(s["density"].max())+3), color=colors.get(sp,"blue"), edgecolor="white")
        ax.set_xlabel("Number of Heads"); ax.set_ylabel("Images"); ax.set_title(f"Density ({sp})")
    plt.suptitle("Fig 4: Density Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/"fig4_density.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig4")


def plot_scale_aspect(stats, out_dir):
    splits = list(stats.keys())
    fig, axes = plt.subplots(2, len(splits), figsize=(5*len(splits), 8))
    colors = {"train":"blue","val":"green","test":"red"}
    for col, sp in enumerate(splits):
        s = stats[sp]
        axes[0,col].hist(s["area_ratio"],   bins=50, range=(0,0.18), color=colors.get(sp,"blue"), edgecolor="white")
        axes[0,col].set_xlabel("Area Ratio"); axes[0,col].set_title(f"Scale ({sp})")
        axes[1,col].hist(s["aspect_ratio"], bins=50, range=(0,3.5),  color=colors.get(sp,"blue"), edgecolor="white")
        axes[1,col].set_xlabel("Aspect Ratio (w/h)"); axes[1,col].set_title(f"Aspect ({sp})")
    plt.suptitle("Fig 5: Scale & Aspect Ratio", fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/"fig5_scale_aspect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig5")


def visualize_samples(data_root, out_dir, split="train", n=6):
    data_root = Path(data_root)
    ids = [l.strip() for l in open(data_root/"ImageSets"/"Main"/f"{split}.txt") if l.strip()][:n]
    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    axes = axes.flatten()
    for ax, img_id in zip(axes, ids):
        img = cv2.cvtColor(cv2.imread(str(data_root/"JPEGImages"/f"{img_id}.jpg")), cv2.COLOR_BGR2RGB)
        boxes, _, _ = parse_xml(data_root/"Annotations"/f"{img_id}.xml")
        ax.imshow(img)
        for x1,y1,x2,y2 in boxes:
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1, linewidth=1, edgecolor="red", facecolor="none"))
        ax.set_title(f"{img_id} | {len(boxes)} heads", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"Samples ({split})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f"samples_{split}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved samples_{split}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stats = collect(args.data_root)
    plot_positions(stats, args.out_dir)
    plot_density(stats, args.out_dir)
    plot_scale_aspect(stats, args.out_dir)
    visualize_samples(args.data_root, args.out_dir)
    print("\nAll EDA figures saved to:", args.out_dir)

if __name__ == "__main__":
    main()