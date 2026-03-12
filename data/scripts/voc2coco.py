import argparse, json, os, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/PHD/VOC2007")
    p.add_argument("--out_dir",   default="data/coco_format")
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
        x1,y1 = float(b.find("xmin").text), float(b.find("ymin").text)
        x2,y2 = float(b.find("xmax").text), float(b.find("ymax").text)
        bw,bh = max(0,x2-x1), max(0,y2-y1)
        if bw>0 and bh>0: boxes.append((x1,y1,bw,bh))
    return boxes, w, h

def convert(data_root, split, out_dir):
    data_root = Path(data_root)
    ids = [l.strip() for l in open(data_root/"ImageSets"/"Main"/f"{split}.txt") if l.strip()]
    coco = {"info":{"description":f"PHD {split}"},"categories":[{"id":1,"name":"head"}],"images":[],"annotations":[]}
    ann_id = 1
    for img_idx, img_id in enumerate(ids):
        xp = data_root/"Annotations"/f"{img_id}.xml"
        if not xp.exists(): continue
        boxes, w, h = parse_xml(xp)
        coco["images"].append({"id":img_idx+1,"file_name":f"{img_id}.jpg","width":w,"height":h})
        for (x,y,bw,bh) in boxes:
            coco["annotations"].append({"id":ann_id,"image_id":img_idx+1,"category_id":1,
                "bbox":[round(x,2),round(y,2),round(bw,2),round(bh,2)],"area":round(bw*bh,2),"iscrowd":0,"segmentation":[]})
            ann_id += 1
    os.makedirs(out_dir, exist_ok=True)
    out = Path(out_dir)/f"phd_{split}.json"
    json.dump(coco, open(out,"w"))
    print(f"[{split}] {len(coco['images'])} images, {len(coco['annotations'])} anns → {out}")

def main():
    args = parse_args()
    for sp in ["train","val","test"]:
        convert(args.data_root, sp, args.out_dir)

if __name__ == "__main__":
    main()