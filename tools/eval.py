import argparse, os, sys
from pathlib import Path

# always resolve to project root regardless of where script is called from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from src.models.yolox import build_model
from src.utils.dataset import PHDDataset, collate_fn
from src.utils.metrics import DetectionEvaluator, FPSTimer, print_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--split",   default="test", choices=["val","test"])
    p.add_argument("--device",  default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = yaml.safe_load(open(args.config))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Device: {device}")

    img_size = tuple(cfg["data"]["img_size"])
    H, W     = img_size
    dataset  = PHDDataset(cfg["data"]["root"], args.split, img_size)
    loader   = DataLoader(dataset, cfg["eval"]["batch_size"], shuffle=False,
                          num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn)

    mc    = cfg["model"]
    model = build_model(mc["size"], mc["num_classes"], img_size, mc["neck_type"])
    ckpt  = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    model.conf_thresh = cfg["eval"]["conf_thresh"]
    model.iou_thresh  = cfg["eval"]["iou_thresh"]

    ev    = DetectionEvaluator()
    timer = FPSTimer()

    with torch.no_grad():
        for batch in tqdm(loader):
            imgs    = batch["imgs"].to(device)
            targets = batch["targets"]
            timer.start()
            dets = model(imgs)
            # Add temporarily inside evaluate() after dets = model(imgs), before the loop:
            for _i, _det in enumerate(dets):
                if len(_det) > 0:
                    print(f"[DEBUG] pred boxes[0]: {_det[0,:4]}  score: {_det[0,4]:.4f}")
                    print(f"[DEBUG] pred box ranges: x=[{_det[:,0].min():.1f},{_det[:,2].max():.1f}], y=[{_det[:,1].min():.1f},{_det[:,3].max():.1f}]")
                    break
            break  # only check first batch
            timer.stop()
            for i, det in enumerate(dets):
                # NEW — correctly converts normalized cx/cy/w/h → absolute x1/y1/x2/y2
                bt = targets[targets[:,0]==i]
                if len(bt) > 0:
                    cxcywh = bt[:,2:6].numpy().copy()          # normalized cx, cy, w, h
                    cx = cxcywh[:,0] * W
                    cy = cxcywh[:,1] * H
                    w  = cxcywh[:,2] * W
                    h  = cxcywh[:,3] * H
                    gb = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1).astype(np.float32)
                else:
                    gb = np.zeros((0,4), dtype=np.float32)
                pb = det[:,:4].cpu().numpy() if len(det)>0 else np.zeros((0,4),dtype=np.float32)
                ps = det[:,4].cpu().numpy()  if len(det)>0 else np.zeros(0,dtype=np.float32)
                ev.update(pb, ps, gb)

    metrics = ev.compute()
    metrics["FPS"] = timer.fps()
    print_metrics(metrics, f"YoloX-{mc['size'].upper()} [{args.split}]")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/f"metrics_{args.split}.txt","w") as f:
        [f.write(f"{k}: {v:.4f}\n") for k,v in metrics.items()]
    logger.info(f"Saved to {out_dir}/metrics_{args.split}.txt")

if __name__ == "__main__":
    main()