import argparse, math, os, sys, time
from pathlib import Path

# always resolve to project root regardless of where script is called from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import torch, torch.nn as nn, yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import numpy as np

from src.models.yolox import build_model
from src.loss.yolox_loss import YoloXLoss
from src.utils.dataset import PHDDataset, collate_fn
from src.utils.metrics import DetectionEvaluator, print_metrics
from src.utils.visualization import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--resume",  default=None)
    p.add_argument("--device",  default=None)
    p.add_argument("--amp",     action="store_true")
    return p.parse_args()


def build_optimizer(model, cfg):
    oc = cfg["train"]["optimizer"]
    no_wd, with_wd = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim < 2 or "bn" in name or "bias" in name: no_wd.append(p)
        else: with_wd.append(p)
    groups = [{"params":no_wd,"weight_decay":0.0},{"params":with_wd,"weight_decay":oc["weight_decay"]}]
    if oc["type"] == "sgd":
        return torch.optim.SGD(groups, lr=oc["lr"], momentum=oc["momentum"], nesterov=True)
    return torch.optim.AdamW(groups, lr=oc["lr"])


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total   = cfg["train"]["epochs"] * steps_per_epoch
    warmup  = cfg["train"]["warmup_epochs"] * steps_per_epoch
    min_lr  = cfg["train"]["scheduler"]["min_lr"]
    base_lr = cfg["train"]["optimizer"]["lr"]
    def lr_fn(step):
        if step < warmup: return (step+1)/warmup
        t = (step-warmup)/max(1,total-warmup)
        return max(min_lr/base_lr, 0.5*(1+math.cos(math.pi*t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, cfg, writer, scaler):
    model.train()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(loader):
        imgs    = batch["imgs"].to(device)
        targets = batch["targets"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            raw = model(imgs)
            loss, ld = criterion(raw, targets)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item(); n += 1
        if (i+1) % cfg["logging"]["log_interval"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Ep[{epoch+1}][{i+1}/{len(loader)}] loss={loss.item():.4f} obj={ld['loss_obj']:.3f} box={ld['loss_box']:.3f} lr={lr:.6f}")
            if writer:
                step = epoch*len(loader)+i
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
    return total_loss / max(1,n)


@torch.no_grad()
def evaluate(model, loader, device, cfg, epoch=0, writer=None):
    model.eval()
    ev = DetectionEvaluator()
    H, W = cfg["data"]["img_size"]
    for batch in loader:
        imgs    = batch["imgs"].to(device)
        targets = batch["targets"]
        dets    = model(imgs)
        # Add temporarily inside evaluate() after dets = model(imgs), before the loop:
        for _i, _det in enumerate(dets):
            if len(_det) > 0:
                print(f"[DEBUG] pred boxes[0]: {_det[0,:4]}  score: {_det[0,4]:.4f}")
                print(f"[DEBUG] pred box ranges: x=[{_det[:,0].min():.1f},{_det[:,2].max():.1f}], y=[{_det[:,1].min():.1f},{_det[:,3].max():.1f}]")
                break
        break  # only check first batch
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
    if writer:
        for k,v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
    return metrics


def save_ckpt(model, optimizer, epoch, metrics, out_dir, tag):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"epoch":epoch,"model":model.state_dict(),"optim":optimizer.state_dict(),"metrics":metrics},
               Path(out_dir)/f"{tag}.pth")


def main():
    args    = parse_args()
    cfg     = yaml.safe_load(open(args.config))
    device  = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = cfg["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.add(f"{out_dir}/train.log")
    logger.info(f"Device: {device}")
    logger.info(f"Working dir: {os.getcwd()}")

    img_size  = tuple(cfg["data"]["img_size"])
    data_root = cfg["data"]["root"]

    train_ds = PHDDataset(data_root, "train", img_size)
    val_ds   = PHDDataset(data_root, "val",   img_size)
    train_loader = DataLoader(train_ds, cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"], drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, cfg["eval"]["batch_size"], shuffle=False,
                              num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn)

    mc    = cfg["model"]
    model = build_model(mc["size"], mc["num_classes"], img_size, mc["neck_type"]).to(device)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    criterion = YoloXLoss(mc["num_classes"], img_size=img_size)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = torch.cuda.amp.GradScaler() if args.amp and device.type=="cuda" else None

    start_epoch, best_map = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1

    writer = SummaryWriter(f"{out_dir}/tb") if cfg["logging"]["use_tensorboard"] else None
    train_losses, val_maps = [], []

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, cfg, writer, scaler)
        train_losses.append(loss)
        logger.info(f"Epoch {epoch+1} loss={loss:.4f}")

        if (epoch+1) % cfg["train"]["eval_every"] == 0 or epoch == cfg["train"]["epochs"]-1:
            # lower conf_thresh for proper mAP evaluation
            model.conf_thresh = 0.001
            metrics = evaluate(model, val_loader, device, cfg, epoch, writer)
            model.conf_thresh = cfg["model"]["conf_thresh"]
            print_metrics(metrics, f"YoloX-{mc['size'].upper()} ep{epoch+1}")
            val_maps.append(metrics["mAP@0.5"])
            if metrics["mAP@0.5"] > best_map:
                best_map = metrics["mAP@0.5"]
                save_ckpt(model, optimizer, epoch, metrics, out_dir, "best")
                logger.info(f"New best mAP@0.5={best_map:.4f}")

        if (epoch+1) % cfg["train"]["save_every"] == 0:
            save_ckpt(model, optimizer, epoch, {}, out_dir, f"epoch_{epoch+1}")

    save_ckpt(model, optimizer, cfg["train"]["epochs"]-1, {}, out_dir, "last")
    plot_training_curves(train_losses, val_maps if val_maps else None, f"{out_dir}/curves.png")
    logger.info(f"Done. Best mAP@0.5={best_map:.4f}")
    if writer: writer.close()

if __name__ == "__main__":
    main()