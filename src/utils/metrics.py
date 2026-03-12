import time
import numpy as np
from collections import defaultdict


def _iou_single_vs_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-7)


def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    n_pred = len(pred_boxes)
    n_gt   = len(gt_boxes)
    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)

    if n_gt == 0:
        fp[:] = 1
        return tp, fp, n_gt
    if n_pred == 0:
        return tp, fp, n_gt

    order = np.argsort(-pred_scores)
    pred_boxes  = pred_boxes[order]
    gt_matched  = np.zeros(n_gt, dtype=bool)

    for i, pb in enumerate(pred_boxes):
        iou      = _iou_single_vs_many(pb, gt_boxes)
        best_idx = np.argmax(iou)
        if iou[best_idx] >= iou_threshold and not gt_matched[best_idx]:
            tp[i] = 1
            gt_matched[best_idx] = True
        else:
            fp[i] = 1

    unsort = np.argsort(order)
    return tp[unsort], fp[unsort], n_gt


def compute_ap(tp_all, fp_all, n_gt_total):
    if n_gt_total == 0:
        return 0.0
    tp_cum = np.cumsum(tp_all)
    fp_cum = np.cumsum(fp_all)
    recall    = tp_cum / (n_gt_total + 1e-7)
    precision = tp_cum / (tp_cum + fp_cum + 1e-7)
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p = precision[recall >= thr]
        if len(p) > 0:
            ap += np.max(p)
    return ap / 101


class DetectionEvaluator:
    IOT_THRESHOLDS   = np.arange(0.5, 1.0, 0.05)
    SMALL_AREA_THRESH = 32 * 32

    def __init__(self):
        self.reset()

    def reset(self):
        self._records = []

    def update(self, pred_boxes, pred_scores, gt_boxes):
        self._records.append({
            "pred_boxes":  pred_boxes,
            "pred_scores": pred_scores,
            "gt_boxes":    gt_boxes,
        })

    def compute(self):
        tps  = defaultdict(list)
        fps  = defaultdict(list)
        scrs = defaultdict(list)
        n_gts_total = defaultdict(int)
        tps_s  = defaultdict(list)
        fps_s  = defaultdict(list)
        scrs_s = defaultdict(list)
        n_gts_small = defaultdict(int)

        for rec in self._records:
            pb = rec["pred_boxes"]
            ps = rec["pred_scores"]
            gb = rec["gt_boxes"]

            for iou_thr in self.IOT_THRESHOLDS:
                k = round(iou_thr, 2)
                tp, fp, n_gt = match_predictions(pb, ps, gb, iou_thr)
                tps[k].extend(tp.tolist())
                fps[k].extend(fp.tolist())
                scrs[k].extend(ps.tolist() if len(ps) else [])
                n_gts_total[k] += n_gt

                if len(gb) > 0:
                    small = ((gb[:,2]-gb[:,0]) * (gb[:,3]-gb[:,1])) < self.SMALL_AREA_THRESH
                    gb_s  = gb[small]
                else:
                    gb_s = gb
                tp_s, fp_s, n_gt_s = match_predictions(pb, ps, gb_s, iou_thr)
                tps_s[k].extend(tp_s.tolist())
                fps_s[k].extend(fp_s.tolist())
                scrs_s[k].extend(ps.tolist() if len(ps) else [])
                n_gts_small[k] += n_gt_s

        def _ap(tps_d, fps_d, scrs_d, n_gts_d, key):
            sc = np.array(scrs_d[key])
            tp = np.array(tps_d[key])
            fp = np.array(fps_d[key])
            if len(sc) == 0:
                return 0.0
            order = np.argsort(-sc)
            return compute_ap(tp[order], fp[order], n_gts_d[key])

        aps   = [_ap(tps,   fps,   scrs,   n_gts_total, round(t,2)) for t in self.IOT_THRESHOLDS]
        aps_s = [_ap(tps_s, fps_s, scrs_s, n_gts_small, round(t,2)) for t in self.IOT_THRESHOLDS]

        map_50     = aps[0]
        map_50_95  = float(np.mean(aps))
        maps_50_95 = float(np.mean(aps_s))

        k50  = 0.5
        sc50 = np.array(scrs[k50])
        tp50 = np.array(tps[k50])
        fp50 = np.array(fps[k50])
        if len(sc50) > 0:
            order50  = np.argsort(-sc50)
            tp_cum   = np.cumsum(tp50[order50])
            fp_cum   = np.cumsum(fp50[order50])
            prec     = tp_cum / (tp_cum + fp_cum + 1e-7)
            rec      = tp_cum / (n_gts_total[k50] + 1e-7)
            f1_arr   = 2 * prec * rec / (prec + rec + 1e-7)
            best     = np.argmax(f1_arr)
            precision, recall, f1 = float(prec[best]), float(rec[best]), float(f1_arr[best])
        else:
            precision = recall = f1 = 0.0

        return {
            "P":               precision,
            "R":               recall,
            "F1":              f1,
            "mAP@0.5":         map_50,
            "mAP@[0.5:0.95]":  map_50_95,
            "mAPs@[0.5:0.95]": maps_50_95,
        }


class FPSTimer:
    def __init__(self):
        self._start = None
        self._count = 0
        self._total = 0.0

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        if self._start:
            self._total += time.perf_counter() - self._start
            self._count += 1
            self._start  = None

    def fps(self):
        return self._count / self._total if self._total > 0 else 0.0


def print_metrics(metrics, model_name="Model"):
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")
    print(f"{'='*55}\n")