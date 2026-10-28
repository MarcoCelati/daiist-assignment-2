"""
Detection evaluation in plain numpy, shared by train.py and app.py.

All three models (finetuned, zero-shot COCO, trivial) are scored by this same
code on the same test split, so differences between them come from the models
and not from how they were measured.

Formats used throughout:
  ground truth per image: array (k, 5)  ->  class, x1, y1, x2, y2
  predictions per image:  array (p, 6)  ->  class, confidence, x1, y1, x2, y2
Coordinates are normalised to 0..1. IoU does not change when both axes are
rescaled, so normalised boxes give the same IoU as pixel boxes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)  # COCO's 0.50:0.05:0.95
EMPTY_GT = np.zeros((0, 5))
EMPTY_PRED = np.zeros((0, 6))


# ---------------------------------------------------------------------------
# Reading labels
# ---------------------------------------------------------------------------

def read_yolo_labels(path: Path) -> np.ndarray:
    """YOLO txt (class cx cy w h, normalised) -> (k, 5) class, x1, y1, x2, y2.

    Some Roboflow exports store polygons (class x1 y1 x2 y2 ... xn yn). Those
    are converted to their bounding box.
    """
    if not path.is_file():
        return EMPTY_GT.copy()
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, vals = int(float(parts[0])), np.array(parts[1:], float)
        if len(vals) == 4:
            cx, cy, w, h = vals
            rows.append([c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        else:
            xs, ys = vals[0::2], vals[1::2]
            rows.append([c, xs.min(), ys.min(), xs.max(), ys.max()])
    return np.clip(np.array(rows, float), [0, 0, 0, 0, 0], [np.inf, 1, 1, 1, 1]) if rows else EMPTY_GT.copy()


# ---------------------------------------------------------------------------
# Matching predictions to ground truth
# ---------------------------------------------------------------------------

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (n, 4) and every box in b (m, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match_image(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Which predictions are true positives, at each IoU threshold.

    COCO rule: go through predictions from most to least confident; each one
    claims the unclaimed ground-truth box of the same class it overlaps most,
    if that overlap reaches the IoU threshold. Returns bool (p, 10), in the
    order of `pred`.
    """
    tp = np.zeros((len(pred), len(IOU_THRESHOLDS)), bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    iou = box_iou(pred[:, 2:], gt[:, 1:])
    iou[pred[:, 0][:, None] != gt[:, 0][None, :]] = 0.0  # classes must agree
    order = np.argsort(-pred[:, 1], kind="stable")
    for t, thr in enumerate(IOU_THRESHOLDS):
        claimed = np.zeros(len(gt), bool)
        for i in order:
            cand = np.where(~claimed, iou[i], 0.0)
            j = int(np.argmax(cand))
            if cand[j] >= thr:
                claimed[j] = True
                tp[i, t] = True
    return tp


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve, COCO 101-point interpolation."""
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    p = np.flip(np.maximum.accumulate(np.flip(p)))  # make precision monotone
    x = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(x, r, p)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def detection_metrics(gts: list, preds: list, names: list[str], conf: float) -> dict:
    """Per-class AP50, AP50-95, and precision/recall at confidence `conf`.

    Precision and recall are read at one fixed confidence (the same one the
    confusion matrix uses), so the numbers in the table and the matrix agree.
    """
    nc = len(names)
    all_cls, all_conf, all_tp = [], [], []
    for gt, pred in zip(gts, preds):
        if len(pred):
            all_cls.append(pred[:, 0])
            all_conf.append(pred[:, 1])
            all_tp.append(match_image(gt, pred))
    cls = np.concatenate(all_cls) if all_cls else np.zeros(0)
    cf = np.concatenate(all_conf) if all_conf else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros((0, len(IOU_THRESHOLDS)), bool)
    gt_cls = np.concatenate([g[:, 0] for g in gts]).astype(int) if gts else np.zeros(0, int)
    n_gt = np.bincount(gt_cls, minlength=nc)[:nc]

    per_class = []
    for c in range(nc):
        sel = cls == c
        order = np.argsort(-cf[sel], kind="stable")
        tpc, cfc = tp[sel][order], cf[sel][order]
        aps = np.zeros(len(IOU_THRESHOLDS))
        if n_gt[c] and len(tpc):
            ctp = np.cumsum(tpc, axis=0)
            cfp = np.cumsum(~tpc, axis=0)
            for t in range(len(IOU_THRESHOLDS)):
                aps[t] = average_precision(ctp[:, t] / n_gt[c], ctp[:, t] / (ctp[:, t] + cfp[:, t]))
        keep = cfc >= conf
        n_pred = int(keep.sum())
        n_tp = int(tpc[keep, 0].sum()) if n_pred else 0
        per_class.append({
            "class": names[c],
            "test_instances": int(n_gt[c]),
            "predictions": n_pred,
            "precision": n_tp / n_pred if n_pred else 0.0,
            "recall": n_tp / n_gt[c] if n_gt[c] else float("nan"),
            "AP50": float(aps[0]),
            "AP50-95": float(aps.mean()),
        })

    present = [r for r in per_class if r["test_instances"] > 0]
    return {
        "conf": conf,
        "per_class": per_class,
        "mAP50": float(np.mean([r["AP50"] for r in present])) if present else 0.0,
        "mAP50-95": float(np.mean([r["AP50-95"] for r in present])) if present else 0.0,
    }


def confusion_matrix(gts: list, preds: list, nc: int, conf: float, iou_thr: float = 0.5) -> np.ndarray:
    """Object-level confusion matrix, (nc + 1) x (nc + 1).

    Rows are the true class, columns the predicted class; the last row and
    column are "background". Ground truth nobody found lands in the background
    column (a miss); a prediction matching no object lands in the background
    row (a false alarm). Matching ignores class, so a box found but labelled
    wrong shows up off the diagonal rather than as a miss plus a false alarm.
    """
    cm = np.zeros((nc + 1, nc + 1), int)
    for gt, pred in zip(gts, preds):
        pred = pred[pred[:, 1] >= conf] if len(pred) else pred
        iou = box_iou(gt[:, 1:], pred[:, 2:])
        g_used, p_used = set(), set()
        if iou.size:
            pairs = np.argwhere(iou >= iou_thr)
            pairs = pairs[np.argsort(-iou[pairs[:, 0], pairs[:, 1]], kind="stable")]
            for gi, pi in pairs:
                if gi in g_used or pi in p_used:
                    continue
                g_used.add(gi)
                p_used.add(pi)
                cm[int(gt[gi, 0]), int(pred[pi, 0])] += 1
        for gi in range(len(gt)):
            if gi not in g_used:
                cm[int(gt[gi, 0]), nc] += 1
        for pi in range(len(pred)):
            if pi not in p_used:
                cm[nc, int(pred[pi, 0])] += 1
    return cm


# ---------------------------------------------------------------------------
# From boxes to one decision per image
# ---------------------------------------------------------------------------

def image_label(gt: np.ndarray, positive_ids: list[int]) -> int:
    """1 if the image truly contains any positive-class object."""
    return int(np.isin(gt[:, 0], positive_ids).any()) if len(gt) else 0


def image_score(pred: np.ndarray, positive_ids: list[int]) -> float:
    """The model's highest confidence for any positive-class object, else 0."""
    if len(pred) == 0:
        return 0.0
    hit = np.isin(pred[:, 0], positive_ids)
    return float(pred[hit, 1].max()) if hit.any() else 0.0


def ranking_quality(scores, labels) -> dict:
    """Threshold-free quality of the image scores.

    ROC AUC: chance a random positive image outscores a random negative one.
    Average precision: area under the image-level precision-recall curve.
    """
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    # Mann-Whitney U with average ranks for ties.
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    o = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[o])
    ap = average_precision(tp / npos, tp / np.arange(1, len(s) + 1))
    return {"roc_auc": float(auc), "average_precision": ap}
