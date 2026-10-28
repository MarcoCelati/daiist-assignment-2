"""
Assignment 2: training pipeline. Run with

    uv run python main.py train

Stages, in order, with no manual steps:

  1. Read decisions.py and stop if any decision is still missing.
  2. Download the Roboflow Universe dataset version named there.
  3. Describe it: images and instances per class per split, object sizes.
  4. Finetune the pretrained YOLO checkpoint and save weights/best.pt.
  5. Score the finetuned model, the zero-shot COCO model and a trivial
     baseline on the same test split, with the same code (evaluation.py).
  6. Turn per-image scores into euros (costing.py): threshold sweep, review
     band, staffing, alternatives, bootstrap and sensitivity.
  7. Save everything app.py needs into artifacts/, which you commit.

REUSE_WEIGHTS=1 skips stage 4 and re-runs 5-7 on the committed weights, for
when you change costs or staffing in decisions.py but not the model.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

import costing
import decisions as D
import evaluation as ev
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
ART = ROOT / "artifacts"
DATA = ROOT / "datasets"
SPLITS = {"train": "train", "val": "valid", "test": "test"}  # name -> Roboflow folder
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEEP_PRED_CONF = 0.05  # predictions saved for the dashboard's live confusion matrix

# Okabe-Ito colours: distinguishable with the common forms of colour blindness.
C_MODEL, C_ZERO, C_BLUE, C_GREY = "#D55E00", "#E69F00", "#0072B2", "#7f7f7f"


def log(msg: str) -> None:
    print(f"\n[train] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Decisions
# ---------------------------------------------------------------------------

def require_decisions() -> None:
    gaps = D.missing()
    if gaps:
        raise SystemExit(
            "decisions.py still has decisions only you can make:\n  "
            + "\n  ".join(gaps)
            + "\nFill them in (and justify them in REPORT.md), then run again."
        )
    if not 0 < D.BASE_RATE < 1:
        raise SystemExit("BASE_RATE must be a fraction between 0 and 1, e.g. 0.02 for 2%.")
    if D.MODEL not in {"yolo11n", "yolo11s"}:
        raise SystemExit('MODEL must be "yolo11n" or "yolo11s".')
    if not D.POSITIVE_CLASSES:
        raise SystemExit("POSITIVE_CLASSES needs at least one class name.")


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------

def download_dataset() -> Path:
    # Offline escape hatch for testing the pipeline on a local YOLO-format
    # folder. Your submitted run must download from Roboflow (leave it unset).
    if os.environ.get("DATASET_DIR"):
        return Path(os.environ["DATASET_DIR"]).resolve()

    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    project = rf.workspace(D.ROBOFLOW_WORKSPACE).project(D.ROBOFLOW_PROJECT)
    kind = getattr(project, "type", "object-detection")
    if kind not in ("object-detection", "instance-segmentation"):
        raise SystemExit(
            f"{D.ROBOFLOW_PROJECT} is a {kind} project. This pipeline is written "
            "for object detection; a classification dataset needs the -cls variant."
        )
    dest = DATA / f"{D.ROBOFLOW_WORKSPACE}-{D.ROBOFLOW_PROJECT}-v{D.ROBOFLOW_VERSION}"
    ds = project.version(int(D.ROBOFLOW_VERSION)).download("yolov11", location=str(dest))
    return Path(ds.location).resolve()


def dataset_layout(ds_dir: Path) -> tuple[Path, list[str]]:
    """Write a data.yaml with absolute paths and return it with the class names.

    Roboflow's own data.yaml uses relative paths like ../train/images that
    break depending on where training is launched from.
    """
    raw = yaml.safe_load((ds_dir / "data.yaml").read_text())
    names = raw["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    for folder in SPLITS.values():
        if not (ds_dir / folder / "images").is_dir():
            raise SystemExit(
                f"The dataset has no {folder}/ split. The brief compares models on "
                f"a test split, so pick a dataset version that has train, valid and test."
            )
    fixed = ds_dir / "data_abs.yaml"
    fixed.write_text(yaml.safe_dump({
        "path": str(ds_dir), "train": "train/images", "val": "valid/images",
        "test": "test/images", "names": dict(enumerate(names)),
    }, sort_keys=False))
    return fixed, names


def load_split(ds_dir: Path, folder: str) -> tuple[list[Path], list[np.ndarray]]:
    images = sorted(p for p in (ds_dir / folder / "images").iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES)
    gts = [ev.read_yolo_labels(ds_dir / folder / "labels" / f"{p.stem}.txt") for p in images]
    return images, gts


def class_ids(names: list[str], wanted: list[str], what: str) -> list[int]:
    bad = [w for w in wanted if w not in names]
    if bad:
        raise SystemExit(f"{what} {bad} not in the dataset. Its classes are: {names}")
    return [names.index(w) for w in wanted]


# ---------------------------------------------------------------------------
# 3. Describe the dataset
# ---------------------------------------------------------------------------

def describe_dataset(names, splits, positive_ids) -> dict:
    rows, sizes = [], []
    summary = {"classes": names, "positive_classes": list(D.POSITIVE_CLASSES), "splits": {}}
    for split, (paths, gts) in splits.items():
        counts = np.zeros(len(names), int)
        for g in gts:
            counts += np.bincount(g[:, 0].astype(int), minlength=len(names))[: len(names)]
            for c, x1, y1, x2, y2 in g:
                sizes.append({"split": split, "class": names[int(c)],
                              "area_share": (x2 - x1) * (y2 - y1)})
        labels = [ev.image_label(g, positive_ids) for g in gts]
        summary["splits"][split] = {
            "images": len(paths),
            "images_without_objects": int(sum(len(g) == 0 for g in gts)),
            "positive_images": int(sum(labels)),
            "positive_image_share": float(np.mean(labels)) if labels else 0.0,
            "instances": int(counts.sum()),
        }
        for i, n in enumerate(names):
            rows.append({"split": split, "class": n, "instances": int(counts[i]),
                         "images_containing": int(sum((g[:, 0] == i).any() for g in gts if len(g)))})

    inst = pd.DataFrame(rows)
    size = pd.DataFrame(sizes, columns=["split", "class", "area_share"])
    bins = [0, 0.01, 0.05, 0.25, 1.0001]
    bucket_names = ["<1% of image", "1-5%", "5-25%", ">25%"]
    size_rows = []
    for n in names:
        a = size.loc[size["class"] == n, "area_share"]
        hist = np.histogram(a, bins=bins)[0] if len(a) else np.zeros(4, int)
        size_rows.append({"class": n, "boxes": len(a),
                          "median_area_share": float(a.median()) if len(a) else float("nan"),
                          "p10_area_share": float(a.quantile(0.1)) if len(a) else float("nan"),
                          "p90_area_share": float(a.quantile(0.9)) if len(a) else float("nan"),
                          **{b: int(h) for b, h in zip(bucket_names, hist)}})
    size_tbl = pd.DataFrame(size_rows)

    # Image resolutions (reads headers only, so this is fast).
    res = pd.Series([f"{w}x{h}" for paths, _ in splits.values() for w, h in
                     (Image.open(p).size for p in paths)]).value_counts()
    summary["image_resolutions"] = {k: int(v) for k, v in res.head(5).items()}
    summary["distinct_resolutions"] = int(res.size)

    inst.to_csv(ART / "dataset_instances.csv", index=False)
    size_tbl.to_csv(ART / "object_sizes.csv", index=False)
    (ART / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    # Plots for REPORT.md
    pivot = inst.pivot(index="class", columns="split", values="instances")[list(SPLITS)]
    ax = pivot.plot.barh(figsize=(8, 0.5 * len(names) + 2), color=[C_BLUE, C_ZERO, C_MODEL])
    ax.set_xlabel("instances")
    ax.set_title("Instances per class and split")
    for c in ax.containers:
        ax.bar_label(c, fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(ART / "plots" / "class_instances.png", dpi=130)
    plt.close()

    if len(size):
        fig, ax = plt.subplots(figsize=(8, 4))
        for n in names:
            a = size.loc[size["class"] == n, "area_share"].clip(lower=1e-5)
            if len(a):
                ax.hist(a, bins=np.logspace(-5, 0, 40), histtype="step", lw=1.5, label=n)
        ax.set_xscale("log")
        ax.set_xlabel("box area as share of image (log scale)")
        ax.set_ylabel("boxes")
        ax.legend()
        ax.set_title("Object size relative to image")
        plt.tight_layout()
        plt.savefig(ART / "plots" / "object_size.png", dpi=130)
        plt.close()

    return summary | {"_instances": inst, "_sizes": size_tbl}


# ---------------------------------------------------------------------------
# 4. Finetune
# ---------------------------------------------------------------------------

def finetune(data_yaml: Path) -> dict:
    from ultralytics import YOLO

    if os.environ.get("REUSE_WEIGHTS") == "1":
        if not WEIGHTS.is_file():
            raise SystemExit("REUSE_WEIGHTS=1 but weights/best.pt does not exist yet.")
        log(f"REUSE_WEIGHTS=1: skipping finetune, evaluating {WEIGHTS.relative_to(ROOT)}")
        old = ART / "run_info.json"
        return json.loads(old.read_text()).get("training", {}) if old.is_file() else {}

    log(f"Finetuning {D.MODEL}.pt for up to {D.EPOCHS} epochs at imgsz {D.IMGSZ}")
    started = time.time()
    model = YOLO(f"{D.MODEL}.pt")
    model.train(
        data=str(data_yaml), epochs=D.EPOCHS, imgsz=D.IMGSZ, batch=D.BATCH,
        patience=D.PATIENCE, seed=D.SEED, deterministic=True, device=D.DEVICE,
        project=str(ROOT / "runs"), name="finetune", exist_ok=True, plots=True,
    )
    best = Path(model.trainer.best)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy2(best, WEIGHTS)

    train_dir = best.parent.parent
    (ART / "training").mkdir(exist_ok=True)
    for f in ("results.csv", "results.png"):
        if (train_dir / f).is_file():
            shutil.copy2(train_dir / f, ART / "training" / f)
    results_csv = train_dir / "results.csv"
    epochs_run = len(pd.read_csv(results_csv)) if results_csv.is_file() else None

    import torch

    return {
        "model": D.MODEL, "epochs_max": D.EPOCHS, "epochs_run": epochs_run,
        "imgsz": D.IMGSZ, "batch": D.BATCH, "patience": D.PATIENCE, "seed": D.SEED,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "minutes": round((time.time() - started) / 60, 1),
    }


# ---------------------------------------------------------------------------
# 5. Predict on the test split
# ---------------------------------------------------------------------------

def predict(model, paths, class_map: dict[int, int] | None) -> list[np.ndarray]:
    """Run the model on every image at a very low confidence floor.

    Keeping low-confidence boxes is what lets the threshold be swept later.
    class_map turns model class ids into dataset class ids; boxes of classes
    it does not mention are dropped (used for the zero-shot COCO model).
    """
    out = []
    for r in model.predict(source=[str(p) for p in paths], conf=0.001, iou=0.7,
                           imgsz=D.IMGSZ, max_det=300, stream=True, verbose=False,
                           device=D.DEVICE):
        b = r.boxes
        cls = b.cls.cpu().numpy().astype(int)
        arr = (np.column_stack([cls, b.conf.cpu().numpy(), b.xyxyn.cpu().numpy()])
               if len(cls) else ev.EMPTY_PRED.copy())
        if class_map is not None and len(arr):
            arr = arr[np.isin(cls, list(class_map))]
            arr[:, 0] = [class_map[int(c)] for c in arr[:, 0]]
        out.append(arr if len(arr) else ev.EMPTY_PRED.copy())
    return out


def zero_shot_map(coco_names: dict, names: list[str]) -> dict[int, int]:
    bad = [k for k in D.COCO_TO_DATASET if k not in coco_names.values()]
    if bad:
        raise SystemExit(f"COCO_TO_DATASET keys {bad} are not COCO classes. "
                         f"COCO classes: {sorted(coco_names.values())}")
    class_ids(names, list(D.COCO_TO_DATASET.values()), "COCO_TO_DATASET targets")
    return {i: names.index(D.COCO_TO_DATASET[n]) for i, n in coco_names.items()
            if n in D.COCO_TO_DATASET}


# ---------------------------------------------------------------------------
# Plots and failure gallery
# ---------------------------------------------------------------------------

def plot_cost_curve(sweeps: dict, strat: dict, threshold: float, band: tuple) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(band[0], band[1], color=C_BLUE, alpha=0.12,
               label=f"review band {band[0]:.2f}-{band[1]:.2f}")
    for name, colour in (("finetuned", C_MODEL), ("zero_shot", C_ZERO)):
        ax.plot(sweeps[name]["threshold"], sweeps[name]["cost_per_1000"], color=colour,
                lw=2.2 if name == "finetuned" else 1.4, label=f"{name.replace('_', '-')} model alone")
    for key, style in (("Do nothing", ":"), ("Person checks everything", "--"), ("Fail everything", "-.")):
        k = next(s for s in strat if s.startswith(key))
        ax.axhline(strat[k]["cost_per_1000"], color=C_GREY, ls=style, lw=1, label=k)
    y = float(np.interp(threshold, sweeps["finetuned"]["threshold"], sweeps["finetuned"]["cost_per_1000"]))
    ax.plot([threshold], [y], "o", color="black", ms=8, zorder=5)
    ax.annotate(f"chosen t = {threshold:.2f}\n€{y:,.0f} per 1,000", (threshold, y),
                xytext=(10, 18), textcoords="offset points", fontsize=9)
    ax.set_xlabel("confidence threshold (image score ≥ t → fail)")
    ax.set_ylabel("expected cost per 1,000 cases (€)")
    ax.set_title(f"Cost against threshold at base rate {D.BASE_RATE:.2%}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ART / "plots" / "cost_vs_threshold.png", dpi=130)
    plt.close()


def plot_confusion(cm: np.ndarray, names: list[str], conf: float) -> None:
    labels = names + ["background"]
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3, 1.0 * len(labels) + 2.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Finetuned model, test split, conf ≥ {conf:.2f}, IoU ≥ 0.5")
    plt.tight_layout()
    plt.savefig(ART / "plots" / "confusion_matrix.png", dpi=130)
    plt.close()


def save_failures(paths, gts, preds, scores, labels, names, threshold, n=6) -> pd.DataFrame:
    """The most confident misses and false alarms, drawn, for the report and app.

    Green boxes are the true labels, orange boxes what the model predicted at
    the chosen threshold.
    """
    out_dir = ART / "failures"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    scores, labels = np.asarray(scores), np.asarray(labels)
    fn = [i for i in np.argsort(scores, kind="stable") if labels[i] == 1 and scores[i] < threshold][:n]
    fp = [i for i in np.argsort(-scores, kind="stable") if labels[i] == 0 and scores[i] >= threshold][:n]
    rows = []
    for kind, idxs in (("missed", fn), ("false_alarm", fp)):
        for rank, i in enumerate(idxs):
            img = cv2.imread(str(paths[i]))
            if img is None:
                continue
            h, w = img.shape[:2]
            for c, x1, y1, x2, y2 in gts[i]:
                p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
                cv2.rectangle(img, p1, p2, (115, 158, 0), 2)
                cv2.putText(img, f"true {names[int(c)]}", (p1[0], max(p1[1] - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (115, 158, 0), 1, cv2.LINE_AA)
            shown = preds[i][preds[i][:, 1] >= threshold] if len(preds[i]) else []
            for c, cf, x1, y1, x2, y2 in shown:
                p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
                cv2.rectangle(img, p1, p2, (0, 94, 213), 2)
                cv2.putText(img, f"{names[int(c)]} {cf:.2f}", (p1[0], min(p2[1] + 14, h - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 94, 213), 1, cv2.LINE_AA)
            scale = 640 / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            fname = f"{kind}_{rank + 1}.jpg"
            cv2.imwrite(str(out_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            rows.append({"file": fname, "kind": kind, "image": paths[i].name,
                         "image_score": round(float(scores[i]), 4),
                         "true_classes": ", ".join(sorted({names[int(c)] for c in gts[i][:, 0]})) or "(none)"})
    df = pd.DataFrame(rows, columns=["file", "kind", "image", "image_score", "true_classes"])
    df.to_csv(out_dir / "failures.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Markdown tables to paste into REPORT.md (numbers only; the analysis is yours)
# ---------------------------------------------------------------------------

def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_tables(stats, det, op, a, sens, switches, failures, training) -> None:
    md = ["# Results tables\n",
          "Generated by train.py. Numbers only: the interpretation in REPORT.md is yours.\n",
          f"Training settings: `{json.dumps(training)}`\n"]

    md.append("## Dataset composition\n")
    md.append(md_table(pd.DataFrame([{"split": s, **v} for s, v in stats["splits"].items()])) + "\n")
    inst = stats["_instances"].pivot(index="class", columns="split", values="instances")[list(SPLITS)]
    md.append(md_table(inst.reset_index()) + "\n")
    md.append("Object size, box area as a share of the image, all splits:\n")
    md.append(md_table(stats["_sizes"], "{:.4f}") + "\n")
    md.append(f"Most common resolutions: {stats['image_resolutions']} "
              f"({stats['distinct_resolutions']} distinct)\n")

    md.append("## Models on the test split\n")
    rows = [{"model": m, "mAP50": d["mAP50"], "mAP50-95": d["mAP50-95"],
             "image-level ROC AUC": d["image_level"]["roc_auc"],
             "image-level AP": d["image_level"]["average_precision"]} for m, d in det.items()]
    md.append(md_table(pd.DataFrame(rows)) + "\n")
    for m, d in det.items():
        md.append(f"Per class, {m}. Precision and recall at confidence ≥ {d['conf']:.2f}, IoU ≥ 0.5:\n")
        md.append(md_table(pd.DataFrame(d["per_class"])) + "\n")

    md.append("## Operating point and strategies, per 1,000 cases at the assumed base rate\n")
    md.append(f"Assumptions: `{json.dumps(a.to_dict())}`\n")
    cap = op["band_ignoring_capacity_outcome"]
    md.append(f"Threshold {op['threshold']:.2f} ({op['threshold_source']}). "
              f"Review band {op['band'][0]:.2f}-{op['band'][1]:.2f} ({op['band_source']}). "
              f"With no capacity limit the cheapest band is {op['band_ignoring_capacity'][0]:.2f}-"
              f"{op['band_ignoring_capacity'][1]:.2f}, costing €{cap['cost_per_1000']:,.2f} per 1,000 "
              f"and needing {cap['reviewers_needed']:.2f} reviewers. "
              f"Test split positive share: {op['test_positive_share']:.1%} of {op['test_images']} images.\n")
    rows = [{"strategy": k, "cost €/1000": v["cost_per_1000"], "errors €": v["error_cost_per_1000"],
             "review €": v["review_cost_per_1000"], "missed (FN)": v["fn"],
             "false alarms (FP)": v["fp"], "review share": v["review_share"],
             "reviews/day": v["reviews_per_day"], "reviewers needed": v["reviewers_needed"],
             f"staffable with {a.reviewers_available:g}": "yes" if v["feasible"] else "no"}
            for k, v in op["strategies"].items()]
    z, t = op["zero_shot_alone"], op["trivial_majority"]
    rows.append({"strategy": f"Zero-shot COCO model alone (t={z['threshold']:.2f})",
                 "cost €/1000": z["cost_per_1000"], "errors €": z["error_cost_per_1000"],
                 "review €": 0.0, "missed (FN)": z["fn"], "false alarms (FP)": z["fp"],
                 "review share": 0.0, "reviews/day": 0.0, "reviewers needed": 0.0,
                 f"staffable with {a.reviewers_available:g}": "yes"})
    rows.append({"strategy": f"Trivial: majority class from train ({t['predicts']})",
                 "cost €/1000": t["cost_per_1000"], "errors €": t["error_cost_per_1000"],
                 "review €": 0.0, "missed (FN)": t["fn"], "false alarms (FP)": t["fp"],
                 "review share": 0.0, "reviews/day": 0.0, "reviewers needed": 0.0,
                 f"staffable with {a.reviewers_available:g}": "yes"})
    md.append(md_table(pd.DataFrame(rows), "{:,.2f}") + "\n")
    b = op["bootstrap"]
    md.append(f"Bootstrap over test images ({b['resamples']} resamples), 95% intervals: "
              f"model alone €{b['model_alone_cost_95ci'][0]:,.0f}-{b['model_alone_cost_95ci'][1]:,.0f}; "
              f"model + review €{b['model_review_cost_95ci'][0]:,.0f}-{b['model_review_cost_95ci'][1]:,.0f}; "
              f"cheapest threshold {b['best_threshold_95ci'][0]:.2f}-{b['best_threshold_95ci'][1]:.2f}.\n")

    md.append("## Where the cheapest strategy changes, one assumption varied at a time\n")
    md.append((md_table(pd.DataFrame(switches), "{:.4g}") if switches
               else "No change between 1/100x and 100x of any assumption.") + "\n")
    md.append("## Sensitivity\n")
    md.append(md_table(sens, "{:,.3g}") + "\n")
    if len(failures):
        md.append("## Most confident failures at the chosen threshold (images in artifacts/failures/)\n")
        md.append(md_table(failures, "{:.3f}") + "\n")
    (ART / "results_tables.md").write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    require_decisions()
    for d in (ART, ART / "plots"):
        d.mkdir(parents=True, exist_ok=True)

    log("Getting the dataset")
    ds_dir = download_dataset()
    data_yaml, names = dataset_layout(ds_dir)
    positive_ids = class_ids(names, list(D.POSITIVE_CLASSES), "POSITIVE_CLASSES")
    splits = {s: load_split(ds_dir, f) for s, f in SPLITS.items()}
    log(f"Classes: {names}. Positive: {list(D.POSITIVE_CLASSES)}. "
        + ", ".join(f"{s}: {len(p)} images" for s, (p, _) in splits.items()))

    log("Describing the dataset")
    stats = describe_dataset(names, splits, positive_ids)

    training = finetune(data_yaml)

    from ultralytics import YOLO

    test_paths, test_gts = splits["test"]
    log(f"Predicting on {len(test_paths)} test images with the finetuned model")
    ft_preds = predict(YOLO(str(WEIGHTS)), test_paths, None)

    log("Predicting with the pretrained COCO model, zero-shot")
    coco = YOLO(f"{D.MODEL}.pt")
    cmap = zero_shot_map(coco.names, names)
    zs_preds = predict(coco, test_paths, cmap) if cmap else [ev.EMPTY_PRED.copy() for _ in test_paths]
    trivial_preds = [ev.EMPTY_PRED.copy() for _ in test_paths]  # predicts nothing

    # One decision per image: label from ground truth, score from the model.
    labels = np.array([ev.image_label(g, positive_ids) for g in test_gts])
    if labels.min() == labels.max():
        raise SystemExit("Every test image has the same label, so no threshold can be evaluated. "
                         "Choose positive classes that appear in some test images but not all.")
    train_labels = [ev.image_label(g, positive_ids) for g in splits["train"][1]]
    majority_positive = float(np.mean(train_labels)) > 0.5  # decided on train, not test
    scores = {
        "finetuned": np.array([ev.image_score(p, positive_ids) for p in ft_preds]),
        "zero_shot": np.array([ev.image_score(p, positive_ids) for p in zs_preds]),
        "trivial": np.full(len(labels), 1.0 if majority_positive else 0.0),
    }

    log("Costing: sweeping thresholds and review bands")
    a = costing.Assumptions.from_decisions(D)
    sweeps = {m: costing.threshold_sweep(scores[m], labels, a) for m in ("finetuned", "zero_shot")}
    best_t = {m: costing.best_threshold(scores[m], labels, a) for m in ("finetuned", "zero_shot")}
    threshold = float(D.THRESHOLD_OVERRIDE) if D.THRESHOLD_OVERRIDE is not None else best_t["finetuned"]
    band = (tuple(map(float, D.REVIEW_BAND_OVERRIDE)) if D.REVIEW_BAND_OVERRIDE is not None
            else costing.best_band(scores["finetuned"], labels, a))
    band_uncapped = costing.best_band(scores["finetuned"], labels, a, respect_capacity=False)
    strat = costing.strategies(scores["finetuned"], labels, a, threshold, band)
    zs_alone = costing.band_outcomes(scores["zero_shot"], labels, best_t["zero_shot"], best_t["zero_shot"], a)
    trivial = costing.band_outcomes(scores["trivial"], labels, 0.5, 0.5, a)
    uncapped = costing.band_outcomes(scores["finetuned"], labels, *band_uncapped, a)
    boot = costing.bootstrap(scores["finetuned"], labels, a, threshold, band)
    sens = pd.DataFrame(costing.sensitivity(scores["finetuned"], labels, a))
    switches = costing.switch_points(scores["finetuned"], labels, a)

    log("Detection metrics per class, for all three models")
    # Each model's precision/recall is read at its own cost-minimising threshold.
    model_conf = {"finetuned": threshold, "zero_shot": best_t["zero_shot"], "trivial": threshold}
    preds = {"finetuned": ft_preds, "zero_shot": zs_preds, "trivial": trivial_preds}
    det = {m: ev.detection_metrics(test_gts, preds[m], names, model_conf[m]) for m in preds}
    for m in det:
        det[m]["image_level"] = ev.ranking_quality(scores[m], labels)
    cm = ev.confusion_matrix(test_gts, ft_preds, len(names), threshold)
    det["finetuned"]["confusion_matrix"] = cm.tolist()

    log("Saving artifacts")
    plot_cost_curve(sweeps, strat, threshold, band)
    plot_confusion(cm, names, threshold)
    failures = save_failures(test_paths, test_gts, ft_preds, scores["finetuned"], labels, names, threshold)

    pd.DataFrame({"image": [p.name for p in test_paths], "label": labels,
                  **{m: np.round(s, 5) for m, s in scores.items()}}).to_csv(ART / "image_scores.csv", index=False)
    sweep_df = pd.DataFrame({"threshold": costing.GRID})
    for m, sw in sweeps.items():
        for k in ("cost_per_1000", "recall_auto", "false_alarm_rate_auto", "fp", "fn"):
            sweep_df[f"{m}_{k}"] = np.round(sw[k], 4)
    sweep_df.to_csv(ART / "cost_sweep.csv", index=False)
    sens.to_csv(ART / "sensitivity.csv", index=False)
    (ART / "detection_metrics.json").write_text(json.dumps(det, indent=2))
    (ART / "test_predictions.json").write_text(json.dumps({
        "images": [p.name for p in test_paths],
        "gt": [np.round(g, 5).tolist() for g in test_gts],
        "finetuned": [np.round(p[p[:, 1] >= KEEP_PRED_CONF], 4).tolist() if len(p) else []
                      for p in ft_preds],
    }))
    (ART / "assumptions.json").write_text(json.dumps(
        a.to_dict() | {"labels": [D.LABEL_PASS, D.LABEL_REVIEW, D.LABEL_FAIL]}, indent=2))
    operating = {
        "threshold": threshold,
        "threshold_source": "override" if D.THRESHOLD_OVERRIDE is not None else "sweep minimum",
        "band": list(band),
        "band_source": ("override" if D.REVIEW_BAND_OVERRIDE is not None
                        else "sweep minimum within reviewer capacity"),
        "band_ignoring_capacity": list(band_uncapped),
        "band_ignoring_capacity_outcome": costing._scalar(uncapped),
        "strategies": strat,
        "zero_shot_alone": {"threshold": best_t["zero_shot"]} | costing._scalar(zs_alone),
        "trivial_majority": {"predicts": "positive" if majority_positive else "negative"}
                            | costing._scalar(trivial),
        "bootstrap": boot,
        "switch_points": switches,
        "test_positive_share": float(labels.mean()),
        "test_images": int(len(labels)),
    }
    (ART / "operating_point.json").write_text(json.dumps(operating, indent=2))
    (ART / "run_info.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "workspace": D.ROBOFLOW_WORKSPACE, "project": D.ROBOFLOW_PROJECT,
            "version": D.ROBOFLOW_VERSION,
            "url": f"https://universe.roboflow.com/{D.ROBOFLOW_WORKSPACE}/"
                   f"{D.ROBOFLOW_PROJECT}/dataset/{D.ROBOFLOW_VERSION}",
        },
        "classes": names, "positive_classes": list(D.POSITIVE_CLASSES),
        "coco_to_dataset": D.COCO_TO_DATASET, "training": training,
        "python": platform.python_version(),
    }, indent=2))

    write_tables(stats, det, operating, a, sens, switches, failures, training)
    log("Done. Commit weights/best.pt and artifacts/. "
        "Tables to paste into REPORT.md: artifacts/results_tables.md")


if __name__ == "__main__":
    main()
