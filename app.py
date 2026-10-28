"""
Assignment 2: Gradio dashboard. Run with

    uv run python main.py app

Built only from what `main.py train` produced: weights/best.pt and artifacts/.
It never trains. Every cost is recomputed with costing.py, the same code the
pipeline used, so the dashboard and REPORT.md cannot disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import costing
import evaluation as ev

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"

# One colour per outcome, used everywhere (Okabe-Ito, colour-blind safe).
PASS_C, REVIEW_C, FAIL_C, ZERO_C, INK = "#0072B2", "#E69F00", "#D55E00", "#999999", "#1d2731"


# ---------------------------------------------------------------------------
# Load what training produced
# ---------------------------------------------------------------------------

def _load():
    needed = ["run_info.json", "assumptions.json", "operating_point.json", "image_scores.csv",
              "detection_metrics.json", "test_predictions.json", "dataset_summary.json"]
    gaps = [n for n in needed if not (ART / n).is_file()]
    if gaps or not WEIGHTS.is_file():
        raise SystemExit(
            f"Missing {gaps or ''} {'' if WEIGHTS.is_file() else 'weights/best.pt'}. "
            "Run `uv run python main.py train` and commit weights/ and artifacts/."
        )
    j = lambda n: json.loads((ART / n).read_text())  # noqa: E731
    return (j("run_info.json"), j("assumptions.json"), j("operating_point.json"),
            pd.read_csv(ART / "image_scores.csv"), j("detection_metrics.json"),
            j("test_predictions.json"), j("dataset_summary.json"))


RUN, ASSUME, OP, SCORES, DET, TESTPRED, DSUM = _load()
NAMES = RUN["classes"]
POS_IDS = [NAMES.index(c) for c in RUN["positive_classes"]]
LBL_PASS, LBL_REVIEW, LBL_FAIL = ASSUME.get("labels", ["Pass", "Human review", "Fail"])
BASE = costing.Assumptions.from_dict(ASSUME)
T_CHOSEN = OP["threshold"]
BAND = tuple(OP["band"])
S_FT, S_ZS, Y = SCORES["finetuned"].to_numpy(), SCORES["zero_shot"].to_numpy(), SCORES["label"].to_numpy()

# Predictions on the test split, matched once at startup so per-class
# precision/recall can be read at any confidence instantly.
GTS = [np.array(g, float).reshape(-1, 5) for g in TESTPRED["gt"]]
PREDS = [np.array(p, float).reshape(-1, 6) for p in TESTPRED["finetuned"]]
_flat = [(p[:, 0], p[:, 1], ev.match_image(g, p)[:, 0]) for g, p in zip(GTS, PREDS) if len(p)]
P_CLS = np.concatenate([f[0] for f in _flat]) if _flat else np.zeros(0)
P_CONF = np.concatenate([f[1] for f in _flat]) if _flat else np.zeros(0)
P_TP = np.concatenate([f[2] for f in _flat]) if _flat else np.zeros(0, bool)
N_GT = np.bincount(np.concatenate([g[:, 0] for g in GTS]).astype(int), minlength=len(NAMES))

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(WEIGHTS))
    return _MODEL


# ---------------------------------------------------------------------------
# Decision tab: everything that moves with the two thresholds
# ---------------------------------------------------------------------------

def _assumptions(cost_fp, cost_fn, base_rate_pct, review_minutes, reviewers):
    d = BASE.to_dict()
    d.update(cost_fp=float(cost_fp), cost_fn=float(cost_fn), base_rate=float(base_rate_pct) / 100,
             review_minutes=float(review_minutes), reviewers_available=float(reviewers))
    return costing.Assumptions.from_dict(d)


def _kpis(o: dict, a) -> str:
    staffed = o["reviewers_needed"] <= a.reviewers_available + 1e-9
    staff_note = ("within" if staffed else "over") + f" the {a.reviewers_available:g} available"
    cells = [
        ("Expected cost per 1,000 cases", f"€{o['cost_per_1000']:,.0f}",
         f"€{o['error_cost_per_1000']:,.0f} errors + €{o['review_cost_per_1000']:,.0f} review time"),
        ("Sent to a person", f"{o['review_share']:.1%}", f"{o['reviews_per_day']:,.0f} cases a day"),
        ("Reviewers needed", f"{o['reviewers_needed']:.2f}", staff_note),
        ("Positives missed per 1,000", f"{o['fn']:.1f}", f"{o['fp']:.1f} false alarms per 1,000"),
    ]
    warn = "" if staffed else " kpi-warn"
    html = "".join(
        f'<div class="kpi{warn if i == 2 else ""}"><div class="kpi-label">{t}</div>'
        f'<div class="kpi-value">{v}</div><div class="kpi-note">{n}</div></div>'
        for i, (t, v, n) in enumerate(cells))
    return f'<div class="kpis">{html}</div>'


def _matrices(o: dict, lo: float, hi: float):
    pos = "Actually " + " / ".join(RUN["positive_classes"])
    per1000 = pd.DataFrame({
        "": [pos, "Actually negative"],
        LBL_PASS: [o["pos_pass"], o["neg_pass"]],
        LBL_REVIEW: [o["pos_review"], o["neg_review"]],
        LBL_FAIL: [o["pos_fail"], o["neg_fail"]],
    }).round(1)
    zone = np.where(S_FT < lo, 0, np.where(S_FT >= hi, 2, 1))
    raw = pd.DataFrame({
        "": [pos, "Actually negative"],
        **{lab: [int(((Y == 1) & (zone == k)).sum()), int(((Y == 0) & (zone == k)).sum())]
           for k, lab in enumerate((LBL_PASS, LBL_REVIEW, LBL_FAIL))},
    })
    return per1000, raw


def _cost_figure(a, lo: float, hi: float) -> go.Figure:
    ft = costing.threshold_sweep(S_FT, Y, a)
    zs = costing.threshold_sweep(S_ZS, Y, a)
    strat = costing.strategies(S_FT, Y, a, T_CHOSEN, (lo, hi))
    fig = go.Figure()
    fig.add_vrect(x0=lo, x1=hi, fillcolor=REVIEW_C, opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.add_trace(go.Scatter(x=zs["threshold"], y=zs["cost_per_1000"], name="Zero-shot COCO model alone",
                             line=dict(color=ZERO_C, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=ft["threshold"], y=ft["cost_per_1000"], name="Finetuned model alone",
                             line=dict(color=FAIL_C, width=3)))
    for key, dash in (("Do nothing", "dash"), ("Person checks everything", "longdash")):
        k = next(s for s in strat if s.startswith(key))
        fig.add_hline(y=strat[k]["cost_per_1000"], line=dict(color="#555", width=1, dash=dash),
                      annotation_text=k, annotation_position="top right")
    y_chosen = float(np.interp(T_CHOSEN, ft["threshold"], ft["cost_per_1000"]))
    fig.add_trace(go.Scatter(x=[T_CHOSEN], y=[y_chosen], mode="markers+text", name="Chosen threshold",
                             marker=dict(size=13, color=INK, symbol="diamond"),
                             text=[f"chosen {T_CHOSEN:.2f}"], textposition="top center"))
    t_now = costing.best_threshold(S_FT, Y, a)
    if abs(t_now - T_CHOSEN) > 1e-9:
        y_now = float(np.interp(t_now, ft["threshold"], ft["cost_per_1000"]))
        fig.add_trace(go.Scatter(x=[t_now], y=[y_now], mode="markers", name="Cheapest under these inputs",
                                 marker=dict(size=11, color=INK, symbol="diamond-open")))
    top = max(float(np.max(ft["cost_per_1000"])), strat[next(iter(strat))]["cost_per_1000"])
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white",
        xaxis_title="Confidence threshold (image score at or above it is failed)",
        yaxis_title="Expected cost per 1,000 cases (€)", yaxis_range=[0, top * 1.08],
        legend=dict(orientation="h", y=-0.22), hovermode="x unified",
    )
    return fig


def _strategy_table(a, lo: float, hi: float) -> pd.DataFrame:
    s = costing.strategies(S_FT, Y, a, T_CHOSEN, (lo, hi))
    rows = [{"Strategy": k, "€ per 1,000": round(v["cost_per_1000"]),
             "Missed": round(v["fn"], 1), "False alarms": round(v["fp"], 1),
             "Reviews a day": round(v["reviews_per_day"]),
             "Reviewers needed": round(v["reviewers_needed"], 2),
             "Staffable": "yes" if v["feasible"] else "no"} for k, v in s.items()]
    return pd.DataFrame(rows).sort_values("€ per 1,000", kind="stable")


def update_decision(low, high, cost_fp, cost_fn, base_rate_pct, review_minutes, reviewers):
    lo, hi = sorted((float(low), float(high)))
    a = _assumptions(cost_fp, cost_fn, base_rate_pct, review_minutes, reviewers)
    o = costing._scalar(costing.band_outcomes(S_FT, Y, lo, hi, a))
    per1000, raw = _matrices(o, lo, hi)
    note = "" if float(low) <= float(high) else "The left slider is above the right one, so they were swapped."
    return (_kpis(o, a), per1000, raw, _cost_figure(a, lo, hi), _strategy_table(a, lo, hi), note)


def reset_band():
    return BAND[0], BAND[1]


def reset_assumptions():
    return (BASE.cost_fp, BASE.cost_fn, BASE.base_rate * 100, BASE.review_minutes,
            BASE.reviewers_available)


# ---------------------------------------------------------------------------
# Image tab
# ---------------------------------------------------------------------------

def predict_image(img, low, high):
    if img is None:
        return None, "Upload an image to see what the model finds in it.", None
    lo, hi = sorted((float(low), float(high)))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Ultralytics expects BGR arrays
    imgsz = RUN.get("training", {}).get("imgsz")  # predict at the size the pipeline used
    r = model().predict(bgr, conf=max(lo, 0.001), verbose=False, **({"imgsz": imgsz} if imgsz else {}))[0]
    boxes = r.boxes
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    out = img.copy()
    rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))  # noqa: E731
    rows = []
    for c, cf, (x1, y1, x2, y2) in zip(cls, conf, xyxy):
        zone = LBL_FAIL if cf >= hi else LBL_REVIEW
        colour = rgb(FAIL_C if cf >= hi else REVIEW_C)
        thick = 3 if c in POS_IDS else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)
        cv2.putText(out, f"{NAMES[c]} {cf:.2f}", (x1, max(y1 - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 2, cv2.LINE_AA)
        rows.append({"Class": NAMES[c], "Confidence": round(float(cf), 3),
                     "Counts toward decision": "yes" if c in POS_IDS else "no", "Band": zone})
    pos_conf = [cf for c, cf in zip(cls, conf) if c in POS_IDS]
    score = float(max(pos_conf)) if pos_conf else 0.0
    decision, colour = ((LBL_FAIL, FAIL_C) if score >= hi else
                        (LBL_REVIEW, REVIEW_C) if score >= lo else (LBL_PASS, PASS_C))
    verdict = (f'<div class="verdict" style="border-color:{colour}"><span style="color:{colour}">'
               f'{decision}</span><br><small>Image score {score:.3f}: the highest confidence for '
               f'{", ".join(RUN["positive_classes"])}. Pass below {lo:.2f}, fail at or above {hi:.2f}.'
               f'</small></div>')
    table = pd.DataFrame(rows) if rows else pd.DataFrame({"Class": ["Nothing found above the pass threshold"]})
    return out, verdict, table


# ---------------------------------------------------------------------------
# Per-class tab
# ---------------------------------------------------------------------------

def model_table() -> pd.DataFrame:
    labels = {"finetuned": "Finetuned", "zero_shot": "Zero-shot COCO", "trivial": "Trivial (predicts nothing)"}
    return pd.DataFrame([{
        "Model": labels[m], "mAP50": round(d["mAP50"], 3), "mAP50-95": round(d["mAP50-95"], 3),
        "Image-level ROC AUC": round(d["image_level"]["roc_auc"], 3),
        "Image-level AP": round(d["image_level"]["average_precision"], 3),
    } for m, d in DET.items()])


def per_class(conf):
    conf = float(conf)
    keep = P_CONF >= conf
    rows = []
    for i, n in enumerate(NAMES):
        sel = keep & (P_CLS == i)
        npred, ntp = int(sel.sum()), int(P_TP[sel].sum())
        fixed = {r["class"]: r for r in DET["finetuned"]["per_class"]}[n]
        zs = {r["class"]: r for r in DET["zero_shot"]["per_class"]}[n]
        rows.append({
            "Class": n + (" (positive)" if i in POS_IDS else ""),
            "Test instances": int(N_GT[i]),
            "Precision": round(ntp / npred, 3) if npred else 0.0,
            "Recall": round(ntp / N_GT[i], 3) if N_GT[i] else None,
            "AP50": round(fixed["AP50"], 3), "AP50-95": round(fixed["AP50-95"], 3),
            "Zero-shot AP50": round(zs["AP50"], 3),
        })
    cm = ev.confusion_matrix(GTS, PREDS, len(NAMES), conf)
    labels = NAMES + ["background"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", showscale=False,
                               text=cm, texttemplate="%{text}", hovertemplate="true %{y}<br>predicted %{x}<br>%{z} boxes<extra></extra>"))
    fig.update_layout(height=120 + 60 * len(labels), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed",
                      template="plotly_white")
    return pd.DataFrame(rows), fig


def failure_gallery():
    f = ART / "failures" / "failures.csv"
    if not f.is_file():
        return []
    df = pd.read_csv(f)
    words = {"missed": "Missed", "false_alarm": "False alarm"}
    return [(str(ART / "failures" / r.file),
             f"{words.get(r.kind, r.kind)}: score {r.image_score:.3f}, truly {r.true_classes}")
            for r in df.itertuples() if (ART / "failures" / r.file).is_file()]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

CSS = f"""
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.kpi {{ border-left:4px solid {INK}; padding:6px 12px; }}
.kpi-warn {{ border-left-color:{FAIL_C}; }}
.kpi-label {{ font-size:0.85rem; opacity:0.75; }}
.kpi-value {{ font-size:1.9rem; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.15; }}
.kpi-note {{ font-size:0.8rem; opacity:0.7; }}
.verdict {{ border-left:6px solid; padding:8px 14px; font-size:1.6rem; font-weight:700; }}
.verdict small {{ font-size:0.85rem; font-weight:400; }}
.band-legend span {{ display:inline-block; padding:1px 10px; margin-right:6px; border-radius:3px; color:white; }}
"""


def build() -> gr.Blocks:
    ds = RUN["dataset"]
    test = DSUM["splits"]["test"]
    with gr.Blocks(title="Pass, review or fail") as demo:
        gr.Markdown(
            f"# {ds['project']}: pass, review or fail\n"
            f"Finetuned **{RUN['training'].get('model', 'YOLO')}** on "
            f"[{ds['workspace']}/{ds['project']} v{ds['version']}]({ds['url']}). "
            f"Every number comes from {test['images']} test images, reweighted to a production "
            f"base rate of {BASE.base_rate:.2%} positive. Positive means the image contains "
            f"**{', '.join(RUN['positive_classes'])}**."
        )
        gr.HTML(f'<div class="band-legend"><span style="background:{PASS_C}">{LBL_PASS}</span>'
                f'<span style="background:{REVIEW_C};color:{INK}">{LBL_REVIEW}</span>'
                f'<span style="background:{FAIL_C}">{LBL_FAIL}</span></div>')
        with gr.Row():
            low = gr.Slider(0, 1, value=BAND[0], step=0.01, label=f"{LBL_PASS} below this score")
            high = gr.Slider(0, 1, value=BAND[1], step=0.01, label=f"{LBL_FAIL} at or above this score")
            reset = gr.Button("Reset to chosen band", scale=0, min_width=180)
        swap_note = gr.Markdown()
        kpis = gr.HTML()

        with gr.Tabs():
            with gr.Tab("Decision and cost"):
                gr.Markdown(f"Chosen operating point from the pipeline: threshold **{T_CHOSEN:.2f}** "
                            f"for the model alone, band **{BAND[0]:.2f} to {BAND[1]:.2f}** with review "
                            f"({OP['band_source']}).")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Where 1,000 production cases land** (before review)")
                        m1000 = gr.Dataframe(interactive=False)
                    with gr.Column():
                        gr.Markdown(f"**Where the {test['images']} test images land** (raw counts)")
                        mraw = gr.Dataframe(interactive=False)
                cost_plot = gr.Plot(label="Cost against threshold")
                gr.Markdown("**Your band against the alternatives**, cheapest first")
                strat = gr.Dataframe(interactive=False)
                with gr.Accordion("What if the assumptions were different?", open=False):
                    gr.Markdown("These start at the values the pipeline used. Changing them recomputes "
                                "every number above from the same test-set scores.")
                    with gr.Row():
                        c_fp = gr.Number(BASE.cost_fp, label="Cost of a false alarm (€)")
                        c_fn = gr.Number(BASE.cost_fn, label="Cost of a missed positive (€)")
                        br = gr.Number(round(BASE.base_rate * 100, 4), label="Base rate (% positive)")
                        rmin = gr.Number(BASE.review_minutes, label="Minutes per review")
                        nrev = gr.Number(BASE.reviewers_available, label="Reviewers available")
                    reset_a = gr.Button("Reset assumptions")

            with gr.Tab("Try an image"):
                with gr.Row():
                    inp = gr.Image(type="numpy", label="Image", height=420)
                    out = gr.Image(type="numpy", label="Predictions at the current band", height=420)
                verdict = gr.HTML()
                dets = gr.Dataframe(interactive=False, label="Detections at or above the pass threshold")

            with gr.Tab("Per-class results"):
                gr.Markdown("**All three models on the same test split**")
                gr.Dataframe(model_table(), interactive=False)
                conf = gr.Slider(0.05, 1, value=max(T_CHOSEN, 0.05), step=0.01,
                                 label="Confidence for precision, recall and the confusion matrix")
                pc = gr.Dataframe(interactive=False, label="Finetuned model per class (AP is threshold-free)")
                cm = gr.Plot(label="Object-level confusion matrix, IoU at least 0.5")
                gr.Markdown("**Most confident failures at the chosen threshold.** Green is the true "
                            "label, orange is what the model predicted.")
                gr.Gallery(failure_gallery(), columns=4, height="auto", label="Failures")

            with gr.Tab("Dataset"):
                gr.Dataframe(pd.DataFrame([{"Split": s, **v} for s, v in DSUM["splits"].items()]),
                             interactive=False)
                if (ART / "dataset_instances.csv").is_file():
                    inst = pd.read_csv(ART / "dataset_instances.csv")
                    gr.Dataframe(inst.pivot(index="class", columns="split", values="instances")
                                 .reset_index(), interactive=False, label="Instances per class")
                if (ART / "object_sizes.csv").is_file():
                    gr.Dataframe(pd.read_csv(ART / "object_sizes.csv").round(4), interactive=False,
                                 label="Box area as a share of the image")
                with gr.Row():
                    for p in ("class_instances.png", "object_size.png"):
                        if (ART / "plots" / p).is_file():
                            gr.Image(str(ART / "plots" / p), show_label=False, interactive=False)

        decision_in = [low, high, c_fp, c_fn, br, rmin, nrev]
        decision_out = [kpis, m1000, mraw, cost_plot, strat, swap_note]
        for ctl in decision_in:
            ctl.change(update_decision, decision_in, decision_out)
        reset.click(reset_band, None, [low, high])
        reset_a.click(reset_assumptions, None, [c_fp, c_fn, br, rmin, nrev])
        for ctl in (inp, low, high):
            ctl.change(predict_image, [inp, low, high], [out, verdict, dets])
        conf.change(per_class, conf, [pc, cm])
        demo.load(update_decision, decision_in, decision_out)
        demo.load(per_class, conf, [pc, cm])
    return demo


if __name__ == "__main__":
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
    )
    build().launch(theme=theme, css=CSS)
