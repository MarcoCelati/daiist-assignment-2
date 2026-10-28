"""
Turns model scores into euros. Shared by train.py and app.py, so the pipeline
and the dashboard compute every number with exactly the same formulas.

The unit of decision is one case (one image). Each test image has:
  - a label: 1 if it really contains a positive-class object, else 0
  - a score: the model's highest confidence for any positive-class object in it

A single threshold t gives two outcomes:   score >= t -> FAIL,  else PASS.
A review band (low, high) gives three:     score <  low  -> PASS
                                           low <= score < high -> REVIEW
                                           score >= high -> FAIL

Why costs are reweighted to the base rate
-----------------------------------------
The test split might be 50% positive while production is 2% positive. Counting
the test confusion matrix directly would price the wrong world. So the test set
is used only to estimate rates *within* each class (what share of real
positives land in PASS/REVIEW/FAIL, same for negatives), and those rates are
then applied to 1,000 cases drawn at the production base rate:

    positives per 1,000 = 1000 * base_rate
    negatives per 1,000 = 1000 * (1 - base_rate)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

GRID = np.round(np.linspace(0.0, 1.0, 101), 2)  # thresholds swept, step 0.01
PER = 1000  # costs are reported per 1,000 cases


@dataclass
class Assumptions:
    cost_fp: float
    cost_fn: float
    cost_tp: float
    cost_tn: float
    base_rate: float
    daily_volume: float
    review_minutes: float
    reviewer_hourly_cost: float
    productive_hours: float
    reviewers_available: float
    reviewer_miss_rate: float = 0.0
    reviewer_false_alarm_rate: float = 0.0

    @property
    def review_cost(self) -> float:
        """Euros of reviewer time spent on one case."""
        return self.review_minutes / 60.0 * self.reviewer_hourly_cost

    def to_dict(self) -> dict:
        return asdict(self) | {"review_cost_per_case": self.review_cost}

    @classmethod
    def from_dict(cls, d: dict) -> "Assumptions":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_decisions(cls, dec) -> "Assumptions":
        return cls(
            cost_fp=float(dec.COST_FALSE_POSITIVE),
            cost_fn=float(dec.COST_FALSE_NEGATIVE),
            cost_tp=float(dec.COST_TRUE_POSITIVE),
            cost_tn=float(dec.COST_TRUE_NEGATIVE),
            base_rate=float(dec.BASE_RATE),
            daily_volume=float(dec.DAILY_VOLUME),
            review_minutes=float(dec.REVIEW_MINUTES_PER_CASE),
            reviewer_hourly_cost=float(dec.REVIEWER_HOURLY_COST),
            productive_hours=float(dec.REVIEWER_PRODUCTIVE_HOURS_PER_DAY),
            reviewers_available=float(dec.REVIEWERS_AVAILABLE),
            reviewer_miss_rate=float(dec.REVIEWER_MISS_RATE),
            reviewer_false_alarm_rate=float(dec.REVIEWER_FALSE_ALARM_RATE),
        )


# ---------------------------------------------------------------------------
# Core: outcome shares and expected cost for a review band
# ---------------------------------------------------------------------------

def _share_below(scores: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """Share of `scores` strictly below each value in `cut` (vectorised)."""
    if scores.size == 0:
        return np.zeros_like(cut, dtype=float)
    s = np.sort(scores)
    return np.searchsorted(s, cut, side="left") / s.size


def band_outcomes(scores, labels, low, high, a: Assumptions) -> dict:
    """Expected outcomes and cost per 1,000 cases for band(s) (low, high).

    `low` and `high` may be scalars or same-shape arrays; every output then has
    that shape. A single threshold t is the band (t, t): nothing is reviewed.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    pos, neg = scores[labels == 1], scores[labels == 0]

    # Shares of real positives / negatives landing in each outcome.
    pos_pass = _share_below(pos, low)
    pos_fail = 1.0 - _share_below(pos, high)
    pos_rev = 1.0 - pos_pass - pos_fail
    neg_pass = _share_below(neg, low)
    neg_fail = 1.0 - _share_below(neg, high)
    neg_rev = 1.0 - neg_pass - neg_fail

    P = PER * a.base_rate
    N = PER * (1.0 - a.base_rate)
    m, f = a.reviewer_miss_rate, a.reviewer_false_alarm_rate

    # Final decisions per 1,000 cases, after reviewers decide the middle band.
    tp = P * (pos_fail + pos_rev * (1 - m))
    fn = P * (pos_pass + pos_rev * m)
    fp = N * (neg_fail + neg_rev * f)
    tn = N * (neg_pass + neg_rev * (1 - f))
    reviews = P * pos_rev + N * neg_rev

    error_cost = tp * a.cost_tp + fn * a.cost_fn + fp * a.cost_fp + tn * a.cost_tn
    review_cost = reviews * a.review_cost
    reviews_per_day = a.daily_volume * reviews / PER
    reviewers_needed = reviews_per_day * a.review_minutes / 60.0 / a.productive_hours

    return {
        "cost_per_1000": error_cost + review_cost,
        "error_cost_per_1000": error_cost,
        "review_cost_per_1000": review_cost,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "review_share": reviews / PER,
        "reviews_per_day": reviews_per_day,
        "reviewers_needed": reviewers_needed,
        # Where cases land before any person looks (per 1,000), for the matrix.
        "pos_pass": P * pos_pass, "pos_review": P * pos_rev, "pos_fail": P * pos_fail,
        "neg_pass": N * neg_pass, "neg_review": N * neg_rev, "neg_fail": N * neg_fail,
        # Rates within each class, as measured on the test split.
        "recall_auto": pos_fail, "false_alarm_rate_auto": neg_fail,
    }


def _scalar(d: dict) -> dict:
    return {k: float(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Sweeps and optimisation
# ---------------------------------------------------------------------------

def threshold_sweep(scores, labels, a: Assumptions) -> dict:
    """Model alone: cost at every threshold on GRID."""
    out = band_outcomes(scores, labels, GRID, GRID, a)
    return {"threshold": GRID} | out


def best_threshold(scores, labels, a: Assumptions) -> float:
    """Cheapest single threshold. Ties go to the lowest threshold."""
    sweep = threshold_sweep(scores, labels, a)
    return float(GRID[int(np.argmin(sweep["cost_per_1000"]))])


def best_band(scores, labels, a: Assumptions, respect_capacity: bool = True):
    """Cheapest (low, high) with low <= high, optionally within reviewer capacity.

    Searches every pair on GRID. A band with low == high reviews nothing, so a
    feasible answer always exists.
    """
    lo, hi = np.meshgrid(GRID, GRID, indexing="ij")
    valid = lo <= hi
    out = band_outcomes(scores, labels, lo, hi, a)
    cost = np.where(valid, out["cost_per_1000"], np.inf)
    if respect_capacity:
        cost = np.where(out["reviewers_needed"] <= a.reviewers_available + 1e-9, cost, np.inf)
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return float(GRID[i]), float(GRID[j])


# ---------------------------------------------------------------------------
# The strategies the brief asks to compare
# ---------------------------------------------------------------------------

def strategies(scores, labels, a: Assumptions, threshold: float, band: tuple) -> dict:
    """Cost per 1,000 cases and staffing for every strategy in the brief."""
    ones = np.ones_like(np.asarray(scores, float))
    pass_all = band_outcomes(ones * 0.0, labels, 0.5, 0.5, a)   # every case passes
    fail_all = band_outcomes(ones * 1.0, labels, 0.5, 0.5, a)   # every case fails
    check_all = band_outcomes(ones * 0.5, labels, 0.0, 1.0, a)  # every case reviewed
    alone = band_outcomes(scores, labels, threshold, threshold, a)
    hybrid = band_outcomes(scores, labels, band[0], band[1], a)
    out = {
        "Do nothing (pass everything)": _scalar(pass_all),
        "Fail everything": _scalar(fail_all),
        "Person checks everything": _scalar(check_all),
        f"Model alone (t={threshold:.2f})": _scalar(alone),
        f"Model + review ({band[0]:.2f}-{band[1]:.2f})": _scalar(hybrid),
    }
    for v in out.values():  # can the people you have actually do it?
        v["feasible"] = bool(v["reviewers_needed"] <= a.reviewers_available + 1e-9)
    return out


# ---------------------------------------------------------------------------
# How much to trust the numbers: bootstrap and sensitivity
# ---------------------------------------------------------------------------

def bootstrap(scores, labels, a: Assumptions, threshold: float, band: tuple,
              n: int = 500, seed: int = 0) -> dict:
    """Resample the test images to see how much the costs move.

    Answers "if I had drawn a different test set of the same size, how
    different would these numbers be?". Wide intervals mean the test set is
    too small to separate the options.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    alone, hybrid, best_t = [], [], []
    for _ in range(n):
        idx = rng.integers(0, scores.size, scores.size)
        s, y = scores[idx], labels[idx]
        if y.min() == y.max():  # a resample with only one class says nothing
            continue
        alone.append(float(band_outcomes(s, y, threshold, threshold, a)["cost_per_1000"]))
        hybrid.append(float(band_outcomes(s, y, *band, a)["cost_per_1000"]))
        best_t.append(best_threshold(s, y, a))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {
        "resamples": len(alone),
        "model_alone_cost_95ci": ci(alone),
        "model_review_cost_95ci": ci(hybrid),
        "best_threshold_95ci": ci(best_t),
    }


def _best_strategy(scores, labels, a: Assumptions) -> tuple[str, dict]:
    """Cheapest strategy the available reviewers can actually staff."""
    t = best_threshold(scores, labels, a)
    band = best_band(scores, labels, a)
    s = strategies(scores, labels, a, t, band)
    costs = {k.split(" (")[0]: v["cost_per_1000"] for k, v in s.items()}
    feasible = {k.split(" (")[0]: v["cost_per_1000"] for k, v in s.items() if v["feasible"]}
    return min(feasible, key=feasible.get), {"threshold": t, "band": band, "costs": costs}


def sensitivity(scores, labels, a: Assumptions,
                multipliers=(0.25, 0.5, 1.0, 2.0, 4.0)) -> list[dict]:
    """Re-solve the problem with one assumption scaled up or down at a time.

    "cheapest_strategy" only considers strategies the available reviewers can
    staff; the cost columns show every strategy regardless.
    """
    rows = []
    for param in ("base_rate", "cost_fn", "cost_fp", "review_minutes"):
        for m in multipliers:
            d = a.to_dict()
            d[param] = min(d[param] * m, 0.999) if param == "base_rate" else d[param] * m
            b = Assumptions.from_dict(d)
            winner, info = _best_strategy(scores, labels, b)
            rows.append({
                "assumption": param, "multiplier": m, "value": d[param],
                "best_threshold": info["threshold"],
                "best_band_low": info["band"][0], "best_band_high": info["band"][1],
                "cheapest_strategy": winner,
                **{f"cost: {k}": v for k, v in info["costs"].items()},
            })
    return rows


def switch_points(scores, labels, a: Assumptions) -> list[dict]:
    """Values of each assumption at which the cheapest *feasible* strategy changes.

    This is the "my answer changes if X drops below Z" evidence: each
    assumption is scanned from 1/100x to 100x its stated value, holding the
    others fixed. Strategies needing more reviewers than REVIEWERS_AVAILABLE
    are excluded, so change that number to ask "what if we hired?".
    """
    found = []
    grid = np.logspace(-2, 2, 81)
    for param in ("base_rate", "cost_fn", "cost_fp", "review_minutes"):
        prev = None
        for m in grid:
            d = a.to_dict()
            d[param] = d[param] * m
            if param == "base_rate" and d[param] >= 1:
                break
            winner, _ = _best_strategy(scores, labels, Assumptions.from_dict(d))
            if prev is not None and winner != prev[1]:
                found.append({"assumption": param, "at_about": float(d[param]),
                              "stated_value": float(getattr(a, param)),
                              "from": prev[1], "to": winner})
            prev = (m, winner)
    return found
