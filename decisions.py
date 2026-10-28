"""
Every judgement call this assignment grades lives in this one file.

train.py reads it, runs the pipeline, and copies the values it used into
artifacts/, so the dashboard always shows the assumptions behind its numbers.

Anything set to None is a decision you have to make yourself. train.py refuses
to run until every required value is filled in, and tells you which are
missing. Each value should trace back to a sentence in REPORT.md.

After changing only costs, base rate or staffing (not the dataset or the model),
re-run the costing on your committed weights without retraining:

    REUSE_WEIGHTS=1 uv run python main.py train          # macOS / Linux
    $env:REUSE_WEIGHTS=1; uv run python main.py train    # Windows PowerShell
"""

# ---------------------------------------------------------------------------
# 1. Dataset (README step 2). Copy these from the dataset's Universe URL:
#    https://universe.roboflow.com/<workspace>/<project>/dataset/<version>
#    Use the public dataset as it is. Do not fork it.
# ---------------------------------------------------------------------------
ROBOFLOW_WORKSPACE = "roboflow-100"
ROBOFLOW_PROJECT = "construction-safety-gsnvb"
ROBOFLOW_VERSION = 1

# ---------------------------------------------------------------------------
# 2. What counts as a "positive" case (README step 5).
#    A case is one image. An image is positive if it contains at least one
#    object of any class listed here. Spell names exactly as in the dataset.
#    Positive = the thing you want to catch, which the model routes to FAIL.
# ---------------------------------------------------------------------------
POSITIVE_CLASSES = ["no-vest"]

# What each outcome is called in your business. Purely cosmetic.
LABEL_PASS = "Pass"
LABEL_REVIEW = "Human review"
LABEL_FAIL = "Fail"

# ---------------------------------------------------------------------------
# 3. Cost of each outcome, in euros per case (README step 6).
#    Justify every figure in REPORT.md: where it comes from, and the value at
#    which your conclusion would change.
# ---------------------------------------------------------------------------
COST_FALSE_POSITIVE = 1.0  # TEMP placeholder: replace with your own estimate before the final run
COST_FALSE_NEGATIVE = 1.0  # TEMP placeholder: replace with your own estimate before the final run
COST_TRUE_POSITIVE = 0.0    # a positive case caught (e.g. rework cost); 0 if none
COST_TRUE_NEGATIVE = 0.0    # a negative case passed; usually 0

# How often the positive class actually occurs in production, as a fraction
# (0.02 means 2 in 100 cases). This is NOT the share of positives in the test
# split, which is usually far higher. Costs are reweighted to this rate.
BASE_RATE = 0.5  # TEMP placeholder: replace with your own estimate before the final run

# ---------------------------------------------------------------------------
# 4. Human review (README step 7).
# ---------------------------------------------------------------------------
DAILY_VOLUME = 1  # TEMP placeholder: replace with your own estimate before the final run
REVIEW_MINUTES_PER_CASE = 1.0  # TEMP placeholder: replace with your own estimate before the final run
REVIEWER_HOURLY_COST = 1.0  # TEMP placeholder: replace with your own estimate before the final run
REVIEWER_PRODUCTIVE_HOURS_PER_DAY = 1.0  # TEMP placeholder: replace with your own estimate before the final run
REVIEWERS_AVAILABLE = 1  # TEMP placeholder: replace with your own estimate before the final run

# Reviewers make mistakes too. 0.0 means "a person is always right", which is
# itself an assumption you should be able to defend.
REVIEWER_MISS_RATE = 0.0         # share of reviewed positives a person passes
REVIEWER_FALSE_ALARM_RATE = 0.0  # share of reviewed negatives a person fails

# ---------------------------------------------------------------------------
# 5. Operating point. Leave as None to use the cost-minimising values the
#    sweep finds (that is what "chosen by sweeping" means). Only override if
#    you can say in REPORT.md why the sweep's minimum is not the one to ship.
# ---------------------------------------------------------------------------
THRESHOLD_OVERRIDE = None        # single model-alone threshold
REVIEW_BAND_OVERRIDE = None      # (low, high): below low -> pass, at/above high -> fail

# ---------------------------------------------------------------------------
# 6. Zero-shot baseline (README step 4). The pretrained COCO model knows 80
#    COCO classes. Map any that mean one of your classes, e.g.
#    {"person": "worker"}. Leave {} if none match: then the zero-shot model
#    can never find your classes, and reporting that is a valid finding.
# ---------------------------------------------------------------------------
COCO_TO_DATASET = {"person": "person"}  # COCO's "person" means the same as the dataset's "person"

# ---------------------------------------------------------------------------
# 7. Training settings. Record where you trained and these values in REPORT.md.
# ---------------------------------------------------------------------------
MODEL = "yolo11n"     # "yolo11n" or "yolo11s"
EPOCHS = 50
IMGSZ = 640
BATCH = 16            # lower to 8 if you run out of GPU memory
PATIENCE = 15         # stop early after this many epochs without improvement
SEED = 0
DEVICE = None         # None lets Ultralytics pick (GPU if available)


REQUIRED = [
    "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT", "ROBOFLOW_VERSION",
    "POSITIVE_CLASSES",
    "COST_FALSE_POSITIVE", "COST_FALSE_NEGATIVE", "BASE_RATE",
    "DAILY_VOLUME", "REVIEW_MINUTES_PER_CASE", "REVIEWER_HOURLY_COST",
    "REVIEWER_PRODUCTIVE_HOURS_PER_DAY", "REVIEWERS_AVAILABLE",
]


def missing() -> list[str]:
    """Names of required decisions that are still None."""
    return [name for name in REQUIRED if globals()[name] is None]
