# Confidence-Gated LMS Triage & Grading

An ML pipeline for a Learning Management System (LMS) that does two jobs:

1. **Grades code submissions** — predicts a quality score for a student's code.
2. **Triages student doubts** — figures out which course a forum post belongs to,
   and whether it's urgent enough to skip the queue and go straight to a teacher.

The second task is the interesting one: instead of just classifying urgent vs.
not, the system only **auto-handles** a doubt when it is *confident the doubt
is NOT urgent*. Anything it's unsure about — or confident is urgent — gets
escalated to a human. That asymmetry is deliberate: missing a genuinely urgent
doubt is far more costly than a teacher reviewing one extra doubt that turns
out to be fine.

---

## Prerequisites

To re-run the notebook (`LMS_ML_Pipeline_v3_RealData.ipynb`) yourself:

- Python 3.9+ (Colab's default runtime works out of the box)
- Internet access (both datasets are pulled live — GitHub raw CSVs and a
  Hugging Face parquet file)
- Packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`,
  `lightgbm`, `radon` — installed by the notebook's setup cell
  (`!pip install -q lightgbm radon datasets`)

No GPU is required — every model here (LightGBM, logistic regression) trains
in well under a minute on CPU.

---

## The datasets

### 1. Grading data — `sjelassi/new_omi_code_100k` (Hugging Face)

100k LLM-generated coding solutions. A 15,000-row sample is used. Each row
has:

| Field | What it is |
|---|---|
| `answer` | the actual generated code |
| `pass_rate` | fraction of automated tests the code passed |
| `test_count` | number of tests run against it |
| `quality_score` | the target — a composite quality label |
| `total_tokens`, `def_count`, `has_docstring` | structural proxies shipped with the dataset |

**Important caveat:** `pass_rate` and `quality_score` are both produced by an
automated grading harness, not a human — they're a noisy proxy for real code
quality, not ground truth. The pipeline is built around that limitation
rather than pretending it isn't there.

### 2. Triage data — `pcla-code/forum-posts-urgency` (GitHub, MIT license)

Real student discussion-forum posts from 9 Stanford MOOCs (accounting,
calculus, design, game theory, globalization, modern poetry, mythology,
probability, vaccines), hand-coded by researchers for urgency on a 1–7 scale
(Almatrafi et al.). Each of the 9 courses ships as two partition files
(e.g. `acc1`, `acc2`) — these are two chunks of the *same* course, not two
different courses.

- **Topic label:** the course the post came from (9 classes)
- **Urgency label:** `Urgency_1_7 >= 4` → urgent (the threshold used in the
  published literature on this dataset). Urgent posts are a minority class
  (~19%), matching the imbalance reported in that literature.

---

## What was done to the data

**Triage data:**
- `id` is a *per-course* sequential ID from the original corpus, not a global
  primary key — the same `id` shows up in multiple courses by coincidence. A
  composite key (`course` + `id`) was built before deduplicating, otherwise
  valid rows get dropped as false duplicates.
- A small number of genuine duplicate rows (same course + id) were removed.
- Rows missing `post_text` or `Urgency_1_7` were dropped.

**Grading data:**
- Checked every numeric column's correlation with `quality_score` to rule out
  leakage (nothing came back suspiciously high — `pass_rate` and `test_count`
  are legitimately predictive, not leaky, since they exist *before* quality
  is scored).
- Cyclomatic complexity and lines-of-code were computed directly from the
  `answer` column using `radon`. A handful of samples don't parse (truncated
  generations, syntax errors) — those become `NaN` and are **median-imputed**
  rather than dropped, since a failed-to-parse submission is itself
  informative (it likely also failed most tests).

---

## Models

| Task | Model | Why |
|---|---|---|
| Grading | Linear Regression (baseline) vs. LightGBM (tuned via `RandomizedSearchCV`, 5-fold CV) | Winner picked on the validation set, reported once on test |
| Topic classification | Logistic Regression on TF-IDF (unigrams + bigrams, 5000 features), `class_weight="balanced"` | Simple, fast, strong baseline for 9-class text classification |
| Urgency classification | Logistic Regression wrapped in `CalibratedClassifierCV` (sigmoid, 5-fold) | Calibration matters here because the routing decision below depends on the *probability* being trustworthy, not just the class label |

All splits are 70/15/15 (train/val/test), stratified where relevant, with the
test set touched exactly once at the end.

### Routing rule

Auto-handle a doubt only when `P(urgent) < 0.15`. This threshold was chosen
by sweeping it on the validation set: below 0.15, too few doubts get
auto-handled to be useful; above it, the share of auto-handled doubts that
were actually urgent climbs past 6–7% and keeps rising. At 0.15, roughly 60%
of doubts can be auto-handled while only ~6% of those are missed urgent
cases.

---

## Visualizations

The notebook includes a plotting cell (`generate_plots.py` in this repo —
paste it in as a new cell after the grading and triage cells have run) that
saves the following to a `plots/` folder:

| Plot | Shows |
|---|---|
| `01_topic_distribution.png` | How many forum posts came from each course |
| `02_urgency_balance.png` | Urgent vs. non-urgent post split |
| `03_quality_score_distribution.png` | Spread of code quality scores |
| `04_feature_correlation_heatmap.png` | Correlation between grading features (the leakage check, visualized) |
| `05_complexity_vs_quality.png` | Does more complex code score higher or lower? |
| `06_grading_model_comparison.png` | Baseline vs. LightGBM validation RMSE |
| `07_urgency_probability_distribution.png` | Predicted urgency probability, split by true label, with the 0.15 threshold marked |
| `08_threshold_tradeoff.png` | Coverage vs. missed-urgent rate across thresholds — the actual justification for picking 0.15 |

Once generated, drop the images here:

![Topic distribution](plots/01_topic_distribution.png)
![Urgency balance](plots/02_urgency_balance.png)
![Quality score distribution](plots/03_quality_score_distribution.png)
![Feature correlation](plots/04_feature_correlation_heatmap.png)
![Complexity vs quality](plots/05_complexity_vs_quality.png)
![Grading model comparison](plots/06_grading_model_comparison.png)
![Urgency probability distribution](plots/07_urgency_probability_distribution.png)
![Threshold tradeoff](plots/08_threshold_tradeoff.png)

---

## Using the live API

The trained models are served as two endpoints. Replace
`https://YOUR-PROJECT.vercel.app` with your actual deployment URL.

### `POST /api/triage`

Classify a student doubt by topic and urgency.

**Request**
```json
{
  "post_text": "I cannot submit my assignment and the deadline is in 20 minutes, please help urgently"
}
```

**Response**
```json
{
  "predicted_topic": "calc",
  "urgency_probability": 0.82,
  "auto_handle": false,
  "threshold_used": 0.15
}
```

`auto_handle: false` means this doubt gets routed to a teacher instead of
being closed automatically.

### `POST /api/grade`

Score a code submission. `pass_rate` and `test_count` are optional — if you
don't have test-harness results yet, leave them out and the saved imputer
fills in a sensible default.

**Request**
```json
{
  "code": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b",
  "pass_rate": 0.9,
  "test_count": 5
}
```

**Response**
```json
{
  "predicted_quality_score": 7.42,
  "model_used": "LightGBM",
  "cyclomatic_complexity": 1.0,
  "lines_of_code": 2
}
```

### `GET /api/health`

Returns `{"status": "ok"}` — useful for confirming the deployment is live
before sending real requests.

---

## Summary

| Task | Model | Metric | Result |
|---|---|---|---|
| Grading | Baseline vs. LightGBM (selected on val) | Test RMSE / R² | *fill in from your run — depends on the 15k-row sample* |
| Triage — topic | Logistic Regression (TF-IDF) | Val macro-F1 | *fill in from your run* |
| Triage — urgency | Calibrated Logistic Regression | Val macro-F1 @ 0.5 | *fill in from your run* |
| Routing | Confidence threshold = 0.15 | Test coverage / missed-urgent rate | *fill in from your run* |

The point of using real (rather than synthetic) data here wasn't a better
leaderboard number — real numbers are messier than a hand-built generative
process produces. The value is that every modeling decision above (the
composite key, the calibration, the asymmetric threshold) was a response to
something the data actually did, not something anticipated in advance.
