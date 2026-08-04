# Confidence-Gated LMS Triage & Grading

An ML pipeline for a Learning Management System (LMS) that does two jobs:

1. **Grades code submissions** — predicts a continuous quality score using AST structural metrics, surface-level code ratios, and dual-token TF-IDF features.
2. **Triages student doubts** — classifies forum posts by course topic and determines urgency for confidence-gated queue escalation.

The triage system uses an asymmetric decision rule: instead of simply classifying urgent vs. non-urgent, it only **auto-handles** a post when it's confident the post is *NOT* urgent. Anything uncertain — or confidently urgent — is escalated to a human instructor. This asymmetry is deliberate: missing a genuinely urgent doubt is far more costly than a teacher reviewing one extra doubt that turns out to be fine.

---

## Prerequisites

- Python 3.9+
- Internet access (datasets pulled live from GitHub and Hugging Face)
- Packages:
  ```bash
  pip install -q lightgbm scikit-learn pandas numpy joblib scipy radon
  ```

No GPU required — feature extraction, vectorization, and LightGBM fitting run in under two minutes on standard CPU.

---

## Datasets

### 1. Grading data — `sjelassi/new_omi_code_100k` (Hugging Face)

LLM-generated Python solutions, sampled to 20,000 rows.

| Field | Description |
|---|---|
| `answer` | Raw Python code string |
| `pass_rate` | Fraction of automated unit tests passed |
| `test_count` | Number of test cases executed |
| `quality_score` | Composite ground-truth quality score |
| `has_docstring` | Binary flag for docstring presence |

**Caveat:** `pass_rate` and `quality_score` come from an automated grading harness, not a human — they're a noisy proxy for code quality, not ground truth.

### 2. Triage data — `pcla-code/forum-posts-urgency` (GitHub, MIT license)

Real student discussion posts from **9 Stanford MOOC courses** (accounting, calculus, design, game theory, globalization, modern poetry, mythology, probability, vaccines), hand-coded for urgency on a 1–7 scale (Almatrafi et al.). Each course ships as **two partition files** (e.g. `acc1`, `acc2`) — these are two chunks of the *same* course, not two different courses.

- **Topic label:** the course a post came from (9 classes)
- **Urgency label:** `Urgency_1_7 >= 4` → urgent, the threshold used in the published literature on this dataset. Urgent posts are a minority class (~19% positive balance).

**Data handling notes:**
- `id` is a *per-course* sequential ID, not a global primary key — the same `id` recurs across courses by coincidence. A composite key (`course` + `id`) was built before deduplicating, otherwise valid rows get dropped as false duplicates.
- Genuine duplicate rows (same course + id) were removed.
- Rows missing `post_text` or `Urgency_1_7` were dropped.

---

## Feature Engineering (Grading)

Target undergoes normalization + rank transformation to correct for density clustering around central `quality_score` values:

$$\text{target\_raw} = \text{clip}\left(\frac{\text{quality\_score} - 15.1}{15.3 - 15.1}, 0.0, 1.0\right)$$

$$y = \text{RankTransform}(\text{target\_raw})$$

The LightGBM grading engine uses a **1,018-dimensional** sparse feature set:

**AST structural metrics (16 features)**
- Node count, depth, function/class defs, loop constructs
- Try-except blocks, return statements, assignments, type annotations (`ast.AnnAssign`)
- Comprehension counts, argument density, nested loop depth
- Cyclomatic complexity (from AST control flow: `1 + conditionals + loops + tries + boolean ops`)
- Identifier style ratios (proportion of single-character names like `i`, `x`, `temp`)

**Code ratios & surface metrics (10 features)**
- Character length, line count, docstring flag, test execution metrics
- Average line length, comment density, indentation frequency, operator counts/density
- AST node density per line, AST complexity density per node

**Dual sparse TF-IDF (1,000 features)**
- Character n-grams (3–5): 600 features — syntactic structure, formatting, keyword combinations
- Word n-grams (1–2): 400 features — domain-specific keywords/identifiers (`r"(?u)\b\w+\b"`)

Cyclomatic complexity and line counts are computed via `radon`. Rows that fail to parse (truncated generations, syntax errors) become `NaN` and are **median-imputed** rather than dropped, since a failed-to-parse submission is itself informative.

---

## Models

| Task | Model | Notes |
|---|---|---|
| Code grading | **Enhanced LightGBM Regressor** | AST + surface + dual TF-IDF features, quantile rank target scaling |
| Topic triage | Logistic Regression (TF-IDF), `class_weight="balanced"` | 9-class classification |
| Urgency triage | Logistic Regression + `CalibratedClassifierCV` (sigmoid) | Calibration matters because routing depends on trustworthy probabilities, not just the label |
| Escalation routing | Confidence threshold (`P(urgent) < 0.15`) | Auto-handle only below threshold |

### Routing rule

Auto-handle a doubt only when `P(urgent) < 0.15`, chosen by sweeping the threshold on the validation set: below it, too few doubts get auto-handled to be useful; above it, the share of auto-handled doubts that were actually urgent climbs past 6–7% and keeps rising.

---

## Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Code grading | Enhanced LightGBM Regressor | Test R² | 0.4309 |
| Code grading | Enhanced LightGBM Regressor | Test RMSE | 0.1791 |
| Code grading | Enhanced LightGBM Regressor | Test MAE | 0.1038 |
| Topic triage | Balanced Logistic Regression (TF-IDF) | Val Macro-F1 | 0.6670 |
| Urgency triage | Calibrated Logistic Regression (Sigmoid) | Val Macro-F1 | 0.7030 |
| Escalation routing | Confidence threshold (P < 0.15) | Auto-handle coverage / missed urgency | 59.3% / 5.92% |

Min predicted quality score: 0.0159 · Max predicted quality score: 0.9629

---

## Visualizations

The notebook includes a plotting cell (`generate_plots.py` in this repo — paste it in as a new cell after the grading and triage cells have run) that saves the following to a `plots/` folder:

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

![Topic distribution](vercel-deploy/plots/01_topic_distribution.png)
![Urgency balance](vercel-deploy/plots/02_urgency_balance.png)
![Quality score distribution](vercel-deploy/plots/03_quality_score_distribution.png)
![Feature correlation](vercel-deploy/plots/04_feature_correlation_heatmap.png)
![Complexity vs quality](vercel-deploy/plots/05_complexity_vs_quality.png)
![Grading model comparison](vercel-deploy/plots/06_grading_model_comparison.png)
![Urgency probability distribution](vercel-deploy/plots/07_urgency_probability_distribution.png)
![Threshold tradeoff](vercel-deploy/plots/08_threshold_tradeoff.png)

---

## Live API

### `POST /api/grade`

Evaluates raw Python code against the trained AST-LightGBM pipeline artifact (`code_grading_lgbm.joblib`).

**Request**
```json
{
  "code": "def fibonacci(n:\n    \"\"\"Calculate nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "pass_rate": 1.0,
  "test_count": 10
}
```

**Response**
```json
{
  "predicted_quality_score": 0.8842,
  "ast_cyclomatic_complexity": 3,
  "ast_depth": 4,
  "syntax_valid": 1,
  "model_version": "code_grading_lgbm_v2"
}
```

### `POST /api/triage`

Classifies a student doubt by topic and urgency, and returns the routing decision.

**Request**
```json
{
  "post_text": "I keep getting an IndexOutOfBounds error on line 42 when passing an empty array, assignment due in 10 mins!"
}
```

**Response**
```json
{
  "predicted_topic": "cs106a",
  "urgency_probability": 0.8412,
  "auto_handle": false,
  "threshold_used": 0.15
}
```

---

## Artifact Export

The trained model and vectorizers are serialized for inference into `code_grading_lgbm.joblib`:

```python
import joblib

artifact = joblib.load("code_grading_lgbm.joblib")
model = artifact["model"]
tfidf_char = artifact["tfidf_char"]
tfidf_word = artifact["tfidf_word"]
feature_names = artifact["feature_names"]
```
