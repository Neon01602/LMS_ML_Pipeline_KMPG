# Confidence-Gated LMS Triage & Grading

An ML pipeline for a Learning Management System (LMS) that does two jobs:

1. **Grades code submissions** — predicts a continuous quality score using AST structural metrics, surface-level code ratios, and (at training time) dual-token TF-IDF features.
2. **Triages student doubts** — classifies forum posts by course topic and determines urgency for confidence-gated queue escalation.

The triage system uses an asymmetric decision rule: instead of simply classifying urgent vs. non-urgent, it only **auto-handles** a post when it's confident the post is *NOT* urgent. Anything uncertain — or confidently urgent — is escalated to a human instructor. This asymmetry is deliberate: missing a genuinely urgent doubt is far more costly than a teacher reviewing one extra doubt that turns out to be fine.

---

## Prerequisites

- Python 3.9+
- Internet access (datasets pulled live from GitHub and Hugging Face) — training only, not required at inference
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

## Feature Engineering (Grading — Training Pipeline)

Target undergoes normalization + rank transformation to correct for density clustering around central `quality_score` values:

$$\text{target\_raw} = \text{clip}\left(\frac{\text{quality\_score} - 15.1}{15.3 - 15.1}, 0.0, 1.0\right)$$

$$y = \text{RankTransform}(\text{target\_raw})$$

`RankTransform` here is a **percentile rank (0–100 scale)**, not a 0–1 probability — this matters at inference time (see [Serving Format](#serving-format--inference-only-feature-subset) below), since the deployed booster's raw output lands in roughly the 0–100 range, not 0–1.

The full training pipeline builds a **1,018-dimensional** sparse feature set:

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

## Serving Format — Inference-Only Feature Subset

**Important discrepancy between training and deployment:** the model actually shipped to `/api/grade` (via the raw LightGBM text dump, e.g. `lgbm_model.txt` / `lgbm_model_data.py`) is trained on a **15-feature subset only** — the 1,018-dimensional TF-IDF-augmented feature set above describes the full training/experimentation pipeline, but the exported booster's `feature_names=` header lists just:

```
test_pass_rate, cyclomatic_complexity, lines_of_code, num_functions,
runtime_ms, memory_kb, num_compile_errors, num_warnings, comment_density,
num_attempts, hours_before_deadline, student_avg_past_score,
runtime_ms_missing, memory_kb_missing, comment_density_missing
```

This is the **exact positional order** the pure-Python tree evaluator indexes against via each node's `split_feature` integer — it does not do name-based lookup. Any inference wrapper must build the feature vector in this order, regardless of what order feature engineering code produces internally.

**Target scale at inference:** the booster's raw summed output is on a **0–100 scale** (consistent with the percentile-rank training target above), not 0–1. Serving code should:
1. Sum all tree outputs (the root tree carries `shrinkage=1` and starts near the population mean, ~59–60; subsequent trees are `shrinkage=0.03` residual corrections).
2. Clip the sum to `[0, 100]`.
3. Divide by 100 if a `[0, 1]` score is required downstream (e.g. to match a `predicted_quality_score` API contract).

Clipping to `[0, 1]` *before* this normalization silently saturates almost every real prediction to `1.0` and was a production bug in earlier deployments — see inline comments in `PurePythonLGBMRegressor.predict()`.

Missing-value routing (`default_left`) is currently hardcoded to `True` for every node in the pure-Python evaluator rather than reading LightGBM's per-node default-direction bit. This only affects predictions where a feature is genuinely `NaN` post-imputation, which the current pipeline avoids by imputing upstream — flagged here as a known limitation, not yet fixed.

---

## Models

| Task | Model | Notes |
|---|---|---|
| Code grading | **Enhanced LightGBM Regressor** | Trained on AST + surface + dual TF-IDF features (1,018-dim); **deployed booster uses the 15-feature AST/surface/metadata subset only** (no TF-IDF at inference), percentile-rank (0–100) target scaling |
| Topic triage | Logistic Regression (TF-IDF), `class_weight="balanced"` | 9-class classification |
| Urgency triage | Logistic Regression (TF-IDF), `class_weight="balanced"` | Binary, asymmetric confidence-gated decision rule |

### Deployed model artifacts (`models/`)

| File | Purpose |
|---|---|
| `lgbm_model.json` / `lgbm_model_trees.json` (or `lgbm_model_data.py` `LGBM_MODEL_TEXT`) | Raw LightGBM text-format model dump, parsed by a pure-Python evaluator (no native `lightgbm` runtime dependency at serve time) |
| `grading_artifacts.joblib` | `feature_cols`, fitted `imputer`, `numeric_with_na` — used to impute `runtime_ms` / `memory_kb` / `comment_density` before scoring |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer for triage text |
| `topic_classifier.pkl` | 9-class topic classifier (expects vectorized text, not raw strings) |
| `urgency_classifier.pkl` | Binary urgency classifier (expects vectorized text, not raw strings) |
| `tfidf_and_features.joblib` | Combined vectorizer + any additional engineered features used during triage training (if present, must be concatenated with TF-IDF output before calling `.predict()`) |
| `metadata.json` | Serving metadata, including `confidence_threshold` for the urgency auto-handle gate |

There is **no single bundled `triage_models.joblib`** — the triage pipeline is served by loading `tfidf_vectorizer.pkl`, `topic_classifier.pkl`, and `urgency_classifier.pkl` independently and vectorizing `post_text` before calling `.predict()` / `.predict_proba()` on each.

---

## API

### `POST /api/grade`

```json
{
  "code": "def add(a, b):\n    return a + b",
  "pass_rate": 1.0,
  "test_count": 10,
  "runtime_ms": 12,
  "memory_kb": 300,
  "comment_density": 0.1,
  "num_attempts": 1,
  "hours_before_deadline": 48,
  "student_avg_past_score": 92
}
```

Returns a `predicted_quality_score` in `[0, 1]` (normalized from the model's native 0–100 scale), plus `cyclomatic_complexity` and `lines_of_code` computed via `radon` where available.

### `POST /api/triage`

```json
{
  "post_text": "I'm confused about how gradient descent updates the weights in backpropagation"
}
```

Returns `predicted_topic`, `urgency_probability`, `auto_handle` (`true` only when the model is confident the post is *not* urgent, per the asymmetric threshold in `metadata.json`), and `threshold_used`.
