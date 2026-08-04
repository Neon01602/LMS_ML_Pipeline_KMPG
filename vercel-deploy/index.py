"""
FastAPI app serving the LMS Triage + Grading models.
Runs as a single Vercel Python serverless function (all routes handled
internally by FastAPI, see vercel.json for the catch-all rewrite).
"""
import json
import os
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# radon is only needed for the grading endpoint (computes complexity from code)
try:
    from radon.complexity import cc_visit
    from radon.raw import analyze
    from radon.metrics import h_visit, mi_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

app = FastAPI(title="LMS ML Pipeline API")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---------------------------------------------------------------
# Load models once at cold start (reused across warm invocations).
# Triage and grading load independently so a failure in one
# (e.g. a missing/incompatible model file) doesn't take the other down.
# ---------------------------------------------------------------
_state = {}


def _load_metadata():
    if "metadata" not in _state:
        with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
            _state["metadata"] = json.load(f)
    return _state["metadata"]


def _load_triage():
    """Load only what /api/triage needs, independent of the grading model."""
    if "vec" not in _state:
        _state["vec"] = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        _state["topic_clf"] = joblib.load(os.path.join(MODELS_DIR, "topic_classifier.pkl"))
        _state["urgency_clf"] = joblib.load(os.path.join(MODELS_DIR, "urgency_classifier.pkl"))
    _load_metadata()
    return _state


def _load_grading():
    """Load only what /api/grade needs, independent of the triage models.

    grading_model_v5.joblib bundles everything the grading pipeline needs
    into a single dict artifact: {model, imputer, feature_cols,
    final_feature_idx, winner_name} -- this keeps the feature schema and
    the final-feature selection tied to the exact model that was trained
    on them, so there's no separate metadata file to drift out of sync.
    """
    if "grading_artifact" not in _state:
        artifact = joblib.load(os.path.join(MODELS_DIR, "grading_model_v5.joblib"))
        _state["grading_artifact"] = artifact
        _state["grading_model"] = artifact["model"]
        _state["grading_imputer"] = artifact["imputer"]
        _state["grading_feature_cols"] = artifact["feature_cols"]
        _state["grading_final_feature_idx"] = artifact["final_feature_idx"]
        _state["grading_model_name"] = artifact["winner_name"]
    return _state


# ---------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------
class TriageRequest(BaseModel):
    post_text: str = Field(..., description="Raw text of the student's forum post/doubt")


class TriageResponse(BaseModel):
    predicted_topic: str
    urgency_probability: float
    auto_handle: bool
    threshold_used: float


class GradeRequest(BaseModel):
    code: str = Field(..., description="Source code submission to grade")
    pass_rate: Optional[float] = Field(None, description="Fraction of tests passed, 0-1, if known")
    test_count: Optional[int] = Field(None, description="Number of tests run against this submission, if known")


class GradeResponse(BaseModel):
    predicted_quality_score: float
    model_used: str
    cyclomatic_complexity: Optional[float]
    lines_of_code: Optional[float]


# ---------------------------------------------------------------
# Feature helpers (mirror the notebook's feature engineering /
# extract_code_features + the derived-feature step from training)
# ---------------------------------------------------------------
def _complexity_stats(code: str) -> dict:
    """Returns both mean and max cyclomatic complexity across all blocks
    (functions/methods) in the submitted code."""
    try:
        blocks = cc_visit(code)
        complexities = [b.complexity for b in blocks] if blocks else [1.0]
        return {
            "cyclomatic_complexity": float(np.mean(complexities)),
            "max_complexity": float(np.max(complexities)),
        }
    except Exception:
        return {"cyclomatic_complexity": float("nan"), "max_complexity": float("nan")}


def _lines_of_code(code: str) -> float:
    try:
        return float(analyze(code).loc)
    except Exception:
        return float("nan")


def _full_code_metrics(code: str) -> dict:
    nan_result = {
        "logical_lines": float("nan"),
        "comment_lines": float("nan"),
        "comment_ratio": float("nan"),
        "blank_lines": float("nan"),
        "blank_ratio": float("nan"),
        "halstead_volume": float("nan"),
        "halstead_difficulty": float("nan"),
        "halstead_effort": float("nan"),
        "maintainability_index": float("nan"),
        "parse_failed": 1,
    }
    try:
        raw = analyze(code)
        loc = raw.loc or 1  # avoid div-by-zero
        comment_lines = raw.comments + raw.single_comments

        h = h_visit(code).total
        mi = mi_visit(code, True)

        # Min-Max Normalize MI to range [0, 1]
        # Radon MI scale theoretically ranges from 0 to 100
        normalized_mi = max(0.0, min(1.0, float(mi) / 100.0))

        return {
            "logical_lines": float(raw.lloc),
            "comment_lines": float(comment_lines),
            "comment_ratio": float(comment_lines) / loc,
            "blank_lines": float(raw.blank),
            "blank_ratio": float(raw.blank) / loc,
            "halstead_volume": float(h.volume),
            "halstead_difficulty": float(h.difficulty),
            "halstead_effort": float(h.effort),
            "maintainability_index": normalized_mi,
            "parse_failed": 0,
        }
    except Exception:
        return nan_result


def _code_structural_features(code: str) -> dict:
    """Cheap stand-ins for the dataset's own def_count / total_tokens / has_docstring,
    computed directly from the submitted code at inference time -- callers only
    need to send raw code, not pre-computed structural features."""
    def_count = code.count("def ")
    total_tokens = len(code.split())
    has_docstring = int('"""' in code or "'''" in code)
    return {
        "def_count": def_count,
        "total_tokens": total_tokens,
        "has_docstring": has_docstring,
    }


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/")
@app.get("/api")
def root():
    return {
        "message": "LMS ML Pipeline API is live",
        "endpoints": {
            "health": "GET /api/health",
            "triage": "POST /api/triage",
            "grade": "POST /api/grade",
        },
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    state = _load_triage()
    X_txt = state["vec"].transform([req.post_text])

    predicted_topic = state["topic_clf"].predict(X_txt)[0]
    urgency_proba = float(state["urgency_clf"].predict_proba(X_txt)[0, 1])
    threshold = state["metadata"]["urgency_threshold"]

    return TriageResponse(
        predicted_topic=predicted_topic,
        urgency_probability=round(urgency_proba, 4),
        auto_handle=urgency_proba < threshold,
        threshold_used=threshold,
    )


@app.post("/api/grade", response_model=GradeResponse)
def grade(req: GradeRequest):
    if not RADON_AVAILABLE:
        raise HTTPException(500, "radon not installed on server")

    try:
        state = _load_grading()
    except Exception as e:
        raise HTTPException(500, f"Grading model failed to load: {type(e).__name__}: {e}")

    complexity_stats = _complexity_stats(req.code)
    complexity = complexity_stats["cyclomatic_complexity"]
    max_complexity = complexity_stats["max_complexity"]
    loc = _lines_of_code(req.code)
    struct = _code_structural_features(req.code)
    full_metrics = _full_code_metrics(req.code)

    row = {
        "pass_rate": req.pass_rate if req.pass_rate is not None else np.nan,
        "test_count": req.test_count if req.test_count is not None else np.nan,
        "total_tokens": struct["total_tokens"],
        "def_count": struct["def_count"],
        "has_docstring": struct["has_docstring"],
        "cyclomatic_complexity": complexity,
        "max_complexity": max_complexity,
        "lines_of_code": loc,
        **full_metrics,
    }
    
    row["tokens_per_def"] = row["total_tokens"] / (row["def_count"] + 1)
    row["complexity_per_line"] = row["cyclomatic_complexity"] / ((row["lines_of_code"] or 0) + 1)

    feature_cols = state["grading_feature_cols"]
    final_feature_idx = state["grading_final_feature_idx"]

    missing = [c for c in feature_cols if c not in row]
    if missing:
        raise HTTPException(
            500,
            f"Feature schema mismatch -- missing columns {missing} in computed features."
        )

    x = np.array([[row[c] for c in feature_cols]], dtype=float)
    x_imputed = state["grading_imputer"].transform(x)[:, final_feature_idx]
    
    # Raw prediction from model (e.g., 15.2013)
    raw_pred = float(state["grading_model"].predict(x_imputed)[0])

    # --- MIN-MAX SCALING TO RANGE [0, 1] ---
    MIN_VAL = 15.1000
    MAX_VAL = 15.3000
    
    # Scale to 0-1 and clamp to handle slight out-of-bound predictions
    scaled_pred = (raw_pred - MIN_VAL) / (MAX_VAL - MIN_VAL)
    scaled_pred = float(np.clip(scaled_pred, 0.0, 1.0))

    return GradeResponse(
        predicted_quality_score=round(scaled_pred, 4),
        model_used=state["grading_model_name"],
        cyclomatic_complexity=None if np.isnan(complexity) else round(complexity, 2),
        lines_of_code=None if np.isnan(loc) else loc,
    )
