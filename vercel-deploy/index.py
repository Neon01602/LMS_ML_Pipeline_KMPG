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


import re
from scipy.sparse import hstack

# ---------------------------------------------------------------
# Updated Artifact Loader
# ---------------------------------------------------------------
def _load_grading():
    """Load the new TF-IDF + XGBoost grading artifact."""
    if "grading_artifact" not in _state:
        artifact = joblib.load(os.path.join(MODELS_DIR, "code_grading_classifier_v2.joblib"))
        _state["grading_artifact"] = artifact
        _state["grading_model"] = artifact["model"]
        _state["tfidf_char"] = artifact["tfidf_char"]
        _state["tfidf_word"] = artifact["tfidf_word"]
        _state["target_map"] = artifact["target_map"]
        # Reverse map to convert class predictions [0, 1, 2] -> [15.1, 15.2, 15.3]
        _state["rev_target_map"] = {v: k for k, v in artifact["target_map"].items()}
    return _state

# Helper to clean code fences before TF-IDF extraction
def _clean_code(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    return re.sub(r"\n?```$", "", text).strip()

# Helper to extract the exact numerical features expected by V2 model
def _extract_v2_num_features(clean_code_str: str, raw_code: str) -> np.ndarray:
    code_len = len(clean_code_str)
    line_count = clean_code_str.count("\n") + 1
    has_docstring = int('"""' in raw_code or "'''" in raw_code)
    
    if_count = len(re.findall(r"\bif\b", clean_code_str))
    for_count = len(re.findall(r"\bfor\b", clean_code_str))
    while_count = len(re.findall(r"\bwhile\b", clean_code_str))
    try_count = len(re.findall(r"\btry\b", clean_code_str))
    return_count = len(re.findall(r"\breturn\b", clean_code_str))
    
    indent_lengths = [len(line) - len(line.lstrip(" ")) for line in clean_code_str.split("\n")]
    max_indent = max(indent_lengths) if indent_lengths else 0

    return np.array([[
        has_docstring, code_len, line_count, if_count,
        for_count, while_count, try_count, return_count, max_indent
    ]], dtype=float)



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


# ---------------------------------------------------------------
# Updated Grade Endpoint
# ---------------------------------------------------------------
@app.post("/api/grade", response_model=GradeResponse)
def grade(req: GradeRequest):
    try:
        state = _load_grading()
    except Exception as e:
        raise HTTPException(500, f"Grading model failed to load: {type(e).__name__}: {e}")

    # 1. Clean submission code
    clean_code_str = _clean_code(req.code)

    # 2. Compute non-leakage numerical features
    num_features = _extract_v2_num_features(clean_code_str, req.code)

    # 3. Transform via Dual TF-IDF Vectorizers
    X_char = state["tfidf_char"].transform([clean_code_str])
    X_word = state["tfidf_word"].transform([clean_code_str])

    # 4. Stack features to create the sparse matrix
    X_input = hstack([num_features, X_char, X_word]).tocsr()

    # 5. Predict discrete class label [0, 1, 2]
    class_pred = int(state["grading_model"].predict(X_input)[0])
    raw_quality_score = state["rev_target_map"].get(class_pred, 15.2)

    # 6. Optional min-max normalization to [0, 1] range
    MIN_VAL, MAX_VAL = 15.1, 15.3
    scaled_pred = float(np.clip((raw_quality_score - MIN_VAL) / (MAX_VAL - MIN_VAL), 0.0, 1.0))

    # Optional: Extract Radon complexity stats for metadata response
    complexity, loc = None, None
    if RADON_AVAILABLE:
        c_stats = _complexity_stats(req.code)
        complexity = None if np.isnan(c_stats["cyclomatic_complexity"]) else round(c_stats["cyclomatic_complexity"], 2)
        l_stat = _lines_of_code(req.code)
        loc = None if np.isnan(l_stat) else l_stat

    return GradeResponse(
        predicted_quality_score=round(scaled_pred, 4),
        model_used="XGBoost_V2_DualTFIDF",
        cyclomatic_complexity=complexity,
        lines_of_code=loc,
    )
