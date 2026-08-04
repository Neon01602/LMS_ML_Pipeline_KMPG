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
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

app = FastAPI(title="LMS ML Pipeline API")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---------------------------------------------------------------
# Load models once at cold start (reused across warm invocations)
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
    """Load only what /api/grade needs, independent of the triage models."""
    if "grading_model" not in _state:
        _state["grading_model"] = joblib.load(os.path.join(MODELS_DIR, "grading_model.pkl"))
        _state["grading_imputer"] = joblib.load(os.path.join(MODELS_DIR, "grading_imputer.pkl"))
    _load_metadata()
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
# Feature helpers (mirror the notebook's feature engineering)
# ---------------------------------------------------------------
def _cyclomatic_complexity(code: str) -> float:
    try:
        blocks = cc_visit(code)
        return float(np.mean([b.complexity for b in blocks])) if blocks else 1.0
    except Exception:
        return float("nan")


def _lines_of_code(code: str) -> float:
    try:
        return float(analyze(code).loc)
    except Exception:
        return float("nan")


def _code_structural_features(code: str) -> dict:
    """Cheap stand-ins for the dataset's own def_count / total_tokens / has_docstring,
    computed directly from the submitted code at inference time."""
    def_count = code.count("def ")
    total_tokens = len(code.split())
    has_docstring = int('"""' in code or "'''" in code)
    return {
        "def_count": def_count,
        "total_tokens": total_tokens,
        "has_docstring": has_docstring,
    }


def _normalize_quality_score(raw_pred: float, meta: dict) -> float:
    """Rescale the raw prediction (which lives in the original dataset's
    narrow quality_score range, e.g. ~15.1-15.3) into a 0-1 score.
    q_min/q_max come from metadata.json (subs['quality_score'].min()/.max()
    from the training notebook) so the API doesn't hardcode magic numbers.
    Clipped to [0, 1] in case a new submission's raw prediction falls
    slightly outside the observed training range."""
    q_min = meta.get("quality_score_min", 15.1)
    q_max = meta.get("quality_score_max", 15.3)
    span = q_max - q_min
    if span <= 0:
        return 0.0
    normalized = (raw_pred - q_min) / span
    return max(0.0, min(1.0, normalized))


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


from feature_extraction import extract_code_features

artifact = joblib.load("grading_model_v5.joblib")
model = artifact["model"]
imputer = artifact["imputer"]
feature_cols = artifact["feature_cols"]
final_feature_idx = artifact["final_feature_idx"]

app = FastAPI()

class CodeSubmission(BaseModel):
    answer: str
    pass_rate: float
    test_count: int
    total_tokens: int
    def_count: int
    has_docstring: bool

@app.post("/grade")
def grade(sub: CodeSubmission):
    feats = extract_code_features(sub.answer)
    row = {
        "pass_rate": sub.pass_rate, "test_count": sub.test_count,
        "total_tokens": sub.total_tokens, "def_count": sub.def_count,
        "has_docstring": int(sub.has_docstring), **feats,
    }
    row["tokens_per_def"] = row["total_tokens"] / (row["def_count"] + 1)
    row["complexity_per_line"] = row["cyclomatic_complexity"] / ((row["lines_of_code"] or 0) + 1)

    x = np.array([[row[c] for c in feature_cols]])
    x_imputed = imputer.transform(x)[:, final_feature_idx]
    score = float(model.predict(x_imputed)[0])
    return {"quality_score": score, "model_version": artifact["winner_name"]}
