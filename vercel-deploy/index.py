"""
FastAPI app serving the LMS Triage + Grading models.
Runs as a single Vercel Python serverless function (all routes handled
internally by FastAPI, see vercel.json for the catch-all rewrite).
"""

import json
import os
import re
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from lgbm_interpreter import PurePythonLGBM
from pydantic import BaseModel, Field
from scipy.sparse import hstack

# radon is only needed for the grading endpoint (computes complexity from code)
try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
    from radon.raw import analyze

    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

app = FastAPI(title="LMS ML Pipeline API")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---------------------------------------------------------------
# Load models once at cold start (reused across warm invocations).
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
        _state["vec"] = joblib.load(
            os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        )
        _state["topic_clf"] = joblib.load(
            os.path.join(MODELS_DIR, "topic_classifier.pkl")
        )
        _state["urgency_clf"] = joblib.load(
            os.path.join(MODELS_DIR, "urgency_classifier.pkl")
        )
    _load_metadata()
    return _state


def _load_grading():
    """Loads Pure Python LightGBM tree structure and feature vectorizers."""
    if "grading_engine" not in _state:
        tree_path = os.path.join(MODELS_DIR, "lgbm_model_trees.json")
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_and_features.joblib")

        if not os.path.exists(tree_path):
            raise FileNotFoundError(f"Tree structure file not found at {tree_path}")

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Feature vectorizer file not found at {vectorizer_path}"
            )

        _state["grading_engine"] = PurePythonLGBM(tree_path)

        feature_artifacts = joblib.load(vectorizer_path)
        _state["tfidf_char"] = feature_artifacts["tfidf_char"]
        _state["tfidf_word"] = feature_artifacts["tfidf_word"]

    return _state


def _clean_code(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    return re.sub(r"\n?```$", "", text).strip()


# ---------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------
class TriageRequest(BaseModel):
    post_text: str = Field(
        ..., description="Raw text of the student's forum post/doubt"
    )


class TriageResponse(BaseModel):
    predicted_topic: str
    urgency_probability: float
    auto_handle: bool
    threshold_used: float


class GradeRequest(BaseModel):
    code: str = Field(..., description="Source code submission to grade")
    pass_rate: Optional[float] = Field(
        None, description="Fraction of tests passed, 0-1, if known"
    )
    test_count: Optional[int] = Field(
        None, description="Number of tests run against this submission, if known"
    )


class GradeResponse(BaseModel):
    predicted_quality_score: float
    model_used: str
    cyclomatic_complexity: Optional[float]
    lines_of_code: Optional[float]


# ---------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------
def _complexity_stats(code: str) -> dict:
    try:
        blocks = cc_visit(code)
        complexities = [b.complexity for b in blocks] if blocks else [1.0]
        return {
            "cyclomatic_complexity": float(np.mean(complexities)),
            "max_complexity": float(np.max(complexities)),
        }
    except Exception:
        return {
            "cyclomatic_complexity": float("nan"),
            "max_complexity": float("nan"),
        }


def _lines_of_code(code: str) -> float:
    try:
        return float(analyze(code).loc)
    except Exception:
        return float("nan")


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
    try:
        state = _load_grading()
    except Exception as e:
        raise HTTPException(
            500, f"Grading model failed to load: {type(e).__name__}: {e}"
        )

    raw_code = req.code
    clean_code_str = _clean_code(raw_code)

    # 1. Compute numerical features
    code_len = float(len(clean_code_str))
    line_count = float(clean_code_str.count("\n") + 1)
    has_docstring = float(int('"""' in raw_code or "'''" in raw_code))

    X_num = np.array([[has_docstring, code_len, line_count]], dtype=float)

    # 2. Extract dual TF-IDF features
    X_char = state["tfidf_char"].transform([clean_code_str])
    X_word = state["tfidf_word"].transform([clean_code_str])

    # 3. Stack into unified feature vector and convert to dense array
    X_combined = hstack([X_num, X_char, X_word]).tocsr()
    dense_features = X_combined.toarray().ravel()

    # 4. Predict continuous quality score using Pure-Python LightGBM engine
    predicted_score = state["grading_engine"].predict_one(dense_features)

    # Radon Complexity Metrics
    complexity, loc = None, None
    if RADON_AVAILABLE:
        c_stats = _complexity_stats(raw_code)
        complexity = (
            None
            if np.isnan(c_stats["cyclomatic_complexity"])
            else round(c_stats["cyclomatic_complexity"], 2)
        )
        l_stat = _lines_of_code(raw_code)
        loc = None if np.isnan(l_stat) else l_stat

    return GradeResponse(
        predicted_quality_score=round(predicted_score, 4),
        model_used="LightGBM_Pure_Python_Interpreter",
        cyclomatic_complexity=complexity,
        lines_of_code=loc,
    )
