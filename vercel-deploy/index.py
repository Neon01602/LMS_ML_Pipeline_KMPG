"""
FastAPI app serving the LMS Triage + Grading models.
Runs as a single Vercel Python serverless function using split JSON/Joblib artifacts.
"""

import ast
import json
import os
import re
from typing import Optional

# CRITICAL: Prevent LightGBM from trying to load missing system OpenMP shared libraries on Vercel
os.environ["LGB_VERBOSITY"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except OSError:
    LIGHTGBM_AVAILABLE = False

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from radon.complexity import cc_visit
    from radon.raw import analyze

    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

app = FastAPI(title="LMS ML Pipeline API")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_state = {}


def _load_metadata():
    if "metadata" not in _state:
        metadata_path = os.path.join(MODELS_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                _state["metadata"] = json.load(f)
        else:
            _state["metadata"] = {}
    return _state["metadata"]


def _load_triage():
    if "triage_bundle" not in _state:
        _state["triage_bundle"] = joblib.load(
            os.path.join(MODELS_DIR, "triage_models.joblib")
        )
    _load_metadata()
    return _state


def _load_grading():
    if "grading_bundle" not in _state:
        json_path = os.path.join(MODELS_DIR, "lgbm_model.json")
        artifacts_path = os.path.join(MODELS_DIR, "grading_artifacts.joblib")

        if not os.path.exists(json_path) or not os.path.exists(artifacts_path):
            raise FileNotFoundError(
                f"Missing grading files in {MODELS_DIR}. Ensure lgbm_model.json and grading_artifacts.joblib are present."
            )

        with open(json_path, "r") as f:
            model_str = f.read()
        
        try:
            booster = lgb.Booster(model_str=model_str)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LightGBM Booster from JSON: {e}")

        artifacts = joblib.load(artifacts_path)
        _state["grading_bundle"] = {
            "model": booster,
            "feature_cols": artifacts["feature_cols"],
            "imputer": artifacts["imputer"],
            "numeric_with_na": artifacts["numeric_with_na"],
        }
    return _state


def _clean_code(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    return re.sub(r"\n?```$", "", text).strip()


def _extract_features(
    code_str: str,
    pass_rate: Optional[float] = None,
    test_count: Optional[int] = None,
    runtime_ms: Optional[float] = None,
    memory_kb: Optional[float] = None,
    comment_density: Optional[float] = None,
    num_attempts: Optional[float] = None,
    hours_before_deadline: Optional[float] = None,
    student_avg_past_score: Optional[float] = None,
) -> dict:
    try:
        tree = ast.parse(code_str)
        ast_nodes = list(ast.walk(tree))
        syntax_valid = 1.0

        def_count = sum(1 for n in ast_nodes if isinstance(n, ast.FunctionDef))
        class_count = sum(1 for n in ast_nodes if isinstance(n, ast.ClassDef))
        loop_count = sum(
            1 for n in ast_nodes if isinstance(n, (ast.For, ast.While))
        )
        try_count = sum(1 for n in ast_nodes if isinstance(n, ast.Try))
        if_count = sum(1 for n in ast_nodes if isinstance(n, ast.If))
        return_count = sum(1 for n in ast_nodes if isinstance(n, ast.Return))
        assign_count = sum(1 for n in ast_nodes if isinstance(n, ast.Assign))
        type_ann_count = sum(
            1 for n in ast_nodes if isinstance(n, ast.AnnAssign)
        )
        comp_count = sum(
            1
            for n in ast_nodes
            if isinstance(
                n,
                (
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                ),
            )
        )

        node_count = float(len(ast_nodes))
        cyclomatic = float(
            1 + if_count + loop_count + try_count + comp_count
        )

        identifiers = [
            n.id for n in ast_nodes if isinstance(n, ast.Name) and hasattr(n, "id")
        ]
        single_char_ids = sum(1 for i in identifiers if len(i) == 1)
        single_char_ratio = (
            single_char_ids / len(identifiers) if identifiers else 0.0
        )

    except Exception:
        syntax_valid = 0.0
        node_count, def_count, class_count = 0.0, 0.0, 0.0
        loop_count, try_count, if_count = 0.0, 0.0, 0.0
        return_count, assign_count, type_ann_count = 0.0, 0.0, 0.0
        comp_count, cyclomatic, single_char_ratio = 0.0, 1.0, 0.0

    char_len = float(len(code_str))
    lines = float(max(1, code_str.count("\n") + 1))
    has_docstring = float(int('"""' in code_str or "'''" in code_str))
    avg_line_len = char_len / lines
    comment_count = float(len(re.findall(r"#.*", code_str)))
    calc_comment_density = comment_density if comment_density is not None else (comment_count / lines)
    indent_spaces = float(len(re.findall(r"^[ ]+", code_str, re.MULTILINE)))
    operator_count = float(
        len(re.findall(r"[\+\-\*\/\%\=\<\>\!\&\|\^\~]", code_str))
    )
    operator_density = operator_count / max(1.0, char_len)
    ast_density = node_count / lines
    complexity_density = cyclomatic / max(1.0, node_count)

    return {
        "test_pass_rate": pass_rate if pass_rate is not None else 0.5,
        "cyclomatic_complexity": cyclomatic,
        "lines_of_code": lines,
        "num_functions": def_count,
        "runtime_ms": runtime_ms,
        "memory_kb": memory_kb,
        "num_compile_errors": 0.0,
        "num_warnings": 0.0,
        "comment_density": calc_comment_density,
        "num_attempts": num_attempts if num_attempts is not None else 1.0,
        "hours_before_deadline": hours_before_deadline if hours_before_deadline is not None else 24.0,
        "student_avg_past_score": student_avg_past_score if student_avg_past_score is not None else 80.0,
        "runtime_ms_missing": 1.0 if runtime_ms is None else 0.0,
        "memory_kb_missing": 1.0 if memory_kb is None else 0.0,
        "comment_density_missing": 1.0 if comment_density is None else 0.0,
        "has_docstring": has_docstring,
        "char_len": char_len,
        "avg_line_len": avg_line_len,
        "comment_count": comment_count,
        "indent_spaces": indent_spaces,
        "operator_count": operator_count,
        "operator_density": operator_density,
        "syntax_valid": syntax_valid,
        "node_count": node_count,
        "class_count": class_count,
        "loop_count": loop_count,
        "try_count": try_count,
        "if_count": if_count,
        "return_count": return_count,
        "assign_count": assign_count,
        "type_ann_count": type_ann_count,
        "comp_count": comp_count,
        "single_char_ratio": single_char_ratio,
        "ast_density": ast_density,
        "complexity_density": complexity_density,
    }


class TriageRequest(BaseModel):
    post_text: str = Field(..., description="Raw text of student doubt")


class TriageResponse(BaseModel):
    predicted_topic: str
    urgency_probability: float
    auto_handle: bool
    threshold_used: float


class GradeRequest(BaseModel):
    code: str = Field(..., description="Source code submission to grade")
    pass_rate: Optional[float] = Field(None)
    test_count: Optional[int] = Field(None)
    runtime_ms: Optional[float] = Field(None)
    memory_kb: Optional[float] = Field(None)
    comment_density: Optional[float] = Field(None)
    num_attempts: Optional[float] = Field(None)
    hours_before_deadline: Optional[float] = Field(None)
    student_avg_past_score: Optional[float] = Field(None)


class GradeResponse(BaseModel):
    predicted_quality_score: float
    model_used: str
    cyclomatic_complexity: Optional[float]
    lines_of_code: Optional[float]


@app.get("/")
@app.get("/api")
def root():
    return {"message": "LMS ML Pipeline API is live"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    state = _load_triage()
    triage_bundle = state["triage_bundle"]
    topic_model = triage_bundle["topic_model"]
    urgency_model = triage_bundle["urgency_model"]
    threshold = triage_bundle["confidence_threshold"]

    predicted_topic = topic_model.predict([req.post_text])[0]
    urg_proba = urgency_model.predict_proba([req.post_text])[0]
    max_urgency_proba = float(urg_proba.max())

    return TriageResponse(
        predicted_topic=predicted_topic,
        urgency_probability=round(max_urgency_proba, 4),
        auto_handle=max_urgency_proba >= threshold,
        threshold_used=threshold,
    )


@app.post("/api/grade", response_model=GradeResponse)
def grade(req: GradeRequest):
    try:
        state = _load_grading()
    except Exception as e:
        raise HTTPException(
            500, f"Grading initialization failed: {type(e).__name__}: {e}"
        )

    bundle = state["grading_bundle"]
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    imputer = bundle["imputer"]
    numeric_with_na = bundle["numeric_with_na"]

    clean_code_str = _clean_code(req.code)
    features_dict = _extract_features(
        code_str=clean_code_str,
        pass_rate=req.pass_rate,
        test_count=req.test_count,
        runtime_ms=req.runtime_ms,
        memory_kb=req.memory_kb,
        comment_density=req.comment_density,
        num_attempts=req.num_attempts,
        hours_before_deadline=req.hours_before_deadline,
        student_avg_past_score=req.student_avg_past_score,
    )

    row_values = []
    for col in numeric_with_na:
        val = features_dict.get(col, np.nan)
        row_values.append(np.nan if val is None else val)

    imputed_array = imputer.transform([row_values])
    for idx, col in enumerate(numeric_with_na):
        features_dict[col] = imputed_array[0][idx]

    X_input = [[features_dict.get(col, 0.0) for col in feature_cols]]

    raw_predicted_score = float(model.predict(X_input)[0])
    final_score = float(np.clip(raw_predicted_score, 0.0, 1.0))

    computed_complexity = float(features_dict["cyclomatic_complexity"])
    computed_lines = float(features_dict["lines_of_code"])

    if RADON_AVAILABLE:
        try:
            blocks = cc_visit(req.code)
            if blocks:
                computed_complexity = round(float(np.mean([b.complexity for b in blocks])), 2)
            analysis = analyze(req.code)
            if analysis:
                computed_lines = float(analysis.loc)
        except Exception:
            pass

    return GradeResponse(
        predicted_quality_score=round(final_score, 4),
        model_used="lgbm_model.json",
        cyclomatic_complexity=computed_complexity,
        lines_of_code=computed_lines,
    )
