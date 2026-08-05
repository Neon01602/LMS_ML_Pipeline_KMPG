"""
FastAPI app serving the LMS Triage + Grading models.
Pure-Python LightGBM text-format parser + evaluator (no native lib needed).
"""

import ast
import json
import os
import re
from typing import Optional

import joblib
import numpy as np

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


# ---------------------------------------------------------------------------
# Pure-Python LightGBM text-format parser
# ---------------------------------------------------------------------------

def _parse_lgbm_text(text: str) -> list:
    lines = [ln.strip() for ln in text.splitlines()]
    trees = []
    current = None

    def flush():
        if current is not None and "split_feature" in current:
            trees.append(current)

    for ln in lines:
        if not ln:
            continue
        if ln.startswith("Tree="):
            flush()
            current = {}
            continue
        if ln.startswith("end of trees") or ln.startswith("feature_importances"):
            flush()
            current = None
            continue
        if current is None:
            continue
        if "=" not in ln:
            continue
        key, _, val = ln.partition("=")
        key = key.strip()
        val = val.strip()

        if key in ("split_feature", "decision_type", "left_child", "right_child"):
            current[key] = [int(v) for v in val.split()]
        elif key in ("threshold", "leaf_value"):
            current[key] = [float(v) for v in val.split()]
        elif key == "num_leaves":
            current["num_leaves"] = int(val)

    flush()
    return trees


def _build_tree_struct(tree: dict, node_idx: int) -> dict:
    if node_idx < 0:
        leaf_idx = -(node_idx + 1)
        return {"leaf_value": tree["leaf_value"][leaf_idx]}

    decision_type = tree["decision_type"][node_idx]
    default_left = True

    return {
        "split_feature": tree["split_feature"][node_idx],
        "threshold": tree["threshold"][node_idx],
        "decision_type": decision_type,
        "default_left": default_left,
        "left_child": _build_tree_struct(tree, tree["left_child"][node_idx]),
        "right_child": _build_tree_struct(tree, tree["right_child"][node_idx]),
    }


class PurePythonLGBMRegressor:
    def __init__(self, model_text: str):
        raw_trees = _parse_lgbm_text(model_text)
        if not raw_trees:
            raise ValueError(
                "Parsed 0 trees from LGBM_MODEL_TEXT — the text dump is empty "
                "or malformed. Check lgbm_model_data.py generation."
            )
        self.trees = [_build_tree_struct(t, 0) for t in raw_trees]
        self.base_score = 0.0  # Tree=0 (shrinkage=1) already carries the base level

    def _predict_tree(self, node: dict, x: np.ndarray) -> float:
        if "leaf_value" in node:
            return float(node["leaf_value"])

        feat_idx = int(node["split_feature"])
        val = x[feat_idx] if feat_idx < len(x) else np.nan
        threshold = float(node["threshold"])

        if np.isnan(val):
            next_node = node["left_child"] if node["default_left"] else node["right_child"]
        else:
            is_left = val <= threshold
            next_node = node["left_child"] if is_left else node["right_child"]

        return self._predict_tree(next_node, x)

    def predict(self, X, clip=True):
        predictions = []
        for x in X:
            if hasattr(x, "toarray"):
                x = x.toarray().ravel()
            x_arr = np.asarray(x, dtype=float)
            total = self.base_score
            for tree in self.trees:
                total += self._predict_tree(tree, x_arr)
            # Target scale is 0-100 (percentage/quality score), NOT 0-1.
            predictions.append(float(np.clip(total, 0.0, 100.0)) if clip else float(total))
        return np.array(predictions)


# ---------------------------------------------------------------------------
# Grading model loading — isolated so a broken lgbm_model_data.py
# can NEVER crash the whole app / triage endpoint at import time.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Grading model loading — isolated so a broken lgbm_model_data.py
# can NEVER crash the whole app / triage endpoint at import time.
# ---------------------------------------------------------------------------

_GRADING_IMPORT_ERROR = None


def _get_lgbm_model_text():
    """Lazily import LGBM_MODEL_TEXT so import errors surface as a clean
    500 on /api/grade instead of killing the entire serverless function."""
    global _GRADING_IMPORT_ERROR
    try:
        from lgbm_model_data import LGBM_MODEL_TEXT
        return LGBM_MODEL_TEXT
    except Exception as e:
        _GRADING_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def _load_grading():
    if "grading_bundle" not in _state:
        artifacts_path = os.path.join(MODELS_DIR, "grading_artifacts.joblib")
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Missing grading_artifacts.joblib in {MODELS_DIR}.")

        model_text = _get_lgbm_model_text()
        model = PurePythonLGBMRegressor(model_text)
        artifacts = joblib.load(artifacts_path)

        _state["grading_bundle"] = {
            "model": model,
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
        loop_count = sum(1 for n in ast_nodes if isinstance(n, (ast.For, ast.While)))
        try_count = sum(1 for n in ast_nodes if isinstance(n, ast.Try))
        if_count = sum(1 for n in ast_nodes if isinstance(n, ast.If))
        return_count = sum(1 for n in ast_nodes if isinstance(n, ast.Return))
        assign_count = sum(1 for n in ast_nodes if isinstance(n, ast.Assign))
        type_ann_count = sum(1 for n in ast_nodes if isinstance(n, ast.AnnAssign))
        comp_count = sum(
            1 for n in ast_nodes
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
        )

        node_count = float(len(ast_nodes))
        cyclomatic = float(1 + if_count + loop_count + try_count + comp_count)

        identifiers = [n.id for n in ast_nodes if isinstance(n, ast.Name) and hasattr(n, "id")]
        single_char_ids = sum(1 for i in identifiers if len(i) == 1)
        single_char_ratio = single_char_ids / len(identifiers) if identifiers else 0.0

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
    operator_count = float(len(re.findall(r"[\+\-\*\/\%\=\<\>\!\&\|\^\~]", code_str)))
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
    try:
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
    except Exception as e:
        raise HTTPException(500, f"Triage failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Canonical order the LightGBM booster indexes split_feature against.
# MUST match the "feature_names=" line in the LGBM text dump exactly.
# ---------------------------------------------------------------------------
LGBM_FEATURE_ORDER = [
    "test_pass_rate",
    "cyclomatic_complexity",
    "lines_of_code",
    "num_functions",
    "runtime_ms",
    "memory_kb",
    "num_compile_errors",
    "num_warnings",
    "comment_density",
    "num_attempts",
    "hours_before_deadline",
    "student_avg_past_score",
    "runtime_ms_missing",
    "memory_kb_missing",
    "comment_density_missing",
]

@app.post("/api/grade", response_model=GradeResponse)
def grade(req: GradeRequest):
    try:
        state = _load_grading()
    except Exception as e:
        detail = f"Grading initialization failed: {type(e).__name__}: {e}"
        if _GRADING_IMPORT_ERROR:
            detail += f" | lgbm_model_data import error: {_GRADING_IMPORT_ERROR}"
        raise HTTPException(500, detail)

    bundle = state["grading_bundle"]
    model = bundle["model"]
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

    X_input = [[features_dict.get(col, 0.0) for col in LGBM_FEATURE_ORDER]]

    raw_score_0_100 = float(model.predict(X_input)[0])   # clipped to [0, 100] inside model.predict
    final_score = round(raw_score_0_100 / 100.0, 4)       # normalize to [0, 1]

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
        predicted_quality_score=final_score,
        model_used="lgbm_model.txt (pure-python parser)",
        cyclomatic_complexity=computed_complexity,
        lines_of_code=computed_lines,
    )
