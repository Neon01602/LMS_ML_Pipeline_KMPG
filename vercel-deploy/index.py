"""
FastAPI app serving the LMS Triage + Grading models.
Runs as a single Vercel Python serverless function.
"""

import ast
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
        with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
            _state["metadata"] = json.load(f)
    return _state["metadata"]


def _load_triage():
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
    if "grading_engine" not in _state:
        tree_path = os.path.join(MODELS_DIR, "lgbm_model_trees.json")
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_and_features.joblib")

        if not os.path.exists(tree_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Missing model files in {MODELS_DIR}. Check deployment bundle."
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


def _extract_ast_and_surface_features(
    code_str: str,
    pass_rate: Optional[float] = None,
    test_count: Optional[int] = None,
):
    """Reconstructs the numerical feature vector matching model training."""
    pass_val = pass_rate if pass_rate is not None else 0.5
    tests_val = float(test_count) if test_count is not None else 0.0

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
    comment_density = comment_count / lines
    indent_spaces = float(len(re.findall(r"^[ ]+", code_str, re.MULTILINE)))
    operator_count = float(
        len(re.findall(r"[\+\-\*\/\%\=\<\>\!\&\|\^\~]", code_str))
    )
    operator_density = operator_count / max(1.0, char_len)
    ast_density = node_count / lines
    complexity_density = cyclomatic / max(1.0, node_count)

    # Return dense numerical vector matching feature order
    num_feats = [
        pass_val,
        tests_val,
        has_docstring,
        char_len,
        lines,
        avg_line_len,
        comment_count,
        comment_density,
        indent_spaces,
        operator_count,
        operator_density,
        syntax_valid,
        node_count,
        def_count,
        class_count,
        loop_count,
        try_count,
        if_count,
        return_count,
        assign_count,
        type_ann_count,
        comp_count,
        cyclomatic,
        single_char_ratio,
        ast_density,
        complexity_density,
    ]
    return np.array([num_feats], dtype=float)


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
            500, f"Grading initialization failed: {type(e).__name__}: {e}"
        )

    clean_code_str = _clean_code(req.code)

    # 1. Extract numerical AST and surface features
    X_num = _extract_ast_and_surface_features(
        clean_code_str, req.pass_rate, req.test_count
    )

    # 2. Extract TF-IDF features
    X_char = state["tfidf_char"].transform([clean_code_str])
    X_word = state["tfidf_word"].transform([clean_code_str])

    # 3. Stack features into a single sparse array
    X_combined = hstack([X_num, X_char, X_word]).tocsr()
    dense_features = X_combined.toarray().ravel()

    # 4. Pad array if total feature count is less than 1028
    REQUIRED_FEATURES = 1028
    if len(dense_features) < REQUIRED_FEATURES:
        padding = np.zeros(REQUIRED_FEATURES - len(dense_features))
        dense_features = np.concatenate([dense_features, padding])

    # 5. Predict quality score
    predicted_score = state["grading_engine"].predict_one(dense_features)

    # 6. Extract Radon Code Complexity Metrics
    complexity, loc = None, None
    if RADON_AVAILABLE:
        # Cyclomatic Complexity
        try:
            blocks = cc_visit(req.code)
            if blocks:
                complexities = [b.complexity for b in blocks]
                complexity = round(float(np.mean(complexities)), 2)
            else:
                complexity = 1.0
        except Exception:
            complexity = None
    
        # Lines of Code
        try:
            analysis = analyze(req.code)
            loc = float(analysis.loc)
        except Exception:
            loc = None

    return GradeResponse(
        predicted_quality_score=round(predicted_score, 4),
        model_used="Pure_Python_LightGBM_Tree_Interpreter",
        cyclomatic_complexity=complexity,
        lines_of_code=loc,
    )
