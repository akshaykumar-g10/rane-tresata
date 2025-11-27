#!/usr/bin/env python3
"""
semantic_model_svm.py

Utility functions to load the SVM-based semantic classifier and
run predictions on individual cells and entire columns.

Relies on the model trained and saved by train_classifier_svm.py:
    ./models/semantic_classifier_svm.joblib
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

# IMPORTANT: ensure TextStatsExtractor is importable for joblib
import text_features  # noqa: F401

DEFAULT_MODEL_PATH = os.path.join("models", "semantic_classifier_svm.joblib")

_model_cache = None
_model_path_cache = None


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over 1D array of scores."""
    scores = np.asarray(scores, dtype=float)
    scores = scores - np.max(scores)
    exp = np.exp(scores)
    return exp / exp.sum() if exp.sum() > 0 else np.ones_like(exp) / len(exp)


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained sklearn Pipeline (SVM-based) from disk.
    Uses a simple cache to avoid reloading on every call.
    """
    global _model_cache, _model_path_cache

    if _model_cache is not None and _model_path_cache == model_path:
        return _model_cache

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Run train_classifier_svm.py first or check the path."
        )

    model = joblib.load(model_path)
    _model_cache = model
    _model_path_cache = model_path
    return model


def predict_cell_type(
    text: str,
    model=None
) -> Tuple[str, Dict[str, float]]:
    """
    Predict the semantic type of a single cell string using the SVM model.

    Returns:
        (pred_label, proba_dict) where proba_dict is obtained by applying
        softmax to the decision_function scores.
    """
    if model is None:
        model = load_model()

    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    X = [text]

    # LinearSVC exposes decision_function, not predict_proba.
    scores = model.decision_function(X)[0]  # shape: (n_classes,) or scalar for binary
    labels = model.classes_

    if np.ndim(scores) == 0:
        # Binary case: decision_function returns scalar; map to 2D scores
        scores = np.array([-scores, scores])

    probs = _softmax(scores)
    proba_dict = {label: float(p) for label, p in zip(labels, probs)}
    pred_label = labels[int(np.argmax(probs))]

    return pred_label, proba_dict


def predict_column_type(
    df: pd.DataFrame,
    column_name: str,
    model=None,
    sample_size: int = 200
) -> Tuple[str, Dict[str, float]]:
    """
    Predict the semantic type of an entire column by aggregating
    cell-level pseudo-probabilities using the SVM model.
    """
    if model is None:
        model = load_model()

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    col_series = df[column_name].dropna().astype(str)
    col_series = col_series[col_series.str.strip() != ""]

    if len(col_series) == 0:
        labels = list(model.classes_)
        agg_proba_dict = {label: 0.0 for label in labels}
        if "OTHER" in agg_proba_dict:
            agg_proba_dict["OTHER"] = 1.0
            return "OTHER", agg_proba_dict
        else:
            pred = labels[0]
            agg_proba_dict[pred] = 1.0
            return pred, agg_proba_dict

    if len(col_series) > sample_size:
        col_series = col_series.sample(n=sample_size, random_state=42)

    texts = col_series.tolist()

    # Get decision scores for all rows
    scores_matrix = model.decision_function(texts)  # shape: (n_samples, n_classes)
    labels = model.classes_

    # Convert each row's scores to probs, then average
    probs_matrix = np.vstack([_softmax(scores) for scores in scores_matrix])
    mean_proba = np.mean(probs_matrix, axis=0)

    agg_proba_dict = {label: float(p) for label, p in zip(labels, mean_proba)}
    pred_label = labels[int(np.argmax(mean_proba))]

    return pred_label, agg_proba_dict


def predict_all_columns(
    df: pd.DataFrame,
    model=None,
    sample_size: int = 200
):
    """
    Predict semantic type for every column in a DataFrame using the SVM model.

    Returns:
        {
          column_name: (pred_label, proba_dict),
          ...
        }
    """
    if model is None:
        model = load_model()

    results = {}
    for col in df.columns:
        label, probs = predict_column_type(df, col, model=model, sample_size=sample_size)
        results[col] = (label, probs)
    return results
