#!/usr/bin/env python3
"""
semantic_model.py

Utilities to:
  - load the trained semantic classifier
  - run cell-level predictions
  - run robust column-level predictions
"""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib

# Ensure custom transformers are importable when unpickling
import text_features  # noqa: F401

DEFAULT_MODEL_PATH = os.path.join("models", "semantic_classifier.joblib")

_model_cache = None
_model_path_cache = None


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained sklearn Pipeline from disk.
    Uses a simple in-process cache to avoid re-loading repeatedly.
    """
    global _model_cache, _model_path_cache

    if _model_cache is not None and _model_path_cache == model_path:
        return _model_cache

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Run train_classifier.py first or check the path."
        )

    model = joblib.load(model_path)
    _model_cache = model
    _model_path_cache = model_path
    return model


# -------------------------------------------------------------------
# Cell-level prediction
# -------------------------------------------------------------------

def predict_cell_type(
    text: str,
    model=None
) -> Tuple[str, Dict[str, float]]:
    """
    Predict the semantic type of a single cell string.

    Returns:
        (pred_label, proba_dict)
    """
    if model is None:
        model = load_model()

    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    proba = model.predict_proba([text])[0]  # shape: (n_classes,)
    labels = model.classes_

    proba_dict = {label: float(p) for label, p in zip(labels, proba)}
    pred_label = labels[int(np.argmax(proba))]

    return pred_label, proba_dict


# -------------------------------------------------------------------
# Column-level aggregation (improved)
# -------------------------------------------------------------------

# Tunable heuristics for column decision
_MAJORITY_STRONG = 0.7           # fraction of rows agreeing on same label
_PROBA_THRESHOLD = 0.5           # min mean/weighted proba for top label
_MARGIN_THRESHOLD = 0.05         # min gap between top and second label


def _aggregate_column_probs(
    texts,
    model,
) -> Tuple[str, Dict[str, float]]:
    """
    Aggregate cell-level predictions for a column into a single label +
    probability distribution.

    Improvements over simple mean:
      - use confidence-weighted averaging
      - respect majority label when it's strong
    """
    # Predict probabilities for all sampled texts at once
    proba_matrix = model.predict_proba(texts)        # (n_samples, n_classes)
    labels = model.classes_
    n_samples = proba_matrix.shape[0]

    # Confidence per row = max probability
    confidences = proba_matrix.max(axis=1)           # (n_samples,)

    # Confidence-normalized weights
    weight_sum = confidences.sum()
    if weight_sum <= 0:
        weights = np.ones_like(confidences) / n_samples
    else:
        weights = confidences / weight_sum           # (n_samples,)

    # Confidence-weighted mean probabilities
    weighted_mean = (proba_matrix * weights[:, None]).sum(axis=0)  # (n_classes,)

    # Majority label based on argmax per row
    pred_indices = proba_matrix.argmax(axis=1)
    label_counts = {label: 0 for label in labels}
    for idx in pred_indices:
        label_counts[labels[idx]] += 1

    label_fracs = {label: count / n_samples for label, count in label_counts.items()}
    majority_label = max(labels, key=lambda l: label_fracs[l])
    majority_frac = label_fracs[majority_label]

    # Top-2 labels by weighted mean probability
    sorted_idx = np.argsort(weighted_mean)[::-1]
    top_idx = int(sorted_idx[0])
    if len(sorted_idx) > 1:
        second_idx = int(sorted_idx[1])
    else:
        second_idx = top_idx

    top_label = labels[top_idx]
    top_proba = float(weighted_mean[top_idx])
    second_proba = float(weighted_mean[second_idx])
    margin = top_proba - second_proba

    # Decision logic:
    #  1) If a strong majority of rows agree, trust majority label.
    #  2) Else, if top label is clearly above threshold with margin, choose it.
    #  3) Else, fall back to majority if it's at least 50%.
    #  4) Else, default to OTHER if available, otherwise top label.
    if majority_frac >= _MAJORITY_STRONG:
        chosen_label = majority_label
    else:
        if top_proba >= _PROBA_THRESHOLD and margin >= _MARGIN_THRESHOLD:
            chosen_label = top_label
        else:
            if majority_frac >= 0.5:
                chosen_label = majority_label
            else:
                chosen_label = "OTHER" if "OTHER" in labels else top_label

    agg_proba_dict = {label: float(p) for label, p in zip(labels, weighted_mean)}
    return chosen_label, agg_proba_dict


def predict_column_type(
    df: pd.DataFrame,
    column_name: str,
    model=None,
    sample_size: int = 200
) -> Tuple[str, Dict[str, float]]:
    """
    Predict the semantic type of an entire column by aggregating
    cell-level probabilities in a robust way.

    - Drops NaNs / empty strings.
    - Samples up to `sample_size` rows for speed.
    - Uses confidence-weighted mean and majority logic for stability.
    """
    if model is None:
        model = load_model()

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    col_series = df[column_name].dropna().astype(str)
    col_series = col_series[col_series.str.strip() != ""]

    # Empty column → default to OTHER if possible
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

    # Sample for efficiency
    if len(col_series) > sample_size:
        col_series = col_series.sample(n=sample_size, random_state=42)

    texts = col_series.tolist()
    chosen_label, agg_proba_dict = _aggregate_column_probs(texts, model)
    return chosen_label, agg_proba_dict


def predict_all_columns(
    df: pd.DataFrame,
    model=None,
    sample_size: int = 200
):
    """
    Predict semantic type for every column in a DataFrame.

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
