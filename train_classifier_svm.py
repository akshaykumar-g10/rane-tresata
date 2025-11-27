#!/usr/bin/env python3
"""
Train an alternative semantic column-value classifier using Linear SVM
instead of Logistic Regression.

This reuses the same dataset and feature construction as train_classifier.py,
but swaps the final classifier.

Usage:
    python train_classifier_svm.py \
        --data_dir ./data \
        --model_path ./models/semantic_classifier_svm.joblib
"""

import os
import argparse

import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Reuse dataset + utils from the LR trainer
from train_classifier import build_labeled_dataset, load_list_from_txt
from text_features import TextStatsExtractor


# ----------------------------
# Build SVM-based pipeline
# ----------------------------

def build_svm_pipeline(countries, legal_terms) -> Pipeline:
    """
    Build a sklearn Pipeline:
      [FeatureUnion(TextStatsExtractor + char-TFIDF)] -> LinearSVC
    """
    country_set = {c.strip().lower() for c in countries if c.strip()}
    legal_terms_norm = [t.strip().lower() for t in legal_terms if t.strip()]

    stats_extractor = TextStatsExtractor(
        country_set=country_set,
        legal_terms=legal_terms_norm
    )

    stats_pipeline = Pipeline([
        ("stats", stats_extractor),
        ("scaler", StandardScaler()),
    ])

    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2
    )

    feats = FeatureUnion([
        ("char_tfidf", char_tfidf),
        ("stats", stats_pipeline),
    ])

    base_svm = LinearSVC()  # linear margin-based classifier

    pipe = Pipeline([
        ("features", feats),
        ("clf", base_svm),
    ])

    return pipe


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SVM-based semantic type classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing Company.csv, Countries.txt, Dates.csv, phoneNumber.csv, legal.txt"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/semantic_classifier_svm.joblib",
        help="Where to save the trained SVM-based model pipeline"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"[INFO] Loading training data from: {data_dir}")
    df = build_labeled_dataset(data_dir)
    print(f"[INFO] Dataset size: {len(df)} rows")
    print(df["label"].value_counts())

    countries_path = os.path.join(data_dir, "countries.txt")
    legal_path = os.path.join(data_dir, "legal.txt")
    countries = load_list_from_txt(countries_path)
    legal_terms = load_list_from_txt(legal_path)

    print("[INFO] Building SVM pipeline...")
    pipe = build_svm_pipeline(countries, legal_terms)

    X = df["text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Training SVM classifier...")
    pipe.fit(X_train, y_train)

    print("[INFO] Evaluating on hold-out set...")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"[INFO] Saving SVM model to: {model_path}")
    joblib.dump(pipe, model_path)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
