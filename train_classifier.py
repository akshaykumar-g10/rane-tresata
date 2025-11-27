#!/usr/bin/env python3
"""
Train a semantic column-value classifier with 5 classes:
PHONE, COMPANY, COUNTRY, DATE, OTHER.

Expected data layout (you can adjust paths with --data_dir):

data/
    Company.csv
    Countries.txt
    Dates.csv
    phoneNumber.csv
    legal.txt

Usage:
    python train_classifier.py \
        --data_dir ./data \
        --model_path ./models/semantic_classifier.joblib
"""

import os
import argparse
import random
from typing import List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from text_features import TextStatsExtractor, normalize_text


# ----------------------------
# 1. Load training data
# ----------------------------

def load_list_from_txt(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def load_first_column_from_csv(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df.shape[1] == 0:
        return []
    col = df.iloc[:, 0].astype(str)
    return [normalize_text(x) for x in col.tolist()]


# ----------------------------
# 2. Build labeled dataset (no synthetic aug)
# ----------------------------

def build_labeled_dataset(data_dir: str) -> pd.DataFrame:
    """
    Build a labeled dataset with columns: 'text', 'label'.
    Labels: PHONE, COMPANY, COUNTRY, DATE, OTHER
    """

    phone_path = os.path.join(data_dir, "phone.csv")
    company_path = os.path.join(data_dir, "company.csv")
    countries_path = os.path.join(data_dir, "countries.txt")
    dates_path = os.path.join(data_dir, "dates.csv")
    legal_path = os.path.join(data_dir, "legal.txt")  # just to ensure exists

    phone_values = load_first_column_from_csv(phone_path)
    company_values = load_first_column_from_csv(company_path)
    country_values = load_list_from_txt(countries_path)
    date_values = load_first_column_from_csv(dates_path)

    phone_values = [normalize_text(x) for x in phone_values if str(x).strip()]
    company_values = [normalize_text(x) for x in company_values if str(x).strip()]
    country_values = [normalize_text(x) for x in country_values if str(x).strip()]
    date_values = [normalize_text(x) for x in date_values if str(x).strip()]

    rows = []

    for v in phone_values:
        rows.append({"text": v, "label": "PHONE"})
    for v in company_values:
        rows.append({"text": v, "label": "COMPANY"})
    for v in country_values:
        rows.append({"text": v, "label": "COUNTRY"})
    for v in date_values:
        rows.append({"text": v, "label": "DATE"})

    # Generate OTHER examples (simple random junk)
    other_samples = []
    num_other = max(500, int(0.5 * len(rows)))  # at least 500, or 50% of size

    def random_numeric():
        return str(random.randint(0, 10**10))

    def random_alpha():
        length = random.randint(3, 10)
        return "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))

    def random_alphanum():
        length = random.randint(5, 12)
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        return "".join(random.choice(chars) for _ in range(length))

    generators = [random_numeric, random_alpha, random_alphanum]

    for _ in range(num_other):
        g = random.choice(generators)
        v = g()
        other_samples.append(v)

    for v in other_samples:
        rows.append({"text": v, "label": "OTHER"})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


# ----------------------------
# 3. Build model pipeline
# ----------------------------

def build_pipeline(country_list: List[str], legal_terms: List[str]) -> Pipeline:
    """
    Build a sklearn Pipeline:
      [FeatureUnion(TextStatsExtractor + char-TFIDF)] -> LogisticRegression
    """

    country_set = {c.strip().lower() for c in country_list if c.strip()}
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

    clf = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        n_jobs=-1
        # no class_weight here – simpler, more stable
    )

    pipe = Pipeline([
        ("features", feats),
        ("clf", clf),
    ])

    return pipe


# ----------------------------
# 4. Main training entrypoint
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train semantic type classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing Company.csv, Countries.txt, Dates.csv, phoneNumber.csv, legal.txt"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/semantic_classifier.joblib",
        help="Where to save the trained model pipeline"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"[INFO] Loading training data from: {data_dir}")
    df = build_labeled_dataset(data_dir)
    print(f"[INFO] Dataset size: {len(df)} rows")
    print(df["label"].value_counts())

    countries_path = os.path.join(data_dir, "Countries.txt")
    legal_path = os.path.join(data_dir, "legal.txt")
    countries = load_list_from_txt(countries_path)
    legal_terms = load_list_from_txt(legal_path)

    print("[INFO] Building model pipeline...")
    pipe = build_pipeline(countries, legal_terms)

    X = df["text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Training classifier...")
    pipe.fit(X_train, y_train)

    print("[INFO] Evaluating on hold-out set...")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"[INFO] Saving model to: {model_path}")
    joblib.dump(pipe, model_path)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
