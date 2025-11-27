#!/usr/bin/env python3
"""
test_improved_pipeline.py

Quick end-to-end checks for:
  - cell-level predictions
  - column-level predictions (with improved aggregation)
  - column selection (best phone / company)
"""

import pandas as pd

from semantic_model import load_model, predict_cell_type, predict_column_type, predict_all_columns
from parser import find_best_columns


def test_cell_level(model):
    print("=== CELL-LEVEL TESTS ===")
    samples = [
        "+91 9876543210",
        "2024-11-27",
        "01/01/2000",
        "Tresata Pvt Ltd",
        "India",
        "1234567890",
        "foo bar 123",
    ]

    for s in samples:
        label, probs = predict_cell_type(s, model=model)
        print(f"  '{s}': {label} {probs}")


def test_column_level(model):
    print("\n=== COLUMN-LEVEL TESTS (TRAINING FILES) ===")
    # Adjust column names if your CSVs use different headers
    df_phone = pd.read_csv("data/phone.csv")
    phone_col = df_phone.columns[0]  # assume first column is the phone numbers
    label_p, probs_p = predict_column_type(df_phone, phone_col, model=model)
    print(f"  phoneNumber.csv [{phone_col}]: {label_p} {probs_p}")

    df_company = pd.read_csv("data/company.csv")
    company_col = df_company.columns[0]  # assume first column is the company names
    label_c, probs_c = predict_column_type(df_company, company_col, model=model)
    print(f"  Company.csv [{company_col}]: {label_c} {probs_c}")


def test_file_level_selection(model):
    print("\n=== FILE-LEVEL SELECTION (PARSER INPUT FILES) ===")

    # 1) phone.csv – should find a phone column, no company column
    try:
        df_phone_file = pd.read_csv("data/phone.csv")
        best_phone_col, best_company_col, results = find_best_columns(df_phone_file)
        print("  data/phone.csv:")
        print(f"    best_phone_col   = {best_phone_col}")
        print(f"    best_company_col = {best_company_col}")
        for col, (label, probs) in results.items():
            print(f"      - {col}: {label} {probs}")
    except FileNotFoundError:
        print("  [SKIP] data/phone.csv not found")

    # 2) company.csv – should find a company column, no phone column
    try:
        df_company_file = pd.read_csv("data/company.csv")
        best_phone_col, best_company_col, results = find_best_columns(df_company_file)
        print("\n  data/company.csv:")
        print(f"    best_phone_col   = {best_phone_col}")
        print(f"    best_company_col = {best_company_col}")
        for col, (label, probs) in results.items():
            print(f"      - {col}: {label} {probs}")
    except FileNotFoundError:
        print("  [SKIP] data/company.csv not found")


def main():
    model = load_model()
    test_cell_level(model)
    test_column_level(model)
    test_file_level_selection(model)


if __name__ == "__main__":
    main()
