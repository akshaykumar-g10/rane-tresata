#!/usr/bin/env python3
"""
predict.py

Executable for Part A: Given a file path and a column name,
print the semantic classification of that column.

Usage:
    python3 predict.py --input /path/to/file.csv --column columnName

Example output (single line):
    CompanyName
"""

import argparse
import sys
import pandas as pd

from semantic_model import load_model, predict_column_type


# Map internal labels -> required output strings
LABEL_TO_OUTPUT = {
    "PHONE": "PhoneNumber",
    "COMPANY": "CompanyName",
    "COUNTRY": "Country",
    "DATE": "Date",
    "OTHER": "Other",
}


def main():
    parser = argparse.ArgumentParser(description="Semantic column classifier")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--column",
        type=str,
        required=True,
        help="Name of the column to classify",
    )

    args = parser.parse_args()

    input_path = args.input
    column_name = args.column

    # Load CSV
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error: failed to read CSV file '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if column_name not in df.columns:
        print(f"Error: column '{column_name}' not found in file '{input_path}'.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Load model and classify
    model = load_model()
    pred_label, proba_dict = predict_column_type(df, column_name, model=model, sample_size=200)

    # Map to required output string
    output_label = LABEL_TO_OUTPUT.get(pred_label, pred_label)

    # IMPORTANT: print only the label as per problem statement
    print(output_label)


if __name__ == "__main__":
    main()
