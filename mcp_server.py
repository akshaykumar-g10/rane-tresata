#!/usr/bin/env python3
"""
mcp_server.py

MCP server exposing:
  - list_files: list available CSV files
  - column_prediction: classify a column's semantic type
  - parse_file: run end-to-end parsing (Part B) on a file

Intended to be used by MCP-compatible clients (ChatGPT, Claude, etc.).
"""

import os
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

# MCP SDK / FastMCP
from mcp.server.fastmcp import FastMCP

# Reuse our existing logic
from semantic_model import load_model, predict_column_type, predict_all_columns
from parsers import parse_phone, parse_company, load_legal_suffixes

# NOTE: we duplicate a bit of logic from parser.py here to avoid circular imports.


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Base directory where input files live (you can adjust)
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Thresholds for deciding if a file really has a phone/company column
PHONE_MIN_PROB = 0.60          # min avg PHONE prob for best column
COMPANY_MIN_PROB = 0.55        # min avg COMPANY prob for best column
PHONE_MIN_MARGIN = 0.05        # min gap vs 2nd-best PHONE column
COMPANY_MIN_MARGIN = 0.05      # min gap vs 2nd-best COMPANY column

# Label mapping to human-facing names (same as predict.py)
LABEL_TO_OUTPUT = {
    "PHONE": "PhoneNumber",
    "COMPANY": "CompanyName",
    "COUNTRY": "Country",
    "DATE": "Date",
    "OTHER": "Other",
}

# Initialize MCP server
mcp = FastMCP("tresata_semantic_server")


# -------------------------------------------------------------------
# Helper functions (reused from parser.py logic)
# -------------------------------------------------------------------



def find_best_columns(df: pd.DataFrame):
    """
    Use the classifier to predict all columns and pick the best
    Phone and Company columns in a robust way.

    Returns:
        (best_phone_col, best_company_col, all_results)

        - best_phone_col: column name or None
        - best_company_col: column name or None
        - all_results: dict[col] = (label, proba_dict)
    """
    model = load_model()
    results = predict_all_columns(df, model=model, sample_size=200)

    # Track best & second-best PHONE columns
    best_phone_col = None
    best_phone_prob = -1.0
    second_phone_prob = -1.0

    # Track best & second-best COMPANY columns
    best_company_col = None
    best_company_prob = -1.0
    second_company_prob = -1.0

    for col, (_label, probs) in results.items():
        phone_prob = float(probs.get("PHONE", 0.0))
        company_prob = float(probs.get("COMPANY", 0.0))

        # Update PHONE ranking
        if phone_prob > best_phone_prob:
            second_phone_prob = best_phone_prob
            best_phone_prob = phone_prob
            best_phone_col = col
        elif phone_prob > second_phone_prob:
            second_phone_prob = phone_prob

        # Update COMPANY ranking
        if company_prob > best_company_prob:
            second_company_prob = best_company_prob
            best_company_prob = company_prob
            best_company_col = col
        elif company_prob > second_company_prob:
            second_company_prob = company_prob

    # Compute margins between best and 2nd-best columns for each type
    phone_margin = best_phone_prob - max(second_phone_prob, 0.0)
    company_margin = best_company_prob - max(second_company_prob, 0.0)

    # Apply robust thresholds:
    #  - require that best column is strong enough (min prob)
    #  - and clearly better than the second-best for that type (margin)
    if best_phone_prob < PHONE_MIN_PROB or phone_margin < PHONE_MIN_MARGIN:
        best_phone_col = None

    if best_company_prob < COMPANY_MIN_PROB or company_margin < COMPANY_MIN_MARGIN:
        best_company_col = None

    return best_phone_col, best_company_col, results


def build_output_dataframe(
    df: pd.DataFrame,
    phone_col: Optional[str],
    company_col: Optional[str]
) -> pd.DataFrame:
    """
    Construct the output DataFrame according to spec.

    - If only phone_col is not None:
        columns: PhoneNumber, Country, Number
    - If only company_col is not None:
        columns: CompanyName, Name, Legal
    - If both present:
        columns: PhoneNumber, Country, Number, CompanyName, Name, Legal
    """

    n_rows = len(df.index)

    phone_series = pd.Series([""] * n_rows, index=df.index, dtype=object)
    country_series = pd.Series([""] * n_rows, index=df.index, dtype=object)
    number_series = pd.Series([""] * n_rows, index=df.index, dtype=object)

    company_series = pd.Series([""] * n_rows, index=df.index, dtype=object)
    name_series = pd.Series([""] * n_rows, index=df.index, dtype=object)
    legal_series = pd.Series([""] * n_rows, index=df.index, dtype=object)

    legal_terms = load_legal_suffixes()

    # Phone parsing
    if phone_col is not None:
        original_phone_col = df[phone_col].astype(str)
        phone_series = original_phone_col.copy()

        parsed_countries = []
        parsed_numbers = []

        for v in original_phone_col:
            ctry, num = parse_phone(v)
            parsed_countries.append(ctry)
            parsed_numbers.append(num)

        country_series = pd.Series(parsed_countries, index=df.index, dtype=object)
        number_series = pd.Series(parsed_numbers, index=df.index, dtype=object)

    # Company parsing
    if company_col is not None:
        original_company_col = df[company_col].astype(str)
        company_series = original_company_col.copy()

        parsed_names = []
        parsed_legals = []

        for v in original_company_col:
            name, legal = parse_company(v, legal_terms=legal_terms)
            parsed_names.append(name)
            parsed_legals.append(legal)

        name_series = pd.Series(parsed_names, index=df.index, dtype=object)
        legal_series = pd.Series(parsed_legals, index=df.index, dtype=object)

    if phone_col is not None and company_col is not None:
        out_df = pd.DataFrame({
            "PhoneNumber": phone_series,
            "Country": country_series,
            "Number": number_series,
            "CompanyName": company_series,
            "Name": name_series,
            "Legal": legal_series,
        })
    elif phone_col is not None:
        out_df = pd.DataFrame({
            "PhoneNumber": phone_series,
            "Country": country_series,
            "Number": number_series,
        })
    elif company_col is not None:
        out_df = pd.DataFrame({
            "CompanyName": company_series,
            "Name": name_series,
            "Legal": legal_series,
        })
    else:
        # Fallback: no relevant columns; return original df
        out_df = df.copy()

    return out_df


def _resolve_file_path(relative_or_abs: str) -> str:
    """
    Resolve a file path. If it's absolute and exists, use it.
    Otherwise, treat it as a filename inside BASE_DATA_DIR.
    """
    p = os.path.abspath(relative_or_abs)
    if os.path.exists(p):
        return p

    candidate = os.path.join(BASE_DATA_DIR, relative_or_abs)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    raise FileNotFoundError(f"File not found: {relative_or_abs}")


# -------------------------------------------------------------------
# MCP tools
# -------------------------------------------------------------------

@mcp.tool()
def list_files() -> List[str]:
    """
    List available CSV files that this MCP server can process.

    Returns a list of absolute paths (or relative names) to CSV files
    under the configured BASE_DATA_DIR.
    """
    if not os.path.isdir(BASE_DATA_DIR):
        return []

    files = []
    for name in os.listdir(BASE_DATA_DIR):
        if name.lower().endswith(".csv"):
            files.append(os.path.abspath(os.path.join(BASE_DATA_DIR, name)))

    return files


@mcp.tool()
def column_prediction(file_path: str, column_name: str) -> Dict[str, Any]:
    """
    Classify a single column in a file.

    Args:
        file_path: path to the CSV file (absolute or just filename in data/)
        column_name: name of the column to classify

    Returns:
        {
          "file": resolved_file_path,
          "column": column_name,
          "internal_label": "PHONE" | "COMPANY" | "COUNTRY" | "DATE" | "OTHER",
          "semantic_type": "PhoneNumber" | "CompanyName" | "Country" | "Date" | "Other",
          "probabilities": {label: prob, ...}
        }
    """
    resolved = _resolve_file_path(file_path)

    df = pd.read_csv(resolved)
    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' not found in file '{resolved}'. "
            f"Available columns: {list(df.columns)}"
        )

    model = load_model()
    internal_label, proba_dict = predict_column_type(df, column_name, model=model, sample_size=200)
    semantic_type = LABEL_TO_OUTPUT.get(internal_label, internal_label)

    return {
        "file": resolved,
        "column": column_name,
        "internal_label": internal_label,
        "semantic_type": semantic_type,
        "probabilities": proba_dict,
    }


@mcp.tool()
def parse_file(file_path: str) -> Dict[str, Any]:
    """
    Run end-to-end classification + parsing on a CSV file.

    Steps:
      - Load file.
      - Identify best Phone and Company columns.
      - Parse them into:
            PhoneNumber, Country, Number, CompanyName, Name, Legal
      - Write output.csv in the same directory as the input.

    Args:
        file_path: path to the CSV file (absolute or just filename in data/).

    Returns:
        {
          "input_file": resolved_input_path,
          "output_file": output_path,
          "phone_column": best_phone_col or null,
          "company_column": best_company_col or null,
        }
    """
    resolved = _resolve_file_path(file_path)
    df = pd.read_csv(resolved)

    phone_col, company_col, _results = find_best_columns(df)

    out_df = build_output_dataframe(df, phone_col, company_col)

    out_dir = os.path.dirname(resolved)
    output_path = os.path.join(out_dir, "output.csv")
    out_df.to_csv(output_path, index=False)

    return {
        "input_file": resolved,
        "output_file": output_path,
        "phone_column": phone_col,
        "company_column": company_col,
    }


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    # This will start the MCP server over stdio for compatible clients.
    mcp.run()
