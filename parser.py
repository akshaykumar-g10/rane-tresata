#!/usr/bin/env python3
"""
parser.py

Part B: Parsing & Normalization

Once a column is classified as:
  - PhoneNumber  -> parse into: Country, Number
  - CompanyName  -> parse into: Name, Legal

Given an input CSV, this script:
  1. Uses the semantic classifier to pick the best Phone & Company columns.
  2. Parses/normalizes those columns row by row.
  3. Writes an output CSV with columns:
       PhoneNumber, Country, Number, CompanyName, Name, Legal

Usage:
    python parser.py --input /path/to/file.csv
    # optional:
    # python parser.py --input /path/to/file.csv --output /path/to/output.csv
"""

import argparse
import os
import re
from typing import Tuple, List, Optional

import pandas as pd

from semantic_model import load_model, predict_all_columns, predict_cell_type
from text_features import normalize_text, _is_valid_date_like  # reused heuristics

# ----------------------------------------------------------------------
# Column-selection thresholds
# ----------------------------------------------------------------------

PHONE_MIN_PROB = 0.60          # min avg PHONE prob for best column
COMPANY_MIN_PROB = 0.55        # min avg COMPANY prob for best column
PHONE_MIN_MARGIN = 0.05        # min gap vs 2nd-best PHONE column
COMPANY_MIN_MARGIN = 0.05      # min gap vs 2nd-best COMPANY column

# Cell-level threshold: how sure we must be to parse a value as PHONE
CELL_PHONE_MIN_PROB = 0.50

# ----------------------------------------------------------------------
# Phone country-code map (extend as needed)
# Codes are without '+' sign
# ----------------------------------------------------------------------

DIAL_CODE_TO_COUNTRY = {
    "91": "India",
    "1": "US",
    "44": "UK",
    # You can add more if needed, e.g.:
    "49": "Germany",
    "33": "France",
    "39": "Italy",
    "81": "Japan",
    "61": "Australia",
}


# ----------------------------------------------------------------------
# Column selection logic
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Phone parsing helpers
# ----------------------------------------------------------------------

def _strip_extension(text: str) -> str:
    """
    Remove common extension markers (ext, extension, x, #) and anything after them.
    """
    if not text:
        return text
    # Split on common extension markers (case-insensitive)
    parts = re.split(r'\b(ext|extension|x|#)\b', text, maxsplit=1, flags=re.IGNORECASE)
    return parts[0]


def _looks_like_valid_phone_digits(digits: str) -> bool:
    """
    Simple length-based sanity check for phone numbers.
    """
    n = len(digits)
    return 7 <= n <= 15  # common international range


def parse_phone_value(raw: str) -> Tuple[str, str]:
    """
    Parse a raw phone string into (Country, Number).

    Rules:
      - If looks like a valid date (e.g. 2024-11-27, 01/01/2000), skip: return ("", "").
      - If starts with '+<country_code>', map code to Country using DIAL_CODE_TO_COUNTRY.
      - Number is always returned as digits only.
      - If no country code is found, Country = "" and Number = digits.
    """
    if raw is None:
        return "", ""

    text = str(raw).strip()
    if not text:
        return "", ""

    # Date guard: if this looks like a real date, do NOT treat as phone
    try:
        if _is_valid_date_like(text) >= 0.5:
            return "", ""
    except Exception:
        # If anything goes wrong, just fall back to normal phone logic
        pass

    # Strip common extensions
    text_main = _strip_extension(text)

    # If starts with '+', try to parse country code
    country = ""
    number_digits = ""

    if text_main.startswith("+"):
        # Extract digits after '+'
        m = re.match(r"\+(\d+)", text_main)
        if m:
            all_digits = m.group(1)
            # Try codes of length 3,2,1 (longest first)
            code = ""
            for length in (3, 2, 1):
                if len(all_digits) >= length:
                    candidate = all_digits[:length]
                    if candidate in DIAL_CODE_TO_COUNTRY:
                        code = candidate
                        break

            if code:
                country = DIAL_CODE_TO_COUNTRY[code]
                # Rest digits after the country code
                remaining_digits = all_digits[len(code):]
                # Also capture any digits after the first digit sequence
                tail_digits = re.sub(r"\D", "", text_main[m.end():])
                number_digits = remaining_digits + tail_digits
            else:
                # No known code: treat all digits as local number
                number_digits = all_digits + re.sub(r"\D", "", text_main[m.end():])
        else:
            # No digits after '+', fallback to non-plus handling
            number_digits = re.sub(r"\D", "", text_main)
    else:
        # No '+': just take all digits
        number_digits = re.sub(r"\D", "", text_main)

    if not number_digits:
        return "", ""

    # Length sanity: if absurdly short/long, mark as no parse
    if not _looks_like_valid_phone_digits(number_digits):
        # For assignment, we keep Number as digits even if length is odd,
        # but don't try to infer Country.
        return "", number_digits

    return country, number_digits


def parse_phone_column(
    df: pd.DataFrame,
    phone_col: str,
    model,
    cell_phone_min_prob: float = CELL_PHONE_MIN_PROB
):
    """
    Parse a phone column row-by-row, using cell-level predictions
    to avoid parsing obvious non-phone entries (e.g., DOBs).
    """
    phone_raw = df[phone_col].astype(str)

    phone_out: List[str] = []
    country_out: List[str] = []
    number_out: List[str] = []

    for raw in phone_raw:
        raw_str = str(raw)

        # Cell-level classification
        label, probs = predict_cell_type(raw_str, model=model)
        phone_prob = float(probs.get("PHONE", 0.0))

        # If not confident that this is a phone, skip parsing
        if label != "PHONE" and phone_prob < cell_phone_min_prob:
            phone_out.append(raw_str)
            country_out.append("")
            number_out.append("")
            continue

        country, number = parse_phone_value(raw_str)
        phone_out.append(raw_str)
        country_out.append(country)
        number_out.append(number)

    return phone_out, country_out, number_out


# ----------------------------------------------------------------------
# Company parsing helpers
# ----------------------------------------------------------------------

def load_legal_terms(data_dir: str) -> List[str]:
    """
    Load legal suffixes from legal.txt in the given data directory.
    Each line should contain one suffix, e.g.:
      pvt ltd
      gmbh co kg
      ag
    """
    legal_path = os.path.join(data_dir, "legal.txt")
    if not os.path.exists(legal_path):
        return []

    terms = []
    with open(legal_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().lower()
            if t:
                terms.append(t)

    # Sort by descending length so multi-word/longer suffixes match first
    terms = sorted(terms, key=len, reverse=True)
    return terms


def parse_company_value(raw: str, legal_terms: List[str]) -> Tuple[str, str]:
    """
    Parse a company name into (Name, Legal).

    - Normalize using the same normalize_text as in training.
    - Match the longest legal suffix from legal_terms at the end.
    - Example:
        "Tresata pvt ltd"            -> ("tresata", "pvt ltd")
        "Enno Roggemann GmbH & Co. KG" -> ("enno roggemann", "gmbh co kg")
        "First National Bank"        -> ("first national bank", "")
    """
    if raw is None:
        return "", ""

    raw_str = str(raw)
    if not raw_str.strip():
        return "", ""

    # Use same normalization as training (lowercase, strip punctuation, collapse spaces)
    norm = normalize_text(raw_str)

    if not norm:
        return "", ""

    # Try to match the longest legal suffix at the end
    for suffix in legal_terms:
        # exact match
        if norm == suffix:
            return "", suffix

        # suffix match with a space boundary
        if norm.endswith(" " + suffix):
            name_part = norm[: -len(suffix)].rstrip()
            return name_part, suffix

    # No legal suffix found
    return norm, ""


def parse_company_column(
    df: pd.DataFrame,
    company_col: str,
    data_dir: str
):
    """
    Parse a company column row-by-row into (CompanyName, Name, Legal).
    """
    legal_terms = load_legal_terms(data_dir)
    company_raw = df[company_col].astype(str)

    company_out: List[str] = []
    name_out: List[str] = []
    legal_out: List[str] = []

    for raw in company_raw:
        raw_str = str(raw)
        name, legal = parse_company_value(raw_str, legal_terms)
        company_out.append(raw_str)
        name_out.append(name)
        legal_out.append(legal)

    return company_out, name_out, legal_out


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse PhoneNumber / CompanyName columns")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: input_dir/output.csv)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing legal.txt (for company suffixes)",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    data_dir = args.data_dir

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        out_dir = os.path.dirname(os.path.abspath(input_path))
        output_path = os.path.join(out_dir, "output.csv")

    print(f"[INFO] Loading input file: {input_path}")
    df = pd.read_csv(input_path)

    best_phone_col, best_company_col, results = find_best_columns(df)

    print(f"[INFO] Column predictions:")
    for col, (label, probs) in results.items():
        print(f"  - {col}: {label} {probs}")

    print(f"[INFO] Best phone column: {best_phone_col}")
    print(f"[INFO] Best company column: {best_company_col}")

    model = load_model()

    # Prepare output columns (even if some are None, we include them with blanks)
    phone_number_col: List[str] = []
    country_col: List[str] = []
    number_col: List[str] = []

    company_name_col: List[str] = []
    name_col: List[str] = []
    legal_col: List[str] = []

    n_rows = len(df)

    # Initialize all as empty
    phone_number_col = ["" for _ in range(n_rows)]
    country_col = ["" for _ in range(n_rows)]
    number_col = ["" for _ in range(n_rows)]

    company_name_col = ["" for _ in range(n_rows)]
    name_col = ["" for _ in range(n_rows)]
    legal_col = ["" for _ in range(n_rows)]

    # If we detected a phone column, parse it
    if best_phone_col is not None:
        ph_raw, c_raw, num_raw = parse_phone_column(df, best_phone_col, model=model)
        phone_number_col = ph_raw
        country_col = c_raw
        number_col = num_raw

    # If we detected a company column, parse it
    if best_company_col is not None:
        comp_raw, name_raw, legal_raw = parse_company_column(df, best_company_col, data_dir=data_dir)
        company_name_col = comp_raw
        name_col = name_raw
        legal_col = legal_raw

    out_df = pd.DataFrame({
        "PhoneNumber": phone_number_col,
        "Country": country_col,
        "Number": number_col,
        "CompanyName": company_name_col,
        "Name": name_col,
        "Legal": legal_col,
    })

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(output_path, index=False)
    print(f"[INFO] Wrote output to: {output_path}")


if __name__ == "__main__":
    main()
