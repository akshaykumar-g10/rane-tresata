# Semantic Column Classification & Parsing

This repository implements a complete pipeline for:

1. **Semantic column classification**  
   Classify cell/column values into one of:
   - `PHONE`
   - `COMPANY`
   - `COUNTRY`
   - `DATE`
   - `OTHER`

2. **Parsing & normalization**  
   Once columns are identified as:
   - **PhoneNumber** → parse into `Country`, `Number`
   - **CompanyName** → parse into `Name`, `Legal` suffix

3. **Tooling / integration support**  
   - Simple CLIs for training, testing, and parsing.
   - An MCP server (`mcp_server.py`) that exposes the model as a tool.

The focus is on **robust behavior on messy, real-world CSVs** rather than just raw model accuracy:
- Good **feature engineering**
- Careful **column-level aggregation**
- Conservative **parser behavior** to avoid over-confident mistakes

---

## Repository Structure

Recommended structure for this project:

```text
.
├─ data/
│  ├─ phoneNumber.csv        # Training values for PHONE
│  ├─ Company.csv            # Training values for COMPANY
│  ├─ Dates.csv              # Training values for DATE
│  ├─ Countries.txt          # Training values for COUNTRY (one per line)
│  ├─ legal.txt              # Legal suffixes (e.g. "pvt ltd", "gmbh co kg", "ag", ...)
│  ├─ phone.csv              # Example test file with phone columns
│  └─ company.csv            # Example test file with company columns
│
├─ models/
│  ├─ semantic_classifier.joblib       # Main Logistic Regression model (trained)
│  └─ semantic_classifier_svm.joblib   # Alternative Linear SVM model (trained)
│
├─ train_classifier.py        # Train the main LR-based classifier
├─ train_classifier_svm.py    # Train the alternative SVM-based classifier
├─ text_features.py           # Feature extraction: TF–IDF + text stats
├─ semantic_model.py          # Load & use LR model (cell + column level)
├─ semantic_model_svm.py      # Load & use SVM model (alternative)
├─ parser.py                  # Column selection + phone/company parsing
├─ sanity_check.py            # Quick sanity tests on a few hard-coded strings
├─ test_parsers.py            # Unit-like tests for phone/company parsers
├─ compare_models.py          # Compare LR vs SVM on real files
├─ test_improved_pipeline.py  # End-to-end tests (cell, column, file-level)
├─ mcp_server.py              # MCP server exposing the classifier as a tool
│
├─ requirements.txt           # Python dependencies (optional but recommended)
└─ README.md                  # This file
```

The structure is flexible as long as:
- Code entrypoints remain in the root
- Training data stays under data/
- Models are saved under models/

Setup
Python version

Tested with Python 3.11. Anything ≥ 3.9 should work with minor adjustments.

Install dependencies

Create a virtual environment (recommended) and install dependencies:

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt


If you don’t use requirements.txt, at minimum you need:

pip install pandas numpy scikit-learn joblib


(Optional) If you use the MCP server, also install the relevant SDK / framework as needed.

Data Layout

Place the following files under data/:

phoneNumber.csv

First column: phone number strings for training (e.g. "+91 9876543210", "044-12345678", etc.)

Company.csv

First column: company names (e.g. "Tresata Pvt Ltd", "First National Bank")

Dates.csv

First column: date strings in various formats (e.g. "2024-11-27", "01/01/2000")

Countries.txt

One country per line (e.g. India, United States, Germany, …)

legal.txt

One legal suffix per line, in lowercase, e.g.:

pvt ltd
private limited
gmbh co kg
ag
ltd
llc


Optional test files:

phone.csv – CSV with at least one column that contains phone numbers.

company.csv – CSV with at least one column that contains company names.

1. Training the Classifier (Logistic Regression)

The main model is a multinomial Logistic Regression over:

Character-level TF–IDF features (ngrams 2–4)

Hand-crafted text features from text_features.py, e.g.:

String length

Fraction of digits, letters, punctuation

Digit run lengths

Flags like looks_like_date, has_plus, has_country_name, etc.

Command
python train_classifier.py \
  --data_dir ./data \
  --model_path ./models/semantic_classifier.joblib

What it does

Reads training data from:

phoneNumber.csv, Company.csv, Dates.csv, Countries.txt.

Builds a labeled dataset with labels:

PHONE, COMPANY, COUNTRY, DATE, OTHER.

Shuffles and splits into train / validation.

Builds a scikit-learn Pipeline:

FeatureUnion(char-TFIDF, TextStatsExtractor) → LogisticRegression(multi_class="multinomial")

Trains the classifier and prints:

Classification report (precision/recall/F1 per class).

Confusion matrix.

Saves the model pipeline to:

models/semantic_classifier.joblib.

2. Alternative Model: Linear SVM (Optional)

For comparison, you can also train a Linear SVM with the same features.

Train SVM model
python train_classifier_svm.py \
  --data_dir ./data \
  --model_path ./models/semantic_classifier_svm.joblib


This script:

Reuses the same dataset construction (build_labeled_dataset from train_classifier.py).

Uses a LinearSVC classifier instead of Logistic Regression.

Prints a classification report and confusion matrix.

Saves the model to models/semantic_classifier_svm.joblib.

In practice, performance is similar; Logistic Regression is used as the main production model, while SVM is included as a documented experiment.

3. Semantic Model API

The main interface is in semantic_model.py.

Loading the model
from semantic_model import load_model

model = load_model()  # loads models/semantic_classifier.joblib by default

Predict a single cell
from semantic_model import predict_cell_type

label, probs = predict_cell_type("+91 9876543210", model=model)
# label -> "PHONE"
# probs -> {"PHONE": 0.999..., "DATE": ..., ...}

Predict a single column
import pandas as pd
from semantic_model import predict_column_type

df = pd.read_csv("data/phoneNumber.csv")
col_name = df.columns[0]

label, probs = predict_column_type(df, col_name, model=model)
# label -> "PHONE"
# probs -> class probability distribution for the whole column

Predict all columns in a DataFrame
from semantic_model import predict_all_columns

results = predict_all_columns(df, model=model)
# results: dict[col_name] = (label, probs)


Column-level logic is robust:

Uses confidence-weighted averaging of cell probabilities.

Uses a majority-vote signal over row-level predictions.

Applies thresholds and margins to reduce ambiguous decisions.

4. Parsing & Normalization (parser.py)

The parser combines:

Column-level classification (PHONE, COMPANY, …)

Heuristics to pick the best Phone column and best Company column in a file.

Row-level parsing / normalization for those columns.

CLI Usage
python parser.py --input data/phone.csv
# or
python parser.py --input data/company.csv
# or with a custom output path:
python parser.py --input data/phone.csv --output data/output_phone.csv


By default, the output is written to output.csv next to the input file, unless --output is specified.

Column selection

parser.py uses:

best_phone_col, best_company_col, results = find_best_columns(df)


Where:

best_phone_col is the name of the best column to treat as Phone numbers (or None if no strong candidate).

best_company_col is the best Company column (or None).

results contains per-column label + probabilities.

Selection is based on:

Mean probability for PHONE / COMPANY.

Margin over the second-best column for that type.

Minimum probability thresholds:

PHONE_MIN_PROB, COMPANY_MIN_PROB.

This allows a “no-phone-column” / “no-company-column” decision when appropriate.

Output schema

The parser always produces an output CSV with columns:

PhoneNumber – original string from the chosen phone column (or empty if none).

Country – inferred country name from international dialing code (if present).

Number – digits-only phone number (normalized).

CompanyName – original string from the chosen company column (or empty).

Name – base company name (normalized).

Legal – extracted legal suffix (normalized), e.g. pvt ltd, gmbh co kg.

If a phone/company column is not found, relevant output columns remain empty.

Phone parsing details

For each row in the chosen phone column:

Run predict_cell_type to get per-row label & probabilities.

If the row doesn’t look sufficiently like a phone (PHONE prob below threshold and not predicted as PHONE), parsing is skipped for that row.

Else, parse_phone_value:

Strips common extensions: ext, extension, x, #, etc.

Checks if value looks like a date (using _is_valid_date_like); if yes, skip.

If it starts with '+', tries to parse a country code using DIAL_CODE_TO_COUNTRY:

Tries 3, 2, then 1 digit (longest match first).

Maps to a country name (e.g. +91 → India).

Extracts digits-only main number.

Applies simple length sanity check (7–15 digits).

Returns (Country, Number).

Company parsing details

For each row in the chosen company column:

Normalize using normalize_text:

Lowercase

Remove punctuation

Collapse whitespace

Load legal suffix list from data/legal.txt (already normalized).

Match the longest legal suffix that appears at the end of the normalized string:

E.g. "tresata pvt ltd" → "tresata" + "pvt ltd"

E.g. "enno roggemann gmbh & co. kg" → "enno roggemann" + "gmbh co kg" (assuming legal.txt has gmbh co kg)

If no suffix matches:

Name = full normalized string

Legal = ""

5. Sanity & Test Scripts
sanity_check.py

Runs a few hard-coded examples through the main model and prints predictions + probabilities. Useful for quickly checking that the model has loaded correctly and basic behavior makes sense.

python sanity_check.py

test_parsers.py

Contains simple unit-like tests for the phone and company parsers:

Ensures example phone/company strings parse into the expected (Name, Legal) / (Country, Number) splits.

python test_parsers.py

test_improved_pipeline.py

End-to-end smoke tests:

Cell-level predictions on a small set of strings.

Column-level predictions on phoneNumber.csv and Company.csv.

File-level column selection on data/phone.csv and data/company.csv.

python test_improved_pipeline.py

compare_models.py

Loads both the LR and SVM models and compares their behavior on real CSVs (e.g., which column they choose as PHONE/COMPANY, probability distributions, etc.).

python compare_models.py

6. MCP Server (Optional)

mcp_server.py exposes the classifier as a tool (MCP-compatible server).
Conceptually it:

Loads the trained classifier via semantic_model.py.

Implements handlers to:

Predict semantic types for example values.

Predict types for columns in a given CSV.

Return structured results to an MCP-compatible client.

You can plug this into an MCP-capable environment to query the classifier interactively.

7. Design Choices & Future Work

You can highlight these points in your report/slides:

Why Logistic Regression?

Linear model, fast to train, robust with high-dimensional TF–IDF.

Well-calibrated probabilities, which matter for column-level aggregation.

Interpretable and simple to debug.

Why char-level TF–IDF?

Works well for short, noisy strings.

Captures prefixes/suffixes like +91, pvt, ltd, date formats, etc.

Language-agnostic and robust to typos.

Why add hand-crafted features?

Simple numeric signals are very informative:

Fraction of digits / letters / punctuation.

Presence of +, parentheses, etc.

Length ranges.

Date-like patterns.

Country-name matches from Countries.txt.

These complement TF–IDF and help the model separate tricky cases (e.g. numeric IDs vs phone numbers vs dates).

Column-level aggregation

Confidence-weighted averaging reduces the impact of noisy rows.

Majority vote over row-level labels gives another signal.

Thresholds and margins prevent ambiguous decisions from being forced into a type.

Explicit “no such column” decision avoids random false positives.

Parser robustness

Per-row gating: don’t parse rows that don’t look like phone/company.

Date-aware phone parsing avoids DOBs becoming phone numbers.

Longest-suffix logic gives clean Name / Legal splits.

All original values are preserved in PhoneNumber / CompanyName.

Potential future improvements

Expand DIAL_CODE_TO_COUNTRY map to more countries.

Smarter date parsing and normalization.

Additional feature types (word-level TF–IDF, pretrained embeddings).

Model ensembling (e.g. LR + SVM voting) if needed.

Better handling of highly mixed columns (e.g. free-text + structured).

8. How to Run Everything (Quick Summary)

Train main model

python train_classifier.py \
  --data_dir ./data \
  --model_path ./models/semantic_classifier.joblib


Optional: train SVM model

python train_classifier_svm.py \
  --data_dir ./data \
  --model_path ./models/semantic_classifier_svm.joblib


Basic sanity check

python sanity_check.py


End-to-end test

python test_improved_pipeline.py


Parse a new CSV

python parser.py \
  --input path/to/your_file.csv \
  --output path/to/output.csv


This README should be enough for someone to:

Understand the problem,

Reproduce your training,

Run the classifier,

Parse real CSVs,

And see the reasoning/design behind the choices you made.
