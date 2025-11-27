#!/usr/bin/env python3
"""
text_features.py

Shared text utilities and custom feature extractor used in the
semantic type classifier.
"""

import re
from typing import Iterable, List, Optional, Set

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def normalize_text(x: str) -> str:
    """Basic normalization: ensure string + strip whitespace."""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


def _extract_digits(s: str) -> str:
    """Return only digit characters from s."""
    return re.sub(r"\D", "", s)


def _is_valid_date_like(s: str) -> float:
    """
    Heuristic date detection:
    Returns 1.0 if the string matches a common date pattern where
    day/month/year ranges are plausible, else 0.0.

    Patterns checked:
      - YYYY-MM-DD or YYYY/MM/DD
      - DD-MM-YYYY or DD/MM/YYYY
      - DD-MM-YY or DD/MM/YY
    """
    if not s:
        return 0.0

    s = s.strip()

    # Normalize separators to '-' for easier handling
    s_norm = s.replace("/", "-")

    # Pattern 1: YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s_norm)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return 1.0

    # Pattern 2: DD-MM-YYYY or DD-MM-YY
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{2,4})", s_norm)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year = int(m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            # basic plausibility on year (00-99, 1900-2099)
            if (0 <= year <= 99) or (1900 <= year <= 2099):
                return 1.0

    return 0.0


def _is_phone_length_like(num_digits: int) -> float:
    """
    Simple phone-length heuristic:
    Returns 1.0 if number of digits is in a typical phone range (8–15),
    else 0.0.
    """
    return 1.0 if 8 <= num_digits <= 15 else 0.0


class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """
    Extract numeric / rule-based features from strings.

    Parameters:
        country_set: set of lowercased country names
        legal_terms: list of lowercased legal suffix phrases
    """

    def __init__(
        self,
        country_set: Optional[Set[str]] = None,
        legal_terms: Optional[List[str]] = None
    ):
        self.country_set = set(country_set) if country_set is not None else set()
        self.legal_terms = list(legal_terms) if legal_terms is not None else []

        # month names for simple date detection
        self.month_names = [
            "jan", "january",
            "feb", "february",
            "mar", "march",
            "apr", "april",
            "may",
            "jun", "june",
            "jul", "july",
            "aug", "august",
            "sep", "sept", "september",
            "oct", "october",
            "nov", "november",
            "dec", "december",
        ]

    def fit(self, X, y=None):
        # Transformer is stateless
        return self

    def transform(self, X: Iterable[str]):
        features = []

        for text in X:
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            lower = text.lower()

            length = len(text)
            num_digits = sum(ch.isdigit() for ch in text)
            num_alpha = sum(ch.isalpha() for ch in text)
            num_space = sum(ch.isspace() for ch in text)
            num_plus = text.count('+')
            num_minus = text.count('-')
            num_lparen = text.count('(')
            num_rparen = text.count(')')
            num_slash = text.count('/')
            num_dot = text.count('.')
            num_comma = text.count(',')

            digit_ratio = num_digits / length if length > 0 else 0.0
            alpha_ratio = num_alpha / length if length > 0 else 0.0
            space_ratio = num_space / length if length > 0 else 0.0

            has_plus_prefix = 1.0 if text.startswith('+') else 0.0
            has_parentheses = 1.0 if ('(' in text or ')' in text) else 0.0
            mostly_digits = 1.0 if digit_ratio > 0.8 and length > 0 else 0.0

            # Year-like pattern
            contains_year_4digits = (
                1.0 if re.search(r"\b(19|20)\d{2}\b", text) else 0.0
            )

            # Generic date separators
            has_date_sep = 1.0 if any(sep in text for sep in ['/', '-', ':']) else 0.0

            # Month-name presence
            lower_no_punct = re.sub(r"[^\w\s]", " ", lower)
            contains_month_name = 0.0
            for m in self.month_names:
                if re.search(r"\b" + re.escape(m) + r"\b", lower_no_punct):
                    contains_month_name = 1.0
                    break

            is_all_caps = 1.0 if (length > 0 and text.isupper()) else 0.0
            is_title_case = 1.0 if (length > 0 and text.istitle()) else 0.0

            tokens = [t for t in re.split(r"\s+", text) if t]
            num_tokens = len(tokens)
            if num_tokens > 0:
                token_lengths = [len(t) for t in tokens]
                avg_token_len = float(sum(token_lengths)) / num_tokens
            else:
                avg_token_len = 0.0

            numeric_tokens = [t for t in tokens if t.isdigit()]
            alpha_tokens = [t for t in tokens if t.isalpha()]

            frac_tokens_numeric = len(numeric_tokens) / num_tokens if num_tokens > 0 else 0.0
            frac_tokens_alpha = len(alpha_tokens) / num_tokens if num_tokens > 0 else 0.0

            # Country dictionary features
            stripped_lower = lower.strip()
            is_exact_country = 1.0 if stripped_lower in self.country_set else 0.0

            # Legal suffix features
            legal_suffix_count = 0.0
            ends_with_legal_suffix = 0.0
            for term in self.legal_terms:
                if not term:
                    continue
                if term in stripped_lower:
                    legal_suffix_count += 1.0
                if stripped_lower.endswith(term) or stripped_lower.endswith(" " + term):
                    ends_with_legal_suffix = 1.0

            # --- NEW: explicit phone/date helpers ---

            # 1) Number of digits as separate feature
            total_digits = float(num_digits)

            # 2) Is digit length in typical phone range?
            phone_length_like = _is_phone_length_like(num_digits)

            # 3) Is string a "valid date-like" pattern?
            is_valid_date_pattern = _is_valid_date_like(text)

            # 4) Digits-only string (for cases like "20241127" vs "9876543210")
            digits_only = _extract_digits(text)
            digits_only_len = float(len(digits_only))

            feats = [
                # basic length & ratios
                length,
                digit_ratio,
                alpha_ratio,
                space_ratio,

                # raw counts of punctuation-like chars
                num_plus,
                num_minus,
                num_lparen,
                num_rparen,
                num_slash,
                num_dot,
                num_comma,

                # structural flags
                has_plus_prefix,
                has_parentheses,
                mostly_digits,
                contains_year_4digits,
                has_date_sep,
                contains_month_name,
                is_all_caps,
                is_title_case,

                # token stats
                num_tokens,
                avg_token_len,
                frac_tokens_numeric,
                frac_tokens_alpha,

                # dictionary-based
                is_exact_country,
                legal_suffix_count,
                ends_with_legal_suffix,

                # NEW: numeric helpers
                total_digits,
                phone_length_like,
                is_valid_date_pattern,
                digits_only_len,
            ]
            features.append(feats)

        return np.array(features, dtype=float)
