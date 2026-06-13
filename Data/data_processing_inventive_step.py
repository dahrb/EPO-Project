"""
Data Processing Pipeline — Inventive Step Subset

Takes the full EPO_Data.pkl (or Train&TestData_1.0_PatentRefusal.pkl) and
produces a balanced Inventive-Step-only dataset for experiments, plus a
small human-validation sample.

Pipeline:
    1. Load Patent Refusal data from Train&TestData_1.0_PatentRefusal.pkl
    2. Filter to cases whose Matched Articles == ['Inventive Step'] only
    3. Remove auxiliary-request keyword cases (regex filter on Keywords)
    4. Convert Reference format  (e.g. "T112564EU1" → "T256411")
    5. Balanced stratified sampling by Outcome × Year
    6. Draw a small validation sample (excluded from training data)
    7. Save outputs

Outputs (saved to ../Data/):
    - Inv_Step_Test.pkl          : balanced Inventive Step dataset
    - Inv_Step_Sampled_Valid.pkl : small sample for manual validation
    - Data_Inv_Step/             : (directory for any per-year splits)

Last Updated: 02.04.26

Status: Done
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Data")

INPUT_PKL = os.path.join(DATA_DIR, "Train&TestData_1.0_PatentRefusal.pkl")

RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & filter to Inventive Step
# ═══════════════════════════════════════════════════════════════════════════════

def load_patent_refusal(path: str) -> pd.DataFrame:
    """Load the Patent Refusal dataset."""
    print(f"[1/6] Loading {os.path.basename(path)} …")
    df = pd.read_pickle(path)
    print(f"       {len(df):,} rows loaded")
    return df


def filter_inventive_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only cases where Matched Articles == ['Inventive Step']
    (exactly, not combined with Novelty etc.).
    """
    print("[2/6] Filtering to Inventive Step only …")
    mask = df["Matched Articles"].apply(lambda x: x == ["Inventive Step"])
    inv = df[mask].copy()
    print(f"       {len(inv):,} Inventive Step cases")
    print(f"       Outcome distribution:")
    for k, v in inv["Outcome"].value_counts().items():
        print(f"         {k:12s}  {v:>5,}")
    return inv


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Remove auxiliary-request cases via Keywords
# ═══════════════════════════════════════════════════════════════════════════════

# Regex to detect auxiliary request keywords — matches patterns like:
#   "Inventive step - (no)", "Inventive Step - main request",
#   "Inventive step - auxiliary request", etc.
AUXILIARY_RE = re.compile(
    r"Inventive step - \((no|yes)\)"
    r"|Inventive Step - main request[\W_]*"
    r"|Inventive step - main request[\W_]*"
    r"|Inventive Step - auxiliary request[\W_]*"
    r"|Inventive step - auxiliary request[\W_]*"
    r"|Inventive [Ss]tep.*auxiliary"
    r"|Inventive [Ss]tep.*main request",
    re.IGNORECASE,
)


def _keywords_to_str(kw) -> str:
    """Convert Keywords (list or str or NaN) to a string for regex matching."""
    if isinstance(kw, list):
        return " | ".join(str(k) for k in kw)
    if pd.isna(kw):
        return ""
    return str(kw)


def remove_auxiliary_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cases whose Keywords mention auxiliary/main request patterns.
    These often represent multi-issue cases that are not purely Inventive Step.
    """
    print("[3/6] Removing auxiliary-request keyword cases …")
    before = len(df)
    kw_str = df["Keywords"].apply(_keywords_to_str)
    mask = kw_str.apply(lambda x: bool(AUXILIARY_RE.search(x)))
    df = df[~mask].copy()
    removed = before - len(df)
    print(f"       Removed {removed:,} auxiliary cases → {len(df):,} remaining")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Reference conversion
# ═══════════════════════════════════════════════════════════════════════════════

_REF_RE = re.compile(r"^([A-Z])(\d{2})(\d+)(EU|EP)[12]$")


def convert_reference(ref: str) -> str:
    """
    Convert raw EPO reference to compact format.

    Examples:
        "T112564EU1"  →  "T256411"
        "T000019EU1"  →  "T001900"   (keeps leading zeros in case number)

    Format: <letter><case_number><year_digits>
    The raw format encodes year as digits 2-3 after the letter, and the
    case number as the remaining digits before the EU/EP suffix.
    """
    m = _REF_RE.match(str(ref))
    if m:
        letter, year, case_num, _ = m.groups()
        return f"{letter}{case_num}{year}"
    return str(ref)


def convert_references(df: pd.DataFrame) -> pd.DataFrame:
    """Apply reference conversion to the Reference column."""
    print("[4/6] Converting Reference format …")
    df["Reference"] = df["Reference"].apply(convert_reference)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Balanced stratified sampling
# ═══════════════════════════════════════════════════════════════════════════════

def balanced_stratified_sample(
    df: pd.DataFrame,
    outcome_col: str = "Outcome",
    year_col: str = "Year",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Create a balanced dataset where each (Year, Outcome) stratum has the
    same number of samples — i.e. for each year, sample min(affirmed, reversed)
    from each class.
    """
    print("[5/6] Balanced stratified sampling by Outcome × Year …")
    balanced_parts = []

    for year in sorted(df[year_col].dropna().unique()):
        year_df = df[df[year_col] == year]
        affirmed = year_df[year_df[outcome_col] == "Affirmed"]
        reversed_ = year_df[year_df[outcome_col] == "Reversed"]
        n = min(len(affirmed), len(reversed_))
        if n > 0:
            balanced_parts.append(affirmed.sample(n=n, random_state=random_state))
            balanced_parts.append(reversed_.sample(n=n, random_state=random_state))

    balanced = pd.concat(balanced_parts, ignore_index=False)
    print(f"       Balanced dataset: {len(balanced):,} rows")
    print(f"       Outcome distribution:")
    for k, v in balanced[outcome_col].value_counts().items():
        print(f"         {k:12s}  {v:>5,}")
    print(f"       Year range: {balanced[year_col].min()} – {balanced[year_col].max()}")
    return balanced


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Validation sample
# ═══════════════════════════════════════════════════════════════════════════════

def draw_validation_sample(
    df: pd.DataFrame,
    n: int = 20,
    outcome_col: str = "Outcome",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Draw a small stratified sample for human validation (50/50 by Outcome).
    """
    print(f"[6/6] Drawing validation sample (n={n}) …")
    affirmed = df[df[outcome_col] == "Affirmed"]
    reversed_ = df[df[outcome_col] == "Reversed"]
    half = n // 2
    sample = pd.concat([
        affirmed.sample(n=half, random_state=random_state),
        reversed_.sample(n=half, random_state=random_state),
    ], ignore_index=False)
    print(f"       Validation sample: {len(sample):,} rows")
    return sample


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Select output columns & save
# ═══════════════════════════════════════════════════════════════════════════════

# Columns to keep in the Inventive Step output (matching existing format)
OUTPUT_COLS = [
    "Reference", "Court Type", "Summary Facts", "Decision Reasons",
    "Order", "Legal Provisions", "Date", "Invention Title",
    "Board Code", "Classification", "Keywords", "Year",
    "Matched Articles", "Outcome",
]


def save_outputs(
    balanced: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Save the balanced dataset and validation sample."""
    print()
    print("Saving outputs …")

    # Ensure output directory exists
    inv_step_dir = os.path.join(DATA_DIR, "Data_Inv_Step")
    os.makedirs(inv_step_dir, exist_ok=True)

    # Select output columns (only those present)
    out_cols = [c for c in OUTPUT_COLS if c in balanced.columns]

    # Save balanced dataset
    test_path = os.path.join(DATA_DIR, "Inv_Step_Test.pkl")
    balanced[out_cols].to_pickle(test_path)
    print(f"  Saved Inv_Step_Test.pkl          ({len(balanced):,} rows)")

    # Save validation sample
    valid_path = os.path.join(DATA_DIR, "Inv_Step_Sampled_Valid.pkl")
    validation[out_cols].to_pickle(valid_path)
    print(f"  Saved Inv_Step_Sampled_Valid.pkl  ({len(validation):,} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(" Inventive Step Data Processing Pipeline")
    print("=" * 70)

    # 1. Load
    df = load_patent_refusal(INPUT_PKL)

    # 2. Filter to Inventive Step
    inv = filter_inventive_step(df)

    # 3. Remove auxiliary keyword cases
    inv = remove_auxiliary_cases(inv)

    # 4. Convert references
    inv = convert_references(inv)

    # 5. Balanced stratified sampling
    balanced = balanced_stratified_sample(inv)

    # 6. Draw validation sample (from balanced set)
    validation = draw_validation_sample(balanced, n=20)

    # 7. Save
    save_outputs(balanced, validation)

    print()
    print("✓ Inventive Step pipeline complete.")


if __name__ == "__main__":
    main()
