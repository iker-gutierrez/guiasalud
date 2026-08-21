# -*- coding: utf-8 -*-
"""datasetting_postcuration.py

Continues datasetting_precuration.py after the manual curation step
(scripts/manual_curation.py, its own §8): compares the pre-curation dataset
(dataset_precuration.csv) against the post-curation dataset (dataset.csv,
built by manual_curation.py from the curated train.csv/dev.csv/test.csv)
side by side, on the same two axes datasetting_precuration.py's own report
already covers pre-curation:

  - Feature completeness (mandatory/optional presence rates, incomplete
    samples): confirms manual curation actually fixed what it was supposed
    to, and quantifies what changed.
  - Descriptive statistics (sentence/token counts, per-field lengths,
    guidebook distribution): shows curation's effect in numbers, not just
    "0 incomplete samples now" -- e.g. a field that used to average 400
    tokens because of a runaway extraction bug should shrink back down to a
    realistic length once curated.

Both are computed against BOTH files and printed together, so this is a
before/after comparison, not just a final-state report -- the pre-curation
numbers alone aren't representative of the finished dataset (curation
hasn't happened yet), and the post-curation numbers alone don't show what
curation actually changed.

Run after scripts/manual_curation.py, which builds dataset.csv from the
curated train.csv/dev.csv/test.csv.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "interim" / "guiasalud"

REQUIRED_COLS = ["guidebook", "topic", "question", "judgement", "evidence"]
OPTIONAL_COLS = ["subtopic", "focus", "considerations"]
TEXT_COLS = ["question", "focus", "judgement", "evidence", "considerations"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing dataset_precuration.csv and dataset.csv.",
    )
    return parser.parse_args()


def _load(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"SKIPPED: {path.name} not found")
        return None
    return pd.read_csv(path)


def _count_sentences(text) -> int:
    if pd.isna(text):
        return 0
    sentences = re.split(r"[.!?]+", str(text))
    return len([s for s in sentences if s.strip()])


def _count_tokens(text) -> int:
    if pd.isna(text):
        return 0
    return len(re.findall(r"\b\w+\b", str(text)))


def _non_empty_mask(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].notna() & (df[col].astype(str).str.strip() != "")


def completeness_report(df: pd.DataFrame) -> dict:
    mandatory = {col: _non_empty_mask(df, col).mean() * 100 for col in REQUIRED_COLS}
    optional = {col: df[col].notna().mean() * 100 for col in OPTIONAL_COLS}
    valid_mask = pd.concat([_non_empty_mask(df, col) for col in REQUIRED_COLS], axis=1).all(axis=1)
    incomplete_ids = df.loc[~valid_mask, "id"].tolist()
    return {"mandatory": mandatory, "optional": optional, "incomplete_ids": incomplete_ids}


def quantitative_report(df: pd.DataFrame) -> dict:
    sentences_per_row = df[TEXT_COLS].fillna("").apply(
        lambda row: sum(_count_sentences(x) for x in row), axis=1
    )
    tokens_per_row = df[TEXT_COLS].fillna("").apply(
        lambda row: sum(_count_tokens(x) for x in row), axis=1
    )
    lengths = {}
    for col in TEXT_COLS:
        mask = _non_empty_mask(df, col)
        token_lengths = df[col].astype(str).str.split().str.len()
        lengths[col] = {"avg": token_lengths[mask].mean(), "std": token_lengths[mask].std()}
    return {
        "sentences": int(sentences_per_row.sum()),
        "tokens": int(tokens_per_row.sum()),
        "lengths": lengths,
        "per_guidebook": df["guidebook"].value_counts(),
    }


def print_completeness_comparison(before: dict, after: dict) -> None:
    print("Completeness (before curation -> after curation):")
    print("- Mandatory features:")
    for col in REQUIRED_COLS:
        print(f"  - {col}: {before['mandatory'][col]:.2f}% -> {after['mandatory'][col]:.2f}%")
    print("- Optional features:")
    for col in OPTIONAL_COLS:
        print(f"  - {col}: {before['optional'][col]:.2f}% -> {after['optional'][col]:.2f}%")
    print(
        "- Incomplete samples:",
        len(before["incomplete_ids"]), "->", len(after["incomplete_ids"]),
    )
    if after["incomplete_ids"]:
        print("- IDs of incomplete samples (after curation):")
        print(after["incomplete_ids"])


def print_quantitative_comparison(before: dict, after: dict) -> None:
    print("Quantitative statistics (before curation -> after curation):")
    print(f"- Sentences: {before['sentences']} -> {after['sentences']}")
    print(f"- Tokens: {before['tokens']} -> {after['tokens']}")
    print()
    for col in TEXT_COLS:
        b, a = before["lengths"][col], after["lengths"][col]
        print(f"- Avg {col} length (tokens): {b['avg']:.2f} -> {a['avg']:.2f}")
        print(f"- Std {col} length (tokens): {b['std']:.2f} -> {a['std']:.2f}")
    print()
    print("Samples per guidebook (before curation vs. after curation):")
    comparison = pd.DataFrame(
        {"before": before["per_guidebook"], "after": after["per_guidebook"]}
    ).fillna(0).astype(int)
    print(comparison.to_string())


def main() -> None:
    global DATA_DIR
    DATA_DIR = parse_args().data_dir.resolve()
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {DATA_DIR}")
    before_df = _load(DATA_DIR / "dataset_precuration.csv")
    after_df = _load(DATA_DIR / "dataset.csv")
    if before_df is None or after_df is None:
        return

    print("-----")
    print()
    print_completeness_comparison(completeness_report(before_df), completeness_report(after_df))
    print()
    print("-----")
    print()
    print_quantitative_comparison(quantitative_report(before_df), quantitative_report(after_df))
    print()


if __name__ == "__main__":
    main()
