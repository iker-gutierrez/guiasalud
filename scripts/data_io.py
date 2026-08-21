from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import pandas as pd


STANDARD_COLUMNS = [
    "id",
    "source",
    "split",
    "guidebook",
    "topic",
    "subtopic",
    "question",
    "focus",
    "judgement",
    "short_answer",
    "evidence",
    "considerations",
    "options",
    "correct_answer",
]


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Union[str, Path]) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def write_csv(records: Iterable[Mapping[str, Any]], path: Union[str, Path]) -> None:
    rows = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def canonical_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def column_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {canonical_name(column): column for column in columns}


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    lookup = column_lookup(df.columns)
    for candidate in candidates:
        key = canonical_name(candidate)
        if key in lookup:
            return lookup[key]
    if required:
        options = ", ".join(str(column) for column in df.columns)
        expected = ", ".join(candidates)
        raise ValueError(f"Could not find any of [{expected}] in columns: {options}")
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = clean_text(value)
            if value == "":
                continue
        if isinstance(value, (list, dict)) and not value:
            continue
        compacted[key] = value
    return compacted


def ensure_id(prefix: str, index: int, value: Any = None) -> str:
    candidate = clean_text(value)
    if candidate:
        return candidate
    return f"{prefix}_{index:05d}"


def records_to_dataframe(records: list[Mapping[str, Any]]) -> pd.DataFrame:
    ordered = []
    for record in records:
        row = {column: record.get(column, "") for column in STANDARD_COLUMNS if column in record}
        for key, value in record.items():
            if key not in row:
                row[key] = value
        ordered.append(row)
    return pd.DataFrame(ordered)
