#!/usr/bin/env python
"""Converts this repository's curated train/dev/test.csv into the JSONL
layout published to the Hugging Face dataset (es/{train,dev,test}.jsonl):
adds the composite `query`/`justification` fields, a `source` label, and
copies `judgement` into `short_answer`.

This does not re-derive a train/dev/test split: the split is the one
already fixed by manual_curation.py in train.csv/dev.csv/test.csv, read
here as given.

Adapted from med_rag_thesis's scripts/prepare_sns1064.py (the downstream
thesis repo this dataset was built for; despite the filename, that script
was GuiaSalud's own CSV-to-JSONL converter), pointed at this repo's own
layout and dependencies (scripts/data_io.py, scripts/translation/
question_format.py) instead of the thesis repo's src/medical_rag_thesis
package.

Usage:
  python scripts/prepare_jsonl.py \\
      --train-df train.csv \\
      --dev-df   dev.csv \\
      --test-df  test.csv \\
      --output-dir data/processed/es
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "translation"))

from data_io import (  # noqa: E402
    clean_text,
    compact_record,
    ensure_id,
    find_column,
    read_table,
    records_to_dataframe,
    write_jsonl,
)
from question_format import format_question  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert this repository's curated train/dev/test CSVs into es/{split}.jsonl."
    )
    parser.add_argument("--train-df", required=True, help="train.csv")
    parser.add_argument("--dev-df", required=True, help="dev.csv")
    parser.add_argument("--test-df", required=True, help="test.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for all/train/dev/test files (data/processed/es).")
    parser.add_argument("--source", default="GuiaSalud")
    return parser.parse_args()


def build_justification(evidence: str, considerations: str) -> str:
    """Compose the `justification` field the same way format_question
    composes `query`: one dash-bulleted labeled line per non-empty part,
    skipping any blank field entirely (considerations is empty on roughly
    a fifth of GuiaSalud records, so this must not emit a bare
    "- Consideraciones adicionales:" line for those)."""
    lines = []
    if evidence:
        lines.append(f"- Evidencia procedente de la investigación: {evidence}")
    if considerations:
        lines.append(f"- Consideraciones adicionales: {considerations}")
    return "\n".join(lines)


def normalize_split(input_path: str, source: str, split: str, id_offset: int) -> list[dict[str, str]]:
    """Normalize one pre-split CSV (train.csv/dev.csv/test.csv) into the
    published record schema: id, source, guidebook, topic, subtopic,
    question, focus, judgement, short_answer, evidence, considerations,
    query, justification, split."""
    df = read_table(input_path)
    id_col = find_column(df, ["id", "sample_id", "question_id"], required=False)
    guidebook_col = find_column(df, ["guidebook", "guide_book", "guia", "guía"], required=False)
    topic_col = find_column(df, ["Topic", "topic", "tema"], required=False)
    subtopic_col = find_column(df, ["subtopic", "sub_topic", "subtema"], required=False)
    question_col = find_column(df, ["Question", "question", "pregunta"])
    focus_col = find_column(
        df,
        ["focus", "foco", "subquestion", "sub_question", "subpregunta", "sub pregunta"],
        required=False,
    )
    judgement_col = find_column(
        df,
        ["judgement", "juicio", "Short answer", "short_answer", "answer", "respuesta corta", "respuesta"],
        required=False,
    )
    evidence_col = find_column(df, ["evidence"], required=False)
    considerations_col = find_column(df, ["considerations"], required=False)
    split_col = find_column(df, ["split"], required=False)

    records = []
    for index, row in df.iterrows():
        question = clean_text(row[question_col])
        subtopic = clean_text(row[subtopic_col]) if subtopic_col else ""
        focus = clean_text(row[focus_col]) if focus_col else ""
        judgement = clean_text(row[judgement_col]) if judgement_col else ""
        evidence = clean_text(row[evidence_col]) if evidence_col else ""
        considerations = clean_text(row[considerations_col]) if considerations_col else ""
        row_split = clean_text(row[split_col]) if split_col else split
        topic = clean_text(row[topic_col]) if topic_col else ""
        query = format_question(
            {"topic": topic, "subtopic": subtopic, "question": question, "focus": focus}
        )
        justification = build_justification(evidence, considerations)
        record = compact_record(
            {
                "id": ensure_id("guiasalud", id_offset + int(index), row[id_col] if id_col else None),
                "source": source,
                "guidebook": clean_text(row[guidebook_col]) if guidebook_col else "",
                "topic": topic,
                "subtopic": subtopic,
                "question": question,
                "focus": focus,
                "judgement": judgement,
                "short_answer": judgement,
                "evidence": evidence,
                "considerations": considerations,
                "query": query,
                "justification": justification,
                "split": row_split or split,
            }
        )
        if record.get("question") and record.get("judgement"):
            records.append(record)
    if not records:
        raise ValueError(f"No usable GuiaSalud rows found in {input_path}.")
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = normalize_split(args.train_df, args.source, "train", id_offset=0)
    dev_records = normalize_split(args.dev_df, args.source, "dev", id_offset=len(train_records))
    test_records = normalize_split(
        args.test_df, args.source, "test", id_offset=len(train_records) + len(dev_records)
    )
    records = [*train_records, *dev_records, *test_records]

    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids across train/dev/test after normalization.")

    write_jsonl(records, output_dir / "all.jsonl")
    records_to_dataframe(records).to_csv(output_dir / "all.csv", index=False)

    summary = {
        "inputs": {"train_df": args.train_df, "dev_df": args.dev_df, "test_df": args.test_df},
        "num_records": len(records),
        "splits": dict(Counter(record["split"] for record in records)),
        "topics": dict(Counter(record.get("topic", "") for record in records if record.get("topic"))),
    }
    for split in ["train", "dev", "test"]:
        split_records = [record for record in records if record["split"] == split]
        write_jsonl(split_records, output_dir / f"{split}.jsonl")
        records_to_dataframe(split_records).to_csv(output_dir / f"{split}.csv", index=False)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
