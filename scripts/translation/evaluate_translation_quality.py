#!/usr/bin/env python
"""Reference-less (quality-estimation) evaluation of the Spanish->Basque
translation, complementing check_translation_integrity.py rather than
replacing it.

check_translation_integrity.py only asks whether content went missing (a
length-ratio floor): it cannot tell a fluent, faithful Basque translation
from a shorter-but-still-plausible one, since it has no notion of meaning.
This script scores each (Spanish source, Basque translation) PAIR directly
with a COMET quality-estimation model (no Basque reference needed, hence
"reference-less"), the same family of metric WMT's shared quality-estimation
task uses to rank MT systems without human references.

Default model: Unbabel/wmt22-cometkiwi-da, a stronger reference-free QE model
than the older wmt20-comet-qe-da. It is gated on the HF Hub, but access has
been confirmed granted for this account (verified by an actual successful
download, not just the acceptance page). If a future run ever hits a 403
here (e.g. a different account/environment), fall back with
`--model Unbabel/wmt20-comet-qe-da`, which is not gated.

Usage:
  python scripts/translation/evaluate_translation_quality.py \\
      --source data/processed/es/dev.jsonl \\
      --target data/processed/eu/dev.jsonl \\
      --output reports/eu_dev_translation_quality.json

Adapted from med_rag_thesis's scripts/evaluate_translation_quality.py (the
downstream thesis repo this dataset was built for); no other changes, this
script was already generic and self-contained.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Changed this session from the old 7-field schema (topic/subtopic/question/
# focus/judgement/evidence/considerations, translate_to_basque.py's
# pre-migration TRANSLATABLE_FIELDS) to the three fields actually consumed
# downstream: `query` (what's shown to the model as the question),
# `short_answer` (copied from the translated `judgement`), and
# `justification` (what the model is scored against, replacing the old sole
# use of `evidence`). Once query/justification exist and are rebuilt from
# the translated individual fields (see translate_to_basque.py's
# rebuild_query_and_justification), scoring the individual topic/subtopic/
# focus/evidence/considerations parts separately is redundant: they are no
# longer directly consumed anywhere in prompting, only query/short_answer/
# justification are. This also affects GuiaSalud's own already-built
# evaluate_guiasalud_eu_translation.sh, since QE_FIELDS is shared code; that
# is intentional, it is a correctness improvement there too, not a
# CasiMedicos-only change.
QE_FIELDS = ("query", "short_answer", "justification")

DEFAULT_MODEL = "Unbabel/wmt22-cometkiwi-da"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference-less QE scoring of a Basque translation.")
    parser.add_argument("--source", required=True, help="Spanish JSONL (translation source).")
    parser.add_argument("--target", required=True, help="Basque JSONL (translation output), same ids as --source.")
    parser.add_argument("--output", required=True, help="Output metrics JSON.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="COMET QE model name on the HF Hub.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--min-source-chars",
        type=int,
        default=1,
        help="Skip fields shorter than this (matches check_translation_integrity.py's "
        "MIN_SOURCE_CHARS=40 by default being intentionally lower here: QE scores single "
        "short segments fine, unlike the length-ratio check, so there is no need to "
        "exclude them the same way).",
    )
    parser.add_argument(
        "--extra-fields",
        nargs="+",
        default=(),
        help="Score these fields in addition to QE_FIELDS, without changing the shared "
        "module default other callers (e.g. GuiaSalud's own eval) rely on. Used to score "
        "fields excluded from QE_FIELDS because they never reach `query`/`justification` "
        "(e.g. CasiMedicos-Exp's `topic`, the original dataset's `type` column, the MIR "
        "exam-section label, reused as `specialty` in the mixed guiasalud_casimedicos "
        "schema -- same underlying text, different column name per dataset).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[record["id"]] = record
    return records


def collect_pairs(
    source: dict[str, dict],
    target: dict[str, dict],
    min_source_chars: int,
    extra_fields: tuple[str, ...] = (),
) -> list[dict]:
    """(id, field, src, mt) rows for every field present, non-empty, and
    long enough in both the Spanish source and the Basque target."""
    pairs = []
    fields = QE_FIELDS + tuple(f for f in extra_fields if f not in QE_FIELDS)
    for record_id, source_record in source.items():
        target_record = target.get(record_id)
        if not target_record:
            continue
        for field in fields:
            src_text = str(source_record.get(field) or "").strip()
            mt_text = str(target_record.get(field) or "").strip()
            if len(src_text) < min_source_chars or not mt_text:
                continue
            pairs.append({"id": record_id, "field": field, "src": src_text, "mt": mt_text})
    return pairs


def load_qe_model(model_name: str, gpus: int):
    from comet import download_model, load_from_checkpoint

    checkpoint_path = download_model(model_name)
    model = load_from_checkpoint(checkpoint_path)
    return model


def score_pairs(model, pairs: list[dict], batch_size: int, gpus: int) -> list[float]:
    samples = [{"src": p["src"], "mt": p["mt"]} for p in pairs]
    output = model.predict(samples, batch_size=batch_size, gpus=gpus, progress_bar=True)
    return list(output.scores)


def summarize(pairs: list[dict], scores: list[float]) -> dict:
    for pair, score in zip(pairs, scores):
        pair["qe_score"] = float(score)

    by_field: dict[str, list[float]] = {}
    for pair in pairs:
        by_field.setdefault(pair["field"], []).append(pair["qe_score"])

    field_summary = {
        field: {
            "n": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for field, values in sorted(by_field.items())
    }

    all_scores = [p["qe_score"] for p in pairs]
    overall = {
        "n": len(all_scores),
        "mean": statistics.mean(all_scores) if all_scores else None,
        "median": statistics.median(all_scores) if all_scores else None,
        "min": min(all_scores) if all_scores else None,
        "max": max(all_scores) if all_scores else None,
        "stdev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
    }

    # Lowest-scoring pairs are the most useful single artifact this script
    # produces: a reviewer can spot-check these first rather than reading
    # every translated field, the same way check_translation_integrity.py's
    # own failures list surfaces the worst truncation cases first.
    worst = sorted(pairs, key=lambda p: p["qe_score"])[:20]

    return {
        "overall": overall,
        "by_field": field_summary,
        "worst_20": [
            {"id": p["id"], "field": p["field"], "qe_score": p["qe_score"], "src": p["src"][:200], "mt": p["mt"][:200]}
            for p in worst
        ],
    }


def main() -> None:
    args = parse_args()
    source = load_jsonl(Path(args.source))
    target = load_jsonl(Path(args.target))

    if set(source) != set(target):
        missing_in_target = set(source) - set(target)
        missing_in_source = set(target) - set(source)
        print(
            f"WARNING: id sets differ (source={len(source)}, target={len(target)}, "
            f"missing_in_target={len(missing_in_target)}, missing_in_source={len(missing_in_source)}). "
            "Scoring only the ids present in both."
        )

    pairs = collect_pairs(source, target, args.min_source_chars, tuple(args.extra_fields))
    print(f"{len(pairs)} (id, field) pairs to score with {args.model}", flush=True)
    if not pairs:
        print("No pairs to score, nothing written.")
        return

    model = load_qe_model(args.model, args.gpus)
    scores = score_pairs(model, pairs, args.batch_size, args.gpus)
    summary = summarize(pairs, scores)
    summary["model"] = args.model
    summary["source_path"] = str(args.source)
    summary["target_path"] = str(args.target)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nOverall QE score: mean={summary['overall']['mean']:.4f}  "
          f"median={summary['overall']['median']:.4f}  n={summary['overall']['n']}")
    for field, stats in summary["by_field"].items():
        print(f"  {field:15} mean={stats['mean']:.4f}  n={stats['n']}")
    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()
