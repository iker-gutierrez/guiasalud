#!/usr/bin/env python
"""Create and curate copies of the pre-curation GuiaSalud split files.

``dataset_precuration.csv`` and ``*_precuration.csv`` are immutable audit
baselines.  This script first creates ``train.csv``, ``dev.csv``, and
``test.csv`` as copies where absent, applies the reviewed fixes only to those
curated copies, and rebuilds ``dataset.csv`` from them.  Existing curated
splits are never overwritten wholesale, so further manual edits remain safe;
the exact substitutions below are idempotent and may be applied to them on a
later run.

datasetting_postcuration.py now only re-runs the completeness checks
(mandatory/optional feature presence) against the dataset.csv this script
produces; the descriptive statistics (sentence/token counts, per-field
lengths, guidebook distribution) that used to live in postcuration.py's own
§10 have moved to datasetting_precuration.py's own log, since none of them
actually depend on manual curation having happened.

These are 19 of the 20 cases from the "automatically unresolved line-wrap
hyphenation artifacts" list (§6.4 of datasetting_precuration.py) that
remained after the automatic corpus-frequency/BSC-tokenizer approach: either
the split word never repeats unsplit anywhere in its own guidebook (so
corpus-frequency has no signal) and isn't common enough to be the BSC
tokenizer's own single vocabulary token (so the fallback can't confirm it
either), even though a human reader can tell at a glance what the correct
form is. This is exactly the manual-judgment step
_merge_hyphenated_line_wraps' own docstring says automation was not trusted
to make (see datasetting_precuration.py §1). Recorded here as an explicit,
reviewed (id, field, find, replace) table rather than a general regex rule,
so it's reproducible without re-doing the manual review, and so it can
never accidentally touch a DIFFERENT row that happens to contain the same
substring.

The 20th case (guiasalud_203, "Scale- y" in "PPS-Palliative Performance
Scale- y RASS-Richmond...") is deliberately NOT included: the "-" there is
plausibly a PDF-to-TXT em-dash-to-hyphen conversion artifact running through
that whole phrase, not this row's own line-wrap split, so neither "join into
one word" nor "join with a space" has been confirmed as the right fix. Left
for a dedicated look at that broader em-dash-vs-hyphen question.

Usage:
    python guiasalud_datasetting/manual_curation.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "interim" / "guiasalud"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the pre-curation and curated CSV files.",
    )
    return parser.parse_args()

# (id, field, exact substring to find, exact replacement), reviewed by hand
# against each row's actual current text. "find" is the literal hyphenated
# fragment as it appears in the CSV (single space after the hyphen, matching
# _merge_hyphenated_line_wraps' own normalization), not a regex.
CURATION_TABLE: list[tuple[str, str, str, str]] = [
    # Genuine line-wrap splits: join, dropping the hyphen entirely.
    ("guiasalud_272", "evidence", "levopremo- mazina", "levomepromazina"),
    ("guiasalud_312", "evidence", "interacciona- ban", "interaccionaban"),
    ("guiasalud_790", "evidence", "es- cogida", "escogida"),
    ("guiasalud_224", "considerations", "for- zarlo", "forzarlo"),
    ("guiasalud_211", "considerations", "interdis- ciplinares", "interdisciplinares"),
    ("guiasalud_250", "considerations", "malinter- pretada", "malinterpretada"),
    ("guiasalud_260", "considerations", "admi- nistrarlas", "administrarlas"),
    ("guiasalud_281", "considerations", "vigi- lando", "vigilando"),
    ("guiasalud_293", "evidence", "cum- plimentadas", "cumplimentadas"),
    ("guiasalud_313", "evidence", "mioclo- nías", "mioclonías"),
    ("guiasalud_323", "evidence", "in- terrumpieron", "interrumpieron"),
    ("guiasalud_324", "evidence", "hipofrac- cionada", "hipofraccionada"),
    ("guiasalud_785", "considerations", "compen- sado", "compensado"),
    # guiasalud_203 ("Scale- y" in "PPS-Palliative Performance Scale- y
    # RASS-Richmond...") deliberately left untouched: the "-" throughout
    # that phrase is plausibly a PDF-to-TXT em-dash-to-hyphen conversion
    # artifact (a document-wide style issue, not this row's own line-wrap
    # split), so neither the "join into one word" nor "join with a space"
    # correction has been confirmed as the right one. Left for a dedicated
    # look at the em-dash-vs-hyphen question, not fixed as part of this batch.
    # Genuine hyphenated compounds: normalize the internal spacing (drop the
    # stray space the line-wrap left after the hyphen) WITHOUT removing the
    # hyphen itself, since these are real compound words in Spanish/English,
    # not accidental splits ("coste-efectividad", "high-density",
    # "dietistas-nutricionistas").
    ("guiasalud_894", "evidence", "coste- efectividad", "coste-efectividad"),
    ("guiasalud_14", "evidence", "coste- efectivas", "coste-efectivas"),
    ("guiasalud_359", "considerations", "high- density", "high-density"),
    ("guiasalud_394", "evidence", "coste- efectividad", "coste-efectividad"),
    ("guiasalud_498", "considerations", "dietistas- nutricionistas", "dietistas-nutricionistas"),
    ("guiasalud_857", "evidence", "coste- efectividad", "coste-efectividad"),
]


def apply_curation(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        return 0
    applied = 0
    for sample_id, field, find, replace in CURATION_TABLE:
        mask = df["id"] == sample_id
        if not mask.any() or field not in df.columns:
            continue
        row_index = df.index[mask][0]
        value = str(df.at[row_index, field])
        if find not in value:
            continue
        df.at[row_index, field] = value.replace(find, replace, 1)
        applied += 1
    if applied:
        df.to_csv(csv_path, index=False)
    return applied


def copy_curated_splits() -> None:
    # datasetting_precuration.py §8 describes this copy as the manual,
    # off-script step that starts real curation ("copy train_precuration.csv
    # to train.csv, ... then curate the copies"). Refuse to overwrite an
    # existing curated copy: it may contain edits beyond this fixed table.
    for split in ("train", "dev", "test"):
        source = DATA_DIR / f"{split}_precuration.csv"
        destination = DATA_DIR / f"{split}.csv"
        if not source.exists():
            print(f"{destination.name}: SKIPPED ({source.name} not found)")
            continue
        if destination.exists():
            print(f"{destination.name}: SKIPPED (already exists, not overwritten)")
            continue
        shutil.copy2(source, destination)
        print(f"{destination.name}: created from {source.name}")


# Same sort convention datasetting_precuration.py's own
# _sort_by_guidebook_then_id uses: (guidebook, id-as-number), not a plain
# guidebook-only sort, so document order within each guidebook is preserved
# rather than left in whatever order train/dev/test's own row order put it.
def _sort_by_guidebook_then_id(df: pd.DataFrame) -> pd.DataFrame:
    id_num = df["id"].str.removeprefix("guiasalud_").astype(int)
    return (
        df.assign(_id_num=id_num)
        .sort_values(by=["guidebook", "_id_num"])
        .drop(columns="_id_num")
        .reset_index(drop=True)
    )


def build_dataset_csv() -> None:
    split_paths = {split: DATA_DIR / f"{split}.csv" for split in ("train", "dev", "test")}
    missing = [name for name, path in split_paths.items() if not path.exists()]
    if missing:
        print(f"dataset.csv: SKIPPED (missing {', '.join(missing)}.csv)")
        return
    frames = []
    for split, path in split_paths.items():
        split_df = pd.read_csv(path)
        split_df["split"] = split
        frames.append(split_df)
    dataset_df = pd.concat(frames, ignore_index=True)
    dataset_df = _sort_by_guidebook_then_id(dataset_df)
    dataset_df.insert(1, "split", dataset_df.pop("split"))
    dataset_df.to_csv(DATA_DIR / "dataset.csv", index=False)
    print(f"dataset.csv: rebuilt from train.csv + dev.csv + test.csv ({len(dataset_df)} rows)")


def main() -> None:
    global DATA_DIR
    DATA_DIR = parse_args().data_dir.resolve()
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {DATA_DIR}")
    copy_curated_splits()
    # Only the copies are curated.  Pre-curation files remain an unmodified
    # before-state suitable for auditing the exact changes below.
    for split in ("train", "dev", "test"):
        csv_path = DATA_DIR / f"{split}.csv"
        applied = apply_curation(csv_path)
        noun = "fix" if applied == 1 else "fixes"
        print(f"{csv_path.name}: {applied} {noun} applied")
    build_dataset_csv()
    print()


if __name__ == "__main__":
    main()
