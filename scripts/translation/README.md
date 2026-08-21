# Basque translation pipeline

Translates the Spanish `data/processed/es/` GuiaSalud JSONL files into
`data/processed/eu/`, the Basque version published alongside `es/` on the
[Hugging Face dataset](https://huggingface.co/datasets/ikergf/guiasalud).

Adapted from the downstream thesis repo
([`medical-rag-es-eu`](https://github.com/iker-gutierrez/evirag)) that
consumes this dataset for RAG experiments, trimmed to GuiaSalud only (the
original pipeline also drove two other, unrelated datasets).

## Pipeline

```bash
python scripts/translation/translate_to_basque.py \
    --input  data/processed/es/train.jsonl data/processed/es/dev.jsonl data/processed/es/test.jsonl \
    --output data/processed/eu/train.jsonl data/processed/eu/dev.jsonl data/processed/eu/test.jsonl

python scripts/translation/check_translation_integrity.py
```

`translate_to_basque.py` runs the HiTZ/medical_es-eu MarianMT model over
every translatable field, with paragraph- and sentence-boundary chunking to
avoid silent truncation at the model's 512-token input limit, then rebuilds
the composite `query`/`justification` fields from the translated parts.

`check_translation_integrity.py` fails if any Basque field is short enough,
relative to its Spanish source, to indicate lost content rather than
ordinary cross-language compactness.

## Quality

```bash
python scripts/translation/evaluate_translation_quality.py \
    --source data/processed/es/dev.jsonl --target data/processed/eu/dev.jsonl \
    --output reports/eu_dev_translation_quality.json
```

Reference-less quality-estimation scoring (COMET, `Unbabel/wmt22-cometkiwi-da`)
of each Spanish/Basque pair, complementing the integrity check: it can tell
a fluent, faithful translation from a shorter-but-still-plausible one, which
the length-ratio check alone cannot.

## Publishing

`data/processed/` is gitignored: `es/` and `eu/` are built and kept locally
during dataset creation, not committed to this repo. After regenerating or
patching them, push the updated files to the
[Hugging Face dataset](https://huggingface.co/datasets/ikergf/guiasalud)
(`es/{split}.jsonl`, `eu/{split}.jsonl`), which is the published source of
truth downstream consumers, including `evirag`, read from.
