# GuiaSalud: structured Spanish clinical QA dataset

GuiaSalud is a structured Spanish clinical question-answering dataset built
from six clinical practice guidelines of the Spanish National Health System.
It is intended for evidence-grounded clinical QA and retrieval-augmented
generation research.

## Dataset

The repository publishes the final curated CSV files:

| File | Contents |
| --- | --- |
| `dataset.csv` | Full structured dataset. |
| `train.csv` | Training split. |
| `dev.csv` | Development split. |
| `test.csv` | Held-out test split. |
| `clinical_guidebooks_txt.zip` | Source guideline text used by the construction pipeline. |

Each record contains a stable identifier and structured fields for the source
guidebook, clinical topic, question, optional refinement, clinical judgement,
supporting evidence, and optional considerations. The split files are fixed;
they should be used as published rather than recreated with a new random
split.

## Rebuild the dataset

The maintained construction pipeline is in `scripts/`. It preserves every
pre-curation file as an audit baseline and applies reviewed corrections only
to a separate curated copy.

```bash
python -m pip install -r requirements.txt
mkdir -p data/raw/clinical_guidebooks_txt
unzip -oj clinical_guidebooks_txt.zip 'clinical_guidebooks_txt/*.txt' \
  -d data/raw/clinical_guidebooks_txt

python scripts/datasetting_precuration.py
python scripts/manual_curation.py
python scripts/datasetting_postcuration.py
```

The pre-curation script writes its intermediate files to
`data/interim/guiasalud/`. `manual_curation.py` creates `train.csv`,
`dev.csv`, and `test.csv` from the corresponding `_precuration.csv` files
where needed, keeps the pre-curation files unchanged, and rebuilds
`dataset.csv`. `datasetting_postcuration.py` compares the two states and
reports completeness and descriptive statistics.

The manual-curation and post-curation scripts accept `--data-dir`, allowing
those audit steps to run in a separate working directory without altering the
checked-in dataset files.

## Basque translation

`es/` and `eu/` hold the same records as the CSV files above, in JSON Lines
form, in Spanish and a machine-translated Basque version respectively. These
are the files published to the
[Hugging Face dataset](https://huggingface.co/datasets/ikergf/guiasalud).
The translation pipeline that produces `eu/` from `es/` is in
`scripts/translation/`, see its own README for details.

## Relationship to eviRAG

This repository owns dataset construction and the published fixed split.
The retrieval, generation, evaluation, and final experimental predictions
are maintained separately in
[`evirag`](https://github.com/iker-gutierrez/evirag).

## License and contact

See [LICENSE](LICENSE). For questions, contact Iker Gutierrez Fandiño at
igutierrez134@ikasle.ehu.eus.
