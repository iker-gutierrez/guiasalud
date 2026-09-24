# GuiaSalud: structured clinical dataset for evidence-grounded open-answer QA

GuiaSalud is an open-answer medical question-answering dataset built from six
Spanish clinical practice guidebooks published by GuíaSalud (an organism that belongs to the Spanish Ministry of Health
/ Ministerio de Sanidad).
The dataset is built in two languages: Spanish (original) and Basque
(machine-translated).
It is intended for evidence-grounded clinical QA and retrieval-augmented
generation research.

## Dataset at a glance

| Property | Value |
| :--- | :--- |
| Instances | 919 |
| Splits | 731 train / 63 development / 125 test |
| Languages | Spanish (original) and Basque (machine-translated) |
| Sources | Six Spanish clinical practice guidebooks |
| Task | Evidence-grounded open-answer medical QA |
| Structure | Question, short judgement, supporting evidence, and optional considerations |

## Dataset

This repository does not publish the dataset itself: it publishes the
reproducible process that builds it. The curated CSVs (`dataset.csv`,
`train.csv`, `dev.csv`, `test.csv`) and the `data/processed/es`/`eu` JSON
Lines files are all built and kept locally, not committed here. The
published dataset is available on Hugging Face:
[https://huggingface.co/datasets/ikergf/guiasalud](https://huggingface.co/datasets/ikergf/guiasalud).
Downstream consumers should read it from there.

Load either language as a separate configuration:

```python
from datasets import load_dataset

spanish = load_dataset("ikergf/guiasalud", "es")
basque = load_dataset("ikergf/guiasalud", "eu")
```

`clinical_guidebooks_txt.zip`, the source guideline text the construction
pipeline reads, is the one input file this repository does commit.

Each instance contains a stable identifier and structured fields for the source
guidebook, clinical topic, question, optional refinement, clinical judgement,
supporting evidence, and optional considerations. The train/dev/test split is
fixed; it should be used as published rather than recreated with a new random
split.

## Construction of the dataset

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
`dev.csv`, and `test.csv` inside that same directory, as copies of the
corresponding `_precuration.csv` files where needed, keeps the pre-curation
files unchanged, and rebuilds `dataset.csv` there.
`datasetting_postcuration.py` compares the two states and reports
completeness and descriptive statistics.

The manual-curation and post-curation scripts accept `--data-dir`, allowing
those audit steps to run in a separate working directory without altering the
locally built dataset files.

To prepare the dataset for publication, copy
`data/interim/guiasalud/{train,dev,test,dataset}.csv` to the repository root,
then convert the split CSVs to
`data/processed/es/{split}.jsonl`:

```bash
cp data/interim/guiasalud/{train,dev,test,dataset}.csv .
python scripts/prepare_jsonl.py \
    --train-df train.csv --dev-df dev.csv --test-df test.csv \
    --output-dir data/processed/es
```

## Basque translation

`data/processed/es/` and `data/processed/eu/` hold the same records as the
CSV files above, in JSON Lines form, in Spanish and a machine-translated
Basque version respectively. These files are built and kept locally during
dataset creation (`data/processed/` is gitignored, not committed here) and then
published in the Hugging Face dataset linked above, which is the source of
truth downstream consumers read from. The translation pipeline that produces
`eu/` from `es/` is documented in
[scripts/translation/README.md](scripts/translation/README.md).

As an automatic check beyond translation-integrity validation, the
reference-free `Unbabel/wmt22-cometkiwi-da` quality-estimation model was
applied to three normalized composite fields for every Spanish--Basque
instance: `query` combines the native topic, subtopic, question, and focus;
`short_answer` corresponds to the judgement; and `justification` combines the
evidence and considerations.

| Split | n | COMET-QE (mean ± SD) |
| :--- | :---: | :---: |
| Train | 2,193 | 0.766 ± 0.136 |
| Development | 189 | 0.787 ± 0.124 |
| Test | 375 | 0.770 ± 0.131 |
| Combined | 2,757 | 0.768 ± 0.135 |

This is an automatic estimate, not a substitute for human assessment of
medical translation quality.

## Relationship to MeviRAG

This repository owns dataset construction and the published fixed split.
The retrieval, generation, evaluation, and final experimental predictions
are maintained separately in the MeviRAG GitHub repository:
[https://github.com/iker-gutierrez/mevirag](https://github.com/iker-gutierrez/mevirag).

## Citation

If you use this dataset, please cite the Master’s thesis that introduces
GuiaSalud and describes its construction, validation, and use in MeviRAG:

```bibtex
@mastersthesis{gutierrezfandino2026mevirag,
  author = {Gutierrez Fandiño, Iker},
  title  = {{GuiaSalud Dataset and MeviRAG}: Towards Evidence-Grounded Medical QA in Spanish and Basque},
  school = {University of the Basque Country (EHU)},
  year   = {2026},
  type   = {Master's thesis}
}
```

## License

[Creative Commons Attribution-NonCommercial 4.0 International Public License](LICENSE). 

## Contact

For questions, contact Iker Gutierrez Fandiño at ikergutierrezfandino@gmail.com.
