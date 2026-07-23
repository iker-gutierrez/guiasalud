# SNS1064: Clinical dataset for RAG-based QA

SNS1064 is a structured Spanish clinical dataset designed for Retrieval-Augmented Generation (RAG) and question answering (QA) systems. It is derived from six clinical practice guidelines from the Spanish National Health System.

## Dataset Overview

- Samples: 1,064 QA instances.
- Domains:
  - Anxiety.
  - Diabetes.
  - Ictus management.
  - Palliative care.
  - Pedriatic palliative care.
  - Secondary ictus prevention.
- Structure:
  - `guidebook` : source document.
  - `topic` : clinical context.
  - `question` : main question.
  - `subquestion` : optional refinement.
  - `judgement` : short clinical answer.
  - `evidence` : supporting text.
  - `considerations` : optional interpretation.

The dataset explicitly separates answers from their supporting evidence, enabling grounded and explainable QA.

---

## Repository Contents

- `dataset.csv` : full dataset.
- `train_df.csv` : train split (80%).
- `test_df.csv` : test split (20%).
- `clinical_guidebooks_txt.zip` : source texts (TXT format).
- `datasetting.ipynb` : dataset construction pipeline.
- `paper.pdf` : full description of dataset and building process.

---

## Construction Pipeline

The dataset is built using a rule-based pipeline:

1. PDF → TXT conversion.
2. Text cleaning (whitespace and bullet removal).
3. Regex-based text segmentation.
4. Sample filtering (removal of non-informative placeholders).
5. Manual curation (entire test set, partial training set).

---

## Use Case

SNS1064 is designed for:
- RAG-based clinical QA.
- evidence-grounded generation.
- explainable NLP in healthcare.

The separation between `judgement` and `evidence`/`considerations` supports factuality, explainability, and traceability, which are essential in high-stakes domains such as healthcare.

---

## Remarks

- The test set is fully curated to ensure reliable RAG evaluation.
- Some noise is expected in the training set.


---

## Contact

Iker Gutierrez Fandiño  
University of the Basque Country (EHU)  
igutierrez134@ikasle.ehu.eus

