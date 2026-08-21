"""Builds the composite `query` field (Tema/Subtema/Pregunta/Foko block) shown
to the model as the question, in Spanish or Basque.

Extracted from med_rag_thesis's src/medical_rag_thesis/prompts.py (the
downstream thesis repo that consumes this dataset for RAG experiments),
which the translation scripts here otherwise have no reason to depend on.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

# The letter prefix ("b) ") and focus prefix ("b.1. ") are source-document
# labels with no meaning to the model, and there's no indication that
# topic/subtopic/question/focus narrow down to ONE answer, not four. Each
# field is instead given its own labeled line, stripping the source's own
# lettering (GRADE letter, "letter.number." focus prefix) since that
# labeling is redundant once the field has its own name here.
_GRADE_LETTER_PREFIX_PATTERN = re.compile(r'^[a-z]\)\s*')
_FOCUS_PREFIX_PATTERN = re.compile(r'^[a-z]\.\d+\.\s*')

QUESTION_FIELD_LABELS = {
    "es": {"topic": "Tema", "subtopic": "Subtema", "question": "Pregunta", "focus": "Foco"},
    "eu": {"topic": "Gaia", "subtopic": "Azpigaia", "question": "Galdera", "focus": "Foku"},
}


def format_question(record: Mapping[str, Any], language: str = "es") -> str:
    # A record with a pre-built `query` field (this dataset's GuiaSalud
    # records carry one, in both Spanish and Basque) is preferred directly,
    # so the composite is not built twice. Any record without `query` at all
    # degrades gracefully to the live per-field composite below.
    query = str(record.get("query", "") or "").strip()
    if query:
        return query

    labels = QUESTION_FIELD_LABELS[language]
    topic = str(record.get("topic", "") or "").strip()
    subtopic = str(record.get("subtopic", "") or "").strip()
    question = _GRADE_LETTER_PREFIX_PATTERN.sub("", str(record.get("question", "") or "").strip())
    focus = _FOCUS_PREFIX_PATTERN.sub("", str(record.get("focus", "") or "").strip())

    lines = []
    if topic:
        lines.append(f"- {labels['topic']}: {topic}")
    if subtopic:
        lines.append(f"- {labels['subtopic']}: {subtopic}")
    if question:
        lines.append(f"- {labels['question']}: {question}")
    if focus:
        lines.append(f"- {labels['focus']}: {focus}")
    return "\n".join(lines)
