#!/usr/bin/env python3
"""
Translate Spanish medical JSONL datasets to Basque using HiTZ/medical_es-eu.

Translates text fields in-place, keeps IDs, splits, and numeric keys unchanged.
Long texts are split into paragraphs (double-newline), any paragraph still over
MAX_CHARS is further split on sentence boundaries (see sec:translation-artefact
in the manuscript). The paragraph-only guard this replaces silently truncated
any single paragraph without an internal break at MarianMT's 512-token limit,
confirmed to have lost more than half the content of roughly a fifth of
long `evidence` fields in the development split. Sentence-splitting closes
that gap, since a single sentence essentially never exceeds 512 tokens in this
corpus.

Usage:
  python scripts/translation/translate_to_basque.py \
      --input  data/processed/es/train.jsonl \
      --output data/processed/eu/train.jsonl

Adapted from med_rag_thesis's scripts/translate_to_basque.py (the downstream
thesis repo this dataset was built for), trimmed to GuiaSalud only: the
original also drove sns1064/casimedicos, which do not exist in this repo.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from question_format import format_question  # noqa: E402

# "subquestion" no longer exists (split into "subtopic"/"focus", see
# datasetting_precuration.py's own current 7-field schema). GuiaSalud's
# short_answer duplicates judgement, whereas CasiMedicos has no judgement and
# therefore needs its own short_answer translated. collect_texts handles that
# conditional case so duplicate GuiaSalud content is not translated twice.
TRANSLATABLE_FIELDS = ("topic", "subtopic", "question", "focus", "judgement", "evidence", "considerations")
MAX_CHARS = 1800  # rough guard, MarianMT max is ~512 tokens ≈ 2000 chars
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", help="Input JSONL file(s).")
    parser.add_argument("--output", required=True, nargs="+", help="Output JSONL file(s).")
    parser.add_argument("--model", default="HiTZ/medical_es-eu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading translation model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # transformers>=5 refuses to force-tie model.shared.weight / lm_head.weight
    # when the checkpoint stores them as distinct (but supposed-to-be-tied)
    # tensors, leaving lm_head at its untrained init and producing incoherent
    # output. The checkpoint's shared/input embedding is the trained one,
    # copy it into the output head to restore the intended tying.
    input_emb = model.get_input_embeddings().weight
    output_emb = model.get_output_embeddings().weight
    if not torch.equal(input_emb, output_emb):
        with torch.no_grad():
            output_emb.copy_(input_emb)

    model = model.to(device)
    model.eval()
    return tokenizer, model


def translate_batch(texts: list[str], tokenizer: Any, model: Any, device: str) -> list[str]:
    import torch

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # A flat max_length=512 lets the model generate arbitrarily long output for
    # a short input. Observed on isolated short fragments (page headers,
    # "Recomendaciones 1.", single-clause sentences produced by sentence-level
    # chunking) with no repetition guard: beam search degenerates into a
    # repeating-word loop that runs to the full 512-token cap, producing
    # Basque output 20-30x longer than the source instead of a translation.
    # A generous but input-relative cap (plus a repetition penalty and
    # no-repeat n-gram constraint) keeps normal-length translations unaffected
    # while making runaway generation on short/odd inputs structurally
    # impossible rather than merely unlikely.
    input_len = int(inputs["input_ids"].shape[1])
    gen_max_length = min(512, max(32, input_len * 3))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=gen_max_length,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]


def split_into_sentences(paragraph: str) -> list[str]:
    """Split a single paragraph (no internal double-newline) into sentences,
    used when the paragraph alone is still over the token budget."""
    sentences = [s for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    return sentences if sentences else [paragraph]


# Tabular / enumerated-list source text (dosage tables, numbered subgroup
# breakdowns) has few or no sentence-ending periods, so SENTENCE_SPLIT_RE
# leaves it as one long "sentence" (observed directly on a real record: a
# 940-char / 277-token run with 18 colons and 11 periods, well UNDER the
# 400-token pack budget, so size-based splitting alone never touches it). Fed
# to the model whole, it degenerated into repeating garbled fragments: the
# problem is not length, it is that the text has no real prose structure
# (short colon-delimited "label: value" runs), which the model cannot handle
# as a single unit regardless of token count. Any sentence with several
# colons/semicolons, the structural signature of this failure mode, is
# therefore split on them BEFORE the size check, not only as an
# over-budget fallback, a sentence with too few colons to split usefully
# falls through to the word-count split as a last resort.
SECONDARY_SPLIT_RE = re.compile(r"(?<=[:;])\s+")
TABULAR_COLON_THRESHOLD = 3
WORD_CHUNK_SIZE = 40


def split_oversized_sentence(sentence: str, tokenizer: Any) -> tuple[list[str], bool]:
    """Returns (pieces, is_tabular). is_tabular is True when the split was
    triggered by the colon/semicolon heuristic (tabular source text) rather
    than by the token budget alone, pack_sentences must NOT recombine
    those pieces, since re-merging them recreates the exact structureless
    blob that caused the model to degenerate in the first place."""
    within_budget = len(tokenizer(sentence, add_special_tokens=False)["input_ids"]) <= SENTENCE_TOKEN_BUDGET
    looks_tabular = sentence.count(":") + sentence.count(";") >= TABULAR_COLON_THRESHOLD
    if within_budget and not looks_tabular:
        return [sentence], False
    pieces = [p for p in SECONDARY_SPLIT_RE.split(sentence) if p.strip()]
    if len(pieces) <= 1:
        if within_budget:
            return [sentence], False
        words = sentence.split()
        pieces = [
            " ".join(words[i : i + WORD_CHUNK_SIZE])
            for i in range(0, len(words), WORD_CHUNK_SIZE)
        ] or [sentence]
        return pieces, False
    out: list[str] = []
    for piece in pieces:
        if len(tokenizer(piece, add_special_tokens=False)["input_ids"]) > SENTENCE_TOKEN_BUDGET:
            words = piece.split()
            out.extend(
                " ".join(words[i : i + WORD_CHUNK_SIZE])
                for i in range(0, len(words), WORD_CHUNK_SIZE)
            )
        else:
            out.append(piece)
    return (out if out else [sentence]), True


# Isolated short/fragment sentences (page headers, "Recomendaciones 1.",
# numbered-list stubs produced by splitting dense clinical prose) starve
# MarianMT of context: translated alone, beam search on this model reliably
# degenerates into a repeating-word loop, observed directly, not a
# hypothetical: 12 of 21 single-sentence calls on one real record produced
# incoherent output (ratios of 5-13x the source length) even with a
# repetition penalty and a length cap. Packing several consecutive sentences
# into one call restores enough context that this does not happen. The
# packed group is kept under SENTENCE_TOKEN_BUDGET (measured with the real
# tokenizer, not a character proxy) so it still fits MarianMT's 512-token
# limit with margin for the Basque translation, which can run longer than
# the Spanish source.
SENTENCE_TOKEN_BUDGET = 400

# A multi-sentence text within this fraction of SENTENCE_TOKEN_BUDGET is
# treated as "near budget" by translate_text and forced through sentence
# packing at a TIGHTER budget (SENTENCE_TOKEN_BUDGET * this same margin)
# rather than sent as one call, even though it technically fits: MarianMT's
# beam search can stop generating early, mid-sentence, on a long, dense,
# citation-heavy paragraph well before any hard length cap is reached
# (confirmed directly: guiasalud_89's 399-token evidence field, previously
# admitted whole, produced a 135-token Basque output that trails off
# incoherently, not a clean cut). 0.75 keeps genuinely short/simple text on
# the cheap single-call path (most records) while catching the dense,
# borderline-budget case that caused this.
SINGLE_CALL_SAFETY_MARGIN = 0.75


def pack_sentences(sentences: list[str], tokenizer: Any, budget: int = SENTENCE_TOKEN_BUDGET) -> list[str]:
    """Greedily group consecutive sentences into chunks whose tokenized
    length stays under `budget` (SENTENCE_TOKEN_BUDGET by default). Any
    sentence that alone exceeds the budget, or looks tabular (few sentence-
    ending periods, many colons), is split further by split_oversized_sentence
    before packing. Pieces from a tabular split are kept as their OWN chunk,
    never merged with neighbours: re-combining colon-delimited "label: value"
    fragments back into one call recreates the exact structureless blob that
    made the model degenerate in the first place (observed directly on a
    real record), even though the merged chunk stays under the token budget.

    `budget` is overridable (below SENTENCE_TOKEN_BUDGET) for text that is
    known to need extra generation headroom despite fitting the default
    budget as a raw token count, translate_text's own near-budget,
    multi-sentence fallback path (see its docstring) is the only caller that
    does this: a whole 12-sentence, 399-token paragraph packed at the
    default 400-token budget collapses right back into one ~399-token
    group, identical to the single-call path it was meant to avoid, so that
    caller passes a tighter budget to force it into genuinely smaller,
    multi-call chunks instead."""
    expanded: list[tuple[str, bool]] = []
    for sentence in sentences:
        pieces, is_tabular = split_oversized_sentence(sentence, tokenizer)
        expanded.extend((piece, is_tabular) for piece in pieces)

    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence, is_tabular in expanded:
        if is_tabular:
            if current:
                groups.append(" ".join(current))
                current, current_tokens = [], 0
            groups.append(sentence)
            continue
        sentence_tokens = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
        if current and current_tokens + sentence_tokens > budget:
            groups.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        groups.append(" ".join(current))
    return groups


def fits_token_budget(text: str, tokenizer: Any) -> bool:
    """The real constraint is MarianMT's 512-token input limit, not
    characters. MAX_CHARS is a cheap pre-filter (most text is prose, where
    char count is a fine proxy), but number/citation-dense clinical text can
    have a much lower chars-per-token ratio (observed directly: a
    1,723-char record (under MAX_CHARS=1800) tokenized to 554 tokens (over
    512) and, translated whole on the strength of the char check alone, was
    silently truncated after its first two sentences with the rest of the
    paragraph, an entire dosage-comparison table, dropped with no error).
    Any text passed here is therefore re-checked with the actual tokenizer
    before being trusted as a single call."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= SENTENCE_TOKEN_BUDGET


def translate_text(text: str, tokenizer: Any, model: Any, device: str) -> str:
    """Translate a single text, chunking by paragraph, then packing sentences
    for any paragraph still over the token budget after that split.

    A whole text is only ever sent as ONE call when it is short enough that
    the generation cap (gen_max_length = min(512, input_len * 3),
    translate_batch's own comment) leaves real headroom, not merely when it
    fits under SENTENCE_TOKEN_BUDGET at all: a dense, multi-clause, multi-
    citation paragraph that is just UNDER that budget (confirmed directly on
    a real record, guiasalud_89's evidence field: 399 tokens, admitted whole
    by the old `<= SENTENCE_TOKEN_BUDGET` check) can still make MarianMT's
    beam search stop early mid-sentence well before hitting any length cap,
    producing Basque output that trails off incoherently rather than being
    cleanly cut, an integrity-check failure (check_translation_integrity.py)
    distinct from, but with the same root cause as, the over-budget silent-
    truncation incident fits_token_budget's own docstring already documents.
    Multi-sentence text within a fraction of the budget is therefore always
    routed through per-sentence packing instead, so each MarianMT call gets
    a shorter input and proportionally more output headroom; only text that
    is ALSO effectively one atomic sentence (nothing left to usefully split)
    still goes through as a single call regardless of its token count."""
    if not text or not text.strip():
        return text
    if len(text) <= MAX_CHARS and fits_token_budget(text, tokenizer):
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        near_budget = token_count > SENTENCE_TOKEN_BUDGET * SINGLE_CALL_SAFETY_MARGIN
        has_multiple_sentences = len(split_into_sentences(text)) > 1
        if not (near_budget and has_multiple_sentences):
            return translate_batch([text], tokenizer, model, device)[0]
    # chunk by double-newline (paragraph break), further split any paragraph
    # still over the token budget into token-budgeted sentence groups, so no
    # chunk handed to MarianMT can silently exceed its 512-token limit or be
    # so short/decontextualized that generation degenerates.
    paragraphs = [c for c in text.split("\n\n") if c.strip()]
    if not paragraphs:
        return text
    pieces: list[str] = []
    piece_is_paragraph_end: list[bool] = []
    for paragraph in paragraphs:
        paragraph_fits = len(paragraph) <= MAX_CHARS and fits_token_budget(paragraph, tokenizer)
        paragraph_near_budget = paragraph_fits and (
            len(tokenizer(paragraph, add_special_tokens=False)["input_ids"])
            > SENTENCE_TOKEN_BUDGET * SINGLE_CALL_SAFETY_MARGIN
        )
        paragraph_has_multiple_sentences = paragraph_fits and len(split_into_sentences(paragraph)) > 1
        if paragraph_fits and not (paragraph_near_budget and paragraph_has_multiple_sentences):
            pieces.append(paragraph)
            piece_is_paragraph_end.append(True)
        else:
            # A paragraph that only landed here because it was near-budget
            # (not because it was actually over SENTENCE_TOKEN_BUDGET) needs
            # a TIGHTER pack budget, or greedy packing just recombines it
            # back into one near-400-token group, identical to the
            # single-call path this branch exists to avoid.
            pack_budget = (
                int(SENTENCE_TOKEN_BUDGET * SINGLE_CALL_SAFETY_MARGIN)
                if paragraph_fits
                else SENTENCE_TOKEN_BUDGET
            )
            groups = pack_sentences(split_into_sentences(paragraph), tokenizer, budget=pack_budget)
            for j, group in enumerate(groups):
                pieces.append(group)
                piece_is_paragraph_end.append(j == len(groups) - 1)
    translated_pieces = translate_batch(pieces, tokenizer, model, device)
    # Re-assemble: a paragraph boundary gets "\n\n", a group boundary within
    # a re-split paragraph gets a single space.
    out_parts: list[str] = []
    for i, (piece, is_end) in enumerate(zip(translated_pieces, piece_is_paragraph_end)):
        out_parts.append(piece)
        if i == len(translated_pieces) - 1:
            continue
        out_parts.append("\n\n" if is_end else " ")
    return "".join(out_parts)


def collect_texts(records: list[dict[str, Any]]) -> tuple[list[str], list[tuple[int, str]]]:
    """Return a flat list of texts and their (record_idx, field) locations."""
    texts: list[str] = []
    locations: list[tuple[int, str]] = []
    for i, record in enumerate(records):
        for field in TRANSLATABLE_FIELDS:
            value = record.get(field)
            if value and isinstance(value, str) and value.strip():
                texts.append(value.strip())
                locations.append((i, field))
        # CasiMedicos has no judgement: its short_answer is distinct gold
        # content and must be translated. GuiaSalud's is copied from its
        # translated judgement in apply_translations below.
        short_answer = record.get("short_answer")
        if not record.get("judgement") and isinstance(short_answer, str) and short_answer.strip():
            texts.append(short_answer.strip())
            locations.append((i, "short_answer"))
        # options: translate string values, skip "nan" and numeric-only strings
        options = record.get("options")
        if isinstance(options, dict):
            for key, val in options.items():
                if isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
                    texts.append(val.strip())
                    locations.append((i, f"options.{key}"))
    return texts, locations


def run_translation(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
) -> list[str]:
    results: list[str] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        # split long texts individually
        translated: list[str] = []
        short, short_idx = [], []
        for j, t in enumerate(batch):
            if len(t) <= MAX_CHARS and fits_token_budget(t, tokenizer):
                short.append(t)
                short_idx.append(j)
            else:
                translated.append((j, translate_text(t, tokenizer, model, device)))
        if short:
            batch_out = translate_batch(short, tokenizer, model, device)
            for j, out in zip(short_idx, batch_out):
                translated.append((j, out))
        translated.sort()
        results.extend(out for _, out in translated)
        print(f"  Translated {min(start + batch_size, total)}/{total}", flush=True)
    return results


def apply_translations(
    records: list[dict[str, Any]],
    locations: list[tuple[int, str]],
    translations: list[str],
) -> None:
    for (rec_idx, field), translation in zip(locations, translations):
        if field.startswith("options."):
            key = field[len("options."):]
            records[rec_idx]["options"][key] = translation
        else:
            records[rec_idx][field] = translation
    # GuiaSalud short_answer duplicates judgement pre-translation, so copy the
    # now-translated judgement. CasiMedicos has no judgement and its own
    # short_answer was translated above; do not overwrite it with Spanish.
    for record in records:
        if record.get("judgement"):
            record["short_answer"] = record["judgement"]


def build_justification_eu(evidence: str, considerations: str, evidence_label: str, considerations_label: str) -> str:
    """Basque-language equivalent of prepare_sns1064.py's build_justification,
    with the two Spanish label phrases replaced by their real MT-translated
    Basque equivalents (evidence_label/considerations_label), passed in
    rather than hardcoded. Mirrors build_justification's exact skip-when-blank
    structure: no bare considerations line is emitted when considerations is
    empty (~18% of GuiaSalud records)."""
    lines = []
    if evidence:
        lines.append(f"{evidence_label}: {evidence}")
    if considerations:
        lines.append(f"{considerations_label}: {considerations}")
    return "\n".join(lines)


def rebuild_query_and_justification(
    records: list[dict[str, Any]],
    evidence_label_eu: str,
    considerations_label_eu: str,
) -> None:
    """Rebuild `query` and `justification` from each record's own
    now-TRANSLATED individual fields, using the Basque-language versions of
    the same composition logic prompts.format_question / prepare_sns1064.
    build_justification use for Spanish. Must run AFTER apply_translations,
    never before: rebuilding from pre-translation (still-Spanish) fields
    would silently leave stale Spanish text in the two fields the model is
    actually shown/scored against.

    GuiaSalud vs CasiMedicos is discriminated by the "guidebook" field:
    confirmed present on every GuiaSalud record and absent from every
    CasiMedicos record in data/processed/{guiasalud,casimedicos}/dev.jsonl.
    "subtopic"/"focus" were not used for this since both are legitimately
    blank on some GuiaSalud records, so their presence is a less reliable
    discriminator than "guidebook", which is structural (always set or
    never set per source), not content-dependent.
    """
    for record in records:
        if "guidebook" in record:
            # GuiaSalud: rebuild the Tema/Subtema/Pregunta/Foko composite from
            # the record's own translated topic/subtopic/question/focus.
            # format_question(language="eu") never reads a "query" field (its
            # query-shortcut is gated on `language != "eu"`), so it always
            # recomputes live from the translated parts, never reusing a
            # stale Spanish query left on the record.
            record["query"] = format_question(record, language="eu")
            evidence = str(record.get("evidence", "") or "").strip()
            considerations = str(record.get("considerations", "") or "").strip()
            record["justification"] = build_justification_eu(
                evidence, considerations, evidence_label_eu, considerations_label_eu
            )
        else:
            # CasiMedicos: query/justification are verbatim copies of the
            # (now-translated) question/evidence fields, mirroring the
            # existing short_answer<-judgement copy-after-translate pattern.
            if "question" in record:
                record["query"] = record["question"]
            if "evidence" in record:
                record["justification"] = record["evidence"]


def process_file(
    input_path: Path,
    output_path: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    evidence_label_eu: str,
    considerations_label_eu: str,
) -> None:
    print(f"\n{input_path} → {output_path}", flush=True)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts, locations = collect_texts(records)
    print(f"  {len(records)} records, {len(texts)} text fields to translate", flush=True)
    translations = run_translation(texts, tokenizer, model, device, batch_size)
    apply_translations(records, locations, translations)
    # Must run strictly after apply_translations: query/justification are
    # rebuilt from each record's own now-translated fields, never from the
    # pre-translation Spanish ones.
    rebuild_query_and_justification(records, evidence_label_eu, considerations_label_eu)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    print(f"  Saved {len(records)} records to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.output):
        print("ERROR: --input and --output must have the same number of paths.", file=sys.stderr)
        sys.exit(1)

    tokenizer, model = load_model(args.model, args.device)

    # Translate the two GuiaSalud justification label phrases through the
    # SAME translate_batch() used for all other content, once, up front (not
    # invented/hardcoded), so build_justification_eu uses real MT output for
    # its labels just like every other translated field.
    evidence_label_es = "Evidencia procedente de la investigación"
    considerations_label_es = "Consideraciones adicionales"
    evidence_label_eu, considerations_label_eu = translate_batch(
        [evidence_label_es, considerations_label_es], tokenizer, model, args.device
    )
    print(
        f"Translated justification labels: evidence={evidence_label_eu!r}, "
        f"considerations={considerations_label_eu!r}",
        flush=True,
    )

    for inp, out in zip(args.input, args.output):
        process_file(
            Path(inp), Path(out), tokenizer, model, args.device, args.batch_size,
            evidence_label_eu, considerations_label_eu,
        )

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
