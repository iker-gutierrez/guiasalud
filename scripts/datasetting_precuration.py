# -*- coding: utf-8 -*-
"""datasetting_precuration.py

This script implements a regex-based pipeline to construct a structured clinical QA dataset from TXT guidebooks. The process includes text cleaning, segmentation using regular expressions, filtering, and evaluation.

## 0. Imports and data

Import the required libraries:
"""

import csv
import re
from pathlib import Path

import pandas as pd

"""Define the list of clinical guidebook files to be processed.

The TXT files themselves are not stored in this repository, they are
published in the guiasalud repository
(https://github.com/iker-gutierrez/guiasalud) as
`clinical_guidebooks_txt.zip`. Fetch and unzip them into
`data/raw/clinical_guidebooks_txt/` before running this script:

    mkdir -p data/raw/clinical_guidebooks_txt
    curl -L -o /tmp/clinical_guidebooks_txt.zip \\
        https://github.com/iker-gutierrez/guiasalud/raw/main/clinical_guidebooks_txt.zip
    unzip -oj /tmp/clinical_guidebooks_txt.zip "clinical_guidebooks_txt/*.txt" \\
        -d data/raw/clinical_guidebooks_txt

(-j flattens the zip's own nested clinical_guidebooks_txt/ subdirectory and
the "*.txt" filter skips its __MACOSX/ metadata entries.)
"""

ROOT = Path(__file__).resolve().parents[1]
GUIDEBOOKS_DIR = ROOT / "data" / "raw" / "clinical_guidebooks_txt"
# Derived, not raw: this script's own CSV outputs (pre-curation, not yet the
# final processed dataset scripts/prepare_sns1064.py produces) live under
# data/interim/, matching this repo's raw -> interim -> processed convention.
DATA_DIR = ROOT / "data" / "interim" / "guiasalud"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Used only for de-hyphenating PDF line-wrap word splits (§1), as a fallback
# for candidates corpus-frequency can't resolve on its own. A general-purpose
# hunspell Spanish dictionary was tried first and dropped: it has no
# clinical-domain vocabulary (missed real terms like "sobrehidratación"), and
# using its suggestion-list output as a validity signal made it ~800x slower
# than needed for no benefit (fixed by never requesting suggestions, but the
# vocabulary gap remained the real problem). BSC's clinical-Spanish tokenizer
# (trained on real Spanish clinical/biomedical text) is used purely as a
# deterministic vocabulary-membership check, NOT as a language model doing
# fill-mask/generation, which would be nondeterministic run-to-run. Only its
# tokenizer (a fixed lookup table, no forward pass) is loaded. See
# _bsc_is_single_token for exactly what "membership" means here and why.
BSC_TOKENIZER_NAME = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"

files = [
    GUIDEBOOKS_DIR / "ansiedad.txt",
    GUIDEBOOKS_DIR / "atencion_paliativa.txt",
    GUIDEBOOKS_DIR / "cuidados_paliativos_pediatria.txt",
    GUIDEBOOKS_DIR / "diabetes.txt",
    GUIDEBOOKS_DIR / "manejo_ictus.txt",
    GUIDEBOOKS_DIR / "prevencion_secundaria_ictus.txt",
]

"""## 1. Text cleaning

Each line is normalized to reduce noise introduced during PDF-to-TXT conversion. This includes trimming whitespace and removing leading bullet-like characters.

Page headers/footers (the guidebook's own running title, plus the generic
"GUÍAS DE PRÁCTICA CLÍNICA EN EL SNS", both followed by an optional page
number) are also stripped here: left in, they get silently appended to
whatever field is accumulating when a page break falls mid-field. The
titles are matched as a full line (anchored both ends, only trailing
digits/whitespace allowed after the title) rather than as a prefix, so a
body sentence that happens to start with the same words (e.g. "GPC en
ningún caso...") is never affected.
"""

_FOOTER_TITLES = [
    r"GPC PARA EL TRATAMIENTO DEL TRASTORNO DE ANSIEDAD GENERALIZADA EN ATENCIÓN PRIMARIA",
    r"GUÍAS DE PRÁCTICA CLÍNICA EN EL SNS",
    r"GUÍA DE PRÁCTICA CLÍNICA SOBRE ATENCIÓN PALIATIVA AL ADULTO EN SITUACIÓN DE ÚLTIMOS DÍAS",
    r"GUÍA DE PRÁCTICA CLÍNICA SOBRE CUIDADOS PALIATIVOS EN PEDIATRÍA",
    r"GUÍA DE PRÁCTICA CLÍNICA SOBRE DIABETES MELLITUS TIPO 1",
    r"GUÍA DE PRÁCTICA CLÍNICA SOBRE EL MANEJO DEL ICTUS EN ATENCIÓN PRIMARIA",
    r"GUÍA DE PRÁCTICA CLÍNICA SOBRE PREVENCIÓN SECUNDARIA DEL ICTUS\. ACTUALIZACIÓN\.?",
]
# The page number can appear on either side of the title depending on which
# page it fell on ("<title> <n>" or "<n> <title>"). Confirmed 302
# reversed-order occurrences across 4 guidebooks, 92 of which land inside a
# field (e.g. "...comparación indirecta. 82 GUÍAS DE PRÁCTICA CLÍNICA EN EL
# SNS"). Matching only the title-first form left every one of those
# uncaught. Verified this combined pattern has zero false positives against
# ordinary numbered content (e.g. "1. Introducción", numbered
# recommendation lists) corpus-wide. The footer titles are long, specific
# strings no such content coincidentally matches.
footer_pattern = re.compile(
    rf'^(?:(?:{"|".join(_FOOTER_TITLES)})\s*\d*|\d+\s*(?:{"|".join(_FOOTER_TITLES)}))\s*$',
    re.IGNORECASE
)

_bullet_prefix_pattern = re.compile(r'^[\•\●]')

# The dividida narration sentence (see pregunta_dividida_pattern above)
# names how many refinement questions follow, sometimes names the
# anticoagulants/comparisons involved, and sometimes cites the methodology
# tool used to structure them ("La estructura de estas subpreguntas siguió
# la estructura propuesta por GRADE Pro Software19.", wording varies),
# before the real per-subquestion content starts. None of this is clinical
# content, it's a document-authoring aside describing the source guideline's
# own editorial process, not something a reader needs to answer the
# question. Now that subtopic is its own column carrying the actual
# refinement question, this narration is also redundant (previously it was
# the only place a reader saw that the topic had been split at all), and
# left in, it dangles mid-topic with no framing (e.g. "...reduce el riesgo
# de nuevos episodios? Esta pregunta se dividió en cinco subpreguntas en
# función de los anticoagulantes..."). Confirmed corpus-wide (11
# occurrences, diabetes.txt and prevencion_secundaria_ictus.txt) that this
# sentence, plus the GRADE Pro sentence when present, is always the LAST
# thing appended to the topic buffer before the real per-subquestion cutoff
# (a bulleted preview line or a "Subpregunta N.M." trigger) closes it, with
# nothing legitimate ever following it within the same buffer. Rather than
# match the sentence's own (highly variable) start and end (three different
# openings exist corpus-wide, see pregunta_dividida_pattern's comment),
# this finds the trigger phrase (same pattern used to detect the narration
# during collection) and truncates everything from the END of the
# PRECEDING sentence onward, which is robust to all three wordings and
# both "...:"/"...." endings without needing to describe them individually.
def _strip_dividida_narration(topic):
    m = pregunta_dividida_pattern.search(topic)
    if not m:
        return topic
    before = topic[:m.start()]
    cutoff = max(before.rfind(". "), before.rfind("? "), before.rfind("! "))
    if cutoff == -1:
        return ""
    return topic[:cutoff + 1].strip()

# judgement/evidence/considerations are meant to read as standalone
# sentences (they're the fields BERTScore compares model output against),
# but the extraction cuts them out of a larger paragraph, so they often
# start lowercase mid-sentence in the source. Capitalizes the first letter
# found, skipping any leading non-letter characters (whitespace, "(", a
# stray digit), so a properly-cased reference sentence is what the model
# is actually scored against. topic/subtopic/question/subquestion are
# deliberately NOT touched here: those are frequently not full sentences
# at all (e.g. "b.1. Adultos", a GRADE letter fragment), so capitalizing
# them wouldn't reflect a real casing judgment to score the model on.
def _capitalize_first_letter(t):
    for i, ch in enumerate(t):
        if ch.isalpha():
            return t[:i] + ch.upper() + t[i + 1:]
    return t

# PDF-to-TXT conversion collapses superscript citation numbers onto the end
# of whatever word (or closing parenthesis, e.g. a parenthetical statistic
# like "(p=0,02)42.") they followed (e.g. "...para la vida18 o que..." for
# what was originally "...para la vida¹⁸ o que...", the footnote a
# superscript right after "vida"). A footnote marker is always glued
# directly onto a lowercase word, a ")", a "]" (e.g. a bracketed p-value,
# "p<0,0001]38."), a "%" (e.g. "77%38", "7,1 %50,52"), or a closing curly
# quote "'" (e.g. "'estadio final'26"), with no space, and always
# immediately followed by sentence-ending punctuation, another footnote
# marker chained the same way ("...vida59.60. En..."), or end of
# string/field, never by more letters mid-word. That boundary condition is
# what makes this safely distinguishable from real numeric content: doses,
# percentages, sample sizes, and confidence intervals are never glued onto a
# lowercase word, ")", "]", "%", or "'" with no space in exactly this shape.
#
# Uppercase-letter-preceded digits are handled separately and more
# narrowly: a corpus-wide scan of every uppercase-acronym-plus-digit token
# found this space dominated by real, load-bearing clinical abbreviations
# (DM1, DM2, DPP4, GLP1, HbA1c, HN1, HNF1, IA2, IC95/IC9, I2, MD6/MD12,
# MODY3, OR3, P5/P95, RD1015, SGLT2, ABCD2, ZnT8, vitamina B12), an
# open-ended set an exclusion allowlist can't safely guarantee coverage of.
# Only 6 acronyms in the corpus actually carry footnote numbers (FA, AVK,
# OMS, RT, FOP, ACOD, e.g. "FA36,55,56", "ACOD31,44,67,68", "AVK29",
# "OMS15", "RT40", "FOP63"), confirmed via manual review with no colliding
# real-data use, so those are matched by name (a positive allowlist)
# instead of accepting any uppercase-preceded digit.
_UNIT_SUFFIX_PATTERN = re.compile(r'(?:kg|mg|cm)/m$', re.IGNORECASE)
_MEDIA_SUBSCRIPT_PATTERN = re.compile(r'Media$')
_FOOTNOTE_MARKER_PATTERN = re.compile(
    r"(?<=[a-záéíóúñ)%\]’])(\d{1,3}(?:[,\-–]\d{1,3})*)((?:\.\d{1,3}(?:[,\-–]\d{1,3})*)*)(?=[\s.,;:)]|$)"
)
_FOOTNOTE_ACRONYM_PATTERN = re.compile(
    r"(?<![A-ZÁÉÍÓÚÑa-záéíóúñ])(?P<acronym>FA|AVK|OMS|RT|FOP|ACOD)"
    r"(?P<digits>\d{1,3}(?:[,\-–]\d{1,3})*(?:\.\d{1,3}(?:[,\-–]\d{1,3})*)*)(?=[\s.,;:)]|$)"
)

# Running total of individual footnote NUMBERS actually removed (not
# candidates found: the unit/Media exclusions mean some regex matches are
# left alone; not regex matches either: one match like "29,50,52" or a
# chained "59.60" is a single citation reference but names 3 and 2 distinct
# footnote numbers respectively), so the log can report a true per-number
# count without a second, separate counting pass.
footnote_markers_removed = 0
_FOOTNOTE_NUMBER_SEPARATOR_PATTERN = re.compile(r'[,\-–.]')

def _strip_footnote_markers(t):
    def _count_and_clear(match_text):
        global footnote_markers_removed
        pieces = _FOOTNOTE_NUMBER_SEPARATOR_PATTERN.split(match_text)
        footnote_markers_removed += sum(1 for piece in pieces if piece.isdigit())
        return ""

    def _replace(match):
        prefix = t[: match.start()]
        if _UNIT_SUFFIX_PATTERN.search(prefix) or _MEDIA_SUBSCRIPT_PATTERN.search(prefix):
            return match.group(0)
        return _count_and_clear(match.group(0))

    def _replace_acronym(match):
        _count_and_clear(match.group("digits"))
        return match.group("acronym")

    t = _FOOTNOTE_MARKER_PATTERN.sub(_replace, t)
    t = _FOOTNOTE_ACRONYM_PATTERN.sub(_replace_acronym, t)
    return t

def clean_text(t):
    t = t.strip() #To strip leading and trailing whitespaces.
    # \x07/\x1f are stray non-printable bytes left over from PDF extraction
    # (confirmed in atencion_paliativa.txt, cuidados_paliativos_pediatria.txt,
    # manejo_ictus.txt: 216 occurrences corpus-wide), landing mid-word
    # ("Fase \x1fnal" for "Fase final") or right after a bullet/heading
    # number ("b) \x07¿Cuál es..."). Left in, they surface as visible
    # garbage characters in the final dataset fields.
    t = re.sub(r'[\x07\x1f]', '', t)
    t = re.sub(r'^[\•\●\-\*]+\s*', '', t)
    t = re.sub(r'^\$?\\bullet\$?\s*', '', t)
    t = t.strip() #To apply final whitespace trimming after normalization.
    if footer_pattern.match(t):
        return ""
    return t

# PDF-to-text extraction preserves the source's line-wrap hyphens verbatim
# (e.g. a line ending "...control de los sín-" followed by "tomas?"), which
# later gets space-joined into visible garbage like "sín- tomas" once rows
# are assembled. This resolves each such split BEFORE clean_text/line
# splitting ever separates the two fragments into different list entries,
# so the space-join never happens for genuine splits.
#
# Not every trailing hyphen is a split though, some are genuine compound
# words that happen to fall at a line break (e.g. "coste-" / "efectividad"
# -> "coste-efectividad", not "costeefectividad"). Distinguishing the two:
# whichever of the JOINED form ("síntomas") or the HYPHENATED-COMPOUND form
# ("coste-efectividad") independently appears elsewhere in the SAME
# guidebook, unsplit, is treated as confirmed correct (corpus-frequency).
# Words that never repeat can't be resolved this way, so a Spanish
# clinical-domain tokenizer (BSC's) is used as a fallback for those.
#
# Only the "join" (merge, drop hyphen) decision is applied here, NOT the
# "keep as compound" decision. Manual review of ~146 compound-classified
# candidates found roughly half were wrong: PDF two-column-layout artifacts
# (a margin label like "BPC"/"favor" or a table cell interleaved mid-word,
# e.g. "expectati-" + "adaptada" where the real continuation "vas" appears
# further down, after the interloper) fooled BOTH the corpus-frequency and
# dictionary checks into treating the interloper as a legitimate compound's
# second half. No reliable way to detect that interleaving was found, so
# compounds are left exactly as extracted (visible "word- word" artifact)
# rather than risk silently corrupting them. The "join" bucket does not
# show this problem (spread + random samples of 1772 cases, zero garbling
# found) since genuine line-wrap fragments are typically longer/more
# specific and don't collide with interloper words the way short compound
# stems can.
# [a-záéíóúñA-ZÁÉÍÓÚÑ] (not just lowercase): the word-char class used to
# exclude uppercase entirely, so a split word starting with a capital letter
# (a proper noun, a drug name, anything sentence- or clause-initial, e.g.
# "European So-" / "ciety") silently truncated frag1 by its own leading
# letter, since (.*?) is non-greedy and happily absorbs a capital the char
# class itself can't match, leaving frag1="o" instead of "So". Confirmed via
# guiasalud_203/317/327/718 (this session): "So-ciety", "Fen-tanilo",
# "Quimiotera-pia" were all silently corrupted this way, and their
# corpus-frequency/BSC checks ran against the WRONG fragment as a result,
# hiding real, resolvable joins as apparently-unresolved candidates.
_hyphen_split_pattern = re.compile(r'^(.*?)([a-záéíóúñA-ZÁÉÍÓÚÑ]+)-\s*$')
_hyphen_next_word_pattern = re.compile(r'^\s*([a-záéíóúñA-ZÁÉÍÓÚÑ]+)')

_bsc_tokenizer = None

def _get_bsc_tokenizer():
    global _bsc_tokenizer
    if _bsc_tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            return None
        _bsc_tokenizer = AutoTokenizer.from_pretrained(BSC_TOKENIZER_NAME)
    return _bsc_tokenizer

def _bsc_is_single_token(word):
    """A subword tokenizer always produces SOME segmentation for any input,
    real or garbage (e.g. "asdfqwerty" still splits into pieces), so
    piece-count/length is NOT a reliable validity signal on its own: a
    genuinely wrong concatenation like "costeefectividad" ("coste" +
    "efectividad" wrongly joined) tokenizes into the same shape (2
    real-looking pieces) as a genuinely correct word like "sobrehidratación"
    ("sobre" + "hidratación"). The one signal that IS reliable: whether the
    model's vocabulary contains this EXACT surface form as its own single
    token, which only happens for forms the tokenizer's training corpus saw
    often enough to warrant a dedicated entry. A wrong concatenation like
    "costeefectividad" essentially never earns that (confirmed: 0 false
    "compound" resolutions across the full corpus using this check, vs.
    hunspell's dictionary-based approach producing ~45% wrong compound
    classifications on manual review).
    """
    tok = _get_bsc_tokenizer()
    if tok is None:
        return False
    return len(tok.encode(word, add_special_tokens=False)) == 1

def _merge_hyphenated_line_wraps(raw_lines):
    """Returns (new list of raw lines, number of splits merged), with
    confirmed line-wrap hyphen splits joined into one line (hyphen removed).
    Only touches lines the corpus-frequency or dictionary check confirms as
    a genuine split. Everything else (including likely compounds) is left
    untouched. The count is this function's own tally of the "join"
    decisions it made, i.e. how many hyphen artifacts the automatic
    approach resolved, kept separate from _residual_hyphen_pattern's count
    (§6.4) of what's left for manual curation, so both are visible in the
    log rather than only the unresolved remainder."""
    text = "".join(raw_lines)
    n = len(raw_lines)

    candidates = []  # (line_idx, frag1, next_line_idx, frag2)
    for i, line in enumerate(raw_lines):
        m = _hyphen_split_pattern.match(line.rstrip())
        if not m:
            continue
        frag1 = m.group(2)
        j = i + 1
        # Skip blank lines AND page-footer lines (guidebook running title,
        # generic "GUÍAS DE PRÁCTICA CLÍNICA EN EL SNS", both with an
        # optional page number, same footer_pattern §1 clean_text uses):
        # a word split right at a page boundary has the footer sitting
        # between its two fragments in the raw text (clean_text/footer
        # stripping hasn't run yet at this point in the pipeline), so
        # only skipping blanks left the real continuation word
        # unreachable, silently failing candidates that were otherwise
        # perfectly resolvable (confirmed via guiasalud_212/314/624/641/
        # 695, this session: "comple-jas", "ni-ños", "hos-pital",
        # "en-contrar", "respec-tivamente", all interrupted by a footer
        # line, not by anything ambiguous about the split itself).
        while j < n and (not raw_lines[j].strip() or footer_pattern.match(raw_lines[j])):
            j += 1
        if j >= n:
            continue
        m2 = _hyphen_next_word_pattern.match(raw_lines[j])
        if not m2:
            continue
        frag2 = m2.group(1)
        if not (2 <= len(frag1) <= 20 and 2 <= len(frag2) <= 20):
            continue
        candidates.append((i, frag1, j, frag2))

    decisions = {}  # line_idx -> True (merge) / absent (leave alone)
    bsc_needed = []
    for i, frag1, j, frag2 in candidates:
        joined = frag1 + frag2
        compound = frag1 + "-" + frag2
        joined_present = bool(re.search(
            rf'(?<![a-záéíóúñ]){re.escape(joined)}(?![a-záéíóúñ])', text, re.IGNORECASE))
        compound_present = bool(re.search(
            rf'(?<![a-záéíóúñ-]){re.escape(compound)}(?![a-záéíóúñ])', text, re.IGNORECASE))
        if joined_present and not compound_present:
            decisions[i] = True
        elif compound_present and not joined_present:
            pass  # confirmed compound, leave hyphen as-is
        else:
            bsc_needed.append((i, joined))

    # BSC fallback only ever confirms "join" (see _bsc_is_single_token).
    # It deliberately has no "confirmed compound" outcome, since a wrong
    # concatenation and a genuine split can tokenize into the same shape.
    # Single-token status is the one signal reliable enough to trust, and it
    # only ever points toward the joined form being a known real word.
    for i, joined in bsc_needed:
        if _bsc_is_single_token(joined):
            decisions[i] = True

    if not decisions:
        return raw_lines, 0

    merged = list(raw_lines)
    for i, frag1, j, frag2 in candidates:
        if i not in decisions:
            continue
        # Remove the trailing "frag1-" (and any trailing whitespace) from
        # line i, and strip frag2, plus any punctuation immediately
        # following it with no space (e.g. "tomas?" -> take "tomas?", not
        # just "tomas", so the "?" doesn't end up stranded as its own
        # space-joined fragment downstream, producing "síntomas ?"), off
        # the front of line j. The merged word is written entirely onto
        # line i. Line j keeps whatever followed that punctuation.
        line_i = merged[i]
        cut = line_i.rstrip().rfind(frag1 + "-")
        line_j = merged[j]
        stripped = line_j.lstrip()
        take_len = len(frag2)
        while take_len < len(stripped) and stripped[take_len] in ".,;:?!)]¿¡":
            take_len += 1
        # clean_text() strips whitespace unconditionally, so the exact line
        # ending here doesn't matter for downstream processing.
        merged[i] = line_i[:cut] + frag1 + stripped[:take_len]
        prefix_len = len(line_j) - len(stripped)
        merged[j] = line_j[:prefix_len] + stripped[take_len:]

    return merged, len(decisions)

# A handful of "Contexto" blocks have no "Pregunta"/"Subpregunta"/"Pregunta
# clínica" heading before them at all. The block starts directly at a
# numbered section title instead (e.g. "7.2 Dolor", "7.4.4. Adición de la
# psicoterapia..."). A generic numbered-heading trigger is unsafe to add
# corpus-wide (500-800+ matches per guidebook: citations, list items, GRADE
# sub-bullets, and the same numbering also reappears verbatim in each
# guidebook's own table of contents), so instead of guessing, these 13 known
# locations (found by manually auditing every "Contexto" with no active
# topic-collection state) are mapped explicitly. Keyed by (guidebook
# filename, Nth orphan Contexto encountered in that file, 0-indexed) rather
# than by heading text, since some section numbers (e.g. "7.3.2.1.") also
# appear verbatim in the ToC and are not unique text keys on their own.
#
# Each value is (topic, subtopic). The bare section title alone (e.g.
# "Dolor") is too generic to stand as topic on its own: none of these 13
# sections state their own clinical question, they're subsections of a
# broader question stated elsewhere, in one of three source patterns:
#   - atencion_paliativa.txt's 5 symptom sections (Dolor/Disnea/Náuseas y
#     vómitos/Ansiedad y delirium/Estertores) all share ONE umbrella
#     question from "7. Manejo de síntomas" 's own Pregunta/Contexto block,
#     four pages above the first of them.
#   - manejo_ictus.txt's Dieta/Ejercicios para la disfagia (7.3.2.1/7.3.2.2)
#     share their OWN closer umbrella question, stated once for their
#     direct parent "7.3.2. Plan terapéutico en el domicilio" ("¿Qué plan
#     terapéutico puede realizarse en el domicilio? (Dietas (espesantes),
#     ejercicios…)"), same "adjacent question, no Pregunta label" pattern
#     as the 3 entries below, not "7.3.1. Evaluación de la disfagia en AP"
#     's own, more distant, less specific Pregunta/Contexto question.
#   - manejo_ictus.txt's depresión/labilidad/psicoterapia trio (7.4.1/7.4.3/
#     7.4.4) share "7.4."'s own 3-question preview list. No single one of
#     the 3 questions maps cleanly 1:1 to all 3 subsections (the 2nd
#     question covers antidepressants generically, not split by condition),
#     so the first, broadest question is used for all three.
# The remaining 3 (Intervenciones multidisciplinares/Terapia ocupacional/
# Tratamiento dual antiagregante) each have their OWN adjacent question
# right after the section number, just missing the literal "Pregunta"
# heading label a normal topic block would have, so subtopic is left empty:
# the section title added nothing the question doesn't already state.
ORPHAN_CONTEXTO_TOPICS = {
    ("atencion_paliativa.txt", 0): (
        "Para pacientes en los últimos días de vida, ¿qué fármacos son más "
        "efectivos para aliviar el dolor, la disnea, las náuseas y vómitos, "
        "la ansiedad, el delirium y los estertores?",
        "Dolor",
    ),
    ("atencion_paliativa.txt", 1): (
        "Para pacientes en los últimos días de vida, ¿qué fármacos son más "
        "efectivos para aliviar el dolor, la disnea, las náuseas y vómitos, "
        "la ansiedad, el delirium y los estertores?",
        "Disnea",
    ),
    ("atencion_paliativa.txt", 2): (
        "Para pacientes en los últimos días de vida, ¿qué fármacos son más "
        "efectivos para aliviar el dolor, la disnea, las náuseas y vómitos, "
        "la ansiedad, el delirium y los estertores?",
        "Náuseas y vómitos",
    ),
    ("atencion_paliativa.txt", 3): (
        "Para pacientes en los últimos días de vida, ¿qué fármacos son más "
        "efectivos para aliviar el dolor, la disnea, las náuseas y vómitos, "
        "la ansiedad, el delirium y los estertores?",
        "Ansiedad y delirium",
    ),
    ("atencion_paliativa.txt", 4): (
        "Para pacientes en los últimos días de vida, ¿qué fármacos son más "
        "efectivos para aliviar el dolor, la disnea, las náuseas y vómitos, "
        "la ansiedad, el delirium y los estertores?",
        "Estertores",
    ),
    ("manejo_ictus.txt", 0): (
        "¿Qué plan terapéutico puede realizarse en el domicilio? (Dietas "
        "(espesantes), ejercicios…)",
        "Dieta",
    ),
    ("manejo_ictus.txt", 1): (
        "¿Qué plan terapéutico puede realizarse en el domicilio? (Dietas "
        "(espesantes), ejercicios…)",
        "Ejercicios para la disfagia",
    ),
    ("manejo_ictus.txt", 2): (
        "¿Deben tratarse farmacológicamente la depresión, ansiedad y "
        "labilidad emocional tras un ictus?",
        "Tratamiento farmacológico de la depresión post-ictus",
    ),
    ("manejo_ictus.txt", 3): (
        "¿Deben tratarse farmacológicamente la depresión, ansiedad y "
        "labilidad emocional tras un ictus?",
        "Tratamiento farmacológico de la labilidad emocional",
    ),
    ("manejo_ictus.txt", 4): (
        "¿Deben tratarse farmacológicamente la depresión, ansiedad y "
        "labilidad emocional tras un ictus?",
        "Adición de la psicoterapia al tratamiento farmacológico de la depresión y la ansiedad",
    ),
    ("manejo_ictus.txt", 5): (
        "¿Son eficaces las intervenciones multidisciplinares (fisioterapia "
        "junto con terapia ocupacional, logopedia, etc.) en la mejoría de "
        "la independencia para las AVD en personas que han sufrido un ictus?",
        "",
    ),
    ("manejo_ictus.txt", 6): (
        "¿Es eficaz la terapia ocupacional en la mejoría de la "
        "independencia para las AVD en personas que han sufrido un ictus?",
        "",
    ),
    ("manejo_ictus.txt", 7): (
        "Aquellas personas que han sufrido un ictus isquémico leve o un AIT "
        "no cardioembólico, que no son candidatos a trombólisis y que "
        "reciben tratamiento dual antiagregante, ¿cuánto tiempo deberían "
        "estar recibiendo dicho tratamiento?",
        "",
    ),
}

# A guidebook's LAST GRADE letter in a block (whichever letter, k), l),
# etc., happens to be final for that particular question) has no lettered
# bullet after it to stop judgement/evidence/considerations accumulation.
# Normally the next topic/subpregunta/pregunta-clinica/Contexto heading
# closes it instead (flush_pending_row fires there too), but in these 11
# known, hand-verified spots the block is immediately followed by an
# unrelated TOP-LEVEL numbered document section ("5. Tratamiento
# psicológico", "11. Plan de actualización", etc.) with no heading marker of
# any kind recognized elsewhere in this script. A generic "stop at any
# numbered heading" rule is unsafe corpus-wide (500+ false positives per
# guidebook: numbered recommendation lists, numbered questions, ToC entries
# all start with digits), so, same approach as ORPHAN_CONTEXTO_TOPICS,
# these are the exact, verified section-title strings to stop at, checked
# only while a field is actively accumulating.
SECTION_BOUNDARY_TITLES = {
    "ansiedad.txt": [
        "5. Tratamiento psicológico",
        "6. Tratamiento farmacológico",
        "7. Otros tratamientos",
        "8. Derivación a atención especializada",
        "9. Estrategias diagnósticas y terapéuticas",
        "10. Difusión e implementación",
        "11. Líneas de investigación futura",
        "12. Plan de actualización",
        "5.1. Terapia cognitivo conductual",
        "5.2. Terapia de relajación",
        "5.3. Terapia metacognitiva",
        "5.4. Terapia de aceptación",
        "5.5. Terapia psicodinámica",
        "5.6. Mindfulness",
        "6.1. Antidepresivos",
        "6.1.1. Inhibidores selectivos de la recaptación de",
        "6.1.2. Inhibidores de la recaptación de serotonina y",
        "6.1.3. Otros antidepresivos",
        "6.1.3.1. Agomelatina",
        "6.1.3.2. Vortioxetina",
        "6.2. Ansiolíticos",
        "6.2.1. Benzodiacepinas",
        "6.2.2. Otros ansiolíticos",
        "6.2.2.1. Buspirona",
        "6.3. Otros fármacos",
        "6.3.1. Pregabalina",
        "6.3.2. Quetiapina",
        "6.3.3. Opipramol",
        "7.1. Programas de ejercicio físico",
        "7.2. Hierbas medicinales",
        "7.2.2. Pasiflora",
        "7.2.3. Valeriana",
        "7.2.4. Galphimia",
        "7.2.5. Kava",
        "7.2.6. Silexan",
        "7.2.7. Granulado de hierbas Jiu Wei Zhen Xin",
        "7.2.8. Medicina herbal oriental",
        "7.2.8.1. Medicina herbal oriental como tratamiento único",
        "7.2.8.2. Medicina herbal oriental combinada con ansiolíticos",
    ],
    "diabetes.txt": [
        "5. Manejo glucémico",
        "6. Educación terapéutica en diabetes",
        "7. Alimentación",
        "8. Manejo de la DM1 en situaciones",
        "9. Difusión e implementación",
        "10. Líneas de investigación futura",
        "11. Plan de actualización",
        "4.2. Diagnóstico diferencial de diabetes monogénica",
        "4.3.2. Enfermedad celíaca",
        "4.3.3. Gastritis atrófica autoinmune",
        "5.2. Monitorización de glucosa",
        "5.2.2. Sistemas de monitorización Flash (intermitente) de",
        "5.3.2. Múltiples dosis de inyecciones (MDI) +",
        "5.3.3. Sistemas híbridos de asa cerrada",
        "5.3.4. Reutilización de agujas en personas con DM1",
        "5.4.2. Inhibidores de dipeptidil peptidasa-4 (iDPP-4)",
        "5.4.3. Metformina",
        "5.4.4. Inhibidores del cotransportador de sodio-glucosa",
        "6.2. Apoyo comunitario",
        "7.2. Recuento de hidratos de carbono",
        "8.2.2. Intervención educativa",
        "8.2.3. Determinaciones de laboratorio",
        "8.2.4. Otras medidas",
        "8.3.2. Programa de manejo de citas",
        "8.4. Cuidados preconcepcionales",
        "8.5. Manejo de complicaciones en el embarazo",
    ],
    "manejo_ictus.txt": [
        "6.\t Manejo del ictus comunicado",
        "7.2.\t Dolor central post-ictus",
        "7.3.2.2.  Ejercicios para la disfagia",
        "7.4.1.\t Tratamiento farmacológico de la depresión post-ictus",
        "7.4.4.\t Adición de la psicoterapia al tratamiento farmacológico de",
        "7.6.\t Terapia ocupacional",
        "7.7.\t Tratamiento dual antiagregante",
        "8.\t Estrategias diagnósticas y terapéuticas",
        "9.\t Difusión e implementación",
        "10.\tLíneas de investigación futura",
        "11.\tPlan de actualización",
        "11. Plan de actualización",
        "12.\tAnexos",
        "13.\tBibliografía",
    ],
    "prevencion_secundaria_ictus.txt": [
        "8. Difusión e implementación",
        "9. Líneas de investigación futura",
        "10. Plan de actualización",
    ],
    # Previously had NO entries at all (unlike every other guidebook), so
    # every internal section boundary was unguarded. Confirmed via
    # guiasalud_310 (this session): its considerations field ran to 246,958
    # characters, all the way to the bibliography, because the "l) Otras
    # consideraciones" catch-all item (see manejo_ictus.txt's own comment on
    # this same label pattern) has no Juicio:/Evidencia: to trigger a real
    # flush, and NOTHING in this guidebook's boundary list existed to stop
    # it early. Same root cause found in 10 more atencion_paliativa.txt rows
    # at smaller scale (a few characters to ~5.4KB each), wherever the
    # runaway happened to hit a section title sooner. This is the
    # guidebook's full top-level table of contents (§3-§13), verified each
    # title's real body occurrence is unique corpus-wide (34 raw-line
    # matches total: exactly 17 ToC + 17 real headings, zero unexpected
    # hits against ordinary body text, including the guidebook's own
    # "Dolor"/"Disnea"/"Náuseas y vómitos"/etc. subtopic values, which never
    # appear at a raw line's start with this exact "N.M.\t" numbered prefix).
    "atencion_paliativa.txt": [
        "3\t Reconocimiento de",
        "4\t Comunicación e información",
        "5\t Toma de decisiones compartida y",
        "6\t Hidratación",
        "7\t Manejo de síntomas",
        "7.1\t Consideraciones generales",
        "7.2\t Dolor",
        "7.3\t Disnea",
        "7.4\t Náuseas y vómitos",
        "7.5\t Ansiedad y delirium",
        "7.6\t Estertores",
        "8\t Sedación paliativa",
        "9\t Estrategias diagnósticas",
        "10\t Metodología",
        "11\t Difusión e implementación",
        "12\t Líneas de investigación futura",
        "13\t Plan de actualización",
    ],
    "cuidados_paliativos_pediatria.txt": [
        "4.\t Quimioterapia y radioterapia",
        "5. \tParticipación del menor en la toma",
        "11. Difusión e implementación",
        "12. Líneas de investigación futura",
    ],
}
_section_boundary_patterns = {
    gb: [re.compile(r'^' + re.escape(title)) for title in titles]
    for gb, titles in SECTION_BOUNDARY_TITLES.items()
}

def _is_section_boundary(filename, text):
    return any(p.match(text) for p in _section_boundary_patterns.get(filename, []))

"""## 2. Text segmentation

Regular expressions are used to detect structural elements in the text, including:

- topic
- subtopic
- question
- focus
- judgement
- evidence
- considerations

These patterns drive the rule-based extraction process.
"""

pregunta_pattern = re.compile(r'^[a-z](?:\)|\.(?!\d))')
# "(?:\)|\.(?!\d))" also accepts a bare period after the letter (e.g. "a. ¿Es
# prioritario el problema?"), not just the standard close-paren "a)": one
# GRADE-criteria block in diabetes.txt (the "manejo de las complicaciones...
# en el embarazo" topic) uses period-form lettering throughout, confirmed as
# the ONLY such block corpus-wide. The negative lookahead "(?!\d)" excludes
# subcomparacion_pattern's own "a.1."/"b.2."-style labels (digit right after
# the dot), which would otherwise collide with this pattern and be wrongly
# treated as GRADE-criteria boundaries instead of sub-comparison labels.
# Widening this DOES introduce real false positives corpus-wide (30 lines,
# mostly atencion_paliativa.txt's own unrelated lettered instruction lists,
# e.g. "b. Lavar la vía con suero..."), all confirmed suppressed by the
# existing _is_real_pregunta lookahead guard (requires a nearby "Juicio:"),
# verified corpus-wide: 0 spurious rows in any guidebook other than
# diabetes.txt, whose row count is the only one that changed.
# Case-sensitive, whole-line: the case-insensitive prefix-only version also
# matched lowercase "contexto" at the start of an ordinary line-wrapped
# sentence (e.g. "...en el / contexto del estudio PROSPER..."), which, when
# it happened to fall during an active topic/subquestion collection,
# prematurely closed the buffer partway through, truncating real content
# (confirmed in 3 places in diabetes.txt).
contexto_pattern = re.compile(r'^Contexto\s*$')

juicio_pattern = re.compile(r'Juicio:\s*', re.IGNORECASE)
# [\d,\-]* tolerates citation-number suffixes between "investigación" and the
# colon (e.g. "investigación29,44,45:", "investigación47:", "investigación29-31:"),
# which the original bare pattern missed, leaving mode stuck on "judgement" and
# the entire evidence paragraph silently absorbed into the judgement field.
# "(?:procedente )?de (?:la )?" tolerates manejo_ictus.txt's own one-off
# "Evidencia de investigación:" wording (missing "procedente"/"la"), confirmed
# as the ONLY occurrence of that shortened phrasing across all six guidebooks
# (corpus-wide check found exactly 2 distinct phrasings total). Without this,
# guiasalud_890's judgement field silently absorbed the whole c) criterion's
# evidence paragraph, the same failure mode as the citation-suffix case above.
evidencia_pattern = re.compile(r'Evidencia (?:procedente )?de (?:la )?investigación[\d,\-]*:\s*', re.IGNORECASE)
# Unanchored (search, not match), like juicio_pattern/evidencia_pattern
# above: atencion_paliativa.txt's own bullet character is an en-dash "–",
# not one of the "•"/"●"/"-"/"*" chars clean_text's bullet-strip regex
# covers, so it survives at the front of the line untouched. An anchored
# ^-only pattern silently never matched any of the 91 real occurrences in
# that guidebook, and their content was absorbed as plain evidence text
# instead. Confirmed corpus-wide that this phrase never appears preceded by
# real prose (only by bullet-type characters), so unanchoring introduces no
# new false positives.
consideraciones_pattern = re.compile(
    r'(?:Consideraciones adicionales|Información adicional):\s*',
    re.IGNORECASE
)

# Anchored at both ends: only a line that IS ENTIRELY "Pregunta(s)"/"Pregunta
# para/a responder" (optional trailing colon) triggers topic collection. The
# original prefix-only pattern matched any body sentence that merely started
# with the word "pregunta" (e.g. "pregunta de investigación..."), which opened
# collecting_tema on ordinary prose and silently swallowed everything up to the
# next "Contexto", in one diabetes.txt case, an entire multi-question block.
# Deliberately NOT "Preguntas para/a responder" (plural + suffix): every
# guidebook uses that exact phrase once, always as a document-overview /
# priority-questions summary heading with no Contexto for 450-1300+ lines
# (confirmed corpus-wide). Letting it trigger reproduces the same
# runaway-buffer bug this pattern exists to prevent. Bare "Pregunta"/
# "Preguntas" (no suffix) and singular "Pregunta para/a responder" (with
# suffix) are the real per-question headings and all close within a handful
# of lines everywhere they appear.
tema_start_pattern = re.compile(
    r'^(?:Pregunta(?: para responder| a responder)?|Preguntas)\s*:?\s*$',
    re.IGNORECASE
)
# Replaces the old subpregunta_pattern (r'^[a-z]\.\d+\.'), which across the
# whole corpus matched ZERO genuine subquestions and only ever matched internal
# sub-headings inside a single evidence block in diabetes.txt (e.g. "b.1.
# Adultos", "b.2. Embarazadas"), wrongly splitting one record into several
# truncated ones. The genuine subquestion marker is "Subpregunta para
# responder:", matched here the same anchored, whole-line way as
# tema_start_pattern so it never fires on the unrelated "Subpregunta 1.1. ..."
# table-of-contents/summary-list lines seen in prevencion_secundaria_ictus.txt.
# "Subpregunta N.M." alone on its own line (no "para responder", no inline
# text after the period) is a second genuine-subquestion style, used only in
# prevencion_secundaria_ictus.txt. Matched case-sensitively (no IGNORECASE):
# the same phrase also appears lowercase mid-sentence in cross-references
# ("...ver información adicional en la subpregunta 1.1."), which line-wraps
# so "subpregunta 1.1." lands alone on its own line too. Case is the only
# thing distinguishing those 4 false positives from the 11 real headings.
subpregunta_start_pattern = re.compile(
    r'^(?:Subpregunta para responder\s*:?\s*|Subpregunta \d+\.\d+\.\s*)$'
)

# prevencion_secundaria_ictus.txt's umbrella heading ("Pregunta clínica nº X")
# has no Contexto of its own. Its content is delegated entirely to child
# "Subpregunta N.M." blocks above (questions 1-3), or in one case (question
# 4) to no child at all. It still carries real topic content though: a
# bullet summary + explanatory paragraph between the heading and the first
# inline subquestion-preview line ("Subpregunta N.M. <text>", as opposed to
# the real per-subquestion trigger above which is alone on its line). Matched
# case-sensitively so the ALL-CAPS "PREGUNTA CLÍNICA Nº X" occurrences (which
# only appear inside an unrelated table-of-contents block, each with no
# Contexto for 1000+ lines) never trigger this.
pregunta_clinica_start_pattern = re.compile(r'^Pregunta clínica nº \d+\s*$')
pregunta_clinica_end_pattern = re.compile(
    r'^(?:Subpregunta \d+\.\d+\.\s+\S|Pregunta clínica nº \d+\s*$)'
)

# Within one topic, some guidebooks split the a)-k) GRADE-criteria block by
# specific intervention, e.g. "Para el cribado de disfagia por la prueba de
# deglución de agua" vs. "Para la herramienta de autocumplimentación EAT-10"
# under the same "¿Cómo debe realizarse la evaluación de la disfagia...?"
# topic (manejo_ictus.txt). Without capturing this, rows for different
# interventions share the same (topic, subtopic, question, focus) dedup key
# and get wrongly collapsed into one. "Para ..." also appears constantly as
# ordinary prose (43 of 47 corpus-wide occurrences), so it is only treated as
# an intervention heading when the very next line is a "a)" GRADE-criteria
# bullet, the one shape real headings have and body prose never does.
intervencion_pattern = re.compile(r'^Para (?:el|la|los|las) .+$')

# manejo_ictus.txt's "Tratamiento dual antiagregante" topic uses a second,
# unrelated intervencion-heading style for its own two branches: a bare
# drug-combination name ("Clopidogrel y AAS", "Ticagrelor y AAS") with no
# "Para..." prefix at all. Confirmed exactly these 2 occurrences corpus-wide
# (both manejo_ictus.txt only), matched via the same next-line-is-GRADE-letter
# lookahead as intervencion_pattern, so this can't over-match unrelated prose.
_drug_combo_intervencion_pattern = re.compile(
    r'^(?:Clopidogrel y AAS|Ticagrelor y AAS)$'
)

# diabetes.txt (only) further splits some GRADE letters into distinct
# sub-comparisons, labeled "b.1.", "b.2.", etc. (e.g. "b.1. Adultos" vs
# "b.2. Adultos con riesgo de hipoglucemia grave" under one "b) ¿Cuál es la
# magnitud...?" item), each with its own Juicio:/Evidencia:/Consideraciones,
# genuinely distinct findings, not internal noise (verified against source:
# 53/53 occurrences, all diabetes.txt, all close cleanly within 5-82 lines at
# the next "letter.N+1." or the next lettered "x)" bullet). This replaces the
# earlier, broader subpregunta_pattern (r'^[a-z]\.\d+\.') that was removed
# from the row-boundary role entirely: that version fired mid-block on the
# SAME text seen here without the immediately-following-bullet requirement
# below, wrongly splitting single records apart. Only treated as a real
# sub-comparison label when the very next non-empty line is NOT itself
# another sub-comparison/bullet (i.e. is prose, the label itself, possibly
# wrapped onto one more line before the "•"/"●" bullet content starts).
subcomparacion_pattern = re.compile(r'^[a-z]\.\d+\.\s+\S')

# Several topics (diabetes.txt: 7 occurrences, prevencion_secundaria_ictus.txt:
# 3, there closing collecting_pregunta_clinica instead of collecting_tema)
# state their umbrella question, then explicitly preview their own
# refinement questions with a bare bulleted list before the section title
# and the first real "Subpregunta para responder:" trigger. Wording for the
# narration sentence itself varies more than a single prefix can cover:
# most start "Esta pregunta se dividió en..." (still matched as a prefix
# check elsewhere for the wrapped-continuation case), but
# prevencion_secundaria_ictus.txt's "Pregunta clínica nº 3" instead reads
# "Sobre la base de la patología valvular especificada, la pregunta de
# investigación general se dividió en 3 subpreguntas..." (the trigger phrase
# "se dividió en" only appears line-wrapped onto the SECOND line of that
# sentence, with "pregunta" on the line before it). This pattern is used
# unanchored (search, not match) specifically so it also catches that case:
# either "pregunta" appears anywhere before "se dividió en" on the SAME
# line, or "se dividió en" is directly followed by a digit count and
# "subpreguntas" (unambiguous even without "pregunta" on the same line).
# Confirmed corpus-wide zero false positives (the one unrelated corpus hit,
# cuidados_paliativos_pediatria.txt's "Este rango se dividió en diferen-",
# about a medication dosage range, matches neither alternative). The
# narration sentence itself is never kept in topic (see
# _strip_dividida_narration below): topic collection must still stop
# right before the following bulleted preview list. That list's items
# become their own subquestion values via "Subpregunta para responder:"
# further down, not part of topic. Left unhandled entirely, the topic
# buffer would keep absorbing that whole preview list and the section title
# too, since only "Subpregunta para responder:" (much further down)
# eventually closes it.
pregunta_dividida_pattern = re.compile(
    r'pregunta.*se dividió en|se dividió en \d+ subpreguntas',
    re.IGNORECASE
)

# "a)"/"b)"/... also matches plenty of ordinary lettered lists that are NOT
# GRADE-criteria row boundaries. Narrative sub-headings ("a) Magnitud de
# los efectos"), quality-indicator formulas ("a) N.º de personas con..."),
# evidence-summary sub-items ("a) ¿Cuál es la evidencia identificada?"). 107
# such false positives corpus-wide wrongly reset current_subpregunta (and
# spuriously flush a row) before the real GRADE block starts. Concretely,
# this caused every real per-subquestion row in prevencion_secundaria_ictus's
# first umbrella question to lose its subquestion tag and collide together
# as false duplicates. A real GRADE item always has "Juicio:" within the
# next couple of lines (confirmed: 1175/1176 within 1-6 lines, corpus-wide).
# The false positives never do. Stops looking as soon as another "a)"-style
# bullet appears first, so it can't leak into a neighboring real item's own
# "Juicio:".
def _is_real_pregunta(texts, idx, lookahead=8):
    for j in range(idx, min(idx + lookahead, len(texts))):
        if j > idx and pregunta_pattern.match(texts[j]):
            return False
        if juicio_pattern.search(texts[j]):
            return True
    return False

"""## 3. Dataframe creation

The text is processed line by line and segmented into structured QA instances. Segmentation logic relies on local patterns such as:

- "Juicio:"
- "Evidencia procedente de la investigación:"
- "a)", "a.1."

These markers typically appear at the beginning of lines in the TXT files Hence, per-line processing makes it easier to:

- detect section boundaries
- switch modes (e.g., judgement, evidence)
- accumulate text until the next trigger appears


Each instance is stored as a row with fields for topic, question, answers, and metadata.
"""

rows = []
hyphen_wraps_merged = 0

for filepath in files:

    filename = filepath.name

    with open(filepath, encoding="utf8") as f:
        raw_lines = [line for line in f if line.strip()]
    raw_lines, _merged_count = _merge_hyphenated_line_wraps(raw_lines)
    hyphen_wraps_merged += _merged_count
    texts = []
    bullet_flags = []
    for line in raw_lines:
        t = clean_text(line)
        if t:
            texts.append(t)
            bullet_flags.append(bool(_bullet_prefix_pattern.match(line.strip())))

    current_tema = ""
    current_pregunta = ""
    current_subpregunta = ""
    current_intervencion = ""
    current_subcomparacion_full = ""
    pending_intervencion = None
    orphan_contexto_count = 0

    juicio = []
    evidencia = []
    consideraciones = []

    mode = None
    collecting_tema = False
    collecting_subpregunta = False
    collecting_pregunta_clinica = False
    dividida_seen = False
    # Persists across the topic's own Subpregunta/GRADE-letter cycles (does
    # NOT get cleared by subpregunta_start_pattern, only by a genuinely
    # new topic/pregunta-clinica/orphan-Contexto starting): tracks whether
    # the CURRENT topic was explicitly split into refinement questions
    # ("Esta pregunta se dividió en..."). When it was, the refinement
    # (current_subpregunta) is a subtopic under the umbrella topic, not
    # itself the row's question. The GRADE letter (current_pregunta) is
    # always the question, regardless of whether the topic was divided.
    topic_was_dividida = False
    tema_buffer = []
    subpregunta_buffer = []
    pregunta_clinica_buffer = []
    pregunta_clinica_dividida_seen = False
    mode_before_subcomp = None
    # Some GRADE-letter blocks (diabetes.txt only, e.g. "b) ... / Juicio:
    # Moderada." followed by a b.1./b.2. sub-comparison split) state the
    # judgement ONCE, before the split, and never repeat it per sub-group:
    # the source genuinely intends one shared judgement for every
    # sub-comparison row under that letter, it is not a per-sub-group
    # field there. Tracks the last non-empty judgement seen for the
    # CURRENT GRADE letter, survives subcomparacion_pattern boundaries
    # (which do not reset current_pregunta either), and is cleared
    # whenever current_pregunta itself resets (a genuine new GRADE
    # letter/topic/subpregunta/pregunta-clinica boundary), so it can never
    # leak into an unrelated letter's rows.
    last_juicio_for_pregunta = []

    # A row is only complete once a GRADE-criteria item's "Juicio:"/
    # "Evidencia procedente..."/"Consideraciones adicionales:" content has
    # been accumulated. That content is flushed as a row when the NEXT
    # "a)"-style bullet starts (see the "pregunta" branch below), or at
    # end-of-file. But the last item in a block ("l) ¿Es factible...")  is
    # often followed not by another bullet but by a new
    # topic/subpregunta/pregunta-clinica heading. Every such heading must
    # flush the pending row too, or that last item's content silently gets
    # absorbed into the FIRST row of the next block instead (with that next
    # block's topic/subquestion wrongly attached to it).
    #
    # current_pregunta/mode always belong to the item just flushed, not to
    # whatever comes next, so they always reset. current_subpregunta is
    # different: it's shared by EVERY "a)"-"l)" row within one subquestion
    # block, so the "pregunta" branch (fired once per bullet, i.e. constantly
    # within a single subquestion) must NOT clear it, only a genuine new
    # subpregunta/tema/pregunta-clinica/orphan-Contexto boundary should
    # (reset_subpregunta=True from those 4 call sites only).
    #
    # This function can't use `nonlocal` (it's defined in top-level script
    # scope, inside a `for` loop, not inside another `def`), so every call
    # site must assign its four return values back to current_pregunta/
    # current_subpregunta/mode/current_subcomparacion_full itself,
    # immediately after calling it and before changing any other
    # current_*/collecting_* state.
    #
    # Four columns, filled from four independent pieces of state:
    #   topic       -- current_tema (the umbrella question)
    #   subtopic    -- current_intervencion, or current_subpregunta when
    #                  topic_was_dividida (the more specific unit the
    #                  umbrella topic was split or branched into)
    #   question    -- current_pregunta (the GRADE letter, always)
    #   subquestion -- current_subcomparacion_full when a b.1./b.2.-style
    #                  sub-comparison is active, else current_subpregunta
    #                  when NOT topic_was_dividida (its normal role)
    # A row can have both a subtopic AND a subquestion at once (e.g. a
    # dividida topic whose refinement also has b.1./b.2. sub-comparisons):
    # each of the four fields is independent, so no swap or priority logic
    # between them is needed. reset_pregunta defaults to True since most
    # boundaries (topic/subpregunta/pregunta-clinica/orphan-Contexto) start
    # a genuinely new GRADE letter. Only the subcomparacion_pattern branch
    # passes reset_pregunta=False, since b.1./b.2. share the SAME parent
    # GRADE letter and current_pregunta must survive that boundary intact.
    def flush_pending_row(reset_subpregunta, reset_pregunta=True):
        if juicio or evidencia or consideraciones:
            if topic_was_dividida and current_subpregunta:
                row_subtopic = current_subpregunta
            else:
                row_subtopic = current_intervencion
            if current_subcomparacion_full:
                row_subquestion = current_subcomparacion_full
            elif not (topic_was_dividida and current_subpregunta):
                row_subquestion = current_subpregunta
            else:
                row_subquestion = ""
            # A sub-comparison row (current_subcomparacion_full set) with no
            # judgement of its own inherits the last judgement seen for this
            # SAME GRADE letter (see last_juicio_for_pregunta above): the
            # source states it once before the b.1./b.2. split and intends
            # it to cover every sub-group under that letter. A row with its
            # OWN judgement is never touched, this only fills a genuine gap.
            row_juicio = juicio if juicio else (
                last_juicio_for_pregunta if current_subcomparacion_full else juicio
            )
            rows.append({
                "guidebook": filename,
                "topic": _strip_dividida_narration(current_tema),
                "subtopic": row_subtopic,
                "question": current_pregunta,
                "focus": row_subquestion,
                "judgement": _capitalize_first_letter(_strip_footnote_markers(" ".join(row_juicio))),
                "evidence": _capitalize_first_letter(_strip_footnote_markers(" ".join(evidencia))),
                "considerations": _capitalize_first_letter(_strip_footnote_markers(" ".join(consideraciones)))
            })
            if juicio:
                last_juicio_for_pregunta.clear()
                last_juicio_for_pregunta.extend(juicio)
            juicio.clear()
            evidencia.clear()
            consideraciones.clear()
        if reset_pregunta:
            last_juicio_for_pregunta.clear()
        return (
            ("" if reset_pregunta else current_pregunta),
            ("" if reset_subpregunta else current_subpregunta),
            None, ""
        )

    for idx, text in enumerate(texts):

        # -------- intervencion heading (lookahead) --------
        # Only a candidate if the very next line is a "a)" GRADE-criteria
        # bullet, confirmed once that bullet is actually reached below.
        if intervencion_pattern.match(text) or _drug_combo_intervencion_pattern.match(text):
            nxt = texts[idx + 1] if idx + 1 < len(texts) else ""
            if pregunta_pattern.match(nxt):
                pending_intervencion = text
                continue

        # -------- pregunta clinica (umbrella) start --------
        if pregunta_clinica_start_pattern.match(text):
            current_pregunta, current_subpregunta, mode, current_subcomparacion_full = flush_pending_row(reset_subpregunta=True)
            collecting_pregunta_clinica = True
            pregunta_clinica_buffer = []
            pregunta_clinica_dividida_seen = False
            continue

        # -------- pregunta clinica (umbrella) end --------
        # The terminator line (an inline subquestion-preview, or the next
        # umbrella heading with no preview list at all) is not itself part of
        # the topic, but still needs to be re-evaluated against the patterns
        # below (it may be a real trigger in its own right), so this only
        # closes the buffer and falls through instead of using `continue`.
        if collecting_pregunta_clinica and pregunta_clinica_end_pattern.match(text):
            collecting_pregunta_clinica = False
            current_tema = " ".join(pregunta_clinica_buffer).strip()
            current_intervencion = ""
            topic_was_dividida = pregunta_clinica_dividida_seen

        # Same dividida narration as §2's pregunta_dividida_pattern, here as
        # the umbrella (Pregunta clínica) closing sentence rather than
        # inside collecting_tema. Still appended to the buffer, same
        # reasoning as §2's comment: it line-wraps with no reliable
        # single-line end marker, so it's removed at flush time by
        # _strip_dividida_narration once joined (prevencion_secundaria_
        # ictus.txt's Pregunta clínica nº 1/nº 2/nº 3, each with different
        # wording, see pregunta_dividida_pattern's own comment).
        if collecting_pregunta_clinica and pregunta_dividida_pattern.search(text):
            pregunta_clinica_dividida_seen = True
            pregunta_clinica_buffer.append(text)
            continue

        if collecting_pregunta_clinica:
            pregunta_clinica_buffer.append(text)
            continue

        # -------- subpregunta start --------
        # A genuine subquestion also closes any still-open topic collection:
        # "Pregunta para responder:" for the umbrella question is sometimes
        # followed directly by "Subpregunta para responder:" for its first
        # refinement with no "Contexto" line in between (the umbrella question
        # has no Contexto of its own in that case).
        if subpregunta_start_pattern.match(text):
            current_pregunta, current_subpregunta, mode, current_subcomparacion_full = flush_pending_row(reset_subpregunta=True)
            if collecting_tema:
                collecting_tema = False
                current_tema = " ".join(tema_buffer).strip()
            collecting_subpregunta = True
            subpregunta_buffer = []
            continue

        # -------- subpregunta end --------
        if collecting_subpregunta and contexto_pattern.match(text):
            collecting_subpregunta = False
            current_subpregunta = " ".join(subpregunta_buffer).strip()
            continue

        if collecting_subpregunta:
            subpregunta_buffer.append(text)
            continue

        # -------- topic start --------
        if tema_start_pattern.match(text):
            current_pregunta, current_subpregunta, mode, current_subcomparacion_full = flush_pending_row(reset_subpregunta=True)
            collecting_tema = True
            tema_buffer = []
            dividida_seen = False
            topic_was_dividida = False
            current_intervencion = ""
            continue

        # -------- topic end --------
        if collecting_tema and contexto_pattern.match(text):
            collecting_tema = False
            current_tema = " ".join(tema_buffer).strip()
            continue

        # See pregunta_dividida_pattern definition (§2). Its narration
        # sentence is used only as a signal that the topic was subdivided
        # (topic_was_dividida drives the subtopic routing in
        # flush_pending_row), NOT kept as part of topic itself: now that
        # subtopic is its own column, the narration text adds nothing the
        # model needs and left in, it dangles mid-prompt with no framing.
        # Still appended to tema_buffer here (unlike a simple skip) because
        # the sentence line-wraps across several raw lines with no reliable
        # single-line end marker (some guidebooks end it "...:", others
        # "....", followed immediately by the GRADE Pro sentence in some
        # cases); it's removed in one pass by _strip_dividida_narration
        # once the whole buffer is joined, at flush time, since in every
        # corpus occurrence this sentence (plus the optional GRADE Pro one)
        # is the LAST thing in the buffer before the real cutoff below.
        # Topic collection must still stop right before the bulleted preview
        # list that follows it. dividida_seen marks that the trigger line has
        # been seen, so the NEXT bulleted line (not the trigger line itself)
        # is the real cutoff point.
        if collecting_tema and dividida_seen and bullet_flags[idx]:
            collecting_tema = False
            current_tema = " ".join(tema_buffer).strip()
            dividida_seen = False
            topic_was_dividida = True
            continue

        if collecting_tema and pregunta_dividida_pattern.search(text):
            tema_buffer.append(text)
            dividida_seen = True
            continue

        if collecting_tema:
            tema_buffer.append(text)
            continue

        # -------- orphan Contexto --------
        # No active topic/subpregunta/pregunta-clinica collection to close:
        # this "Contexto" starts directly at a bare numbered section title
        # instead of a proper heading. Rather than silently keep whatever
        # topic was last set (wrong, confirmed to mislabel unrelated
        # questions, e.g. a "Disnea" block inheriting the previous "Dolor"
        # topic), the topic resets to unknown, falling back to the verified
        # ORPHAN_CONTEXTO_TOPICS lookup where available. current_intervencion
        # carries the subtopic half of that lookup (the section title, or
        # empty when the topic half is already specific enough on its own),
        # routed into the subtopic column via flush_pending_row the same way
        # a real "Para..." intervencion heading would be.
        if contexto_pattern.match(text):
            current_pregunta, current_subpregunta, mode, current_subcomparacion_full = flush_pending_row(reset_subpregunta=True)
            key = (filename, orphan_contexto_count)
            current_tema, current_intervencion = ORPHAN_CONTEXTO_TOPICS.get(key, ("", ""))
            topic_was_dividida = False
            orphan_contexto_count += 1
            continue

        # -------- pregunta --------
        if pregunta_pattern.match(text) and _is_real_pregunta(texts, idx):

            if pending_intervencion is not None:
                current_intervencion = pending_intervencion
                pending_intervencion = None

            current_pregunta, current_subpregunta, mode, current_subcomparacion_full = flush_pending_row(reset_subpregunta=False)

            current_pregunta = text
            mode = "question"

            continue

        # -------- sub-comparison label (diabetes.txt only) --------
        # A new "letter.N." label closes whatever sub-comparison came before
        # it (flushing that row) and opens the next one under the SAME
        # parent GRADE letter. current_pregunta/current_subpregunta must
        # NOT reset here (reset_subpregunta=False, reset_pregunta=False):
        # b.1./b.2./etc. all share the SAME parent GRADE letter, so
        # current_pregunta must survive this boundary intact, unlike a real
        # topic/subpregunta/pregunta-clinica boundary. mode_before_subcomp
        # captures what mode was doing right before this trigger (e.g.
        # "evidence"): some guidebook blocks restart with a fresh "b.2. "
        # label straight into MORE evidence prose for the new sub-group,
        # with no restated "Evidencia procedente..." trigger at all (see
        # the wrapped-continuation check below), so mode needs to be
        # restorable to whatever it was, not just reset to None.
        if subcomparacion_pattern.match(text):
            mode_before_subcomp = mode
            current_pregunta, current_subpregunta, mode, current_subcomparacion_full = flush_pending_row(
                reset_subpregunta=False, reset_pregunta=False
            )
            current_subcomparacion_full = text
            mode = "subcomparacion_label"
            continue

        # A wrapped continuation of the label itself (see intervencion's own
        # "Adición de la psicoterapia..." case for the same PDF-extraction
        # artifact): only while still in subcomparacion_label mode, before
        # any Juicio:/Evidencia:/Consideraciones: content has started. Once
        # one of those triggers is reached, mode must fall through to their
        # own branches below instead of being swallowed as more label text.
        #
        # Requires the continuation line to start lowercase: confirmed
        # corpus-wide (diabetes.txt) that every GENUINE wrapped label (e.g.
        # "b.1. Recuento de hidratos de carbono frente a consejos
        # nutricionales" -> "convencionales y dosis fijas...") continues
        # with a lowercase word, since it's a line-wrap mid-phrase, not a
        # new sentence. Some "b.2. Embarazadas"-style labels are instead
        # immediately followed by a capitalized, grammatically complete new
        # sentence/heading with no restated Evidencia: trigger at all
        # ("Estado metabólico (HbA1c, %)", "Tiempo en rango..."), which
        # this lowercase check correctly rejects as NOT a label
        # continuation. Once rejected, mode falls back to
        # mode_before_subcomp (the SAME field type active before the
        # subcomparacion trigger fired, since that content never got a
        # fresh header either) so it accumulates into the correct field
        # for the rest of the block.
        if mode == "subcomparacion_label":
            if (
                text[:1].islower()
                and not juicio_pattern.search(text)
                and not evidencia_pattern.search(text)
                and not consideraciones_pattern.search(text)
            ):
                current_subcomparacion_full = (current_subcomparacion_full + " " + text).strip()
                continue
            mode = mode_before_subcomp

        # -------- section boundary (known, hand-verified spots only) --------
        # Only flushes the pending row and stops accumulation, like
        # pregunta_pattern's own flush. Does NOT touch current_tema. A
        # real topic/subpregunta/pregunta-clinica/Contexto trigger reliably
        # follows shortly after every one of these known spots and sets
        # current_tema correctly itself. Clobbering it here left current_tema
        # stuck empty for any case where the next real trigger is
        # "Subpregunta para responder:" (its own branch only re-derives
        # current_tema from an actively open tema_buffer, not from scratch).
        if mode in ("judgement", "evidence", "considerations") and _is_section_boundary(filename, text):
            current_pregunta, current_subpregunta, mode, current_subcomparacion_full = flush_pending_row(reset_subpregunta=True)
            current_intervencion = ""
            continue

        pending_intervencion = None

        # -------- juicio --------
        if juicio_pattern.search(text):
            mode = "judgement"
            text = juicio_pattern.split(text, 1)[1]
            juicio.append(text)
            continue

        # -------- evidencia --------
        if evidencia_pattern.search(text):
            mode = "evidence"
            text = evidencia_pattern.split(text, 1)[1]
            evidencia.append(text)
            continue

        # -------- consideraciones --------
        if consideraciones_pattern.search(text):
            mode = "considerations"
            text = consideraciones_pattern.split(text, 1)[1]
            consideraciones.append(text)
            continue

        # -------- accumulate --------
        if mode == "question":
            current_pregunta = (current_pregunta + " " + text).strip()

        elif mode == "judgement":
            juicio.append(text)

        elif mode == "evidence":
            evidencia.append(text)

        elif mode == "considerations":
            consideraciones.append(text)

    # -------- save last row --------
    flush_pending_row(reset_subpregunta=True)

df = pd.DataFrame(rows)
df.to_csv(DATA_DIR / "dataset_prededup.csv", index=False, encoding="utf8")

"""## 4. Deduplication

Duplicate model inputs are removed: two rows with the same
(topic, subtopic, question, focus) are the same query to the model
even if their judgement/evidence/considerations differ, so only one is kept
(the one with the longer evidence+considerations text).
"""

input_key = df["topic"] + "|" + df["subtopic"] + "|" + df["question"] + "|" + df["focus"]
ec_len = (df["evidence"].fillna("") + df["considerations"].fillna("")).str.len()
deduped_df = (
    df.assign(_input_key=input_key, _ec_len=ec_len)
    .sort_values("_ec_len", ascending=False)
    .drop_duplicates("_input_key", keep="first")
    # The length-sort above is only to pick WHICH duplicate survives
    # (longer evidence+considerations wins); it otherwise scrambles row
    # order into "longest text first" across the whole corpus, unrelated
    # to guidebook or document order. Restoring the original index here
    # (df's own row order: guidebook by guidebook per `files`, and within
    # each guidebook, GRADE-letter order as the state machine emits rows)
    # means ids assigned below reflect true order of appearance, and
    # question letters (a), b), c)...) read in order within each topic.
    .sort_index()
    .drop(columns=["_input_key", "_ec_len"])
)
deduped_df.to_csv(DATA_DIR / "dataset_prefiltering.csv", index=False, encoding="utf8")

num_original_samples = len(df)
num_duplicated_input_features_removed = len(df) - len(deduped_df)

"""## 5. Sample filtering

Noisy samples are removed using regex. In particular, entries with placeholder evidence are excluded: *"ver apartado(s) anterior(es)"*, which means, "see previous section(s)".
"""

# Regex for skipping samples containing "ver apartado(s) anterior(es)":
skip_pattern = re.compile(r"ver apartado(?:s)? anterior(?:es)?", re.IGNORECASE)

no_ver_apartado_df = deduped_df[~deduped_df["evidence"].str.contains(skip_pattern, na=False)]

num_cross_referenced_placeholder_removed = len(deduped_df) - len(no_ver_apartado_df)

# A second, distinct class of noisy sample: evidence is a boilerplate
# "no evidence/studies were found" statement (the GAG/GEG found no formal
# evidence review to cite, in any of its ~25 real corpus phrasings, e.g.
# "No se identificaron estudios que respondieran a la pregunta.", "No se
# encontró evidencia...", "No se realizó una revisión de la evidencia
# disponible."), AND considerations offers nothing to compensate: either
# empty, or itself a "nothing to add" boilerplate ("No se han tenido en
# cuenta.", "No se realizaron consideraciones al respecto.", "No se han
# señalado.", "No no se han tenido en cuenta." (typo variant), "No se puede
# valorar porque no se han incluido estudios."). Rows where evidence is
# empty but considerations carries real GAG/GEG panel reasoning (cost
# estimates, feasibility/acceptability judgement, equity notes, explicit
# reasoning for why no formal review applied) are kept: that reasoning is
# substantive, source-faithful content, not noise, even though it did not
# come from a literature synthesis.
_no_evidence_found_pattern = re.compile(
    r"^(?:"
    r"No se identificaron estudios|"
    r"No se han podido identificar estudios|"
    r"No se ha identificado\.|"
    r"No se han identificado estudios|"
    r"No se han identificado\.|"
    r"No se ha localizado ningún estudio|"
    r"No se encontró evidencia|"
    r"No se localizaron estudios|"
    r"No se identificaron evaluaciones económicas|"
    r"No se identificó evidencia|"
    r"No se han incluido estudios|"
    r"No se ha identificado evidencia|"
    r"No se localizó evidencia|"
    r"No se han localizado estudios|"
    r"No se detectaron efectos|"
    r"No se facilitó al GEG ningún estudio|"
    r"No se han encontrado ECAs|"
    r"No se pudieron identificar estudios|"
    r"No se ha identificado ningún estudio|"
    r"No se ha identificado ningún daño|"
    r"No se han identificado estudios diagnósticos|"
    r"No se aportan datos|"
    r"No se describieron en los estudios|"
    r"No se han descrito|"
    r"No se realizó una revisión de la evidencia"
    r")",
    re.IGNORECASE,
)
_NO_CONSIDERATIONS_PLACEHOLDERS = {
    "No se han tenido en cuenta.",
    "No se han tenido en cuenta",
    "No no se han tenido en cuenta.",
    "No se realizaron consideraciones al respecto.",
    "No se han señalado.",
    "No se puede valorar porque no se han incluido estudios.",
}

# .str.strip() before matching: _no_evidence_found_pattern is anchored with
# ^, and some rows carry a leading space in evidence (a PDF-extraction
# artifact from a stray bullet/label byte upstream), which would otherwise
# silently defeat the anchor and leave those rows out of the filter.
evidence_stripped = no_ver_apartado_df["evidence"].fillna("").str.strip()
no_evidence_mask = evidence_stripped.str.contains(_no_evidence_found_pattern, na=False)
considerations_stripped = no_ver_apartado_df["considerations"].fillna("").str.strip()
# Reported as two separate counts (empty considerations vs. considerations
# itself a placeholder), even though both are discarded the same way, since
# they are two distinct real-world cases worth being able to tell apart in
# the log: one has genuinely nothing in considerations, the other has a
# placeholder there too, both leaving the row with no salvageable content.
empty_considerations_mask = no_evidence_mask & considerations_stripped.eq("")
placeholder_considerations_mask = no_evidence_mask & considerations_stripped.isin(_NO_CONSIDERATIONS_PLACEHOLDERS)
empty_evidence_and_considerations_mask = empty_considerations_mask | placeholder_considerations_mask

filtered_df = no_ver_apartado_df[~empty_evidence_and_considerations_mask].copy()

num_empty_considerations_removed = int(empty_considerations_mask.sum())
num_placeholder_considerations_removed = int(placeholder_considerations_mask.sum())
num_samples_after_filtering = len(filtered_df)

# Any row still reaching this point with an empty evidence field has real
# judgement + considerations content (the no-evidence-AND-no-considerations
# case was already discarded above), so it is kept, not dropped, its
# evidence cell is just genuinely absent from the source guideline. A short,
# deliberately neutral placeholder is filled in instead of leaving NaN:
# neutral in the specific sense of describing a gap in what THIS DOCUMENT
# records ("no evidence was documented for this question"), not a claim
# about whether evidence exists in the wider literature ("no studies were
# found", which several real corpus phrasings assert) or about whether a
# formal review was or wasn't performed (also unverifiable from the source
# text alone). Neither of those two claims can be verified for these rows,
# so this placeholder avoids making either one.
_EVIDENCE_PLACEHOLDER = "No se documentó evidencia relevante para esta cuestión."
evidence_missing_mask = filtered_df["evidence"].isna() | (filtered_df["evidence"].astype(str).str.strip() == "")
filtered_df.loc[evidence_missing_mask, "evidence"] = _EVIDENCE_PLACEHOLDER

num_evidence_placeholder_filled = int(evidence_missing_mask.sum())

# Stable, globally-unique id, assigned once here (final row set, pre-split)
# and carried through train_precuration.csv/dev_precuration.csv/test_precuration.csv,
# so no two rows across any split ever share an id, unlike the pandas
# index, which each split file resets to 0,1,2... independently after sorting.
filtered_df = filtered_df.reset_index(drop=True)
filtered_df.insert(0, "id", [f"guiasalud_{i}" for i in range(1, len(filtered_df) + 1)])

try:
    filtered_df.to_csv(DATA_DIR / "dataset_precuration.csv", index=False, encoding="utf8")
except Exception as e:
    print("Error while saving CSV:", e)

"""## 6. Intermediate automatic evaluation

Quality checks are performed on the extracted dataset, including:

- Presence rate of optional features
- Presence rate of mandatory features
"""

#Checkpoint
import re
import pandas as pd

filtered_df = pd.read_csv(DATA_DIR / "dataset_precuration.csv")

"""### 6.1 Presence rate of optional features

Let's measure how often optional features (subtopic, focus, considerations) appear in the dataset:
"""

subtopic_ratio = filtered_df["subtopic"].notna().mean()
focus_ratio = filtered_df["focus"].notna().mean()
cons_ratio = filtered_df["considerations"].notna().mean()

"""### 6.2 Presence rate of mandatory features

Let's verify if all cells of required features (e.g., question, judgement, evidence) are non-empty across all samples:
"""

required_cols = ["guidebook", "topic", "question", "judgement", "evidence"]

required_col_percentages = {}
for col in required_cols:
    valid = filtered_df[col].notna() & (filtered_df[col].astype(str).str.strip() != "")
    required_col_percentages[col] = valid.mean() * 100

# True if cell is valid (non-empty, non-null)
valid_mask = (
    filtered_df[required_cols]
    .notna()
    & (filtered_df[required_cols].astype(str).apply(lambda x: x.str.strip() != ""))
).all(axis=1)

# ids of incomplete samples
incomplete_ids = filtered_df.loc[~valid_mask, "id"].tolist()

"""The output cell above shows that there are 11 empty cells in mandatory features. The indices of those samples are provided for later manual curation (§8).

### 6.3 Section-heading leak check

Completeness (§6.2) only asks whether a field is non-empty. It says nothing
about whether the field's content was cut off at the right place. A field
should NEVER contain a numbered document heading (e.g. "5. Tratamiento
psicológico", "6.1.3.2. Vortioxetina"). A heading is exactly the kind of
marker that's supposed to terminate accumulation of a field, so if one shows
up INSIDE topic/subquestion/judgement/evidence/considerations, accumulation
ran past it instead of stopping there, and the field has swallowed the
start of an unrelated section.

Checked across all 5 free-text fields, not just judgement/evidence/
considerations: topic is built by the same kind of accumulate-until-a-
terminator logic (§2-3, e.g. collecting_tema/collecting_pregunta_clinica)
and is equally susceptible. Confirmed via a real case (a topic that
absorbed an entire subquestion-preview list plus the start of the next
section's own heading, "4.3.1. Enfermedad tiroidea autoinmune", before the
pregunta_dividida_pattern generalization in §2 fixed it corpus-wide instead
of only at the single location first found).

This is how the known leaks fixed via SECTION_BOUNDARY_TITLES and
pregunta_dividida_pattern (§2) were originally found: not by verifying
every row's content is correct (which this pipeline's guidebooks make
unreliable to automate. GRADE questions and even some evidence sentences
are reused verbatim across many unrelated locations, so re-deriving "does
this text belong here" from scratch is ambiguous in a way the generator's
own left-to-right pass isn't), but by searching for a structural marker
that should categorically never appear mid-field. This check re-runs that
same search on every pipeline run, so a future edit to the extraction logic
that reopens one of these holes (or a new guidebook with its own unlisted
heading) is caught immediately rather than requiring another manual audit
to rediscover.
"""

# [^.:]* (up to end of string, no internal period or colon) replaces an
# earlier {0,4}-word cap, which silently caught ZERO real leaks: genuine
# headings run well past 5 words ("Inhibidores del cotransportador de
# sodio-glucosa tipo 2", 7 words) and the old \s (exactly one whitespace
# char) missed headings separated by a tab-after-space ("5. \tParticipación
# ..."), both confirmed against real leaked rows. The colon exclusion is
# needed because the same shape ("N. Capitalized text") also matches a
# genuine in-field numbered list item ending in ":" (e.g. a considerations
# field's own "5. Escasez de material educativo para... ceguera:"), which is
# real content, not a leaked heading, headings never end in a colon.
_section_heading_leak_pattern = re.compile(
    r'\.\s+\d+(?:\.\d+){0,3}\.\s+[A-ZÁÉÍÓÚÑ][^.:]*\s*$'
)

section_leak_ids = []
for _, row in filtered_df.iterrows():
    for field in ["topic", "subtopic", "focus", "judgement", "evidence", "considerations"]:
        value = row[field]
        if pd.notna(value) and _section_heading_leak_pattern.search(str(value).rstrip()):
            section_leak_ids.append((row["id"], field))
            break

"""### 6.4 Residual line-wrap hyphen check

_merge_hyphenated_line_wraps (§1) only merges a split ("sín-" + "tomas" ->
"síntomas") when corpus-frequency or the BSC tokenizer confirms it.
Everything else, including every case classified as a genuine compound
("coste-efectividad"), is deliberately left with the source's original
"word- word" spacing, since manual review found ~45% of compound
classifications were themselves wrong (a PDF two-column-layout artifact
that neither check could reliably detect, see §1's comment for the full
explanation). That means a real, uncorrected artifact can still be sitting
in the final data, silently, unless it's surfaced somewhere.

This does not try to re-decide join-vs-compound (that ambiguity is exactly
what could not be resolved automatically). It only flags every remaining
"<lowercase letter>- <lowercase letter>" occurrence so it's visible during
manual curation, instead of only being found by chance while reading a
field (as the original sín- tomas report was).
"""

_residual_hyphen_pattern = re.compile(r'[a-záéíóúñ]-\s[a-záéíóúñ]')

# The bare regex match is only 3-4 characters ("n- s"), not enough on its own
# to judge join-vs-compound-vs-interleaved-artifact from the printed log
# without opening the CSV. This extracts just the word(s) the match
# overlaps (no surrounding context), stripped of leading/trailing
# punctuation but keeping the hyphen itself, so the log shows enough to make
# that call, e.g. distinguishing a genuine split ("compen- sado" ->
# "compensado") from a two-column-layout interleaving artifact ("pacien-
# tiempo", where the real continuation "tes" sits several words further on,
# separated from "pacien-" by unrelated interloper text, see §6.4's own
# comment for the full explanation of this failure mode).
def _hyphen_match_words(text, match):
    tokens = list(re.finditer(r"\S+", text))
    match_start, match_end = match.start(), match.end()
    overlapping = [
        token.group() for token in tokens if token.end() > match_start and token.start() < match_end
    ]
    cleaned = []
    for word in overlapping:
        word = re.sub(r"^[^a-záéíóúñA-ZÁÉÍÓÚÑ]+", "", word)
        word = re.sub(r"[^a-záéíóúñA-ZÁÉÍÓÚÑ\-]+$", "", word)
        cleaned.append(word)
    return " ".join(cleaned)

residual_hyphen_ids = []
residual_hyphen_words = {}
for _, row in filtered_df.iterrows():
    for field in ["topic", "subtopic", "question", "focus", "judgement", "evidence", "considerations"]:
        value = row[field]
        if pd.notna(value):
            match = _residual_hyphen_pattern.search(str(value))
            if match:
                residual_hyphen_ids.append((row["id"], field))
                residual_hyphen_words[(row["id"], field)] = _hyphen_match_words(str(value), match)
                break

# The detailed id/field/split listing (and every count computed above) is
# printed together at the very end of the script (see the final report
# section), since split isn't assigned to any row until §7 runs, and
# grouping every summary statistic into one place at the end makes the
# printed run log easier to read than statistics interleaved with
# processing steps throughout.

"""## 7. Train/dev/test split

The dataset is split into training, development, and test sets, stratified
by guidebook. Dev and test are sized to match CasiMédicos-Exp's own dev
(63) and test (125) splits exactly.

Requires `dataset_precuration.csv`, written by §5 (over the deduplicated
data from §4) and evaluated in §6.
"""

from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA_DIR / "dataset_precuration.csv")

DEV_SIZE = 63
TEST_SIZE = 125

train_dev_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=df["guidebook"],
)
train_df, dev_df = train_test_split(
    train_dev_df,
    test_size=DEV_SIZE,
    random_state=42,
    stratify=train_dev_df["guidebook"],
)

assert len(dev_df) == DEV_SIZE and len(test_df) == TEST_SIZE

# Sorts by (guidebook, id-as-number) rather than plain guidebook (pandas'
# sort_values is stable, so a guidebook-only sort would leave rows in
# whatever order train_test_split's random shuffle put them in, not
# document order) or plain id (which groups by id across ALL guidebooks at
# once, not per guidebook). id-as-number, not the raw string, so
# guiasalud_2 sorts before guiasalud_10 rather than lexicographically.
def _sort_by_guidebook_then_id(df):
    id_num = df["id"].str.removeprefix("guiasalud_").astype(int)
    return df.assign(_id_num=id_num).sort_values(
        by=["guidebook", "_id_num"]
    ).drop(columns="_id_num").reset_index(drop=True)

# None of the three splits are curated yet at this point, so all are written
# under a _precuration name, not train.csv/dev.csv/test.csv. Curation
# (next, manual) copies them into train.csv/dev.csv/test.csv, keeping these
# pre-curation snapshots around as a diff baseline for what curation changed.
train_df = _sort_by_guidebook_then_id(train_df)
train_df.to_csv(DATA_DIR / "train_precuration.csv", index=False)

dev_df = _sort_by_guidebook_then_id(dev_df)
dev_df.to_csv(DATA_DIR / "dev_precuration.csv", index=False)

test_df = _sort_by_guidebook_then_id(test_df)
test_df.to_csv(DATA_DIR / "test_precuration.csv", index=False)

# Overwrite dataset_precuration.csv with a version that also records each
# row's split assignment, so the whole corpus can be audited (e.g. presence
# rates or guidebook balance per split) without joining the three split files.
# Sorted the same way as the three split files above (guidebook, then id
# order within each guidebook), with split placed right after id.
with_split_df = pd.concat(
    [train_df.assign(split="train"), dev_df.assign(split="dev"), test_df.assign(split="test")],
    ignore_index=True,
)
with_split_df = _sort_by_guidebook_then_id(with_split_df)
with_split_df.insert(1, "split", with_split_df.pop("split"))
with_split_df.to_csv(DATA_DIR / "dataset_precuration.csv", index=False)

"""## Run summary

Every count computed throughout the pipeline (§4-§7) is printed together
here, in one place, instead of interleaved with the processing steps that
computed them: filtering counts, mandatory/optional feature completeness,
extraction-leak diagnostics, and split sizes. The detailed id/field/split
listing for unresolved line-wrap hyphenation artifacts needs split, which
isn't assigned to any row until §7 runs, so it could not be printed earlier
than this regardless.
"""

# Sorted test -> dev -> train (rather than left in insertion/file order),
# so the split the curation policy (§8) treats as highest-priority --
# the entire test set is manually reviewed, train only partially -- sorts
# to the top of each listing instead of being scattered throughout.
_SPLIT_SORT_ORDER = {"test": 0, "dev": 1, "train": 2}
id_to_split = dict(zip(with_split_df["id"], with_split_df["split"]))

def _by_split_priority(id_field_pairs):
    return sorted(id_field_pairs, key=lambda pair: _SPLIT_SORT_ORDER[id_to_split[pair[0]]])

print()
print("-----")
print()
print("Number of original samples:", num_original_samples)
print("- Number of removed samples due to duplicated input features:", num_duplicated_input_features_removed)
print("- Number of removed samples due to cross-referenced evidence placeholder:", num_cross_referenced_placeholder_removed)
print("- Number of removed samples due to no-evidence placeholder and empty considerations field:", num_empty_considerations_removed)
print("- Number of removed samples due to no-evidence placeholder and no-considerations placeholder:", num_placeholder_considerations_removed)
print("- Number of empty evidence fields automatically filled with no-evidence placeholder:", num_evidence_placeholder_filled)
print()
print("Number of samples after filtering:", num_samples_after_filtering)
print(f"- Train size: {len(train_df)}")
print(f"- Dev size: {len(dev_df)}")
print(f"- Test size: {len(test_df)}")
print()
print("-----")
print()
print("Presence of mandatory features:")
for col in required_cols:
    print(f"- {col}: {required_col_percentages[col]:.2f}%")
if len(incomplete_ids) > 0:
    print("- Number of incomplete samples:", len(incomplete_ids))
    print("- IDs of incomplete samples:")
    print(incomplete_ids)
print()
print("Presence of optional features:")
print(f"- subtopic: {subtopic_ratio*100:.2f}%")
print(f"- focus: {focus_ratio*100:.2f}%")
print(f"- considerations: {cons_ratio*100:.2f}%")
print()
print("-----")
print()
print("Extraction leaking statistics:")
print("- Number of section-heading leaks:", len(section_leak_ids))
if len(section_leak_ids) > 0:
    print("- IDs, fields, and splits:")
    for sample_id, field in _by_split_priority(section_leak_ids):
        print(f"  {sample_id} ({id_to_split[sample_id]}, {field})")
print("- Number of automatically removed numeric footnote markers:", footnote_markers_removed)
print("- Number of automatically resolved line-wrap hyphenation artifacts:", hyphen_wraps_merged)
print("- Number of automatically unresolved line-wrap hyphenation artifacts (manual curation needed):", len(residual_hyphen_ids))
if len(residual_hyphen_ids) > 0:
    print("- IDs, fields, splits, and words of automatically unresolved line-wrap hyphenation artifacts (manual curation needed):")
    for sample_id, field in _by_split_priority(residual_hyphen_ids):
        words = residual_hyphen_words[(sample_id, field)]
        print(f'  {sample_id} ({id_to_split[sample_id]}, {field}): "{words}"')
print()
print("-----")
print()
print("CSVs created successfully.")
print()

"""## 8. Manual curation

The entire test set is manually reviewed to fix segmentation errors and ensure data quality. However, the training set is only partially inspected, as noise is less critical in training data and full manual curation would be highly time-consuming.

This is a manual, off-script step: copy `train_precuration.csv` to
`train.csv`, `dev_precuration.csv` to `dev.csv`, and `test_precuration.csv`
to `test.csv` (same directory), then curate the copies (train only
partially, per the paragraph above). The `_precuration` files are left
untouched as the pre-curation baseline. Sections 9-10 (final evaluation,
quantitative analysis) continue in `datasetting_postcuration.py`, which
builds `dataset.csv` from the current `train.csv`/`dev.csv`/`test.csv`.
"""
