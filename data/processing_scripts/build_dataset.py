"""build_dataset.py - Intermediate ingestion and extraction script for q3as.

Discovers Ada source files (.ads, .adb) and .gpr project files
across target directory trees (including ../adacovex and ../Ada_CRDT), pairs
specification and implementation files, detects the target Ada standard via
heuristic keyword analysis, sanitizes content, and writes a standardized JSONL
dataset formatted with the OpenAI/Qwen chat template.

Turn kinds produced by this script:

- code pairs        spec/body completion turns from .ads/.adb trees
- doc-QA            Ada code blocks extracted from ../learn course material
- defect pairs      correct code next to deliberately broken variants
                    (syntax, context clause, visibility, contract,
                    spec/body mismatch) with an STE-compliant diagnosis
- STE doc-QA        doc explanation turns whose answers follow
                    ASD-STE100 rules and define technical terms
- toolchain QA      question/answer turns distilled from the AdaCore
                    agent skills (gnatprove, alire, gnatdoc, gnattest,
                    gnatfuzz)

Guidance ingestion (embedded into system prompts on a subset of turns):

- ../ada-spark       the ada-spark agent skill (MIT): current-toolchain
                     correction map and SPARK assurance guidance
- ../SimpleEnglish   the simple-english agent skill (MIT): ASD-STE100
                     writing rules (no em-dashes, active voice, one word
                     one meaning) and the slop-to-plain word map. These
                     rules govern every assistant explanation this script
                     generates, and a distilled rule block is embedded into
                     system prompts so the model learns to write docs that
                     obey them.
- ../skills          AdaCore's agent skills (Apache-2.0): gnatprove, alire,
                     gnatdoc, gnattest, gnatfuzz. The SKILL.md files are
                     loaded as toolchain guidance and converted to
                     question/answer training turns.

Also loads evaluation methodology from ../ada-eval for dataset structure
and metric definitions.

Usage:
    python build_dataset.py --input-dir data/raw/
    python build_dataset.py --input-dir data/raw/  # extra sources come from the cache
    python build_dataset.py --input-dir /path/to/submodule --output-dir data/processed/
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import random
import re
import textwrap
import zlib
from pathlib import Path
from typing import Any

import code_variants
import eval_guard
import progress as progress_log
import stage_state

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ADA_SPEC_EXTENSIONS = {".ads", ".adb"}
PROJECT_EXTENSIONS = {".gpr"}
DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "dataset.jsonl"
# Default extra input directories come from the source cache at runtime
# (source_paths.default_code_dirs()); nothing is hard-coded here anymore.
# Directories treated as documentation sources: Ada code blocks are extracted
# from their RST/Markdown content instead of pairing .ads/.adb files.
DOC_SOURCE_DIRS = ["learn"]
# Agent-skill sources whose markdown guidance is embedded into system prompts.
# Each entry maps a sibling directory name to the repo-relative paths of its
# skill documents so the loader works across the three repo layouts we use.
AGENT_SKILL_SOURCE_DIRS = ["ada-spark", "SimpleEnglish", "skills"]


def _resolved_source(name: str) -> Path:
    """Cache directory for a source repo (or its legacy sibling location)."""
    import source_paths

    resolved = source_paths.resolve(name)
    if resolved is None:
        return Path("data/raw_repos/_missing") / name
    return resolved


ADA_SPARK_DIR = _resolved_source("ada-spark")
SIMPLE_ENGLISH_DIR = _resolved_source("SimpleEnglish")
ADACORE_SKILLS_DIR = _resolved_source("skills")
# Default path to ada-eval methodology directory
ADA_EVAL_DIR = _resolved_source("ada-eval")

# Deterministic pseudo-randomness for defect selection and guidance sampling
_RNG_SEED = 42
_rng = random.Random(_RNG_SEED)

# ---- Sanitization patterns ------------------------------------------------ #
_SECRET_PATTERNS = re.compile(
    r"""
    (?:password|secret|api_key|apikey|token|private_key)\s*[:=]\s*
    ['"]?[A-Za-z0-9_\-+/=]{8,}['"]?
    """,
    re.IGNORECASE | re.VERBOSE,
)
_BINARY_MAGIC = {b"\x00", b"\xff\xfe", b"\xfe\xff", b"\x7fELF"}

# ---- Ada standard detection (feature-based, newest-first) ----
# Newer Ada standards are supersets of older ones, so a feature that
# standard X introduced is conclusive evidence for an X floor. Detection
# strips -- comments first (prose such as "since Ada 95" must not vote),
# then counts weighted features newest-first; the first standard whose
# weighted score clears its threshold wins. SPARK is tested before Ada
# 2012 because SPARK-specific annotations (SPARK_Mode, Loop_Invariant,
# Contract_Cases) are what the model must imitate; a plain Ada 2012 file
# without them falls through to the Ada 2012 tier.

_COMMENT_LINE_RE = re.compile(r"--[^\n]*")
# Ada string literal: doubled quotes are the escape ("say ""hi""").
_ADA_STRING_RE = re.compile(r'"(?:[^"]|"")*"')


def _strip_ada_comments(code: str) -> str:
    """Remove -- comments and string contents so prose cannot vote in
    standard detection (a comment or message naming SPARK_Mode is not
    evidence the file is SPARK)."""
    code = _COMMENT_LINE_RE.sub(" ", code)
    return _ADA_STRING_RE.sub('""', code)


# Each feature is (label, pattern, weight). Weight 2 marks features that
# are conclusive for their standard, weight 1 strong hints. Thresholds
# require more than one hint, so a single false positive cannot decide.
_STANDARD_FEATURES: list[tuple[str, list[tuple[str, re.Pattern[str], int]], int]] = [
    ("Ada 2022", [
        ("reduce", re.compile(r"\bReduce\s*\(", re.IGNORECASE), 2),
        ("put_image", re.compile(r"\bPut_Image\b", re.IGNORECASE), 1),
        ("parallel", re.compile(r"\bparallel\s+(do|block|and|loop)\b", re.IGNORECASE), 2),
        ("delta_aggregate", re.compile(r"\bwith\s+delta\b", re.IGNORECASE), 2),
        ("update_aspect", re.compile(r"\b'Update\s*=>", re.IGNORECASE), 2),
        ("static_predicate", re.compile(r"\bStatic_Predicate\b", re.IGNORECASE), 2),
        ("then_abort", re.compile(r"\bselect\s+[^;]+\s+then\s+abort\b", re.IGNORECASE), 2),
    ], 2),
    ("Ada 2012", [
        ("pre_aspect", re.compile(r"\bPre\s*=>", re.IGNORECASE), 2),
        ("post_aspect", re.compile(r"\bPost\s*=>", re.IGNORECASE), 2),
        ("type_invariant", re.compile(r"\bType_Invariant('Class)?\s*=>", re.IGNORECASE), 2),
        ("global_aspect", re.compile(r"\bGlobal\s*=>", re.IGNORECASE), 2),
        ("depends_aspect", re.compile(r"\bDepends\s*=>", re.IGNORECASE), 2),
        ("predicate_aspect", re.compile(r"\b(Dynamic_)?Predicate('Class)?\s*=>", re.IGNORECASE), 2),
        ("quantified", re.compile(r"\(\s*for\s+(all|some)\b", re.IGNORECASE), 2),
        ("if_expression", re.compile(r"\(\s*if\s[^()]*\bthen\s[^()]*\belse\b", re.IGNORECASE), 2),
        ("case_expression", re.compile(r"\(\s*case\s[^()]+\bis\s[^()]+\bwhen\b", re.IGNORECASE), 2),
        ("declare_expression", re.compile(r"\(\s*declare\b", re.IGNORECASE), 2),
        ("expression_function", re.compile(r"\bfunction\s+[\w.]+\s*(\([^)]*\))?\s*(return\s+[\w.]+\s*)?is\s*\(", re.IGNORECASE), 2),
        ("in_iterator", re.compile(r"\bfor\s+\w+\s+of\b", re.IGNORECASE), 2),
        ("wide_wide", re.compile(r"\bWide_Wide_\w", re.IGNORECASE), 1),
    ], 2),
    ("SPARK 2014", [
        ("spark_mode", re.compile(r"\bSPARK_Mode\b", re.IGNORECASE), 2),
        ("loop_invariant", re.compile(r"\bLoop_Invariant\b", re.IGNORECASE), 2),
        ("loop_variant", re.compile(r"\bLoop_Variant\b", re.IGNORECASE), 2),
        ("contract_cases", re.compile(r"\bContract_Cases\b", re.IGNORECASE), 2),
        ("loop_entry", re.compile(r"\bLoop_Entry\b", re.IGNORECASE), 2),
        ("relaxed_init", re.compile(r"\bRelaxed_Initialization\b", re.IGNORECASE), 2),
        ("ghost", re.compile(r"\bGhost\b", re.IGNORECASE), 1),
        ("assume_cut", re.compile(r"\bpragma\s+(Assume|Assert_And_Cut|Guard)\b", re.IGNORECASE), 1),
        ("gnatprove", re.compile(r"\bgnatprove\b"), 1),
    ], 2),
    ("Ada 2005", [
        ("interface_type", re.compile(r"\btype\s+\w+\s+is\s+interface\b", re.IGNORECASE), 2),
        ("overriding", re.compile(r"\boverriding\b", re.IGNORECASE), 1),
        ("containers", re.compile(r"\bAda\.Containers\b"), 2),
        ("use_type", re.compile(r"\buse\s+type\b", re.IGNORECASE), 1),
        ("null_procedure", re.compile(r"\bis\s+null\s*;", re.IGNORECASE), 1),
        ("not_null_access", re.compile(r"\bnot\s+null\s+access\b", re.IGNORECASE), 2),
        ("assert_pragma", re.compile(r"\bpragma\s+Assert\b", re.IGNORECASE), 1),
    ], 2),
    ("Ada 95", [
        ("tagged", re.compile(r"\btagged\b", re.IGNORECASE), 2),
        ("abstract", re.compile(r"\babstract\s+(tagged|type)\b", re.IGNORECASE), 1),
        ("child_unit", re.compile(r"\bpackage\s+(body\s+)?\w+\.\w+\s+is\b", re.IGNORECASE), 2),
        ("controlled", re.compile(r"\bAda\.Finalization\.(Root_)?(Limited_)?Controlled\b"), 2),
        ("protected", re.compile(r"\bprotected\s+(type|body)\b", re.IGNORECASE), 2),
        ("aliased", re.compile(r"\baliased\b", re.IGNORECASE), 1),
        ("wide_char", re.compile(r"\bWide_(String|Character|Text)\b"), 1),
    ], 2),
    ("Ada 83", [
        ("package_spec", re.compile(r"\bpackage\s+\w+\s+is\b", re.IGNORECASE), 1),
        ("package_body", re.compile(r"\bpackage\s+body\b", re.IGNORECASE), 1),
        ("subprogram", re.compile(r"\b(procedure|function)\s+\w+", re.IGNORECASE), 1),
        ("pragma", re.compile(r"\bpragma\s+\w+", re.IGNORECASE), 1),
        ("task", re.compile(r"\btask\s+(body\s+)?\w+", re.IGNORECASE), 1),
    ], 2),
]


def detect_ada_standard(content: str) -> str:
    """Detect the Ada standard (or SPARK sublanguage) of *content*.

    Comment-stripped, weighted feature matching, newest standard first.
    The first standard whose weighted feature score reaches its threshold
    wins. Returns the standard label, or 'Unknown' when nothing matches.
    """
    code = _strip_ada_comments(content)
    for standard_name, features, threshold in _STANDARD_FEATURES:
        score = sum(weight for _, pattern, weight in features if pattern.search(code))
        if score >= threshold:
            return standard_name
    return "Unknown"


# --------------------------------------------------------------------------- #
# STE writing rules (distilled from ../SimpleEnglish, ASD-STE100 Issue 9)
# --------------------------------------------------------------------------- #
# These rules govern every explanation this script generates. The rule
# block is also embedded into system prompts so the fine-tuned model
# learns the same constraints. The wording is our paraphrase of the
# SimpleEnglish skill's rules; the standard itself (ASD-STE100) forbids
# reproduction of its text and dictionary.

# Word swaps distilled from SimpleEnglish references/word-swaps.md. AI
# overuses these words; STE writes the plain form or deletes the word.
_SLOP_SWAPS: list[tuple[str, str]] = [
    ("in order to", "to"),
    ("prior to", "before"),
    ("due to the fact that", "because"),
    ("in the event that", "if"),
    ("it is worth noting that", ""),
    ("it's important to", ""),
    ("it is important to", ""),
    ("leverage", "use"),
    ("utilize", "use"),
    ("harness", "use"),
    ("seamlessly", ""),
    ("effortlessly", ""),
    ("simply", ""),
    ("robust", ""),
    ("comprehensive", ""),
    ("crucial", "important"),
    ("pivotal", "important"),
    ("paramount", "important"),
    ("facilitate", "help"),
    ("delve into", "examine"),
    ("dive into", "examine"),
    ("showcase", "show"),
    ("underscore", "show"),
    ("furthermore", "also"),
    ("moreover", "also"),
    ("in conclusion", ""),
    ("in summary", ""),
    ("holistic", "full"),
    ("plethora", "many"),
    ("myriad", "many"),
    ("cutting-edge", "new"),
    ("state-of-the-art", "new"),
    ("groundbreaking", "new"),
    ("streamline", "simplify"),
    ("when it comes to", "for"),
]

# Words SimpleEnglish rules out as hedges. can/will/must survive.
_HEDGES = re.compile(
    r"\b(should|would|may|might|could)\b(?!\s*(not|be\s+used))", re.IGNORECASE
)

_PAIRED_EM_DASH = re.compile(r"\s*[—–]\s*([^—–]*?)\s*[—–]\s*")
_EM_DASHES = re.compile(r"\s*[—–]\s*")
_SEMICOLONS = re.compile(r"\s*;\s*")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r" ([,.:!?])")


def apply_word_swaps(text: str) -> str:
    """Replace slop words with their plain equivalents or delete them."""
    for slop, plain in _SLOP_SWAPS:
        pattern = re.compile(r"\b" + re.escape(slop) + r"\b", re.IGNORECASE)
        if plain:
            text = pattern.sub(plain, text)
        else:
            # Deleting: drop the whole clause if it was a filler phrase.
            text = pattern.sub("", text)
    return text


def sanitize_prose(text: str) -> str:
    """Rewrite prose to the STE style distilled from SimpleEnglish.

    Applies, in order: slop word swaps, paired em-dash to comma appositive,
    remaining em-dash and semicolon removal (split into sentences), hedge
    removal, whitespace cleanup. Newlines are preserved: paragraph
    structure survives sanitization. Code, identifiers, and quoted strings
    must not run through this function.
    """
    text = apply_word_swaps(text)
    # A paired em-dash is an appositive: "the reader - a user - confirms"
    # becomes "the reader, a user, confirms". SimpleEnglish: no em-dashes.
    text = _PAIRED_EM_DASH.sub(r", \1, ", text)
    # Remaining em-dashes become sentence breaks. Semicolons likewise:
    # write two sentences, or name the relation.
    text = _EM_DASHES.sub(". ", text)
    text = _SEMICOLONS.sub(". ", text)
    # Hedging modal sentences become direct statements. "may fail" -> "can fail".
    text = _HEDGES.sub("can", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    # Collapse artifacts left by deletions.
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"(^|\n)\s*\.\s*", r"\1", text)
    text = re.sub(r",\s*\.", ".", text)
    return text.strip()


def has_style_violations(text: str) -> list[str]:
    """List STE style violations found in a prose string (used on our
    own generated text as a self-check, and to tag metadata)."""
    violations: list[str] = []
    if re.search(r"[—–]", text):
        violations.append("em-dash")
    if ";" in text:
        violations.append("semicolon")
    if re.search(r"\b(should|would|may|might)\b", text, re.IGNORECASE):
        violations.append("hedge-modal")
    for slop in ("leverage", "utilize", "seamless", "robust", "crucial",
                 "comprehensive", "in order to", "it is worth noting"):
        if re.search(r"\b" + re.escape(slop), text, re.IGNORECASE):
            violations.append(f"slop:{slop}")
    return violations


# ---- STE system-prompt rule block ----------------------------------------- #

STE_RULE_BLOCK = """\
Simplified Technical English (ASD-STE100) writing rules:

1. Use active voice and simple tenses. Name the actor.
2. Keep instructions to 20 words and descriptions to 25 words per sentence.
3. Write the condition before the command: "If the build fails, read the log."
4. No em-dashes and no semicolons. Write two sentences, or name the relation.
5. Modals: can, will, must. Never should, would, may, might, could.
6. No contractions. Keep articles. Keep "that".
7. One word, one meaning, for the whole document.
8. Define a technical term at its first use, in a few words.
9. State the fact, not its importance. Delete: simply, robust, crucial,
   comprehensive, "it is worth noting".
10. Code, identifiers, commands, and quoted errors are exempt. Never edit them.
"""

_TECH_TERM_DEFINITION_PROMPT = (
    "Define each technical term at its first use. Keep the definition to a "
    "few words inside parentheses, for example: \"contract (a checked "
    "assertion about a subprogram's inputs and outputs)\"."
)


def build_technical_term_glossary() -> str:
    """Build the Ada/SPARK technical-term glossary block.

    This is our own collation of standard Ada/SPARK terminology, written
    to teach one-word-one-meaning discipline: each term is defined once,
    in under ten words, the way SimpleEnglish rule 9 handles concept
    terms. The model learns to use the canonical term and to define it
    at first use instead of switching between synonyms.
    """
    terms = [
        ("specification (.ads)", "the public contract of a package: types and subprogram signatures"),
        ("body (.adb)", "the implementation that completes a specification"),
        ("aspect", "a property attached to a declaration, written with =>"),
        ("precondition (Pre)", "an assertion that must hold on entry to a subprogram"),
        ("postcondition (Post)", "an assertion that must hold when a subprogram returns"),
        ("invariant", "an assertion that must hold for every object of a type"),
        ("contract", "a checked assertion about a subprogram or type: preconditions, postconditions, invariants"),
        ("context clause", "the with/use lines before a compilation unit"),
        ("with clause", "a context clause that makes a library unit visible"),
        ("pragma", "a compiler directive"),
        ("discriminant", "a parameter of a record type fixed at object creation"),
        ("variant part", "the case structure inside a record type"),
        ("access type", "a pointer type"),
        ("tagged type", "a record type that supports inheritance and dynamic dispatch"),
        ("primitive operation", "a subprogram declared in the same package as a tagged type"),
        ("controlled type", "a type with user-defined initialization and finalization"),
        ("generic", "a template instantiated with types or values"),
        ("task", "a unit of concurrent execution"),
        ("protected object", "a passive unit that serializes access to shared data"),
        ("Ravenscar profile", "a restricted tasking subset for high-integrity systems"),
        ("exception", "a runtime error condition that propagates until handled"),
        ("renames", "a new name for an existing entity"),
        ("instantiation", "the creation of a concrete unit from a generic"),
        ("elaboration", "the pre-execution phase that initializes library units"),
        ("SPARK_Mode", "a pragma that marks code as inside the SPARK subset"),
        ("ghost code", "code that exists only for proof and is erased at runtime"),
        ("loop invariant", "an assertion that holds on every iteration of a loop"),
        ("proof obligation", "a check that gnatprove must discharge"),
        ("flow analysis", "gnatprove's check that reads and writes respect data dependencies"),
        ("Alire crate", "a reusable Ada project managed by the alr tool"),
        ("GNAT project file (.gpr)", "the build description that names sources and options"),
        ("scenario variable", "a -X NAME=VALUE switch that selects build variants"),
    ]
    lines = [f"- {term}: {definition}." for term, definition in terms]
    return "Canonical Ada/SPARK terminology (define at first use, use consistently):\n" + "\n".join(lines)


# ---- STE / SimpleEnglish guidance loading --------------------------------- #

def load_simple_english_rules() -> str:
    """Load writing rules from the SimpleEnglish agent skill (../SimpleEnglish).

    Returns the distilled STE rule block plus the slop-to-plain word map
    from the skill's references. Falls back to the built-in STE_RULE_BLOCK
    when the sibling repo is absent. The full SKILL.md text is deliberately
    not embedded: it would eat most of a 2048-token training window.
    """
    parts: list[str] = [STE_RULE_BLOCK]

    word_swaps = SIMPLE_ENGLISH_DIR / "skills" / "simple-english" / "references" / "word-swaps.md"
    if word_swaps.exists():
        try:
            swaps_text = word_swaps.read_text(encoding="utf-8").strip()
            # Keep the table but cap it: the first rows carry most signal.
            lines = swaps_text.splitlines()
            table_rows = [ln for ln in lines if ln.startswith("|")]
            if len(table_rows) > 24:
                swaps_text = "\n".join(lines[: lines.index(table_rows[24])])
            parts.append("Slop-to-plain word map (from the same skill):\n" + swaps_text)
        except (UnicodeDecodeError, OSError):
            pass
        logger.info("Loaded SimpleEnglish STE rules from %s", SIMPLE_ENGLISH_DIR)
    else:
        logger.info(
            "SimpleEnglish skill not found at %s - using built-in STE rules",
            SIMPLE_ENGLISH_DIR,
        )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Agent-skill guidance loading (ada-spark, AdaCore skills)
# --------------------------------------------------------------------------- #

# Per-repo skill-document discovery rules: dir name -> repo-relative glob
# patterns for files whose content forms the guidance text.
_SKILL_DOC_PATTERNS: dict[str, list[str]] = {
    "ada-spark": ["SKILL.md", "agent-knowledge/*.md"],
    "SimpleEnglish": ["skills/*/SKILL.md", "skills/*/references/*.md"],
    "skills": ["plugins/*/skills/*/SKILL.md"],
}

# AdaCore skills: keep SKILL.md files compact in prompts, but load a
# bounded subset of reference files so toolchain QA turns have substance.
_ADACORE_REF_LIMIT = 6
_ADACORE_REF_MAX_CHARS = 6000


def _collect_skill_docs(root: Path, dir_name: str) -> list[Path]:
    """Collect skill markdown files from an agent-skill repo by its layout."""
    patterns = _SKILL_DOC_PATTERNS.get(dir_name, [])
    docs: list[Path] = []
    for pattern in patterns:
        docs.extend(sorted(root.glob(pattern)))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for doc in docs:
        if doc.is_file() and doc not in seen:
            seen.add(doc)
            unique.append(doc)
    return unique


_ADACORE_SKILL_PRIORITY = ["gnatprove", "alire", "gnatdoc", "gnattest", "gnatfuzz"]


def _source_label(source: str) -> str:
    """Machine-independent source label for dataset text.

    Input directories are resolved to absolute paths at discovery time, so
    embedding them in user messages would hard-code the build machine's
    filesystem layout into the trained model. Only the final component is
    meaningful across machines (repo or file name), and it is deterministic.
    """
    normalized = source.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1] or source


def _standard_label(standard: str) -> str:
    """Map the detector's 'Unknown' to a natural prompt label."""
    return "Ada" if standard in ("Unknown", "") else standard


def load_agent_skill_guidance(
    source_dirs: list[Path],
    per_file_cap: int = 12000,
    per_source_cap: int = 16000,
) -> tuple[str, dict[str, list[str]]]:
    """Load agent-skill markdown guidance from sibling repos.

    Works across the three repo layouts we consume:
    - ada-spark:       SKILL.md + agent-knowledge/*.md
    - SimpleEnglish:   skills/simple-english/SKILL.md + references
    - skills (AdaCore): plugins/adacore/skills/*/SKILL.md

    Each source repo gets its own *per_source_cap* budget so one large repo
    does not starve the others. Within a repo, files load in discovery order
    until the budget is reached. AdaCore skill files are reordered so the
    proof-relevant skills (gnatprove, alire) load first.

    Returns (guidance_text, per_source_skill_names).
    """
    parts: list[str] = []
    per_source: dict[str, list[str]] = {}

    for root in source_dirs:
        if not root.exists():
            logger.warning("Agent-skill source not found, skipping: %s", root)
            continue
        dir_name = root.name
        docs = _collect_skill_docs(root, dir_name)
        if dir_name == "skills":
            docs = _prioritize_adacore_docs(docs)
        if not docs:
            logger.warning("No skill documents found under %s", root)
            continue

        per_source[dir_name] = []
        source_chars = 0
        for md_path in docs:
            if source_chars >= per_source_cap:
                logger.info(
                    "Guidance budget for %s reached (%d chars) - skipping %s",
                    dir_name, source_chars, md_path,
                )
                break
            try:
                text = md_path.read_text(encoding="utf-8").strip()
            except (UnicodeDecodeError, OSError):
                continue
            if not text:
                continue
            skill_name = _skill_display_name(dir_name, md_path)
            per_source[dir_name].append(skill_name)
            capped = text[:per_file_cap]
            parts.append(f"### Skill: {skill_name}\n\n{capped}")
            source_chars += len(capped)

    if parts:
        logger.info(
            "Loaded %d skill documents (%d chars total) from %s",
            len(parts),
            sum(len(p) for p in parts),
            ", ".join(str(d) for d in source_dirs),
        )
    return "\n\n".join(parts), per_source


def _prioritize_adacore_docs(docs: list[Path]) -> list[Path]:
    """Order AdaCore skill docs so gnatprove and alire load within budget."""
    def order_key(path: Path) -> tuple[int, str]:
        for i, skill in enumerate(_ADACORE_SKILL_PRIORITY):
            if f"/skills/{skill}/" in str(path):
                return (i, str(path))
        return (len(_ADACORE_SKILL_PRIORITY), str(path))
    return sorted(docs, key=order_key)


def _skill_display_name(dir_name: str, path: Path) -> str:
    """Human-readable skill name for headers, e.g. 'gnatprove' or 'simple-english'."""
    if dir_name == "skills":
        # plugins/adacore/skills/<name>/SKILL.md
        for parent in path.parents:
            if parent.parent.name == "skills" and parent.name in {
                "gnatprove", "alire", "gnatdoc", "gnattest", "gnatfuzz",
            }:
                return parent.name
        return path.parent.name
    if dir_name == "SimpleEnglish":
        return "simple-english"
    if dir_name == "ada-spark":
        return path.stem if path.stem != "SKILL" else "ada-spark"
    return path.stem


def load_adacore_toolchain_docs() -> dict[str, str]:
    """Load the AdaCore skills (../skills) for toolchain QA generation.

    Returns a mapping skill name -> concatenated SKILL.md + reference text.
    Used by build_toolchain_qa_turns() to produce question/answer turns that
    teach gnatprove, alire, gnatdoc, gnattest, and gnatfuzz usage.
    """
    docs: dict[str, str] = {}
    skills_root = ADACORE_SKILLS_DIR / "plugins" / "adacore" / "skills"
    if not skills_root.exists():
        logger.info("AdaCore skills not found at %s - skipping toolchain QA", skills_root)
        return docs

    for skill_dir in sorted(skills_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        chunks: list[str] = []
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            try:
                chunks.append(skill_md.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, OSError):
                pass
        refs = sorted((skill_dir / "references").glob("*.md")) if (skill_dir / "references").is_dir() else []
        for ref in refs[:_ADACORE_REF_LIMIT]:
            try:
                text = ref.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            chunks.append(text[:_ADACORE_REF_MAX_CHARS])
        if chunks:
            docs[skill_dir.name] = "\n\n".join(chunks)
            logger.info(
                "Loaded AdaCore skill '%s' (%d chars, %d reference files)",
                skill_dir.name, len(docs[skill_dir.name]), len(chunks) - 1,
            )
    return docs


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("q3as_build_dataset")


# --------------------------------------------------------------------------- #
# Evaluation Methodology Loader
# --------------------------------------------------------------------------- #

def load_eval_methodology() -> dict[str, Any]:
    """Load evaluation methodology and dataset configuration from ../ada-eval.

    Reads the ada-eval project structure to derive:
    - Dataset splitting strategy
    - Evaluation metric definitions
    - Sample categories (spark_learn, spark_custom, spark_human_eval_silver)

    Returns a dict with methodology metadata. Returns empty dict if
    ada-eval is not found.
    """
    if not ADA_EVAL_DIR.exists():
        logger.info("ada-eval not found at %s: skipping methodology load", ADA_EVAL_DIR)
        return {}

    methodology: dict[str, Any] = {"source": str(ADA_EVAL_DIR)}

    # Discover compacted JSONL datasets
    compacted_dir = ADA_EVAL_DIR / "data" / "base" / "compacted"
    if compacted_dir.exists():
        jsonl_files = list(compacted_dir.glob("*.jsonl"))
        methodology["compacted_datasets"] = [f.name for f in jsonl_files]
        logger.info(
            "Found %d compacted datasets in ada-eval: %s",
            len(jsonl_files), methodology["compacted_datasets"],
        )

    # Discover expanded sample categories
    expanded_dir = ADA_EVAL_DIR / "data" / "base" / "expanded"
    if expanded_dir.exists():
        categories = [d.name for d in expanded_dir.iterdir() if d.is_dir()]
        methodology["expanded_categories"] = categories
        logger.info(
            "Found %d expanded categories in ada-eval: %s",
            len(categories), categories,
        )

    # Load any available evaluation config
    for config_file in [ADA_EVAL_DIR / "pyproject.toml", ADA_EVAL_DIR / "setup.py"]:
        if config_file.exists():
            methodology["config_file"] = str(config_file)
            break

    return methodology


# --------------------------------------------------------------------------- #
# File Discovery
# --------------------------------------------------------------------------- #

def discover_files(root: Path) -> dict[str, list[Path]]:
    """Walk *root* and classify discovered Ada source files by category.

    Only discovers .ads, .adb, and .gpr files. Non-Ada files are skipped.
    Symlinks are resolved automatically via ``Path.resolve()``.
    """
    discovered: dict[str, list[Path]] = {"ads": [], "adb": [], "gpr": []}

    if not root.exists():
        logger.warning("Input directory does not exist: %s", root)
        return discovered

    for dirpath, dirnames, filenames in os_walk_safe(root):
        for fname in filenames:
            full_path = (Path(dirpath) / fname).resolve()
            ext = full_path.suffix.lower()

            if ext in ADA_SPEC_EXTENSIONS:
                discovered["ads" if ext == ".ads" else "adb"].append(full_path)
            elif ext in PROJECT_EXTENSIONS:
                discovered["gpr"].append(full_path)

    logger.info(
        "Discovered %d .ads, %d .adb, %d .gpr files in %s",
        len(discovered["ads"]), len(discovered["adb"]),
        len(discovered["gpr"]), root,
    )
    return discovered


def os_walk_safe(root: Path):
    """Safely walk a directory tree, pruning large/dangerous directories.

    Uses os.walk with in-place directory pruning to skip .git, .cache,
    __pycache__, .pytest_cache, and .github without descending into them.
    """
    skip_dirs = {".git", ".cache", "__pycache__", ".pytest_cache", ".github"}
    try:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            yield dirpath, dirnames, filenames
    except PermissionError:
        logger.warning("Permission denied walking: %s", root)
        return
    except OSError as exc:
        logger.warning("OS error walking %s: %s", root, exc)
        return


# --------------------------------------------------------------------------- #
# File Pairing
# --------------------------------------------------------------------------- #

def pair_files(discovered: dict[str, list[Path]]) -> list[dict[str, Path | str | None]]:
    """Pair .ads (specification) with .adb (body) files by package name.

    If a direct package-name match fails, falls back to filename convention
    (e.g., ``foo.ads`` pairs with ``foo.adb``). Unpaired files are emitted
    as standalone entries.
    """
    pairs: list[dict[str, Path | str | None]] = []
    ads_files = discovered.get("ads", [])
    adb_files = discovered.get("adb", [])

    adb_index: dict[str, Path] = {p.stem: p for p in adb_files}
    paired_adb: set[str] = set()

    for ads_path in ads_files:
        stem = ads_path.stem
        adb_path = adb_index.get(stem)
        package_name = extract_package_name(ads_path)

        if adb_path and adb_path.name not in paired_adb:
            paired_adb.add(adb_path.name)
            pairs.append({"spec": ads_path, "impl": adb_path, "package": package_name})
        elif adb_path and adb_path.name in paired_adb:
            alt_ads = [p for p in ads_files if p.stem == stem and p != ads_path]
            if not alt_ads:
                pairs.append({"spec": ads_path, "impl": None, "package": package_name})
        else:
            pairs.append({"spec": ads_path, "impl": None, "package": package_name})

    for adb_path in adb_files:
        if adb_path.name not in paired_adb:
            package = extract_package_name(adb_path)
            pairs.append({"spec": None, "impl": adb_path, "package": package})

    logger.info("Paired %d spec/impl file sets.", len(pairs))
    return pairs


def extract_package_name(file_path: Path) -> str | None:
    """Extract the Ada package name from a source file's first declaration.

    Falls back to the filename stem if no explicit package declaration is found.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return file_path.stem

    m = re.search(
        r"(?:package\s+body\s+)?package\s+(\w+)\s+is",
        content,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)

    return file_path.stem


# --------------------------------------------------------------------------- #
# Content Sanitization
# --------------------------------------------------------------------------- #

def is_binary(file_path: Path) -> bool:
    """Check if a file contains binary data by inspecting magic bytes."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(4)
        for magic in _BINARY_MAGIC:
            if head.startswith(magic):
                return True
    except OSError:
        return True
    return False


def sanitize_content(content: str) -> str:
    """Remove secrets and non-UTF-8 artifacts from source content.

    Returns the sanitized content or raises ValueError if the file is
    deemed unparseable.
    """
    sanitized = _SECRET_PATTERNS.sub("[REDACTED]", content)

    if sanitized.count("\x00") > len(sanitized) * 0.01:
        raise ValueError("File contains excessive null bytes: likely binary/corrupted")

    return sanitized


def read_and_sanitize(file_path: Path) -> str | None:
    """Read a source file, sanitize it, and return the cleaned content.

    Returns ``None`` if the file should be skipped (binary, empty, unparseable).
    """
    if not file_path.exists():
        logger.warning("File does not exist: %s", file_path)
        return None

    if is_binary(file_path):
        logger.warning("Skipping binary file: %s", file_path)
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Skipping non-UTF-8 file: %s", file_path)
        return None
    except OSError as exc:
        logger.warning("Cannot read file %s: %s", file_path, exc)
        return None

    if not content.strip():
        logger.warning("Skipping empty file: %s", file_path)
        return None

    try:
        return sanitize_content(content)
    except ValueError as exc:
        logger.warning("Skipping unparseable file %s: %s", file_path, exc)
        return None


# --------------------------------------------------------------------------- #
# Code-block comment cleaning (STE for inline comments)
# --------------------------------------------------------------------------- #

_COMMENT_RE = re.compile(r"^(\s*)--(.*)$")


def strip_noisy_comments(code: str) -> str:
    """Clean inline comments so the model learns STE-compliant comments.

    Rewrites -- comment lines: removes AI-slop words, hedges, em-dashes,
    and semicolons. Leaves code and non-comment lines untouched. Comments
    that become empty after cleaning are dropped.
    """
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        m = _COMMENT_RE.match(line)
        if not m:
            cleaned_lines.append(line)
            continue
        indent, body = m.group(1), m.group(2)
        if not body.strip() or re.fullmatch(r"\s*[-=*#]+\s*", body):
            # Pure decoration (e.g. -- ----) or empty comment: drop it.
            continue
        cleaned = sanitize_prose(body.strip())
        if not cleaned:
            continue
        # Guarantee sentence casing only if the original had it; preserve
        # identifiers exactly (sanitize_prose never edits them).
        cleaned_lines.append(f"{indent}-- {cleaned}")
    return "\n".join(cleaned_lines)


# --------------------------------------------------------------------------- #
# Defect Injection: correct-vs-wrong training pairs
# --------------------------------------------------------------------------- #
# The model must recognize broken code, not only produce correct code.
# Each defect family below maps a correct snippet to a broken variant plus
# an STE-compliant diagnosis naming the error the compiler or gnatprove
# would report. Defects are chosen per-snippet by applicability checks, so
# every broken variant is plausible and the diagnosis is factual.

_DEFECT_FAMILIES = (
    "syntax", "context", "visibility", "contract", "mismatch",
    "typo", "hallucinated", "ordering", "scoping", "type", "lang_confusion",
    "wrong_ref", "nonexistent_call", "bad_typing", "arity", "stray_aspect",
    "old_misuse",
)

# Roots of the GNAT predefined library. A with clause corrupted under one of
# these roots makes GNAT report the unit as not predefined; any other root
# makes it search for a source file. The injector only corrupts predefined
# units so the claimed message is deterministic.
_PREDEFINED_ROOTS = ("Ada", "System", "GNAT", "Interfaces")

# Verified against GNAT 14 (see scripts/validate_defects.py):
# - a misspelled identifier:            error: "Xx" is undefined
# - a hallucinated predefined unit:     error: "Ada.Text_Iox" is not a
#                                       predefined library unit
# - a declaration after begin:          error: declarations mixed with
#                                       statements is a GNAT-specific
#                                       extension
# - a numeric literal in a Boolean:     error: expected type
#                                       "Standard.Boolean"
# - a C-style '=' assignment:           error: "=" should be ":="
# - a wrong selected component:         error: "Foo_Line" not declared in
#                                       "Text_IO"
# - a call to an undeclared name:       error: "Compute_Stat" is undefined
# - a string into a numeric object:     error: expected type
#                                       "Standard.Integer" / found a
#                                       string type
# - extra argument in a call:           error: too many arguments in call
#                                       to "P"
# - aspect inside a package spec:       error: aspect specifications not
#                                       allowed here
# - 'Old outside a postcondition:       error: attribute "Old" can only
#                                       appear in postcondition


def _gnat_unit_display(unit: str) -> str:
    """Return *unit* the way GNAT echoes it in error messages.

    GNAT canonicalizes identifiers the way it builds file names: the first
    letter and letters right after underscores stay uppercase, other
    uppercase letters are lowercased (verified: 'Ada.Text_IOx' is echoed as
    'Ada.Text_Iox'). The claimed compiler message must use the echoed form.
    """
    out: list[str] = []
    upper_next = True
    for ch in unit:
        if ch in "_.":
            upper_next = True
            out.append(ch)
        elif upper_next and ch.isalpha():
            out.append(ch.upper())
            upper_next = False
        else:
            out.append(ch.lower())
    return "".join(out)


def _code_without_comments(code: str) -> str:
    """Strip -- line comments so name searches do not hit prose."""
    return re.sub(r"--[^\n]*", "", code)


_COMMENT_SPAN = re.compile(r"--[^\n]*")
# Ada strings are single-line; "" is the escaped quote.
_STRING_SPAN = re.compile(r'"(?:[^"]|"")*"')


def _protected_spans(code: str) -> list[tuple[int, int]]:
    """Spans of comments and string literals: identifiers there are prose."""
    spans = [(m.start(), m.end()) for m in _COMMENT_SPAN.finditer(code)]
    spans.extend((m.start(), m.end()) for m in _STRING_SPAN.finditer(code))
    return spans


def _first_use(code: str, name: str, start: int, spans: list[tuple[int, int]]) -> int | None:
    """Offset of the first real (non-comment, non-string) use of *name*.

    Offsets are computed directly on *code* so they can be used to cut the
    original text: searching comment-stripped text and applying the index
    to the original corrupts an unrelated span whenever a comment precedes
    the use site.
    """
    for match in re.finditer(rf"\b{re.escape(name)}\b", code[start:]):
        pos = start + match.start()
        if not any(s <= pos < e for s, e in spans):
            return pos
    return None


def _extract_with_clauses(code: str) -> list[str]:
    return re.findall(r"^\s*with\s+([\w.]+)\s*;", code, re.MULTILINE)


def _first_subprogram(code: str) -> tuple[str, str] | None:
    """Return (name, 'procedure'|'function') of the first subprogram declaration."""
    m = re.search(r"\b(procedure|function)\s+(\w+)", code)
    if m:
        return m.group(2), m.group(1)
    return None


def _split_context_and_unit(code: str) -> tuple[str, str]:
    """Split Ada source into (context clauses, rest of the compilation unit).

    The context part keeps its trailing newline so that clause-removal
    regexes operating on it can anchor on whole lines.
    """
    lines = code.splitlines()
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith(("with ", "use ", "pragma ", "--")) or not stripped:
            idx += 1
        else:
            break
    context = "\n".join(lines[:idx]) + ("\n" if lines[:idx] else "")
    unit = "\n".join(lines[idx:])
    return context, unit


def inject_defect(code: str, family: str) -> tuple[str, str, str] | None:
    """Inject one defect of *family* into *code*.

    Returns (broken_code, diagnosis, compiler_expectation) or None when the
    family does not apply to this snippet. The diagnosis is written in the
    STE style this project trains on: short sentences, active voice, no
    em-dashes, no hedges.
    """
    context, unit = _split_context_and_unit(code)

    if family == "syntax":
        # Drop the 'is' of a package declaration: a classic compile error
        # (verified against GNAT 14 for spec, body, and 'with aspect'
        # forms: the compiler reports missing "is" at the unit name).
        # The 'is' must follow the unit name on the SAME line: a later
        # line 'is' would belong to a nested construct, and removing the
        # wrong 'is' leaves the snippet valid (verified: a body whose
        # spec is absent kept compiling with the mutated line inside it).
        m = re.search(r"(\bpackage\s+\w+(\s+\w+)?)\s+is\b[^\n]*$", code, re.MULTILINE)
        if m:
            broken = code[:m.start()] + m.group(1) + " " + code[m.end():]
            return (
                broken,
                (
                    "The package declaration misses the keyword is. "
                    'The compiler reports missing "is" at the unit name.'
                ),
                'error: missing "is"',
            )
        # Drop the semicolon of a simple object declaration.
        m = re.search(r"^\s*(\w[\w.]*)\s*:\s*(?:constant\s+)?[\w.]+", code, re.MULTILINE)
        if m and m.end() < len(code) and code[m.end()] == ";":
            broken = code[:m.end()] + code[m.end() + 1:]
            if broken.strip():
                return (
                    broken,
                    (
                        "The declaration misses its terminating semicolon. "
                        "The compiler reports the error at the next token."
                    ),
                    'error: missing ";"',
                )
        return None

    if family == "context":
        withs = _extract_with_clauses(code)
        referenced = [w for w in withs if re.search(r"\b" + re.escape(w.split(".")[-1]) + r"\b", unit)]
        # A with clause carried by both spec and body survives the removal
        # of its first copy: the snippet would compile clean and the claimed
        # error would be a false claim.
        referenced = [
            w for w in referenced
            if len(re.findall(rf"^\s*with\s+{re.escape(w)}\s*;", code, re.MULTILINE)) == 1
        ]
        if referenced:
            victim = referenced[0]
            broken_context = re.sub(
                rf"^(\s*)with\s+{re.escape(victim)}\s*;\s*\n", "", context, flags=re.MULTILINE
            )
            if broken_context == context:
                return None
            broken = broken_context + unit
            # The compiler message depends on how the code names the unit
            # (verified against GNAT 14):
            # - full dotted name (Ada.Text_IO.Put_Line): the parent chain is
            #   undefined, plus the missing-with hint.
            # - short prefix (Text_IO.Put_Line) or the unit name itself
            #   (Spark_Discrete_Set.Contains): the name is not visible.
            # - dot-free names: a remaining use clause anchors the
            #   missing-with hint.
            last = victim.split(".")[-1]
            full_prefix_used = "." in victim and bool(
                re.search(rf"(?<![.\w]){re.escape(victim)}\s*\.", unit)
            )
            short_prefix_used = bool(
                re.search(rf"(?<![.\w]){re.escape(last)}\s*\.", unit)
            )
            if full_prefix_used:
                return (
                    broken,
                    (
                        f"The code uses {victim} but the context clause does not import it. "
                        "The compiler reports an undefined name and names the missing "
                        "with clause."
                    ),
                    f'error: missing "with {victim};"',
                )
            if short_prefix_used:
                return (
                    broken,
                    (
                        f"The code uses {victim} through the short prefix {last} "
                        "but the context clause does not import it. "
                        f'The compiler reports {last} as not visible.'
                    ),
                    f'error: "{last}" is not visible',
                )
            return (
                broken,
                (
                    f"The code uses names of {victim} but the context clause "
                    "does not import it. The compiler names the missing "
                    "with clause."
                ),
                f'error: missing "with {victim};"',
            )
        return None

    if family == "visibility":
        # Verified against GNAT 14: removing the use clause while dot-free
        # names of the package stay in use yields
        #   error: "Put_Line" is not visible
        #   error: non-visible declaration at <pkg spec>:<line>
        # The with-clause case is the context family's job: it knows how
        # the code names the unit and picks the matching message.
        use_m = re.search(r"^\s*use\s+([\w.]+)\s*;", code, re.MULTILINE)
        if use_m:
            used_pkg = use_m.group(1)
            # A dotted reference through the package prefix makes the use
            # clause redundant: removing it changes nothing.
            short = used_pkg.split(".")[-1]
            has_dotted_ref = bool(re.search(rf"(?<![.\w]){re.escape(short)}\.[A-Za-z_]", unit))
            # Dot-free reference = a call statement: an identifier that is
            # not a language keyword and is immediately followed by an
            # argument parenthesis.
            keywords = {
                "begin", "end", "is", "in", "out", "null", "return",
                "procedure", "function", "package", "type", "subtype",
                "declare", "if", "then", "else", "elsif", "for", "while",
                "loop", "case", "when", "exit", "pragma", "new", "range",
                # Reserved words that can legitimately precede "(" in real
                # code but never start a call: separate (Parent), not (X).
                "separate", "overriding", "generic", "not", "and", "or",
                "xor", "mod", "rem", "abs", "delay", "raise", "select",
                "accept", "entry", "task", "protected", "body", "renames",
                "access", "all", "abstract", "limited", "at", "goto",
            }
            candidates = re.findall(r"(?<![.\w:])([A-Za-z_]\w*)\s*\(", unit)
            # Names declared inside the snippet stay visible without the
            # use clause, so they are not offenders.
            declared = set(re.findall(r"\b(?:procedure|function)\s+(\w+)", code))
            declared |= {
                m for m in re.findall(
                    r"^\s*(\w+)\s*:\s*(?:constant\s+)?[\w.]+", unit, re.MULTILINE,
                )
            }
            offenders = [
                c for c in candidates
                if c.lower() not in keywords and c not in declared
            ]
            # A use clause carried by both spec and body survives the
            # removal of one copy: skip duplicated use clauses.
            if len(re.findall(rf"^\s*use\s+{re.escape(used_pkg)}\s*;", code, re.MULTILINE)) > 1:
                return None
            if not has_dotted_ref and offenders:
                broken = re.sub(
                    rf"^\s*use\s+{re.escape(used_pkg)}\s*;\s*\n", "", code, flags=re.MULTILINE
                )
                if broken != code:
                    # The first offender in source order is the first name
                    # the compiler rejects.
                    return (
                        broken,
                        (
                            f"The code references names of {used_pkg} directly. "
                            "The use clause that makes them visible is missing. "
                            f'The compiler reports {offenders[0]} as not visible.'
                        ),
                        f'error: "{offenders[0]}" is not visible',
                    )
                return None
        return None

    if family == "contract":
        # Precondition aspect misspelled: GNAT 14 emits 'not a valid aspect
        # identifier' as a warning (with a misspelling hint) and then
        # IGNORES the clause. The harm is silent: the subprogram compiles
        # with no contract, and gnatprove has nothing to check. In SPARK
        # builds with warnings as errors the build stops.
        m = re.search(r"\b(Pre|Post)\s*=>", code)
        if m:
            aspect = m.group(1)
            broken = code[:m.start()] + f"{aspect}s =>" + code[m.end():]
            return (
                broken,
                (
                    f"The aspect name {aspect}s does not exist. "
                    f"The correct aspect for the {('entry' if aspect == 'Pre' else 'return')} "
                    f"check is {aspect}. The compiler emits a warning and then "
                    "ignores the clause. The subprogram keeps no contract, so "
                    "gnatprove has nothing to check. This is a silent failure."
                ),
                f'warning: "{aspect}s" is not a valid aspect identifier',
            )
        return None

    if family == "mismatch":
        # Remove a parameter from a procedure body that the spec declares:
        # spec/body profile mismatch (verified against GNAT 14: 'not type
        # conformant with declaration' + 'too few parameters'). We must
        # edit the BODY occurrence, not the spec declaration: find the
        # procedure inside 'package body ... is' whose parameter list has
        # at least two parameters, and drop the first one there.
        body_m = re.search(r"\bpackage\s+body\s+\w+\s+is\b", code)
        search_from = body_m.start() if body_m else 0
        for m in re.finditer(r"\bprocedure\s+(\w+)\s*\(([^)]*)\)", code[search_from:]):
            param_list = [p.strip() for p in m.group(2).split(";")]
            if len(param_list) < 2 or not param_list[0]:
                continue
            dropped_name = param_list[0].split(":")[0].strip()
            new_params = "; ".join(param_list[1:])
            abs_start = search_from + m.start(2)
            abs_end = search_from + m.end(2)
            broken = code[:abs_start] + new_params + code[abs_end:]
            return (
                broken,
                (
                    "The body parameter list misses the parameter "
                    f"{dropped_name} that the specification declares. The "
                    "profiles do not match. The compiler reports the body "
                    "as not type conformant with the declaration."
                ),
                "error: not type conformant with declaration",
            )
        return None

    if family == "typo":
        # Misspell one identifier at a single use site (the declaration and
        # the other references keep the correct spelling). GNAT reports the
        # typo'd name as undefined. Use sites are located span-aware on the
        # original text: a comment-stripped index would corrupt unrelated
        # code whenever a comment precedes the use site.
        spans = _protected_spans(code)
        for dm in re.finditer(r"^\s*(\w+)\s*:\s*(?:constant\s+)?[\w.]+", code, re.MULTILINE):
            name = dm.group(1)
            if len(name) < 3:
                continue
            typo = name[0] + name[2] + name[1] + name[3:]
            if typo == name or _first_use(code, typo, 0, spans) is not None:
                continue
            pos = _first_use(code, name, dm.end(), spans)
            if pos is None:
                continue
            broken = code[:pos] + typo + code[pos + len(name):]
            return (
                broken,
                (
                    f"The name {name} is misspelled as {typo} at one use site. "
                    "The declaration and the other references keep the original "
                    f"spelling. The compiler reports {typo} as undefined."
                ),
                f'error: "{typo}" is undefined',
            )
        return None

    if family == "hallucinated":
        # Corrupt one predefined with clause into a unit that does not
        # exist: the model-style hallucination of a plausible library name.
        # Only predefined roots are corrupted: GNAT then reports the unit
        # as not predefined, a deterministic message. A non-predefined root
        # would make the message depend on whether the root file resolves.
        for victim in _extract_with_clauses(code):
            root_component = victim.split(".")[0]
            if root_component not in _PREDEFINED_ROOTS:
                continue
            if len(re.findall(rf"^\s*with\s+{re.escape(victim)}\s*;", code, re.MULTILINE)) > 1:
                continue  # the body's duplicate with keeps the build alive
            bad_unit = f"{victim}x"
            display = _gnat_unit_display(bad_unit)
            broken = re.sub(
                rf"^\s*with\s+{re.escape(victim)}\s*;",
                f"with {bad_unit};",
                code,
                count=1,
                flags=re.MULTILINE,
            )
            if broken == code:
                return None
            return (
                broken,
                (
                    f"The context clause names {display}, a unit that does "
                    "not exist. The intended unit is the one the code uses. "
                    "The compiler reports the name as not a predefined "
                    "library unit."
                ),
                f'error: "{display}" is not a predefined library unit',
            )
        return None

    if family == "ordering":
        # Move one object declaration out of a subprogram's declarative part
        # and drop it between the statements: Standard Ada requires
        # declarations to precede the statements of the same part.
        body_m = re.search(
            r"\b(?:procedure|function)\s+\w+[^\n]*\bis\b(.*?)\bbegin\b(.*?)\bend\b",
            code, re.DOTALL,
        )
        if body_m:
            decl_m = re.search(
                r"^\s*\w[\w.]*\s*:\s*[\w.]+[^;]*;\s*$", body_m.group(1), re.MULTILINE,
            )
            if decl_m:
                line_text = decl_m.group(0).strip()
                # Remove exactly the declaration LINE (the aspect-spanning
                # match may include surrounding whitespace/newlines), then
                # re-insert it before the first end after begin.
                line_m = re.search(
                    rf"^[ \t]*{re.escape(line_text)}[ \t]*$", code, re.MULTILINE,
                )
                if line_m:
                    line_end = code.find("\n", line_m.end())
                    broken = code[:line_m.start()] + (
                        "" if line_end == -1 else code[line_end + 1:]
                    )
                    begin_m = re.search(r"\bbegin\b", broken)
                    end_m = re.search(r"\bend\b", broken[begin_m.end():]) if begin_m else None
                    if begin_m and end_m:
                        insert_at = begin_m.end() + end_m.start()
                        broken = (
                            broken[:insert_at]
                            + "   " + line_text + "\n"
                            + broken[insert_at:]
                        )
                        if broken != code and broken.strip():
                            return (
                                broken,
                                (
                                    "A declaration sits between the statements. "
                                    "Standard Ada requires declarations before the "
                                    "statements of the same part. The compiler "
                                    "reports the mixed declarations and refuses the "
                                    "unit."
                                ),
                                "error: declarations mixed with statements",
                            )
        return None

    if family == "scoping":
        # Remove an object declaration whose name stays in use: the name is
        # no longer visible in the scope that needs it.
        spans = _protected_spans(code)
        for dm in re.finditer(r"^\s*(\w+)\s*:\s*(?:constant\s+)?[\w.]+", code, re.MULTILINE):
            name = dm.group(1)
            # Declaration shape only: "Name :=" (an assignment statement)
            # also contains a colon, so the type name after it must be part
            # of the pattern or every assignment inflates the count.
            decl_count = len(re.findall(rf"^\s*{name}\s*:\s*(?:constant\s+)?[\w.]", code, re.MULTILINE))
            if decl_count != 1 or _first_use(code, name, dm.end(), spans) is None:
                continue
            line_start = code.rfind("\n", 0, dm.start()) + 1
            line_end = code.find("\n", dm.end())
            broken = code[:line_start] + ("" if line_end == -1 else code[line_end + 1:])
            if broken == code or not broken.strip():
                continue
            return (
                broken,
                (
                    f"The declaration of {name} is missing from the scope that "
                    f"uses it. The remaining references to {name} name an "
                    "entity the compiler cannot see."
                ),
                f'error: "{name}" is undefined',
            )
        return None

    if family == "type":
        # Change a numeric object's type to Boolean: the numeric initializer
        # or the later numeric assignment no longer matches. Only corrupt an
        # object that demonstrably receives a numeric value, so the broken
        # variant cannot compile clean.
        numeric = r"(?:Integer|Natural|Positive|Long_Integer|Long_Long_Integer|Short_Integer|Float|Long_Float)"
        for dm in re.finditer(rf"^\s*(\w+)\s*:\s*({numeric})\s*(:=[^;]*)?;", code, re.MULTILINE):
            name, old_type, initializer = dm.group(1), dm.group(2), dm.group(3) or ""
            rest = _code_without_comments(code[dm.end():])
            gets_numeric = bool(
                re.search(r":=\s*[-+]?\d", initializer)
                or re.search(rf"\b{name}\s*:=\s*[-+]?\d", rest)
            )
            if not gets_numeric or re.search(rf"\b{name}\s*:=\s*(?:True|False)\b", rest):
                continue
            broken = code[:dm.start(2)] + "Boolean" + code[dm.end(2):]
            if initializer.strip():
                where = f"the initializer of {name}"
            else:
                where = f"the later assignments to {name}"
            return (
                broken,
                (
                    f"The object {name} is declared Boolean but was {old_type}. "
                    f"{where.capitalize()} supply a numeric value. The compiler "
                    "expects a Boolean type and rejects the unit."
                ),
                'error: expected type "Standard.Boolean"',
            )
        return None

    if family == "lang_confusion":
        # Replace the first statement-position ':=' with '=': the C-style
        # assignment operator does not exist in Ada. Occurrences inside
        # string literals are skipped: rewriting a string cannot fail the
        # build and would fake a clean compile.
        begin_m = re.search(r"\bbegin\b", code)
        if not begin_m:
            return None
        spans = _protected_spans(code)
        for am in re.finditer(r":=", code[begin_m.end():]):
            pos = begin_m.end() + am.start()
            if any(s <= pos < e for s, e in spans):
                continue  # inside a comment or string literal
            broken = code[:pos] + "=" + code[pos + 2:]
            return (
                broken,
                (
                    "The statement assigns with =, the C-style operator. Ada "
                    "writes assignment as := and uses = for equality only. The "
                    "compiler reports the operator and suggests the Ada form."
                ),
                'error: "=" should be ":="',
            )
        return None

    if family == "wrong_ref":
        # Corrupt a selected component name: Foo_Line -> Fo_Line. The unit
        # compiles, the component does not exist in it. Only predefined
        # units are corrupted so the claimed message is deterministic.
        spans = _protected_spans(code)
        for wm in re.finditer(r"\b(Ada|System|GNAT|Interfaces)[\w.]*\.\s*(\w+)", code):
            unit, component = wm.group(1), wm.group(2)
            prefix = code[wm.start():wm.start(2)].rstrip(". \t")
            pos = wm.start(2)
            if any(s <= pos < e for s, e in spans) or len(component) < 4:
                continue
            if not re.search(r"^\s*with\s+" + re.escape(prefix) + r"\s*;", code, re.MULTILINE):
                continue
            cut = max(1, len(component) // 4)
            corrupted = component[:len(component) - cut]
            if corrupted == component:
                continue
            broken = code[:wm.start(2)] + corrupted + code[wm.end(2):]
            display = prefix.split(".")[-1]
            return (
                broken,
                (
                    f"The code calls {corrupted}, which is not declared in the "
                    f"unit {display}. The reference points at a component that "
                    "does not exist. The compiler reports the missing "
                    "declaration."
                ),
                f'error: "{corrupted}" not declared in "{display}"',
            )
        return None

    if family == "nonexistent_call":
        # Corrupt the name of a locally declared procedure at a real call
        # site (never an ``end`` label); the call then names an entity that
        # does not exist.
        spans = _protected_spans(code)
        for pm in re.finditer(r"^\s*procedure\s+(\w+)\s*(?:\([^)]*\))?\s*is\b", code, re.MULTILINE):
            name = pm.group(1)
            if len(name) < 4:
                continue
            for call_m in re.finditer(rf"(?<!end\s)\b{name}\b\s*(?:\([^)]*\))?\s*;", code):
                pos = call_m.start()
                if any(s <= pos < e for s, e in spans):
                    continue
                before = code[max(0, pos - 30):pos]
                if re.search(r"\bend\s*$", before):
                    continue  # an end label, not a call
                cut = max(1, len(name) // 3)
                corrupted = name[:len(name) - cut]
                if corrupted == name:
                    continue
                broken = code[:pos] + corrupted + code[pos + len(name):]
                if broken != code:
                    return (
                        broken,
                        (
                            f"The code calls {corrupted}, which is not declared "
                            f"anywhere in scope. The declared procedure is {name}. "
                            "A wrong or invented name in a call produces this "
                            "error."
                        ),
                        f'error: "{_gnat_unit_display(corrupted)}" is undefined',
                    )
        return None

    if family == "bad_typing":
        # Assign a string literal to a numeric object: the value's type
        # cannot match the declared numeric type.
        begin_m = re.search(r"\bbegin\b", code)
        if begin_m:
            numeric = r"(?:Integer|Natural|Positive|Long_Integer|Long_Long_Integer|Short_Integer|Float|Long_Float)"
            for dm in re.finditer(rf"^\s*(\w+)\s*:\s*{numeric}\s*(:=[^;]*)?;", code[:begin_m.start()], re.MULTILINE):
                name = dm.group(1)
                assign_m = re.search(rf"\b{name}\s*:=\s*[^;]+;", code[begin_m.end():])
                if not assign_m:
                    continue
                pos = begin_m.end() + assign_m.start()
                assign_part = assign_m.group(0)
                rhs_m = re.search(r":=\s*(.+?);", assign_part, re.DOTALL)
                rhs = rhs_m.group(1).strip() if rhs_m else ""
                if rhs.startswith('"') and rhs.endswith('"'):
                    continue  # already a string; nothing to corrupt
                new_rhs = '"' + rhs.strip('"\'') + '"'
                broken = code[:pos] + re.sub(r":=\s*.+?;", f":= {new_rhs};", assign_part, flags=re.DOTALL) + code[pos + len(assign_part):]
                if broken != code:
                    return (
                        broken,
                        (
                            f"The statement assigns a string value to {name}, a "
                            "numeric object. The value type cannot match the "
                            "declared type. The compiler expects the numeric "
                            "type and rejects the string."
                        ),
                        f'error: expected type "Standard.{_old_type_for(name, code)}"',
                    )
        return None

    if family == "arity":
        # Add a spurious argument to a call of a locally declared
        # subprogram: the visible declaration cannot accept it.
        begin_m = re.search(r"\bbegin\b", code)
        if begin_m:
            for pm in re.finditer(r"^\s*(?:procedure|function)\s+(\w+)\s*\(([^)]*)\)", code[:begin_m.start()], re.MULTILINE):
                name, params = pm.group(1), pm.group(2)
                if not params.strip():
                    continue
                arg_call: re.Match[str] | None = re.search(rf"\b{name}\s*(\([^)]*\))\s*;", code[begin_m.end():])
                if not arg_call or "=>" in arg_call.group(1):
                    # Named associations would trigger a different error
                    # (positional after named), not the arg-count one.
                    continue
                pos = begin_m.end() + arg_call.start()
                broken = (
                    code[:pos]
                    + re.sub(r"\(([^)]*)\)\s*;", lambda mm: "(" + mm.group(1) + ", 0);", arg_call.group(0))
                    + code[pos + len(arg_call.group(0)):]
                )
                if broken != code:
                    count = len([p for p in params.split(";") if p.strip()])
                    return (
                        broken,
                        (
                            f"The call of {name} passes more arguments than the "
                            f"declaration accepts. The declaration takes {count} "
                            "argument(s). The compiler counts the arguments and "
                            "rejects the call."
                        ),
                        f'error: too many arguments in call to "{_gnat_unit_display(name)}"',
                    )
        return None

    if family == "stray_aspect":
        # Insert an aspect clause where aspects are not allowed: directly
        # inside a package spec body (before the first declaration).
        pkg_m = re.search(r"^\s*package\s+(?:body\s+)?(\w[\w.]*)\s+is\s*$", code, re.MULTILINE | re.IGNORECASE)
        if pkg_m:
            insert_at = pkg_m.end()
            broken = code[:insert_at] + "\n   with Pre => True;" + code[insert_at:]
            if broken != code:
                return (
                    broken,
                    (
                        "An aspect specification sits inside the package "
                        "declarative part. Aspects attach to declarations such "
                        "as subprograms and objects, not to the package itself "
                        "at this position. The compiler rejects the clause."
                    ),
                    "error: aspect specifications not allowed here",
                )
        return None

    if family == "old_misuse":
        # Put 'Old into a precondition: 'Old reads the value at the entry
        # point, and only postconditions can do that. Probed message:
        # error: attribute "Old" can only appear in postcondition.
        if "'Old" in code:
            return None
        fm = re.search(
            r"\bfunction\s+(\w+)\s*\(([^)]*)\)[^;]*?\bwith\s+Pre\s*=>\s*([^;]+);",
            code, re.DOTALL,
        )
        if fm:
            params = fm.group(2)
            param_m = re.search(r"\b(\w+)\s*:\s*", params)
            pre_expr = fm.group(3)
            if param_m and param_m.group(1) in pre_expr and not re.search(r"\b\w+'", pre_expr):
                param = param_m.group(1)
                pre_start = fm.start(3)
                pre_ref: re.Match[str] | None = re.search(rf"\b{param}\b", pre_expr)
                if pre_ref is None:
                    return None
                broken = code[:pre_start + pre_ref.start()] + param + "'Old" + code[pre_start + pre_ref.end():]
                if broken != code:
                    return (
                        broken,
                        (
                            f"The precondition of {fm.group(1)} reads {param}'Old. "
                            "Old refers to the value of an expression at the "
                            "entry point. Only postconditions can read that "
                            "value, so the compiler rejects the clause."
                        ),
                        'error: attribute "Old" can only appear in postcondition',
                    )
        return None

    return None


def _prefix_up_to(code: str, pos: int) -> str:
    """Dotted prefix (Ada, Ada.Text_IO, ...) ending just before *pos*."""
    start = max(code.rfind(";", 0, pos), code.rfind("\n", 0, pos), 0)
    line = code[start:pos]
    m = re.search(r"\b(Ada|System|GNAT|Interfaces)(?:\.\w+)+$", line)
    return m.group(0) if m else (m.group(0) if (m := re.search(r"\b\w+(?:\.\w+)+$", line)) else "")


def _old_type_for(name: str, code: str) -> str:
    """Base type of *name*'s declaration for GNAT's 'expected type' message.

    GNAT reports the base type, not the subtype: an object declared Natural
    expects "Standard.Integer" (verified). Subtypes with a differently
    named root map to that root; everything else is its own base.
    """
    m = re.search(rf"^\s*{name}\s*:\s*(\w+)\s*(:=)?\s*;", code, re.MULTILINE)
    declared = m.group(1) if m else "Integer"
    return {"Natural": "Integer", "Positive": "Integer"}.get(declared, declared)


_DEFECT_FAMILY_ORDER: tuple[str, ...] = _DEFECT_FAMILIES
# Cap defect turns per pair so the dataset does not skew negative.
_MAX_DEFECTS_PER_PAIR = 3

# The same defect question asked several ways: the fine-tune must recognize
# the defect class regardless of the natural-language wrapper around it.
# Selection is content-derived (crc32 of the broken snippet), so the choice
# stays identical no matter how many worker processes built the dataset.
_DEFECT_USER_PHRASINGS = (
    "The following {standard} code for `{package}` does not compile. Find the defect and provide the corrected code.",
    "This {standard} snippet for `{package}` fails to build. Identify the defect and give the corrected code.",
    "Something is wrong with this {standard} code for `{package}`. Find the defect and provide the corrected code.",
    "The compiler rejects this {standard} unit `{package}`. Name the defect and supply the corrected code.",
    "Review this {standard} code for `{package}`. It contains one defect. Find it and provide the corrected code.",
)


def build_defect_turns(
    code: str,
    standard: str,
    package: str | None,
    source: str,
    ste_rules: str,
    glossary: str,
) -> list[dict[str, list[dict[str, str]]]]:
    """Build correct-vs-wrong training turns for one code snippet.

    For each applicable defect family (capped), emit one turn: the user
    asks to find and fix the bug, the assistant shows the broken variant,
    names the error, and provides the corrected code with an STE-compliant
    diagnosis. This teaches the model when code goes wrong instead of
    hallucinating correctness.
    """
    turns: list[dict[str, list[dict[str, str]]]] = []
    emitted = 0
    # Rotate the family order per snippet (content-derived) so the cap does
    # not always starve the same late families on every pair.
    rotation = zlib.crc32(code.encode("utf-8")) % len(_DEFECT_FAMILY_ORDER)
    family_order = _DEFECT_FAMILY_ORDER[rotation:] + _DEFECT_FAMILY_ORDER[:rotation]
    for family in family_order:
        if emitted >= _MAX_DEFECTS_PER_PAIR:
            break
        result = inject_defect(code, family)
        if not result:
            continue
        broken, diagnosis, compiler_error = result
        if not broken.strip() or broken == code:
            continue
        diagnosis = sanitize_prose(diagnosis)
        emitted += 1
        phrase = _DEFECT_USER_PHRASINGS[zlib.crc32(broken.encode("utf-8")) % len(_DEFECT_USER_PHRASINGS)]
        user_msg = (
            phrase.format(standard=_standard_label(standard), package=package or "this unit")
            + "\n\n"
            + f"```ada\n{broken}\n```\n\n"
            + f"Source: {_source_label(source)}"
        )
        assistant_msg = (
            f"The code has this defect. {diagnosis}\n\n"
            f"The compiler reports: `{compiler_error}`\n\n"
            f"Corrected code:\n\n```ada\n{code}\n```"
        )
        turns.append({
            "messages": [
                {"role": "system", "content": _compose_system_prompt(standard, ste_rules, glossary)},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
        })
    return turns


# --------------------------------------------------------------------------- #
# System-prompt composition (STE rules + glossary + optional guidance)
# --------------------------------------------------------------------------- #

def _compose_system_prompt(
    standard: str,
    ste_rules: str,
    glossary: str,
    guidance_text: str = "",
) -> str:
    """Compose a system prompt that carries the STE writing contract.

    The writing rules and terminology glossary are always present. The
    agent-skill guidance (ada-spark current-toolchain map, SimpleEnglish
    full rules) is embedded on a deterministic subset of turns via the
    caller sampling, so the model sees the full guidance without every
    prompt carrying its token cost.
    """
    parts = [
        (
            "You are an Ada/SPARK language expert. You work with the full Ada "
            "language spectrum from Ada 83 to Ada 2022, including SPARK 2014. "
            f"This task targets {_standard_label(standard)}. "
            "You produce safe, correct, standards-compliant Ada code, and you "
            "diagnose broken code accurately. You write explanations that follow "
            "Simplified Technical English (ASD-STE100) rules."
        ),
        ste_rules,
        _TECH_TERM_DEFINITION_PROMPT,
    ]
    if glossary:
        parts.append(glossary)
    if guidance_text:
        parts.append("Additional Ada/SPARK guidance:\n" + guidance_text)
    return "\n\n".join(p for p in parts if p)


def _sample_guidance(guidance_text: str, every_n: int = 4, key: str = "") -> str:
    """Return guidance_text on a deterministic 1-in-N subset of items.

    With *key* (usually the snippet content), the choice derives from its
    crc32, so the subset is identical no matter how many worker processes
    or what order the dataset is built in. Without a key, the module RNG
    decides (serial callers only).
    """
    if not guidance_text:
        return ""
    if key:
        return guidance_text if zlib.crc32(key.encode("utf-8")) % every_n == 0 else ""
    if _rng.random() < 1.0 / every_n:
        return guidance_text
    return ""


# --------------------------------------------------------------------------- #
# Documentation & Guidance Ingestion (learn, agent skills)
# --------------------------------------------------------------------------- #

_RST_ADA_BLOCK = re.compile(
    r".. code-block::\s+(?:ada|Ada)\s*\n\s*\n((?:[^\n]*\n?)+?)(?=\n\S|\Z)",
    re.MULTILINE,
)
_MD_ADA_BLOCK = re.compile(r"```(?:ada|Ada)\s*\n(.*?)```", re.DOTALL)


def _looks_like_ada(code: str) -> bool:
    """Heuristically reject shell transcripts and non-Ada snippets."""
    lowered = code.lower()
    keywords = (
        "procedure", "function", "package", "task", "protected",
        "is begin", "end ", ":=", "with ", "type ", "subtype ",
    )
    return any(k in lowered for k in keywords)


def _ada_blocks_from_file(doc_path: Path) -> list[str]:
    """Extract candidate Ada code blocks from one documentation file.

    Module-level so worker processes can map over files (picklable).
    """
    try:
        content = doc_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    found = _RST_ADA_BLOCK.findall(content) + _MD_ADA_BLOCK.findall(content)
    blocks: list[str] = []
    for raw in found:
        code = textwrap.dedent(raw).strip("\n")
        if not (30 <= len(code) <= 4000):
            continue
        if not _looks_like_ada(code):
            continue
        blocks.append(code)
    return blocks


def _collect_doc_blocks(
    files: list[Path],
    imap=None,
) -> list[str]:
    """Extract and deduplicate Ada code blocks across documentation files.

    *imap* is a ``Pool.imap``-like callable mapping one file to its blocks;
    when absent, files are processed serially. Either way the output order
    follows the input file order, so results are identical.
    """
    blocks: list[str] = []
    seen: set[str] = set()
    per_file = imap(_ada_blocks_from_file, files, chunksize=4) if imap else map(_ada_blocks_from_file, files)
    for file_blocks in per_file:
        for code in file_blocks:
            key = re.sub(r"\s+", " ", code)
            if key in seen:
                continue
            seen.add(key)
            blocks.append(code)
    return blocks


def extract_ada_code_blocks(doc_root: Path, imap=None) -> list[str]:
    """Extract Ada code blocks from RST and Markdown files under doc_root.

    Deduplicates blocks and keeps only snippets that look like Ada units.
    """
    files: list[Path] = []
    for pattern in ("*.rst", "*.md"):
        files.extend(sorted(doc_root.rglob(pattern)))
    blocks = _collect_doc_blocks(files, imap)
    logger.info("Extracted %d unique Ada code blocks from %s", len(blocks), doc_root)
    return blocks


def build_doc_training_turns(
    doc_blocks: list[str],
    guidance_text: str = "",
) -> list[tuple[str, dict[str, list[dict[str, str]]]]]:
    """Build documentation-QA style training turns from Ada code blocks.

    Two turn kinds per code block:
    - a plain code-explanation turn (explanation distilled from the
      surrounding documentation is not available, so the assistant
      restates and completes the snippet),
    - an STE explanation turn where the assistant explains the code in
      Simplified Technical English and defines each technical term at
      its first use. These turns teach the model to write docs that
      obey the STE rules it is prompted with.

    Returns (group_id, turn) pairs: both turns of one code block share a
    group so the train/val/test split never separates them.
    """
    grouped: list[tuple[str, dict[str, list[dict[str, str]]]]] = []
    for code in doc_blocks:
        group = f"doc:{zlib.crc32(code.encode('utf-8'))}"
        standard = detect_ada_standard(code)
        cleaned_code = strip_noisy_comments(code)
        user_msg = (
            f"Explain the following {_standard_label(standard)} code and describe what it demonstrates.\n\n"
            f"```ada\n{cleaned_code}\n```"
        )
        # Plain completion answer
        grouped.append((group, {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"```ada\n{cleaned_code}\n```"},
            ],
        }))
        # STE explanation answer: written by the rules we distill,
        # with each technical term defined at first use.
        ste_explanation = _ste_explanation(cleaned_code, standard)
        ste_user = (
            f"Explain the following {_standard_label(standard)} code for documentation. "
            "Follow Simplified Technical English rules and define technical "
            "terms at first use.\n\n"
            f"```ada\n{cleaned_code}\n```"
        )
        grouped.append((group, {
            "messages": [
                {
                    "role": "system",
                    "content": _compose_system_prompt(standard, STE_RULE_BLOCK, build_technical_term_glossary()),
                },
                {"role": "user", "content": ste_user},
                {"role": "assistant", "content": ste_explanation},
            ],
        }))
    if grouped and guidance_text:
        # Variant system prompts carrying the agent-skill guidance teach the
        # model to write current-toolchain Ada/SPARK (contracts, Alire, proof).
        sample_turns = grouped[: max(len(grouped) // 4, 1)]
        for _group, turn in sample_turns:
            turn["messages"].insert(0, {
                "role": "system",
                "content": (
                    "You are a specialized Ada/SPARK AI agent. You work with the "
                    "full Ada language spectrum from Ada 83 to Ada 2022, including "
                    "SPARK 2014. You write strictly "
                    "conforming, idiomatic code, prioritize contract annotations "
                    "(Pre/Post), and target the current GNAT/Alire toolchain.\n\n"
                    f"{guidance_text}"
                ),
            })
    return grouped


def _ste_explanation(code: str, standard: str) -> str:
    """Write an STE-compliant explanation of an Ada snippet.

    The explanation follows the distilled rules: short sentences, active
    voice, no em-dashes, no hedges, and one technical term per definition.
    It names the constructs the snippet actually contains, so the answer
    stays factual instead of hallucinating behavior.
    """
    facts: list[str] = []
    pkg = re.search(r"\bpackage\s+(?:body\s+)?(\w+)", code)
    if pkg:
        facts.append(f"This {_standard_label(standard)} code declares package {pkg.group(1)}.")
    procs = re.findall(r"\b(?:procedure|function)\s+(\w+)", code)
    for proc in procs[:3]:
        kind = "procedure" if re.search(rf"\bprocedure\s+{proc}\b", code) else "function"
        facts.append(f"The {kind} {proc} performs one step of the unit's work.")
    if re.search(r"\bPre\s*=>", code):
        facts.append("A precondition states what must hold on entry. The compiler and gnatprove check it.")
    if re.search(r"\bPost\s*=>", code):
        facts.append("A postcondition states what must hold on return. The compiler and gnatprove check it.")
    if re.search(r"\bLoop_Invariant\b", code):
        facts.append("A loop invariant holds on every iteration. gnatprove uses it to verify the loop.")
    if re.search(r"\btagged\b", code):
        facts.append("A tagged type supports inheritance and dynamic dispatch.")
    if re.search(r"\btask\s+", code):
        facts.append("A task runs concurrently with its caller.")
    if re.search(r"\bprotected\s+", code):
        facts.append("A protected object serializes access to shared data.")
    if re.search(r"\bgeneric\b", code):
        facts.append("A generic is a template. Instantiation creates the concrete unit.")
    if re.search(r"\bGhost\b", code):
        facts.append("Ghost code exists only for proof. The compiler erases it at runtime.")
    if not facts:
        facts.append(
            f"This {_standard_label(standard)} snippet shows a declaration and its use. "
            "Read the specification first, then the body."
        )
    # First sentence: the answer. Then facts as short sentences. No em-dashes,
    # no hedges; sanitize_prose enforces the mechanical rules.
    explanation = sanitize_prose(" ".join(facts))
    return (
        f"```ada\n{code}\n```\n\n{explanation}\n\n"
        "Terms used: specification (the public contract of a package), "
        "contract (a checked assertion about a subprogram or type)."
    )


# --------------------------------------------------------------------------- #
# AdaCore toolchain QA turns (gnatprove, alire, gnatdoc, gnattest, gnatfuzz)
# --------------------------------------------------------------------------- #

# Per-skill question templates. Each entry: (question, answer_builder).
# Answers are extracted from the loaded skill docs where possible and are
# sanitized to the STE style.

_TOOLCHAIN_QUESTIONS: dict[str, list[tuple[str, str, str]]] = {
    # skill name -> list of (user question, regex to locate the answer chunk,
    #                        fallback keyword used if regex misses)
    "gnatprove": [
        (
            "How do I run gnatprove on a SPARK project and read its output?",
            r"## Quick Start",
            "Quick Start",
        ),
        (
            "How do I use loop invariants and contracts when writing SPARK?",
            r"## Use Cases",
            "Writing SPARK",
        ),
        (
            "What is the workflow for debugging failed proof obligations?",
            r"## Use Cases",
            "Proving SPARK",
        ),
    ],
    "alire": [
        (
            "How do I create an Alire crate and manage its dependencies?",
            r"## Quick Reference Chart",
            "Quick Reference",
        ),
        (
            "How do I add tests (AUnit or GNATtest) to an Alire crate?",
            r"## Quick Reference Chart",
            "testing",
        ),
    ],
    "gnatdoc": [
        (
            "How do I generate API documentation for an Ada project with gnatdoc?",
            r"## Quick Start",
            "Quick Start",
        ),
        (
            "Where must documentation comments sit for gnatdoc to pick them up?",
            r"## Core principles",
            "Comments must be adjacent",
        ),
    ],
    "gnattest": [
        (
            "How do I generate and run unit tests with gnattest?",
            r"## Quick Start",
            "Phase 1",
        ),
        (
            "Which gnattest files belong in version control and why?",
            r"## Core Principles",
            "skeleton",
        ),
    ],
    "gnatfuzz": [
        (
            "What is the gnatfuzz workflow from analysis to a fuzzing campaign?",
            r"## Quick Start",
            "analyze",
        ),
        (
            "What does a crash found by gnatfuzz mean about my Ada code?",
            r"## Core principles",
            "oracle",
        ),
    ],
}


def _extract_answer_chunk(doc_text: str, anchor_regex: str, fallback: str) -> str:
    """Extract the section after an anchor heading, bounded in size."""
    m = re.search(anchor_regex, doc_text)
    if not m:
        idx = doc_text.find(fallback)
        if idx < 0:
            return ""
        chunk = doc_text[idx:idx + 2500]
    else:
        chunk = doc_text[m.end():m.end() + 2500]
    # Cut at the next top-level heading, if any.
    nxt = re.search(r"\n## ", chunk)
    if nxt:
        chunk = chunk[:nxt.start()]
    return chunk.strip()


def build_toolchain_qa_turns(
    toolchain_docs: dict[str, str],
    ste_rules: str,
    glossary: str,
) -> list[tuple[str, dict[str, list[dict[str, str]]]]]:
    """Build question/answer turns from the AdaCore agent skills.

    For each skill, asks its canonical questions and answers with the
    relevant section of the skill documentation, rewritten to the STE
    style. These turns ground the model in real toolchain usage: gnatprove
    invocation rules, Alire workflows, gnattest VCS rules, and so on.

    Returns (group_id, turn) pairs; one question's turns share a group.
    """
    grouped: list[tuple[str, dict[str, list[dict[str, str]]]]] = []
    for skill_name, questions in _TOOLCHAIN_QUESTIONS.items():
        doc_text = toolchain_docs.get(skill_name, "")
        if not doc_text:
            continue
        for question_idx, (question, anchor_regex, fallback) in enumerate(questions):
            chunk = _extract_answer_chunk(doc_text, anchor_regex, fallback)
            if not chunk:
                continue
            # STE-clean the prose but keep fenced code blocks intact.
            answer = _ste_clean_markdown(chunk)
            intro = (
                f"This answer uses the {skill_name} skill from AdaCore. "
                "Follow the commands exactly. Code, flags, and paths stay unchanged."
            )
            assistant_msg = sanitize_prose(intro) + "\n\n" + answer
            grouped.append((f"qa:{skill_name}:{question_idx}", {
                "messages": [
                    {
                        "role": "system",
                        "content": _compose_system_prompt("SPARK 2014", ste_rules, glossary),
                    },
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": assistant_msg},
                ],
            }))
    logger.info("Built %d toolchain QA turns from AdaCore skills", len(grouped))
    return grouped


def _ste_clean_markdown(text: str) -> str:
    """Apply STE prose rules to a markdown chunk without touching code.

    Prose between code fences is sanitized and rejoined with paragraph
    breaks. Markdown bold is stripped (STE: no bold lead-ins); headings,
    lists, and code stay untouched.
    """
    fence_pattern = re.compile(r"(```.*?```)", re.DOTALL)
    parts = fence_pattern.split(text)
    cleaned: list[str] = []
    for part in parts:
        if part.startswith("```"):
            cleaned.append(part)
        else:
            prose = sanitize_prose(part)
            prose = re.sub(r"\*\*([^*]+)\*\*", r"\1", prose)
            cleaned.append(prose)
    return "\n\n".join(p for p in cleaned if p.strip())


# --------------------------------------------------------------------------- #
# JSONL Formatting
# --------------------------------------------------------------------------- #

def format_training_turn(
    standard: str,
    spec_content: str | None,
    impl_content: str | None,
    package: str | None,
    context: str,
    source: str = "",
    guidance_text: str = "",
    ste_rules: str = "",
    glossary: str = "",
) -> dict[str, list[dict[str, str]]]:
    """Format a single training turn using the OpenAI/Qwen chat template.

    Produces a dict with a ``messages`` key containing the ``system``,
    ``user``, and ``assistant`` message objects. System prompts carry the
    STE writing rules and the technical-term glossary. When *guidance_text*
    is provided (from the agent skills), a portion of turns embed it so the
    model internalizes current-toolchain conventions.
    """
    system_msg = _compose_system_prompt(standard, ste_rules, glossary)
    if guidance_text:
        system_msg += "\n\n" + (
            "You are a specialized Ada/SPARK AI agent. You work with the full "
            "Ada language spectrum from Ada 83 to Ada 2022, including SPARK "
            "2014. You write strictly "
            "conforming, idiomatic code, prioritize contract annotations "
            "(Pre/Post), and target the current GNAT/Alire toolchain.\n\n"
            f"{guidance_text}"
        )
    _ = context  # context kept for CLI compatibility; system prompt covers safety

    if spec_content and impl_content:
        user_msg = (
            f"Ada {_standard_label(standard)} - Package specification for `{package}`.\n\n"
            f"Source: {_source_label(source)}\n\n"
            f"Please complete the following package body based on the "
            f"specification below.\n\n---\n\n"
            f"```ada\n{spec_content}\n```\n\n"
            f"Provide the corresponding package body implementation."
        )
        assistant_msg = f"```ada\n{impl_content}\n```"
    elif spec_content and not impl_content:
        user_msg = (
            f"Ada {_standard_label(standard)} - Package specification for `{package}`.\n\n"
            f"Source: {_source_label(source)}\n\n"
            f"Please provide the full package body implementation for "
            f"the following specification.\n\n---\n\n"
            f"```ada\n{spec_content}\n```\n\n"
            f"Provide the corresponding package body."
        )
        assistant_msg = ""
    elif impl_content and not spec_content:
        user_msg = (
            f"Ada {_standard_label(standard)} - Implementation unit for `{package}`.\n\n"
            f"Source: {_source_label(source)}\n\n"
            f"Please provide the corresponding package specification "
            f"(`.ads` file) for this implementation.\n\n---\n\n"
            f"```ada\n{impl_content}\n```\n\n"
            f"Provide the corresponding package specification."
        )
        assistant_msg = ""
    else:
        return {"messages": []}

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
    }


# --------------------------------------------------------------------------- #
# Multiprocessing helpers (stdlib multiprocessing, fork start method)
# --------------------------------------------------------------------------- #

# Context shared with worker processes. Set once in the parent before the
# Pool forks; workers inherit it, so per-task payloads stay small.
_BUILD_CONTEXT: dict[str, Any] = {}


def _worker_init() -> None:
    """Quiet the dataset logger inside worker processes.

    Parent-side progress lines stay; per-file skip warnings are folded into
    the aggregated counts instead of interleaving from every worker.
    """
    logging.getLogger("q3as_build_dataset").setLevel(logging.ERROR)


def _make_pool(workers: int, task_count: int):
    """Create a fork-based Pool, or None for the serial path.

    Serial when only one worker is requested or the workload is tiny: a
    Pool costs more than it saves below a handful of tasks.
    """
    if workers <= 1 or task_count < 8:
        return None
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        logger.warning("Fork start method unavailable - building serially")
        return None
    return ctx.Pool(processes=min(workers, task_count), initializer=_worker_init)


def _process_pair_task(
    task: dict[str, Any],
) -> tuple[str, list[dict[str, list[dict[str, str]]]], dict[str, int]]:
    """Read, sanitize, and build turns for one spec/body pair.

    Pure function of *task* plus the inherited _BUILD_CONTEXT, so it runs
    in any worker process. Returns (group_id, turns, per-kind counts).
    """
    ste_rules = _BUILD_CONTEXT["ste_rules"]
    glossary = _BUILD_CONTEXT["glossary"]
    guidance_text = _BUILD_CONTEXT["guidance_text"]
    context = _BUILD_CONTEXT["context"]

    spec_path = task["spec"]
    impl_path = task["impl"]
    package = task["package"]
    source = task["source"]

    spec_content: str | None = read_and_sanitize(Path(spec_path)) if spec_path else None
    impl_content: str | None = read_and_sanitize(Path(impl_path)) if impl_path else None
    if spec_content is None and impl_content is None:
        return task["group"], [], {}

    combined_content = ""
    if spec_content:
        combined_content += spec_content + "\n"
    if impl_content:
        combined_content += impl_content + "\n"

    standard = detect_ada_standard(combined_content)

    # Clean noisy inline comments so the model learns STE comments.
    if spec_content:
        spec_content = strip_noisy_comments(spec_content)
    if impl_content:
        impl_content = strip_noisy_comments(impl_content)

    # Include .gpr context if available (resolved once per input directory).
    if task["gpr_content"]:
        combined_content += (
            f"\n---\nProject file reference ({task['gpr_name']}):\n{task['gpr_content']}\n"
        )

    counts: dict[str, int] = {}
    turns: list[dict[str, list[dict[str, str]]]] = []
    turn = format_training_turn(
        standard, spec_content, impl_content, package, context,
        source=source,
        guidance_text=_sample_guidance(guidance_text, key=combined_content),
        ste_rules=ste_rules,
        glossary=glossary,
    )
    if turn["messages"]:
        turns.append(turn)
        counts["code_pair"] = 1

    # Correct-vs-wrong defect pairs from the combined unit.
    if task["enable_defect_pairs"] and combined_content.strip():
        defect_turns = build_defect_turns(
            combined_content.strip(), standard, package, source,
            ste_rules, glossary,
        )
        turns.extend(defect_turns)
        counts["defect_pair"] = len(defect_turns)

    return task["group"], turns, counts


# Cap on how many AST-derived records with the same structural (alpha-renamed)
# code signature are kept. The Ada-Algorithms corpus is highly templated:
# hundreds of
# records are the same algorithm with different identifier spellings, and the
# variant-renaming pass makes them exactly equal. Cap, not drop: a few copies
# of an idiom are useful signal, hundreds are duplication. Records with no
# structural signature (plain prose turns, short code) are never capped.
#
# Value chosen by experiment (scripts/run_cap_experiment.sh, 40-step QLoRA
# probes on the cap-variant datasets): cap 2 reached the best val loss
# (1.050 vs 1.117 for cap 3) and tied cap 5 on test (1.065 vs 1.070 ppl 2.90
# vs 2.92) with a smaller, more diverse dataset. See
# docs/datasets-and-training.md.
AST_STRUCTURAL_CAP = 10
AST_STRUCTURAL_CAP_UNLIMITED = -1


def dedup_grouped(
    all_grouped: list[tuple[str, dict[str, list[dict[str, str]]]]],
    ast_structural_cap: int = AST_STRUCTURAL_CAP,
) -> tuple[list[tuple[str, dict[str, list[dict[str, str]]]]], int, dict[str, int]]:
    """Drop duplicate turns, keeping the first occurrence.

    Two duplicate families are removed:

    1. **Exact duplicates** - identical message lists generated in two
       different groups (a package spec extracted through different source
        paths, a variant turn that renames structurally identical code
        to the same fresh names). Group-aware splitting cannot catch these,
        and copies that straddle splits would leak eval answers into
        training.
    2. **Structural duplicates of AST-derived records** - records whose
        assistant Ada code has the same alpha-renamed token shape
        (``eval_guard.structural_text``). The Ada-Algorithms corpus contains
        large families of the same algorithm under different spellings; more than
       ``AST_STRUCTURAL_CAP`` copies of one shape add duplication, not
       signal, so extras are dropped.

    What is deliberately preserved: the dataset's intentional
    natural-language and code variety. Correct-variant turns
    (``code_variants.variant_turns``) rename or reorder a *specific*
    snippet and keep the surrounding prose, so each variant has a distinct
    structural signature and distinct wording - it survives dedup. The
    paraphrase turns (plain vs STE doc answers, prose vs diagnosis) differ
    in their assistant text and never collide. Only verbatim repeats and
    beyond-cap structural clones of AST code are removed.

    Returns (deduplicated list, dropped count, breakdown by family with
    the keys ``exact`` and ``ast_structural_capped``).
    """
    seen: set[str] = set()
    ast_seen: dict[str, int] = {}
    deduped: list[tuple[str, dict[str, list[dict[str, str]]]]] = []
    breakdown = {"exact": 0, "ast_structural_capped": 0}
    for group, turn in all_grouped:
        signature = json.dumps(turn["messages"], sort_keys=True)
        if signature in seen:
            breakdown["exact"] += 1
            continue
        structural = _ast_structural_signature(turn["messages"])
        if structural is not None and ast_structural_cap != AST_STRUCTURAL_CAP_UNLIMITED:
            count = ast_seen.get(structural, 0)
            if count >= ast_structural_cap:
                breakdown["ast_structural_capped"] += 1
                continue
            ast_seen[structural] = count + 1
        seen.add(signature)
        deduped.append((group, turn))
    dropped = breakdown["exact"] + breakdown["ast_structural_capped"]
    return deduped, dropped, breakdown


def _ast_structural_signature(messages: list[dict[str, str]]) -> str | None:
    """Alpha-renamed signature of a record's assistant Ada code, or None.

    Only assistant fences count: the answer is what must not repeat.
    The user prompt (question phrasing, spec shown) may legitimately repeat
    across records. Code shorter than ``eval_guard.MIN_STRUCTURAL_LEN``
    tokens carries too little shape to cluster on and is never capped.
    """
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for block in _FENCE_BLOCK_RE.findall(str(message.get("content", ""))):
            structural = eval_guard.structural_text(block)
            if len(structural) >= eval_guard.MIN_STRUCTURAL_LEN:
                return structural
    return None


def split_file_paths(output_file: Path) -> dict[str, Path]:
    """The three split files that accompany *output_file*.

    Single definition so the writer and the staleness check in main() cannot
    drift apart and leave a stamp describing files that are never written.
    """
    return {
        name: output_file.parent / f"dataset_{name}.jsonl"
        for name in ("train", "val", "test")
    }


def assign_splits(
    groups: list[str],
    seed: int = _RNG_SEED,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> dict[str, str]:
    """Assign each turn group to train/val/test deterministically.

    Whole groups go to one split, so turns derived from the same source
    (a code pair and its defect pairs, both doc turns of one block) never
    leak across splits. Groups are sorted, then shuffled with *seed*, so
    the assignment is independent of build order and worker count.
    """
    unique = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    if n == 1:
        return {unique[0]: "train"}
    if n == 2:
        return {unique[0]: "train", unique[1]: "val"}
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - n_test - 1)
    assignment: dict[str, str] = {g: "train" for g in unique}
    for g in unique[:n_test]:
        assignment[g] = "test"
    for g in unique[n_test:n_test + n_val]:
        assignment[g] = "val"
    return assignment


# --------------------------------------------------------------------------- #
# Extra-turn ingestion (parser outputs: docs chunks, Ada AST units)
# --------------------------------------------------------------------------- #

# Turn-kind buckets for records produced by the parser modules.
def _extra_turn_kind(meta: dict[str, Any]) -> str:
    kind = str(meta.get("kind", "extra"))
    if kind == "doc_section":
        return "doc_section"
    if kind.startswith("ast_"):
        return "ast_qa"
    if kind.startswith("contract_synth_"):
        return "contract_synth"
    return "extra"


# Fence blocks in arbitrary messages (assistant replies, user prompts).
_FENCE_BLOCK_RE = re.compile(r"```(?:ada)?\n(.*?)```", re.DOTALL)


def _empty_assistant(turn: dict[str, list[dict[str, str]]]) -> bool:
    """True when the turn's last assistant message is empty.

    The pair pipeline emits spec-only/impl-only records whose assistant
    reply is the empty string ("provide the body" with no body). Training
    on them teaches immediate-EOS behavior, so they are dropped.
    """
    messages = turn.get("messages") or []
    return bool(messages) and messages[-1].get("role") == "assistant" and not str(messages[-1].get("content", "")).strip()


def _detect_standard_for_block(code: str) -> str:
    """Standard for ingested code: content-derived, defaults to SPARK 2014."""
    std = detect_ada_standard(code)
    return std if std not in ("Unknown", "") else "SPARK 2014"


def _extra_turn_group(record: dict[str, Any], index: int) -> str:
    """Split group for one extra record.

    The Ada AST parser tags its records (impl + contract turns of one
    subprogram share a group). Doc-section records group by source and
    section so the same section never straddles two splits.
    """
    meta = record.get("meta") or {}
    group = meta.get("group")
    if group:
        return str(group)
    source = str(meta.get("source", ""))
    section = str(meta.get("section", ""))
    if source or section:
        key = (source + "\x00" + section).encode("utf-8")
        return f"extra:{zlib.crc32(key)}"
    return f"extra:record:{index}"


def _ingest_extra_turns(
    extra_paths: list[Path],
    all_grouped: list[tuple[str, dict[str, list[dict[str, str]]]]],
    turn_counts: dict[str, int],
) -> list[dict[str, int]]:
    """Load parser-produced JSONL turn files into the grouped turn pool.

    Records whose assistant reply carries an Ada code block (impl, contract
    write, type turns) also get defect turns: the injected families run on
    the extracted code itself, so AST-derived training data gets the same
    broken-example coverage as the pair pipeline.

    Empty-assistant records ("provide the body" with no body) are counted
    per file under ``empty_assistant`` and dropped instead of ingested.

    Returns one stats dict per input file, in the same order, with the keys
    ``records``, ``ingested`` (2+ message records), ``defects`` (derived
    defect turns), ``variants`` (derived correct-variant turns) and
    ``empty_assistant`` (records dropped for an empty assistant reply).
    Missing and unreadable files still produce their stats entry so the
    metadata exposes every requested source, not only the ones that worked.
    """
    per_file_stats: list[dict[str, int]] = []
    for extra_path in extra_paths:
        stats = {"records": 0, "ingested": 0, "defects": 0, "variants": 0, "empty_assistant": 0}
        per_file_stats.append(stats)
        if not extra_path.exists():
            logger.warning("Extra turns file not found, skipping: %s", extra_path)
            continue
        ingested = 0
        defect_turns = 0
        variant_turns_count = 0
        skipped_empty_assistant = 0
        try:
            with open(extra_path, "r", encoding="utf-8") as f:
                for index, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    stats["records"] += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON in %s line %d", extra_path, index + 1)
                        continue
                    messages = record.get("messages") or []
                    if len(messages) < 2:
                        continue
                    meta = record.get("meta") or {}
                    group = _extra_turn_group(record, index)
                    if _empty_assistant({"messages": messages}):
                        skipped_empty_assistant += 1
                        continue
                    all_grouped.append((group, {"messages": messages}))
                    kind = _extra_turn_kind(meta)
                    turn_counts[kind] = turn_counts.get(kind, 0) + 1
                    ingested += 1

                    # Defect pairs from the record's own code (same group, so
                    # they never straddle splits).
                    code_blocks = _FENCE_BLOCK_RE.findall(
                        "\n".join(m.get("content", "") for m in messages)
                    )
                    if code_blocks:
                        std = meta.get("standard") or _detect_standard_for_block(code_blocks[0])
                        new_defects = build_defect_turns(
                            code_blocks[0], std,
                            meta.get("unit") or None,
                            meta.get("source", ""),
                            STE_RULE_BLOCK,
                            build_technical_term_glossary(),
                        )
                        for dturn in new_defects:
                            all_grouped.append((group, dturn))
                            defect_turns += 1

                        # Correct-variant turns (renamed, reordered) from the
                        # original record only - never from its defect turns,
                        # whose broken code must stay tied to the original.
                        if not meta.get("kind", "").endswith(("_fix", "_defect")):
                            vturns = code_variants.variant_turns(record)
                            for _vgroup, vturn in vturns:
                                all_grouped.append((group, vturn))
                                variant_turns_count += 1
        except OSError as exc:
            logger.warning("Cannot read extra turns file %s: %s", extra_path, exc)
        finally:
            # Attribute every derived turn of this file, even when the read
            # failed midway: partial ingestion still lands in the dataset.
            stats["ingested"] += ingested
            stats["defects"] += defect_turns
            stats["variants"] += variant_turns_count
            stats["empty_assistant"] += skipped_empty_assistant
            if defect_turns:
                turn_counts["ast_defect"] = turn_counts.get("ast_defect", 0) + defect_turns
            if variant_turns_count:
                turn_counts["variant"] = turn_counts.get("variant", 0) + variant_turns_count
            if ingested:
                logger.info(
                    "Ingested %d extra turns from %s (+%d defect, +%d variant turns)",
                    ingested, extra_path, defect_turns, variant_turns_count,
                )
            else:
                logger.warning("No usable turns in extra turns file: %s", extra_path)
    return per_file_stats


# --------------------------------------------------------------------------- #
# Main Pipeline
# --------------------------------------------------------------------------- #

def build_dataset(
    input_dirs: list[Path],
    extra_input_dirs: list[Path],
    output_file: Path,
    context: str = "high-integrity, safety-critical systems",
    eval_methodology: dict[str, Any] | None = None,
    doc_dirs: list[Path] | None = None,
    guidance_dirs: list[Path] | None = None,
    enable_defect_pairs: bool = True,
    workers: int = 0,
    extra_turns: list[Path] | None = None,
    ast_structural_cap: int = AST_STRUCTURAL_CAP,
) -> int:
    """Run the full ingestion -> pairing -> sanitization -> JSONL pipeline.

    Processes the supplied input directories, extracts Ada code blocks from
    documentation sources (learn), embeds agent-skill guidance
    (ada-spark, SimpleEnglish, AdaCore skills) into system prompts, and
    generates correct-vs-wrong defect pairs plus toolchain QA turns.
    Derives dataset structure from ada-eval methodology if available.

    Pair processing and documentation extraction run on a stdlib
    multiprocessing Pool (*workers* processes, fork start method; 0 = one
    per CPU core, 1 = serial). All randomness is content-derived or seeded,
    so the output is byte-identical for any worker count.

    Every record gets a ``split`` field (train/val/test, ~90/5/5 by turn
    groups) and the same records are written to dataset_train.jsonl,
    dataset_val.jsonl, and dataset_test.jsonl beside the main file.

    *extra_turns* are pre-built chat JSONL files from the parser modules
    (parse_docs.py heading chunks, parse_ada_ast.py semantic units); they
    join the same seeded, group-aware split as the rest of the corpus.

    Returns the number of valid training turns written to the output file.
    """
    all_grouped: list[tuple[str, dict[str, list[dict[str, str]]]]] = []
    turn_counts: dict[str, int] = {}
    eval_info = eval_methodology or {}
    empty_assistant_dropped = 0

    # Load writing rules, terminology glossary, and agent-skill guidance.
    ste_rules = load_simple_english_rules()
    glossary = build_technical_term_glossary()
    guidance_text, skills_loaded = load_agent_skill_guidance(guidance_dirs or [])
    toolchain_docs = load_adacore_toolchain_docs()

    logger.info("Agent skills loaded: %s", {k: len(v) for k, v in skills_loaded.items()})
    logger.info("Toolchain skills with QA templates: %s", sorted(toolchain_docs.keys()))

    # Log evaluation methodology info
    if eval_info:
        logger.info(
            "Using evaluation methodology from %s: %s categories, %d compacted datasets",
            eval_info.get("source", "unknown"),
            len(eval_info.get("expanded_categories", [])),
            len(eval_info.get("compacted_datasets", [])),
        )

    # One task per spec/body pair. File reads, sanitization, standard
    # detection, and turn building are pure functions of the task, so they
    # distribute across worker processes without shared state.
    _BUILD_CONTEXT.update({
        "ste_rules": ste_rules,
        "glossary": glossary,
        "guidance_text": guidance_text,
        "context": context,
    })
    tasks: list[dict[str, Any]] = []
    records_by_input_dir: dict[str, int] = {}
    with progress_log.phase(
        "discover sources", logger, roots=len(input_dirs) + len(extra_input_dirs),
    ):
        for input_dir in input_dirs + extra_input_dirs:
            resolved_input = input_dir.resolve()
            logger.info("Processing input directory: %s -> %s", input_dir, resolved_input)

            if not resolved_input.exists():
                logger.warning("Directory does not exist, skipping: %s", resolved_input)
                continue

            discovered = discover_files(resolved_input)
            if not discovered["ads"] and not discovered["adb"]:
                logger.warning("No Ada source files found in %s", resolved_input)

            pairs = pair_files(discovered)

            # Resolve one .gpr reference per input directory (as the serial
            # loop did) so workers do not all re-read the same project file.
            gpr_content = ""
            gpr_name = ""
            if discovered.get("gpr"):
                for gpr_path in discovered["gpr"][:3]:
                    candidate = read_and_sanitize(gpr_path)
                    if candidate:
                        gpr_content = candidate
                        gpr_name = gpr_path.name
                        break

            source = str(resolved_input)
            dir_pair_count = 0
            for idx, pair in enumerate(pairs):
                spec_path = pair["spec"]
                impl_path = pair["impl"]
                package = pair["package"]
                tasks.append({
                    "group": f"pair:{source}:{idx}",
                    "spec": str(spec_path) if isinstance(spec_path, Path) else None,
                    "impl": str(impl_path) if isinstance(impl_path, Path) else None,
                    "package": str(package) if package else None,
                    "source": source,
                    "gpr_content": gpr_content,
                    "gpr_name": gpr_name,
                    "enable_defect_pairs": enable_defect_pairs,
                })
                dir_pair_count += 1
            records_by_input_dir[source] = records_by_input_dir.get(source, 0) + dir_pair_count

    if workers <= 0:
        workers = os.cpu_count() or 1
    pool = _make_pool(workers, len(tasks))
    imap = pool.imap if pool else None
    with progress_log.phase("build pairs", logger, pairs=len(tasks), workers=workers):
        bar = progress_log.Progress("Pairs", len(tasks), log=logger)
        results = []
        if pool:
            logger.info("Building %d pairs with %d worker processes", len(tasks), workers)
            for result in pool.imap(
                _process_pair_task, tasks, chunksize=max(1, len(tasks) // (workers * 4)),
            ):
                results.append(result)
                bar.advance()
        else:
            for task in tasks:
                results.append(_process_pair_task(task))
                bar.advance()
        bar.close()

    with progress_log.phase("merge pair turns", logger, pairs=len(results)):
        for group, turns, counts in results:
            for turn in turns:
                if _empty_assistant(turn):
                    # Spec-only/impl-only pairs ask for a completion but carry
                    # no answer; training on them teaches empty replies.
                    empty_assistant_dropped += 1
                    if counts.get("code_pair"):
                        counts["code_pair"] -= 1
                    continue
                all_grouped.append((group, turn))
            for key, value in counts.items():
                turn_counts[key] = turn_counts.get(key, 0) + value
    if empty_assistant_dropped:
        logger.info(
            "Dropped %d pair turns with empty assistant replies",
            empty_assistant_dropped,
        )

    # Documentation-QA turns from code blocks embedded in course material
    doc_files: list[Path] = []
    with progress_log.phase("discover docs", logger, roots=len(doc_dirs or [])):
        for doc_root in doc_dirs or []:
            if not doc_root.exists():
                logger.warning("Documentation source not found, skipping: %s", doc_root)
                continue
            for pattern in ("*.rst", "*.md"):
                doc_files.extend(sorted(doc_root.rglob(pattern)))
    if doc_files:
        with progress_log.phase("doc code blocks", logger, files=len(doc_files)):
            doc_blocks = _collect_doc_blocks(doc_files, imap)
        logger.info("Extracted %d unique Ada code blocks from documentation", len(doc_blocks))
        with progress_log.phase("doc QA turns", logger, blocks=len(doc_blocks)):
            doc_grouped = build_doc_training_turns(doc_blocks, guidance_text)
        all_grouped.extend(doc_grouped)
        turn_counts["doc_qa"] = turn_counts.get("doc_qa", 0) + len(doc_grouped)

    if pool:
        pool.close()
        pool.join()

    # Toolchain QA turns from the AdaCore skills
    if toolchain_docs:
        with progress_log.phase("toolchain QA turns", logger, docs=len(toolchain_docs)):
            qa_grouped = build_toolchain_qa_turns(toolchain_docs, ste_rules, glossary)
        all_grouped.extend(qa_grouped)
        turn_counts["toolchain_qa"] = turn_counts.get("toolchain_qa", 0) + len(qa_grouped)

    # Pre-built turns from the parser modules (doc chunks, Ada AST units,
    # synthetic contract turns). The per-file stats double as provenance:
    # a parser output that is missing or empty shows up in the metadata
    # with ingested=0 instead of vanishing silently.
    with progress_log.phase("ingest parser output", logger, files=len(extra_turns or [])):
        extra_turns_stats = _ingest_extra_turns(extra_turns or [], all_grouped, turn_counts)

    # Eval-integrity guard: drop any group whose content matches the ada-eval
    # evaluation suite (verbatim, reformatted, or identifier-renamed). One
    # bad turn poisons its whole group. Runs before dedup and split so no
    # eval-derived record can reach any split file.
    records_before_guard = len(all_grouped)
    with progress_log.phase("eval guard", logger, records=records_before_guard):
        all_grouped, guard_dropped_groups, guard_reasons = eval_guard.contaminated_groups(all_grouped)
    guard_sigs = eval_guard.load_eval_signatures()
    guard_dropped_records = records_before_guard - len(all_grouped)
    if guard_dropped_groups:
        # Report records and groups separately: contaminated_groups returns a
        # group count, and calling it a record count overstated the drop.
        logger.warning(
            "Eval guard dropped %d records in %d contaminated groups: %s",
            guard_dropped_records,
            guard_dropped_groups,
            ", ".join(sorted(set(guard_reasons))[:10]),
        )

    # Remove exact duplicates and beyond-cap structural duplicates before
    # splitting (see dedup_grouped docstring for what is kept on purpose).
    # eval_guard is imported at module top; the build_dataset ->
    # eval_guard -> parse_ada_ast -> build_dataset import cycle is benign
    # because all cross-uses are call-time attribute lookups.
    with progress_log.phase("dedup", logger, records=len(all_grouped)):
        all_grouped, deduped_count, dedup_detail = dedup_grouped(
            all_grouped, ast_structural_cap=ast_structural_cap,
        )
    if deduped_count:
        logger.info(
            "Removed %d duplicate turns before splitting (%d verbatim, %d AST structural over cap)",
            deduped_count,
            dedup_detail["exact"],
            dedup_detail["ast_structural_capped"],
        )

    all_turns = [turn for _group, turn in all_grouped]

    # Self-check: count STE violations in our own generated assistant prose.
    # Messages with code fences are exempt: the fence itself triggers the
    # semicolon/identifier checks, and code is exempt from STE by rule 10.
    violation_count = 0
    with progress_log.phase("STE self-check", logger, turns=len(all_turns)):
        for turn in all_turns:
            for message in turn["messages"]:
                if message["role"] != "assistant" or "```" in message["content"]:
                    continue
                if has_style_violations(message["content"]):
                    violation_count += 1
    if violation_count:
        logger.warning(
            "%d assistant messages still contain STE style violations "
            "(sanitize_prose should have removed most)",
            violation_count,
        )

    # Deterministic, group-aware train/val/test split (~90/5/5).
    split_seed = _RNG_SEED
    split_of = assign_splits([group for group, _ in all_grouped], seed=split_seed)
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for group, _turn in all_grouped:
        split_counts[split_of[group]] += 1

    # Write JSONL output: the main file carries every record with its split
    # tag; the three split files hold the same records pre-filtered.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    split_paths = split_file_paths(output_file)
    with progress_log.phase("write dataset", logger, turns=len(all_grouped), path=str(output_file)):
        write_bar = progress_log.Progress("Records written", len(all_grouped), log=logger)
        with (
            open(output_file, "w", encoding="utf-8") as f_all,
            open(split_paths["train"], "w", encoding="utf-8") as f_train,
            open(split_paths["val"], "w", encoding="utf-8") as f_val,
            open(split_paths["test"], "w", encoding="utf-8") as f_test,
        ):
            handles = {"train": f_train, "val": f_val, "test": f_test}
            for group, turn in all_grouped:
                record: dict[str, Any] = dict(turn)
                record["split"] = split_of[group]
                line = json.dumps(record, ensure_ascii=False) + "\n"
                f_all.write(line)
                handles[split_of[group]].write(line)
                write_bar.advance()
        write_bar.close()

    # Write evaluation methodology alongside the dataset
    meta_path = output_file.parent / "dataset_metadata.json"
    metadata: dict[str, Any] = {
        "total_turns": len(all_turns),
        "turn_counts_by_kind": turn_counts,
        "splits": {
            "seed": split_seed,
            "strategy": "group-aware: every turn from one source pair or doc block stays in one split",
            "deduped_duplicates": deduped_count,
            "dedup_detail": {**dedup_detail, "ast_structural_cap": ast_structural_cap},
            "eval_guard": {
                "degraded": guard_sigs.degraded,
                "blocked_signatures": len(guard_sigs),
                "dropped_groups": guard_dropped_groups,
                "dropped_records": guard_dropped_records,
            },
            "ratios": {"train": 0.90, "val": 0.05, "test": 0.05},
            "counts": split_counts,
            "files": {name: str(path) for name, path in split_paths.items()},
        },
        "workers": workers,
        "empty_assistant_dropped": empty_assistant_dropped,
        "extra_turns_files": [
            {"path": str(path), **stats}
            for path, stats in zip(extra_turns or [], extra_turns_stats)
        ],
        "records_by_input_dir": records_by_input_dir,
        "writing_style": {
            "standard": "ASD-STE100 (distilled from the SimpleEnglish agent skill)",
            "em_dashes": "banned",
            "technical_terms": "defined at first use from the built-in glossary",
            "agent_skills": {k: len(v) for k, v in skills_loaded.items()},
        },
        "evaluation_methodology": eval_info,
        "input_dirs": [str(d.resolve()) for d in input_dirs + extra_input_dirs],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Dataset metadata written to %s", meta_path)

    logger.info(
        "Dataset written to %s - %d training turns (%s); splits: train=%d val=%d test=%d.",
        output_file, len(all_turns),
        ", ".join(f"{k}={v}" for k, v in sorted(turn_counts.items())) or "no turns",
        split_counts["train"], split_counts["val"], split_counts["test"],
    )
    return len(all_turns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build q3as training dataset from Ada source trees.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Root directory containing Ada source files. "
             "Accepts relative paths, symlinks, and git submodule targets. "
             "Resolved via pathlib.Path.resolve().",
    )
    parser.add_argument(
        "--extra-input-dir", type=Path, action="append", default=[],
        help="Additional input directories for training data "
             "(e.g., cache dirs of adacovex, Ada_CRDT, Ada-83-TLALOC). "
             "Can be specified multiple times.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory where dataset.jsonl will be written.",
    )
    parser.add_argument(
        "--context", type=str, default="high-integrity, safety-critical systems",
        help="Safety context string embedded in the system prompt.",
    )
    parser.add_argument(
        "--doc-dir", type=Path, action="append", default=[],
        help="Documentation source to extract Ada code blocks from "
             "(e.g., cache dirs of learn, training_material). "
             "Can be specified multiple times.",
    )
    parser.add_argument(
        "--guidance-dir", type=Path, action="append", default=[],
        help="Agent-skill source whose markdown guidance is embedded into "
             "system prompts (e.g., cache dirs of ada-spark, SimpleEnglish, "
             "skills). "
             "Repeatable.",
    )
    parser.add_argument(
        "--no-defect-pairs", action="store_true",
        help="Disable correct-vs-wrong defect pair generation.",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes for pair/doc processing (0 = one per CPU "
             "core, 1 = serial). Output is identical for any value.",
    )
    parser.add_argument(
        "--extra-turns", type=Path, action="append", default=None,
        help="Pre-built chat JSONL from the parser modules to merge into "
             "the dataset (repeatable). Defaults to the three standard "
             "parser outputs when they exist.",
    )
    parser.add_argument(
        "--ast-structural-cap", type=int, default=AST_STRUCTURAL_CAP,
        help="Max AST records sharing one alpha-renamed code signature "
             f"({AST_STRUCTURAL_CAP_UNLIMITED} = unlimited). Cap 0 keeps only "
             "the first record of each structural family.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even when every input is unchanged since the last build.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default extra input directories if none provided. Defaults come
    # from the source cache (source_paths); legacy siblings are the last
    # resort handled inside resolve().
    import source_paths

    if args.extra_input_dir:
        extra_dirs = args.extra_input_dir
    else:
        extra_dirs = source_paths.default_code_dirs()
    doc_dirs = (
        args.doc_dir
        if args.doc_dir
        else source_paths.default_doc_dirs()
        or [Path("..") / d for d in DOC_SOURCE_DIRS]
    )
    guidance_dirs = (
        args.guidance_dir if args.guidance_dir
        else source_paths.default_guidance_dirs()
        or [Path("..") / d for d in AGENT_SKILL_SOURCE_DIRS]
    )

    # Load evaluation methodology from ada-eval
    eval_methodology = load_eval_methodology()

    # Ensure all input directories exist (warn if not)
    all_dirs = [args.input_dir] + extra_dirs
    for d in all_dirs:
        if not d.exists():
            logger.warning("Input directory not found: %s", d)

    output_file = args.output_dir / "dataset.jsonl"
    extra_turns = args.extra_turns
    if extra_turns is None:
        # Auto-integrate the standard parser outputs when they exist, so
        # a bare `build_dataset.py` run picks up new data without extra
        # flags. The Makefile passes the same list explicitly (and makes
        # the files first), so both entry points stay in sync.
        extra_turns = [
            path for path in (
                DEFAULT_OUTPUT_DIR / "docs_chunks.jsonl",
                DEFAULT_OUTPUT_DIR / "ada_ast_units.jsonl",
                DEFAULT_OUTPUT_DIR / "contract_mutations.jsonl",
            )
            if path.exists()
        ]

    # Skip the rebuild when nothing this stage reads has changed. Without this
    # the target is re-run on every `make all`, which is minutes of work in
    # front of training for output that is already on disk. --workers is not
    # part of the fingerprint: the builder documents identical output for any
    # worker count, so changing DATASET_WORKERS must not invalidate the stamp.
    code_suffixes = tuple(sorted(ADA_SPEC_EXTENSIONS | PROJECT_EXTENSIONS))
    doc_suffixes = (".rst", ".md")
    guidance_suffixes = (".md", ".markdown")
    # ada-eval supplies the evaluation methodology block. Its expanded
    # categories are directory names, so they go in as a param: a file-suffix
    # digest cannot see a new category directory.
    expanded_dir = ADA_EVAL_DIR / "data" / "base" / "expanded"
    expanded_categories = (
        sorted(d.name for d in expanded_dir.iterdir() if d.is_dir())
        if expanded_dir.is_dir() else []
    )
    spec = stage_state.make_spec(
        name="dataset",
        outputs=[
            output_file,
            *split_file_paths(output_file).values(),
            output_file.parent / "dataset_metadata.json",
        ],
        input_trees=[
            *((root, code_suffixes) for root in [args.input_dir, *extra_dirs]),
            *((root, doc_suffixes) for root in doc_dirs),
            *((root, guidance_suffixes) for root in guidance_dirs),
            (ADA_EVAL_DIR, (".jsonl", ".toml", ".py")),
        ],
        input_files=extra_turns or (),
        scripts=[Path(__file__).resolve()],
        params=[
            ("context", args.context),
            ("defect_pairs", not args.no_defect_pairs),
            ("ast_structural_cap", args.ast_structural_cap),
            ("split_seed", _RNG_SEED),
            ("eval_expanded_categories", ",".join(expanded_categories)),
        ],
    )
    if spec.skip_if_fresh(force=args.force):
        print(f"Dataset up to date at {output_file} (use --force to rebuild).")
        return

    count = build_dataset(
        input_dirs=[args.input_dir],
        extra_input_dirs=extra_dirs,
        output_file=output_file,
        context=args.context,
        eval_methodology=eval_methodology,
        doc_dirs=doc_dirs,
        guidance_dirs=guidance_dirs,
        enable_defect_pairs=not args.no_defect_pairs,
        workers=args.workers,
        extra_turns=extra_turns,
        ast_structural_cap=args.ast_structural_cap,
    )
    print(f"Dataset built: {count} training turns -> {output_file}")
    spec.mark_fresh({"total_turns": count})


if __name__ == "__main__":
    main()
