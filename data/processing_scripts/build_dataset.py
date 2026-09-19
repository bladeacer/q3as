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
    python build_dataset.py --input-dir data/raw/ --extra-input-dir ../adacovex --extra-input-dir ../Ada_CRDT
    python build_dataset.py --input-dir /path/to/submodule --output-dir data/processed/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import textwrap
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ADA_SPEC_EXTENSIONS = {".ads", ".adb"}
PROJECT_EXTENSIONS = {".gpr"}
DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "dataset.jsonl"
# Default extra input directories for additional training data sources
DEFAULT_EXTRA_INPUT_DIRS = [
    Path("../adacovex"),
    Path("../Ada_CRDT"),
    Path("../Ada-83-TLALOC"),
    Path("../ada-eval"),
]
# Directories treated as documentation sources: Ada code blocks are extracted
# from their RST/Markdown content instead of pairing .ads/.adb files.
DOC_SOURCE_DIRS = ["learn"]
# Agent-skill sources whose markdown guidance is embedded into system prompts.
# Each entry maps a sibling directory name to the repo-relative paths of its
# skill documents so the loader works across the three repo layouts we use.
AGENT_SKILL_SOURCE_DIRS = ["ada-spark", "SimpleEnglish", "skills"]
ADA_SPARK_DIR = Path("../ada-spark")
SIMPLE_ENGLISH_DIR = Path("../SimpleEnglish")
ADACORE_SKILLS_DIR = Path("../skills")
# Default path to ada-eval methodology directory
ADA_EVAL_DIR = Path("../ada-eval")

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

# ---- Ada standard detection (robust, scoring-based, newest-first) ----
# Newer Ada standards are supersets of older ones. We score each file
# against features from Ada 2022 down to Ada 83. The first standard
# whose matched feature count meets the minimum threshold wins.
# This avoids brittle per-keyword regex matching and handles the
# superset relationship between Ada standards correctly.

_STANDARD_FEATURES: list[tuple[str, list[tuple[str, re.Pattern[str]]], int]] = [
    ("Ada 2022", [
        ("Static_Pure", re.compile(r"\bStatic_Pure\b", re.IGNORECASE)),
        ("Pure_Global", re.compile(r"\bPure_Global\b", re.IGNORECASE)),
        ("Contract_Cases", re.compile(r"\bContract_Cases\b", re.IGNORECASE)),
        ("Loop_Invariant", re.compile(r"\bLoop_Invariant\b", re.IGNORECASE)),
        ("Dynamic_Pure", re.compile(r"\bDynamic_Pure\b", re.IGNORECASE)),
        ("Wide_Wide_String", re.compile(r"\bWide_Wide_String\b", re.IGNORECASE)),
        ("String_Literals", re.compile(r"\bString_Literal\b", re.IGNORECASE)),
        ("Aspect_Specs", re.compile(r"\bwith\s+\w+\s+=>", re.IGNORECASE)),
    ], 3),
    ("Ada 2012", [
        ("Pre_aspect", re.compile(r"\bPre\s*=>", re.IGNORECASE)),
        ("Post_aspect", re.compile(r"\bPost\s*=>", re.IGNORECASE)),
        ("Type_Invariant", re.compile(r"\bType_Invariant\b", re.IGNORECASE)),
        ("Global_aspect", re.compile(r"\bGlobal\s*=>", re.IGNORECASE)),
        ("Depends_aspect", re.compile(r"\bDepends\s*=>", re.IGNORECASE)),
        ("Subtype_Contract", re.compile(r"\bSubtype_Contract\b", re.IGNORECASE)),
        ("Iterate_aspect", re.compile(r"\bIterate\s*=>", re.IGNORECASE)),
        ("Concurrent", re.compile(r"\bSynchronous_Queue\b|\bProtected_Type\b", re.IGNORECASE)),
    ], 3),
    ("SPARK 2014", [
        ("SPARK_Mode", re.compile(r"\bSPARK_Mode\b", re.IGNORECASE)),
        ("Ghost", re.compile(r"\bGhost\b", re.IGNORECASE)),
        ("GNATprove", re.compile(r"\bGNATprove\b", re.IGNORECASE)),
        ("Praxis", re.compile(r"\bPraxis\b", re.IGNORECASE)),
        ("SPARK_Keyword", re.compile(r"\bSPARK\b", re.IGNORECASE)),
        ("Pre_Post", re.compile(r"\bPre\s*=>|Post\s*=>", re.IGNORECASE)),
        ("Ghost_Var", re.compile(r"\bGhost\s+\w+", re.IGNORECASE)),
    ], 2),
    ("Ada 2005", [
        ("interfaces", re.compile(r"\binterfaces\b", re.IGNORECASE)),
        ("aliased", re.compile(r"\baliased\b", re.IGNORECASE)),
        ("protected_type", re.compile(r"\bprotected\s+type\b", re.IGNORECASE)),
        ("abstract_interface", re.compile(r"\babstract\s+interface\b", re.IGNORECASE)),
        ("assertion", re.compile(r"\bassert\b|\bassertion_policy\b", re.IGNORECASE)),
        ("container_types", re.compile(r"\bAda\.Containers\b", re.IGNORECASE)),
    ], 2),
    ("Ada 95", [
        ("tagged", re.compile(r"\btagged\b", re.IGNORECASE)),
        ("abstract_tagged", re.compile(r"\babstract\s+tagged\b", re.IGNORECASE)),
        ("override", re.compile(r"\boverride\b", re.IGNORECASE)),
        ("interface", re.compile(r"\binterface\b\b", re.IGNORECASE)),
        ("protected", re.compile(r"\bprotected\b", re.IGNORECASE)),
        ("task_type", re.compile(r"\btask\s+type\b", re.IGNORECASE)),
        ("generic_formal", re.compile(r"\bgeneric\s+formal\b", re.IGNORECASE)),
    ], 2),
    ("Ada 83", [
        ("procedure", re.compile(r"\bprocedure\b", re.IGNORECASE)),
        ("function", re.compile(r"\bfunction\b", re.IGNORECASE)),
        ("package_body", re.compile(r"\bpackage\s+body\b", re.IGNORECASE)),
        ("package_spec", re.compile(r"\bpackage\s+\w+\s+is\b", re.IGNORECASE)),
        ("pragma", re.compile(r"\bpragma\b", re.IGNORECASE)),
    ], 2),
]


def detect_ada_standard(content: str) -> str:
    """Detect the Ada language standard using a scoring-based approach.

    Newer standards are supersets of older ones. We score the content
    against features from Ada 2022 down to Ada 83. The first standard
    whose matched feature count meets or exceeds its minimum threshold
    wins. Returns the matched standard label or 'Unknown'.
    """
    for standard_name, features, min_match in _STANDARD_FEATURES:
        match_count = sum(1 for _, pattern in features if pattern.search(content))
        if match_count >= min_match:
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

_DEFECT_FAMILIES = ("syntax", "context", "visibility", "contract", "mismatch")


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

    return None


_DEFECT_FAMILY_ORDER: tuple[str, ...] = _DEFECT_FAMILIES
# Cap defect turns per pair so the dataset does not skew negative.
_MAX_DEFECTS_PER_PAIR = 2


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
    for family in _DEFECT_FAMILY_ORDER:
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
        user_msg = (
            f"The following {_standard_label(standard)} code for `{package or 'this unit'}` does not "
            f"compile. Find the defect and provide the corrected code.\n\n"
            f"```ada\n{broken}\n```\n\n"
            f"Source: {source}"
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
            f"You are an Ada language expert specializing in {_standard_label(standard)}. "
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


def _sample_guidance(guidance_text: str, every_n: int = 4) -> str:
    """Return guidance_text on a deterministic 1-in-N subset of calls."""
    if not guidance_text:
        return ""
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


def extract_ada_code_blocks(doc_root: Path) -> list[str]:
    """Extract Ada code blocks from RST and Markdown files under doc_root.

    Deduplicates blocks and keeps only snippets that look like Ada units.
    """
    blocks: list[str] = []
    seen: set[str] = set()
    for pattern in ("*.rst", "*.md"):
        for doc_path in sorted(doc_root.rglob(pattern)):
            try:
                content = doc_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            found = _RST_ADA_BLOCK.findall(content) + _MD_ADA_BLOCK.findall(content)
            for raw in found:
                code = textwrap.dedent(raw).strip("\n")
                if not (30 <= len(code) <= 4000):
                    continue
                if not _looks_like_ada(code):
                    continue
                key = re.sub(r"\s+", " ", code)
                if key in seen:
                    continue
                seen.add(key)
                blocks.append(code)
    logger.info("Extracted %d unique Ada code blocks from %s", len(blocks), doc_root)
    return blocks


def build_doc_training_turns(
    doc_sources: list[Path],
    guidance_text: str = "",
) -> list[dict[str, list[dict[str, str]]]]:
    """Build documentation-QA style training turns from Ada code blocks.

    Two turn kinds per code block:
    - a plain code-explanation turn (explanation distilled from the
      surrounding documentation is not available, so the assistant
      restates and completes the snippet),
    - an STE explanation turn where the assistant explains the code in
      Simplified Technical English and defines each technical term at
      its first use. These turns teach the model to write docs that
      obey the STE rules it is prompted with.
    """
    turns: list[dict[str, list[dict[str, str]]]] = []
    for doc_root in doc_sources:
        if not doc_root.exists():
            logger.warning("Documentation source not found, skipping: %s", doc_root)
            continue
        for code in extract_ada_code_blocks(doc_root):
            standard = detect_ada_standard(code)
            cleaned_code = strip_noisy_comments(code)
            user_msg = (
                f"Explain the following {_standard_label(standard)} code and describe what it demonstrates.\n\n"
                f"```ada\n{cleaned_code}\n```"
            )
            # Plain completion answer
            turns.append({
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"```ada\n{cleaned_code}\n```"},
                ],
            })
            # STE explanation answer: written by the rules we distill,
            # with each technical term defined at first use.
            ste_explanation = _ste_explanation(cleaned_code, standard)
            ste_user = (
                f"Explain the following {_standard_label(standard)} code for documentation. "
                "Follow Simplified Technical English rules and define technical "
                "terms at first use.\n\n"
                f"```ada\n{cleaned_code}\n```"
            )
            turns.append({
                "messages": [
                    {
                        "role": "system",
                        "content": _compose_system_prompt(standard, STE_RULE_BLOCK, build_technical_term_glossary()),
                    },
                    {"role": "user", "content": ste_user},
                    {"role": "assistant", "content": ste_explanation},
                ],
            })
    if turns and guidance_text:
        # Variant system prompts carrying the agent-skill guidance teach the
        # model to write current-toolchain Ada/SPARK (contracts, Alire, proof).
        sample_turns = turns[: max(len(turns) // 4, 1)]
        for turn in sample_turns:
            turn["messages"].insert(0, {
                "role": "system",
                "content": (
                    "You are a specialized Ada/SPARK AI agent. You write strictly "
                    "conforming, idiomatic code, prioritize contract annotations "
                    "(Pre/Post), and target the current GNAT/Alire toolchain.\n\n"
                    f"{guidance_text}"
                ),
            })
    return turns


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
) -> list[dict[str, list[dict[str, str]]]]:
    """Build question/answer turns from the AdaCore agent skills.

    For each skill, asks its canonical questions and answers with the
    relevant section of the skill documentation, rewritten to the STE
    style. These turns ground the model in real toolchain usage: gnatprove
    invocation rules, Alire workflows, gnattest VCS rules, and so on.
    """
    turns: list[dict[str, list[dict[str, str]]]] = []
    for skill_name, questions in _TOOLCHAIN_QUESTIONS.items():
        doc_text = toolchain_docs.get(skill_name, "")
        if not doc_text:
            continue
        for question, anchor_regex, fallback in questions:
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
            turns.append({
                "messages": [
                    {
                        "role": "system",
                        "content": _compose_system_prompt("SPARK 2014", ste_rules, glossary),
                    },
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": assistant_msg},
                ],
            })
    logger.info("Built %d toolchain QA turns from AdaCore skills", len(turns))
    return turns


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
            "You are a specialized Ada/SPARK AI agent. You write strictly "
            "conforming, idiomatic code, prioritize contract annotations "
            "(Pre/Post), and target the current GNAT/Alire toolchain.\n\n"
            f"{guidance_text}"
        )
    _ = context  # context kept for CLI compatibility; system prompt covers safety

    if spec_content and impl_content:
        user_msg = (
            f"Ada {_standard_label(standard)} - Package specification for `{package}`.\n\n"
            f"Source: {source}\n\n"
            f"Please complete the following package body based on the "
            f"specification below.\n\n---\n\n"
            f"```ada\n{spec_content}\n```\n\n"
            f"Provide the corresponding package body implementation."
        )
        assistant_msg = f"```ada\n{impl_content}\n```"
    elif spec_content and not impl_content:
        user_msg = (
            f"Ada {_standard_label(standard)} - Package specification for `{package}`.\n\n"
            f"Source: {source}\n\n"
            f"Please provide the full package body implementation for "
            f"the following specification.\n\n---\n\n"
            f"```ada\n{spec_content}\n```\n\n"
            f"Provide the corresponding package body."
        )
        assistant_msg = ""
    elif impl_content and not spec_content:
        user_msg = (
            f"Ada {_standard_label(standard)} - Implementation unit for `{package}`.\n\n"
            f"Source: {source}\n\n"
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
) -> int:
    """Run the full ingestion -> pairing -> sanitization -> JSONL pipeline.

    Processes all input directories plus extra input directories
    (adacovex, Ada_CRDT, TLALOC, ada-eval), extracts Ada code blocks from
    documentation sources (learn), embeds agent-skill guidance
    (ada-spark, SimpleEnglish, AdaCore skills) into system prompts, and
    generates correct-vs-wrong defect pairs plus toolchain QA turns.
    Derives dataset structure from ada-eval methodology if available.

    Returns the number of valid training turns written to the output file.
    """
    all_turns: list[dict[str, list[dict[str, str]]]] = []
    turn_counts: dict[str, int] = {}
    eval_info = eval_methodology or {}

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

        for idx, pair in enumerate(pairs):
            spec_path = pair["spec"]
            impl_path = pair["impl"]
            package = pair["package"]
            if isinstance(package, Path):
                package = str(package)
            source = str(resolved_input)

            spec_content: str | None = None
            impl_content: str | None = None

            if isinstance(spec_path, Path):
                spec_content = read_and_sanitize(spec_path)
            if isinstance(impl_path, Path):
                impl_content = read_and_sanitize(impl_path)

            if spec_content is None and impl_content is None:
                logger.warning("Skipping pair %d - both spec and impl unparseable", idx)
                continue

            combined_content = ""
            if spec_content:
                combined_content += spec_content + "\n"
            if impl_content:
                combined_content += impl_content + "\n"

            standard = detect_ada_standard(combined_content)
            logger.debug(
                "Pair %d - Package: %s | Standard: %s | Source: %s",
                idx, package, standard, source,
            )

            # Clean noisy inline comments so the model learns STE comments.
            if spec_content:
                spec_content = strip_noisy_comments(spec_content)
            if impl_content:
                impl_content = strip_noisy_comments(impl_content)

            # Include .gpr context if available
            if discovered.get("gpr"):
                for gpr_path in discovered["gpr"][:3]:
                    gpr_content = read_and_sanitize(gpr_path)
                    if gpr_content:
                        combined_content += f"\n---\nProject file reference ({gpr_path.name}):\n{gpr_content}\n"
                        break

            turn = format_training_turn(
                standard, spec_content, impl_content, package, context,
                source=source,
                guidance_text=_sample_guidance(guidance_text),
                ste_rules=ste_rules,
                glossary=glossary,
            )
            if turn["messages"]:
                all_turns.append(turn)
                turn_counts["code_pair"] = turn_counts.get("code_pair", 0) + 1

            # Correct-vs-wrong defect pairs from the combined unit.
            if enable_defect_pairs and combined_content.strip():
                defect_turns = build_defect_turns(
                    combined_content.strip(), standard, package, source,
                    ste_rules, glossary,
                )
                all_turns.extend(defect_turns)
                turn_counts["defect_pair"] = turn_counts.get("defect_pair", 0) + len(defect_turns)

    # Documentation-QA turns from code blocks embedded in course material
    if doc_dirs:
        doc_turns = build_doc_training_turns(doc_dirs, guidance_text)
        all_turns.extend(doc_turns)
        turn_counts["doc_qa"] = turn_counts.get("doc_qa", 0) + len(doc_turns)

    # Toolchain QA turns from the AdaCore skills
    if toolchain_docs:
        qa_turns = build_toolchain_qa_turns(toolchain_docs, ste_rules, glossary)
        all_turns.extend(qa_turns)
        turn_counts["toolchain_qa"] = turn_counts.get("toolchain_qa", 0) + len(qa_turns)

    # Self-check: count STE violations in our own generated assistant prose.
    # Messages with code fences are exempt: the fence itself triggers the
    # semicolon/identifier checks, and code is exempt from STE by rule 10.
    violation_count = 0
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

    # Write JSONL output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for turn in all_turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    # Write evaluation methodology alongside the dataset
    meta_path = output_file.parent / "dataset_metadata.json"
    metadata: dict[str, Any] = {
        "total_turns": len(all_turns),
        "turn_counts_by_kind": turn_counts,
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
        "Dataset written to %s - %d training turns (%s).",
        output_file, len(all_turns),
        ", ".join(f"{k}={v}" for k, v in sorted(turn_counts.items())) or "no turns",
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
             "(e.g., ../adacovex, ../Ada_CRDT, ../Ada-83-TLALOC). "
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
             "(e.g., ../learn). Can be specified multiple times.",
    )
    parser.add_argument(
        "--guidance-dir", type=Path, action="append", default=[],
        help="Agent-skill source whose markdown guidance is embedded into "
             "system prompts (e.g., ../ada-spark, ../SimpleEnglish, ../skills). "
             "Repeatable.",
    )
    parser.add_argument(
        "--no-defect-pairs", action="store_true",
        help="Disable correct-vs-wrong defect pair generation.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default extra input directories if none provided
    extra_dirs = args.extra_input_dir if args.extra_input_dir else DEFAULT_EXTRA_INPUT_DIRS
    doc_dirs = args.doc_dir if args.doc_dir else [Path("..") / d for d in DOC_SOURCE_DIRS]
    guidance_dirs = (
        args.guidance_dir if args.guidance_dir
        else [Path("..") / d for d in AGENT_SKILL_SOURCE_DIRS]
    )

    # Load evaluation methodology from ada-eval
    eval_methodology = load_eval_methodology()

    # Ensure all input directories exist (warn if not)
    all_dirs = [args.input_dir] + extra_dirs
    for d in all_dirs:
        if not d.exists():
            logger.warning("Input directory not found: %s", d)

    output_file = args.output_dir / "dataset.jsonl"
    count = build_dataset(
        input_dirs=[args.input_dir],
        extra_input_dirs=extra_dirs,
        output_file=output_file,
        context=args.context,
        eval_methodology=eval_methodology,
        doc_dirs=doc_dirs,
        guidance_dirs=guidance_dirs,
        enable_defect_pairs=not args.no_defect_pairs,
    )
    print(f"Dataset built: {count} training turns -> {output_file}")


if __name__ == "__main__":
    main()
