"""build_dataset.py - Intermediate ingestion and extraction script for q3as.

Discovers Ada source files (.ads, .adb) and .gpr project files
across target directory trees (including ../adacovex and ../Ada_CRDT),
pairs matching specification/implementation files, detects the target Ada
standard via heuristic keyword analysis, sanitizes content, and writes a
standardized JSONL dataset formatted with the OpenAI/Qwen chat template.

Additionally ingests:
- ../ada-eval          - eval sample sources as extra Ada code trees
- ../learn             - AdaCore's learn.adacore.com courses; Ada code blocks
                         embedded in the RST course material are extracted as
                         documentation-QA style training turns (CC-BY-4.0)
- ../ada-spark         - the ada-spark agent skill (MIT); SKILL.md and
                         agent-knowledge guidance is embedded into the system
                         prompt so the model learns current-toolchain
                         conventions (contracts, Alire, SPARK proof)

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
import re
import sys
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
# Directories whose markdown guidance (SKILL.md, agent-knowledge) is embedded
# into the system prompt as Ada/SPARK conventions.
GUIDANCE_SOURCE_DIRS = ["ada-spark"]
# Default path to ada-eval methodology directory
ADA_EVAL_DIR = Path("../ada-eval")

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
        logger.info("ada-eval not found at %s — skipping methodology load", ADA_EVAL_DIR)
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

def pair_files(discovered: dict[str, list[Path]]) -> list[dict[str, Path | None]]:
    """Pair .ads (specification) with .adb (body) files by package name.

    If a direct package-name match fails, falls back to filename convention
    (e.g., ``foo.ads`` pairs with ``foo.adb``). Unpaired files are emitted
    as standalone entries.
    """
    pairs: list[dict[str, Path | None]] = []
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
        raise ValueError("File contains excessive null bytes - likely binary/corrupted")

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
# Documentation & Guidance Ingestion (learn, ada-spark)
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


def load_guidance_text(source_dirs: list[Path]) -> str:
    """Concatenate SKILL.md / agent-knowledge guidance from agent-skill repos."""
    parts: list[str] = []
    for root in source_dirs:
        if not root.exists():
            logger.warning("Guidance source not found, skipping: %s", root)
            continue
        candidates = [root / "SKILL.md", *sorted(root.glob("agent-knowledge/*.md"))]
        for md_path in candidates:
            if not md_path.is_file():
                continue
            try:
                text = md_path.read_text(encoding="utf-8").strip()
            except (UnicodeDecodeError, OSError):
                continue
            if text:
                parts.append(text)
    if parts:
        logger.info(
            "Loaded %d guidance documents from %s",
            len(parts), ", ".join(str(d) for d in source_dirs),
        )
    return "\n\n".join(parts)


def build_doc_training_turns(
    doc_sources: list[Path],
    guidance_text: str = "",
) -> list[dict[str, list[dict[str, str]]]]:
    """Build documentation-QA style training turns from Ada code blocks."""
    turns: list[dict[str, list[dict[str, str]]]] = []
    for doc_root in doc_sources:
        if not doc_root.exists():
            logger.warning("Documentation source not found, skipping: %s", doc_root)
            continue
        for code in extract_ada_code_blocks(doc_root):
            standard = detect_ada_standard(code)
            user_msg = (
                f"Explain the following {standard} code and describe what it demonstrates.\n\n"
                f"```ada\n{code}\n```"
            )
            turns.append({
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"```ada\n{code}\n```"},
                ],
            })
    if turns and guidance_text:
        # Variant system prompts carrying the ada-spark guidance teach the
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
) -> dict[str, list[dict[str, str]]]:
    """Format a single training turn using the OpenAI/Qwen chat template.

    Produces a dict with a ``messages`` key containing the ``system``,
    ``user``, and ``assistant`` message objects. When *guidance_text* is
    provided (from the ada-spark skill), a portion of turns embed it in the
    system prompt so the model internalizes current-toolchain conventions.
    """
    system_msg = (
        f"You are an Ada language expert specializing in {standard}. "
        f"You produce safe, correct, and standards-compliant Ada code. "
        f"Safety requirements: {context}. "
        "Always use the appropriate Ada standard syntax and conventions."
    )
    if guidance_text and hash(package or "") % 4 == 0:
        system_msg = (
            "You are a specialized Ada/SPARK AI agent. You write strictly "
            "conforming, idiomatic code, prioritize contract annotations "
            "(Pre/Post), and target the current GNAT/Alire toolchain.\n\n"
            f"{guidance_text}\n\n"
            + system_msg
        )

    if spec_content and impl_content:
        user_msg = (
            f"Ada {standard} - Package specification for `{package}`.\n\n"
            f"Source: {source}\n\n"
            f"Please complete the following package body based on the "
            f"specification below.\n\n---\n\n"
            f"```ada\n{spec_content}\n```\n\n"
            f"Provide the corresponding package body implementation."
        )
        assistant_msg = f"```ada\n{impl_content}\n```"
    elif spec_content and not impl_content:
        user_msg = (
            f"Ada {standard} - Package specification for `{package}`.\n\n"
            f"Source: {source}\n\n"
            f"Please provide the full package body implementation for "
            f"the following specification.\n\n---\n\n"
            f"```ada\n{spec_content}\n```\n\n"
            f"Provide the corresponding package body."
        )
        assistant_msg = ""
    elif impl_content and not spec_content:
        user_msg = (
            f"Ada {standard} - Implementation unit for `{package}`.\n\n"
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
) -> int:
    """Run the full ingestion -> pairing -> sanitization -> JSONL pipeline.

    Processes all input directories plus extra input directories
    (adacovex, Ada_CRDT, TLALOC, ada-eval), extracts Ada code blocks from
    documentation sources (learn), and embeds agent-skill guidance
    (ada-spark) into system prompts. Derives dataset structure from
    ada-eval methodology if available.

    Returns the number of valid training turns written to the output file.
    """
    all_turns: list[dict[str, list[dict[str, str]]]] = []
    eval_info = eval_methodology or {}
    guidance_text = load_guidance_text(guidance_dirs or [])

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
            source = str(resolved_input)

            spec_content: str | None = None
            impl_content: str | None = None

            if spec_path:
                spec_content = read_and_sanitize(spec_path)
            if impl_path:
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

            # Include .gpr context if available
            if discovered.get("gpr"):
                for gpr_path in discovered["gpr"][:3]:
                    gpr_content = read_and_sanitize(gpr_path)
                    if gpr_content:
                        combined_content += f"\n---\nProject file reference ({gpr_path.name}):\n{gpr_content}\n"
                        break

            turn = format_training_turn(
                standard, spec_content, impl_content, package, context,
                source=source, guidance_text=guidance_text,
            )
            if turn["messages"]:
                all_turns.append(turn)

    # Documentation-QA turns from code blocks embedded in course material
    if doc_dirs:
        all_turns.extend(build_doc_training_turns(doc_dirs, guidance_text))

    # Write JSONL output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for turn in all_turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    # Write evaluation methodology alongside the dataset
    if eval_info:
        meta_path = output_file.parent / "dataset_metadata.json"
        metadata = {
            "total_turns": len(all_turns),
            "evaluation_methodology": eval_info,
            "input_dirs": [str(d.resolve()) for d in input_dirs + extra_input_dirs],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info("Dataset metadata written to %s", meta_path)

    logger.info(
        "Dataset written to %s - %d training turns.", output_file, len(all_turns)
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
        help="Agent-skill source whose SKILL.md/agent-knowledge guidance is "
             "embedded into system prompts (e.g., ../ada-spark). Repeatable.",
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
        else [Path("..") / d for d in GUIDANCE_SOURCE_DIRS]
    )

    # Load evaluation methodology from ada-eval
    eval_methodology = load_eval_methodology()

    # Ensure all input directories exist (warn if not)
    all_dirs = [args.input_dir] + extra_dirs
    for d in all_dirs:
        if not d.exists():
            logger.warning("Extra input directory not found: %s", d)

    output_file = args.output_dir / "dataset.jsonl"
    count = build_dataset(
        input_dirs=[args.input_dir],
        extra_input_dirs=extra_dirs,
        output_file=output_file,
        context=args.context,
        eval_methodology=eval_methodology,
        doc_dirs=doc_dirs,
        guidance_dirs=guidance_dirs,
    )
    print(f"Dataset built: {count} training turns -> {output_file}")


if __name__ == "__main__":
    main()
