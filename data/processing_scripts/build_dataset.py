"""build_dataset.py - Intermediate ingestion and extraction script for q3as.

Discovers Ada source files (.ads, .adb), .gpr project files, and alire.toml
metadata across target directory trees (including ../adacovex and ../Ada_CRDT),
pairs matching specification/implementation files, detects the target Ada
standard via heuristic keyword analysis, sanitizes content, and writes a
standardized JSONL dataset formatted with the OpenAI/Qwen chat template.

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
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ADA_SPEC_EXTENSIONS = {".ads", ".adb"}
PROJECT_EXTENSIONS = {".gpr"}
METADATA_FILES = {"alire.toml"}
DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "dataset.jsonl"
# Default extra input directories for additional training data sources
DEFAULT_EXTRA_INPUT_DIRS = [
    Path("../adacovex"),
    Path("../Ada_CRDT"),
    Path("../Ada-83-TLALOC"),
]
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

# ---- Ada standard detection heuristics ------------------------------------ #
# Ordered from most specific to least specific; first match wins.
_STANDARD_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    ("SPARK 2014", [
        re.compile(r"\bSPARK_Mode\b", re.IGNORECASE),
        re.compile(r"\bGhost\b", re.IGNORECASE),
        re.compile(r"\bGNATprove\b", re.IGNORECASE),
        re.compile(r"\bPraxis\b", re.IGNORECASE),
    ]),
    ("Ada 2022", [
        re.compile(r"\bStatic_Pure\b", re.IGNORECASE),
        re.compile(r"\bPure_Global\b", re.IGNORECASE),
        re.compile(r"\bContract_Cases\b", re.IGNORECASE),
        re.compile(r"\bLoop_Invariant\b", re.IGNORECASE),
        re.compile(r"\bDynamic_Pure\b", re.IGNORECASE),
    ]),
    ("Ada 2012", [
        re.compile(r"\bPre\s*=>", re.IGNORECASE),
        re.compile(r"\bPost\s*=>", re.IGNORECASE),
        re.compile(r"\bType_Invariant\b", re.IGNORECASE),
        re.compile(r"\bSubtype_Contract\b", re.IGNORECASE),
        re.compile(r"\bGlobal\s*=>", re.IGNORECASE),
        re.compile(r"\bDepends\s*=>", re.IGNORECASE),
        re.compile(r"with\s+Pre\b", re.IGNORECASE),
        re.compile(r"with\s+Post\b", re.IGNORECASE),
    ]),
    ("Ada 2005", [
        re.compile(r"\binterfaces\b", re.IGNORECASE),
        re.compile(r"\baliased\b", re.IGNORECASE),
        re.compile(r"\bprotected\s+type\b", re.IGNORECASE),
    ]),
    ("Ada 95", [
        re.compile(r"\btagged\b", re.IGNORECASE),
        re.compile(r"\babstract\s+tagged\b", re.IGNORECASE),
        re.compile(r"\boverride\b", re.IGNORECASE),
        re.compile(r"\binterface\b\b", re.IGNORECASE),
    ]),
    ("Ada 83", [
        re.compile(r"procedure\b", re.IGNORECASE),
        re.compile(r"function\b", re.IGNORECASE),
        re.compile(r"package\s+body\b", re.IGNORECASE),
    ]),
]

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
    """Walk *root* and classify discovered files by category.

    Returns a dict with keys 'ads', 'adb', 'gpr', 'metadata', and 'other'.
    Symlinks are resolved automatically via ``Path.resolve()``.
    """
    discovered: dict[str, list[Path]] = {"ads": [], "adb": [], "gpr": [], "metadata": [], "other": []}

    if not root.exists():
        logger.warning("Input directory does not exist: %s", root)
        return discovered

    for dirpath, dirnames, filenames in os_walk_safe(root):
        root_path = Path(dirpath)
        for fname in filenames:
            full_path = (root_path / fname).resolve()
            ext = full_path.suffix.lower()

            if ext in ADA_SPEC_EXTENSIONS:
                discovered["ads" if ext == ".ads" else "adb"].append(full_path)
            elif ext in PROJECT_EXTENSIONS:
                discovered["gpr"].append(full_path)
            elif fname in METADATA_FILES:
                discovered["metadata"].append(full_path)
            else:
                discovered["other"].append(full_path)

    logger.info(
        "Discovered %d .ads, %d .adb, %d .gpr, %d metadata files in %s",
        len(discovered["ads"]), len(discovered["adb"]),
        len(discovered["gpr"]), len(discovered["metadata"]), root,
    )
    return discovered


def os_walk_safe(root: Path, max_depth: int = 5):
    """Safely walk a directory tree, skipping unreadable or broken symlinks.

    Skips .git, .cache, __pycache__, .pytest_cache, and .github.
    Limits recursion depth to avoid walking excessively deep paths.
    """
    skip_dirs = {".git", ".cache", "__pycache__", ".pytest_cache", ".github"}

    def _walk(current: Path, depth: int):
        if depth > max_depth:
            return
        try:
            entries = list(current.iterdir())
        except PermissionError:
            return
        dirnames = []
        filenames = []
        for entry in entries:
            if entry.is_dir() and entry.name not in skip_dirs:
                dirnames.append(entry.name)
                yield from _walk(entry, depth + 1)
            elif entry.is_file():
                filenames.append(entry.name)
        yield current, dirnames, filenames

    try:
        yield from _walk(root, 0)
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
# Standard Detection
# --------------------------------------------------------------------------- #

def detect_ada_standard(content: str) -> str:
    """Detect the Ada language standard based on keyword heuristic analysis.

    Checks patterns ordered from most specific (SPARK 2014, Ada 2022) to
    least specific (Ada 83). Returns the matched standard label.
    """
    for standard_name, patterns in _STANDARD_PATTERNS:
        for pattern in patterns:
            if pattern.search(content):
                return standard_name
    return "Unknown"


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
# JSONL Formatting
# --------------------------------------------------------------------------- #

def format_training_turn(
    standard: str,
    spec_content: str | None,
    impl_content: str | None,
    package: str | None,
    context: str,
    source: str = "",
) -> dict[str, list[dict[str, str]]]:
    """Format a single training turn using the OpenAI/Qwen chat template.

    Produces a dict with a ``messages`` key containing the ``system``,
    ``user``, and ``assistant`` message objects.
    """
    system_msg = (
        f"You are an Ada language expert specializing in {standard}. "
        f"You produce safe, correct, and standards-compliant Ada code. "
        f"Safety requirements: {context}. "
        "Always use the appropriate Ada standard syntax and conventions."
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
) -> int:
    """Run the full ingestion -> pairing -> sanitization -> JSONL pipeline.

    Processes all input directories plus extra input directories
    (adacovex, Ada_CRDT). Derives dataset structure from ada-eval
    methodology if available.

    Returns the number of valid training turns written to the output file.
    """
    all_turns: list[dict[str, list[dict[str, str]]]] = []
    eval_info = eval_methodology or {}

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
                standard, spec_content, impl_content, package, context, source=source,
            )
            if turn["messages"]:
                all_turns.append(turn)

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
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default extra input directories if none provided
    extra_dirs = args.extra_input_dir if args.extra_input_dir else DEFAULT_EXTRA_INPUT_DIRS

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
    )
    print(f"Dataset built: {count} training turns -> {output_file}")


if __name__ == "__main__":
    main()
