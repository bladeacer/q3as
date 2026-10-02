#!/usr/bin/env python3
"""gen_verified_spark.py - completion turns from real SPARK units, verified.

The Ada-Algorithms monorepo ships SPARK2 trees: subprograms with real
contracts (Pre bounds, quantified Post) whose READMEs claim proof
friendliness, but nothing in the tree demonstrates it. This tool closes
that gap with the local prover:

1. Take spec/body pairs from the source trees (skipping units that depend
   on non-predefined library units, which cannot compile standalone).
2. Wrap each pair in a synthetic package and prove it with the real
   ``gnatprove`` in batches, exactly like ``gen_contract_mutations.py``.
3. Keep only units whose every check comes out proved, and emit a
   completion turn in the shape of the evaluation set: the spec
   declaration in, the proved subprogram out.

Unlike the synthetic contract turns, the code here is not authored from
templates: it is real algorithm code, and the prover decides what trains.

Output: ``data/processed/verified_spark.jsonl`` in the chat-record shape
the dataset builder ingests (``--extra-turns``); the builder derives its
defect and variant turns from these records like any other source.

Cached: the run is skipped when this file, ``alire_env.py``, and the
parameters are unchanged since the last successful build; ``--force``
regenerates. A run that verifies nothing does not record a stamp.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data" / "processing_scripts"))

import alire_env
import progress as progress_mod
import stage_state
from alire_env import alire_env_path, find_tool

OUTPUT = ROOT / "data" / "processed" / "verified_spark.jsonl"

MONOREPO = ROOT / "data" / "raw_repos" / "RobertBoettcherSF" / "Ada-Algorithms"


def default_sources() -> list[Path]:
    """Every SPARK2 tree of the monorepo, in topic order."""
    return sorted(
        path for path in MONOREPO.glob("*/SPARK2") if path.is_dir()
    )


def _source_label(spec_path: Path) -> str:
    """``Ada-Algorithms:<topic>/SPARK2/<file>.ads``.

    The topic directory is part of the label: two topics carry SPARK2 units
    of the same file name, so the tree alone does not identify the source.
    """
    try:
        relative = spec_path.resolve().relative_to(MONOREPO)
    except ValueError:
        relative = Path(spec_path.name)
    return f"Ada-Algorithms:{relative.as_posix()}"

# Only units whose context clauses name these roots can be compiled
# standalone: a with of a sibling unit would need that unit's tree, which
# the synthetic one-package-per-unit project does not carry.
_ALLOWED_WITH_ROOTS = frozenset({"Ada", "System", "Interfaces", "GNAT"})

_GPR = """project Prv is
   for Languages use ("Ada");
   for Source_Dirs use ("src");
   for Object_Dir use "obj";
end Prv;
"""

_DECL_RE = re.compile(
    r"\b(?P<kind>procedure|function)\s+(?P<name>\w+)\s*"
    r"(?P<params>\([^;]*?\))?\s*"
    r"(?:return\s+(?P<ret>[\w.']+)\s*)?"
    r"(?P<aspects>with\s+[^;]+)?;",
    re.DOTALL,
)


def _with_clauses(text: str) -> list[str]:
    return re.findall(r"^\s*with\s+([\w.]+)\s*;", text, re.MULTILINE)


def _standalone(spec_text: str, body_text: str) -> bool:
    """True when the unit needs no library beyond the predefined roots."""
    for unit in _with_clauses(spec_text) + _with_clauses(body_text):
        if unit.split(".")[0] not in _ALLOWED_WITH_ROOTS:
            return False
    return True


def _extract_subprogram_body(body_text: str, name: str) -> str | None:
    """The ``procedure Name ... begin ... end Name;`` span, or None."""
    match = re.search(rf"\b(?:procedure|function)\s+{name}\b", body_text)
    if not match:
        return None
    end_match = re.search(rf"\bend\s+{name}\s*;", body_text[match.start():], re.DOTALL)
    if not end_match:
        return None
    return body_text[match.start():match.start() + end_match.end()]


def _normalize_decl(decl: str) -> str:
    return " ".join(decl.split())


def discover_units(
    source_dirs: list[Path], limit: int = 0
) -> list[dict[str, Any]]:
    """Whole spec/body files that are standalone and carry a subprogram.

    The original package spec and body are proven verbatim: the bodies
    reference the types declared in their own package spec, so a synthetic
    spec carrying only the subprogram declaration would not compile. Units
    across trees can share a package name, so the first of each name wins
    (a batch directory cannot hold two of the same package). ``limit`` of 0
    or less means every discovered unit.
    """
    units: list[dict[str, Any]] = []
    skipped_deps = 0
    skipped_name = 0
    seen_packages: set[str] = set()
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            logger.warning("source directory not found, skipping: %s", source_dir)
            continue
        for spec_path in sorted(source_dir.glob("*.ads")):
            if limit > 0 and len(units) >= limit:
                break
            body_path = spec_path.with_suffix(".adb")
            if not body_path.is_file():
                continue
            try:
                spec_text = spec_path.read_text(encoding="utf-8", errors="replace")
                body_text = body_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not _standalone(spec_text, body_text):
                skipped_deps += 1
                continue
            package_match = re.search(
                r"\bpackage\s+(?!body\b)(\w+)\s*(?:with\s+[^;]+?)?\s*is\b",
                spec_text,
            )
            if package_match is None:
                continue
            package = package_match.group(1)
            if package in seen_packages:
                skipped_name += 1
                continue
            decl_match = _DECL_RE.search(spec_text)
            if decl_match is None:
                continue
            name = decl_match.group("name")
            if _extract_subprogram_body(body_text, name) is None:
                continue
            seen_packages.add(package)
            units.append({
                "name": name,
                "package": package,
                "decl": _normalize_decl(decl_match.group(0)),
                "has_contract": "with " in decl_match.group(0),
                "spec": spec_text,
                "body": body_text,
                "source": _source_label(spec_path),
            })
    logger.info(
        "Discovered %d standalone units (%d skipped for local dependencies,"
        " %d duplicate package names)",
        len(units), skipped_deps, skipped_name,
    )
    return units


# --------------------------------------------------------------------------- #
# gnatprove verification (same discipline as gen_contract_mutations.py)
# --------------------------------------------------------------------------- #


def _alire_env() -> dict[str, str]:
    return {**os_environ(), "PATH": alire_env_path()}


def os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def run_gnatprove(project_dir: Path, timeout_s: int = 600) -> tuple[bool, str]:
    """Prove everything in the project. Returns (all_proved, summary).

    A run that hits the timeout yields ``(False, "")`` on purpose. The
    partial summary of a killed run cannot be trusted: a package listed as
    proved may still have had checks outstanding when the prover died, and
    the empty summary makes every package in the batch fall through to
    "not proved", so a timeout drops the batch instead of half-verifying it.
    """
    gpr = project_dir / "main.gpr"
    if not gpr.exists():
        gpr.write_text(_GPR, encoding="utf-8")
    gnatprove = find_tool("gnatprove")
    try:
        proc = subprocess.run(
            [str(gnatprove), f"-P{gpr}", "-j0", "--level=1", "--mode=prove"],
            cwd=project_dir, capture_output=True, text=True, timeout=timeout_s,
            env=_alire_env(), check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("gnatprove did not finish within %ds", timeout_s)
        return False, ""
    out_file = project_dir / "obj" / "gnatprove" / "gnatprove.out"
    summary = out_file.read_text(encoding="utf-8", errors="replace") if out_file.exists() else ""
    output = proc.stdout + proc.stderr
    all_proved = proc.returncode == 0 and "not proved" not in summary and "error:" not in output
    return all_proved, summary


def _unit_proved(package: str, summary: str) -> bool:
    """Per-package status; False when absent, so parse failure never trains."""
    lines = re.findall(rf"^\s*{package}(?:\.\w+)? at \S+ .*$", summary, re.MULTILINE)
    if not lines:
        return False
    return all("and proved" in ln and "not proved" not in ln for ln in lines)


def batch_verify(batch: list[dict[str, Any]], workdir: Path) -> dict[str, bool]:
    """Verify a batch in one gnatprove run. Returns {package: proved}."""
    src = workdir / "src"
    src.mkdir(parents=True, exist_ok=True)
    packages = []
    for unit in batch:
        package = unit["package"]
        packages.append(package)
        (src / f"{package.lower()}.ads").write_text(unit["spec"], encoding="utf-8")
        (src / f"{package.lower()}.adb").write_text(unit["body"], encoding="utf-8")
    all_proved, summary = run_gnatprove(workdir)
    return {
        package: (True if all_proved else _unit_proved(package, summary))
        for package in packages
    }


# --------------------------------------------------------------------------- #
# Turn building
# --------------------------------------------------------------------------- #


def build_turns(units: list[dict[str, Any]], batch_size: int = 8) -> list[dict[str, Any]]:
    """Verify in batches and emit completion turns for the proved units."""
    records: list[dict[str, Any]] = []
    verified = 0
    batches = max(1, -(-len(units) // batch_size))
    progress = progress_mod.Progress("gnatprove batches", batches, log=logger)
    for start in range(0, len(units), batch_size):
        batch = units[start:start + batch_size]
        with tempfile.TemporaryDirectory(prefix="q3as-sparkv-") as tmp:
            try:
                flags = batch_verify(batch, Path(tmp))
            except (TimeoutError, FileNotFoundError, OSError) as exc:
                logger.warning("gnatprove batch failed (%s); skipping %d units", exc, len(batch))
                progress.advance()
                continue
            except subprocess.SubprocessError as exc:
                logger.warning("gnatprove batch errored (%s); skipping %d units", exc, len(batch))
                progress.advance()
                continue
        for unit in batch:
            if not flags.get(unit["package"]):
                logger.info("unit %s did not prove; dropped", unit["package"])
                continue
            verified += 1
            group = f"sparkv:{zlib.crc32(unit['source'].encode('utf-8'))}"
            records.append({
                "messages": [
                    {"role": "user", "content": (
                        f"Write the Ada body for this subprogram declaration "
                        f"of `{unit['name']}`.\n\n```ada\n{unit['decl']}\n```"
                    )},
                    {"role": "assistant", "content": (
                        "```ada\n" + _body_only(unit["body"], unit["name"]) + "\n```"
                    )},
                ],
                "meta": {
                    "kind": "spark_verified_impl", "unit": unit["name"],
                    "source": unit["source"],
                    "verified": "gnatprove proved --level=1",
                    "has_contract": unit["has_contract"],
                    "group": group,
                },
            })
        progress.advance()
    progress.close()
    logger.info("kept %d of %d units (gnatprove proved)", verified, len(units))
    return records


def _body_only(package_body: str, name: str) -> str:
    """The subprogram text alone, without the package wrapper."""
    match = re.search(rf"\b(?:procedure|function)\s+{name}\b", package_body)
    if not match:
        return package_body
    end_match = re.search(rf"\bend\s+{name}\s*;", package_body[match.start():], re.DOTALL)
    if not end_match:
        return package_body
    span = package_body[match.start():match.start() + end_match.end()]
    return span.rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate gnatprove-verified completion turns from real SPARK units.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--source-dir", type=Path, action="append", default=[],
        help="Directory tree to take spec/body pairs from. Repeatable; defaults to the Ada-Algorithms SPARK2 trees.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="maximum number of units to try; 0 (default) takes every discovered unit",
    )
    parser.add_argument("--force", action="store_true", help="regenerate even when the inputs are unchanged")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    source_dirs = args.source_dir or default_sources()
    spec = stage_state.make_spec(
        name="verified_spark",
        outputs=[args.output],
        input_trees=((source_dir, (".ads", ".adb")) for source_dir in source_dirs),
        scripts=[Path(__file__).resolve(), Path(alire_env.__file__).resolve()],
        params=[("limit", args.limit)],
    )
    if spec.skip_if_fresh(force=args.force):
        return 0

    if not alire_env.has_tool("gnatprove"):
        logger.warning(
            "gnatprove is not available (run `make prove`); keeping %s as is",
            args.output if args.output.exists() else "no verified-spark output",
        )
        return 0

    units = discover_units(source_dirs, args.limit)
    records = build_turns(units)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    logger.info("wrote %d gnatprove-verified turns to %s", len(records), args.output)
    if records:
        spec.mark_fresh()
        return 0
    logger.warning("no verified units produced; not recording a fresh stamp")
    return 1


if __name__ == "__main__":
    sys.exit(main())
