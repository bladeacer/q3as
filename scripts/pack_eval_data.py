"""Rebuild ada-eval's compacted benchmark JSONL from its expanded samples.

``data/base/compacted/*.jsonl`` is what ``eval/baseline_eval.py`` scores
against, and it is derived from ``data/base/expanded`` rather than version
controlled. A fresh cache (``make fetch-sources``, ``make update-sources``)
therefore has no compacted data at all, and ``make eval`` exits 1.

Rebuilding it needs one non-obvious guard. ada-eval's packer is git-aware:
``get_contents_git_aware`` shells out to ``git rev-parse
--is-inside-work-tree`` and, when the answer is yes, reads a sample's files
with ``git ls-files``. This repository *is* a worktree and ``data/raw_repos/``
is gitignored, so the check succeeds while ``git ls-files`` returns nothing.
The pack then succeeds, writes one JSONL line per sample, and every line has
an empty ``canonical_solution`` and no ``sources`` -- a hollow benchmark.
``make eval`` scores against it happily and reports BLEU 0.0 with every
standard ``Unknown``, which looks like a model result and is not one.

``GIT_CEILING_DIRECTORIES`` pointing at the ada-eval checkout stops git's
upward repository search there, so the packer takes its plain-directory path
and reads the real files. ``force=True`` is then required as well: the same
git-awareness makes ada-eval's own "uncommitted changes" guard fire, and it
exits rather than packing.

The pack is verified after the fact: any record without solution files is a
failure, so a hollow pack can never reach a report.

Usage:
    uv run python scripts/pack_eval_data.py [--check]

``--check`` only verifies the existing compacted data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "processing_scripts"))

import source_paths

logger = logging.getLogger("pack_eval_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing compacted data instead of rebuilding it",
    )
    return parser.parse_args(argv)


def compacted_dir(ada_eval_dir: Path) -> Path:
    return ada_eval_dir / "data" / "base" / "compacted"


def expanded_dir(ada_eval_dir: Path) -> Path:
    return ada_eval_dir / "data" / "base" / "expanded"


def pack(ada_eval_dir: Path) -> int:
    """Rebuild compacted from expanded. Returns a process exit code."""
    expanded = expanded_dir(ada_eval_dir)
    if not expanded.is_dir():
        logger.error("No expanded benchmark at %s; run `make fetch-sources`.", expanded)
        return 1

    # Must be set before pack_datasets runs: its git calls inherit os.environ.
    os.environ["GIT_CEILING_DIRECTORIES"] = str(ada_eval_dir)

    from ada_eval.datasets.pack_unpack import pack_datasets

    logger.info("Packing %s -> %s", expanded, compacted_dir(ada_eval_dir))
    pack_datasets(expanded, compacted_dir(ada_eval_dir), force=True)
    return 0


def load_packed(ada_eval_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for jsonl_file in sorted(compacted_dir(ada_eval_dir).glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def verify(ada_eval_dir: Path) -> int:
    """Fail unless every packed record carries solution files."""
    if not compacted_dir(ada_eval_dir).is_dir():
        logger.error(
            "No compacted benchmark at %s; run `make eval-data`.", compacted_dir(ada_eval_dir)
        )
        return 1
    records = load_packed(ada_eval_dir)
    if not records:
        logger.error("Compacted benchmark at %s is empty.", compacted_dir(ada_eval_dir))
        return 1
    hollow = [
        r.get("name", "?")
        for r in records
        if not (r.get("canonical_solution") or {})
    ]
    if hollow:
        logger.error(
            "%d of %d packed samples have no canonical solution (%s%s). The pack "
            "was hollow; delete %s and re-run without GIT_CEILING_DIRECTORIES "
            "set for the checkout.",
            len(hollow), len(records), ", ".join(hollow[:5]),
            ", ..." if len(hollow) > 5 else "", compacted_dir(ada_eval_dir),
        )
        return 1
    logger.info("Compacted benchmark verified: %d samples with solutions.", len(records))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ada_eval_dir = source_paths.resolve("ada-eval")
    if ada_eval_dir is None:
        logger.error("ada-eval is not in the source cache; run `make fetch-sources`.")
        return 1
    if not args.check and pack(ada_eval_dir) != 0:
        return 1
    return verify(ada_eval_dir)


if __name__ == "__main__":
    raise SystemExit(main())