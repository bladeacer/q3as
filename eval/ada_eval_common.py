"""ada_eval_common.py - Shared helpers for reading ada-eval results.

The build/test/prove tally used to be implemented three times
(eval/baseline_eval.py, eval/eval_pipeline.py, and scripts/gen_eval_report.py)
and the copies had already drifted: the report counted `proved_incorrectly`
and `subprogram_not_found` as errors while both eval modules counted them as
unproved, so the published "Prove errors" column disagreed with
outputs/comparison_report.txt for the same run.

One implementation lives here so a fix lands once. The classification rule is
deliberately conservative: an incorrect proof is not a proof, and a check the
prover could not find is neither proved nor refuted, so both are reported as
unproved rather than inflating either bucket.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("q3as_ada_eval")

EVAL_KINDS = ("build", "test", "prove")

# ada-eval packs a generated dataset as spark_<dataset>.jsonl, and the ada-eval
# dataset name is itself prefixed with spark_, so the files on disk carry a
# doubled prefix. Matching on the raw stem made `--dataset learn` silently
# select nothing.
PACKED_PREFIX = "spark_"

# Result strings that mean "not proved" rather than "the prover broke".
UNPROVED_RESULTS = ("unproved", "proved_incorrectly", "subprogram_not_found")


def empty_stats() -> dict[str, dict[str, int]]:
    """A zero-populated tally with the same shape as aggregate_eval_results."""
    return {
        "build": {"compiled": 0, "failed": 0, "total": 0},
        "test": {"passed": 0, "failed": 0, "total": 0},
        "prove": {"proved": 0, "unproved": 0, "error": 0, "total": 0},
    }


def strip_packed_prefix(name: str) -> str:
    """Drop the `spark_` prefix, so both naming forms compare equal."""
    return name.removeprefix(PACKED_PREFIX)


def dataset_of_packed_file(path: Path) -> str:
    """Recover the ada-eval dataset name from a packed generated file name."""
    return strip_packed_prefix(path.stem)


def matches_dataset_filter(path: Path, dataset_filter: str | None) -> bool:
    """True when a packed file belongs to *dataset_filter*.

    Accepts either the ada-eval dataset name (``spark_learn``) or the short
    form used in the ada-eval docs (``learn``); both are compared with the
    packing prefix removed.
    """
    if not dataset_filter:
        return True
    return strip_packed_prefix(dataset_of_packed_file(path)) == strip_packed_prefix(dataset_filter)


def has_results(stats: dict[str, Any]) -> bool:
    """True when a tally actually observed at least one result.

    The dicts are always present and always non-empty, so testing truthiness
    made "no data" indistinguishable from "measured zero".
    """
    return any(block.get("total", 0) for block in stats.values())


def rate_pct(block: dict[str, int], numerator: str) -> float | None:
    """Percentage for *numerator*, or None when the block has no samples.

    None rather than 0.0: "measured zero" and "not measured" are different
    facts, and reporting them the same way is what made an empty pipeline look
    like a failed one.
    """
    total = block.get("total", 0)
    if total <= 0:
        return None
    return block.get(numerator, 0) / total * 100


def aggregate_eval_results(
    eval_results_dir: Path, evals: list[str] | None = None
) -> dict[str, Any]:
    """Tally build/test/prove across every ada-eval result file in a tree.

    ada-eval writes ``outputs/eval_results/<model_label>/<dataset>/*.jsonl``,
    one evaluated sample per line with an ``evaluation_results`` list.
    """
    wanted = set(evals) if evals else set(EVAL_KINDS)
    stats = empty_stats()
    if not eval_results_dir.exists():
        return stats

    for result_file in sorted(eval_results_dir.rglob("*.jsonl")):
        try:
            with open(result_file, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for entry in sample.get("evaluation_results", []) or []:
                        kind = entry.get("eval")
                        if kind not in wanted or kind not in stats:
                            continue
                        block = stats[kind]
                        block["total"] += 1
                        if kind == "build":
                            block["compiled" if entry.get("compiled") else "failed"] += 1
                        elif kind == "test":
                            passed = bool(entry.get("compiled") and entry.get("passed_tests"))
                            block["passed" if passed else "failed"] += 1
                        else:
                            result = entry.get("result", "error")
                            if result == "proved":
                                block["proved"] += 1
                            elif result in UNPROVED_RESULTS:
                                block["unproved"] += 1
                            else:
                                block["error"] += 1
        except OSError as exc:
            logger.warning("Cannot read result file %s: %s", result_file, exc)
    return stats
