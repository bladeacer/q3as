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

It also owns the changed/unchanged split
(``split_tally_by_changed_files``): most ada-eval base trees already compile,
so a reply that echoes its input scores on the base project. One total number
for both groups is what let a run read as 15 builds when 12 of them were
inherited.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
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
    for _name, entries in iter_result_samples(eval_results_dir):
        for entry in entries:
            kind = entry.get("eval")
            if kind not in wanted or kind not in stats:
                continue
            tally_result(stats, kind, entry)
    return stats


def iter_result_samples(eval_results_dir: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield ``(sample name, evaluation_results)`` for every evaluated sample.

    The per-file walk and the per-line parse live here so the whole-run tally
    and the changed/unchanged split cannot disagree about which files count or
    how a malformed line is treated.
    """
    if not eval_results_dir.exists():
        return
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
                    if not isinstance(sample, dict):
                        continue
                    entries = sample.get("evaluation_results")
                    if not isinstance(entries, list):
                        continue
                    # A non-dict entry has no "eval" key to classify by, so it
                    # is dropped here rather than in each caller.
                    yield str(sample.get("name", "")), [
                        e for e in entries if isinstance(e, dict)
                    ]
        except OSError as exc:
            logger.warning("Cannot read result file %s: %s", result_file, exc)


def tally_result(stats: dict[str, Any], kind: str, entry: dict[str, Any]) -> None:
    """Add one ada-eval result to the matching bucket of *stats*.

    Single classification for both tallies: a test passes only when the
    project also compiled, an incorrect proof is unproved rather than an
    error, and anything else the prover reports is an error.
    """
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


def load_changed_files(meta_path: Path) -> dict[str, bool] | None:
    """Read ``generation_meta.json`` as {sample name: did the reply change a file}.

    None when the sidecar is absent or unreadable, which is the whole
    difference between "measured: no-op" and "not recorded": older runs have
    no sidecar at all (it was overwritten), and a zero there would read as a
    measured zero. An entry without a usable ``changed_files`` is left out for
    the same reason - it is unknown, not unchanged.
    """
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read %s: %s", meta_path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    changed: dict[str, bool] = {}
    for name, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        count = entry.get("changed_files")
        if isinstance(count, bool) or not isinstance(count, int):
            continue
        changed[str(name)] = count > 0
    return changed


def split_tally_by_changed_files(
    eval_results_dir: Path, meta_path: Path, evals: list[str] | None = None
) -> dict[str, Any]:
    """Tally build/test/prove twice: samples the model changed, and samples it echoed.

    An ada-eval base tree usually already carries a body for the target
    subprogram and already compiles, so a reply that echoes what it was shown
    collects BUILD and TEST for the *base* project. Reporting one number for
    both groups is what let a v0.8.0 run read as 15 builds when 12 of them were
    inherited, so every report now states the two groups apart.

    Returns ``{}`` when the sidecar is absent or when no evaluated sample could
    be joined to it, so the caller renders "not recorded" instead of a zero
    that looks measured.
    """
    changed = load_changed_files(meta_path)
    if not changed:
        return {}

    wanted = set(evals) if evals else set(EVAL_KINDS)
    buckets: dict[str, dict[str, dict[str, int]]] = {
        "edited": empty_stats(),
        "unchanged": empty_stats(),
    }
    joined = 0
    unmatched: list[str] = []
    for name, entries in iter_result_samples(eval_results_dir):
        edited = changed.get(name)
        if edited is None:
            unmatched.append(name or "<unnamed>")
            continue
        joined += 1
        stats = buckets["edited" if edited else "unchanged"]
        for entry in entries:
            kind = entry.get("eval")
            if kind not in wanted or kind not in stats:
                continue
            tally_result(stats, kind, entry)

    if unmatched:
        logger.warning(
            "%d evaluated sample(s) are not in %s and are excluded from the "
            "changed/unchanged split: %s",
            len(unmatched), meta_path, ", ".join(sorted(unmatched)[:5]),
        )
    if not joined:
        return {}
    return buckets
