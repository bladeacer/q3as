#!/usr/bin/env python3
"""make_probe_splits.py - carve a fixed probe val/test for cap experiments.

Carves the first N records of the current dataset_val.jsonl and
dataset_test.jsonl (in file order; the split is group-aware and seeded, so
file order is deterministic for a given build) into probe val/test JSONL
for scripts/run_cap_experiment.sh.

The probe must be held out from EVERY experimental arm's training data.
Split slices depend on the post-dedup group list, so the same source
content can move between train and val across builds with different caps;
a probe carved from one build's val split is not automatically held out
of another build's train split. Every --exclude-train file therefore
supplies one arm's training records, and probe candidates whose chat
messages appear in any of them are skipped. Message content is the
matching key (training text is built from messages only, so meta-only
differences do not matter). Records excluded this way do not shrink the
arms' own splits; they only leave the probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger("q3as_make_probe_splits")


def messages_key(line: str) -> str:
    """Stable content key for one JSONL record: its chat messages."""
    record = json.loads(line)
    return hashlib.sha256(
        json.dumps(record.get("messages"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_excluded(paths: list[Path]) -> set[str]:
    """Hash keys of every record in the given training files."""
    excluded: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    excluded.add(messages_key(line))
        logger.info("Loaded %d exclusion keys from %s", len(excluded), path)
    return excluded


def carve(src: Path, dst: Path, n: int, excluded: set[str]) -> tuple[int, int]:
    """Copy the first *n* non-excluded records of *src* to *dst*.

    Returns (written, skipped).
    """
    written = skipped = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as f_src, open(dst, "w", encoding="utf-8") as f_dst:
        for line in f_src:
            if written >= n:
                break
            if not line.strip():
                continue
            if messages_key(line) in excluded:
                skipped += 1
                continue
            f_dst.write(line)
            written += 1
    if written < n:
        raise SystemExit(f"{src}: only {written} non-excluded records, need {n}")
    return written, skipped


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-src", type=Path, default=Path("data/processed/dataset_val.jsonl"),
        help="Validation split to carve the probe val from.",
    )
    parser.add_argument(
        "--test-src", type=Path, default=Path("data/processed/dataset_test.jsonl"),
        help="Test split to carve the probe test from.",
    )
    parser.add_argument(
        "--exclude-train", type=Path, action="append", default=[],
        help="Arm training file whose records must not appear in the probe (repeatable).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--n-val", type=int, default=200,
        help="Probe val size; capped by the non-excluded records available.",
    )
    parser.add_argument(
        "--n-test", type=int, default=200,
        help="Probe test size; capped by the non-excluded records available.",
    )
    args = parser.parse_args()

    excluded = load_excluded(args.exclude_train)
    wrote_val, skipped_val = carve(args.val_src, args.out_dir / "probe_val.jsonl", args.n_val, excluded)
    wrote_test, skipped_test = carve(
        args.test_src, args.out_dir / "probe_test.jsonl", args.n_test, excluded
    )
    logger.info(
        "Wrote %d probe val (skipped %d) + %d probe test (skipped %d) to %s",
        wrote_val, skipped_val, wrote_test, skipped_test, args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
