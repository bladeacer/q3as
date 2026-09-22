#!/usr/bin/env python3
"""collect_cap_results.py - summarize the cap-tuning experiment.

Reads <out>/run_cap*/training_summary.json and writes one row per run to
<out>/results.csv: final train loss, best eval loss and its step, test
loss/perplexity, and wall seconds. trends.csv carries the same data in
long form for quick plotting.

The output directory defaults to outputs/capexp and can be overridden with
the CAP_OUT environment variable (run_cap_experiment.sh sets it when a
round writes to its own directory).
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("q3as_collect_cap_results")

OUT = Path(os.environ.get("CAP_OUT", "outputs/capexp"))


def best_of(history: list[tuple[int, float]]) -> tuple[int | None, float | None]:
    if not history:
        return None, None
    step, loss = min(history, key=lambda pair: pair[1])
    return step, loss


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []
    for run_dir in sorted(OUT.glob("run_cap*")):
        summary_path = run_dir / "training_summary.json"
        if not summary_path.exists():
            logger.warning("No training_summary.json in %s", run_dir)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cap = run_dir.name.removeprefix("run_cap")
        history = [
            (int(point["step"]), float(point["eval_loss"]))
            for point in summary.get("eval_loss_history", [])
            if "step" in point and "eval_loss" in point
        ]
        best_step, best_loss = best_of(history)
        test = summary.get("test_metrics") or {}
        hyper = summary.get("hyperparameters") or {}
        rows.append({
            "cap": cap,
            "steps": hyper.get("max_steps"),
            "final_train_loss": summary.get("train_loss_final"),
            "best_eval_loss": best_loss,
            "best_step": best_step,
            "test_loss": test.get("test_eval_loss"),
            "test_ppl": test.get("test_perplexity"),
            "early_stopped": summary.get("early_stopped"),
            "seconds": summary.get("train_seconds"),
        })
        for step, loss in history:
            trend_rows.append({"cap": cap, "step": step, "eval_loss": loss})
        # keep the raw histories available for later analysis
        trend_path = OUT / "trends.csv"
        with open(trend_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cap", "step", "eval_loss"])
            writer.writeheader()
            writer.writerows(trend_rows)
    if not rows:
        logger.error("No runs found under %s", OUT)
        return 1
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    width = 8
    header = (f"{'cap':>5} {'steps':>6} {'train':>8} {'best_eval':>10} "
              f"{'@step':>6} {'test':>8} {'ppl':>8} {'early':>6} {'sec':>6}")
    print(header)
    for row in rows:
        print(
            f"{row['cap']!s:>5} {row['steps']!s:>6} "
            f"{row['final_train_loss']!s:>{width}} "
            f"{row['best_eval_loss']!s:>10} "
            f"{row['best_step']!s:>6} "
            f"{row['test_loss']!s:>8} {row['test_ppl']!s:>8} "
            f"{row['early_stopped']!s:>6} {row['seconds']!s:>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
