"""Tests for gen_eval_report training-metrics collection and trend diagnosis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gen_eval_report as ger


def _summary(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "training_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_payload() -> dict:
    return {
        "base_model": "Qwen/Qwen3-8B",
        "train_loss_final": 1.10,
        "train_loss_history": [
            {"step": 10, "loss": 2.0},
            {"step": 20, "loss": 1.6},
            {"step": 30, "loss": 1.10},
        ],
        "eval_loss_history": [
            {"step": 10, "eval_loss": 1.9},
            {"step": 20, "eval_loss": 1.4},
            {"step": 30, "eval_loss": 1.3},
        ],
        "test_metrics": {"test_eval_loss": 1.35, "test_perplexity": 3.86},
    }


def test_collect_training_metrics_happy_path(tmp_path: Path) -> None:
    metrics = ger.collect_training_metrics(_summary(tmp_path, _base_payload()))
    assert metrics["train_loss"]["first"] == 2.0
    assert metrics["train_loss"]["last"] == 1.10
    assert metrics["val_loss"]["best"] == 1.3
    assert metrics["test_loss"] == 1.35
    assert metrics["perplexity"]["test"] == round(2.718281828 ** 1.35, 3)
    assert metrics["trend"]["verdict"] in {"healthy", "plateau"}


def test_trend_diagnosis_healthy() -> None:
    eval_history = [
        {"step": 10, "eval_loss": 2.0},
        {"step": 20, "eval_loss": 1.5},
        {"step": 30, "eval_loss": 1.2},
        {"step": 40, "eval_loss": 1.1},
    ]
    verdict, findings = ger._trend_diagnosis(eval_history, [{"step": 40, "loss": 1.0}])
    assert verdict == "healthy"
    assert any("improvement" in f for f in findings)


def test_trend_diagnosis_unstable_detects_spike() -> None:
    eval_history = [
        {"step": 10, "eval_loss": 2.0},
        {"step": 20, "eval_loss": 1.5},
        {"step": 30, "eval_loss": 2.6},  # +73% jump
        {"step": 40, "eval_loss": 1.2},
    ]
    verdict, findings = ger._trend_diagnosis(eval_history, [])
    assert verdict == "unstable"
    assert any("jumped" in f for f in findings)


def test_trend_diagnosis_overfit() -> None:
    eval_history = [
        {"step": 10, "eval_loss": 1.8},
        {"step": 20, "eval_loss": 1.2},  # best, early
        {"step": 30, "eval_loss": 1.35},  # +12.5% rise
        {"step": 40, "eval_loss": 1.4},
    ]
    train_history = [
        {"step": 20, "loss": 1.4},
        {"step": 30, "loss": 1.2},
        {"step": 40, "loss": 1.0},
    ]
    verdict, findings = ger._trend_diagnosis(eval_history, train_history)
    assert verdict == "overfit"
    assert any("kept falling" in f for f in findings)


def test_trend_diagnosis_plateau_when_best_early_and_train_stalled() -> None:
    eval_history = [
        {"step": 10, "eval_loss": 1.8},
        {"step": 20, "eval_loss": 1.2},
        {"step": 30, "eval_loss": 1.24},
    ]
    train_history = [{"step": 30, "loss": 1.3}]
    verdict, _ = ger._trend_diagnosis(eval_history, train_history)
    assert verdict == "plateau"


def test_trend_diagnosis_sparse() -> None:
    verdict, findings = ger._trend_diagnosis([{"step": 10, "eval_loss": 1.5}], [])
    assert verdict == "sparse"
    assert findings


def test_collect_training_metrics_missing_file(tmp_path: Path) -> None:
    assert ger.collect_training_metrics(tmp_path / "nope.json") == {}


def test_collect_training_metrics_malformed(tmp_path: Path) -> None:
    path = tmp_path / "training_summary.json"
    path.write_text("{not json", encoding="utf-8")
    assert ger.collect_training_metrics(path) == {}


def test_render_training_markdown_missing() -> None:
    lines = ger.render_training_markdown({})
    assert any("No training metrics" in ln for ln in lines)


def test_render_training_markdown_renders_table_and_verdict(tmp_path: Path) -> None:
    metrics = ger.collect_training_metrics(_summary(tmp_path, _base_payload()))
    lines = ger.render_training_markdown(metrics)
    text = "\n".join(lines)
    assert "| Loss |" in text
    assert "Trend:" in text
    assert "| Step | Val loss |" in text
