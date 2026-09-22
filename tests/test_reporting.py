"""Tests for scripts/bump_version.py and scripts/gen_eval_report.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bump_version as bv
import gen_eval_report as ger


@pytest.fixture()
def manifest_dir(tmp_path: Path) -> Path:
    """Two alire manifests with a version line each."""
    (tmp_path / "alire.toml").write_text('name = "q3as"\nversion = "0.1.0"\n', encoding="utf-8")
    (tmp_path / "alire-dev.toml").write_text(
        'name = "q3as"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    return tmp_path


class TestBumpVersion:
    def test_read_version(self, manifest_dir: Path):
        assert bv.read_version(manifest_dir / "alire.toml") == "0.1.0"

    def test_set_updates_both_manifests(self, manifest_dir: Path, monkeypatch):
        monkeypatch.setattr(bv, "MANIFESTS", (manifest_dir / "alire.toml", manifest_dir / "alire-dev.toml"))
        old, results = bv.set_version("0.2.0")
        assert old == "0.1.0"
        assert all(changed for _, changed in results)
        assert bv.read_version(manifest_dir / "alire.toml") == "0.2.0"
        assert bv.read_version(manifest_dir / "alire-dev.toml") == "0.2.0"

    def test_bump_patch(self, manifest_dir: Path, monkeypatch):
        monkeypatch.setattr(bv, "MANIFESTS", (manifest_dir / "alire.toml", manifest_dir / "alire-dev.toml"))
        old, new = bv.bump_version("patch")
        assert (old, new) == ("0.1.0", "0.1.1")

    def test_bump_major_resets(self, manifest_dir: Path, monkeypatch):
        monkeypatch.setattr(bv, "MANIFESTS", (manifest_dir / "alire.toml", manifest_dir / "alire-dev.toml"))
        _old, new = bv.bump_version("major")
        assert new == "1.0.0"

    def test_invalid_version_rejected(self, manifest_dir: Path, monkeypatch):
        monkeypatch.setattr(bv, "MANIFESTS", (manifest_dir / "alire.toml", manifest_dir / "alire-dev.toml"))
        with pytest.raises(ValueError):
            bv.set_version("1.2")

    def test_missing_manifest_raises_filenotfound(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            bv.read_version(tmp_path / "nope.toml")

    def test_extra_version_file_synced(self, manifest_dir: Path, monkeypatch):
        # pyproject-style extra file: carries the version, synced best-effort.
        (manifest_dir / "pyproject.toml").write_text(
            'name = "q3as"\nversion = "0.1.0"\n', encoding="utf-8"
        )
        monkeypatch.setattr(
            bv, "MANIFESTS", (manifest_dir / "alire.toml", manifest_dir / "alire-dev.toml")
        )
        monkeypatch.setattr(bv, "EXTRA_VERSION_FILES", (manifest_dir / "pyproject.toml",))
        _old, results = bv.set_version("0.3.0")
        assert (manifest_dir / "pyproject.toml").read_text(encoding="utf-8").count(
            'version = "0.3.0"'
        ) == 1
        assert any(path.name == "pyproject.toml" and changed for path, changed in results)

    def test_extra_version_file_absent_is_skipped(self, manifest_dir: Path, monkeypatch):
        # No pyproject.toml in the fixture dir: the bump must still succeed.
        monkeypatch.setattr(
            bv, "MANIFESTS", (manifest_dir / "alire.toml", manifest_dir / "alire-dev.toml")
        )
        monkeypatch.setattr(bv, "EXTRA_VERSION_FILES", (manifest_dir / "pyproject.toml",))
        _old, new = bv.bump_version("patch")
        assert new == "0.1.1"
        assert bv.read_version(manifest_dir / "alire.toml") == "0.1.1"


def _sample_result_line(compiled: bool, passed: bool, prove: str) -> str:
    return json.dumps({
        "evaluation_results": [
            {"eval": "build", "compiled": compiled},
            {"eval": "test", "passed_tests": passed},
            {"eval": "prove", "result": prove,
             "unproved_checks": {"VC_OVERFLOW_CHECK": 1} if prove == "unproved" else {},
             "proved_checks": {"UNINITIALIZED": 1} if prove == "unproved" else {}},
        ]
    })


class TestCollectAdaEvalMetrics:
    def test_aggregates_models_and_datasets(self, tmp_path: Path):
        base = tmp_path / "base_qwen3-8b" / "spark_ds"
        ft = tmp_path / "fine_tuned" / "spark_ds"
        base.mkdir(parents=True)
        ft.mkdir(parents=True)
        (base / "r.jsonl").write_text(_sample_result_line(False, False, "error") + "\n", encoding="utf-8")
        (ft / "r.jsonl").write_text(_sample_result_line(True, True, "unproved") + "\n", encoding="utf-8")
        metrics = ger.collect_ada_eval_metrics(tmp_path)
        assert metrics["base_qwen3-8b"]["samples"] == 1
        assert metrics["base_qwen3-8b"]["build"] == 0
        assert metrics["base_qwen3-8b"]["prove_errors"] == 1
        assert metrics["fine_tuned"]["build"] == 1
        assert metrics["fine_tuned"]["build_pct"] == 100.0
        assert metrics["fine_tuned"]["test"] == 1
        assert metrics["fine_tuned"]["unproved_checks"] == {"VC_OVERFLOW_CHECK": 1}
        assert metrics["fine_tuned"]["datasets"]["spark_ds"]["samples"] == 1

    def test_empty_dir_yields_empty(self, tmp_path: Path):
        assert ger.collect_ada_eval_metrics(tmp_path) == {}


class TestRender:
    def _data(self) -> dict:
        return {
            "version": "9.9.9",
            "generated_at": "2026-01-01 00:00 UTC",
            "ada_eval": {
                "base_qwen3-8b": {"samples": 2, "build": 1, "build_pct": 50.0, "test": 0, "test_pct": 0.0,
                                  "proved": 0, "unproved": 1, "prove_errors": 1},
                "fine_tuned": {"samples": 2, "build": 2, "build_pct": 100.0, "test": 1, "test_pct": 50.0,
                               "proved": 0, "unproved": 2, "prove_errors": 0,
                               "unproved_checks": {"VC_OVERFLOW_CHECK": 2},
                               "datasets": {"spark_ds": {"samples": 2, "build": 2, "test": 1}}},
            },
        }

    def test_markdown_has_headline_and_backlink(self):
        text = ger.render_markdown("9.9.9", self._data())
        assert "# Results v9.9.9" in text
        assert "| Build | 1/2 (50.0%) | 2/2 (100.0%) | 50.0 pts |" in text
        assert "[← Back to results index](README.md)" in text
        assert "result-data-v9.9.9.json" in text

    def test_markdown_handles_missing_results(self):
        text = ger.render_markdown("1.0.0", {"version": "1.0.0", "generated_at": "x"})
        assert "No ada-eval JSONL results found" in text

    def test_index_comparison_table_last_three(self):
        entries = [
            {**self._data(), "version": f"0.{i}.0"} for i in (3, 2, 1)
        ]
        entries.append({"version": "0.0.9", "generated_at": "x", "ada_eval": {}})
        text = ger.render_index(entries)
        assert "## Last 3 versions compared" in text
        assert "| v0.3.0 |" in text and "| v0.1.0 |" in text
        assert "v0.0.9" not in text.split("Last 3")[1].split("\n\n")[0]  # outside the table

    def test_index_links_every_version(self):
        entries = [{**self._data(), "version": "0.1.0"}, {"version": "0.2.0", "generated_at": "x", "ada_eval": {}}]
        text = ger.render_index(entries)
        assert "[result-v0.1.0.md](result-v0.1.0.md)" in text
        assert "[result-v0.2.0.md](result-v0.2.0.md)" in text
