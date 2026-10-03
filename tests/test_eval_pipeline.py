"""Tests for eval/eval_pipeline.py: the BUILD/TEST/PROVE driver.

This module had no tests at all, and it is the only place that decides which
directory ada-eval writes per-sample results into, which model directories
are compared, and whether the run exits non-zero. Two of those decisions are
load-bearing for every published number, so they are pinned here:

- the per-dataset output path (``outputs/eval_results/<label>/<stem>``), which
  is what ``baseline_eval`` and ``gen_eval_report.py`` later read;
- the exit codes, because a run that evaluates nothing used to write zeros
  and exit 0, which reads as a measurement of 0.0%.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import eval_pipeline as ep

# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def _stats(compiled=0, total=0, passed=0, proved=0, unproved=0, error=0):
    """An ada_eval_common-shaped stats dict."""
    return {
        "build": {"compiled": compiled, "failed": total - compiled, "total": total},
        "test": {"passed": passed, "failed": total - passed, "total": total},
        "prove": {
            "proved": proved,
            "unproved": unproved,
            "error": error,
            "total": total,
        },
    }


def _evaluated_sample(name, compiled=True, passed=True, prove="unproved"):
    """One line of a packed ada-eval *evaluated* dataset."""
    return json.dumps({
        "name": name,
        "location": {"path": "src/foo.ads"},
        "evaluation_results": [
            {"eval": "build", "compiled": compiled},
            {"eval": "test", "compiled": compiled, "passed_tests": passed},
            {"eval": "prove", "result": prove, "proved_checks": {}, "unproved_checks": {}},
        ],
    })


@pytest.fixture
def generated(tmp_path, monkeypatch):
    """An outputs/generated_solutions tree with two model directories."""
    root = tmp_path / "generated_solutions"
    for label in ("fine_tuned", "base_qwen3-8b"):
        d = root / label
        d.mkdir(parents=True)
        (d / "spark_spark_learn.jsonl").write_text("{}\n", encoding="utf-8")
        (d / "spark_spark_custom.jsonl").write_text("{}\n", encoding="utf-8")
        (d / "notes.txt").write_text("ignored\n", encoding="utf-8")
    monkeypatch.setattr(ep, "GENERATED_DIR", root)
    monkeypatch.setattr(ep, "EVALS_DIR", tmp_path / "eval_results")
    return root


@pytest.fixture
def stub_ada_eval(monkeypatch):
    """Stub the two ada-eval imports run_ada_eval performs lazily.

    ``run_ada_eval`` imports inside the function so that a missing ada-eval
    only breaks the pipeline run, not the module import. Tests register
    modules in sys.modules to intercept exactly that.
    """
    calls: list[dict] = []

    types_mod = types.ModuleType("ada_eval.datasets.types")

    class Eval:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Eval) and other.value == self.value

    types_mod.Eval = Eval

    evals_mod = types.ModuleType("ada_eval.evals")

    def evaluate_directory(evals, path, output_dir, jobs):
        calls.append({
            "evals": list(evals),
            "path": Path(path),
            "output_dir": Path(output_dir),
            "jobs": jobs,
        })

    evals_mod.evaluate_directory = evaluate_directory

    packages = {
        name: types.ModuleType(name)
        for name in ("ada_eval", "ada_eval.datasets", "ada_eval.evals")
    }
    monkeypatch.setitem(sys.modules, "ada_eval", packages["ada_eval"])
    monkeypatch.setitem(sys.modules, "ada_eval.datasets", packages["ada_eval.datasets"])
    monkeypatch.setitem(sys.modules, "ada_eval.datasets.types", types_mod)
    monkeypatch.setitem(sys.modules, "ada_eval.evals", evals_mod)
    return calls


# --------------------------------------------------------------------------- #
# ensure_alire_path
# --------------------------------------------------------------------------- #


class TestEnsureAlirePath:
    def test_exports_the_alire_path(self, monkeypatch):
        # ada-eval resolves gnatprove with shutil.which and inherits
        # os.environ, so the toolchain PATH has to be in this process.
        monkeypatch.setattr(ep, "alire_env_path", lambda: "/opt/alire/bin:/usr/bin")
        monkeypatch.setenv("PATH", "/usr/bin")
        ep.ensure_alire_path()
        import os

        assert os.environ["PATH"] == "/opt/alire/bin:/usr/bin"


# --------------------------------------------------------------------------- #
# check_tools_available
# --------------------------------------------------------------------------- #


class TestCheckToolsAvailable:
    def test_all_present(self, monkeypatch):
        monkeypatch.setattr(ep, "has_tool", lambda t: True)
        ok, missing = ep.check_tools_available(["build", "test", "prove"])
        assert ok is True
        assert missing == []

    def test_missing_tool_reported_once(self, monkeypatch):
        # gprbuild is required by all three kinds; a naive loop lists it three
        # times.
        monkeypatch.setattr(ep, "has_tool", lambda t: t != "gprbuild")
        ok, missing = ep.check_tools_available(["build", "test", "prove"])
        assert ok is False
        assert missing == ["gprbuild"]

    def test_missing_tools_sorted_and_deduplicated(self, monkeypatch):
        monkeypatch.setattr(ep, "has_tool", lambda t: False)
        ok, missing = ep.check_tools_available(["prove"])
        assert ok is False
        assert missing == sorted(missing)
        assert set(missing) == {"gprbuild", "gprls", "gnatprove"}

    def test_unknown_eval_kind_needs_nothing(self, monkeypatch):
        monkeypatch.setattr(ep, "has_tool", lambda t: False)
        assert ep.check_tools_available(["nonexistent"]) == (True, [])

    def test_empty_eval_list_is_ok(self, monkeypatch):
        monkeypatch.setattr(ep, "has_tool", lambda t: False)
        assert ep.check_tools_available([]) == (True, [])


# --------------------------------------------------------------------------- #
# find_packed_datasets
# --------------------------------------------------------------------------- #


class TestFindPackedDatasets:
    def test_missing_model_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ep, "GENERATED_DIR", tmp_path / "nope")
        assert ep.find_packed_datasets("fine_tuned", None) == []

    def test_finds_only_jsonl_sorted(self, generated):
        found = ep.find_packed_datasets("fine_tuned", None)
        assert [f.name for f in found] == [
            "spark_spark_custom.jsonl",
            "spark_spark_learn.jsonl",
        ]

    def test_dataset_filter_narrows(self, generated):
        found = ep.find_packed_datasets("fine_tuned", "learn")
        assert [f.name for f in found] == ["spark_spark_learn.jsonl"]

    def test_filter_that_matches_nothing(self, generated):
        assert ep.find_packed_datasets("fine_tuned", "absent") == []


# --------------------------------------------------------------------------- #
# run_ada_eval
# --------------------------------------------------------------------------- #


class TestRunAdaEval:
    def test_skipped_when_tools_missing(self, generated, stub_ada_eval):
        # No evaluation, and no fabricated results either.
        assert ep.run_ada_eval(["build"], 1, None, tools_ok=False) == {}
        assert stub_ada_eval == []

    def test_no_generated_dir_returns_empty(self, tmp_path, monkeypatch, stub_ada_eval):
        monkeypatch.setattr(ep, "GENERATED_DIR", tmp_path / "absent")
        assert ep.run_ada_eval(["build"], 1, None) == {}

    def test_no_model_dirs_returns_empty(self, tmp_path, monkeypatch, stub_ada_eval):
        root = tmp_path / "empty"
        root.mkdir()
        (root / "stray.jsonl").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(ep, "GENERATED_DIR", root)
        assert ep.run_ada_eval(["build"], 1, None) == {}

    def test_output_dir_is_per_label_and_dataset(self, generated, stub_ada_eval):
        # This path is the contract with baseline_eval and gen_eval_report.
        ep.run_ada_eval(["build", "test"], jobs=1, dataset_filter=None)
        outputs = sorted(str(c["output_dir"]) for c in stub_ada_eval)
        assert outputs == [
            str(ep.EVALS_DIR / "base_qwen3-8b" / "spark_spark_custom"),
            str(ep.EVALS_DIR / "base_qwen3-8b" / "spark_spark_learn"),
            str(ep.EVALS_DIR / "fine_tuned" / "spark_spark_custom"),
            str(ep.EVALS_DIR / "fine_tuned" / "spark_spark_learn"),
        ]

    def test_eval_kinds_are_upper_cased_enums(self, generated, stub_ada_eval):
        ep.run_ada_eval(["build", "prove"], jobs=2, dataset_filter=None)
        assert [e.value for e in stub_ada_eval[0]["evals"]] == ["BUILD", "PROVE"]
        assert stub_ada_eval[0]["jobs"] == 2

    def test_result_key_combines_label_and_stem(self, generated, stub_ada_eval):
        results = ep.run_ada_eval(["build"], 1, None)
        assert sorted(results) == [
            "base_qwen3-8b_spark_spark_custom",
            "base_qwen3-8b_spark_spark_learn",
            "fine_tuned_spark_spark_custom",
            "fine_tuned_spark_spark_learn",
        ]
        assert all(r["status"] == "completed" for r in results.values())

    def test_dataset_filter_applied(self, generated, stub_ada_eval):
        ep.run_ada_eval(["build"], 1, dataset_filter="custom")
        assert len(stub_ada_eval) == 2
        assert all("custom" in c["path"].name for c in stub_ada_eval)

    def test_one_dataset_failure_does_not_stop_the_rest(self, generated, stub_ada_eval, monkeypatch):
        evals_mod = sys.modules["ada_eval.evals"]
        seen: list[str] = []

        def evaluate_directory(evals, path, output_dir, jobs):
            seen.append(Path(path).name)
            if "custom" in Path(path).name:
                raise RuntimeError("gprbuild exploded")

        monkeypatch.setattr(evals_mod, "evaluate_directory", evaluate_directory)
        results = ep.run_ada_eval(["build"], 1, None)
        assert len(seen) == 4, "a failing dataset must not abort the loop"
        failed = [k for k, r in results.items() if r["status"] == "failed"]
        assert len(failed) == 2
        assert "gprbuild exploded" in results[failed[0]]["error"]


# --------------------------------------------------------------------------- #
# rate and compute_stats_from_results
# --------------------------------------------------------------------------- #


class TestRate:
    def test_zero_total_is_zero_percent(self):
        assert ep.rate({"total": 0, "compiled": 0}, ("compiled",)) == 0.0

    def test_single_numerator(self):
        assert ep.rate({"total": 4, "compiled": 3}, ("compiled",)) == 75.0

    def test_several_numerators_summed(self):
        block = {"total": 4, "proved": 1, "unproved": 1}
        assert ep.rate(block, ("proved", "unproved")) == 50.0

    def test_missing_key_counts_as_zero(self):
        assert ep.rate({"total": 2}, ("compiled",)) == 0.0


class TestComputeStatsFromResults:
    def test_reads_the_evaluated_packed_files(self, tmp_path):
        d = tmp_path / "fine_tuned" / "spark_spark_learn"
        d.mkdir(parents=True)
        (d / "spark_spark_learn.jsonl").write_text(
            _evaluated_sample("a") + "\n" + _evaluated_sample("b", compiled=False, passed=False) + "\n",
            encoding="utf-8",
        )
        stats = ep.compute_stats_from_results(tmp_path / "fine_tuned")
        assert stats["build"] == {"compiled": 1, "failed": 1, "total": 2}
        assert stats["test"] == {"passed": 1, "failed": 1, "total": 2}

    def test_missing_dir_is_all_zero(self, tmp_path):
        stats = ep.compute_stats_from_results(tmp_path / "absent")
        assert stats["build"]["total"] == 0


# --------------------------------------------------------------------------- #
# generate_comparison_report
# --------------------------------------------------------------------------- #


class TestGenerateComparisonReport:
    def _write(self, tmp_path, results, ft="fine_tuned", base="base_qwen3-8b"):
        out = tmp_path / "comparison_report.txt"
        ep.generate_comparison_report(results, ft, base, out)
        return out.read_text(encoding="utf-8")

    def test_reports_both_models(self, tmp_path):
        results = {
            "base_qwen3-8b": {"stats": _stats(compiled=8, total=19, passed=5)},
            "fine_tuned": {"stats": _stats(compiled=16, total=19, passed=11)},
        }
        text = self._write(tmp_path, results)
        assert "Compilation: 8/19 passed (42.1%)" in text
        assert "Compilation: 16/19 passed (84.2%)" in text
        assert "Unit Tests:  5/19 passed (26.3%)" in text

    def test_shows_unproved_and_error_counts(self, tmp_path):
        results = {"fine_tuned": {"stats": _stats(total=19, unproved=8, error=11)}}
        text = self._write(tmp_path, results)
        assert "Unproved: 8, Errors: 11" in text

    def test_improvement_column(self, tmp_path):
        results = {
            "base_qwen3-8b": {"stats": _stats(compiled=8, total=19, passed=5)},
            "fine_tuned": {"stats": _stats(compiled=16, total=19, passed=11)},
        }
        text = self._write(tmp_path, results)
        assert "+42.1%" in text
        assert "+31.6%" in text

    def test_missing_model_says_so_instead_of_zero(self, tmp_path):
        # 0/0 must not be rendered as a 0.0% measurement.
        results = {"fine_tuned": {"stats": _stats(compiled=1, total=1)}}
        text = self._write(tmp_path, results)
        assert "(no generated solutions or eval results found)" in text

    def test_comparison_block_skipped_without_both_models(self, tmp_path):
        results = {"fine_tuned": {"stats": _stats(compiled=1, total=1)}}
        text = self._write(tmp_path, results)
        assert "Improvement" not in text
        assert "No ada-eval results for the base model" in text

    def test_comparison_block_asks_for_a_run_when_nothing_scored(self, tmp_path):
        text = self._write(tmp_path, {"fine_tuned": {"stats": _stats()}})
        assert "run `make generate` then this pipeline" in text

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "comparison_report.txt"
        ep.generate_comparison_report({}, "fine_tuned", "base", out)
        assert out.exists()

    def test_result_key_is_the_bare_label_not_the_dataset_key(self, tmp_path):
        # run_ada_eval keys outcomes "<label>_<stem>"; the report is fed the
        # "<label>" dict from main, so a dataset key must not be picked up.
        results = {
            "fine_tuned": {"stats": _stats(compiled=16, total=19)},
            "fine_tuned_spark_spark_learn": {"stats": _stats(compiled=0, total=0)},
        }
        text = self._write(tmp_path, results)
        assert "Compilation: 16/19" in text
        assert "Compilation: 0/0" not in text


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


class TestMain:
    def _argv(self, monkeypatch, *args):
        monkeypatch.setattr(sys, "argv", ["eval_pipeline.py", *args])

    def test_exits_when_nothing_was_generated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ep, "GENERATED_DIR", tmp_path / "absent")
        monkeypatch.setattr(ep, "EVALS_DIR", tmp_path / "eval_results")
        self._argv(monkeypatch)
        monkeypatch.setattr(ep, "ensure_alire_path", lambda: None)
        with pytest.raises(SystemExit) as excinfo:
            ep.main()
        assert "make generate" in str(excinfo.value)

    def test_writes_json_and_report(self, generated, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ep, "ensure_alire_path", lambda: None)
        monkeypatch.setattr(ep, "has_tool", lambda t: True)
        monkeypatch.setattr(ep, "run_ada_eval", lambda **kw: {})
        self._argv(monkeypatch)
        ep.main()
        data = json.loads((tmp_path / "outputs" / "eval_pipeline_results.json").read_text())
        assert sorted(data) == ["base_qwen3-8b", "fine_tuned"]
        assert (tmp_path / "outputs" / "comparison_report.txt").exists()

    def test_exits_nonzero_when_a_dataset_failed(self, generated, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ep, "ensure_alire_path", lambda: None)
        monkeypatch.setattr(ep, "has_tool", lambda t: True)
        monkeypatch.setattr(
            ep, "run_ada_eval",
            lambda **kw: {"fine_tuned_spark_spark_learn": {"status": "failed"}},
        )
        self._argv(monkeypatch)
        with pytest.raises(SystemExit) as excinfo:
            ep.main()
        assert excinfo.value.code == 1

    def test_succeeds_when_every_dataset_completed(self, generated, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ep, "ensure_alire_path", lambda: None)
        monkeypatch.setattr(ep, "has_tool", lambda t: True)
        monkeypatch.setattr(
            ep, "run_ada_eval", lambda **kw: {"fine_tuned_spark_spark_learn": {"status": "completed"}}
        )
        self._argv(monkeypatch)
        ep.main()

    def test_dataset_filter_is_passed_through(self, generated, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ep, "ensure_alire_path", lambda: None)
        monkeypatch.setattr(ep, "has_tool", lambda t: True)
        seen: dict = {}

        def fake_run(**kwargs):
            seen.update(kwargs)
            return {}

        monkeypatch.setattr(ep, "run_ada_eval", fake_run)
        self._argv(monkeypatch, "--dataset", "learn", "--jobs", "4", "--evals", "prove")
        ep.main()
        assert seen["dataset_filter"] == "learn"
        assert seen["jobs"] == 4
        assert seen["evals"] == ["prove"]