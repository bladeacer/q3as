"""Tests for the Makefile's pipeline wiring.

`make all` runs the evaluation steps in series, and one ordering bug cost a
whole published run: `baseline_eval` prints the BUILD/TEST/PROVE tallies but
reads them from `outputs/eval_results/`, which only `eval_pipeline.py`
writes. With `eval` first, the summary showed the *previous* run's numbers,
and on a clean tree `print_stats_block` printed nothing at all, so a report
with no build/test line read like "nothing failed".

Nothing else in the test suite parses the Makefile, so these invariants are
pinned here rather than left to whoever reorders the target list next.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).resolve().parents[1] / "Makefile"


@pytest.fixture(scope="module")
def makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _rule(makefile: str, target: str) -> str:
    """The prerequisite list and recipe of one target, or '' when absent."""
    lines = makefile.splitlines()
    for i, line in enumerate(lines):
        # A target definition starts at column 0; a recipe line starts with a tab.
        if not line.startswith(f"{target}:"):
            continue
        block = [line]
        for follow in lines[i + 1:]:
            if follow.startswith("\t") or (follow and not follow[0].isspace()):
                block.append(follow)
                continue
            break
        return "\n".join(block)
    return ""


def _recipe(makefile: str, target: str) -> str:
    """Only the tab-indented command lines of a target.

    The `##` description on the target line itself names the tools too
    ("Run ruff and mypy ..."), so searching the whole rule for a tool name
    matches the comment before the command that runs it.
    """
    return "\n".join(
        line for line in _rule(makefile, target).splitlines() if line.startswith("\t")
    )


# --------------------------------------------------------------------------- #
# The evaluation pipeline order
# --------------------------------------------------------------------------- #


class TestEvaluationOrder:
    def test_all_runs_eval_pipeline_before_eval(self, makefile):
        # The regression this file exists for: `eval` reads tallies that
        # `eval-pipeline` writes, so the other order reports stale numbers.
        rule = _rule(makefile, "all")
        assert rule, "the all target must exist"
        pipeline_at = rule.index("eval-pipeline")
        eval_at = re.search(r"(?<![\w-])eval(?![\w-])", rule).start()
        assert pipeline_at < eval_at, "eval-pipeline must precede eval in `all`"

    def test_all_ends_with_the_report(self, makefile):
        rule = _rule(makefile, "all")
        assert rule.index("eval-report") > rule.index("eval-pipeline")

    def test_generation_precedes_evaluation(self, makefile):
        rule = _rule(makefile, "all")
        assert rule.index("generate") < rule.index("eval-pipeline")

    def test_eval_pipeline_does_not_depend_on_eval(self, makefile):
        # The reverse dependency would deadlock the ordering guarantee.
        prerequisites = _rule(makefile, "eval-pipeline").splitlines()[0]
        assert "eval:" not in prerequisites


# --------------------------------------------------------------------------- #
# The benchmark data both eval steps need
# --------------------------------------------------------------------------- #


class TestEvalDataTarget:
    def test_eval_depends_on_eval_data(self, makefile):
        # A fresh cache has no compacted JSONL; `make eval` exits 1 without it.
        assert _rule(makefile, "eval").splitlines()[0].startswith("eval: eval-data")

    def test_eval_data_runs_the_packer(self, makefile):
        assert "scripts/pack_eval_data.py" in _rule(makefile, "eval-data")

    def test_eval_data_is_documented_in_help(self, makefile):
        assert "make eval-data" in makefile

    def test_pack_script_is_ruff_checked(self, makefile):
        ruff = _recipe(makefile, "lint").splitlines()[0]
        assert "ruff check" in ruff
        assert "scripts/" in ruff

    def test_pack_script_is_type_checked(self, makefile):
        mypy_line = [ln for ln in _recipe(makefile, "lint").splitlines() if "mypy" in ln]
        assert mypy_line, "the lint target must run mypy"
        assert "scripts/pack_eval_data.py" in mypy_line[0]


# --------------------------------------------------------------------------- #
# Type checking covers the evaluation modules
# --------------------------------------------------------------------------- #


class TestLintCoverage:
    @pytest.mark.parametrize(
        "module", ["eval/generate.py", "eval/eval_pipeline.py", "eval/baseline_eval.py"]
    )
    def test_module_is_type_checked(self, makefile, module):
        # These three decide what is scored; a type error in them must not
        # wait for the next lint run to be noticed.
        mypy_line = next(
            ln for ln in _recipe(makefile, "lint").splitlines() if "mypy" in ln
        )
        assert module in mypy_line

    def test_mypy_path_covers_the_scripts_dir(self, makefile):
        # scripts/ holds modules imported by the eval entry points.
        mypy_line = next(
            ln for ln in _recipe(makefile, "lint").splitlines() if "mypy" in ln
        )
        assert "MYPYPATH=data/processing_scripts:scripts" in mypy_line


# --------------------------------------------------------------------------- #
# The dataset stage chain
# --------------------------------------------------------------------------- #


class TestDatasetStages:
    def test_build_dataset_waits_for_its_parsers(self, makefile):
        prerequisites = _rule(makefile, "build-dataset").splitlines()[0]
        for stage in ("parse-data", "gen-contracts", "gen-verified-spark"):
            assert stage in prerequisites, f"{stage} must feed build-dataset"

    def test_update_sources_rebuilds_the_dataset(self, makefile):
        # Moving a source must not leave a dataset built from the old tree.
        assert "build-dataset" in _rule(makefile, "update-sources")


# --------------------------------------------------------------------------- #
# Help text
# --------------------------------------------------------------------------- #


class TestHelp:
    def test_every_documented_target_is_listed(self, makefile):
        # Targets carry their one-line description in a trailing `##` comment,
        # and `make help` lists them by hand; a target added without a help
        # line is invisible.
        documented = {
            m.group(1)
            for m in re.finditer(r"^([a-z][a-z0-9-]*):[^\n]*##", makefile, re.MULTILINE)
        } - {"help"}  # the default goal; it is what prints the list
        listed = set(re.findall(r"make ([a-z][a-z0-9-]*)", makefile))
        assert documented <= listed, f"missing from help: {sorted(documented - listed)}"