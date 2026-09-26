"""Tests for stage_state.py and progress.py.

The staleness check is what makes `make all` cheap on an unchanged tree, and
a wrong "fresh" verdict is the dangerous direction: the pipeline would train on
output that no longer matches its sources. These tests pin the safe direction
(everything that could have changed forces a rebuild) and the one intended
shortcut (identical inputs and scripts skip).
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "processing_scripts"))

import progress
import stage_state


@pytest.fixture(autouse=True)
def stamp_dir(tmp_path, monkeypatch):
    """Redirect stamps into tmp_path so tests never touch data/processed."""
    target = tmp_path / "stages"
    monkeypatch.setenv("Q3AS_STAGE_DIR", str(target))
    monkeypatch.setattr(stage_state, "STAMP_DIR", target)
    return target


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path):
    """A tiny input tree with one Ada file and one unrelated file."""
    root = tmp_path / "src"
    _write(root / "unit.ads", "package P is\n   procedure Go;\nend P;\n")
    _write(root / "README.md", "# docs\n")
    return root


def _spec(tmp_path, tree, **overrides):
    kwargs = {
        "name": "unit-test",
        "outputs": [tmp_path / "out" / "data.jsonl"],
        "input_trees": [(tree, (".ads", ".adb"))],
        "params": [("cap", 10)],
    }
    kwargs.update(overrides)
    return stage_state.make_spec(**kwargs)


def _produce(spec) -> None:
    """Simulate a successful stage run: write outputs, record the stamp."""
    for output in spec.outputs:
        _write(output, "{}\n")
    spec.mark_fresh()


# --------------------------------------------------------------------------- #
# stage_state: the skip verdict
# --------------------------------------------------------------------------- #


def test_skips_when_nothing_changed(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    _produce(spec)
    assert spec.skip_if_fresh() is True


def test_first_run_has_no_stamp_so_it_builds(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    assert spec.skip_if_fresh() is False


def test_changed_input_content_forces_rebuild(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    _produce(spec)
    # Same size, different bytes: only a content hash catches this.
    path = tree / "unit.ads"
    path.write_text("package Q is\n   procedure Stop;\nend Q;\n", encoding="utf-8")
    assert spec.skip_if_fresh() is False


def test_new_input_file_forces_rebuild(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    _produce(spec)
    _write(tree / "extra.ads", "package R is\nend R;\n")
    assert spec.skip_if_fresh() is False


def test_removed_input_file_forces_rebuild(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    _produce(spec)
    (tree / "unit.ads").unlink()
    assert spec.skip_if_fresh() is False


def test_file_outside_declared_suffixes_is_ignored(tmp_path, tree):
    """The fingerprint covers what the stage reads, not the whole directory."""
    spec = _spec(tmp_path, tree)
    _produce(spec)
    (tree / "README.md").write_text("# changed\n", encoding="utf-8")
    assert spec.skip_if_fresh() is True


def test_changed_param_forces_rebuild(tmp_path, tree):
    _produce(_spec(tmp_path, tree))
    assert _spec(tmp_path, tree, params=[("cap", 20)]).skip_if_fresh() is False


def test_missing_output_forces_rebuild(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    _produce(spec)
    spec.outputs[0].unlink()
    assert spec.skip_if_fresh() is False


def test_truncated_output_forces_rebuild(tmp_path, tree):
    """Same path, different size: a half-written dataset must never look fresh."""
    spec = _spec(tmp_path, tree)
    _produce(spec)
    spec.outputs[0].write_text("", encoding="utf-8")
    assert spec.skip_if_fresh() is False


def test_empty_output_forces_rebuild(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    for output in spec.outputs:
        _write(output, "")
    spec.mark_fresh()
    assert spec.skip_if_fresh() is False


def test_changed_script_forces_rebuild(tmp_path, tree):
    script = _write(tmp_path / "gen.py", "# v1\n")
    spec = _spec(tmp_path, tree, scripts=[script])
    _produce(spec)
    script.write_text("# v2\n", encoding="utf-8")
    assert spec.skip_if_fresh() is False


def test_force_always_rebuilds(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    _produce(spec)
    assert spec.skip_if_fresh(force=True) is False


def test_corrupt_stamp_forces_rebuild(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    _produce(spec)
    spec.stamp_path.write_text("{not json", encoding="utf-8")
    assert spec.skip_if_fresh() is False


def test_stamp_from_another_output_path_is_not_fresh(tmp_path, tree):
    """A run with a custom --output must not mark the default run fresh."""
    _produce(_spec(tmp_path, tree))
    elsewhere = _spec(tmp_path, tree, outputs=[tmp_path / "other" / "data.jsonl"])
    assert elsewhere.skip_if_fresh() is False


def test_missing_input_tree_is_recorded_and_recovers(tmp_path, tree):
    """A source repo that is absent now but appears later must invalidate."""
    absent = tmp_path / "not-fetched-yet"
    spec = _spec(tmp_path, tree, input_trees=[(absent, (".ads",))])
    _produce(spec)
    assert spec.skip_if_fresh() is True
    _write(absent / "new.ads", "package S is\nend S;\n")
    assert spec.skip_if_fresh() is False


def test_no_stage_puts_workers_in_its_fingerprint():
    """Worker count must stay out of every spec.

    The stages document identical output for any worker count, so recording it
    would make `DATASET_WORKERS=8` invalidate a dataset built with 1 for no
    reason. This guards the next edit that adds `("workers", args.workers)`.
    """
    root = Path(__file__).resolve().parents[1]
    producers = [
        root / "data" / "processing_scripts" / "parse_docs.py",
        root / "data" / "processing_scripts" / "parse_ada_ast.py",
        root / "data" / "processing_scripts" / "build_dataset.py",
        root / "scripts" / "gen_contract_mutations.py",
    ]
    for path in producers:
        text = path.read_text(encoding="utf-8")
        calls = [
            node for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Call)
            and (getattr(node.func, "attr", "") or getattr(node.func, "id", "")) == "make_spec"
        ]
        assert calls, f"{path.name} has no stage spec"
        for call in calls:
            segment = ast.get_source_segment(text, call) or ""
            assert "workers" not in segment, (
                f"{path.name} passes workers into its stage fingerprint; worker "
                "count does not change the output and must not invalidate a build"
            )


def test_mark_fresh_refuses_when_an_output_is_missing(tmp_path, tree):
    spec = _spec(tmp_path, tree, outputs=[tmp_path / "out" / "a.jsonl", tmp_path / "out" / "b.jsonl"])
    _write(spec.outputs[0], "{}\n")
    spec.mark_fresh()
    assert not spec.stamp_path.exists()


def test_stamp_is_valid_json_with_version(tmp_path, tree):
    spec = _spec(tmp_path, tree)
    _produce(spec)
    stamp = json.loads(spec.stamp_path.read_text(encoding="utf-8"))
    assert stamp["stage"] == "unit-test"
    assert stamp["stamp_version"] == stage_state.STAMP_VERSION
    assert stamp["outputs"][0]["size"] == 3


def test_shared_scripts_are_always_fingerprinted(tmp_path, tree):
    """Editing a helper every stage imports must invalidate every stage."""
    spec = _spec(tmp_path, tree)
    paths = [entry["path"] for entry in spec.fingerprint()["scripts"]]
    assert str(Path(stage_state.__file__).resolve()) in paths
    assert str(stage_state.SHARED_SCRIPTS[0].resolve()) in [
        str(Path(p).resolve()) for p in stage_state.SHARED_SCRIPTS
    ]


def test_params_order_does_not_matter(tmp_path, tree):
    _produce(_spec(tmp_path, tree, params=[("a", 1), ("b", 2)]))
    assert _spec(tmp_path, tree, params=[("b", 2), ("a", 1)]).skip_if_fresh() is True


# --------------------------------------------------------------------------- #
# progress: throttling and phase timing
# --------------------------------------------------------------------------- #


def test_progress_logs_only_the_final_line_for_a_fast_loop(caplog):
    with caplog.at_level("INFO", logger="q3as_progress"):
        bar = progress.Progress("items", 10, min_interval=60.0)
        for _ in range(10):
            bar.advance()
        bar.close()
    lines = [r.message for r in caplog.records if "items" in r.message]
    assert len(lines) == 2  # "starting" plus the 100% line


def test_progress_closes_only_once(caplog):
    with caplog.at_level("INFO", logger="q3as_progress"):
        bar = progress.Progress("items", 2, min_interval=60.0)
        bar.advance(2)
        bar.close()
        bar.close()
    finals = [r.message for r in caplog.records if "2/2" in r.message]
    assert len(finals) == 1


def test_progress_respects_the_time_gate(caplog):
    """A loop that finishes inside the interval logs no intermediate line."""
    with caplog.at_level("INFO", logger="q3as_progress"):
        bar = progress.Progress("items", 100, every=1, min_interval=30.0)
        for _ in range(99):
            bar.advance()
        bar.advance()
    percent_lines = [r.message for r in caplog.records if "items/s" in r.message]
    assert len(percent_lines) == 1
    assert "100/100" in percent_lines[0]


def test_progress_logs_intermediates_once_the_interval_passes(caplog):
    with caplog.at_level("INFO", logger="q3as_progress"):
        bar = progress.Progress("items", 4, every=1, min_interval=0.0)
        bar.advance()
        bar.advance()
        bar.advance(2)
    percent_lines = [r.message for r in caplog.records if "items/s" in r.message]
    assert len(percent_lines) >= 2


def test_progress_handles_a_zero_total(caplog):
    with caplog.at_level("INFO", logger="q3as_progress"):
        bar = progress.Progress("items", 0)
        bar.advance()
        bar.close()
    assert any("0/0" in r.message for r in caplog.records)


def test_phase_logs_start_and_duration(caplog):
    with caplog.at_level("INFO", logger="q3as_progress"), progress.phase("dedup", records=5):
        time.sleep(0.01)
    messages = [r.message for r in caplog.records]
    assert any("dedup" in m and "starting" in m and "records=5" in m for m in messages)
    assert any("dedup" in m and "done in" in m for m in messages)


def test_phase_logs_duration_even_when_the_step_raises(caplog):
    with (
        caplog.at_level("INFO", logger="q3as_progress"),
        pytest.raises(RuntimeError),
        progress.phase("boom"),
    ):
        raise RuntimeError("kaboom")
    assert any("boom" in r.message and "done in" in r.message for r in caplog.records)


def test_fmt_duration_is_compact():
    assert progress._fmt_duration(0.4) == "0s"
    assert progress._fmt_duration(45) == "45s"
    assert progress._fmt_duration(125) == "2m05s"
    assert progress._fmt_duration(3900) == "1h05m"
