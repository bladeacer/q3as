"""Tests for scripts/pack_eval_data.py.

`data/base/compacted/*.jsonl` is derived from ada-eval's `expanded` samples
and is not version controlled, so a fresh cache has none and `make eval`
exits 1. Rebuilding it has one trap: ada-eval's packer is git-aware, and the
cache lives inside this repository where `data/raw_repos/` is gitignored, so
`git ls-files` returns nothing and the pack succeeds while writing records
with no solution files. `make eval` then reports BLEU 0.0 with every standard
`Unknown` and looks like a model result.

The tests therefore pin three things: the ceiling variable is set, `force` is
passed (the git-aware guard would otherwise exit), and a hollow pack is a
hard failure rather than a silently wrong benchmark.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pack_eval_data as ped


def _record(name="char_count_1", files=None):
    return {
        "name": name,
        "canonical_solution": {"src/string_utils.ads": "package S is end S;"} if files is None else files,
    }


@pytest.fixture
def ada_eval(tmp_path, monkeypatch):
    """A cached ada-eval checkout with expanded samples on disk."""
    root = tmp_path / "ada-eval"
    expanded = root / "data" / "base" / "expanded" / "spark_custom"
    expanded.mkdir(parents=True)
    (root / "data" / "base" / "compacted").mkdir(parents=True)
    (expanded / "char_count_1").mkdir()
    monkeypatch.setattr(ped.source_paths, "resolve", lambda name: root)
    return root


def _write_packed(root: Path, records):
    target = root / "data" / "base" / "compacted" / "spark_custom.jsonl"
    target.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return target


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


class TestPaths:
    def test_dirs_follow_ada_evals_layout(self, tmp_path):
        assert ped.expanded_dir(tmp_path) == tmp_path / "data/base/expanded"
        assert ped.compacted_dir(tmp_path) == tmp_path / "data/base/compacted"


# --------------------------------------------------------------------------- #
# pack
# --------------------------------------------------------------------------- #


class TestPack:
    def test_missing_expanded_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ped.source_paths, "resolve", lambda name: tmp_path)
        assert ped.pack(tmp_path) == 1

    def test_sets_git_ceiling_and_forces(self, ada_eval, monkeypatch):
        seen: dict = {}
        mod = types.ModuleType("ada_eval.datasets.pack_unpack")

        def pack_datasets(src_dir, dest_dir, *, force=False):
            seen["src"] = Path(src_dir)
            seen["dest"] = Path(dest_dir)
            seen["force"] = force
            seen["ceiling"] = os.environ.get("GIT_CEILING_DIRECTORIES")

        mod.pack_datasets = pack_datasets
        for name in ("ada_eval", "ada_eval.datasets"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "ada_eval.datasets.pack_unpack", mod)

        assert ped.pack(ada_eval) == 0
        assert seen["src"] == ada_eval / "data/base/expanded"
        assert seen["dest"] == ada_eval / "data/base/compacted"
        # force=True: with the ceiling set, ada-eval's own "uncommitted
        # changes" guard fires and sys.exits instead of packing.
        assert seen["force"] is True
        # The ceiling is what stops git finding the outer q3as worktree.
        assert seen["ceiling"] == str(ada_eval)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


class TestVerify:
    def test_missing_compacted_dir(self, ada_eval):
        import shutil

        shutil.rmtree(ada_eval / "data" / "base" / "compacted")
        assert ped.verify(ada_eval) == 1

    def test_empty_compacted_dir(self, ada_eval):
        # The directory exists but holds no jsonl: not a benchmark.
        assert ped.verify(ada_eval) == 1

    def test_hollow_records_fail(self, ada_eval):
        _write_packed(ada_eval, [_record(files={}), _record("char_count_2", files={})])
        assert ped.verify(ada_eval) == 1

    def test_hollow_failure_names_the_samples(self, ada_eval, caplog):
        _write_packed(ada_eval, [_record(files={})])
        with caplog.at_level("ERROR"):
            assert ped.verify(ada_eval) == 1
        assert "char_count_1" in caplog.text

    def test_partially_hollow_still_fails(self, ada_eval):
        # One bad record poisons the benchmark: scoring it would report a
        # zero BLEU for a sample the model may well have solved.
        _write_packed(ada_eval, [_record(), _record("char_count_2", files={})])
        assert ped.verify(ada_eval) == 1

    def test_populated_records_pass(self, ada_eval, caplog):
        _write_packed(ada_eval, [_record(), _record("char_count_2")])
        with caplog.at_level("INFO"):
            assert ped.verify(ada_eval) == 0
        assert "2 samples" in caplog.text

    def test_load_packed_reads_every_dataset_file(self, ada_eval):
        _write_packed(ada_eval, [_record()])
        other = ada_eval / "data" / "base" / "compacted" / "spark_learn.jsonl"
        other.write_text(json.dumps(_record("absolute_value")) + "\n", encoding="utf-8")
        assert {r["name"] for r in ped.load_packed(ada_eval)} == {
            "char_count_1", "absolute_value",
        }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


class TestMain:
    def test_reports_a_missing_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ped.source_paths, "resolve", lambda name: None)
        assert ped.main([]) == 1

    def test_check_does_not_repack(self, ada_eval, monkeypatch, caplog):
        _write_packed(ada_eval, [_record()])

        def boom(ada_eval_dir):
            raise AssertionError("--check must not repack")

        monkeypatch.setattr(ped, "pack", boom)
        assert ped.main(["--check"]) == 0

    def test_check_fails_on_a_hollow_pack(self, ada_eval, monkeypatch):
        _write_packed(ada_eval, [_record(files={})])
        assert ped.main(["--check"]) == 1

    def test_default_run_packs_then_verifies(self, ada_eval, monkeypatch):
        _write_packed(ada_eval, [_record()])
        calls: list[str] = []

        def fake_pack(ada_eval_dir):
            calls.append("pack")
            return 0

        monkeypatch.setattr(ped, "pack", fake_pack)
        assert ped.main([]) == 0
        assert calls == ["pack"]

    def test_pack_failure_short_circuits(self, ada_eval, monkeypatch):
        monkeypatch.setattr(ped, "pack", lambda ada_eval_dir: 1)
        # verify is never reached, so no stale error is reported instead.
        monkeypatch.setattr(
            ped, "verify",
            lambda ada_eval_dir: pytest.fail("verify must not run after pack failed"),
        )
        assert ped.main([]) == 1