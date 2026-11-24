"""Tests for scripts/bump_version.py and scripts/gen_eval_report.py."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bump_version as bv
import gen_eval_report as ger

CHANGELOGS_DIR = Path(__file__).resolve().parents[1] / "docs" / "changelogs"


@pytest.fixture(autouse=True)
def isolate_extra_version_files(monkeypatch):
    """Keep set_version away from the repository's own version files.

    EXTRA_VERSION_FILES points at the real pyproject.toml, and set_version
    rewrites it unconditionally. Without this, any test that calls
    set_version or bump_version edits the tracked file: `bump_version("major")`
    bakes 1.0.0 into pyproject.toml on every `make test`. Tests that exercise
    the extra-file path override this with their own file.
    """
    monkeypatch.setattr(bv, "EXTRA_VERSION_FILES", ())


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
    def _result_line(self, name, compiled, prove="unproved"):
        return json.dumps({
            "name": name,
            "evaluation_results": [
                {"eval": "build", "compiled": compiled},
                {"eval": "test", "compiled": compiled, "passed_tests": compiled},
                {"eval": "prove", "result": prove},
            ],
        })

    def _meta(self, tmp_path, label, payload):
        target = tmp_path / "generated_solutions" / label / "generation_meta.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")

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

    def test_records_the_changed_split_when_the_sidecar_exists(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(ger, "GENERATED_DIR", tmp_path / "generated_solutions")
        ft = tmp_path / "fine_tuned" / "spark_ds"
        ft.mkdir(parents=True)
        (ft / "r.jsonl").write_text(
            self._result_line("s1", True) + "\n" + self._result_line("s2", False) + "\n",
            encoding="utf-8",
        )
        self._meta(tmp_path, "fine_tuned", {
            "s1": {"changed_files": 1}, "s2": {"changed_files": 0},
        })

        metrics = ger.collect_ada_eval_metrics(tmp_path)

        by_change = metrics["fine_tuned"]["by_change"]
        assert by_change["edited"]["build"] == {"compiled": 1, "failed": 0, "total": 1}
        assert by_change["unchanged"]["build"] == {"compiled": 0, "failed": 1, "total": 1}

    def test_no_sidecar_leaves_the_split_out(self, tmp_path: Path, monkeypatch):
        # Absent, not zero: an older run has no sidecar, and by_change full of
        # zeros would render as "the model changed nothing", which is a claim.
        monkeypatch.setattr(ger, "GENERATED_DIR", tmp_path / "generated_solutions")
        ft = tmp_path / "fine_tuned" / "spark_ds"
        ft.mkdir(parents=True)
        (ft / "r.jsonl").write_text(self._result_line("s1", True) + "\n", encoding="utf-8")

        metrics = ger.collect_ada_eval_metrics(tmp_path)

        assert "by_change" not in metrics["fine_tuned"]


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

    def _changed_stats(self, edited, unchanged):
        return {
            "edited": {
                "build": {"compiled": edited[0], "failed": edited[1] - edited[0], "total": edited[1]},
                "test": {"passed": edited[0], "failed": edited[1] - edited[0], "total": edited[1]},
                "prove": {"proved": 0, "unproved": edited[1], "error": 0, "total": edited[1]},
            },
            "unchanged": {
                "build": {"compiled": unchanged[0], "failed": unchanged[1] - unchanged[0], "total": unchanged[1]},
                "test": {"passed": unchanged[0], "failed": unchanged[1] - unchanged[0], "total": unchanged[1]},
                "prove": {"proved": 0, "unproved": unchanged[1], "error": 0, "total": unchanged[1]},
            },
        }

    def test_markdown_states_the_changed_and_unchanged_split(self):
        data = self._data()
        data["ada_eval"]["fine_tuned"]["by_change"] = self._changed_stats((3, 4), (12, 15))
        text = ger.render_markdown("9.9.9", data)
        assert "## Samples the model changed" in text
        assert "| fine-tuned | changed | 4 | 3/4 (75.0%) | 3/4 (75.0%) | 0/4 (0.0%) |" in text
        assert "| fine-tuned | unchanged | 15 | 12/15 (80.0%) | 12/15 (80.0%) | 0/15 (0.0%) |" in text
        # The base model is in the same table: the split is per model.
        assert "| base | changed | n/a | n/a | n/a | n/a |" in text

    def test_markdown_says_not_recorded_without_a_sidecar(self):
        text = ger.render_markdown("9.9.9", self._data())
        assert "## Samples the model changed" in text
        assert "_Not recorded:" in text
        assert "| fine-tuned | changed |" not in text

    def test_split_section_absent_without_ada_eval_results(self):
        text = ger.render_markdown("1.0.0", {"version": "1.0.0", "generated_at": "x"})
        assert "Samples the model changed" not in text

    def test_unknown_model_label_is_still_shown(self):
        # --base-model can point anywhere, and generate.py names the directory
        # after it. A label this script has no pretty name for must still get a
        # row rather than vanishing from the table.
        data = self._data()
        stats = self._changed_stats((2, 2), (5, 5))
        data["ada_eval"]["base_llama_3"] = {
            "samples": 4, "build": 3, "by_change": stats,
        }
        text = ger.render_markdown("9.9.9", data)
        assert "| base_llama_3 | changed | 2 | 2/2 (100.0%)" in text

    def test_index_comparison_table_last_three(self):
        entries = [
            {**self._data(), "version": f"0.{i}.0"} for i in (3, 2, 1)
        ]
        entries.append({"version": "0.0.9", "generated_at": "x", "ada_eval": {}})
        text = ger.render_index(entries)
        assert "## Last 3 versions compared" in text
        assert "| v0.3.0 |" in text and "| v0.1.0 |" in text
        assert "v0.0.9" not in text.split("Last 3")[1].split("\n\n")[0]  # outside the table

    def test_index_links_only_versions_whose_report_exists(
        self, tmp_path, monkeypatch
    ):
        """A link in the index must resolve to a file that is really there.

        The index is built from the result-data JSONs, so a version whose
        markdown was deleted used to render a dead link. Reports present get
        a link; absent ones are called out instead.
        """
        monkeypatch.setattr(ger, "RESULTS_DIR", tmp_path)
        (tmp_path / "result-v0.1.0.md").write_text("# Results v0.1.0\n", encoding="utf-8")
        entries = [
            {**self._data(), "version": "0.1.0"},
            {"version": "0.2.0", "generated_at": "x", "ada_eval": {}},
        ]
        text = ger.render_index(entries)

        assert "[result-v0.1.0.md](result-v0.1.0.md)" in text
        assert "_(result-v0.2.0.md missing)_" in text
        # Every relative link the index emits must point at a real file.
        for target in re.findall(r"\]\((result-v[\d.]+\.md)\)", text):
            assert (tmp_path / target).exists(), f"index links a missing report: {target}"

    def test_index_comparison_carries_the_split_rows(self):
        """The cross-version table must state no-op replies and edited-sample build.

    Both are the numbers that stop a total build rate from being read as a
    model result, and both are absent for runs that recorded no sidecar, which
    has to render as n/a rather than as a zero.
    """
        entries = [
            {
                **self._data(),
                "version": "0.9.0",
                "baseline": {"per_model": {"fine_tuned": {"reply_shapes": {"untouched": 15}}}},
                "ada_eval": {
                    **self._data()["ada_eval"],
                    "fine_tuned": {
                        **self._data()["ada_eval"]["fine_tuned"],
                        "by_change": {
                            "edited": {
                                "build": {"compiled": 3, "failed": 1, "total": 4},
                                "test": {"passed": 2, "failed": 2, "total": 4},
                                "prove": {"proved": 1, "unproved": 3, "error": 0, "total": 4},
                            },
                            "unchanged": {
                                "build": {"compiled": 14, "failed": 1, "total": 15},
                                "test": {"passed": 10, "failed": 5, "total": 15},
                                "prove": {"proved": 0, "unproved": 14, "error": 1, "total": 15},
                            },
                        },
                    },
                },
            },
            {"version": "0.8.0", "generated_at": "x", "ada_eval": {}},
        ]
        text = ger.render_index(entries)
        table = text.split("Last 2 versions compared")[1]
        assert "| No-op replies (FT) | 15 | n/a |" in table
        assert "| Build on edited samples (FT) | 75.0% | n/a |" in table

    def test_every_real_result_file_is_linked_from_the_index(self):
        """The shipped index must cover each result file on disk, and every
        link in it must resolve. A withdrawn run may be absent from the index,
        but a present one may not be unlinked."""
        index = (ger.RESULTS_DIR / "README.md")
        if not index.exists():
            pytest.skip("no results index in this checkout")
        text = index.read_text(encoding="utf-8")
        for report in sorted(ger.RESULTS_DIR.glob("result-v*.md")):
            assert report.name in text, f"{report.name} exists but the index never links it"
        for target in re.findall(r"\]\((result-v[\d.]+\.md)\)", text):
            assert (ger.RESULTS_DIR / target).exists(), f"dead link in the index: {target}"

    def test_generated_result_files_match_the_generator(self):
        """docs/results/ is generated; a hand edit there is silently reverted
        by the next `make eval-report`. Rendering from the committed data and
        comparing bytes catches the edit while it is still in the diff."""
        index = ger.RESULTS_DIR / "README.md"
        if not index.exists():
            pytest.skip("no results index in this checkout")
        entries = []
        for data_path in sorted(ger.RESULTS_DIR.glob("result-data-v*.json")):
            entries.append(json.loads(data_path.read_text(encoding="utf-8")))
        if not entries:
            pytest.skip("no result data in this checkout")
        entries.sort(key=lambda e: [int(x) for x in e["version"].split(".")],
                     reverse=True)
        assert ger.render_index(entries) == index.read_text(encoding="utf-8"), (
            "docs/results/README.md differs from gen_eval_report.render_index: "
            "it was hand-edited, and `make eval-report` will overwrite it"
        )
        for data_path in entries:
            version = data_path["version"]
            report = ger.RESULTS_DIR / f"result-v{version}.md"
            if not report.exists():
                continue
            assert ger.render_markdown(version, data_path) == report.read_text(
                encoding="utf-8"
            ), f"{report.name} differs from the generator output: hand-edited"


class TestChangelogIndex:
    """docs/changelogs/ follows the same convention as docs/results/.

    An entry file that nothing links is invisible: the index is the only way
    a reader reaches a version, so a new vX.Y.Z.md must appear in it, and a
    link in it must not dangle.
    """

    def test_every_changelog_file_is_linked_from_the_index(self):
        index = CHANGELOGS_DIR / "index.md"
        if not index.exists():
            pytest.skip("no changelog index in this checkout")
        text = index.read_text(encoding="utf-8")
        for entry in sorted(CHANGELOGS_DIR.glob("v*.md")):
            assert entry.name in text, (
                f"{entry.name} exists but docs/changelogs/index.md never links it"
            )

    def test_index_links_resolve(self):
        index = CHANGELOGS_DIR / "index.md"
        if not index.exists():
            pytest.skip("no changelog index in this checkout")
        text = index.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((v[\d.]+\.md)\)", text):
            assert (CHANGELOGS_DIR / target).exists(), (
                f"dead link in the changelog index: {target}"
            )

    def test_current_version_has_an_entry(self):
        """`make bump-version` moves alire.toml; the release log must follow.

        Without this, a version bump ships with no changelog entry and the
        index silently stops at the previous release.
        """
        if not (CHANGELOGS_DIR / "index.md").exists():
            pytest.skip("no changelog index in this checkout")
        root = Path(bv.__file__).resolve().parents[1]
        version = bv.read_version(root / "alire.toml")
        entry = CHANGELOGS_DIR / f"v{version}.md"
        assert entry.exists(), (
            f"alire.toml is at {version} but docs/changelogs/{entry.name} is missing"
        )


class TestRepoVersionsAgree:
    """The real manifests must not drift apart.

    bump_version.py syncs them, but it can only rewrite what it finds: a
    manifest that is missing (or a pyproject whose version line moved) is
    silently skipped, and the next release quietly ships mismatched numbers.
    These assertions run against the repository, not a fixture.
    """

    def test_all_manifests_carry_the_same_version(self):
        root = Path(bv.__file__).resolve().parents[1]
        assert set(bv.MANIFESTS) == {root / name for name in
                                     ("alire.toml", "alire-dev.toml", "alire-ast.toml")}
        for manifest in bv.MANIFESTS:
            assert manifest.exists(), f"manifest missing: {manifest.name}"
        versions = {m.name: bv.read_version(m) for m in bv.MANIFESTS}
        assert len(set(versions.values())) == 1, f"manifest versions differ: {versions}"

    def test_pyproject_version_matches_the_manifests(self):
        root = Path(bv.__file__).resolve().parents[1]
        pyproject = root / "pyproject.toml"
        assert bv._VERSION_RE.search(pyproject.read_text(encoding="utf-8")), (
            "pyproject.toml has no top-level version line for bump_version to sync"
        )
        assert bv.read_version(root / "alire.toml") == bv.read_version(pyproject), (
            "pyproject.toml version differs from the Alire manifests"
        )
