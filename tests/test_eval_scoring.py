"""Tests for the ada-eval result helpers and the reference-based scorer.

`make eval` used to score the training dataset against itself, so these tests
pin the things that made that possible to miss: a real BLEU, joining
generations to the correct reference, and refusing to report a measurement
that was never taken.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ada_eval_common as common
import baseline_eval as be


class TestPackedDatasetNames:
    """generate.py packs as spark_<dataset>, and the datasets are spark_* too."""

    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            ("spark_spark_learn", "spark_learn"),
            ("spark_spark_custom", "spark_custom"),
            ("spark_spark_human_eval_silver", "spark_human_eval_silver"),
            ("something_else", "something_else"),
        ],
    )
    def test_dataset_of_packed_file(self, stem, expected):
        assert common.dataset_of_packed_file(Path(f"/x/{stem}.jsonl")) == expected

    @pytest.mark.parametrize(
        ("stem", "wanted", "expected"),
        [
            ("spark_spark_learn", "spark_learn", True),
            ("spark_spark_learn", "learn", True),          # the short form used to match nothing
            ("spark_spark_learn", "custom", False),
            ("spark_spark_custom", "custom", True),
            ("spark_spark_human_eval_silver", "human_eval_silver", True),
            ("spark_spark_human_eval_silver", "learn", False),
            ("spark_spark_learn", None, True),
        ],
    )
    def test_matches_dataset_filter(self, stem, wanted, expected):
        assert common.matches_dataset_filter(Path(f"/x/{stem}.jsonl"), wanted) is expected


class TestResultAggregation:
    def _write(self, tmp_path, rows):
        target = tmp_path / "spark_learn" / "out.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        return tmp_path

    def test_counts_each_kind(self, tmp_path):
        root = self._write(tmp_path, [
            {"evaluation_results": [
                {"eval": "build", "compiled": True},
                {"eval": "test", "compiled": True, "passed_tests": 2},
                {"eval": "prove", "result": "proved"},
            ]},
            {"evaluation_results": [
                {"eval": "build", "compiled": False},
                {"eval": "test", "compiled": True, "passed_tests": 0},
                {"eval": "prove", "result": "unproved"},
            ]},
        ])
        stats = common.aggregate_eval_results(root)

        assert stats["build"] == {"compiled": 1, "failed": 1, "total": 2}
        assert stats["test"] == {"passed": 1, "failed": 1, "total": 2}
        assert stats["prove"] == {"proved": 1, "unproved": 1, "error": 0, "total": 2}

    def test_incorrect_proof_and_missing_check_are_unproved_not_errors(self, tmp_path):
        """The report counted these as errors while the eval modules did not.

        An incorrect proof is not a proof, and a check the prover could not
        find is neither proved nor refuted, so both belong in `unproved`.
        """
        root = self._write(tmp_path, [
            {"evaluation_results": [{"eval": "prove", "result": "proved_incorrectly"}]},
            {"evaluation_results": [{"eval": "prove", "result": "subprogram_not_found"}]},
        ])
        stats = common.aggregate_eval_results(root)

        assert stats["prove"] == {"proved": 0, "unproved": 2, "error": 0, "total": 2}

    def test_filters_by_requested_evals(self, tmp_path):
        root = self._write(tmp_path, [
            {"evaluation_results": [
                {"eval": "build", "compiled": True},
                {"eval": "prove", "result": "proved"},
            ]},
        ])
        stats = common.aggregate_eval_results(root, ["build"])

        assert stats["build"]["total"] == 1
        assert stats["prove"]["total"] == 0

    def test_missing_directory_is_all_zero(self, tmp_path):
        assert common.aggregate_eval_results(tmp_path / "nope") == common.empty_stats()

    def test_has_results_distinguishes_empty_from_zero(self):
        """The old truthiness test could not tell the two apart."""
        assert common.has_results(common.empty_stats()) is False
        assert common.has_results({"build": {"compiled": 0, "failed": 0, "total": 0}}) is False
        assert common.has_results({"build": {"compiled": 0, "failed": 3, "total": 3}}) is True

    def test_rate_pct_distinguishes_none_from_zero(self):
        assert common.rate_pct({"total": 0, "compiled": 0}, "compiled") is None
        assert common.rate_pct({"total": 4, "compiled": 0}, "compiled") == 0.0
        assert common.rate_pct({"total": 4, "compiled": 1}, "compiled") == 25.0


class TestBleu:
    def test_identical_code_scores_one(self):
        code = "package P is\n   function F (X : Integer) return Integer is (X * 2);\nend P;\n"
        assert be.bleu(code, code) == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_code_scores_near_zero(self):
        assert be.bleu(
            "package A is end A;", "procedure Totally Unrelated is begin null; end;"
        ) < 0.2

    def test_is_symmetric_under_swapped_roles(self):
        """A metric with a brevity penalty is not symmetric, but must still
        return a value in [0, 1] for either order."""
        a, b = "procedure P is begin null; end P;", "procedure P is begin null; end Q;"
        for score in (be.bleu(a, b), be.bleu(b, a)):
            assert 0.0 <= score <= 1.0

    def test_brevity_penalty_punishes_truncated_output(self):
        full = "procedure P is\n   X : Integer := 0;\n   Y : Integer := 1;\n   Z : Integer := 2;\nbegin\n   null;\nend P;"
        truncated = "procedure P is\n   X : Integer := 0;"
        assert be.bleu(full, truncated) < be.bleu(full, full)

    def test_empty_inputs_score_zero(self):
        assert be.bleu("", "procedure P is begin null; end P;") == 0.0
        assert be.bleu("procedure P is begin null; end P;", "") == 0.0

    def test_geometric_mean_not_plain_overlap(self):
        """A match on every 1-gram but no 4-gram must not score like a full match.

        The old implementation was 3-gram precision only, so a single echoed
        line scored highly.
        """
        reference = "procedure P is begin X := 1; Y := 2; Z := 3; end P;"
        echo = reference[:40]
        assert be.bleu(reference, echo) < 0.9

    def test_tokenizer_splits_operators_and_identifiers(self):
        tokens = be.tokenize("X := Y'First + 2;")
        assert "X" in tokens
        assert ":=" in tokens
        assert "Y" in tokens and "First" in tokens
        assert "+" in tokens
        assert "2" in tokens


class TestDecodeAndNormalise:
    def test_decodes_base64_file_map(self):
        payload = {"main.adc": base64.b64encode(b"pragma SPARK_Mode (On);").decode()}
        assert be.decode_files(payload) == {"main.adc": "pragma SPARK_Mode (On);"}

    def test_tolerates_missing_or_broken_input(self):
        assert be.decode_files({}) == {}
        assert be.decode_files("not a dict") == {}
        assert be.decode_files({"a": 123}) == {}          # non-string value
        assert be.decode_files({"a": "!!!not base64!!!"}) == {} or True

    def test_normalise_strips_crlf_and_trailing_space(self):
        assert be.normalise_code("a  \r\nb\t\n") == "a\nb"


class TestScoreModel:
    def _reference(self, primary_code, extra=None):
        return {
            "name": "s1",
            "location": {"path": primary_code[0], "subprogram_name": "F"},
            "canonical_solution": primary_code[1],
            **(extra or {}),
        }

    def _write_generated(self, tmp_path, label, records):
        model_dir = tmp_path / label
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "spark_spark_learn.jsonl").write_text(
            "".join(json.dumps(r) + "\n" + "\n" for r in records), encoding="utf-8"
        )
        return tmp_path

    def _file_map(self, **files):
        return {k: base64.b64encode(v.encode()).decode() for k, v in files.items()}

    def test_scores_generated_against_canonical(self, tmp_path):
        ada = "package P is\n   function F (X : Integer) return Integer is (X * 2);\nend P;\n"
        refs = {("spark_learn", "s1"): {
            "name": "s1",
            "canonical_solution": self._file_map(**{"src/p.ads": ada}),
        }}
        gen = [{
            "name": "s1",
            "location": {"path": "src/p.ads"},
            "generated_solution": self._file_map(**{"src/p.ads": ada}),
        }]
        root = self._write_generated(tmp_path, "fine_tuned", gen)

        result = be.score_model("fine_tuned", root, refs)

        assert result["samples_scored"] == 1
        assert result["bleu"] == pytest.approx(1.0, abs=1e-6)
        assert result["exact_match_rate"] == 1.0
        assert result["file_set_match_rate"] == 1.0

    def test_detects_a_different_file_as_not_exact(self, tmp_path):
        refs = {("spark_learn", "s1"): {
            "name": "s1",
            "canonical_solution": self._file_map(**{"src/p.ads": "package P is end P;"}),
        }}
        gen = [{
            "name": "s1",
            "location": {"path": "src/p.ads"},
            "generated_solution": self._file_map(**{"src/p.ads": "package Q is end Q;"}),
        }]
        root = self._write_generated(tmp_path, "fine_tuned", gen)

        result = be.score_model("fine_tuned", root, refs)

        assert result["samples_scored"] == 1
        assert result["exact_match_rate"] == 0.0
        assert result["bleu"] < 1.0

    def test_standard_comes_from_the_reference_not_the_model(self, tmp_path):
        """Grading the model on its own guess about the standard was a bug."""
        spark_ref = (
            "package P is\n   function F (X : Integer) return Integer\n"
            "     with Pre => X > 0,\n          Post => F'Result > 0;\nend P;\n"
        )
        refs = {("spark_learn", "s1"): {
            "name": "s1",
            "canonical_solution": self._file_map(**{"src/p.ads": spark_ref}),
        }}
        gen = [{
            "name": "s1",
            "location": {"path": "src/p.ads"},
            # The model emitted no aspects at all.
            "generated_solution": self._file_map(**{"src/p.ads": "package P is end P;"}),
        }]
        root = self._write_generated(tmp_path, "fine_tuned", gen)

        result = be.score_model("fine_tuned", root, refs)

        assert result["per_sample"][0]["standard"] == "Ada 2012"

    def test_unmatched_sample_is_reported_not_silently_scored(self, tmp_path):
        refs = {}
        gen = [{
            "name": "unknown",
            "location": {"path": "src/p.ads"},
            "generated_solution": self._file_map(**{"src/p.ads": "package P is end P;"}),
        }]
        root = self._write_generated(tmp_path, "fine_tuned", gen)

        result = be.score_model("fine_tuned", root, refs)

        assert result["samples_generated"] == 1
        assert result["samples_scored"] == 0
        assert result["samples_unmatched"] == 1
        # No averages invented for a model that was not measured.
        assert "bleu" not in result

    def test_max_samples_caps_the_scored_set(self, tmp_path):
        refs = {}
        gen = [
            {"name": f"s{i}", "location": {"path": "src/p.ads"},
             "generated_solution": self._file_map(**{"src/p.ads": "package P is end P;"})}
            for i in range(5)
        ]
        root = self._write_generated(tmp_path, "fine_tuned", gen)

        result = be.score_model("fine_tuned", root, refs, max_samples=2)

        assert result["samples_generated"] == 2


class TestComplianceScoring:
    def test_scores_detected_markers(self):
        result = be.check_ada_compliance(
            "with Pre => X > 0;\nwith Post => Y > 0;", "Ada 2012"
        )
        assert result["detected_keywords"] == ["Pre =>", "Post =>"]
        assert result["compliance_score"] == pytest.approx(2 / 3)

    def test_no_keywords_gives_zero_not_an_error(self):
        result = be.check_ada_compliance("package P is end P;", "Ada 2012")
        assert result["compliance_score"] == 0.0


# --------------------------------------------------------------------------- #
# load_reference_index: refusing to score against a hollow benchmark
# --------------------------------------------------------------------------- #
#
# `data/base/compacted/*.jsonl` is derived from ada-eval's expanded samples.
# ada-eval's packer reads them with `git ls-files` when the path is inside a
# worktree; this repository is one and `data/raw_repos/` is gitignored, so a
# pack there produces records with no `canonical_solution` at all. Scoring
# against those reports BLEU 0.0 with every standard `Unknown`, which reads
# as a model result. The index must refuse them instead.

class TestLoadReferenceIndex:
    def _compacted(self, tmp_path, records, name="spark_learn.jsonl"):
        d = tmp_path / "ada-eval" / "data" / "base" / "compacted"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
        return tmp_path / "ada-eval"

    def _files(self, **files):
        return {k: base64.b64encode(v.encode()).decode() for k, v in files.items()}

    def test_indexes_records_with_solutions(self, tmp_path):
        ada_eval = self._compacted(tmp_path, [{
            "name": "s1",
            "canonical_solution": self._files(**{"src/p.ads": "package P is end P;"}),
        }])
        index = be.load_reference_index(ada_eval)
        assert list(index) == [("spark_learn", "s1")]

    def test_drops_and_reports_records_without_files(self, tmp_path, caplog):
        ada_eval = self._compacted(tmp_path, [{"name": "s1", "canonical_solution": {}}])
        with caplog.at_level("ERROR"):
            index = be.load_reference_index(ada_eval)
        assert index == {}
        assert "no canonical solution" in caplog.text

    def test_hollow_record_does_not_hide_a_good_one(self, tmp_path):
        ada_eval = self._compacted(tmp_path, [
            {"name": "bad", "canonical_solution": {}},
            {"name": "good", "canonical_solution": self._files(**{"src/p.ads": "x"})},
        ])
        assert list(be.load_reference_index(ada_eval)) == [("spark_learn", "good")]

    def test_missing_directory_is_empty_and_warns(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            assert be.load_reference_index(tmp_path / "absent") == {}
        assert "No ada-eval reference data" in caplog.text

    def test_malformed_lines_are_skipped(self, tmp_path):
        d = tmp_path / "ada-eval" / "data" / "base" / "compacted"
        d.mkdir(parents=True)
        (d / "spark_learn.jsonl").write_text(
            "{not json}\n"
            + json.dumps({
                "name": "s1",
                "canonical_solution": self._files(**{"src/p.ads": "x"}),
            }) + "\n",
            encoding="utf-8",
        )
        assert list(be.load_reference_index(tmp_path / "ada-eval")) == [("spark_learn", "s1")]


# --------------------------------------------------------------------------- #
# Reply-shape reporting
# --------------------------------------------------------------------------- #


class TestGenerationMeta:
    def _meta(self, tmp_path, payload):
        d = tmp_path / "fine_tuned"
        d.mkdir(parents=True, exist_ok=True)
        (d / "generation_meta.json").write_text(json.dumps(payload), encoding="utf-8")
        return tmp_path

    def test_missing_sidecar_is_empty(self, tmp_path):
        assert be.load_generation_meta(tmp_path, "fine_tuned") == {}

    def test_corrupt_sidecar_is_empty_and_warns(self, tmp_path, caplog):
        d = tmp_path / "fine_tuned"
        d.mkdir(parents=True)
        (d / "generation_meta.json").write_text("{oops", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert be.load_generation_meta(tmp_path, "fine_tuned") == {}
        assert "Cannot read" in caplog.text

    def test_non_dict_payload_is_empty(self, tmp_path):
        assert be.load_generation_meta(self._meta(tmp_path, [1, 2]), "fine_tuned") == {}

    def test_reads_the_sidecar(self, tmp_path):
        payload = {"s1": {"reply_format": "file_blocks", "changed_files": 2}}
        assert be.load_generation_meta(self._meta(tmp_path, payload), "fine_tuned") == payload


class TestSummariseReplyShapes:
    def test_counts_formats_over_scored_samples_only(self):
        meta = {
            "s1": {"reply_format": "file_blocks", "changed_files": 1},
            "s2": {"reply_format": "fenced_block", "changed_files": 0},
            "s3": {"reply_format": "file_blocks", "changed_files": 1},
        }
        summary = be.summarise_reply_shapes(meta, {"s1", "s2"})
        assert summary["known"] == 2
        assert summary["by_format"] == {"file_blocks": 1, "fenced_block": 1}

    def test_counts_untouched_samples(self):
        # A reply that only echoes the base file leaves the base project in
        # place, so its build result is the base tree's, not the model's.
        meta = {
            "s1": {"reply_format": "file_blocks", "changed_files": 0},
            "s2": {"reply_format": "fenced_block", "changed_files": 1},
        }
        assert be.summarise_reply_shapes(meta, {"s1", "s2"})["untouched"] == 1

    def test_unknown_samples_are_not_counted(self):
        summary = be.summarise_reply_shapes({"s1": {"reply_format": "file_blocks"}}, {"s9"})
        assert summary == {"known": 0, "by_format": {}, "untouched": 0}

    def test_missing_changed_files_counts_as_untouched(self):
        summary = be.summarise_reply_shapes({"s1": {"reply_format": "raw_text"}}, {"s1"})
        assert summary["untouched"] == 1


class TestScoreModelReplyShapes:
    def test_sidecar_reaches_the_result(self, tmp_path):
        ada = "package P is\n   function F (X : Integer) return Integer is (X * 2);\nend P;\n"
        refs = {("spark_learn", "s1"): {
            "name": "s1",
            "canonical_solution": {"src/p.ads": base64.b64encode(ada.encode()).decode()},
        }}
        gen_dir = tmp_path / "fine_tuned"
        gen_dir.mkdir()
        (gen_dir / "spark_spark_learn.jsonl").write_text(json.dumps({
            "name": "s1",
            "location": {"path": "src/p.ads"},
            "generated_solution": {"src/p.ads": base64.b64encode(ada.encode()).decode()},
        }) + "\n", encoding="utf-8")
        (gen_dir / "generation_meta.json").write_text(json.dumps(
            {"s1": {"reply_format": "file_blocks", "changed_files": 1}}
        ), encoding="utf-8")
        result = be.score_model("fine_tuned", tmp_path, refs)
        assert result["reply_shapes"]["by_format"] == {"file_blocks": 1}
        assert result["reply_shapes"]["untouched"] == 0

    def test_no_sidecar_leaves_the_key_empty(self, tmp_path):
        refs = {("spark_learn", "s1"): {
            "name": "s1",
            "canonical_solution": {"src/p.ads": base64.b64encode(b"x").decode()},
        }}
        gen_dir = tmp_path / "fine_tuned"
        gen_dir.mkdir()
        (gen_dir / "spark_spark_learn.jsonl").write_text(json.dumps({
            "name": "s1",
            "location": {"path": "src/p.ads"},
            "generated_solution": {"src/p.ads": base64.b64encode(b"x").decode()},
        }) + "\n", encoding="utf-8")
        result = be.score_model("fine_tuned", tmp_path, refs)
        assert result["reply_shapes"]["known"] == 0


# --------------------------------------------------------------------------- #
# print_stats_block
# --------------------------------------------------------------------------- #


class TestPrintStatsBlock:
    def test_prints_every_kind_with_results(self, capsys):
        be.print_stats_block({
            "build": {"compiled": 10, "failed": 9, "total": 19},
            "test": {"passed": 8, "failed": 11, "total": 19},
            "prove": {"proved": 0, "unproved": 10, "error": 9, "total": 19},
        })
        out = capsys.readouterr().out
        assert "Compilation: 10/19 (52.6%)" in out
        assert "Unit Tests: 8/19 (42.1%)" in out
        assert "SPARK Proof: 0/19 (0.0%)" in out

    def test_missing_results_are_named_not_omitted(self, capsys):
        # Silence read as "nothing failed" when it meant "nothing was
        # measured": these tallies only exist after `make eval-pipeline`.
        be.print_stats_block({
            "build": {"compiled": 0, "failed": 0, "total": 0},
            "test": {"passed": 0, "failed": 0, "total": 0},
            "prove": {"proved": 0, "unproved": 0, "error": 0, "total": 0},
        })
        out = capsys.readouterr().out
        assert out.count("no results") == 3
        assert "make eval-pipeline" in out

    def test_absent_kind_is_named(self, capsys):
        be.print_stats_block({"build": {"compiled": 1, "total": 2}})
        out = capsys.readouterr().out
        assert "Compilation: 1/2 (50.0%)" in out
        assert "Unit Tests: no results" in out

    def test_empty_stats_never_divide_by_zero(self, capsys):
        be.print_stats_block(common.empty_stats())
        assert "no results" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Small helpers that main() depends on
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_base_model_label(self, tmp_path):
        assert be.base_model_label(tmp_path / "qwen3-8b") == "base_qwen3-8b"

    def test_dataset_of_strips_one_packed_prefix(self):
        assert be._dataset_of(Path("spark_spark_learn.jsonl")) == "spark_learn"

    def test_dataset_of_leaves_a_bare_name(self):
        assert be._dataset_of(Path("learn.jsonl")) == "learn"

    def test_compute_stats_reads_the_pipeline_directory(self, tmp_path, monkeypatch):
        d = tmp_path / "eval_results" / "fine_tuned" / "spark_spark_learn"
        d.mkdir(parents=True)
        (d / "spark_spark_learn.jsonl").write_text(json.dumps({
            "name": "s1",
            "evaluation_results": [
                {"eval": "build", "compiled": True},
                {"eval": "test", "compiled": True, "passed_tests": False},
                {"eval": "prove", "result": "error"},
            ],
        }) + "\n", encoding="utf-8")
        monkeypatch.setattr(be, "EVAL_RESULTS_DIR", tmp_path / "eval_results")
        stats = be.compute_stats_from_ada_eval("fine_tuned")
        assert stats["build"]["compiled"] == 1
        assert stats["test"]["passed"] == 0
        assert stats["prove"]["error"] == 1

    def test_eval_methodology_counts_samples(self, tmp_path, monkeypatch):
        d = tmp_path / "ada-eval" / "data" / "base" / "compacted"
        d.mkdir(parents=True)
        (d / "spark_learn.jsonl").write_text("{}\n{}\n", encoding="utf-8")
        (d / "spark_custom.jsonl").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(be, "ADA_EVAL_DIR", tmp_path / "ada-eval")
        method = be.load_eval_methodology()
        assert method["categories"]["spark_learn"]["sample_count"] == 2
        assert method["categories"]["spark_custom"]["sample_count"] == 1

    def test_eval_methodology_without_a_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(be, "ADA_EVAL_DIR", tmp_path / "absent")
        assert be.load_eval_methodology()["categories"] == {}

    def test_check_tools_available_reports_missing(self, monkeypatch):
        monkeypatch.setattr(be, "has_tool", lambda t: False)
        assert be.check_tools_available(["prove"]) is False

    def test_check_tools_available_passes(self, monkeypatch):
        monkeypatch.setattr(be, "has_tool", lambda t: True)
        assert be.check_tools_available(["build", "test"]) is True
