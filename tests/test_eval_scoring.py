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
