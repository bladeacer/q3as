"""Tests for the eval-integrity guard (data/processing_scripts/eval_guard.py).

Covers the adversarial variants the guard must catch: verbatim copies,
reformatted copies, identifier-renamed copies, and eval prompt containment.
Also covers what it must NOT catch: distinct corpus code that merely
resembles eval content.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "processing_scripts"))

import eval_guard as eg


def _sigs_with_ada(code: str) -> eg.EvalSignatures:
    sigs = eg.EvalSignatures()
    sigs.degraded = False
    eg._add_ada_text(sigs, code)
    return sigs


def _record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


EVAL_SUBPROGRAM = """procedure Checked_Inc (X : in out Natural) is
begin
   X := X + 1;
end Checked_Inc;
"""


class TestStructuralHashing:
    def test_rename_and_case_invariant(self):
        s1 = eg.structural_text(EVAL_SUBPROGRAM)
        renamed = EVAL_SUBPROGRAM.replace("Checked_Inc", "Buffer_Bump").replace("X", "Val")
        s2 = eg.structural_text(renamed.upper())
        assert s1 == s2

    def test_logic_change_changes_hash(self):
        s1 = eg.structural_text(EVAL_SUBPROGRAM)
        s2 = eg.structural_text(EVAL_SUBPROGRAM.replace("X + 1", "X + 2"))
        assert s1 != s2

    def test_comments_ignored(self):
        s1 = eg.structural_text(EVAL_SUBPROGRAM)
        s2 = eg.structural_text(EVAL_SUBPROGRAM.replace("begin", "-- note\nbegin"))
        assert s1 == s2

    def test_string_literal_canonicalized(self):
        s1 = eg.structural_text('Put_Line ("hello world");')
        s2 = eg.structural_text('Put_Line ("a different message");')
        assert s1 == s2


class TestContaminationDetection:
    def test_verbatim_block_is_caught(self):
        sigs = _sigs_with_ada(EVAL_SUBPROGRAM)
        record = _record(
            "Fix this subprogram.",
            f"```ada\n{EVAL_SUBPROGRAM}\n```",
        )
        assert eg._contamination_reason(record, sigs) is not None

    def test_renamed_copy_is_caught_structurally(self):
        sigs = _sigs_with_ada(EVAL_SUBPROGRAM)
        renamed = EVAL_SUBPROGRAM.replace("Checked_Inc", "Queue_Push").replace("X", "Item")
        record = _record(
            "Write this subprogram.",
            f"```ada\n{renamed}\n```",
        )
        reason = eg._contamination_reason(record, sigs)
        assert reason is not None
        assert "structural" in reason

    def test_reformatted_copy_is_caught(self):
        sigs = _sigs_with_ada(EVAL_SUBPROGRAM)
        reformatted = EVAL_SUBPROGRAM.replace("   ", "").upper()
        record = _record("x", f"```ada\n{reformatted}\n```")
        assert eg._contamination_reason(record, sigs) is not None

    def test_distinct_code_is_not_flagged(self):
        sigs = _sigs_with_ada(EVAL_SUBPROGRAM)
        other = """procedure Halve (V : in out Positive) is
begin
   V := V / 2;
end Halve;
"""
        record = _record("x", f"```ada\n{other}\n```")
        assert eg._contamination_reason(record, sigs) is None

    def test_eval_prompt_containment_is_caught(self):
        sigs = eg.EvalSignatures()
        sigs.degraded = False
        sigs.exact_prompts.add(eg.normalized_text("Please can you make Absolute_Value provable."))
        record = _record(
            "Please can you make Absolute_Value provable. Here is the file.",
            "```ada\npackage body P is\nend P;\n```",
        )
        assert eg._contamination_reason(record, sigs) == "contains eval prompt text"


class TestContaminatedGroups:
    def test_whole_group_is_dropped(self):
        sigs = _sigs_with_ada(EVAL_SUBPROGRAM)
        bad = _record("x", f"```ada\n{EVAL_SUBPROGRAM}\n```")
        clean = _record("y", "```ada\npackage body Q is\nend Q;\n```")
        grouped = [
            ("g1", bad),
            ("g1", dict(bad)),  # sibling turn of the same group
            ("g2", clean),
        ]
        kept, dropped, reasons = eg.contaminated_groups(grouped, sigs)
        assert dropped == 2
        assert [g for g, _ in kept] == ["g2"]
        assert reasons == ["g1"]

    def test_degraded_guard_drops_nothing(self):
        sigs = eg.EvalSignatures()
        assert sigs.degraded is True
        grouped = [("g1", _record("x", f"```ada\n{EVAL_SUBPROGRAM}\n```"))]
        kept, dropped, _reasons = eg.contaminated_groups(grouped, sigs)
        assert dropped == 0
        assert len(kept) == 1


class TestBlocklistBuild:
    def test_compacted_sources_are_hashed(self, tmp_path: Path):
        # Minimal ada-eval-like tree: compacted JSONL with base64 source.
        import base64

        compacted = tmp_path / "data" / "base" / "compacted"
        compacted.mkdir(parents=True)
        encoded = base64.b64encode(EVAL_SUBPROGRAM.encode()).decode()
        (compacted / "spark_x.jsonl").write_text(
            '{"name": "s1", "prompt": "Prove the thing.", '
            f'"sources": {{"src/p.adb": "{encoded}"}}}}\n'
        )
        sigs = eg.load_eval_signatures(tmp_path)
        assert not sigs.degraded
        record = _record("x", f"```ada\n{EVAL_SUBPROGRAM}\n```")
        assert eg._contamination_reason(record, sigs) is not None

    def test_expanded_prompt_md_is_hashed(self, tmp_path: Path):
        sample = tmp_path / "data" / "base" / "expanded" / "ds1" / "sample_a"
        sample.mkdir(parents=True)
        (sample / "prompt.md").write_text("Make Foo provable with a range contract.")
        sigs = eg.load_eval_signatures(tmp_path)
        assert not sigs.degraded
        record = _record("Make Foo provable with a range contract.", "ok.")
        assert eg._contamination_reason(record, sigs) == "contains eval prompt text"

    def test_missing_dir_is_degraded(self, tmp_path: Path):
        sigs = eg.load_eval_signatures(tmp_path / "nowhere")
        assert sigs.degraded is True
        assert len(sigs) == 0


class TestCheckJsonl:
    def test_exit_code_contract(self, tmp_path: Path):
        import json as jsonlib

        clean_file = tmp_path / "clean.jsonl"
        clean_file.write_text(
            jsonlib.dumps(_record("x", "```ada\npackage body Q is\nend Q;\n```")) + "\n"
        )
        assert eg.check_jsonl(clean_file, _sigs_with_ada(EVAL_SUBPROGRAM)) == []

        dirty_file = tmp_path / "dirty.jsonl"
        dirty_file.write_text(
            jsonlib.dumps(_record("x", f"```ada\n{EVAL_SUBPROGRAM}\n```")) + "\n"
        )
        violations = eg.check_jsonl(dirty_file, _sigs_with_ada(EVAL_SUBPROGRAM))
        assert len(violations) == 1
        assert "structural" in violations[0] or "normalized" in violations[0]


class TestContractWriteTurns:
    """The spec-to-contract turns must strip aspects correctly."""

    def test_strip_aspects(self):
        import parse_ada_ast as pa

        spec = (
            "function F (X : Natural) return Natural\n"
            "   with Pre  => X > 0,\n"
            "        Post => F'Result > 0;"
        )
        bare = pa._strip_aspects(spec)
        assert "Pre" not in bare
        assert bare.rstrip().endswith("return Natural;")
        assert "function F (X : Natural)" in bare

    def test_strip_aspects_noop_without_aspects(self):
        import parse_ada_ast as pa

        spec = "function G (X : Natural) return Natural;"
        assert pa._strip_aspects(spec) == spec

    @pytest.mark.parametrize(
        "aspect_expr",
        ["X > 0", "F'Result = X * 2"],
    )
    def test_reconstruction_has_valid_aspect_clause(self, aspect_expr: str):
        # Comma-separated single aspect clause, not repeated `with`.
        aspects = {"Pre": aspect_expr, "Post": "F'Result > 0"}
        aspects_str = ",\n        ".join(f"{k} => {v}" for k, v in aspects.items())
        assert aspects_str.count("with") == 0
        assert "Pre =>" in aspects_str and "Post =>" in aspects_str
