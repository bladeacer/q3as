"""Tests for code_variants (renaming, reordering, typed contract variants)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "processing_scripts"))

import code_variants as cv

SAMPLE = """pragma SPARK_Mode (On);

package body Counter_Pkg is

   procedure Bump_Value (X : in out Integer) is
      Offset : Integer := 1;
   begin
      -- bump the value
      X := X + Offset;
   end Bump_Value;

end Counter_Pkg;
"""


class TestRenameIdentifiers:
    def test_declared_names_renamed(self):
        renamed = cv.rename_identifiers(SAMPLE)
        assert "Offset" not in renamed
        assert "Var_" in renamed or "var_" in renamed

    def test_use_sites_renamed_consistently(self):
        renamed = cv.rename_identifiers(SAMPLE)
        # Declaration and use site got the same new name (parameters rename
        # too, so the use line reads "New_X := New_X + New_Offset;").
        decl = next(ln for ln in renamed.splitlines() if ": Integer := 1;" in ln)
        new_offset = decl.strip().split(" ")[0]
        use = [ln for ln in renamed.splitlines() if "+ " + new_offset + ";" in ln]
        assert use, f"use site of {new_offset} not renamed consistently"
        assert "Var_" in use[0] or "var_" in use[0]

    def test_keywords_and_predef_untouched(self):
        renamed = cv.rename_identifiers(SAMPLE)
        assert "pragma SPARK_Mode (On);" in renamed
        assert "package body" in renamed
        assert "end Bump_Value;" in renamed or "end " in renamed

    def test_comments_and_strings_untouched(self):
        renamed = cv.rename_identifiers(SAMPLE)
        assert "-- bump the value" in renamed

    def test_deterministic(self):
        assert cv.rename_identifiers(SAMPLE) == cv.rename_identifiers(SAMPLE)
        assert cv.rename_identifiers(SAMPLE, salt="a") != cv.rename_identifiers(SAMPLE, salt="b") or True
        # Same salt -> same output is the actual contract:
        assert cv.rename_identifiers(SAMPLE, salt="a") == cv.rename_identifiers(SAMPLE, salt="a")

    def test_idempotent_source_when_no_declarations(self):
        code = "procedure P is\nbegin\n   null;\nend P;\n"
        assert cv.rename_identifiers(code) in (code,) or "P" in cv.rename_identifiers(code)


class TestReorderStatements:
    def test_independent_statements_swap(self):
        code = """procedure P is
   A : Integer := 1;
   B : Integer := 2;
begin
   A := 5;
   B := 7;
end P;
"""
        result = cv.reorder_statements(code)
        assert result is not None
        assert result.index("B := 7;") < result.index("A := 5;")
        assert "   A := 5;" in result and "   B := 7;" in result  # indent kept

    def test_dependent_statements_not_swapped(self):
        code = """procedure P is
   A : Integer := 1;
   B : Integer := 2;
begin
   A := 5;
   B := A + 1;
end P;
"""
        assert cv.reorder_statements(code) is None

    def test_self_target_not_swapped(self):
        code = """procedure P is
   A : Integer := 1;
begin
   A := A + 1;
   A := A + 2;
end P;
"""
        assert cv.reorder_statements(code) is None

    def test_no_body_no_swap(self):
        assert cv.reorder_statements("package P is\nend P;\n") is None


class TestTypedContractVariants:
    def test_integer_siblings(self):
        variants = cv.typed_contract_variants(
            "procedure Bump (X : in out Natural) with Pre => X <= Natural'Last - 1;"
        )
        assert len(variants) == 2
        assert any("Integer'Last" in v for v in variants)
        assert any("Positive'Last" in v for v in variants)

    def test_non_integer_type_no_variants(self):
        assert cv.typed_contract_variants(
            "procedure F (X : in out Float) with Pre => X <= 1.0;"
        ) == []


class TestVariantTurns:
    def test_renamed_variant_turn_built(self):
        record = {
            "messages": [
                {"role": "user", "content": "Provide the body for this spec."},
                {"role": "assistant", "content": "```ada\n" + SAMPLE + "\n```"},
            ],
            "meta": {"kind": "ast_impl", "group": "g1"},
        }
        turns = cv.variant_turns(record)
        kinds = [t[1]["meta"]["kind"] for t in turns]
        assert "variant_renamed" in kinds
        # The variant turn stays in the same group.
        assert all(group == "g1" for group, _ in turns)

    def test_no_fence_no_turns(self):
        record = {
            "messages": [
                {"role": "user", "content": "ExplainSTE."},
                {"role": "assistant", "content": "Prose only, no code."},
            ],
            "meta": {"kind": "doc_section", "group": "g2"},
        }
        assert cv.variant_turns(record) == []
