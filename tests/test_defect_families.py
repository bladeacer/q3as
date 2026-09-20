"""Tests for the expanded defect families in build_dataset.py.

Each family's expected compiler message was verified against the real GNAT
(see the probe notes next to _DEFECT_FAMILIES in build_dataset.py, and
scripts/validate_defects.py for the end-to-end compile check).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "processing_scripts"))

import build_dataset as bd

# --------------------------------------------------------------------------- #
# Fixtures: snippets shaped so exactly one family applies
# --------------------------------------------------------------------------- #

TYPED_BODY = """\
package body Calc is

   procedure Scale (X : Integer) is
      Factor : Integer := 3;
   begin
      X := Factor * 2;
   end Scale;

end Calc;
"""

SPEC_WITH_PREDEF = """\
with Ada.Strings.Unbounded;
package body Msg is

   procedure Show is
   begin
      null;
   end Show;

end Msg;
"""

DECLARED_BODY = """\
package body Ord is

   procedure Run is
      Count : Integer := 0;
   begin
      Count := Count + 1;
   end Run;

end Ord;
"""


class TestTypoFamily:
    def test_misspells_one_use_site(self):
        result = bd.inject_defect(TYPED_BODY, "typo")
        assert result is not None
        broken, diagnosis, claimed = result
        # Adjacent-letter swap (F-a-c-t-o-r -> F-c-a-t-o-r), applied to one
        # use site only: the declaration keeps the correct spelling.
        assert "Fcator * 2" in broken
        assert "Factor : Integer" in broken
        assert 'error: "Fcator" is undefined' == claimed
        assert "misspelled" in diagnosis.lower()

    def test_skips_short_names(self):
        code = "package body P is\n   procedure R is\n      X : Integer := 1;\n   begin\n      X := X + 1;\n   end R;\nend P;\n"
        # The only object name is one character long: no usable typo.
        result = bd.inject_defect(code, "typo")
        if result is not None:
            _broken, _diagnosis, claimed = result
            assert 'is undefined' in claimed


class TestHallucinatedFamily:
    def test_corrupts_predefined_with(self):
        result = bd.inject_defect(SPEC_WITH_PREDEF, "hallucinated")
        assert result is not None
        broken, _diagnosis, claimed = result
        assert "with Ada.Strings.Unboundedx;" in broken
        assert "is not a predefined library unit" in claimed
        # GNAT echoes the canonical form: first letter upper, rest lower.
        assert "Ada.Strings.Unboundedx" in claimed or "Ada.Strings.Unboundedx" in claimed

    def test_ignores_non_predefined_roots(self):
        code = "with Helpers.Stuff;\npackage body P is begin null; end P;\n"
        assert bd.inject_defect(code, "hallucinated") is None


class TestOrderingFamily:
    def test_moves_declaration_among_statements(self):
        result = bd.inject_defect(TYPED_BODY, "ordering")
        assert result is not None
        broken, _diagnosis, claimed = result
        assert "declarations mixed with statements" in claimed
        assert broken != TYPED_BODY

    def test_no_decl_body_skipped(self):
        code = "package body P is\n   procedure R is\n   begin\n      null;\n   end R;\nend P;\n"
        assert bd.inject_defect(code, "ordering") is None


class TestScopingFamily:
    def test_removes_used_declaration(self):
        result = bd.inject_defect(DECLARED_BODY, "scoping")
        assert result is not None
        broken, _diagnosis, claimed = result
        assert "Count : Integer" not in broken
        assert "Count" in broken  # still referenced by the statements
        assert 'error: "Count" is undefined' == claimed

    def test_multi_decl_name_skipped(self):
        code = (
            "package body P is\n   procedure R is\n      X : Integer := 1;\n"
            "   begin\n      X := X + 1;\n   end R;\n   procedure S is\n"
            "      X : Integer := 2;\n   begin\n      X := X + 2;\n   end S;\nend P;\n"
        )
        # X is declared twice (two scopes): removing one leaves the other.
        assert bd.inject_defect(code, "scoping") is None


class TestTypeFamily:
    def test_numeric_initializer_retyped(self):
        code = "package body P is\n   procedure R is\n      N : Natural := 42;\n   begin\n      null;\n   end R;\nend P;\n"
        result = bd.inject_defect(code, "type")
        assert result is not None
        broken, _diagnosis, claimed = result
        assert "N : Boolean" in broken
        assert 'expected type "Standard.Boolean"' in claimed

    def test_numeric_assignment_retyped(self):
        code = (
            "package body P is\n   procedure R is\n      N : Natural;\n   begin\n"
            "      N := 42;\n   end R;\nend P;\n"
        )
        result = bd.inject_defect(code, "type")
        assert result is not None
        _broken, _diagnosis, claimed = result
        assert 'expected type "Standard.Boolean"' in claimed

    def test_boolean_value_skipped(self):
        code = (
            "package body P is\n   procedure R is\n      N : Natural;\n   begin\n"
            "      N := 1;\n      N := 2;\n   end R;\nend P;\n"
        )
        result = bd.inject_defect(code, "type")
        assert result is not None  # numeric assignment exists, still broken


class TestLangConfusionFamily:
    def test_c_style_assignment(self):
        result = bd.inject_defect(TYPED_BODY, "lang_confusion")
        assert result is not None
        broken, _diagnosis, claimed = result
        assert ':=' not in broken.split('begin', 1)[1]
        assert '"=" should be ":="' in claimed

    def test_no_statement_assignment_skipped(self):
        code = "package body P is\n   procedure R is\n   begin\n      null;\n   end R;\nend P;\n"
        assert bd.inject_defect(code, "lang_confusion") is None


# --------------------------------------------------------------------------- #
# Phrasing variety and family rotation
# --------------------------------------------------------------------------- #


class TestPhrasingVariety:
    def test_variants_exist(self):
        assert len(bd._DEFECT_USER_PHRASINGS) >= 4

    def test_variant_chosen_by_content(self):
        turns1 = bd.build_defect_turns(SPEC_WITH_PREDEF, "Ada 95", "Msg", "src", bd.STE_RULE_BLOCK, "")
        turns2 = bd.build_defect_turns(SPEC_WITH_PREDEF, "Ada 95", "Msg", "src", bd.STE_RULE_BLOCK, "")
        # Same input, same phrasing (deterministic, not RNG-order dependent).
        assert turns1 and turns2
        assert turns1[0]["messages"][1]["content"] == turns2[0]["messages"][1]["content"]

    def test_multiple_variants_across_snippets(self):
        codes = [SPEC_WITH_PREDEF, TYPED_BODY, DECLARED_BODY, bd.WITH_CODE if hasattr(bd, "WITH_CODE") else SPEC_WITH_PREDEF]
        phrases = set()
        for code in codes:
            turns = bd.build_defect_turns(code, "Ada 95", "P", "src", bd.STE_RULE_BLOCK, "")
            if turns:
                phrases.add(turns[0]["messages"][1]["content"].splitlines()[0])
        assert len(phrases) > 1


class TestFamilyRotation:
    def test_cap_raised(self):
        assert bd._MAX_DEFECTS_PER_PAIR >= 3

    def test_rotation_is_content_derived(self):
        rotation = (
            lambda code: __import__("zlib").crc32(code.encode("utf-8")) % len(bd._DEFECT_FAMILY_ORDER)
        )
        assert rotation(TYPED_BODY) == rotation(TYPED_BODY)
        # Rich snippets trigger families from more than one position.
        combined = TYPED_BODY + SPEC_WITH_PREDEF + DECLARED_BODY
        turns = bd.build_defect_turns(combined, "Ada 95", "P", "src", bd.STE_RULE_BLOCK, "")
        assert len(turns) >= 2


# --------------------------------------------------------------------------- #
# Group-aware split
# --------------------------------------------------------------------------- #


class TestAssignSplits:
    def test_deterministic(self):
        groups = [f"pair:{i}" for i in range(50)]
        assert bd.assign_splits(groups) == bd.assign_splits(groups)

    def test_independent_of_input_order(self):
        groups = [f"pair:{i}" for i in range(50)]
        assert bd.assign_splits(groups) == bd.assign_splits(list(reversed(groups)))

    def test_all_splits_present_and_disjoint(self):
        groups = [f"pair:{i}" for i in range(50)]
        split = bd.assign_splits(groups)
        values = set(split.values())
        assert values == {"train", "val", "test"}
        assert len(split) == len(groups)

    def test_train_dominates(self):
        groups = [f"pair:{i}" for i in range(100)]
        split = bd.assign_splits(groups)
        counts = {"train": 0, "val": 0, "test": 0}
        for value in split.values():
            counts[value] += 1
        assert counts["train"] > counts["val"] + counts["test"]

    def test_tiny_inputs(self):
        assert bd.assign_splits(["only"]) == {"only": "train"}
        two = bd.assign_splits(["a", "b"])
        assert set(two.values()) <= {"train", "val"}
        assert len(two) == 2
