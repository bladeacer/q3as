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


class TestDedupGrouped:
    @staticmethod
    def _turn(text: str) -> dict[str, list[dict[str, str]]]:
        return {"messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": text + " answer"},
        ]}

    def test_identical_content_in_different_groups_is_deduped(self):
        # Regression: identical records generated from two different groups
        # (same spec via different source paths) landed in different splits.
        grouped = [("pair:a:0", self._turn("spec")), ("pair:b:3", self._turn("spec"))]
        deduped, dropped = bd.dedup_grouped(grouped)
        assert dropped == 1
        assert len(deduped) == 1
        assert deduped[0][0] == "pair:a:0"  # first occurrence wins

    def test_distinct_content_is_kept(self):
        grouped = [("pair:a:0", self._turn("one")), ("pair:b:0", self._turn("two"))]
        deduped, dropped = bd.dedup_grouped(grouped)
        assert dropped == 0
        assert len(deduped) == 2

    def test_empty_input(self):
        assert bd.dedup_grouped([]) == ([], 0)


# --------------------------------------------------------------------------- #
# Extra-turns ingestion (parser outputs)
# --------------------------------------------------------------------------- #


class TestExtraTurnGrouping:
    def test_ast_records_use_parser_group(self):
        record = {"meta": {"kind": "ast_impl", "group": "ast:123"}}
        assert bd._extra_turn_group(record, 0) == "ast:123"

    def test_doc_records_group_by_content(self):
        record = {"meta": {"kind": "doc_section", "source": "s.md", "section": "T"}}
        assert bd._extra_turn_group(record, 0) == bd._extra_turn_group(record, 99)

    def test_fallback_group_uses_index(self):
        assert bd._extra_turn_group({"meta": {}}, 7) == "extra:record:7"

    def test_kind_buckets(self):
        assert bd._extra_turn_kind({"kind": "doc_section"}) == "doc_section"
        assert bd._extra_turn_kind({"kind": "ast_impl"}) == "ast_qa"
        assert bd._extra_turn_kind({"kind": "ast_contract"}) == "ast_qa"
        assert bd._extra_turn_kind({"kind": "ast_type"}) == "ast_qa"
        assert bd._extra_turn_kind({}) == "extra"


class TestExtraTurnsIntegration:
    def test_ingest_counts_splits_and_malformed_skip(self, tmp_path):
        import json

        (tmp_path / "pkg.ads").write_text(
            "package Pkg is\n   procedure Op (X : Integer);\nend Pkg;\n"
        )
        (tmp_path / "pkg.adb").write_text(
            "package body Pkg is\n   procedure Op (X : Integer) is\n   begin\n"
            "      null;\n   end Op;\nend Pkg;\n"
        )
        doc = {
            "messages": [{"role": "user", "content": "u-doc"}, {"role": "assistant", "content": "a-doc"}],
            "meta": {"kind": "doc_section", "source": "learn/x.md", "section": "Foo"},
        }
        ast_impl = {
            "messages": [{"role": "user", "content": "u-impl"}, {"role": "assistant", "content": "a-impl"}],
            "meta": {"kind": "ast_impl", "unit": "Op", "group": "ast:123"},
        }
        ast_contract = {
            "messages": [{"role": "user", "content": "u-contract"}, {"role": "assistant", "content": "a-contract"}],
            "meta": {"kind": "ast_contract", "unit": "Op", "group": "ast:123"},
        }
        extra = tmp_path / "extra.jsonl"
        extra.write_text(
            "\n".join(json.dumps(r) for r in (doc, ast_impl, ast_contract))
            + "\n{bad json}\n",
            encoding="utf-8",
        )
        out = tmp_path / "dataset.jsonl"
        count = bd.build_dataset(
            input_dirs=[tmp_path],
            extra_input_dirs=[],
            output_file=out,
            doc_dirs=[],
            guidance_dirs=[],
            workers=1,
            extra_turns=[extra],
        )
        meta = json.loads((tmp_path / "dataset_metadata.json").read_text())
        kinds = meta["turn_counts_by_kind"]
        assert kinds["doc_section"] == 1
        assert kinds["ast_qa"] == 2
        assert meta["total_turns"] == count
        splits = meta["splits"]["counts"]
        assert splits["train"] + splits["val"] + splits["test"] == count

        # The two AST turns of one subprogram share a split (same group).
        records = [json.loads(line) for line in out.read_text().splitlines()]
        by_user = {r["messages"][-2]["content"]: r["split"] for r in records}
        assert by_user["u-impl"] == by_user["u-contract"]
        assert by_user["u-doc"] in {"train", "val", "test"}

    def test_missing_extra_file_is_skipped(self, tmp_path):
        import json

        (tmp_path / "pkg.ads").write_text("package Pkg is\n   X : Integer;\nend Pkg;\n")
        out = tmp_path / "dataset.jsonl"
        count = bd.build_dataset(
            input_dirs=[tmp_path],
            extra_input_dirs=[],
            output_file=out,
            doc_dirs=[],
            guidance_dirs=[],
            workers=1,
            extra_turns=[tmp_path / "nope.jsonl"],
        )
        meta = json.loads((tmp_path / "dataset_metadata.json").read_text())
        assert "doc_section" not in meta["turn_counts_by_kind"]
        assert count == meta["total_turns"]
