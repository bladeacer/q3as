"""Tests for build_dataset.py: STE sanitization, defect injection, guidance loading.

The defect-family expectations are verified against GNAT 14.2 behavior:
each family's expected compiler message matches what the real compiler
emits for the same broken snippet (see scripts/validate_defects.py for
the end-to-end compile check).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "processing_scripts"))

import build_dataset as bd  # noqa: E402


# --------------------------------------------------------------------------- #
# sanitize_prose / apply_word_swaps / has_style_violations
# --------------------------------------------------------------------------- #


class TestSanitizeProse:
    def test_removes_em_dash(self):
        out = bd.sanitize_prose("Check the contract — it fails.")
        assert "—" not in out
        assert out == "Check the contract. it fails." or out.startswith("Check the contract.")

    def test_paired_em_dash_becomes_appositive(self):
        out = bd.sanitize_prose("The reader — a user — confirms the run.")
        assert "—" not in out
        assert "The reader, a user, confirms the run." == out

    def test_removes_semicolons(self):
        out = bd.sanitize_prose("Build first; test after.")
        assert ";" not in out
        assert "Build first." in out

    def test_replaces_hedges(self):
        assert "may" not in bd.sanitize_prose("This may fail.")
        assert "might" not in bd.sanitize_prose("This might fail.")
        assert "can" in bd.sanitize_prose("This may fail.")

    def test_word_swaps(self):
        out = bd.sanitize_prose("In order to leverage the tool, utilize the flag.")
        assert "in order to" not in out.lower()
        assert "leverage" not in out.lower()
        assert "utilize" not in out.lower()
        assert "use" in out.lower()

    def test_deletes_filler_phrases(self):
        out = bd.sanitize_prose("It is worth noting that the build fails.")
        assert "worth noting" not in out

    def test_preserves_newlines(self):
        out = bd.sanitize_prose("First line; fixed.\n\nSecond line — done.")
        assert "\n\n" in out

    def test_preserves_identifiers(self):
        out = bd.sanitize_prose("Call Ada.Text_IO.Put_Line with Pre => X > 0.")
        assert "Ada.Text_IO.Put_Line" in out
        assert "Pre => X > 0" in out

    def test_collapses_multi_space(self):
        out = bd.sanitize_prose("too   many    spaces")
        assert "too many spaces" == out

    def test_no_artifact_periods(self):
        out = bd.sanitize_prose("End — next.")
        assert ".." not in out


class TestHasStyleViolations:
    def test_clean_text(self):
        assert bd.has_style_violations("Run the build. Read the log if it fails.") == []

    def test_em_dash_detected(self):
        assert "em-dash" in bd.has_style_violations("bad — text")

    def test_semicolon_detected(self):
        assert "semicolon" in bd.has_style_violations("bad; text")

    def test_hedge_detected(self):
        assert "hedge-modal" in bd.has_style_violations("This should work")

    def test_slop_detected(self):
        assert any(v.startswith("slop:") for v in bd.has_style_violations("leverage the API"))


# --------------------------------------------------------------------------- #
# strip_noisy_comments
# --------------------------------------------------------------------------- #


class TestStripNoisyComments:
    def test_cleans_comment_prose(self):
        code = "procedure P is begin null; end P;\n-- This is a robust solution — simply build it.\n"
        out = bd.strip_noisy_comments(code)
        assert "robust" not in out
        assert "—" not in out

    def test_never_touches_code(self):
        code = "   X : Integer := 5;  -- set X\n"
        out = bd.strip_noisy_comments(code)
        assert "X : Integer := 5;" in out

    def test_drops_decoration_comments(self):
        code = "--------\n-- real comment\n--------\nprocedure P is begin null; end P;\n"
        out = bd.strip_noisy_comments(code)
        assert "--------" not in out
        assert "real comment" in out

    def test_drops_comments_that_become_empty(self):
        code = "-- simply\nprocedure P is begin null; end P;\n"
        out = bd.strip_noisy_comments(code)
        assert "simply" not in out


# --------------------------------------------------------------------------- #
# inject_defect
# --------------------------------------------------------------------------- #

SPEC_BODY = """\
package Stack is
   procedure Push (X : Integer);
   procedure Pop (Index : in out Integer; X : out Integer);
end Stack;

package body Stack is
   procedure Push (X : Integer) is
   begin
      null;
   end Push;

   procedure Pop (Index : in out Integer; X : out Integer) is
   begin
      null;
   end Pop;
end Stack;
"""

WITH_CODE = """\
with Ada.Text_IO;
package body Outr is
   procedure Show is
   begin
      Ada.Text_IO.Put_Line ("x");
   end Show;
end Outr;
"""

USE_CODE = """\
with Ada.Text_IO;
use Ada.Text_IO;
package body Inr is
   procedure Show is
   begin
      Put_Line ("x");
   end Show;
end Inr;
"""

CONTRACT_CODE = """\
package Sp is
   procedure T (X : in out Integer) with Pre => X > 0;
end Sp;
"""


class TestInjectDefectSyntax:
    def test_package_is_removal(self):
        code = "package Foo is\n   X : Integer;\nend Foo;\n"
        result = bd.inject_defect(code, "syntax")
        assert result is not None
        broken, diagnosis, compiler = result
        assert "package Foo is" not in broken
        assert "package Foo" in broken
        # GNAT 14 reports the dropped 'is' as missing "is" (verified).
        assert 'missing "is"' in compiler
        assert "is" in diagnosis

    def test_semicolon_removal(self):
        code = "package Foo is\n   X : Integer := 1;\nend Foo;\n"
        result = bd.inject_defect(code, "syntax")
        assert result is not None
        broken, diagnosis, compiler = result
        # Either the package-is or the semicolon defect applies; the
        # compiler message must be one of the two verified GNAT errors.
        assert compiler in ('error: missing ";"', 'error: missing "is"')
        assert broken != code


class TestInjectDefectContext:
    def test_with_clause_removal(self):
        result = bd.inject_defect(WITH_CODE, "context")
        assert result is not None
        broken, diagnosis, compiler = result
        assert "with Ada.Text_IO;" not in broken
        assert "Ada.Text_IO.Put_Line" in broken
        assert compiler.startswith("error:")

    def test_no_context_defect_without_with(self):
        code = "package body P is begin null; end P;\n"
        assert bd.inject_defect(code, "context") is None


class TestInjectDefectVisibility:
    def test_use_removal_for_dot_free_use(self):
        result = bd.inject_defect(USE_CODE, "visibility")
        assert result is not None
        broken, diagnosis, compiler = result
        # The use clause is removed; the with clause stays (context family).
        assert "use Ada.Text_IO;" not in broken
        assert "with Ada.Text_IO;" in broken
        assert "Put_Line" in broken
        assert "not visible" in compiler

    def test_redundant_use_clause_is_skipped(self):
        # A dotted reference (Ada.Text_IO.Put_Line) does not need the use
        # clause, so removing it must not produce a defect.
        assert bd.inject_defect(WITH_CODE, "visibility") is None


class TestInjectDefectContract:
    def test_pre_misspelled(self):
        result = bd.inject_defect(CONTRACT_CODE, "contract")
        assert result is not None
        broken, diagnosis, compiler = result
        assert "Pres =>" in broken
        assert "Pre =>" not in broken
        assert "not a valid aspect identifier" in compiler
        assert "warning:" in compiler

    def test_no_contract_defect_without_aspects(self):
        assert bd.inject_defect(SPEC_BODY, "contract") is None


class TestInjectDefectMismatch:
    def test_parameter_dropped(self):
        result = bd.inject_defect(SPEC_BODY, "mismatch")
        assert result is not None
        broken, diagnosis, compiler = result
        assert "not type conformant" in compiler
        assert broken != SPEC_BODY

    def test_single_param_procedure_skipped(self):
        code = "package P is\n   procedure Q (X : Integer);\nend P;\n\npackage body P is\n   procedure Q (X : Integer) is begin null; end Q;\nend P;\n"
        assert bd.inject_defect(code, "mismatch") is None

    def test_body_edited_not_spec(self):
        result = bd.inject_defect(SPEC_BODY, "mismatch")
        assert result is not None
        broken, diagnosis, _compiler = result
        # The spec declaration keeps both parameters; the body loses one.
        assert "procedure Pop (Index : in out Integer; X : out Integer);" in broken
        body_part = broken.split("package body")[1]
        assert "Pop (Index" not in body_part
        assert "Index" in diagnosis  # the diagnosis names the dropped parameter


class TestBuildDefectTurns:
    def test_turns_have_three_messages(self):
        turns = bd.build_defect_turns(
            WITH_CODE, "Ada 95", "Outr", "test", bd.STE_RULE_BLOCK, ""
        )
        for turn in turns:
            roles = [m["role"] for m in turn["messages"]]
            assert roles == ["system", "user", "user"][:len(roles)] or roles == ["system", "user", "assistant"]

    def test_capped_per_pair(self):
        turns = bd.build_defect_turns(
            SPEC_BODY + WITH_CODE, "SPARK 2014", "Stack", "test", bd.STE_RULE_BLOCK, ""
        )
        assert len(turns) <= bd._MAX_DEFECTS_PER_PAIR

    def test_assistant_diagnosis_is_ste_clean(self):
        turns = bd.build_defect_turns(
            CONTRACT_CODE, "SPARK 2014", "Sp", "test", bd.STE_RULE_BLOCK, ""
        )
        for turn in turns:
            assistant = next(m for m in turn["messages"] if m["role"] == "assistant")
            prose = assistant["content"].split("```")[0]
            assert bd.has_style_violations(prose) == []

    def test_correct_code_present_in_answer(self):
        turns = bd.build_defect_turns(
            WITH_CODE, "Ada 95", "Outr", "test", bd.STE_RULE_BLOCK, ""
        )
        assert turns
        assistant = next(m for m in turns[0]["messages"] if m["role"] == "assistant")
        assert "with Ada.Text_IO;" in assistant["content"]


# --------------------------------------------------------------------------- #
# Glossary, system prompt, standard label
# --------------------------------------------------------------------------- #


class TestGlossaryAndPrompt:
    def test_glossary_terms(self):
        g = bd.build_technical_term_glossary()
        assert "precondition" in g
        assert "context clause" in g
        assert "SPARK_Mode" in g
        assert "—" not in g

    def test_system_prompt_carries_ste_rules(self):
        p = bd._compose_system_prompt("Ada 2012", bd.STE_RULE_BLOCK, "GLOSSARY")
        assert "ASD-STE100" in p
        assert "GLOSSARY" in p
        assert "No em-dashes" in p

    def test_standard_label_maps_unknown(self):
        assert bd._standard_label("Unknown") == "Ada"
        assert bd._standard_label("") == "Ada"
        assert bd._standard_label("Ada 2022") == "Ada 2022"


# --------------------------------------------------------------------------- #
# STE rule loading (falls back gracefully when sibling repos are absent)
# --------------------------------------------------------------------------- #


class TestSteRules:
    def test_returns_rule_block(self):
        rules = bd.load_simple_english_rules()
        # The rule block is the base; the word-map table (from the sibling
        # repo when present) may follow. Neither carries em-dashes outside
        # code spans.
        assert "ASD-STE100" in rules
        assert "No em-dashes" in rules

    def test_rule_block_mentions_key_rules(self):
        assert "active voice" in bd.STE_RULE_BLOCK
        assert "em-dashes" in bd.STE_RULE_BLOCK
        assert "One word, one meaning" in bd.STE_RULE_BLOCK


# --------------------------------------------------------------------------- #
# detect_ada_standard regression
# --------------------------------------------------------------------------- #


class TestDetectStandard:
    def test_spark_detected(self):
        code = "pragma SPARK_Mode (On);\nprocedure P with Pre => True is begin null; end P;\n"
        assert bd.detect_ada_standard(code) in ("SPARK 2014", "Ada 2012")

    def test_ada83_minimal_snippet(self):
        # A minimal snippet can fall through every threshold and return
        # Unknown (the scoring detector needs >= 2 Ada-83 keyword hits);
        # callers normalize this to 'Ada' via _standard_label.
        code = "procedure P is begin null; end P;\n"
        assert bd.detect_ada_standard(code) in ("Unknown", "Ada 83")
        assert bd._standard_label(bd.detect_ada_standard(code)) in ("Ada", "Ada 83")


# --------------------------------------------------------------------------- #
# Markdown STE cleaning (toolchain QA answers)
# --------------------------------------------------------------------------- #


class TestSteCleanMarkdown:
    def test_code_fences_untouched(self):
        md = "Run this; it works.\n\n```bash\ngnatprove -P x.gpr; -j0\n```\n\nDone — OK."
        out = bd._ste_clean_markdown(md)
        assert "gnatprove -P x.gpr; -j0" in out  # code keeps its semicolon
        assert "Run this." in out
        assert "—" not in out.split("```")[0]

    def test_bold_stripped(self):
        out = bd._ste_clean_markdown("**Never** run two instances.")
        assert "**" not in out
        assert "Never run two instances." in out

    def test_paragraph_breaks_preserved(self):
        out = bd._ste_clean_markdown("Para one; text.\n\nPara two — text.")
        assert "\n\n" in out

    @pytest.mark.parametrize("chunk", [
        "no code here; just prose — with dashes",
        "```ada\nX : Integer := 1;\n```\nprose after; the fence",
    ])
    def test_no_em_dash_in_prose_portion(self, chunk):
        out = bd._ste_clean_markdown(chunk)
        for prose_part in out.split("```"):
            if not prose_part.startswith("\nada") and "; -j" not in prose_part:
                assert "—" not in prose_part
