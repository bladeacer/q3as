"""Tests for parse_docs.py and parse_ada_ast.py.

The structural scanner is always exercised. The libadalang path is optional
(it needs libadalang.so from `make ast-deps`), so its tests are skipped when
the bindings are not importable; when they are present they also pin the
node attributes the extraction relies on, which are easy to break across
libadalang versions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "processing_scripts"))

import parse_ada_ast as pa
import parse_docs as pd
import source_paths as sp

requires_lal = pytest.mark.skipif(
    not pa.HAS_LIBADALANG, reason="libadalang.so not built (run `make ast-deps`)"
)

# --------------------------------------------------------------------------- #
# parse_docs: markdown and rst splitting
# --------------------------------------------------------------------------- #

MD_DOC = """\
# Intro

Some intro text here that is long enough to matter for the section split test.

## Using Ada.Containers

Body about containers with a code block:

```ada
with Ada.Containers.Vectors;
procedure Show is begin null; end Show;
```

### Vectors

Vector details follow here in enough volume to pass the filter.
"""

RST_DOC = """\
Title One
=========

Body one with enough text to survive the minimum length filter easily.

Subsection
----------

Body two with more text to make it long enough for the filter too.
"""


class TestSplitMarkdown:
    def test_headings_with_hierarchy(self):
        sections = pd.split_markdown(MD_DOC)
        paths = [s["path"] for s in sections]
        assert ["Intro"] in paths
        assert ["Intro", "Using Ada.Containers"] in paths
        assert ["Intro", "Using Ada.Containers", "Vectors"] in paths

    def test_levels(self):
        sections = pd.split_markdown(MD_DOC)
        levels = [s["level"] for s in sections]
        assert levels == [1, 2, 3]

    def test_code_fence_stays_in_its_section(self):
        sections = pd.split_markdown(MD_DOC)
        container = next(s for s in sections if s["path"][-1] == "Using Ada.Containers")
        assert "Ada.Containers.Vectors" in container["body"]


class TestSplitRst:
    def test_underline_titles(self):
        sections = pd.split_rst(RST_DOC)
        assert [s["title"] for s in sections] == ["Title One", "Subsection"]
        assert sections[0]["level"] == 0
        assert sections[1]["path"] == ["Title One", "Subsection"]

    def test_body_separated(self):
        sections = pd.split_rst(RST_DOC)
        assert "Body one" in sections[0]["body"]
        assert "Body two" not in sections[0]["body"]


class TestBuildSectionTurn:
    def test_turn_shape_and_ste(self):
        sections = pd.split_markdown(MD_DOC)
        record = pd.build_section_turn(sections[1])
        assert record is not None
        roles = [m["role"] for m in record["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert record["meta"]["section"] == "Intro > Using Ada.Containers"
        # Code fences survive the STE cleaning byte-for-byte.
        assert "with Ada.Containers.Vectors;" in record["messages"][2]["content"]
        assert "```ada" in record["messages"][2]["content"]

    def test_tiny_sections_skipped(self):
        sections = pd.split_markdown(MD_DOC)
        tiny = dict(sections[-1])
        tiny["body"] = "Short."
        assert pd.build_section_turn(tiny) is None

    def test_dedup(self):
        sections = pd.split_markdown(MD_DOC) * 2
        records = pd.build_section_turns(sections)
        metas = [r["meta"]["section"] for r in records]
        assert len(metas) == len(set(metas))


# --------------------------------------------------------------------------- #
# parse_ada_ast: structural extraction
# --------------------------------------------------------------------------- #

SPEC_TEXT = """\
package Stack is

   procedure Push (X : Integer)
   with Pre => X /= 0;

   function Top return Natural;

end Stack;
"""

BODY_TEXT = """\
package body Stack is

   procedure Push (X : Integer) is
   begin
      null;
   end Push;

   function Top return Natural is
   begin
      return 0;
   end Top;

end Stack;
"""


class TestExtractSpecs:
    def test_subprograms_and_aspects(self):
        specs = pa.extract_spec_subprograms(SPEC_TEXT)
        names = [s["name"] for s in specs]
        assert names == ["Push", "Top"]
        push = specs[0]
        assert push["aspects"] == {"Pre": "X /= 0"}
        assert push["kind"] == "procedure"
        assert "with Pre => X /= 0" in push["text"]

    def test_generic_instantiation_skipped(self):
        text = "generic\npackage V is new Gen (T);\n"
        assert pa.extract_spec_subprograms(text) == []


class TestExtractBodies:
    def test_paired_by_indent(self):
        bodies = pa.extract_body_subprograms(BODY_TEXT)
        assert [b["name"] for b in bodies] == ["Push", "Top"]
        assert bodies[0]["text"].endswith("end Push;")
        assert bodies[1]["text"].endswith("end Top;")

    def test_expression_function(self):
        text = "package body P is\n   function F return Integer is (42);\nend P;\n"
        bodies = pa.extract_body_subprograms(text)
        assert [b["name"] for b in bodies] == ["F"]


class TestExtractTypes:
    def test_constrained_type_kept(self):
        text = "package P is\n   type Buffer_Index is range 0 .. 1024;\n   type Rec is record X : Integer; end record;\nend P;\n"
        types = pa.extract_type_decls(text)
        assert [t["name"] for t in types] == ["Buffer_Index"]
        assert "range 0 .. 1024" in types[0]["text"]


class TestPairAndTurns:
    def test_pairing_and_turn_kinds(self):
        specs = pa.extract_spec_subprograms(SPEC_TEXT)
        for spec in specs:
            spec["package"] = "Stack"
            spec["file"] = "stack.ads"
        bodies = pa.extract_body_subprograms(BODY_TEXT)
        for body in bodies:
            body["package"] = "Stack"
            body["file"] = "stack.adb"
        types = pa.extract_type_decls(SPEC_TEXT)
        records = pa.build_ada_ast_turns(specs, bodies, types)
        kinds = [r["meta"]["kind"] for r in records]
        assert kinds.count("ast_impl") == 2
        assert kinds.count("ast_contract") == 1  # only Push has aspects

        impl = next(r for r in records if r["meta"]["kind"] == "ast_impl")
        assert "```ada" in impl["messages"][0]["content"]  # spec in the question
        assert "end Push;" in impl["messages"][1]["content"]  # body in the answer

        contract = next(r for r in records if r["meta"]["kind"] == "ast_contract")
        assert "X /= 0" in contract["messages"][1]["content"]

    def test_missing_body_still_emits_contract_turn(self):
        specs = pa.extract_spec_subprograms(SPEC_TEXT)
        for spec in specs:
            spec["package"] = "Stack"
        records = pa.build_ada_ast_turns(specs, [], [])
        kinds = [r["meta"]["kind"] for r in records]
        assert "ast_impl" not in kinds
        assert "ast_contract" in kinds

    def test_records_are_json_serializable(self):
        specs = pa.extract_spec_subprograms(SPEC_TEXT)
        bodies = pa.extract_body_subprograms(BODY_TEXT)
        records = pa.build_ada_ast_turns(specs, bodies, [])
        for record in records:
            json.dumps(record)


def test_training_sources_exclude_ada_eval(monkeypatch, tmp_path: Path):
    fake_sources = {
        name: tmp_path / name
        for name in (sp.ADACOVEX, sp.ADA_CRDT, sp.ADA_83_TLALOC, sp.ADA_EVAL, sp.ADA_ALGORITHMS)
    }
    monkeypatch.setattr(sp, "resolve", fake_sources.get)

    assert [path.name for path in sp.default_code_dirs()] == [
        sp.ADACOVEX,
        sp.ADA_CRDT,
        sp.ADA_83_TLALOC,
        sp.ADA_ALGORITHMS,
    ]
    assert [path.name for path in sp.default_eval_sources()] == [
        sp.ADACOVEX,
        sp.ADA_CRDT,
        sp.ADA_EVAL,
        sp.ADA_ALGORITHMS,
    ]


# --------------------------------------------------------------------------- #
# libadalang extraction (optional; needs `make ast-deps`)
# --------------------------------------------------------------------------- #

LAL_SPEC = """\
package Demo is
   type Counter is range 1 .. 100;
   function Bump (C : in out Counter) return Counter
     with Pre => C < Counter'Last,
          Post => Bump'Result > C;
   procedure Reset (C : out Counter);
end Demo;
"""

LAL_BODY = """\
package body Demo is
   function Bump (C : in out Counter) return Counter is
   begin
      C := C + 1;
      return C;
   end Bump;
   procedure Reset (C : out Counter) is
   begin
      C := 1;
   end Reset;
end Demo;
"""


@requires_lal
class TestLibadalangExtraction:
    def test_spec_units_carry_package_kind_and_aspects(self, tmp_path: Path):
        path = tmp_path / "demo.ads"
        path.write_text(LAL_SPEC, encoding="utf-8")
        result = pa._lal_extract_file(str(path))

        assert result["valid"] is True
        specs = {unit["name"]: unit for unit in result["specs"]}
        assert set(specs) == {"Bump", "Reset"}
        assert specs["Bump"]["kind"] == "function"
        assert specs["Reset"]["kind"] == "procedure"
        # A package body is not a BasePackageDecl subclass; the package name
        # must still be recovered from a spec.
        assert specs["Bump"]["package"] == "Demo"
        # Aspects drive the contract-writing turns, so they must survive the
        # trip through the declaration text without the trailing semicolon.
        assert specs["Bump"]["aspects"] == {
            "Pre": "C < Counter'Last",
            "Post": "Bump'Result > C",
        }
        assert specs["Reset"]["aspects"] == {}
        assert {unit["name"] for unit in result["types"]} == {"Counter"}

    def test_bodies_are_separated_from_specs(self, tmp_path: Path):
        path = tmp_path / "demo.adb"
        path.write_text(LAL_BODY, encoding="utf-8")
        result = pa._lal_extract_file(str(path))

        assert result["valid"] is True
        assert result["specs"] == []
        assert {unit["name"] for unit in result["bodies"]} == {"Bump", "Reset"}
        assert result["bodies"][0]["package"] == "Demo"

    def test_syntax_error_marks_the_unit_invalid(self, tmp_path: Path):
        path = tmp_path / "broken.adb"
        path.write_text("procedure P is begin null; end\n", encoding="utf-8")
        result = pa._lal_extract_file(str(path))

        assert result["valid"] is False
        assert all(not unit["valid"] for unit in result["specs"] + result["bodies"])

    def test_worker_prefers_libadalang_over_the_scanner(self, tmp_path: Path):
        path = tmp_path / "demo.ads"
        path.write_text(LAL_SPEC, encoding="utf-8")
        task_result = pa.extract_file_task((str(path), "ads"))

        assert task_result["specs"], "libadalang path returned nothing"
        bump = next(unit for unit in task_result["specs"] if unit["name"] == "Bump")
        assert bump["aspects"]["Post"] == "Bump'Result > C"

    def test_lal_and_scanner_agree_on_subprogram_names(self, tmp_path: Path):
        """The exact AST and the regex scanner must find the same units."""
        path = tmp_path / "demo.ads"
        path.write_text(LAL_SPEC, encoding="utf-8")
        lal_names = {unit["name"] for unit in pa._lal_extract_file(str(path))["specs"]}
        scanner_names = {unit["name"] for unit in pa.extract_spec_subprograms(LAL_SPEC)}

        assert lal_names == scanner_names == {"Bump", "Reset"}

    def test_anonymous_access_type_function_is_skipped(self, tmp_path: Path):
        """An anonymous access type declares an unnamed function.

        It has no name to train on, so the extraction skips it instead of
        raising (which used to push the whole file onto the scanner).
        """
        path = tmp_path / "anon.adb"
        path.write_text(
            "procedure P is\n"
            "   type T is access all Integer'Range;\n"
            "   function Get (S : access all Integer'Range) return Integer;\n"
            "begin\n"
            "   null;\n"
            "end P;\n",
            encoding="utf-8",
        )
        result = pa._lal_extract_file(str(path))

        assert result["valid"] is True
        # The named declaration survives; the unnamed one never appears.
        assert {unit["name"] for unit in result["specs"]} == {"Get"}
