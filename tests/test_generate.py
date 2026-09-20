"""Tests for eval/generate.py: prompt construction and reply parsing.

The generation pipeline feeds the model the task plus the full base project
tree, then parses the reply into project-file overlays. These tests pin the
prompt contract (sources shown, target called out, non-source files
protected) and the parser's fallbacks (multi-file entries, single fenced
block, raw text) plus its path-safety rules.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import generate as gen

# --------------------------------------------------------------------------- #
# build_user_prompt
# --------------------------------------------------------------------------- #


def make_sample(**overrides):
    sample = {
        "prompt": "Please make Absolute_Value provable.",
        "location": {"path": "src/integer_utils.ads", "subprogram_name": "Absolute_Value"},
        "sources_text": {
            Path("main.gpr"): "project Main is end Main;",
            Path("src/integer_utils.ads"): "package Integer_Utils is end Integer_Utils;",
        },
    }
    sample.update(overrides)
    return sample


class TestBuildUserPrompt:
    def test_includes_task_and_all_sources(self):
        prompt = gen.build_user_prompt(make_sample())
        assert "Please make Absolute_Value provable." in prompt
        assert "File: main.gpr" in prompt
        assert "File: src/integer_utils.ads" in prompt
        assert "project Main is end Main;" in prompt

    def test_calls_out_target_file(self):
        prompt = gen.build_user_prompt(make_sample())
        assert "Always include the file src/integer_utils.ads in the reply." in prompt

    def test_restricts_changes_to_target_directory(self):
        prompt = gen.build_user_prompt(make_sample())
        assert "Only files in src may change." in prompt

    def test_target_in_project_root_wording(self):
        sample = make_sample(location={"path": "main.adb"})
        prompt = gen.build_user_prompt(sample)
        assert "Only files in the project root may change." in prompt

    def test_missing_location_falls_back_to_generated_adb(self):
        sample = make_sample(location={})
        prompt = gen.build_user_prompt(sample)
        assert "Always include the file generated.adb in the reply." in prompt

    def test_reply_format_instruction_present(self):
        prompt = gen.build_user_prompt(make_sample())
        assert 'write a line "File: <path>"' in prompt
        assert "```ada" in prompt

    def test_sources_sorted_deterministically(self):
        sample = make_sample(sources_text={
            Path("b.adb"): "b",
            Path("a.ads"): "a",
        })
        prompt = gen.build_user_prompt(sample)
        assert prompt.index("File: a.ads") < prompt.index("File: b.adb")


# --------------------------------------------------------------------------- #
# parse_generated_files
# --------------------------------------------------------------------------- #


MULTI_FILE_REPLY = """\
File: src/foo.ads
```ada
package Foo is
   procedure P (X : Integer)
   with Pre => X > 0;
end Foo;
```

File: src/foo.adb
```ada
package body Foo is
   procedure P (X : Integer) is
   begin
      null;
   end P;
end Foo;
```"""


class TestParseGeneratedFiles:
    def test_multi_file_reply(self):
        files = gen.parse_generated_files(MULTI_FILE_REPLY, Path("src/foo.ads"))
        assert set(files) == {Path("src/foo.ads"), Path("src/foo.adb")}
        assert "Pre => X > 0" in files[Path("src/foo.ads")]

    def test_entries_outside_target_dir_dropped(self):
        files = gen.parse_generated_files(MULTI_FILE_REPLY, Path("src/other.ads"))
        # foo.ads/foo.adb are not next to other.ads... they are (src/), so
        # build a reply whose only entries live elsewhere.
        reply = MULTI_FILE_REPLY + "\n\nFile: main.gpr\n```ada\nproject M is end M;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("main.gpr") not in files

    def test_single_fenced_block_falls_back_to_target(self):
        reply = "Here is the fix:\n```ada\npackage Foo is end Foo;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert set(files) == {Path("src/foo.ads")}
        assert "package Foo is end Foo;" in files[Path("src/foo.ads")]

    def test_raw_text_falls_back_to_target(self):
        reply = "The warning says R might be uninitialized."
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert set(files) == {Path("src/foo.ads")}
        assert "uninitialized" in files[Path("src/foo.ads")]

    def test_trailing_punctuation_stripped_from_path(self):
        reply = "File: src/foo.ads:\n```ada\npackage Foo is end Foo;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("src/foo.ads") in files

    def test_absolute_path_rejected(self):
        reply = "File: /etc/passwd\n```ada\nx\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("/etc/passwd") not in files
        # The reply still has a fenced block, so it lands at the target path.
        assert Path("src/foo.ads") in files

    def test_parent_traversal_rejected(self):
        reply = "File: ../evil.ads\n```ada\nx\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("../evil.ads") not in files

    def test_non_source_suffix_rejected(self):
        reply = "File: src/notes.txt\n```ada\nx\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("src/notes.txt") not in files

    def test_empty_reply_yields_empty_map(self):
        assert gen.parse_generated_files("", Path("src/foo.ads")) == {}
        assert gen.parse_generated_files("   \n  ", Path("src/foo.ads")) == {}


# --------------------------------------------------------------------------- #
# _safe_source_path
# --------------------------------------------------------------------------- #


class TestSafeSourcePath:
    @pytest.mark.parametrize("raw,expected", [
        ("src/foo.ads", "src/foo.ads"),
        (" main.gpr ", "main.gpr"),
        ("`src/foo.adb`", "src/foo.adb"),
    ])
    def test_accepts(self, raw, expected):
        assert gen._safe_source_path(raw) == Path(expected)

    @pytest.mark.parametrize("raw", [
        "/abs/foo.ads",
        "../foo.ads",
        "src/../../foo.ads",
        "notes.txt",
        "src/foo.adbx",
    ])
    def test_rejects(self, raw):
        assert gen._safe_source_path(raw) is None


# --------------------------------------------------------------------------- #
# set_generation_seed (transformers stubbed out to keep the test fast)
# --------------------------------------------------------------------------- #


class TestSetGenerationSeed:
    def test_seeds_python_random_and_calls_hf(self, monkeypatch):
        stub = types.ModuleType("transformers")
        calls = []
        stub.set_seed = lambda s: calls.append(s)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", stub)
        gen.set_generation_seed(42)
        assert calls == [42]

    def test_is_deterministic_call_order(self, monkeypatch):
        stub = types.ModuleType("transformers")
        calls = []
        stub.set_seed = lambda s: calls.append(s)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", stub)
        gen.set_generation_seed(7)
        gen.set_generation_seed(7)
        assert calls == [7, 7]
