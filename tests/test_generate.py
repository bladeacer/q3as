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
# plan_batches
# --------------------------------------------------------------------------- #


class TestPlanBatches:
    def test_empty_input(self):
        assert gen.plan_batches([], 512, 0, 100_000) == []

    def test_no_budget_falls_back_to_singletons(self):
        # budget <= 0 means "could not measure VRAM": one prompt per call.
        assert gen.plan_batches([10, 20, 30], 5, 0, 0) == [[0], [1], [2]]

    def test_explicit_cap_wins_over_budget(self):
        # Budget would allow far more, but max_batch_size clamps it.
        assert gen.plan_batches([100, 100, 100, 100], 10, 2, 10_000_000) == [[0, 1], [2, 3]]

    def test_covered_indices_and_prompt_order_preserved(self):
        # Lengths deliberately unsorted: the plan must still return every
        # index exactly once so results stay in prompt order.
        lengths = [900, 10, 500, 20]
        batches = gen.plan_batches(lengths, 5, 0, 10_000_000)
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(len(lengths)))
        assert len(flat) == len(set(flat))

    def test_batches_respect_token_budget(self):
        # 3 prompts of 1000 tokens + 100 new tokens = 3300 <= 4000 budget.
        batches = gen.plan_batches([1000, 1000, 1000], 100, 0, 4000)
        assert [len(b) for b in batches] == [3]

    def test_long_prompt_splits_the_budget(self):
        # 2 prompts of 2000+100 = 4200 > 4000, so only one fits at a time.
        batches = gen.plan_batches([2000, 2000], 100, 0, 4000)
        assert [len(b) for b in batches] == [1, 1]

    def test_shortest_first_to_limit_padding(self):
        # Sorted by length, so the long prompt never pads a short one.
        batches = gen.plan_batches([2000, 10, 10], 100, 0, 2200)
        assert [len(b) for b in batches] == [2, 1]
        assert batches[0] == [1, 2]
        assert batches[1] == [0]


class TestKvBytesPerToken:
    def test_grouped_query_attention_shrinks_the_cache(self):
        cfg = types.SimpleNamespace(
            num_hidden_layers=36,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        # 36 layers * 2 (K,V) * 8 kv heads * 128 head_dim * 2 bytes.
        assert gen._kv_bytes_per_token(cfg) == 36 * 2 * 8 * 128 * 2

    def test_head_dim_derived_when_absent(self):
        cfg = types.SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
        )
        assert gen._kv_bytes_per_token(cfg) == 2 * 2 * 2 * 64 * 2

    def test_incomplete_config_is_unknown(self):
        assert gen._kv_bytes_per_token(types.SimpleNamespace()) == 0
        assert gen._kv_bytes_per_token(None) == 0


class TestIsOutOfMemory:
    def test_recognises_cuda_oom(self):
        assert gen._is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate"))

    def test_recognises_by_class_name(self):
        class OutOfMemoryError(Exception):
            pass

        assert gen._is_out_of_memory(OutOfMemoryError("no detail given"))

    def test_ordinary_error_is_not_oom(self):
        assert not gen._is_out_of_memory(ValueError("bad prompt"))


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
