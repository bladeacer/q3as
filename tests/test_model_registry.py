"""Tests for the base-model registry.

The registry feeds the downloader, trainer, and Makefile, so a typo in a repo
id or a missing VRAM setting would be copied into three places. These tests
check the invariants that matter without touching the network or a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import model_registry as mr


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
    monkeypatch.delenv("Q3AS_MODEL", raising=False)


class TestRegistryContents:
    def test_expected_keys_are_registered(self):
        assert set(mr.MODELS) == {
            "qwen3-8b", "qwen3.5-4b", "spark-x2.6-4b",
            "k2-horizon-7b", "lfm2.5-2.6b",
        }

    @pytest.mark.parametrize("key", sorted(mr.MODELS))
    def test_every_spec_is_complete_and_vram_safe(self, key):
        spec = mr.get_model(key)
        assert spec.official_repo and "/" in spec.official_repo
        assert spec.train_repo and "/" in spec.train_repo
        assert spec.official_url.startswith("https://huggingface.co/")
        assert spec.train_url.startswith("https://huggingface.co/")
        assert spec.cache_dir.startswith("models/")
        assert spec.local_path == Path(spec.cache_dir)
        assert spec.architecture
        assert spec.backend in {"unsloth", "hf"}
        assert spec.unsloth_support in {"native", "partial", "generic", "unlikely"}
        assert 0 < spec.max_seq_length <= 2048
        assert 0 < spec.lora_rank <= 16
        assert spec.lora_alpha > 0
        assert spec.target_modules
        assert "lm_head" not in spec.target_modules
        assert "embed_tokens" not in spec.target_modules
        # Nothing has been run on the reference machine with these settings.
        assert spec.tested is False

    def test_official_urls_match_the_repo_ids(self):
        for spec in mr.list_models():
            assert spec.official_url == f"https://huggingface.co/{spec.official_repo}"
            assert spec.train_url == f"https://huggingface.co/{spec.train_repo}"

    def test_default_key_is_qwen3_8b(self):
        assert mr.DEFAULT_MODEL_KEY == "qwen3-8b"
        assert mr.default_key() == "qwen3-8b"

    def test_env_can_change_the_default(self, monkeypatch):
        monkeypatch.setenv("Q3AS_MODEL", "k2-horizon-7b")
        assert mr.default_key() == "k2-horizon-7b"
        assert mr.get_model().key == "k2-horizon-7b"

    def test_unknown_key_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="unknown model 'nope'"):
            mr.get_model("nope")
        with pytest.raises(ValueError, match="qwen3-8b"):
            mr.get_model("nope")


class TestResolution:
    def test_resolve_uses_the_registered_repo_and_cache_dir(self):
        spec = mr.resolve_model("qwen3.5-4b")
        assert spec.train_repo == "unsloth/Qwen3.5-4B-Base"
        assert spec.cache_dir == "models/qwen3.5-4b"

    def test_repo_override_does_not_mutate_the_registry(self):
        spec = mr.resolve_model("qwen3-8b", repo="org/custom")
        assert spec.train_repo == "org/custom"
        assert mr.MODELS["qwen3-8b"].train_repo == "unsloth/Qwen3-8B-unsloth-bnb-4bit"

    def test_cache_dir_override_is_stringified(self, tmp_path):
        spec = mr.resolve_model("spark-x2.6-4b", cache_dir=tmp_path)
        assert spec.cache_dir == str(tmp_path)
        assert spec.local_path == tmp_path


class TestCli:
    def test_keys_are_space_separated(self, capsys):
        assert mr.main(["--keys"]) == 0
        keys = capsys.readouterr().out.strip().split()
        assert keys == list(mr.MODELS)

    def test_default_and_field_flags(self, capsys):
        assert mr.main(["--default"]) == 0
        assert capsys.readouterr().out.strip() == "qwen3-8b"
        assert mr.main(["lfm2.5-2.6b", "--field", "backend"]) == 0
        assert capsys.readouterr().out.strip() == "hf"

    def test_table_has_every_key_and_no_tested_run(self, capsys):
        assert mr.main(["--table"]) == 0
        table = capsys.readouterr().out
        for key in mr.MODELS:
            assert f"`{key}`" in table
        assert table.count("| No |") >= len(mr.MODELS)

    def test_json_is_parseable(self, capsys):
        assert mr.main(["--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert [entry["key"] for entry in payload] == list(mr.MODELS)
        assert all(entry["official_url"].startswith("https://") for entry in payload)

    def test_bad_key_exits_two(self, capsys):
        assert mr.main(["nope"]) == 2
        assert "unknown model" in capsys.readouterr().err
