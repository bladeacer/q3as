"""Tests for the registry-aware downloader.

The old completeness check assumed every repo ships configuration.json,
special_tokens_map.json and a sharded index. Unsloth's 4-bit Qwen3-8B has no
configuration.json and Qwen3-4B's quant is one safetensors file, so the check
had to follow the repo's actual layout. No network access here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import download_model as dm


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
    monkeypatch.delenv("Q3AS_MODEL", raising=False)


def _meta(cache: Path) -> None:
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (cache / name).write_text("{}", encoding="utf-8")


def _sharded(cache: Path, names: tuple[str, ...] = ("model-00001-of-00001.safetensors",)) -> None:
    (cache / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"layer{i}.weight": name for i, name in enumerate(names)}}),
        encoding="utf-8",
    )
    for name in names:
        (cache / name).write_bytes(b"st")


class TestCompleteness:
    def test_unsloth_layout_without_configuration_json_is_complete(self, tmp_path):
        cache = tmp_path / "qwen3-8b"
        cache.mkdir()
        _meta(cache)
        _sharded(cache, ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"))
        assert dm.is_model_present(cache)

    def test_single_safetensors_layout_is_complete(self, tmp_path):
        cache = tmp_path / "qwen3-4b"
        cache.mkdir()
        _meta(cache)
        (cache / "model.safetensors").write_bytes(b"weights")
        assert dm.is_model_present(cache)
        assert dm.weight_files(cache) == ["model.safetensors"]

    def test_numbered_shards_without_index_are_complete(self, tmp_path):
        cache = tmp_path / "shards"
        cache.mkdir()
        _meta(cache)
        for name in ("model-00001-of-00003.safetensors", "model-00003-of-00003.safetensors"):
            (cache / name).write_bytes(b"w")
        assert dm.weight_files(cache) == [
            "model-00001-of-00003.safetensors", "model-00003-of-00003.safetensors",
        ]
        assert dm.is_model_present(cache)

    def test_missing_index_shard_is_incomplete_even_with_stray_files(self, tmp_path):
        cache = tmp_path / "partial"
        cache.mkdir()
        _meta(cache)
        (cache / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {
                "layer0.weight": "model-00001-of-00002.safetensors",
                "layer1.weight": "model-00002-of-00002.safetensors",
            }}),
            encoding="utf-8",
        )
        (cache / "model-00001-of-00002.safetensors").write_bytes(b"w")
        assert dm.weight_files(cache) == []
        assert not dm.is_model_present(cache)

    @pytest.mark.parametrize("missing", ["config.json", "tokenizer.json", "tokenizer_config.json"])
    def test_missing_metadata_is_incomplete(self, tmp_path, missing):
        cache = tmp_path / "missing"
        cache.mkdir()
        _meta(cache)
        (cache / missing).unlink()
        assert not dm.is_model_present(cache)

    def test_absent_directory_is_incomplete(self, tmp_path):
        assert not dm.is_model_present(tmp_path / "absent")

    def test_corrupt_index_is_incomplete(self, tmp_path):
        cache = tmp_path / "corrupt"
        cache.mkdir()
        _meta(cache)
        (cache / "model.safetensors.index.json").write_text("{bad", encoding="utf-8")
        assert not dm.is_model_present(cache)


class TestArgResolution:
    def _args(self, monkeypatch, *argv):
        monkeypatch.setattr(sys, "argv", ["download_model.py", *argv])
        return dm.resolve_args(dm.parse_args())

    def test_default_uses_the_unsloth_quant_and_its_cache_dir(self, monkeypatch):
        args = self._args(monkeypatch)
        assert args.model_name == "unsloth/Qwen3-8B-unsloth-bnb-4bit"
        assert args.cache_dir == Path("models/qwen3-8b")
        assert args.model_key == "qwen3-8b"

    def test_registry_key_selects_repo_and_cache(self, monkeypatch):
        args = self._args(monkeypatch, "--model", "k2-horizon-7b")
        assert args.model_name == "IFM/K2-Horizon-7B"
        assert args.cache_dir == Path("models/k2-horizon-7b")

    def test_explicit_repo_gets_a_derived_cache_dir(self, monkeypatch):
        args = self._args(monkeypatch, "--model-name", "org/My-Model")
        assert args.model_name == "org/My-Model"
        assert args.cache_dir == Path("models/my-model")

    def test_explicit_cache_dir_is_kept(self, monkeypatch, tmp_path):
        args = self._args(monkeypatch, "--model-name", "org/My-Model",
                          "--cache-dir", str(tmp_path))
        assert args.cache_dir == tmp_path

    def test_check_only_never_sets_force(self, monkeypatch):
        args = self._args(monkeypatch, "--check-only")
        assert args.check_only is True
        assert args.force is False
