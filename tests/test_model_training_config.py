"""Tests that registry keys drive the 8 GB training configuration.

parse_args and setup_training_environment are pure (no model load), so the
chosen backend, packing, offload and LoRA settings can be checked without a
GPU. unsloth is stubbed because importing it costs over a GB of host RAM.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

if "unsloth" not in sys.modules:
    stub = types.ModuleType("unsloth")
    stub.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader=None)
    sys.modules["unsloth"] = stub

import train_unsloth as tr


def config_for(monkeypatch, *argv) -> dict:
    monkeypatch.setattr(sys, "argv", ["train_unsloth.py", *argv])
    args = tr.parse_args()
    return tr.setup_training_environment(args, has_val=True)


class TestRegistryDefaults:
    def test_default_is_the_unsloth_4_bit_qwen3_config(self, monkeypatch):
        config = config_for(monkeypatch)
        assert config["model_key"] == "qwen3-8b"
        assert config["backend"] == "unsloth"
        assert config["packing"] is True
        assert config["packing_strategy"] == "wrapped"
        assert config["offload_embedding"] is True
        assert config["max_seq_length"] == 1024
        assert config["lora_rank"] == 8
        assert config["lora_alpha"] == 16
        assert config["optim"] == "adamw_8bit"

    def test_k2_gets_the_rank_and_hf_backend_from_the_registry(self, monkeypatch):
        config = config_for(monkeypatch, "--model", "k2-horizon-7b")
        assert config["backend"] == "hf"
        assert config["lora_rank"] == 16
        assert config["lora_alpha"] == 32
        assert config["trust_remote_code"] is True
        assert config["offload_embedding"] is False
        assert config["packing"] is True

    def test_lfm_disables_packing_and_uses_its_own_targets(self, monkeypatch):
        config = config_for(monkeypatch, "--model", "lfm2.5-2.6b")
        assert config["backend"] == "hf"
        assert config["packing"] is False
        assert "in_proj" in config["target_modules"]
        assert "w1" in config["target_modules"]
        assert "lm_head" not in config["target_modules"]

    @pytest.mark.parametrize("key", ["qwen3-8b", "qwen3.5-4b", "spark-x2.6-4b",
                                     "k2-horizon-7b", "lfm2.5-2.6b"])
    def test_every_registered_model_configures(self, monkeypatch, key):
        config = config_for(monkeypatch, "--model", key)
        assert config["model_key"] == key
        assert config["max_seq_length"] in (1024, 2048)
        assert config["target_modules"]


class TestExplicitOverrides:
    def test_cli_flags_win_over_the_registry(self, monkeypatch):
        config = config_for(
            monkeypatch, "--model", "k2-horizon-7b", "--backend", "unsloth",
            "--no-packing", "--packing-strategy", "bfd", "--no-offload-embedding",
            "--no-trust-remote-code", "--lora-rank", "4", "--max-seq-length", "512",
        )
        assert config["backend"] == "unsloth"
        assert config["packing"] is False
        assert config["packing_strategy"] == "bfd"
        assert config["offload_embedding"] is False
        assert config["trust_remote_code"] is False
        assert config["lora_rank"] == 4
        assert config["max_seq_length"] == 512

    def test_explicit_model_name_overrides_the_repo(self, monkeypatch):
        config = config_for(monkeypatch, "--model-name", "org/custom-4b")
        assert config["model_name"] == "org/custom-4b"

    def test_trainer_only_settings_stay_out_of_sft_config(self, monkeypatch):
        # SFTConfig rejects unknown fields. main() filters these trainer-only
        # keys out; keep the list pinned here so a new one is not forgotten.
        config = config_for(monkeypatch)
        # main() adds these two private keys just before the model load.
        config["_load_path"] = "models/example"
        config["_spec"] = object()
        trainer_only = {
            "model_name", "model_key", "lora_rank", "lora_alpha", "backend",
            "offload_embedding", "trust_remote_code", "target_modules",
            "max_seq_length", "_load_path", "_spec",
        }
        assert trainer_only <= config.keys()
        assert {"packing", "packing_strategy"} <= config.keys()

        from trl import SFTConfig

        accepted = set(SFTConfig.__dataclass_fields__)
        leaked = set(config) - trainer_only - accepted
        assert not leaked, f"unknown SFTConfig fields: {sorted(leaked)}"
