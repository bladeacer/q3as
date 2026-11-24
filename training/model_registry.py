"""model_registry.py - one place that describes every base model q3as can train.

The 8 GB training setup used to be hard-wired to one local directory and one
Hugging Face id. That made a second model a scavenger hunt through the
Makefile, the downloader, the trainer, and the eval scripts, and it made the
"does Unsloth actually support this?" question impossible to answer from the
repository. This module is that answer: each entry carries the official source
repo, the repo training loads, where it is cached, and the settings that keep
the run inside the VRAM budget.

Adding a model is one ``ModelSpec`` below plus tests. Nothing else in the
pipeline should hard-code a Hugging Face id or a ``models/<name>`` path.

The ``tested`` field is deliberately False for every entry: the support column
describes what the loader is expected to do, not a claim that a run finished.
Set it only after a real QLoRA run on the target hardware.

Usage:
    uv run python training/model_registry.py --table
    uv run python training/model_registry.py --keys
    uv run python training/model_registry.py qwen3-8b --field cache_dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Dense decoder projections. Applicable to Qwen3/Qwen3.5 and to the custom-code
# models until their modeling_*.py is inspected; it is an explicit list rather
# than "all-linear" because training lm_head on an 8 GB card does not fit.
STANDARD_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# LFM2 hybrids alternate convolution and attention blocks. The attention block
# uses q/k/v/out_proj, the convolution block uses in_proj/out_proj, and the MLP
# uses w1/w2/w3. Verified against transformers' modeling_lfm2.py; names in a
# future revision must be re-checked before a run.
LFM2_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
    "w1", "w2", "w3",
)

# Spark2_5 fuses Q/K/V into one projection and adds a headwise attention gate.
# Names read from the repo's modeling_spark.py, not assumed from Qwen.
SPARK_TARGET_MODULES: tuple[str, ...] = (
    "q_k_v_proj", "g_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)

DEFAULT_MODEL_KEY = "qwen3-8b"


@dataclass(frozen=True)
class ModelSpec:
    """Everything the pipeline needs to know about one candidate base model."""

    key: str
    name: str
    family: str
    architecture: str
    official_repo: str
    train_repo: str
    cache_dir: str
    backend: str  # "unsloth" (FastLanguageModel) or "hf" (transformers + PEFT)
    unsloth_support: str  # "native" | "partial" | "generic" | "unlikely"
    prequantized_4bit: bool
    trust_remote_code: bool
    max_seq_length: int
    lora_rank: int
    lora_alpha: int
    target_modules: tuple[str, ...]
    packing: bool
    offload_embedding: bool
    tested: bool
    notes: str

    @property
    def official_url(self) -> str:
        return f"https://huggingface.co/{self.official_repo}"

    @property
    def train_url(self) -> str:
        return f"https://huggingface.co/{self.train_repo}"

    @property
    def local_path(self) -> Path:
        return Path(self.cache_dir)


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="qwen3-8b",
            name="Qwen 3 8B",
            family="Qwen3",
            architecture="qwen3 / Qwen3ForCausalLM",
            official_repo="Qwen/Qwen3-8B",
            # The training base is Unsloth's pre-quantized 4-bit copy, never the
            # 16 GB BF16 original. Unsloth is no longer framework-only here.
            train_repo="unsloth/Qwen3-8B-unsloth-bnb-4bit",
            cache_dir="models/qwen3-8b",
            backend="unsloth",
            unsloth_support="native",
            prequantized_4bit=True,
            trust_remote_code=False,
            max_seq_length=1024,
            lora_rank=8,
            lora_alpha=16,
            target_modules=STANDARD_TARGET_MODULES,
            packing=True,
            offload_embedding=True,
            tested=False,
            notes=(
                "First-class Unsloth model. The 4-bit base is about 7.5 GB on "
                "disk and needs roughly 5-6 GB of VRAM in QLoRA; never load "
                "Qwen/Qwen3-8B in BF16 on the 8 GB card."
            ),
        ),
        ModelSpec(
            key="qwen3.5-4b",
            name="Qwen 3.5 4B",
            family="Qwen3.5",
            architecture="qwen3_5 / Qwen3_5ForConditionalGeneration",
            official_repo="Qwen/Qwen3.5-4B-Base",
            # Unsloth mirrors Qwen3.5-4B-Base in 16-bit, but no bnb-4bit
            # training repo existed at the time of writing. on-the-fly 4-bit.
            train_repo="unsloth/Qwen3.5-4B-Base",
            cache_dir="models/qwen3.5-4b",
            backend="unsloth",
            unsloth_support="partial",
            prequantized_4bit=False,
            trust_remote_code=False,
            max_seq_length=1024,
            lora_rank=8,
            lora_alpha=16,
            target_modules=STANDARD_TARGET_MODULES,
            packing=True,
            offload_embedding=True,
            tested=False,
            notes=(
                "Hybrid linear/full attention plus a vision tower; use the "
                "Base checkpoint for text training. No verified Unsloth "
                "bnb-4bit base, so 4-bit happens at load time. Check module "
                "names and use --backend hf if FastLanguageModel rejects it."
            ),
        ),
        ModelSpec(
            key="spark-x2.6-4b",
            name="Spark-X2.6-4B",
            family="Spark2.5",
            architecture="spark2_5 / Spark2_5ForCausalLM (custom code)",
            official_repo="XHToken/Spark-X2.6-4B",
            train_repo="XHToken/Spark-X2.6-4B",
            cache_dir="models/spark-x2.6-4b",
            backend="hf",
            unsloth_support="generic",
            prequantized_4bit=False,
            trust_remote_code=True,
            max_seq_length=1024,
            lora_rank=8,
            lora_alpha=16,
            target_modules=SPARK_TARGET_MODULES,
            packing=True,
            offload_embedding=False,
            tested=True,
            notes=(
                "Smoke-tested: 20 steps on a 300-record sample, hf+PEFT 4-bit, "
                "--no-bf16. Ships configuration_spark.py and modeling_spark.py; "
                "needs trust_remote_code=True. No Unsloth quant. Its custom "
                "code needs two compatibility shims (see train_unsloth.py)."
            ),
        ),
        ModelSpec(
            key="k2-horizon-7b",
            name="K2 Horizon 7B",
            family="K2-Horizon",
            architecture="k2_horizon / K2HorizonForCausalLM (custom code)",
            official_repo="IFM/K2-Horizon-7B",
            train_repo="IFM/K2-Horizon-7B",
            cache_dir="models/k2-horizon-7b",
            backend="hf",
            unsloth_support="generic",
            prequantized_4bit=False,
            trust_remote_code=True,
            max_seq_length=1024,
            lora_rank=16,
            lora_alpha=32,
            target_modules=STANDARD_TARGET_MODULES,
            packing=True,
            offload_embedding=False,
            tested=False,
            notes=(
                "Custom code and 9.0B actual parameters despite the 7B name. "
                "Keep max_seq_length at 1024, adamw_8bit on, and LoRA rank at "
                "or below 16; the native 512K context does not fit 8 GB."
            ),
        ),
        ModelSpec(
            key="lfm2.5-2.6b",
            name="LFM 2.5 2.6B",
            family="LFM2.5",
            architecture="lfm2 / Lfm2ForCausalLM (hybrid conv + attention)",
            official_repo="LiquidAI/LFM2.5-2.6B",
            train_repo="LiquidAI/LFM2.5-2.6B",
            cache_dir="models/lfm2.5-2.6b",
            backend="hf",
            unsloth_support="unlikely",
            prequantized_4bit=False,
            trust_remote_code=False,
            max_seq_length=1024,
            lora_rank=8,
            lora_alpha=16,
            target_modules=LFM2_TARGET_MODULES,
            packing=False,
            offload_embedding=False,
            tested=False,
            notes=(
                "Hybrid recurrence/convolution layers: Unsloth's fused MHA "
                "kernels do not apply, so use the HF + PEFT backend. Packing "
                "is off until document boundaries are verified for conv "
                "layers. License is LFM 1.0, not Apache-2.0."
            ),
        ),
    )
}


def get_model(key: str | None = None) -> ModelSpec:
    """Return the spec for *key*, or the default when it is None."""
    key = key or default_key()
    try:
        return MODELS[key]
    except KeyError:
        known = ", ".join(MODELS)
        raise ValueError(f"unknown model '{key}'; known models: {known}") from None


def default_key() -> str:
    """The model key from Q3AS_MODEL, or the shipped default."""
    return os.getenv("Q3AS_MODEL") or DEFAULT_MODEL_KEY


def resolve_model(
    key: str | None = None,
    repo: str | None = None,
    cache_dir: str | Path | None = None,
) -> ModelSpec:
    """Resolve a spec with optional explicit repo/cache-dir overrides."""
    spec = get_model(key)
    if repo:
        spec = replace(spec, train_repo=repo)
    if cache_dir:
        spec = replace(spec, cache_dir=str(cache_dir))
    return spec


def list_models() -> list[ModelSpec]:
    """All specs, in the order they were declared."""
    return list(MODELS.values())


def format_table() -> str:
    """Markdown support table, also used by ``make model-table``."""
    lines = [
        "| Key | Model | Official | Training base | Unsloth | 4-bit base | Packing | Tested |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for spec in list_models():
        tested = "Yes" if spec.tested else "No"
        preq = "Yes" if spec.prequantized_4bit else "No"
        pack = "Yes" if spec.packing else "No"
        official = f"[{spec.official_repo}]({spec.official_url})"
        train = f"[{spec.train_repo}]({spec.train_url})"
        lines.append(
            f"| `{spec.key}` | {spec.name} | {official} | {train} | "
            f"{spec.unsloth_support} | {preq} | {pack} | {tested} |"
        )
    return "\n".join(lines)


_FIELDS = tuple(f for f in ModelSpec.__dataclass_fields__ if f != "key")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("key", nargs="?", help=f"Model key (default: {DEFAULT_MODEL_KEY})")
    parser.add_argument("--keys", action="store_true", help="Print all keys, space separated.")
    parser.add_argument("--table", action="store_true", help="Print the markdown support table.")
    parser.add_argument("--json", action="store_true", help="Print all specs as JSON.")
    parser.add_argument("--default", action="store_true", help="Print the default key.")
    parser.add_argument("--field", choices=_FIELDS, help="Print one field of KEY.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.keys:
        print(" ".join(MODELS))
        return 0
    if args.default:
        print(default_key())
        return 0
    if args.table:
        print(format_table())
        return 0
    if args.json:
        payload: list[dict[str, Any]] = [
            {**spec.__dict__, "official_url": spec.official_url, "train_url": spec.train_url}
            for spec in list_models()
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    try:
        spec = get_model(args.key)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.field:
        print(getattr(spec, args.field))
    else:
        print(spec.train_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
