"""train_unsloth.py - QLoRA fine-tuning training script for q3as using Unsloth.

Trains the Qwen3-8B model on the q3as dataset.jsonl using 4-bit QLoRA
quantization optimized for 8 GB VRAM. Uses the Unsloth library for efficient
LoRA fine-tuning with the OpenAI/Qwen chat template format.

By default this trains the LOCAL base model downloaded by download_model.py
(models/qwen3-8b, Qwen/Qwen3-8B) so that training, generation, and
evaluation all use exactly the same base weights. Passing a HuggingFace id
via --model-name re-downloads/uses that revision instead.

Usage:
    uv run python training/train_unsloth.py
    uv run python training/train_unsloth.py --dataset data/processed/dataset.jsonl --lr 2e-4
    uv run python training/train_unsloth.py --model-name models/qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# transformers v5 materializes tensors on GPU before bitsandbytes quantization
# (https://github.com/huggingface/transformers issues tracked upstream), OOMing
# 8 GB GPUs when loading 8B models in 4-bit. Deactivating the async loader
# restores the synchronous CPU->quantize->GPU path.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Import unsloth before transformers/trl so its patches apply.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

logger = logging.getLogger("q3as_train")

DEFAULT_MODEL_PATH = "models/qwen3-8b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune q3as with Unsloth.")
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/processed/dataset_train.jsonl"),
        help="Path to the training split. Default is the train file written "
        "by build_dataset: dataset.jsonl holds every turn (including the "
        "val/test records), so training on it would leak the held-out "
        "splits.",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (8 GB VRAM: keep at 1).")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps (effective batch = batch-size x this).")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps.")
    parser.add_argument("--save-steps", type=int, default=100, help="Save a checkpoint every N steps.")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument(
        "--model-name", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to the local base model (downloaded by download_model.py) or a HuggingFace identifier.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/q3as", help="Output directory for checkpoints.")
    parser.add_argument(
        "--val-dataset", type=Path, default=Path("data/processed/dataset_val.jsonl"),
        help="Validation split for eval loss and early stopping. "
        "Falls back to a seeded carve-out of --dataset when the file is missing.",
    )
    parser.add_argument(
        "--test-dataset", type=Path, default=Path("data/processed/dataset_test.jsonl"),
        help="Held-out test split evaluated after training (never seen by "
        "early stopping). Skipped when the file is missing.",
    )
    parser.add_argument(
        "--eval-steps", type=int, default=50,
        help="Run validation every N optimizer steps.",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="Stop training after N consecutive evaluations without a "
        "meaningful validation-loss gain.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def load_jsonl_records(path: Path, required: bool = False) -> list[dict[str, Any]]:
    """Load chat-formatted JSONL records; empty list when optional and absent."""
    if not path.exists():
        if required:
            logger.error("Dataset not found: %s", path)
            sys.exit(1)
        logger.info("Optional dataset not present, skipping: %s", path)
        return []

    data: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "messages" in record and len(record["messages"]) >= 2:
                    data.append(record)
                else:
                    logger.warning("Skipping malformed record on line %d", line_num)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON on line %d: %s", line_num, exc)

    logger.info("Loaded %d examples from %s", len(data), path)
    return data


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load the training JSONL dataset into memory (required file)."""
    return load_jsonl_records(path, required=True)


def split_records(
    data: list[dict[str, Any]],
    seed: int,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Carve val/test slices off a dataset that ships without split files.

    Fallback for custom --dataset files that predate the group-aware split
    in build_dataset.py: the tail is held out (deterministic for a fixed
    seed and file), so early stopping and test metrics still work.
    """
    import random

    n = len(data)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_test = min(n_test, max(n // 10, 1))
    n_val = min(n_val, max(n // 10, 1))
    rng = random.Random(seed)
    indexes = list(range(n))
    rng.shuffle(indexes)
    val_idx = set(indexes[:n_val])
    test_idx = set(indexes[n_val:n_val + n_test])
    val = [data[i] for i in sorted(val_idx)]
    test = [data[i] for i in sorted(test_idx)]
    train = [data[i] for i in range(n) if i not in val_idx and i not in test_idx]
    logger.info(
        "Carved fallback splits: train=%d val=%d test=%d (seed=%d)",
        len(train), len(val), len(test), seed,
    )
    return train, val, test


def setup_training_environment(args: argparse.Namespace, has_val: bool) -> dict[str, Any]:
    """Configure training hyperparameters for 8 GB VRAM QLoRA training.

    With a validation split: eval loss every --eval-steps, best checkpoint
    restored at the end, and early stopping once the validation loss has
    not meaningfully improved for --early-stopping-patience evaluations.
    """
    config: dict[str, Any] = {
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "data_seed": args.seed,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "eval_strategy": "steps" if has_val else "no",
        "eval_steps": args.eval_steps if has_val else 500,
        "load_best_model_at_end": has_val,
        "metric_for_best_model": "eval_loss" if has_val else None,
        "greater_is_better": False if has_val else None,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "report_to": "none",
    }
    if has_val:
        # load_best_model_at_end requires save_steps to be a multiple of
        # eval_steps; round the user's save interval up to the next multiple.
        eval_steps = max(args.eval_steps, 1)
        save_steps = max(args.save_steps, eval_steps)
        remainder = save_steps % eval_steps
        if remainder:
            save_steps += eval_steps - remainder
        config["save_steps"] = save_steps
    return config


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Starting q3as QLoRA training pipeline...")

    # Seed every RNG up front (transformers covers random/numpy/torch/CUDA);
    # SFTConfig also receives the seed for data-order reproducibility.
    try:
        from transformers import set_seed

        set_seed(args.seed)
    except ImportError:
        pass

    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("No valid training examples loaded. Exiting.")
        sys.exit(1)

    val_data = load_jsonl_records(args.val_dataset)
    test_data = load_jsonl_records(args.test_dataset)
    if val_data:
        logger.info(
            "Validation split: %d examples (%s)", len(val_data), args.val_dataset
        )
    else:
        # Dataset predates the group-aware split (or a custom --dataset was
        # passed): carve deterministic val/test slices off the training data
        # so eval loss, early stopping, and test metrics still work.
        dataset, val_data, test_data = split_records(dataset, seed=args.seed)

    config = setup_training_environment(args, has_val=bool(val_data))
    logger.info(
        "Training config: lr=%.6f, batch_size=%d, max_steps=%d, lora_rank=%d, "
        "eval=%s, early_stopping_patience=%d",
        args.lr, args.batch_size, args.max_steps, args.lora_rank,
        config["eval_strategy"],
        args.early_stopping_patience if val_data else 0,
    )

    try:
        import unsloth  # noqa: F401  (imported first so its patches apply)
        from unsloth import FastLanguageModel, is_bfloat16_supported

        bfloat_available = is_bfloat16_supported()
        logger.info("bfloat16 supported: %s", bfloat_available)

        # Resolve the base model path. Prefer the local download so training
        # uses the exact same weights the download step verified.
        model_name = config["model_name"]
        if Path(model_name).exists():
            logger.info("Loading local base model: %s", model_name)
        else:
            logger.warning(
                "Local model path %s does not exist - falling back to HuggingFace id '%s'. "
                "Run `make download` first to keep training/eval on the same base model.",
                model_name, model_name,
            )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=config["max_seq_length"],
            load_in_4bit=True,
            token=os.getenv("HF_TOKEN"),
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora_rank"],
            # Attention + MLP projections only. Training embed_tokens/lm_head
            # (622M params each) blows past 8 GB VRAM.
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=config["lora_alpha"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config["seed"],
        )

        from datasets import Dataset

        def format_examples(examples: list[dict[str, Any]]) -> list[dict[str, str]]:
            return [
                {
                    "text": tokenizer.apply_chat_template(
                        example["messages"], tokenize=False, add_generation_prompt=False
                    )
                }
                for example in examples
            ]

        train_dataset = Dataset.from_list(format_examples(dataset))
        eval_dataset = Dataset.from_list(format_examples(val_data)) if val_data else None
        test_dataset = Dataset.from_list(format_examples(test_data)) if test_data else None

        from trl import SFTConfig, SFTTrainer

        training_args = SFTConfig(
            dataset_text_field="text",
            per_device_eval_batch_size=1,
            **{k: v for k, v in config.items()
               if k not in ("model_name", "lora_rank", "lora_alpha")},
        )

        callbacks: list[Any] = []
        if val_data:
            from transformers import EarlyStoppingCallback

            # One evaluation without a meaningful validation-loss gain counts
            # as one wasted turn: stop after the configured patience.
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_threshold=1e-3,
                )
            )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")

        # Traditional held-out metrics: the test split is never seen by
        # early stopping, so its loss and perplexity back the run up.
        test_metrics: dict[str, float] | None = None
        if test_dataset is not None:
            logger.info("Evaluating the held-out test split (%d examples)...", len(test_data))
            test_metrics_raw = trainer.evaluate(
                eval_dataset=test_dataset, metric_key_prefix="test"
            )
            test_metrics = {
                key: float(value) for key, value in test_metrics_raw.items()
                if isinstance(value, (int, float))
            }
            test_loss = test_metrics.get("test_eval_loss")
            if test_loss is not None:
                test_metrics["test_perplexity"] = math.exp(test_loss)
            logger.info("Test metrics: %s", test_metrics)

        output_dir = Path(config["output_dir"])

        # Free training state before exporting. The merged-16bit save
        # dequantizes base weights on GPU; leftover optimizer/gradients and
        # allocator cache OOM an 8 GB card during the merge.
        import gc

        del trainer
        gc.collect()

        import torch

        torch.cuda.empty_cache()

        # Save a merged 16-bit model (base weights + LoRA) plus the adapter.
        # The eval pipeline (eval/generate.py, eval/baseline_eval.py) loads
        # this output with plain transformers, so the merged form is required.
        # maximum_memory_usage caps how much VRAM merged weights may occupy
        # before unsloth spills them to temporary_location on disk (the GPU
        # still holds the 4-bit base at this point, so keep the cap low).

        from unsloth import unsloth_save_model

        unsloth_save_model(
            model,
            tokenizer,
            save_directory=str(output_dir),
            save_method="merged_16bit",
            temporary_location="outputs/_unsloth_save_buffers",
            maximum_memory_usage=0.15,
        )
        logger.info("Merged 16-bit model saved to %s", output_dir)

        adapter_dir = output_dir / "lora_adapter"
        unsloth_save_model(
            model,
            tokenizer,
            save_directory=str(adapter_dir),
            save_method="lora",
        )
        logger.info("LoRA adapter saved to %s", adapter_dir)

        training_summary = {
            "base_model": model_name,
            "dataset": str(args.dataset),
            "dataset_examples": len(dataset),
            "splits": {
                "train": len(dataset),
                "val": len(val_data),
                "test": len(test_data),
                "val_source": str(args.val_dataset) if val_data else "fallback-carve",
                "test_source": str(args.test_dataset) if test_data else "fallback-carve",
            },
            "early_stopping_patience": args.early_stopping_patience if val_data else None,
            "output_dir": str(output_dir),
            "test_metrics": test_metrics,
            "hyperparameters": {
                k: v for k, v in config.items() if k != "model_name"
            },
        }
        with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)
        logger.info("Training summary saved to %s", output_dir / "training_summary.json")

    except ImportError as exc:
        logger.error(
            "Required training libraries not available: %s. "
            "Ensure unsloth, trl, and transformers are installed: uv sync",
            exc,
        )
        sys.exit(1)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
