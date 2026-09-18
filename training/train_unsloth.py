"""train_unsloth.py - QLoRA fine-tuning training script for q3as using Unsloth.

Trains the Qwen3-8B model on the q3as dataset.jsonl using 4-bit QLoRA
quantization optimized for 8 GB VRAM. Uses the Unsloth library for efficient
LoRA fine-tuning with the OpenAI/Qwen chat template format.

Usage:
    uv run python training/train_unsloth.py
    uv run python training/train_unsloth.py --dataset data/processed/dataset.jsonl --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

logger = logging.getLogger("q3as_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune q3as with Unsloth.")
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/processed/dataset.jsonl"),
        help="Path to the training dataset JSONL file.",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps.")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--model-name", type=str, default="unsloth/Qwen3-8B", help="Base model name.")
    parser.add_argument("--output-dir", type=str, default="outputs/q3as", help="Output directory for checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load the JSONL dataset into memory."""
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        sys.exit(1)

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

    logger.info("Loaded %d training examples from %s", len(data), path)
    return data


def setup_training_environment(args: argparse.Namespace) -> dict[str, Any]:
    """Configure training hyperparameters for 8 GB VRAM QLoRA training."""
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
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "report_to": "none",
        "seed": args.seed,
    }
    return config


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Starting q3as QLoRA training pipeline...")

    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("No valid training examples loaded. Exiting.")
        sys.exit(1)

    config = setup_training_environment(args)
    logger.info("Training config: lr=%.6f, batch_size=%d, max_steps=%d, lora_rank=%d",
                args.lr, args.batch_size, args.max_steps, args.lora_rank)

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        import torch

        bfloat_available = is_bfloat16_supported()
        logger.info("bfloat16 supported: %s", bfloat_available)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model_name"],
            max_seq_length=config["max_seq_length"],
            load_in_4bit=True,
            token=None,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora_rank"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head",
            ],
            lora_alpha=config["lora_alpha"],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config["seed"],
        )

        from datasets import Dataset

        formatted_data = []
        for example in dataset:
            messages = example["messages"]
            formatted_data.append({
                "text": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            })

        train_dataset = Dataset.from_list(formatted_data)

        from trl import SFTTrainer
        from transformers import TrainingArguments

        training_args = TrainingArguments(
            **config,
            per_device_eval_batch_size=1,
            evaluation_strategy="no",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            tokenizer=tokenizer,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")

        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        logger.info("Model saved to %s", config["output_dir"])

    except ImportError as exc:
        logger.error(
            "Required training libraries not available: %s. "
            "Ensure unsloth, trl, and transformers are installed: uv sync",
            exc,
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
