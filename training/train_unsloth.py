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
import time
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

import unsloth  # noqa: F401
from transformers import TrainerCallback

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
    parser.add_argument(
        "--skip-merged-save", action="store_true",
        help="Skip the merged 16-bit model export (saves several GB and "
        "minutes per run); the LoRA adapter and training summary are still "
        "written. Intended for short experiment runs.",
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
                    last = record["messages"][-1]
                    if last.get("role") == "assistant" and not str(last.get("content", "")).strip():
                        # Never train on an empty reply: it teaches immediate-EOS.
                        logger.warning("Skipping empty-assistant record on line %d", line_num)
                        continue
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
        # Evaluation is callback-driven (chunked, memory-safe on 8 GB);
        # Trainer-managed eval stays off, as does best-checkpoint restore
        # (which would depend on it). The adapter saved at the end is the
        # final state, which the experiment runner treats as the result.
        "eval_strategy": "no",
        "eval_steps": args.eval_steps if has_val else 500,
        "load_best_model_at_end": False,
        "metric_for_best_model": None,
        "greater_is_better": None,
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

        # -----------------------------------------------------------------
        # Memory-safe eval loss. On this 8 GB card, HF Trainer's eval path
        # OOMs: accelerate wraps the model so every forward's bf16 logits
        # (seq 2048 x vocab ~152k) are converted to fp32 before the metrics
        # gather, a ~1.8 GB transient on top of the 4-bit weights, LoRA
        # state, and CUDA graphs. The prior 500-step run never hit this
        # because it trained before split files existed and never ran eval.
        # Instead: chunked manual evaluation. Labels are tokenized once,
        # the final lm_head is applied in token chunks, and the mean loss
        # over non-padding tokens is computed exactly (sum of token losses
        # divided by their count) with no more than ~100 MB transient.
        # EvalEarlyStoppingCallback (defined below) runs the schedule and
        # the early-stop rule on this curve, replacing both Trainer-managed
        # evaluation and transformers.EarlyStoppingCallback.
        # -----------------------------------------------------------------
        _eval_ctx: dict[str, Any] = {"model": model, "tokenizer": tokenizer}

        def _chunked_eval_loss(dataset: Dataset, chunk: int = 128) -> float | None:
            """Mean token-level NLL over *dataset*, computed in chunks."""
            import torch

            mdl = _eval_ctx["model"]
            tok = _eval_ctx["tokenizer"]
            mdl.eval()
            total_nll = 0.0
            total_tokens = 0
            with torch.no_grad():
                for example in dataset:
                    ids = tok(
                        example["text"], return_tensors="pt",
                        truncation=True, max_length=config["max_seq_length"],
                    ).input_ids.to(mdl.device)
                    if ids.shape[1] < 2:
                        continue
                    causal = mdl.get_base_model()  # Qwen3ForCausalLM (fast)
                    hidden = causal.model(input_ids=ids).last_hidden_state[0]  # [seq, hidden]
                    losses: list[torch.Tensor] = []
                    for start in range(0, hidden.shape[0] - 1, chunk):
                        end = min(start + chunk, hidden.shape[0] - 1)
                        # position t predicts token t+1
                        logits = causal.lm_head(hidden[start:end])
                        targets = ids[0, start + 1:end + 1]
                        losses.append(
                            torch.nn.functional.cross_entropy(
                                logits.float(), targets, reduction="none",
                            )
                        )
                    token_losses = torch.cat(losses)
                    total_nll += float(token_losses.sum())
                    total_tokens += int(token_losses.numel())
            mdl.train()
            return total_nll / total_tokens if total_tokens else None

        class EvalEarlyStoppingCallback(TrainerCallback):
            """Chunked eval on the eval_steps schedule + early-stop rule.

            Replaces Trainer-managed evaluation (which OOMs on 8 GB, see
            above) and transformers.EarlyStoppingCallback (which depends on
            it). The stop rule matches EarlyStoppingCallback: stop when the
            best eval loss has not been beaten by early_stopping_threshold
            for early_stopping_patience consecutive evaluations.
            """

            def __init__(self, patience: int, threshold: float) -> None:
                self.patience = patience
                self.threshold = threshold
                self.history: list[dict[str, float]] = []

            def on_step_end(
                self, args: Any, state: Any, control: Any, **kwargs: Any
            ) -> Any:
                _ = args, kwargs
                if eval_dataset is None:
                    return control
                if state.global_step % max(int(training_args.eval_steps or 1), 1) != 0:
                    return control
                loss = _chunked_eval_loss(eval_dataset)
                if loss is None:
                    return control
                self.history.append({"step": int(state.global_step), "eval_loss": loss})
                best = min(p["eval_loss"] for p in self.history)
                state.best_metric = best
                logger.info(
                    "eval_loss at step %d: %.4f (best %.4f)",
                    state.global_step, loss, best,
                )
                recent = self.history[-(self.patience + 1):]
                if len(recent) > self.patience and all(
                    p["eval_loss"] > best + self.threshold for p in recent[1:]
                ):
                    control.should_training_stop = True
                    logger.info(
                        "Early stopping at step %d: no eval-loss improvement "
                        "for %d evaluations", state.global_step, self.patience,
                    )
                return control

        from trl import SFTConfig, SFTTrainer

        # eval_strategy stays "no": evaluation is callback-driven so the
        # Trainer never materializes logits-sized eval batches.
        training_args = SFTConfig(
            dataset_text_field="text",
            **{k: v for k, v in config.items()
               if k not in ("model_name", "lora_rank", "lora_alpha")},
        )

        callbacks: list[Any] = []
        if val_data:
            callbacks.append(
                EvalEarlyStoppingCallback(
                    patience=args.early_stopping_patience,
                    threshold=1e-3,
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
        train_start = time.monotonic()
        trainer.train()
        train_seconds = time.monotonic() - train_start
        # Whether early stopping fired (before trainer is deleted below).
        # The Trainer sets should_training_stop on normal completion as
        # well, so the signal is "ended before the step budget":
        early_stopped = bool(
            trainer.state.global_step < config["max_steps"]
        )
        logger.info("Training complete!")

        # Training-loss history: the trainer logs train loss every
        # logging_steps. The eval-report trend analysis reads this.
        train_loss_history: list[dict[str, float]] = []
        state = trainer.state
        if state is not None and getattr(state, "log_history", None):
            for entry in state.log_history:
                if isinstance(entry, dict) and "loss" in entry and "step" in entry:
                    train_loss_history.append(
                        {"step": int(entry["step"]), "loss": float(entry["loss"])}
                    )

        # Final train loss: last logged value (the summary entry, when
        # present, carries the run-level mean; prefer the mean).
        train_loss_final: float | None = None
        if state is not None and getattr(state, "log_history", None):
            for entry in reversed(state.log_history):
                if isinstance(entry, dict) and "train_loss" in entry:
                    train_loss_final = float(entry["train_loss"])
                    break

        # Eval-loss curve: from the chunked-eval callback (Trainer-managed
        # eval is off). The report's per-step val trend table is built from
        # this.
        eval_loss_history: list[dict[str, float]] = []
        for callback in callbacks:
            history = getattr(callback, "history", None)
            if history:
                eval_loss_history = list(history)
                break

        # Traditional held-out metrics: the test split is never seen by
        # early stopping, so its loss and perplexity back the run up.
        # Chunked path for the same memory reasons as validation.
        test_metrics: dict[str, float] | None = None
        if test_dataset is not None:
            logger.info("Evaluating the held-out test split (%d examples)...", len(test_data))
            test_loss = _chunked_eval_loss(test_dataset)
            test_metrics = {}
            if test_loss is not None:
                test_metrics["test_eval_loss"] = test_loss
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

        if args.skip_merged_save:
            logger.info("Skipping merged 16-bit save (--skip-merged-save)")
        else:
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
            "experiment": {
                "skip_merged_save": args.skip_merged_save,
            },
            "dataset_examples": len(dataset),
            "train_loss_final": train_loss_final,
            "train_loss_history": train_loss_history,
            "eval_loss_history": eval_loss_history,
            "splits": {
                "train": len(dataset),
                "val": len(val_data),
                "test": len(test_data),
                "val_source": str(args.val_dataset) if val_data else "fallback-carve",
                "test_source": str(args.test_dataset) if test_data else "fallback-carve",
            },
            "early_stopping_patience": args.early_stopping_patience if val_data else None,
            "early_stopped": early_stopped,
            "train_seconds": round(train_seconds, 1),
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
