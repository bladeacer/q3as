"""train_unsloth.py - QLoRA fine-tuning training script for q3as using Unsloth.

Trains the Qwen3-8B model on the q3as dataset.jsonl using 4-bit QLoRA
quantization optimized for 8 GB VRAM. Uses the Unsloth library for efficient
LoRA fine-tuning with the OpenAI/Qwen chat template format.

By default this trains the LOCAL base model downloaded by download_model.py
(models/qwen3-8b, Qwen/Qwen3-8B) so that training, generation, and
evaluation all use exactly the same base weights. Passing a HuggingFace id
via --model-name re-downloads/uses that revision instead.

System RAM matters here as much as VRAM: an 8 GB card is paired with a small
host, and the 423 MB train split used to be held in memory three times over
(parsed records, chat-templated strings, then the Arrow table). The splits are
now streamed from JSONL into Arrow one record at a time, and the resulting
Arrow table is cached under data/processed/.tokenized so a rerun does not
re-tokenize.

Usage:
    uv run python training/train_unsloth.py
    uv run python training/train_unsloth.py --dataset data/processed/dataset.jsonl --lr 2e-4
    uv run python training/train_unsloth.py --model-name models/qwen3-8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_PROCESSING_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "processing_scripts"
if str(_PROCESSING_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROCESSING_SCRIPTS_DIR))

# digest_file is the repo's content-hash helper, reused so the Arrow cache key
# is computed the same way the dataset stages compute their fingerprints.
from stage_state import digest_file

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
from transformers import TrainerCallback, set_seed

logger = logging.getLogger("q3as_train")

DEFAULT_MODEL_PATH = "models/qwen3-8b"

# Where the chat-templated Arrow tables are cached. Under data/processed
# (gitignored) because they are derived from the split files next to them, and
# a rebuild of those files invalidates them through the fingerprint.
TOKENIZED_CACHE_DIR = Path("data/processed/.tokenized")

# Bump when the text produced for a record changes, so an Arrow table written
# by an older loader is rebuilt instead of silently reused. The fingerprint
# also covers the split file content, the chat template, and the tokenizer.
TEXT_PIPELINE_VERSION = 2


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
    parser.add_argument(
        "--max-seq-length", type=int, default=1024,
        help="Maximum training sequence length. Keep at 1024 or lower on an 8 GB GPU.",
    )
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
    parser.add_argument(
        "--eval-sample", type=int, default=2000,
        help="Examples drawn from the val split for each during-training "
        "early-stopping evaluation. 0 uses the whole split. The default keeps "
        "a 500-step run inside --eval-budget-min; a full val pass costs about "
        "37 min on this hardware, so 10 of them would be 6.2 h.",
    )
    parser.add_argument(
        "--test-sample", type=int, default=2000,
        help="Examples drawn from the held-out test split for the final "
        "metrics. 0 uses the whole split.",
    )
    parser.add_argument(
        "--eval-budget-min", type=float, default=300.0,
        help="Soft ceiling in minutes for the whole run's evaluation. Only "
        "used to project the cost up front and warn when the chosen sample "
        "sizes exceed it; it never silently changes them.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def usable_record(record: Any, line_num: int) -> dict[str, Any] | None:
    """*record* if it is trainable, else None (logged with its line number).

    One place for the skip rules so the streaming loader and the in-memory
    fallback loader can never disagree about which records get trained on.
    """
    if not (isinstance(record, dict) and "messages" in record
            and isinstance(record["messages"], list)
            and len(record["messages"]) >= 2):
        logger.warning("Skipping malformed record on line %d", line_num)
        return None
    last = record["messages"][-1]
    if not isinstance(last, dict):
        logger.warning("Skipping malformed record on line %d", line_num)
        return None
    if last.get("role") == "assistant" and not str(last.get("content", "")).strip():
        # Never train on an empty reply: it teaches immediate-EOS.
        logger.warning("Skipping empty-assistant record on line %d", line_num)
        return None
    return record


def iter_usable_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every trainable record of a chat-formatted JSONL split, in order.

    The single reader: the streaming Arrow builders and the in-memory fallback
    all go through it, so they cannot disagree about which records are trainable
    or in what order.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = usable_record(json.loads(line), line_num)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON on line %d: %s", line_num, exc)
                continue
            if record is not None:
                yield record


def load_jsonl_records(path: Path, required: bool = False) -> list[dict[str, Any]]:
    """Load chat-formatted JSONL records; empty list when optional and absent.

    Only used by the fallback carve path (see split_records), which needs the
    records as a list it can index into. The normal path streams the file into
    Arrow instead; see build_text_dataset.
    """
    if not path.exists():
        if required:
            logger.error("Dataset not found: %s", path)
            sys.exit(1)
        logger.info("Optional dataset not present, skipping: %s", path)
        return []

    data = list(iter_usable_records(path))
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


class ChatTemplateJsonl:
    """Yield the chat-templated text of a JSONL split, one record at a time.

    A callable class rather than a closure or a nested generator:
    ``Dataset.from_generator`` pickles the callable when it is allowed to use
    worker processes, and a closure over the tokenizer would not survive that.
    """

    def __init__(self, path: Path, tokenizer: Any) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer

    def __call__(self) -> Iterator[dict[str, Any]]:
        for record in iter_usable_records(self.path):
            yield {
                "text": self.tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False
                )
            }


class TokenizedJsonl:
    """Yield the token ids of a JSONL split, one record at a time.

    This is the format the trainer actually consumes. Rendering to text and
    letting trl tokenize it again costs a four-minute pass over 65k records and
    a second resident copy of them; tokenizing here, once, while streaming,
    removes both. Truncating during tokenization matches what trl does anyway
    (tokenize the whole string, then cut to max_length), so the ids are the
    same either way.
    """

    def __init__(self, path: Path, tokenizer: Any, max_seq_length: int) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self) -> Iterator[dict[str, Any]]:
        for record in iter_usable_records(self.path):
            text = self.tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=False
            )
            ids = self.tokenizer(
                text, truncation=True, max_length=self.max_seq_length
            ).input_ids
            if ids:
                yield {"input_ids": ids}


_TOKENIZER_DIGEST: str | None = None


def tokenizer_digest(tokenizer: Any) -> str:
    """Content hash of the tokenizer's vocabulary, computed once per run.

    The name and vocab size are not enough: a different vocabulary can ship
    under the same model name, and that would silently train on stale ids.
    """
    global _TOKENIZER_DIGEST
    if _TOKENIZER_DIGEST is None:
        vocab = tokenizer.get_vocab()
        blob = json.dumps(vocab, sort_keys=True, ensure_ascii=True).encode("utf-8")
        _TOKENIZER_DIGEST = hashlib.sha256(blob).hexdigest()
    return _TOKENIZER_DIGEST


def text_fingerprint(
    path: Path, tokenizer: Any, max_seq_length: int = 0, kind: str = "text"
) -> str:
    """Identity of the Arrow table *path* should produce with *tokenizer*.

    Covers the pipeline version, the split file's content, the chat template,
    the tokenizer's vocabulary, and (for the tokenized table) the truncation
    length, so any of them changing rebuilds the table instead of reusing a
    stale one.
    """
    digest = digest_file(path)
    template = getattr(tokenizer, "chat_template", None) or ""
    parts = (
        str(TEXT_PIPELINE_VERSION),
        kind,
        str(path.resolve()),
        str(digest.get("digest")),
        str(digest.get("size")),
        hashlib.sha256(template.encode("utf-8")).hexdigest(),
        tokenizer_digest(tokenizer),
        str(max_seq_length),
    )
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]


# Fingerprints of the tables built during this run, so the cache can be pruned
# afterwards. A module-level set because the builders are called from main()
# and threading a collector through every one of them adds nothing.
_USED_FINGERPRINTS: set[str] = set()


def _stream_to_arrow(
    path: Path,
    generator: Any,
    features: Any,
    fingerprint: str,
) -> Any:
    """Materialise *generator* into a cached Arrow table keyed by *fingerprint*.

    The split used to be parsed into a list of records, rendered into a second
    list of templated strings, and only then converted by
    ``Dataset.from_list``, so all three copies of a 423 MB split were resident
    at once: about 2.2 GB of anonymous memory, peaking near 4 GB, on a host
    with 6 GB to spare. Streaming keeps a single record alive and lets the
    Arrow writer do the conversion, which costs about 200 MB instead.
    """
    from datasets import Dataset

    _USED_FINGERPRINTS.add(fingerprint)
    # Best-effort cache-hit detection for the log line. datasets lays the
    # table out under a fingerprint-named directory; if that layout ever
    # changes we just report a miss and rebuild, which is harmless.
    cached = any(TOKENIZED_CACHE_DIR.glob(f"**/*{fingerprint}*/**/*.arrow"))
    started = time.monotonic()
    dataset = Dataset.from_generator(
        generator, features=features, cache_dir=str(TOKENIZED_CACHE_DIR),
        fingerprint=fingerprint,
    )
    logger.info(
        "Streamed %d examples from %s in %.0fs (%s Arrow cache)",
        len(dataset), path, time.monotonic() - started,
        "reused" if cached else "wrote",
    )
    return dataset


def prune_tokenized_cache(keep: set[str]) -> int:
    """Delete cached Arrow tables this run did not just touch.

    The cache key covers the split content, the tokenizer, the truncation
    length, and the pipeline version, so every one of those changing leaves a
    directory behind that nothing will ever read again. datasets does not
    collect them. Returns the number of directories removed.

    Only called once per run, after all three tables are built, so it cannot
    delete a table the running job is using. Two runs with different
    ``--max-seq-length`` values would each evict the other, which costs a
    rebuild (about 40 s) and never correctness.
    """
    root = TOKENIZED_CACHE_DIR / "generator"
    if not root.is_dir():
        return 0
    removed = 0
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        fingerprint = entry.name.removeprefix("default-fingerprint=")
        if fingerprint not in keep:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    if removed:
        logger.info("Pruned %d stale tokenized-cache table(s)", removed)
    return removed


def build_text_dataset(path: Path, tokenizer: Any) -> Any:
    """Stream the split at *path* into a one-column (text) Arrow dataset.

    Used for the eval splits, which the chunked eval callback tokenizes itself
    one example at a time and so has no use for a tokenized column.
    """
    from datasets import Features, Value

    if not path.is_file():
        logger.info("Optional dataset not present, skipping: %s", path)
        return None
    fingerprint = text_fingerprint(path, tokenizer)
    return _stream_to_arrow(
        path, ChatTemplateJsonl(path, tokenizer),
        Features({"text": Value("string")}), fingerprint,
    )


def build_ids_dataset(path: Path, tokenizer: Any, max_seq_length: int) -> Any:
    """Stream the split at *path* into a one-column (input_ids) Arrow dataset.

    This is what trl/Unsloth consume: an ``input_ids`` column marks the dataset
    as already processed, so their tokenization pass is skipped. ids are int32
    because a token id cannot exceed a vocabulary of 2^31, which halves the
    table (a 65,809-record train split is ~135 MB of ids rather than ~470 MB
    of rendered text).
    """
    from datasets import Features, Sequence, Value

    if not path.is_file():
        logger.info("Optional dataset not present, skipping: %s", path)
        return None
    fingerprint = text_fingerprint(path, tokenizer, max_seq_length, kind="ids")
    return _stream_to_arrow(
        path, TokenizedJsonl(path, tokenizer, max_seq_length),
        Features({"input_ids": Sequence(Value("int32"))}), fingerprint,
    )


# Measured on the reference host (RTX 5050 Laptop, 4-bit NF4 Qwen3-8B) as the
# cost of the chunked eval: a transformer forward plus the final lm_head over
# the same tokens, at 128-token chunks. Used only to project eval cost and warn
# when the chosen sample sizes blow the budget; it never changes them.
EVAL_MS_PER_TOKEN = 1.2

# Mean tokens per record in the shipped splits, also measured. Only feeds the
# same projection.
SPLIT_MEAN_TOKENS = 511


def subsample(dataset: Any, limit: int, seed: int) -> Any:
    """Deterministic *limit*-example subset of *dataset*, in file order.

    A seeded, sorted index draw rather than ``Dataset.shuffle``: the same
    subset has to come back on every run, and every evaluation point has to
    score the same examples, or the early-stopping curve would compare
    different data at each step.
    """
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    rng = random.Random(seed)
    return dataset.select(sorted(rng.sample(range(len(dataset)), limit)))


def project_eval_minutes(
    n_val: int, n_test: int, n_evals: int
) -> float:
    """Minutes the whole run's evaluation is expected to take."""
    per_example_s = SPLIT_MEAN_TOKENS * EVAL_MS_PER_TOKEN / 1000.0
    return (n_evals * n_val + n_test) * per_example_s / 60.0


def setup_training_environment(args: argparse.Namespace, has_val: bool) -> dict[str, Any]:
    """Configure training hyperparameters for 8 GB VRAM QLoRA training.

    With a validation split: eval loss every --eval-steps, and early stopping
    once the validation loss has not meaningfully improved for
    --early-stopping-patience evaluations. The adapter saved at the end is the
    final state, not the best one: ``load_best_model_at_end`` is off, so
    nothing is restored.
    """
    config: dict[str, Any] = {
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "max_length": args.max_seq_length,
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
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
        "save_only_model": True,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "report_to": "none",
    }
    if has_val:
        # The adapter saved at the end is the final state, so a checkpoint
        # that is not a multiple of eval_steps can fall between two eval
        # points and never be the state the loss curve was measured at.
        # Keep save_steps a multiple of eval_steps rather than silently
        # rounding, so the flag on disk matches the flag the user passed.
        eval_steps = max(args.eval_steps, 1)
        save_steps = max(args.save_steps, eval_steps)
        if save_steps % eval_steps:
            logger.warning(
                "save_steps=%d is not a multiple of eval_steps=%d; using %d. "
                "Checkpoints will not land on every evaluation point.",
                save_steps, eval_steps, save_steps + (eval_steps - save_steps % eval_steps),
            )
            save_steps += eval_steps - save_steps % eval_steps
        config["save_steps"] = save_steps
    return config


def main() -> None:
    args = parse_args()
    # Configure this module's logger directly instead of calling basicConfig.
    # Importing unsloth (at module scope) already installs a root handler and
    # sets the root level to WARNING, which makes basicConfig a silent no-op:
    # every q3as_train line below was being dropped, and --verbose did nothing.
    # Configuring only our own logger also keeps the third-party INFO firehose
    # (huggingface HTTP requests, dataset maps) out of training.log.
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(_handler)
    logger.propagate = False

    logger.info("Starting q3as QLoRA training pipeline...")

    # Seed every RNG up front (transformers covers random/numpy/torch/CUDA);
    # SFTConfig also receives the seed for data-order reproducibility.
    set_seed(args.seed)

    # Fail before the 8B model load, not after it: the stream path cannot know
    # the split is unusable until it has read it.
    if not args.dataset.is_file():
        logger.error("Dataset not found: %s", args.dataset)
        sys.exit(1)

    # Decide where every split comes from before loading anything, because the
    # fallback carve needs the training records as an indexable list while the
    # normal path streams each file into Arrow.
    val_path = args.val_dataset if args.val_dataset.is_file() else None
    test_path = args.test_dataset if args.test_dataset.is_file() else None

    train_records: list[dict[str, Any]] | None = None
    val_records: list[dict[str, Any]] | None = None
    test_records: list[dict[str, Any]] | None = None
    # Track where each split came from, so the recorded provenance cannot claim
    # a held-out file was used when it was carved out of the training data, nor
    # claim a carve happened when the split was simply absent.
    val_source = str(val_path) if val_path else "fallback-carve"
    test_source = str(test_path) if test_path else "absent"
    split_load = "stream"

    if val_path is None:
        # Dataset predates the group-aware split (or a custom --dataset was
        # passed): carve deterministic val/test slices off the training data
        # so eval loss, early stopping, and test metrics still work. An
        # existing test split is kept: a missing val file must not discard real
        # held-out data. This path holds the records in memory, which the
        # stream path does not.
        split_load = "in-memory-carve"
        logger.warning(
            "No validation split at %s; carving val/test from the training data. "
            "Reported test metrics will not be held-out.",
            args.val_dataset,
        )
        train_records = load_dataset(args.dataset)
        if not train_records:
            logger.error("No valid training examples loaded. Exiting.")
            sys.exit(1)
        train_records, val_records, carved_test = split_records(
            train_records, seed=args.seed
        )
        if test_path is None:
            test_records = carved_test
            test_source = "fallback-carve"
        else:
            test_records = load_jsonl_records(test_path)
            if not test_records:
                test_records = None

    config = setup_training_environment(args, has_val=val_path is not None)
    logger.info(
        "Training config: lr=%.6f, batch_size=%d, max_steps=%d, lora_rank=%d, "
        "eval=%s, early_stopping_patience=%d",
        args.lr, args.batch_size, args.max_steps, args.lora_rank,
        config["eval_strategy"],
        args.early_stopping_patience if val_path else 0,
    )

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported

        bfloat_available = is_bfloat16_supported()
        logger.info("bfloat16 supported: %s", bfloat_available)
        # The probe used to be logged and then ignored, while the config
        # hard-coded bf16, so a CPU-only or pre-Ampere machine was configured
        # for a dtype the trainer cannot use.
        config["bf16"] = bfloat_available
        if not bfloat_available:
            config["fp16"] = True
            logger.warning(
                "bfloat16 is unavailable; falling back to fp16 for training."
            )

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

        # Build the Arrow tables. Streaming keeps peak host memory near the
        # model load's own footprint; the lists only exist on the fallback
        # carve path, and they are released as soon as the tables are built.
        #
        # The train split is stored pre-tokenized: an input_ids column marks
        # the dataset as processed, so trl skips its own tokenization pass
        # (4m07s over 65,809 records, plus a second resident copy of the
        # rendered text). The eval splits stay as text because the chunked
        # eval callback below tokenizes them itself, one example at a time.
        max_seq_length = config["max_seq_length"]
        _USED_FINGERPRINTS.clear()
        if val_path is not None:
            train_dataset = build_ids_dataset(args.dataset, tokenizer, max_seq_length)
            eval_dataset = build_text_dataset(val_path, tokenizer)
            test_dataset = build_text_dataset(test_path, tokenizer) if test_path else None
        else:
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

            train_dataset = Dataset.from_list(format_examples(train_records or []))
            eval_dataset = (
                Dataset.from_list(format_examples(val_records)) if val_records else None
            )
            test_dataset = (
                Dataset.from_list(format_examples(test_records)) if test_records else None
            )

        if train_dataset is None or not len(train_dataset):
            logger.error("No valid training examples loaded. Exiting.")
            sys.exit(1)

        train_count = len(train_dataset)
        val_total = len(eval_dataset) if eval_dataset is not None else 0
        test_total = len(test_dataset) if test_dataset is not None else 0
        if val_total:
            logger.info("Validation split: %d examples (%s)", val_total, val_path)
        elif val_path is not None:
            # The carve fallback needs the records in memory, so it cannot run
            # this late. Say plainly that early stopping is off rather than
            # training silently without eval.
            logger.warning(
                "Validation split %s yielded no usable records; training "
                "without eval loss or early stopping.", val_path,
            )

        # Cap what the eval callback has to score. Every evaluation point
        # scores the same seeded subset, so the curve stays comparable and
        # early stopping still sees a consistent signal; only the size of the
        # sample changes. Training is untouched: all train_count records are
        # still trained on.
        eval_dataset = subsample(eval_dataset, args.eval_sample, args.seed)
        test_dataset = subsample(test_dataset, args.test_sample, args.seed)
        val_count = len(eval_dataset) if eval_dataset is not None else 0
        test_count = len(test_dataset) if test_dataset is not None else 0
        if val_count and val_count < val_total:
            logger.info(
                "Scoring a seeded %d-example subset of the %d-example val "
                "split for early stopping (--eval-sample %d for all).",
                val_count, val_total, args.eval_sample,
            )
        if test_count and test_count < test_total:
            logger.info(
                "Scoring a seeded %d-example subset of the %d-example test "
                "split (--test-sample %d for all).",
                test_count, test_total, args.test_sample,
            )

        # Project the evaluation cost and say so, rather than letting a run
        # discover nine hours of eval the hard way.
        n_evals = max(1, math.ceil(config["max_steps"] / max(int(config["eval_steps"]), 1)))
        projected = project_eval_minutes(val_count, test_count, n_evals)
        logger.info(
            "Evaluation budget: %d eval(s) x %d val + %d test = ~%.0f min "
            "(ceiling %.0f min)", n_evals, val_count, test_count, projected,
            args.eval_budget_min,
        )
        if projected > args.eval_budget_min:
            logger.warning(
                "Projected evaluation time ~%.0f min exceeds the %.0f min "
                "ceiling. Lower --eval-sample / --test-sample, raise "
                "--eval-steps, or raise --eval-budget-min to accept it.",
                projected, args.eval_budget_min,
            )

        prune_tokenized_cache(set(_USED_FINGERPRINTS))

        del train_records, val_records, test_records
        import gc

        gc.collect()
        logger.info(
            "Splits ready: train=%d (all records) val=%d/%d test=%d/%d (%s load)",
            train_count, val_count, val_total, test_count, test_total, split_load,
        )

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

        def _chunked_eval_loss(dataset: Dataset, chunk: int = 64) -> float | None:
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
                    for start in range(0, hidden.shape[0] - 1, chunk):
                        end = min(start + chunk, hidden.shape[0] - 1)
                        # position t predicts token t+1
                        logits = causal.lm_head(hidden[start:end])
                        targets = ids[0, start + 1:end + 1]
                        loss_sum = torch.nn.functional.cross_entropy(
                            logits.float(), targets, reduction="sum",
                        )
                        total_nll += float(loss_sum)
                        total_tokens += int(targets.numel())
                        del logits, targets, loss_sum
                    del hidden, ids
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
        if val_count:
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
            # Deliberately not handed to the Trainer. eval_strategy is "no" and
            # the chunked callback above is what actually measures loss; passing
            # the split in only made trl tokenize all 3,709 val records on every
            # run, to produce a dataset nothing read.
            eval_dataset=None,
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
            logger.info("Evaluating the held-out test split (%d examples)...", test_count)
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
                "split_load": split_load,
                "tokenized_cache": str(TOKENIZED_CACHE_DIR),
            },
            "dataset_examples": train_count,
            "train_loss_final": train_loss_final,
            "train_loss_history": train_loss_history,
            "eval_loss_history": eval_loss_history,
            "splits": {
                # "train" is every record; val/test are what the eval callback
                # actually scored, which is a seeded subset when *_total is
                # larger. Reported losses must be read against these counts.
                "train": train_count,
                "val": val_count,
                "val_total": val_total,
                "test": test_count,
                "test_total": test_total,
                "val_source": val_source,
                "test_source": test_source,
            },
            "eval_budget": {
                "eval_sample": args.eval_sample,
                "test_sample": args.test_sample,
                "budget_minutes": args.eval_budget_min,
                "projected_minutes": round(projected, 1),
                "eval_points": n_evals,
                "ms_per_token": EVAL_MS_PER_TOKEN,
            },
            "early_stopping_patience": args.early_stopping_patience if val_count else None,
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
