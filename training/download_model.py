"""download_model.py - Download and sanity-check the Qwen3-8B model from HuggingFace.

Downloads the base model (Qwen3-8B) to a local cache directory and verifies
that the model can be loaded and accessed correctly before fine-tuning begins.

If the model is already present in the cache directory, the download is skipped
unless --force is passed.

Usage:
    uv run python training/download_model.py
    uv run python training/download_model.py --model-name unsloth/Qwen3-8B --cache-dir models/qwen3-8b
    uv run python training/download_model.py --no-sanity-check
    uv run python training/download_model.py --force
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

logger = logging.getLogger("q3as_download")

# Key files that must exist for the model to be considered fully downloaded
_REQUIRED_FILES = {
    "config.json",
    "configuration.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and sanity-check the Qwen3-8B model.")
    parser.add_argument(
        "--model-name", type=str, default="unsloth/Qwen3-8B",
        help="HuggingFace model identifier to download.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("models/qwen3-8b"),
        help="Local directory to store the downloaded model.",
    )
    parser.add_argument(
        "--sanity-check", dest="sanity_check", action="store_true", default=True,
        help="Run a sanity check after download (default: True).",
    )
    parser.add_argument(
        "--no-sanity-check", dest="sanity_check", action="store_false",
        help="Skip the sanity check after download.",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Force re-download even if model already exists.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug-level logging.",
    )
    return parser.parse_args()


def is_model_present(cache_dir: Path) -> bool:
    """Check if the model files already exist in cache_dir.

    Returns True if all required files are present, False otherwise.
    Verifies that the weight shards referenced by the safetensors index actually
    exist on disk, so a partially downloaded model (metadata only) is not treated
    as complete.
    """
    if not cache_dir.exists():
        return False

    existing_files = {f.name for f in cache_dir.rglob("*") if f.is_file()}
    if not _REQUIRED_FILES.issubset(existing_files):
        return False

    index_path = cache_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        shard_names = set(index.get("weight_map", {}).values())
        if not shard_names.issubset(existing_files):
            missing = shard_names - existing_files
            logger.warning(
                "Model metadata present but weight shards missing: %s",
                ", ".join(sorted(missing)),
            )
            return False

    return True


def download_model(model_name: str, cache_dir: Path, force: bool = False) -> dict[str, Any]:
    """Download the model from HuggingFace to the local cache directory.

    If the model is already present and *force* is False, skips the download
    and returns existing metadata.

    Returns a dict with download metadata and file paths.
    """
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force and is_model_present(cache_dir):
        logger.info(
            "Model already present at %s - skipping download. Use --force to re-download.",
            cache_dir,
        )
        return _read_existing_metadata(cache_dir)

    logger.info("Downloading model '%s' to %s ...", model_name, cache_dir)

    download_info = snapshot_download(
        repo_id=model_name,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )

    downloaded_files = list(cache_dir.rglob("*"))
    file_count = len([f for f in downloaded_files if f.is_file()])
    total_size = sum(f.stat().st_size for f in downloaded_files if f.is_file())

    logger.info(
        "Download complete: %d files, %.2f GB",
        file_count, total_size / (1024 ** 3),
    )

    return {
        "model_name": model_name,
        "cache_dir": str(cache_dir),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "downloaded_files": [str(f.relative_to(cache_dir)) for f in downloaded_files if f.is_file()][:20],
    }


def _read_existing_metadata(cache_dir: Path) -> dict[str, Any]:
    """Read metadata from an already-downloaded model directory."""
    downloaded_files = list(cache_dir.rglob("*"))
    file_count = len([f for f in downloaded_files if f.is_file()])
    total_size = sum(f.stat().st_size for f in downloaded_files if f.is_file())

    metadata_path = cache_dir / "download_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "model_name": "unsloth/Qwen3-8B",
        "cache_dir": str(cache_dir),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "skipped": True,
    }


def sanity_check(cache_dir: Path, model_name: str) -> bool:
    """Verify the downloaded model can be loaded and accessed.

    Returns True if the sanity check passes, False otherwise.
    """
    logger.info("Running sanity check on model at %s ...", cache_dir)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        logger.info("Loading tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(str(cache_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("Set pad_token to eos_token")

        logger.info("Loading model ...")
        model = AutoModelForCausalLM.from_pretrained(
            str(cache_dir),
            torch_dtype=torch.float16,
            device_map="auto",
        )

        logger.info("Model loaded successfully on device: %s", next(model.parameters()).device)

        test_prompt = "Hello, I am an Ada programmer. Please write a simple package."
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Sanity check generation output: %s", generated_text[:200])

        config = model.config
        logger.info(
            "Model verified - name: %s, params: %s, layers: %d",
            config.model_type,
            f"{config.num_parameters:,}",
            config.num_hidden_layers,
        )

        return True

    except Exception as exc:
        logger.error("Sanity check failed: %s", exc, exc_info=True)
        return False


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Starting Qwen3-8B model download pipeline...")

    if not args.force and is_model_present(args.cache_dir):
        print(f"Model already exists at {args.cache_dir}. Use --force to re-download.")
        metadata = _read_existing_metadata(args.cache_dir)
    else:
        metadata = download_model(args.model_name, args.cache_dir, force=args.force)
        print(f"\nDownload complete: {metadata['file_count']} files -> {metadata['cache_dir']}")

    metadata_path = args.cache_dir / "download_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved to %s", metadata_path)

    if args.sanity_check:
        passed = sanity_check(args.cache_dir, args.model_name)
        if passed:
            print("\nSanity check PASSED - model is ready for fine-tuning.")
            print(f"Run: uv run python training/train_unsloth.py")
        else:
            print("\nSanity check FAILED - inspect logs above for details.")
            sys.exit(1)
    else:
        print("\nSkipping sanity check. Use --sanity-check to enable.")


if __name__ == "__main__":
    main()
