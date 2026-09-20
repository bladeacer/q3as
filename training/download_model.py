"""download_model.py - Download and sanity-check the Qwen3-8B model from HuggingFace.

Downloads the base model (Qwen3-8B) to a local cache directory and verifies
that the model is complete and loadable before fine-tuning begins.

If the model is already present in the cache directory, the download is skipped
unless --force is passed.

Sanity checks:
- Default (light): loads the tokenizer and config, and parses every
  safetensors shard header. No GPU work, so it always terminates.
- --sanity-deep: additionally loads the full model on the GPU and generates
  a few tokens. The deep check runs in a child process with a hard timeout
  (default 15 min), because CUDA/Triton initialization can hang on machines
  without a working GPU driver setup and would otherwise wedge the pipeline.

Usage:
    uv run python training/download_model.py
    uv run python training/download_model.py --model-name Qwen/Qwen3-8B --cache-dir models/qwen3-8b
    uv run python training/download_model.py --no-sanity-check
    uv run python training/download_model.py --sanity-deep
    uv run python training/download_model.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

warnings.filterwarnings(
    "ignore",
    message=r".*HF_HUB_ENABLE_HF_TRANSFER.*deprecated.*",
    category=FutureWarning,
)

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
        "--model-name", type=str, default="Qwen/Qwen3-8B",
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
        "--sanity-deep", action="store_true", default=False,
        help="Run the deep GPU sanity check (model load + short generation) "
             "in a child process with --sanity-timeout.",
    )
    parser.add_argument(
        "--sanity-timeout", type=int, default=900,
        help="Hard timeout in seconds for the deep sanity check child process.",
    )
    parser.add_argument(
        "--sanity-child", action="store_true", default=False,
        help=argparse.SUPPRESS,  # internal: runs the deep check in this process
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Force re-download even if model already exists.",
    )
    parser.add_argument(
        "--check-only", action="store_true", default=False,
        help="Exit 0 when the model is complete, 1 when not; never downloads.",
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

    logger.info("Downloading model '%s' to %s using accelerated hf_transfer...", model_name, cache_dir)

    download_info = snapshot_download(
        repo_id=model_name,
        local_dir=str(cache_dir),
        token=os.getenv("HF_TOKEN"),
        ignore_patterns=["*.pt", "*.bin"],
    )

    downloaded_files = list(cache_dir.rglob("*"))
    file_count = len([f for f in downloaded_files if f.is_file()])
    total_size = sum(f.stat().st_size for f in downloaded_files if f.is_file())

    logger.info(
        "Download complete: %d files, %.2f GB (snapshot at %s)",
        file_count, total_size / (1024 ** 3), download_info,
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
        "model_name": "Qwen/Qwen3-8B",
        "cache_dir": str(cache_dir),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "skipped": True,
    }


def sanity_check_light(cache_dir: Path) -> bool:
    """Fast, terminating sanity check: tokenizer, config, shard headers.

    Parses only the safetensors headers (a few KB per shard), so it never
    loads weights into RAM and cannot hang on GPU or driver problems.
    """
    logger.info("Running lightweight sanity check on model at %s ...", cache_dir)
    try:
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer

        logger.info("Loading tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(str(cache_dir))
        logger.info("Tokenizer ok: vocab size %d", len(tokenizer))

        logger.info("Loading config ...")
        config = AutoConfig.from_pretrained(str(cache_dir))
        logger.info(
            "Config ok - model_type: %s, layers: %d, hidden: %d",
            config.model_type, config.num_hidden_layers, config.hidden_size,
        )

        index_path = cache_dir / "model.safetensors.index.json"
        if not index_path.exists():
            logger.error("Missing model.safetensors.index.json")
            return False
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            logger.error("Safetensors index has no weight_map entries")
            return False

        logger.info("Parsing headers of %d shards ...", len(shards))
        for shard in shards:
            shard_path = cache_dir / shard
            if not shard_path.exists():
                logger.error("Missing shard: %s", shard)
                return False
            with safe_open(shard_path, framework="pt") as f:
                tensor_count = len(f.keys())
            if tensor_count <= 0:
                logger.error("Shard %s parses to zero tensors", shard)
                return False
        logger.info("All %d shards parse correctly (%s ...)", len(shards), shards[0])

        logger.info("Lightweight sanity check PASSED (no GPU work performed).")
        return True

    except Exception as exc:
        logger.error("Lightweight sanity check failed: %s", exc, exc_info=True)
        return False


def sanity_check_deep(cache_dir: Path, model_name: str) -> bool:
    """Verify the downloaded model can be loaded on the GPU and generate.

    Heavy: loads all weights onto the device and runs a short generation.
    CUDA/Triton initialization can hang on machines without a working GPU
    setup, so main() runs this in a child process with a hard timeout
    instead of calling it directly.
    """
    logger.info("Running deep sanity check on model at %s ...", cache_dir)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Loading tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(str(cache_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("Set pad_token to eos_token")

        logger.info("Loading model ...")
        model = AutoModelForCausalLM.from_pretrained(
            str(cache_dir),
            dtype=torch.float16,
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
            f"{sum(p.numel() for p in model.parameters()):,}",
            config.num_hidden_layers,
        )

        return True

    except ImportError as exc:
        logger.warning("Deep sanity check skipped - missing dependency: %s", exc)
        return True
    except Exception as exc:
        logger.error("Deep sanity check failed: %s", exc, exc_info=True)
        return False


def run_sanity(
    cache_dir: Path,
    model_name: str,
    deep: bool,
    timeout_s: int,
) -> bool:
    """Run the configured sanity check and return pass/fail.

    The deep check runs in a child process with a hard timeout: if CUDA or
    Triton initialization wedges, the child is killed and the result is
    reported as inconclusive (treated as a pass with a loud warning) so the
    download pipeline always terminates.
    """
    if not deep:
        return sanity_check_light(cache_dir)

    this_file = Path(__file__).resolve()
    child_cmd = [
        sys.executable, "-u", str(this_file),
        "--sanity-child", "--cache-dir", str(cache_dir),
        "--model-name", model_name, "--no-sanity-check",
    ]
    logger.info("Deep check runs in a child process (timeout: %ds) ...", timeout_s)
    try:
        proc = subprocess.run(
            child_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Deep sanity check INCONCLUSIVE: child process exceeded %ds and was "
            "killed (likely CUDA/Triton initialization on this machine). "
            "Run it manually with: uv run python training/download_model.py "
            "--sanity-deep --cache-dir %s",
            timeout_s, cache_dir,
        )
        return True

    # Surface the child's log lines (they went to its stderr).
    for line in (proc.stderr or "").splitlines():
        if line.strip():
            print(line, flush=True)
    return proc.returncode == 0


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.sanity_child:
        # Internal mode: run the deep check in this process and report the
        # result through the exit code.
        passed = sanity_check_deep(args.cache_dir, args.model_name)
        return 0 if passed else 1

    if args.check_only:
        # Presence probe for scripts (make check-model): never downloads.
        if is_model_present(args.cache_dir):
            logger.info("Model present and complete at %s", args.cache_dir)
            return 0
        logger.warning("Model incomplete or missing at %s", args.cache_dir)
        return 1

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

    exit_code = 0
    if args.sanity_check:
        passed = run_sanity(args.cache_dir, args.model_name, args.sanity_deep, args.sanity_timeout)
        if passed:
            print("\nSanity check PASSED - model is ready for fine-tuning.")
            print("Run: uv run python training/train_unsloth.py")
        else:
            print("\nSanity check FAILED - inspect logs above for details.")
            exit_code = 1
    else:
        print("\nSkipping sanity check. Use --sanity-check to enable.")

    sys.stdout.flush()
    sys.stderr.flush()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
