"""run_download.py - Convenience wrapper mirroring `make download`.

Delegates to training/download_model.py main() and exits the process
explicitly (os._exit) so lingering background threads (hf_transfer, tokenizers
parallelism) can never keep a finished run hanging on a dead stdout.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sys.argv = [
    "download_model.py",
    "--model-name", "Qwen/Qwen3-8B",
    "--cache-dir", "models/qwen3-8b",
]

from training.download_model import main  # noqa: E402

if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Hard-exit: no atexit handlers, no thread joins, no hangs.
    os._exit(exit_code)
