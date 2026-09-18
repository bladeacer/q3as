import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
import sys
sys.argv = [
    "download_model.py",
    "--model-name", "unsloth/Qwen3-8B",
    "--cache-dir", "models/qwen3-8b",
]
from training.download_model import main
main()
