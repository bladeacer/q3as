.PHONY: help sync download build-dataset train eval clean

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "q3as - Qwen 3 Ada SPARK - Available targets:"
	@echo ""
	@echo "  make sync          - Install/update all dependencies via uv sync"
	@echo "  make download      - Download and sanity-check the Qwen3-8B model"
	@echo "  make build-dataset - Build the training dataset from Ada source trees"
	@echo "  make train         - Run QLoRA fine-tuning with Unsloth"
	@echo "  make eval          - Run baseline evaluation on the fine-tuned model"
	@echo "  make clean         - Remove generated outputs and caches"
	@echo ""
	@echo "Usage: make [target]   (default: help)"

sync: ## Install/update all dependencies
	uv sync

download: ## Download and sanity-check the Qwen3-8B model
	uv run python training/download_model.py

build-dataset: ## Build the training dataset from Ada source trees
	uv run python data/processing_scripts/build_dataset.py --input-dir data/raw/

train: ## Run QLoRA fine-tuning with Unsloth
	uv run python training/train_unsloth.py

eval: ## Run baseline evaluation
	uv run python eval/baseline_eval.py

clean: ## Remove generated outputs and caches
	rm -rf outputs/ models/ data/processed/dataset.jsonl
	@echo "Cleaned outputs/, models/, and generated dataset."
