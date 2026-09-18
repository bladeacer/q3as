.PHONY: help sync download build-dataset train eval clean

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "q3as - Qwen 3 Ada SPARK - Available targets:"
	@echo ""
	@echo "  make sync          - Install/update all dependencies via uv sync"
	@echo "  make download      - Download and sanity-check the Qwen3-8B model"
	@echo "  make build-dataset - Build the training dataset from Ada source trees"
	@echo "                      (includes ../adacovex and ../Ada_CRDT by default)"
	@echo "  make train         - Run QLoRA fine-tuning with Unsloth"
	@echo "  make eval          - Run baseline evaluation on the fine-tuned model"
	@echo "                      (methodology derived from ../ada-eval)"
	@echo "  make clean         - Remove generated outputs and caches"
	@echo ""
	@echo "Usage: make [target]   (default: help)"

sync: ## Install/update all dependencies
	uv sync

download: ## Download and sanity-check the Qwen3-8B model
	uv run python training/download_model.py --model-name unsloth/Qwen3-8B --cache-dir models/qwen3-8b

build-dataset: ## Build the training dataset from Ada source trees
	## Includes ../adacovex, ../Ada_CRDT, and ../Ada-83-TLALOC by default
	## Evaluation methodology is derived from ../ada-eval
	uv run python data/processing_scripts/build_dataset.py \
		--input-dir data/raw/ \
		--extra-input-dir ../adacovex \
		--extra-input-dir ../Ada_CRDT \
		--extra-input-dir ../Ada-83-TLALOC

train: ## Run QLoRA fine-tuning with Unsloth
	uv run python training/train_unsloth.py

eval: ## Run baseline evaluation (methodology derived from ../ada-eval)
	uv run python eval/baseline_eval.py --model outputs/q3as

clean: ## Remove generated outputs and caches
	rm -rf outputs/ models/ data/processed/dataset.jsonl data/processed/dataset_metadata.json
	@echo "Cleaned outputs/, models/, and generated dataset."
