.PHONY: help sync download build-dataset train generate eval eval-pipeline prove clean all check-model

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "q3as - Qwen 3 Ada SPARK - Available targets:"
	@echo ""
	@echo "  make all           - Run full pipeline: model download (if needed), dataset, train, evaluate"
	@echo "  make sync          - Install/update all dependencies via uv sync"
	@echo "  make download      - Download and sanity-check the Qwen3-8B model"
	@echo "  make build-dataset - Build the training dataset from Ada source trees"
	@echo "                      (includes ../adacovex, ../Ada_CRDT, ../Ada-83-TLALOC, ../ada-eval by default)"
	@echo "  make train         - Run QLoRA fine-tuning with Unsloth"
	@echo "  make generate      - Generate Ada code with base and fine-tuned models"
	@echo "  make eval          - Run baseline evaluation with BLEU + ada-eval metrics"
	@echo "                      (compilation, test, SPARK proof; base model comparison)"
	@echo "  make eval-pipeline - Run full ada-eval BUILD/TEST/PROVE pipeline"
	@echo "  make prove         - Install gnatprove/gprbuild from alire-dev.toml"
	@echo "  make clean         - Remove generated outputs and caches"
	@echo ""
	@echo "Usage: make [target]   (default: help)"

sync: ## Install/update all dependencies
	UV_LINK_MODE=copy uv sync

download: ## Download and sanity-check the Qwen3-8B model
	HF_HUB_DISABLE_XET=1 uv run python training/download_model.py --model-name unsloth/Qwen3-8B --cache-dir models/qwen3-8b

check-model: ## Check if model exists; download if missing
	@if [ -f models/qwen3-8b/config.json ]; then \
		echo "Model already present at models/qwen3-8b - skipping download."; \
	else \
		echo "Model not found - running download..."; \
		$(MAKE) download; \
	fi

build-dataset: ## Build the training dataset from Ada source trees
	## Includes ../adacovex, ../Ada_CRDT, ../Ada-83-TLALOC, and ../ada-eval by default
	## Evaluation methodology is derived from ../ada-eval
	uv run python data/processing_scripts/build_dataset.py \
		--input-dir data/raw/ \
		--extra-input-dir ../adacovex \
		--extra-input-dir ../Ada_CRDT \
		--extra-input-dir ../Ada-83-TLALOC \
		--extra-input-dir ../ada-eval

train: ## Run QLoRA fine-tuning with Unsloth
	HF_HUB_DISABLE_XET=1 uv run python training/train_unsloth.py

generate: ## Generate Ada code with base and fine-tuned models
	uv run python eval/generate.py --model outputs/q3as --base-model unsloth/Qwen3-8B

eval: ## Run baseline evaluation (BLEU + compilation/test/SPARK metrics, base comparison)
	uv run python eval/baseline_eval.py --model outputs/q3as --base-model unsloth/Qwen3-8B

eval-pipeline: ## Run full ada-eval BUILD/TEST/PROVE pipeline
	uv run python eval/eval_pipeline.py --evals build test prove

prove: ## Install gnatprove, gprbuild, gnatformat from dev manifest
	alr build --manifest alire-dev.toml

all: check-model build-dataset train eval eval-pipeline ## Run full pipeline: download (if needed), build dataset, train, evaluate
	@echo ""
	@echo "=== Full pipeline complete ==="
	@echo "  Model: models/qwen3-8b"
	@echo "  Dataset: data/processed/dataset.jsonl"
	@echo "  Checkpoints: outputs/q3as"
	@echo "  Eval results: outputs/eval_results/"

clean: ## Remove generated outputs and caches
	rm -rf outputs/ models/ data/processed/dataset.jsonl data/processed/dataset_metadata.json
	@echo "Cleaned outputs/, models/, and generated dataset."
