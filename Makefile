.PHONY: help sync download build-dataset train generate eval eval-pipeline prove clean all check-model

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "q3as - Qwen 3 Ada SPARK - Available targets:"
	@echo ""
	@echo "  make all           - Run full pipeline: model download (if needed), dataset, train (logs to training.log), generate, evaluate"
	@echo "  make sync          - Install/update all dependencies via uv sync"
	@echo "  make download      - Download and sanity-check the Qwen3-8B model (unsloth/Qwen3-8B -> models/qwen3-8b)"
	@echo "  make build-dataset - Build the training dataset from Ada source trees"
	@echo "                      (includes ../adacovex, ../Ada_CRDT, ../Ada-83-TLALOC, ../ada-eval by default)"
	@echo "  make train         - Run QLoRA fine-tuning with Unsloth on the local base model"
	@echo "  make generate      - Generate Ada code with the fine-tuned and base models"
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
	@if [ -f models/qwen3-8b/config.json ] && [ -f models/qwen3-8b/model.safetensors.index.json ] && \
		uv run python training/download_model.py --no-sanity-check >/dev/null 2>&1; then \
		echo "Model already present at models/qwen3-8b - skipping download."; \
	else \
		echo "Model not found or incomplete - running download..."; \
		$(MAKE) download; \
	fi

build-dataset: ## Build the training dataset from Ada source trees
	## Includes ../adacovex, ../Ada_CRDT, ../Ada-83-TLALOC, and ../ada-eval as Ada code sources,
	## ../learn (AdaCore courses, CC-BY-4.0) for doc-QA turns, and ../ada-spark (MIT) guidance in system prompts
	uv run python data/processing_scripts/build_dataset.py \
		--input-dir data/raw/ \
		--extra-input-dir ../adacovex \
		--extra-input-dir ../Ada_CRDT \
		--extra-input-dir ../Ada-83-TLALOC \
		--extra-input-dir ../ada-eval \
		--doc-dir ../learn \
		--guidance-dir ../ada-spark

train: ## Run QLoRA fine-tuning with Unsloth on the local base model (models/qwen3-8b)
	HF_HUB_DISABLE_XET=1 HF_DEACTIVATE_ASYNC_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
		uv run python -u training/train_unsloth.py 2>&1 | tee -a training.log

generate: ## Generate Ada code with the fine-tuned and base (local download) models
	uv run python eval/generate.py --model outputs/q3as --base-model models/qwen3-8b

eval: ## Run baseline evaluation (BLEU + compilation/test/SPARK metrics, base comparison)
	uv run python eval/baseline_eval.py --model outputs/q3as --base-model models/qwen3-8b

eval-pipeline: ## Run full ada-eval BUILD/TEST/PROVE pipeline
	uv run python eval/eval_pipeline.py --evals build test prove

prove: ## Install gnatprove, gprbuild, gnatformat from dev manifest
	alr build --manifest alire-dev.toml

all: check-model build-dataset train generate eval eval-pipeline ## Run full pipeline: download (if needed), build dataset, train, generate, evaluate
	@echo ""
	@echo "=== Full pipeline complete ==="
	@echo "  Base model: models/qwen3-8b (local download of unsloth/Qwen3-8B)"
	@echo "  Dataset: data/processed/dataset.jsonl"
	@echo "  Fine-tuned checkpoints: outputs/q3as"
	@echo "  Generated solutions: outputs/generated_solutions/"
	@echo "  Eval results: outputs/eval_results/ outputs/eval_results.json"

clean: ## Remove generated outputs and caches
	rm -rf outputs/ models/ data/processed/dataset.jsonl data/processed/dataset_metadata.json training.log
	@echo "Cleaned outputs/, models/, generated dataset, and training.log."
