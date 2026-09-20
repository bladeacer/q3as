.PHONY: help setup sync download build-dataset parse-data check-integrity train generate eval eval-pipeline prove ada-env test lint validate-defects agents-tree clean all check-model

# bash so `set -o pipefail` works for tee'd targets (training.log capture).
SHELL := /bin/bash

.DEFAULT_GOAL := help

# Local Python dev headers (no sudo): python3.13-dev debs extracted to ~/.cache
# so Triton can compile its CUDA driver shim (needs Python.h).
PY_HDR_ROOT ?= $(HOME)/.cache/q3as-python-headers/usr/include
PY_HDRS := $(PY_HDR_ROOT)/python3.13:$(PY_HDR_ROOT)/x86_64-linux-gnu:$(PY_HDR_ROOT)
EXPORT_HEADERS := $(if $(wildcard $(PY_HDR_ROOT)/python3.13/Python.h),CPATH=$(PY_HDRS),)

help: ## Show this help message
	@echo "q3as - Qwen 3 Ada SPARK - Available targets:"
	@echo ""
	@echo "  make all           - Run full pipeline: model download (if needed), dataset, train (logs to training.log), generate, evaluate"
	@echo "  make setup         - One-shot bootstrap: shallow-clone sibling repos (data + skills), .env, uv sync"
	@echo "  make sync          - Install/update all dependencies via uv sync"
	@echo "  make download      - Download and sanity-check the Qwen3-8B model (unsloth/Qwen3-8B -> models/qwen3-8b)"
	@echo "  make build-dataset - Build the training dataset from Ada source trees"
	@echo "                      (includes ../adacovex, ../Ada_CRDT, ../Ada-83-TLALOC, ../ada-eval by default)"
	@echo "                      plus parser outputs (docs chunks, Ada AST units) when present"
	@echo "  make parse-data    - Run the parser modules: heading-aware doc chunking and"
	@echo "                       libadalang/structural Ada AST extraction into JSONL"
	@echo "  make train         - Run QLoRA fine-tuning with Unsloth on the local base model"
	@echo "  make generate      - Generate Ada code with the fine-tuned and base models"
	@echo "  make eval          - Run baseline evaluation with BLEU + ada-eval metrics"
	@echo "                      (compilation, test, SPARK proof; base model comparison)"
	@echo "  make eval-pipeline - Run full ada-eval BUILD/TEST/PROVE pipeline"
	@echo "  make prove         - Fetch the Alire dev toolchain (gnatprove etc. from alire-dev.toml)"
	@echo "  make test          - Run the Python unit tests (pytest)"
	@echo "  make validate-defects - GNAT-compile defect pairs from the sibling repos and check"
	@echo "                       the claimed compiler messages (scripts/validate_defects.py)"
	@echo "  make agents-tree   - Regenerate the project file tree inside AGENTS.md"
	@echo "  make lint          - Run ruff and mypy over the project sources"
	@echo "  make clean         - Remove generated outputs and caches"
	@echo ""
	@echo "Usage: make [target]   (default: help)"

sync: ## Install/update all dependencies
	UV_LINK_MODE=copy uv sync

setup: ## One-shot bootstrap: shallow-clone sibling repos, .env, uv sync, Python headers
	./setup.sh

download: ## Download and sanity-check the Qwen3-8B model
	HF_HUB_DISABLE_XET=1 uv run python training/download_model.py --model-name unsloth/Qwen3-8B --cache-dir models/qwen3-8b

check-model: ## Check if model exists; download if missing
	@if uv run python training/download_model.py --check-only >/dev/null 2>&1; then \
		echo "Model already present at models/qwen3-8b - skipping download."; \
	else \
		echo "Model not found or incomplete - running download..."; \
		$(MAKE) download; \
	fi

build-dataset: ## Build the training dataset from Ada source trees
	## Includes ../adacovex, ../Ada_CRDT, ../Ada-83-TLALOC, and ../ada-eval as Ada code sources,
	## ../learn (AdaCore courses, CC-BY-4.0) for doc-QA turns, and agent-skill guidance in
	## system prompts: ../ada-spark (MIT), ../SimpleEnglish (MIT, STE rules), ../skills (Apache-2.0,
	## AdaCore toolchain skills). Also emits correct-vs-wrong defect pairs.
	uv run python data/processing_scripts/build_dataset.py \
		--input-dir data/raw/ \
		--extra-input-dir ../adacovex \
		--extra-input-dir ../Ada_CRDT \
		--extra-input-dir ../Ada-83-TLALOC \
		--extra-input-dir ../ada-eval \
		--doc-dir ../learn \
		--guidance-dir ../ada-spark \
		--guidance-dir ../SimpleEnglish \
		--guidance-dir ../skills

train: ## Run QLoRA fine-tuning with Unsloth on the local base model (models/qwen3-8b); full stdout captured to training.log
	@set -o pipefail; \
	HF_HUB_DISABLE_XET=1 HF_DEACTIVATE_ASYNC_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(EXPORT_HEADERS) \
		uv run python -u training/train_unsloth.py 2>&1 | tee -a training.log

generate: ## Generate Ada code with the fine-tuned and base (local download) models
	uv run python eval/generate.py --model outputs/q3as --base-model models/qwen3-8b

eval: ## Run baseline evaluation (BLEU + compilation/test/SPARK metrics, base comparison)
	uv run python eval/baseline_eval.py --model outputs/q3as --base-model models/qwen3-8b

eval-pipeline: ## Run full ada-eval BUILD/TEST/PROVE pipeline
	uv run python eval/eval_pipeline.py --evals build test prove

prove: ## Fetch the Alire dev toolchain (gnatprove, gnatdoc, gnatformat)
	## alr 1.2.1 has no --manifest option, so the dev manifest is copied into
	## the gitignored .alire-dev workspace and resolved there. Real manifests
	## are never modified.
	@mkdir -p .alire-dev
	@cmp -s alire-dev.toml .alire-dev/alire.toml || cp alire-dev.toml .alire-dev/alire.toml
	@cd .alire-dev && alr -n update

ada-env: ## Show the PATH alr exec provides (debug helper)
	@bash scripts/ada_env.sh printenv PATH | tr ':' '\n' | head -8

test: ## Run the Python unit tests
	uv run pytest tests/ -q

lint: ## Run ruff and mypy over the project sources
	uv run ruff check data/processing_scripts/ scripts/ eval/ tests/
	uv run mypy data/processing_scripts/build_dataset.py data/processing_scripts/parse_docs.py data/processing_scripts/parse_ada_ast.py data/processing_scripts/eval_guard.py scripts/alire_env.py scripts/gen_agents_tree.py

validate-defects: ## Compile-check dataset defect pairs with the Alire GNAT
	uv run python scripts/validate_defects.py

parse-data: ## Run the parser modules into data/processed/ extra JSONL
	uv run python data/processing_scripts/parse_docs.py --input-dir ../learn --output data/processed/docs_chunks.jsonl
	uv run python data/processing_scripts/parse_ada_ast.py --input-dir ../adacovex --input-dir ../Ada_CRDT --input-dir ../ada-eval --output data/processed/ada_ast_units.jsonl

check-integrity: ## Fail if any split file contains ada-eval evaluation content
	uv run python data/processing_scripts/eval_guard.py data/processed/dataset_train.jsonl data/processed/dataset_val.jsonl data/processed/dataset_test.jsonl

agents-tree: ## Regenerate the project file tree section in AGENTS.md
	uv run python scripts/gen_agents_tree.py

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
