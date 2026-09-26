.PHONY: help setup sync download build-dataset parse-data gen-contracts check-integrity check-links train generate eval eval-pipeline eval-report bump-version prove ast-deps ada-env test lint validate-defects agents-tree fetch-sources clean all check-model

# Prerequisite chain for the dataset: parser outputs (docs chunks, AST units,
# contract turns) must exist before the builder runs. gen-contracts silently
# keeps a cached file when gnatprove is unavailable, so it cannot break a
# build; parse-data re-runs cheaply and regenerates any missing parser output.
DATASET_EXTRA_TURNS := \
	data/processed/docs_chunks.jsonl \
	data/processed/ada_ast_units.jsonl \
	data/processed/contract_mutations.jsonl

DATASET_WORKERS ?= 1
MAX_NEW_TOKENS ?= 512
MAX_PROMPT_CHARS ?= 12000
TRAIN_FLAGS ?= --skip-merged-save

.PHONY: $(DATASET_EXTRA_TURNS)

$(DATASET_EXTRA_TURNS):
	$(MAKE) parse-data gen-contracts

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
	@echo "  make setup         - One-shot bootstrap: fetch source repos into the archive cache, .env, uv sync"
	@echo "  make sync          - Install/update all dependencies via uv sync"
	@echo "  make download      - Download and sanity-check the Qwen3-8B model (Qwen/Qwen3-8B -> models/qwen3-8b)"
	@echo "  make build-dataset - Build the training dataset from Ada source trees"
	@echo "                      (cache: adacovex, Ada_CRDT, Ada-83-TLALOC, and the RobertBoettcherSF Ada-Algorithms monorepo)"
	@echo "                      plus parser outputs (docs chunks, Ada AST units) when present"
	@echo "  make fetch-sources - Fetch source repos into the archive cache (data/raw_repos)"
	@echo "  make parse-data     - Run the parser modules into data/processed/ extra JSONL"
	@echo "                       (doc chunking, libadalang/structural Ada AST extraction)"
	@echo "  make ast-deps       - Build and install libadalang.so for the Ada AST parser"
	@echo "                       (optional; without it the parser uses its structural scanner)"
	@echo "  make train         - Run 8 GB-safe QLoRA fine-tuning with Unsloth (adapter-only by default)"
	@echo "  make generate      - Generate Ada code with bounded memory settings"
	@echo "  make eval          - Run baseline evaluation with BLEU + ada-eval metrics"
	@echo "                      (compilation, test, SPARK proof; base model comparison)"
	@echo "  make eval-pipeline - Run full ada-eval BUILD/TEST/PROVE pipeline"
	@echo "  make eval-report   - Write versioned result summary to docs/results/"
	@echo "  make bump-version  - Bump version in both Alire manifests (VERSION=x.y.z or PART=major|minor|patch)"
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

setup: ## One-shot bootstrap: fetch source repos into the cache, .env, uv sync, Python headers
	./setup.sh

fetch-sources: ## Fetch source repos into the archive cache (data/raw_repos); no-op when cached
	uv run python scripts/fetch_repos.py

download: ## Download and sanity-check the Qwen3-8B model
	HF_HUB_DISABLE_XET=1 uv run python training/download_model.py --model-name Qwen/Qwen3-8B --cache-dir models/qwen3-8b

check-model: ## Check if model exists; download if missing
	@if uv run python training/download_model.py --check-only >/dev/null 2>&1; then \
		echo "Model already present at models/qwen3-8b - skipping download."; \
	else \
		echo "Model not found or incomplete - running download..."; \
		$(MAKE) download; \
	fi

build-dataset: parse-data gen-contracts ## Build the training dataset from cached Ada source trees
	## Ada code sources (cache): adacovex, Ada_CRDT, Ada-83-TLALOC,
	## plus the RobertBoettcherSF Ada-Algorithms monorepo (MIT).
	## Doc sources (cache): learn, training_material (CC-BY-4.0). Guidance in system
	## prompts: ada-spark (MIT), SimpleEnglish (MIT, STE rules), skills (Apache-2.0).
	## Parser outputs (data/processed/*.jsonl from parse-data + gen-contracts) are
	## merged via --extra-turns. Also emits correct-vs-wrong defect pairs.
	uv run python data/processing_scripts/build_dataset.py \
		--input-dir data/raw/ \
		--extra-input-dir data/raw_repos/bladeacer/adacovex \
		--extra-input-dir data/raw_repos/bladeacer/Ada_CRDT \
		--extra-input-dir data/raw_repos/ViMoBr/Ada-83-TLALOC \
		--extra-input-dir data/raw_repos/RobertBoettcherSF/Ada-Algorithms \
		--doc-dir data/raw_repos/AdaCore/learn \
		--doc-dir data/raw_repos/AdaCore/training_material \
		--guidance-dir data/raw_repos/agent-sh/ada-spark \
		--guidance-dir data/raw_repos/AminBlg/SimpleEnglish \
		--guidance-dir data/raw_repos/AdaCore/skills \
		--workers $(DATASET_WORKERS) \
		$(foreach f,$(DATASET_EXTRA_TURNS),--extra-turns $(f))

train: ## Run QLoRA fine-tuning with Unsloth on the local base model (models/qwen3-8b); full stdout captured to training.log
	@set -o pipefail; \
	HF_HUB_DISABLE_XET=1 HF_DEACTIVATE_ASYNC_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(EXPORT_HEADERS) \
		uv run python -u training/train_unsloth.py $(TRAIN_FLAGS) 2>&1 | tee -a training.log

generate: ## Generate Ada code with the fine-tuned and base (local download) models
	uv run python eval/generate.py --model outputs/q3as --base-model models/qwen3-8b \
		--max-new-tokens $(MAX_NEW_TOKENS) --max-prompt-chars $(MAX_PROMPT_CHARS)

eval: ## Run baseline evaluation (BLEU + compilation/test/SPARK metrics, base comparison)
	uv run python eval/baseline_eval.py --model outputs/q3as --base-model models/qwen3-8b

eval-pipeline: ## Run full ada-eval BUILD/TEST/PROVE pipeline
	uv run python eval/eval_pipeline.py --evals build test prove

eval-report: ## Write versioned result summary to docs/results/ (version from alire.toml)
	uv run python scripts/gen_eval_report.py

bump-version: ## Bump version in alire.toml + alire-dev.toml (VERSION=x.y.z or PART=major|minor|patch)
	@if [ -n "$(VERSION)" ]; then \
		uv run python scripts/bump_version.py set $(VERSION); \
	else \
		uv run python scripts/bump_version.py bump $(PART); \
	fi

prove: ## Fetch the Alire dev toolchain (gnatprove, gnatdoc, gnatformat)
	## alr 1.2.1 has no --manifest option, so the dev manifest is copied into
	## the gitignored .alire-dev workspace and resolved there. Real manifests
	## are never modified.
	@mkdir -p .alire-dev
	@cmp -s alire-dev.toml .alire-dev/alire.toml || cp alire-dev.toml .alire-dev/alire.toml
	@cd .alire-dev && alr -n update

ada-env: ## Show the PATH alr exec provides (debug helper)
	@bash scripts/ada_env.sh printenv PATH | tr ':' '\n' | head -8

ast-deps: ## Build and install libadalang.so so the Ada AST parser uses real ASTs
	## The Alire `libadalang` crate ships a static library; the ctypes wrapper
	## in the `ast` dependency group needs a shared one. Resolves alire-ast.toml
	## in the gitignored .alire-ast workspace and builds it there, so the SPARK
	## toolchain in .alire-dev is untouched. Cached: a no-op once installed.
	uv run python scripts/build_libadalang.py $(if $(FORCE),--force,)

test: ## Run the Python unit tests
	uv run pytest tests/ -q

lint: ## Run ruff and mypy over the project sources, then check markdown links
	uv run ruff check data/processing_scripts/ scripts/ eval/ tests/ tools/
	uv run mypy data/processing_scripts/build_dataset.py data/processing_scripts/parse_docs.py data/processing_scripts/parse_ada_ast.py data/processing_scripts/eval_guard.py data/processing_scripts/code_variants.py scripts/alire_env.py scripts/bump_version.py scripts/build_libadalang.py scripts/gen_eval_report.py scripts/gen_agents_tree.py scripts/collect_cap_results.py scripts/make_probe_splits.py
	uv run python tools/check-links.py

validate-defects: ## Compile-check dataset defect pairs with the Alire GNAT
	uv run python scripts/validate_defects.py --source data/raw_repos/bladeacer/adacovex --source data/raw_repos/bladeacer/Ada_CRDT --source data/raw_repos/AdaCore/ada-eval --source data/raw_repos/RobertBoettcherSF/Ada-Algorithms

parse-data: fetch-sources ## Run the parser modules into data/processed/ extra JSONL
	uv run python data/processing_scripts/parse_docs.py --output data/processed/docs_chunks.jsonl --workers $(DATASET_WORKERS)
	uv run python data/processing_scripts/parse_ada_ast.py --output data/processed/ada_ast_units.jsonl --workers $(DATASET_WORKERS)

gen-contracts: ## Generate gnatprove-verified synthetic contract turns (cached; FORCE=1 to regenerate)
	uv run python scripts/gen_contract_mutations.py $(if $(FORCE),--force,)

check-integrity: ## Fail if any split file contains ada-eval evaluation content
	uv run python data/processing_scripts/eval_guard.py data/processed/dataset_train.jsonl data/processed/dataset_val.jsonl data/processed/dataset_test.jsonl

check-links: ## Check every relative markdown link/anchor resolves
	uv run python tools/check-links.py

agents-tree: ## Regenerate the project file tree section in AGENTS.md
	uv run python scripts/gen_agents_tree.py

all: check-model build-dataset train generate eval eval-pipeline eval-report ## Run full pipeline: download (if needed), build dataset, train, generate, evaluate, report
	@echo ""
	@echo "=== Full pipeline complete ==="
	@echo "  Base model: models/qwen3-8b (local download of Qwen/Qwen3-8B)"
	@echo "  Dataset: data/processed/dataset.jsonl"
	@echo "  Fine-tuned checkpoints: outputs/q3as"
	@echo "  Generated solutions: outputs/generated_solutions/"
	@echo "  Eval results: outputs/eval_results/ outputs/eval_results.json"

clean: ## Remove generated outputs and caches
	rm -rf outputs/ models/ data/processed/dataset.jsonl data/processed/dataset_metadata.json training.log
	@echo "Cleaned outputs/, models/, generated dataset, and training.log."
