.PHONY: help setup sync download build-dataset parse-data gen-contracts check-integrity check-links train generate eval eval-pipeline eval-report bump-version prove ast-deps ada-env test lint validate-defects agents-tree fetch-sources clean all check-model

# Prerequisite chain for the dataset: parser outputs (docs chunks, AST units,
# contract turns) must exist before the builder runs. Every stage fingerprints
# its inputs and skips the rebuild when nothing changed (FORCE=1 overrides), so
# these targets stay cheap to re-run. gen-contracts also keeps a cached file
# when gnatprove is unavailable, so it cannot break a build; parse-data
# re-runs cheaply and regenerates any missing parser output.
DATASET_EXTRA_TURNS := \
	data/processed/docs_chunks.jsonl \
	data/processed/ada_ast_units.jsonl \
	data/processed/contract_mutations.jsonl

# Worker processes for the parsers and the builder. 0 means one per CPU core,
# which is what every script documents and what they do when the flag is
# omitted; the old default of 1 left 15 of 16 cores idle. Measured on the full
# corpus (6514 Ada files, 4337 pairs) on 16 cores: 11m30s at 1 worker, 4m39s at
# 8, and a 51s extract phase at 16 versus 7m38s at 1. Worker memory is ~90 MiB
# each and the builder's parent peaks at ~1.7 GiB holding every record, so
# cores run out before RAM does; set DATASET_WORKERS=1 to force serial.
DATASET_WORKERS ?= 0
MAX_NEW_TOKENS ?= 512
MAX_PROMPT_CHARS ?= 12000
TRAIN_FLAGS ?= --skip-merged-save

# Rebuild a stage even when its fingerprint matches.
FORCE_FLAG = $(if $(FORCE),--force,)

.PHONY: $(DATASET_EXTRA_TURNS)

$(DATASET_EXTRA_TURNS):
	$(MAKE) parse-data gen-contracts

# bash so `set -o pipefail` works for tee'd targets (training.log capture).
SHELL := /bin/bash

.DEFAULT_GOAL := help

# Local Python dev headers (no sudo): setup.sh extracts the pythonX.Y-dev
# packages into ~/.cache so Triton can compile its CUDA driver shim (needs
# Python.h). scripts/python_env.sh resolves the include path from the
# interpreter that will actually run training, so a different Python keeps
# working; it reports on stderr when the headers are missing. Recursively
# expanded, so it only runs for the targets that train.
PY_CPATH = $(shell bash scripts/python_env.sh)
EXPORT_HEADERS = $(if $(strip $(PY_CPATH)),CPATH=$(PY_CPATH),)

help: ## Show this help message
	@echo "q3as - Qwen 3 Ada SPARK - Available targets:"
	@echo ""
	@echo "  make all            - Run full pipeline: model download (if needed), dataset,"
	@echo "                        train (logs to training.log), generate, evaluate, report"
	@echo "  make setup          - One-shot bootstrap: fetch source repos into the archive"
	@echo "                        cache, .env, Alire index, uv sync, Python headers"
	@echo "  make sync           - Install/update all dependencies via uv sync"
	@echo "  make download       - Download and sanity-check the Qwen3-8B model"
	@echo "                        (Qwen/Qwen3-8B -> models/qwen3-8b)"
	@echo "  make check-model    - Download the model only when it is missing"
	@echo ""
	@echo "  make fetch-sources  - Fetch source repos into the archive cache (data/raw_repos)"
	@echo "  make parse-data     - Run the parser modules into data/processed/ extra JSONL"
	@echo "                        (doc chunking, libadalang/structural Ada AST extraction)"
	@echo "  make gen-contracts  - Generate gnatprove-verified contract turns (FORCE=1)"
	@echo "  make build-dataset  - Build the training dataset from Ada source trees"
	@echo "                        (cache: adacovex, Ada_CRDT, Ada-83-TLALOC, and the"
	@echo "                        RobertBoettcherSF Ada-Algorithms monorepo) plus the"
	@echo "                        parser outputs from parse-data and gen-contracts"
	@echo "  make ast-deps       - Build and install libadalang.so for the Ada AST parser"
	@echo "                        (optional; without it the parser uses its structural scanner)"
	@echo "  make check-integrity - Fail if any split file contains ada-eval evaluation content"
	@echo "  make check-links    - Check every markdown link/anchor resolves, and that no link points at a gitignored path"
	@echo ""
	@echo "  make train          - Run 8 GB-safe QLoRA fine-tuning with Unsloth (adapter-only)"
	@echo "  make generate       - Generate Ada code with bounded memory settings"
	@echo "  make eval           - Score the generated solutions against the ada-eval canonical"
	@echo "                        solutions (BLEU-4, exact match, compliance) plus ada-eval"
	@echo "                        compilation/test/proof for both models"
	@echo "  make eval-pipeline  - Run full ada-eval BUILD/TEST/PROVE pipeline"
	@echo "  make eval-report    - Write versioned result summary to docs/results/"
	@echo "  make validate-defects - GNAT-compile defect pairs and check the claimed messages"
	@echo ""
	@echo "  make prove          - Fetch the Alire dev toolchain (gnatprove, gnatformat)"
	@echo "  make ada-env        - Show the PATH alr exec provides (debug helper)"
	@echo "  make bump-version   - Bump version in the three Alire manifests + pyproject.toml"
	@echo "                        (VERSION=x.y.z or PART=major|minor|patch)"
	@echo "  make test           - Run the Python unit tests (pytest)"
	@echo "  make lint           - Run ruff, mypy, the link check, and the AGENTS tree check"
	@echo "  make agents-tree    - Regenerate the project file tree inside AGENTS.md"
	@echo "  make clean          - Remove generated outputs and caches"
	@echo ""
	@echo "Options: DATASET_WORKERS=<n> parser/build worker processes (default 0 ="
	@echo "         one per CPU core; 1 = serial. Output is identical for any value),"
	@echo "         FORCE=1 rebuild a stage even when its inputs are unchanged."
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
		$(FORCE_FLAG) \
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

bump-version: ## Bump version in the three Alire manifests + pyproject.toml (VERSION=x.y.z or PART=major|minor|patch)
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
	uv run python scripts/build_libadalang.py $(FORCE_FLAG,)

test: ## Run the Python unit tests
	uv run pytest tests/ -q

lint: ## Run ruff and mypy over the project sources, then check markdown links and the AGENTS tree
	uv run ruff check data/processing_scripts/ scripts/ eval/ tests/ tools/
	uv run mypy data/processing_scripts/build_dataset.py data/processing_scripts/parse_docs.py data/processing_scripts/parse_ada_ast.py data/processing_scripts/eval_guard.py data/processing_scripts/code_variants.py data/processing_scripts/stage_state.py data/processing_scripts/progress.py eval/ada_eval_common.py eval/baseline_eval.py scripts/alire_env.py scripts/bump_version.py scripts/build_libadalang.py scripts/gen_eval_report.py scripts/gen_agents_tree.py scripts/collect_cap_results.py scripts/make_probe_splits.py tools/check-links.py
	uv run python tools/check-links.py
	uv run python scripts/gen_agents_tree.py --check

validate-defects: ## Compile-check dataset defect pairs with the Alire GNAT
	uv run python scripts/validate_defects.py --source data/raw_repos/bladeacer/adacovex --source data/raw_repos/bladeacer/Ada_CRDT --source data/raw_repos/AdaCore/ada-eval --source data/raw_repos/RobertBoettcherSF/Ada-Algorithms

parse-data: fetch-sources ## Run the parser modules into data/processed/ extra JSONL
	uv run python data/processing_scripts/parse_docs.py --output data/processed/docs_chunks.jsonl --workers $(DATASET_WORKERS) $(FORCE_FLAG)
	uv run python data/processing_scripts/parse_ada_ast.py --output data/processed/ada_ast_units.jsonl --workers $(DATASET_WORKERS) $(FORCE_FLAG)

gen-contracts: ## Generate gnatprove-verified contract turns (skipped when unchanged; FORCE=1)
	uv run python scripts/gen_contract_mutations.py $(FORCE_FLAG)

check-integrity: ## Fail if any split file contains ada-eval evaluation content
	uv run python data/processing_scripts/eval_guard.py data/processed/dataset_train.jsonl data/processed/dataset_val.jsonl data/processed/dataset_test.jsonl

check-links: ## Check every markdown link/anchor resolves; fail on a link to a path git ignores
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
	rm -rf outputs/ models/ data/processed/dataset.jsonl data/processed/dataset_metadata.json \
		data/processed/dataset_train.jsonl data/processed/dataset_val.jsonl \
		data/processed/dataset_test.jsonl data/processed/.stages training.log
	@echo "Cleaned outputs/, models/, generated dataset, stage stamps, and training.log."
