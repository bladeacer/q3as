# q3as - Qwen 3 Ada SPARK

Specialized fine-tuning initiative targeting the full Ada language spectrum: **Ada 83, Ada 95, Ada 2005, Ada 2012, SPARK 2014, and Ada 2022**.

## Overview

`q3as` ingests Ada source trees (`.ads`, `.adb`, `.gpr`, `alire.toml`), pairs specification and implementation files, detects the target Ada standard via heuristic rules, and produces a standardized JSONL dataset formatted with the OpenAI/Qwen chat template (`system`, `user`, `assistant` messages) for QLoRA fine-tuning with Unsloth on 8 GB VRAM.

## Project Credits

This project builds on and draws data from the following upstream projects:

- **[adacovex](https://github.com/bladeacer/adacovex)** (`../adacovex`) — Zero-dependency Ada/SPARK command line tool for coverage analysis, proof verification, test-result parsing, and multi-standard safety-compliance assessment (DO-178C / ISO 26262 / IEC 62304). Provides Ada/SPARK source code with contract specifications for fine-tuning data.
- **[Ada_CRDT](https://github.com/bladeacer/Ada_CRDT)** (`../Ada_CRDT`) — Conflict-Free Replicated Data Types library for Ada/SPARK. Provides additional Ada specification/implementation pairs for fine-tuning data diversity.
- **[Ada-83-TLALOC](https://github.com/ViMoBr/Ada-83-TLALOC)** (`../Ada-83-TLALOC`) — Ada 83 compiler and test suite preserving the legacy of Ada 83 (MIL-STD-1815A-1983). Provides Ada 83-era source code for training data covering legacy Ada 83 patterns.
- **[ada-eval](https://github.com/AdaCore/ada-eval)** (`../ada-eval`) — Framework for evaluating LLM-based tools for Ada/SPARK use cases. Provides evaluation methodology, compacted and expanded dataset definitions (spark_learn, spark_custom, spark_human_eval_silver), and benchmark categories.

For training data that is not from my source code repositories, explicit permission was seeked beforehand. We want to be transparent with training data sources.

## Quick Start

```bash
# Install dependencies
uv sync

# Download and sanity-check the base model
uv run python training/download_model.py

# Build the dataset (includes ../adacovex, ../Ada_CRDT, and ../Ada-83-TLALOC by default)
uv run python data/processing_scripts/build_dataset.py --input-dir data/raw/ --extra-input-dir ../adacovex --extra-input-dir ../Ada_CRDT --extra-input-dir ../Ada-83-TLALOC

# Build the dataset with custom extra directories
uv run python data/processing_scripts/build_dataset.py --input-dir data/raw/ --extra-input-dir /path/to/project1 --extra-input-dir /path/to/project2

# Run training
uv run python training/train_unsloth.py

# Generate Ada code with both base and fine-tuned models
uv run python eval/generate.py --model outputs/q3as --base-model unsloth/Qwen3-8B

# Run full evaluation (BLEU + compilation/test/SPARK metrics, base comparison)
uv run python eval/baseline_eval.py --model outputs/q3as --base-model unsloth/Qwen3-8B

# Run the full ada-eval BUILD/TEST/PROVE pipeline
uv run python eval/eval_pipeline.py --evals build test prove
```

Alternatively, you can use the Makefile directly.

## Directory Structure

```
q3as/
├── pyproject.toml
├── README.md
├── data/
│   ├── raw/                      # Optional location for symlinks/git submodules
│   ├── processed/
│   └── processing_scripts/
│       └── build_dataset.py
├── training/
│   ├── train_unsloth.py
│   └── download_model.py
├── eval/
│   ├── baseline_eval.py          # BLEU + compliance + compilation/test/SPARK metrics
│   ├── generate.py               # Generate code with base and fine-tuned models
│   └── eval_pipeline.py          # Full ada-eval BUILD/TEST/PROVE pipeline
├── outputs/
│   ├── generated_solutions/      # Generated Ada code by model
│   ├── eval_results/             # Evaluation results from ada-eval
│   └── eval_results.json         # Combined evaluation summary
├── deploy/
│   └── Modelfile
```

## Standard Detection Heuristics

| Ada Standard | Keywords / Patterns |
|---|---|
| Ada 83 | No aspect specs, no `tagged`, no `interface` |
| Ada 95 | `tagged`, `abstract`, `override`, `interface` |
| Ada 2005 | `interfaces`, `aliased`, `protected type` |
| Ada 2012 | `Pre =>`, `Post =>`, `Type_Invariant`, `Subtype` aspects |
| Ada 2022 | `Static_Pure`, `Pure_Global`, `Contract_Cases`, `Loop_Invariant` |
| SPARK 2014 | `SPARK_Mode`, `Ghost`, `Praxis`, `GNATprove` |

## Evaluation Methodology

Evaluation metrics and dataset categories are derived from the **Ada Eval** project (`../ada-eval`), which defines:

- **spark_learn** - Learning examples with SPARK contracts
- **spark_custom** - Custom SPARK verification challenges
- **spark_human_eval_silver** - HumanEval-style silver standard evaluations

Compacted and expanded datasets are loaded from `../ada-eval/data/base/` to define benchmark categories and sample splits.

### Evaluation Metrics

The evaluation pipeline assesses model-generated Ada code across three dimensions:

1. **Compilation (BUILD)**: Runs `gprbuild` on generated code to verify it compiles without errors
2. **Unit Tests (TEST)**: Builds and runs unit tests via `gprbuild` and `./bin/tests`
3. **SPARK Verification (PROVE)**: Runs `gnatprove` to verify SPARK proof obligations

These metrics are compared between the base Qwen3-8B model and the fine-tuned q3as model to measure the improvement from fine-tuning.

### Alire Toolchain Management

Development tools (gnatprove, gprbuild, gnatformat) are managed via Alire:

- **`alire.toml`** — Clean publishing manifest (no dev toolchain dependencies)
- **`alire-dev.toml`** — Development manifest declaring `gnatprove`, `gnatdoc_bin`, `gnatformat_bin` as dev dependencies (modeled after `../adacovex/alire-dev.toml` and `../Ada_CRDT/alire-dev.toml`)

Install the dev toolchain with:
```bash
alr build --manifest alire-dev.toml
```

Or via Make:
```bash
make prove
```

## License

Apache 2.0.
