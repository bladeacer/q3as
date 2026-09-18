# q3as - Qwen 3 Ada SPARK

Specialized fine-tuning initiative targeting the full Ada language spectrum: **Ada 83, Ada 95, Ada 2005, Ada 2012, SPARK 2014, and Ada 2022**.

## Overview

`q3as` ingests Ada source trees (`.ads`, `.adb`, `.gpr`, `alire.toml`), pairs specification and implementation files, detects the target Ada standard via heuristic rules, and produces a standardized JSONL dataset formatted with the OpenAI/Qwen chat template (`system`, `user`, `assistant` messages) for QLoRA fine-tuning with Unsloth on 8 GB VRAM.

## Project Credits

This project builds on and draws data from the following upstream projects:

- **[Ada Covex](https://github.com/adacovex/adacovex)** (`../adacovex`) — Zero-dependency Ada/SPARK command line tool for coverage analysis, proof verification, test-result parsing, and multi-standard safety-compliance assessment (DO-178C / ISO 26262 / IEC 62304). Provides Ada/SPARK source code with contract specifications for fine-tuning data.
- **[Ada CRDT](https://github.com/bladeacer/Ada_CRDT)** (`../Ada_CRDT`) — Conflict-Free Replicated Data Types library for Ada/SPARK. Provides additional Ada specification/implementation pairs for fine-tuning data diversity.
- **[Ada-83-TLALOC](https://github.com/Ada-83-TLALOC/Ada-83-TLALOC)** (`../Ada-83-TLALOC`) — Ada 83 compiler and test suite preserving the legacy of Ada 83 (MIL-STD-1815A-1983). Provides Ada 83-era source code for training data covering legacy Ada 83 patterns.
- **[Ada Eval](https://github.com/ada-eval/ada-eval)** (`../ada-eval`) — Framework for evaluating LLM-based tools for Ada/SPARK use cases. Provides evaluation methodology, compacted and expanded dataset definitions (spark_learn, spark_custom, spark_human_eval_silver), and benchmark categories.

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

# Run baseline evaluation (methodology derived from ../ada-eval)
uv run python eval/baseline_eval.py --model outputs/q3as
```

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
│   └── baseline_eval.py
└── deploy/
    └── Modelfile
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

## License

Apache 2.0
