# q3as - Qwen 3 Ada SPARK

Specialized fine-tuning initiative targeting the full Ada language spectrum: **Ada 83, Ada 95, Ada 2005, Ada 2012, SPARK 2014, and Ada 2022**.

## Overview

`q3as` ingests Ada source trees (`.ads`, `.adb`, `.gpr`, `alire.toml`), pairs specification and implementation files, detects the target Ada standard via heuristic rules, and produces a standardized JSONL dataset formatted with the OpenAI/Qwen chat template (`system`, `user`, `assistant` messages) for QLoRA fine-tuning with Unsloth on 8 GB VRAM.

## Quick Start

```bash
# Install dependencies
uv sync

# Download and sanity-check the base model
uv run python training/download_model.py

# Build the dataset from a source tree
uv run python data/processing_scripts/build_dataset.py --input-dir data/raw/

# Run training
uv run python training/train_unsloth.py

# Run baseline evaluation
uv run python eval/baseline_eval.py
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

## License

Apache 2.0.
