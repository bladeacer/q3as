# q3as - Qwen 3 Ada SPARK

Specialized fine-tuning initiative targeting the full Ada language spectrum: **Ada 83, Ada 95, Ada 2005, Ada 2012, SPARK 2014, and Ada 2022**.

## Overview

`q3as` ingests Ada source trees (`.ads`, `.adb`, `.gpr`, `alire.toml`), pairs specification and implementation files, detects the target Ada standard via heuristic rules, and produces a standardized JSONL dataset formatted with the OpenAI/Qwen chat template (`system`, `user`, `assistant` messages) for QLoRA fine-tuning with Unsloth on 8 GB VRAM.

### Training turn kinds

The dataset builder emits five kinds of turns:

| Kind | What it teaches |
|---|---|
| `code_pair` | Spec/body completion from real Ada trees |
| `defect_pair` | Correct code next to deliberately broken variants (syntax, missing context clause, visibility, contract, spec/body mismatch) with a diagnosis and the corrected code. The model learns when code goes wrong instead of hallucinating correctness. |
| `doc_qa` | Ada code blocks extracted from AdaCore course material, each with a plain completion answer and an STE-compliant explanation answer |
| `toolchain_qa` | Question and answer turns distilled from the AdaCore agent skills: gnatprove, alire, gnatdoc, gnattest, gnatfuzz |

### Writing style: Simplified Technical English

Every generated explanation follows **ASD-STE100 Simplified Technical English** (distilled from the SimpleEnglish agent skill): active voice, short sentences, no em-dashes, no hedges, one word one meaning, and each technical term defined at first use. The system prompts embed the rule block and a 32-term Ada/SPARK terminology glossary. A sanitizer rewrites prose (never code, identifiers, or quoted errors) and a self-check counts residual violations per build; the counts land in `dataset_metadata.json`.

## Project Credits

This project builds on and draws data from the following upstream projects:

- **[adacovex](https://github.com/bladeacer/adacovex)** (`../adacovex`) - Zero-dependency Ada/SPARK command line tool for coverage analysis, proof verification, test-result parsing, and multi-standard safety-compliance assessment (DO-178C / ISO 26262 / IEC 62304). Provides Ada/SPARK source code with contract specifications for fine-tuning data. *(Apache-2.0)*
- **[Ada_CRDT](https://github.com/bladeacer/Ada_CRDT)** (`../Ada_CRDT`) - Conflict-Free Replicated Data Types library for Ada/SPARK. Provides additional Ada specification/implementation pairs for fine-tuning data diversity. *(MIT)*
- **[Ada-83-TLALOC](https://github.com/ViMoBr/Ada-83-TLALOC)** (`../Ada-83-TLALOC`) - Ada 83 compiler and test suite preserving the legacy of Ada 83 (MIL-STD-1815A-1983). Provides Ada 83-era source code for training data covering legacy Ada 83 patterns. *(GPL-3.0-or-later with GCC runtime exception; test suite CC-BY-SA-4.0)* - **the author has given explicit permission to use this code for model training.**
- **[ada-eval](https://github.com/AdaCore/ada-eval)** (`../ada-eval`) - Framework for evaluating LLM-based tools for Ada/SPARK use cases. Provides evaluation methodology, compacted and expanded dataset definitions (spark_learn, spark_custom, spark_human_eval_silver), and benchmark categories; its sample sources are also used as Ada code for training data. *(Apache-2.0)*
- **[AdaCore/learn](https://github.com/AdaCore/learn)** (`../learn`) - Sources for AdaCore's learn.adacore.com website: courses (intro-to-ada, intro-to-spark, advanced-ada, advanced-spark, Guidelines for Safe and Secure Ada/SPARK, GNAT toolchain intros, domain-specific AdaCore technologies), booklets, and labs. Ada code blocks embedded in the RST course material are extracted as documentation-QA style training turns. *(CC-BY-4.0)*
- **[agent-sh/ada-spark](https://github.com/agent-sh/ada-spark)** (`../ada-spark`) - An agent skill that teaches coding agents to write idiomatic, correct, current Ada and SPARK: a stale-to-current correction map (GNAT Community to Alire + GNAT FSF, `pragma Precondition` to `Pre`/`Post` aspects, CodePeer to GNAT SAS), SPARK assurance levels and proof guidance, and embedded/Ravenscar profiles. SKILL.md and agent-knowledge guidance are embedded into training system prompts. *(MIT)*
- **[AminBlg/SimpleEnglish](https://github.com/AminBlg/SimpleEnglish)** (`../SimpleEnglish`) - An agent skill that makes LLMs write plain English with the discipline of ASD-STE100 Simplified Technical English: short sentences, active voice, simple tenses, one word one meaning, condition before command, no em-dashes, every technical term defined at first use. Its writing rules and slop-to-plain word map govern every explanation this project generates, and a distilled rule block is embedded into training system prompts. *(MIT)*
- **[AdaCore/skills](https://github.com/AdaCore/skills)** (`../skills`) - AdaCore's official agent skills for their toolchain: gnatprove, alire, gnatdoc, gnattest, and gnatfuzz. The SKILL.md files are embedded into training system prompts and converted to toolchain question/answer turns, so the model learns how gnatprove and the rest of the toolchain are actually used (invocation, output reading, proof workflow, Alire crate management). *(Apache-2.0)*

For training data that is not from my source code repositories, explicit permission was seeked beforehand. We want to be transparent with training data sources. Licensing summary: Apache-2.0 and MIT code is compatible with permissive redistribution with attribution; CC-BY-4.0 course material is used with attribution; the GPL-licensed Ada-83-TLALOC code is used for model training only (weights are not source-code redistribution) and is additionally covered by the author's explicit permission. The SimpleEnglish skill paraphrases the ASD-STE100 standard and reproduces no spec text or dictionary content; the same discipline applies to our distilled rule block.

## Quick Start

### 1. Set up the environment

Run the one-shot bootstrap. It shallow-clones the sibling data and guidance repositories into the parent directory, creates `.env` from `.env.dev`, installs dependencies with `uv sync`, and extracts local Python headers for Triton when needed:

```bash
./setup.sh          # everything
./setup.sh --repos  # only the sibling repositories
./setup.sh --deps   # only .env, uv sync, python headers
```

Or via Make: `make setup`.

The pipeline reads training/eval data from shallow clones in the **parent directory** of this repo:

```bash
git clone --depth 1 https://github.com/bladeacer/adacovex.git        ../adacovex
git clone --depth 1 https://github.com/bladeacer/Ada_CRDT.git        ../Ada_CRDT
git clone --depth 1 https://github.com/ViMoBr/Ada-83-TLALOC.git      ../Ada-83-TLALOC
git clone --depth 1 https://github.com/AdaCore/ada-eval.git          ../ada-eval
git clone --depth 1 https://github.com/AdaCore/learn.git             ../learn
git clone --depth 1 https://github.com/agent-sh/ada-spark.git        ../ada-spark
git clone --depth 1 https://github.com/AminBlg/SimpleEnglish.git     ../SimpleEnglish
git clone --depth 1 https://github.com/AdaCore/skills.git            ../skills
```

These stay outside the q3as repository on purpose: the model only reads their code/docs as training data, so there is no reason to vendor copies inside the project (and doing so would bloat the repo and blur licensing provenance).

### 2. Install and run

```bash
# Install dependencies
uv sync

# Set up Hugging Face credentials (see below)
cp .env.dev .env

# Download and sanity-check the base model
uv run python training/download_model.py

# Build the dataset (siblings listed above; see make build-dataset for the exact source list)
uv run python data/processing_scripts/build_dataset.py --input-dir data/raw/ --extra-input-dir ../adacovex --extra-input-dir ../Ada_CRDT --extra-input-dir ../Ada-83-TLALOC --extra-input-dir ../ada-eval

# Build the dataset with custom extra directories
uv run python data/processing_scripts/build_dataset.py --input-dir data/raw/ --extra-input-dir /path/to/project1 --extra-input-dir /path/to/project2

# Run training (console output is appended to training.log for post-mortem debugging)
uv run python training/train_unsloth.py

# Generate Ada code with both base and fine-tuned models
# (--base-model defaults to the local download models/qwen3-8b, i.e. unsloth/Qwen3-8B)
uv run python eval/generate.py --model outputs/q3as --base-model models/qwen3-8b

# Run full evaluation (BLEU + compilation/test/SPARK metrics, base comparison)
uv run python eval/baseline_eval.py --model outputs/q3as --base-model models/qwen3-8b

# Run the full ada-eval BUILD/TEST/PROVE pipeline
uv run python eval/eval_pipeline.py --evals build test prove
```

Alternatively, you can use the Makefile directly.

## Hugging Face Token Setup

The model download script uses `huggingface_hub` with `hf_transfer` for accelerated downloads. If you are downloading gated models (e.g. `unsloth/Qwen3-8B`), you need a Hugging Face access token.

The downloaded base model lives at `models/qwen3-8b` and is the single source of truth for the pipeline: training (`train_unsloth.py`), generation (`generate.py`), and evaluation (`baseline_eval.py`) all default to that local copy of `unsloth/Qwen3-8B`, so the same downloaded weights are used end to end.

1. Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Copy the placeholder environment file:
   ```bash
   cp .env.dev .env
   ```
3. Edit `.env` and replace `your_huggingface_token_here` with your actual token
4. The `.env` file is git-ignored, so your token will not be committed

The token is loaded automatically from `.env` when running `download_model.py`. You can also set it manually:

```bash
export HF_TOKEN="your_huggingface_token_here"
uv run python training/download_model.py
```

## Directory Structure

```
q3as/
├── setup.sh                      # One-shot bootstrap: sibling repos, .env, deps
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

Development tools (gnatprove, gprbuild, gnat) are managed via Alire. All Ada tool invocations run through `alr exec` (never against system binaries):

- **`alire.toml`** - Clean publishing manifest (no dev toolchain dependencies)
- **`alire-dev.toml`** - Development manifest declaring the dev toolchain (modeled after `../adacovex/alire-dev.toml` and `../Ada_CRDT/alire-dev.toml`)
- **`scripts/ada_env.sh`** - Shell wrapper: runs any command inside the Alire environment
- **`scripts/alire_env.py`** - Python helper: `find_tool("gnatprove")` resolves the managed binary
- **`.alire-dev/`** - Gitignored throwaway Alire workspace (a copy of the dev manifest; the real manifests are never modified by tooling)

Fetch the dev toolchain with:
```bash
make prove
```

`alr 1.2.1` has no `--manifest` option, so `make prove` copies `alire-dev.toml` into `.alire-dev/` and resolves there. The eval pipeline and the defect validator resolve every tool through this environment and refuse to run against a missing managed tool.

### Tests and validation

```bash
make test              # pytest unit tests (dataset builder, defect injector, sanitizers)
make lint              # ruff + mypy over the project sources
make validate-defects  # GNAT-compile dataset defect pairs, check claimed compiler messages
make agents-tree       # regenerate the file tree in AGENTS.md
```

## License

Apache 2.0.
