# q3as - Qwen 3 Ada SPARK

Specialized fine-tuning initiative targeting the full Ada language spectrum: **Ada 83, Ada 95, Ada 2005, Ada 2012, SPARK 2014, and Ada 2022**.

`q3as` ingests Ada source trees (`.ads`, `.adb`, `.gpr`, `alire.toml`), pairs specification and implementation files, detects the target Ada standard via heuristic rules, and produces a standardized JSONL dataset formatted with the OpenAI/Qwen chat template (`system`, `user`, `assistant` messages) for QLoRA fine-tuning with Unsloth on 8 GB VRAM.

Every generated explanation follows **ASD-STE100 Simplified Technical English**: active voice, short sentences, no em-dashes, one word one meaning, technical terms defined at first use.

## Documentation

| Doc | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Pipeline overview and module map |
| [Datasets and training](docs/datasets-and-training.md) | Turn kinds, defect families, splits, training configuration |
| [Data provenance](docs/data-provenance.md) | Data sources, licenses, eval-integrity guard |
| [Toolchain setup](docs/toolchain-setup.md) | Alire management, vendored index for outdated `alr` |
| [Evaluation](docs/evaluation.md) | Benchmark, metrics, interpretation, running |
| [Results index](docs/results/README.md) | Per-version eval summaries and the last-3 comparison table |

## Project Credits

This project builds on and draws data from the following upstream projects. Licensing and provenance details are in [docs/data-provenance.md](docs/data-provenance.md).

- **[adacovex](https://github.com/bladeacer/adacovex)** (`../adacovex`) - Ada/SPARK coverage/proof/CLI tool; source with contract specifications. *(Apache-2.0)*
- **[Ada_CRDT](https://github.com/bladeacer/Ada_CRDT)** (`../Ada_CRDT`) - Conflict-free replicated data types for Ada/SPARK. *(MIT)*
- **[Ada-83-TLALOC](https://github.com/ViMoBr/Ada-83-TLALOC)** (`../Ada-83-TLALOC`) - Ada 83 compiler and test suite; [the author has given explicit permission to use this code for model training.](https://forum.ada-lang.io/t/fine-tuning-8b-ai-model-on-ada-spark/4746/3 ) *(GPL-3.0-or-later with GCC runtime exception; tests CC-BY-SA-4.0)*
- **[ada-eval](https://github.com/AdaCore/ada-eval)** (`../ada-eval`) - LLM evaluation framework for Ada/SPARK; provides our benchmark (and is treated as eval-proper, never as training data). *(Apache-2.0)*
- **[AdaCore/learn](https://github.com/AdaCore/learn)** (`../learn`) - AdaCore course material; Ada code blocks and sections become documentation-QA turns. *(CC-BY-4.0)*
- **[agent-sh/ada-spark](https://github.com/agent-sh/ada-spark)** (`../ada-spark`) - Agent skill for idiomatic, current Ada/SPARK; embedded into system prompts. *(MIT)*
- **[AminBlg/SimpleEnglish](https://github.com/AminBlg/SimpleEnglish)** (`../SimpleEnglish`) - ASD-STE100-style writing skill; governs all generated explanations. *(MIT)*
- **[AdaCore/skills](https://github.com/AdaCore/skills)** (`../skills`) - Official AdaCore toolchain skills (gnatprove, alire, gnatdoc, gnattest, gnatfuzz); become toolchain QA turns. *(Apache-2.0)*

## Quick Start

### 1. Set up the environment

```bash
./setup.sh          # everything
./setup.sh --repos  # only the sibling repositories
./setup.sh --deps   # only .env, uv sync, python headers
```

Or via Make: `make setup`.

The bootstrap shallow-clones the sibling repositories into the parent
directory, creates `.env` from `.env.dev`, installs dependencies with
`uv sync`, and (on distributions with an outdated `alr`) registers the
vendored Alire index so `gnatprove` 16.x and `gnatformat_bin` 26.x resolve
(see [docs/toolchain-setup.md](docs/toolchain-setup.md)).

The sibling repos stay outside this repository on purpose: the pipeline only reads their code and docs as training data, so vendoring copies inside the project would bloat the repo and blur licensing provenance.

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
# (--base-model defaults to the local download models/qwen3-8b, i.e. Qwen/Qwen3-8B)
uv run python eval/generate.py --model outputs/q3as --base-model models/qwen3-8b

# Run full evaluation (BLEU + compilation/test/SPARK metrics, base comparison)
uv run python eval/baseline_eval.py --model outputs/q3as --base-model models/qwen3-8b

# Run the full ada-eval BUILD/TEST/PROVE pipeline
uv run python eval/eval_pipeline.py --evals build test prove
```

Alternatively, you can use the Makefile directly.

## Hugging Face Token Setup

q3as downloads the **official [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)** checkpoint. Unsloth is the training framework only (patched kernels and QLoRA); the weights are Qwen's original release, never an unsloth-provisioned copy. If the repo requires accepting a license, you need a Hugging Face access token.

The downloaded base model lives at `models/qwen3-8b` and is the single source of truth for the pipeline: training (`train_unsloth.py`), generation (`generate.py`), and evaluation (`baseline_eval.py`) all default to that local copy of `Qwen/Qwen3-8B`, so the same downloaded weights are used end to end.

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

## Validation

```bash
make test              # pytest unit tests (builders, parsers, guard, injectors)
make lint              # ruff + mypy over the project sources
make validate-defects  # GNAT-compile dataset defect pairs, check claimed compiler messages
make check-integrity   # fail if any training split contains eval content
make agents-tree       # regenerate the file tree in AGENTS.md
```

## License

Apache 2.0.
