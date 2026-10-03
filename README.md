# q3as - Qwen 3 Ada SPARK

Specialized fine-tuning initiative targeting the full Ada language spectrum: **Ada 83, Ada 95, Ada 2005, Ada 2012, SPARK 2014, and Ada 2022**.

`q3as` ingests Ada source trees (`.ads`, `.adb`, `.gpr`,
[`alire.toml`](alire.toml)), pairs specification and implementation files,
detects the target Ada standard via heuristic rules, and produces a standardized
JSONL dataset formatted with the OpenAI/Qwen chat template (`system`, `user`,
`assistant` messages) for QLoRA fine-tuning with Unsloth on 8 GB VRAM.

Every generated explanation follows **ASD-STE100 Simplified Technical English**: active voice, short sentences, no em-dashes, one word one meaning, technical terms defined at first use.

## Documentation

Full index, with a reading order: [docs/README.md](docs/README.md).

| Doc | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Pipeline overview and module map |
| [Datasets and training](docs/datasets-and-training.md) | Turn kinds, defect families, splits, training configuration |
| [Data provenance](docs/data-provenance.md) | Data sources, licenses, eval-integrity guard |
| [Toolchain setup](docs/toolchain-setup.md) | Alire management, vendored index for outdated `alr` |
| [Evaluation](docs/evaluation.md) | Benchmark, metrics, interpretation, running |
| [Docs index](docs/README.md) | Every documentation page, and how they fit together |
| [Results index](docs/results/README.md) | Per-version eval summaries and a cross-version comparison table |
| [Changelog index](docs/changelogs/index.md) | What changed in each version, one file per release |

## Project Credits

This project builds on and draws data from the following upstream projects.
Every one of them is fetched into the gitignored archive cache
`data/raw_repos/<owner>/<repo>` by `make fetch-sources`, and each cache entry
records the upstream commit it came from (`make check-sources` reports what
moved upstream, `make update-sources` re-fetches it). The licenses below were
read from each cached repository's own `LICENSE` file; the authoritative
version of this table, with the cache paths, the per-source detail, and the
eval-integrity rules, is [docs/data-provenance.md](docs/data-provenance.md).

| Source | What q3as takes from it | License |
|---|---|---|
| [bladeacer/adacovex](https://github.com/bladeacer/adacovex) | Ada/SPARK coverage, proof, and CLI tool. Source with contract specifications becomes code, defect, and AST turns; `docs/usage` and `docs/archive` become assurance-ladder QA | Apache-2.0 |
| [bladeacer/Ada_CRDT](https://github.com/bladeacer/Ada_CRDT) | Conflict-free replicated data types for Ada/SPARK. Spec/body pairs add diversity; `docs/proof` and `docs/compliance` become assurance-ladder QA | MIT |
| [ViMoBr/Ada-83-TLALOC](https://github.com/ViMoBr/Ada-83-TLALOC) | Ada 83 compiler and test suite: Ada 83-era source for legacy patterns. [The author has given explicit permission to use this code for model training](https://forum.ada-lang.io/t/fine-tuning-8b-ai-model-on-ada-spark/4746/3) | GPL-3.0-or-later with GCC runtime exception; tests CC-BY-SA-4.0 |
| [AdaCore/ada-eval](https://github.com/AdaCore/ada-eval) | LLM evaluation framework for Ada/SPARK. It provides the 19-sample benchmark and is **eval-proper**: never training data. It is also the source of the eval-integrity guard and the uv path dependency for the eval tooling | Apache-2.0 |
| [AdaCore/learn](https://github.com/AdaCore/learn) | AdaCore course material. Ada code blocks become documentation-QA turns, and the sections become doc-section turns | CC-BY-4.0 |
| [AdaCore/training_material](https://github.com/AdaCore/training_material) | AdaCore training courses (RST): code blocks and sections become documentation turns, and the labs ship `prompt/` and `answer/` trees that become completion and explanation turns | CC-BY-4.0 |
| [agent-sh/ada-spark](https://github.com/agent-sh/ada-spark) | Agent skill for idiomatic, current Ada/SPARK, distilled into the system prompts | MIT |
| [AminBlg/SimpleEnglish](https://github.com/AminBlg/SimpleEnglish) | ASD-STE100-style writing rules and word map, paraphrased (no spec text). It governs every generated explanation | MIT |
| [AdaCore/skills](https://github.com/AdaCore/skills) | Official AdaCore toolchain skills (gnatprove, alire, gnatdoc, gnattest, gnatfuzz), which become toolchain QA turns | Apache-2.0 |
| [RobertBoettcherSF/Ada-Algorithms](https://github.com/RobertBoettcherSF/Ada-Algorithms) | One monorepo of Ada/SPARK algorithm implementations (distributed systems, graph algorithms, image processing, compression, SPARK-verified sheets, parsers, and more) in category directories, several thousand files. The code is AI assisted. The 16 `SPARK2` topic trees are proved with gnatprove, and only the subprograms the prover discharges become training turns. [The author approved training use outside the repository](https://forum.ada-lang.io/t/fine-tuning-8b-ai-model-on-ada-spark/4746/6) | MIT |

The Ada toolchain q3as parses and proves with is a separate set of inputs that
never reach the model: `libadalang` (Apache-2.0 WITH LLVM-exception) and the
GNAT tools resolved through Alire. They are listed with their licenses in
[data provenance](docs/data-provenance.md#toolchain-inputs-not-training-data). 

## Quick Start

### 1. Set up the environment

```bash
./setup.sh          # everything
./setup.sh --repos  # only the source repositories (archive cache)
./setup.sh --deps   # only .env, uv sync, python headers
```

Or via Make: `make setup`.

The bootstrap fetches the source repositories into the local archive cache
(`data/raw_repos/`, gitignored), creates `.env` from [`.env.dev`](.env.dev),
installs dependencies with `uv sync`, and (on distributions with an outdated
`alr`) registers the vendored Alire index so `gnatprove` 16.x and
`gnatformat_bin` 26.x resolve (see
[docs/toolchain-setup.md](docs/toolchain-setup.md)).

Source repos are cached outside version control on purpose: the pipeline
only reads their code and docs as training data, so committing copies would
bloat the repo and blur licensing provenance. Downloads are plain HTTP
tarballs (no git), cached by repository identity: a re-run re-fetches
nothing already on disk (`make fetch-sources`, `--refresh` to force).

### 2. Install and run

```bash
# Install dependencies
uv sync

# Set up Hugging Face credentials (see below)
cp .env.dev .env

# Download and sanity-check the base model
uv run python training/download_model.py

# Build the dataset. Use the Make target: it wires the cached source trees,
# the doc and guidance directories, and the parser outputs together.
make build-dataset

# The builder underneath it, if you want to point it somewhere else. The
# upstream sources live in the archive cache, not in data/raw/, so a run with
# --input-dir alone reads nothing.
uv run python data/processing_scripts/build_dataset.py \
  --input-dir data/raw/ \
  --extra-input-dir data/raw_repos/bladeacer/adacovex \
  --extra-input-dir data/raw_repos/RobertBoettcherSF/Ada-Algorithms

# Run training (console output is appended to training.log for post-mortem
# debugging; make train additionally passes --skip-merged-save)
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

## Approximate runtimes

Measured end to end on the reference machine (RTX 5050 Laptop, 8 GB VRAM;
warm caches: base model downloaded, source cache and parser outputs
present):

| Step | Duration |
|---|---|
| `make build-dataset` (parse-data + contracts cached) | 2 to 5 min |
| `make train` (500 steps, batch 1 x grad-accum 8) | ~2.5 h measured: ~14 s/step plus ~30 min of evaluation passes; the dominant cost |
| `make generate` (19 + 19 samples, both models) | ~28 min |
| `make eval-pipeline` (build/test/prove, 19 + 19) | ~4 min |
| `make eval` (BLEU + compliance) | ~1 min |
| `make eval-report` | seconds |

`make all` takes roughly 3.5 hours with the default 500-step budget. The
default pipeline is tuned for an 8 GB GPU and an 8 GB host: dataset
preparation uses one worker per core, training uses a 1024-token window and one
tokenization process, the adapter-only output skips the merged 16-bit export,
and generation uses 12,000 prompt characters and 512 new tokens per model.
Override `DATASET_WORKERS`, `MAX_PROMPT_CHARS`, `MAX_NEW_TOKENS`, or
`TRAIN_FLAGS` on the `make` command line when more memory is available. Notes:

- **The 500-step budget is deliberate.** One pass over the 72,547 train
  records is 9,013 steps at 8 records per step, so `--max-steps 500` is 0.06
  of one epoch. That is the point: an epoch is 30 to 48 h on this card, and a
  train, generate, evaluate loop that comes back in a few hours is what makes
  the next thing to improve findable. A 500-step run is an under-trained model
  on purpose (train loss was still falling when the budget ran out). Raise
  `--max-steps` for a long run; it resumes from the last checkpoint, so raising
  it on an existing output directory continues rather than restarts.
- Early stopping cannot fire inside a 500-step run: patience is 10
  evaluations at one every 50 steps, so the eleventh point would be step 550.
  Plan for the whole budget, or lower `--max-steps` to end sooner.
- A cold start adds the Qwen3-8B download (~16 GB) and the first
  `make fetch-sources` (one monolithic Ada-Algorithms tarball); both depend on
  bandwidth. One-time extras: `uv sync` and `make prove` (Alire
  toolchain). `make ast-deps` is optional and adds a ~20 minute one-time
  libadalang source build; it upgrades the dataset's Ada AST extraction from
  the regex scanner to real ASTs. See
  [toolchain setup](docs/toolchain-setup.md).

## Hugging Face Token Setup

q3as downloads the **official [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)** checkpoint. Unsloth is the training framework only (patched kernels and QLoRA); the weights are Qwen's original release, never an unsloth-provisioned copy. If the repo requires accepting a license, you need a Hugging Face access token.

The downloaded base model lives at `models/qwen3-8b` and is the single source of
truth for the pipeline: training
([`train_unsloth.py`](training/train_unsloth.py)), generation
([`generate.py`](eval/generate.py)), and evaluation
([`baseline_eval.py`](eval/baseline_eval.py)) all default to that local copy of
`Qwen/Qwen3-8B`, so the same downloaded weights are used end to end.

1. Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Copy the placeholder environment file:
   ```bash
   cp .env.dev .env
   ```
3. Edit `.env` and replace `your_huggingface_token_here` with your actual token
4. The `.env` file is git-ignored, so your token will not be committed

The token is loaded automatically from `.env` when running
[`download_model.py`](training/download_model.py). You can also set it manually:

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
