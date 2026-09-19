# AGENTS.md - q3as repository guide

q3as fine-tunes Qwen3-8B on Ada and SPARK code with Unsloth QLoRA, then
evaluates the result with the ada-eval framework. This file maps the
repository for coding agents: what lives where, how the pieces connect, and
which commands to use. Regenerate the tree below with `make agents-tree`.

## Pipeline at a glance

1. `make download` fetches `unsloth/Qwen3-8B` to `models/qwen3-8b/`.
2. `make build-dataset` walks Ada source trees of the sibling repos and emits
   `data/processed/dataset.jsonl` (chat-format turns: code, doc-QA, defect
   pairs, toolchain QA).
3. `make train` runs QLoRA fine-tuning (Unsloth) and writes checkpoints to
   `outputs/q3as/`.
4. `make generate` produces Ada solutions with both models into
   `outputs/generated_solutions/<label>/`.
5. `make eval` and `make eval-pipeline` score them (BLEU, compilation, unit
   tests, SPARK proofs) and write reports to `outputs/`.

## Where the important files live

- `data/processing_scripts/build_dataset.py` - the dataset builder. Discovers
  and pairs Ada sources, sanitizes prose into Simplified Technical English
  (STE), injects correct-vs-wrong defect pairs (five families), distills
  agent-skill guidance into system prompts, and writes JSONL plus metadata.
- `data/processing_scripts/` - dataset helpers live beside the builder.
- `training/download_model.py` - HF model download plus terminating sanity
  checks (light by default; deep GPU check runs in a child process with a
  timeout so the pipeline can never hang).
- `training/train_unsloth.py` - QLoRA training script (Unsloth, 2048-token
  packing, LoRA adapters on Qwen3-8B).
- `eval/generate.py` - batch generation for the fine-tuned and base models.
- `eval/baseline_eval.py` - BLEU/compliance metrics and result aggregation
  (`outputs/eval_results.json`).
- `eval/eval_pipeline.py` - BUILD/TEST/PROVE comparison report between base
  and fine-tuned models via ada-eval.
- `scripts/validate_defects.py` - compiles the dataset's defect pairs with
  the real GNAT and checks the claimed compiler messages. Exit code is the
  contract: any "compiles clean" defect is a failure.
- `scripts/ada_env.sh` - runs any command inside the Alire toolchain
  environment (`alr exec`); all Ada tool invocations go through it.
- `scripts/alire_env.py` - Python side of the same: `find_tool("gnatprove")`
  resolves binaries through the Alire environment for subprocess calls.
- `scripts/gen_agents_tree.py` - rewrites the file tree section below.
- `alire.toml` / `alire-dev.toml` - publishing and dev manifests. Dev-only
  toolchain deps (gnatprove) live in the dev manifest.
- `setup.sh` - one-shot bootstrap: shallow-clones the sibling repos, creates
  `.env` from `.env.dev` (never overwrites), runs `uv sync`, extracts local
  Python headers for Triton.
- `Makefile` - entry points for every step; `make help` lists them.
- `deploy/Modelfile` - Ollama deployment definition for the fine-tuned model.
- `tests/` - pytest unit tests for the dataset builder and helpers
  (`make test`).

## Ada toolchain through Alire

q3as never calls system `gnat`/`gprbuild`/`gnatprove` directly. The dev
manifest declares the toolchain; commands run through `alr exec`:

- Shell: `scripts/ada_env.sh <cmd>` (used by Makefile targets).
- Python: `from alire_env import find_tool` then pass
  `env=alire_env_path()` to `subprocess.run`.

`make prove` syncs the toolchain. Because alr 1.2.1 has no `--manifest`
option, it copies `alire-dev.toml` into the gitignored `.alire-dev/`
workspace and runs `alr update` there; the real manifests are never modified
by tooling.

## Conventions

- Dataset prose follows STE: short active sentences, no em-dashes, no
  hedging; code, flags, and quoted compiler output are exempt.
- Expected compiler messages in defect pairs are GNAT-verified strings.
  Change them only with `make validate-defects` evidence.
- Python: `uv` for env/deps, ruff + mypy clean for all touched files,
  pytest for the dataset logic.
- The sibling repos (`../adacovex`, `../Ada_CRDT`, `../Ada-83-TLALOC`,
  `../ada-eval`, `../learn`, `../ada-spark`, `../SimpleEnglish`,
  `../skills`) are inputs only; `./setup.sh` clones them. Their licenses are
  credited in `README.md`.

## Project tree

<!-- AGENTS:TREE-BEGIN -->
```text
q3as/
   data/
       processed/
           dataset.jsonl
           dataset_metadata.json
       processing_scripts/
           build_dataset.py
       raw/
   deploy/
       Modelfile
   eval/
       results/
       baseline_eval.py
       eval_pipeline.py
       generate.py
   scripts/
       ada_env.sh
       alire_env.py
       gen_agents_tree.py
       validate_defects.py
   tests/
       test_build_dataset.py
   training/
       download_model.py
       train_unsloth.py
   .env.dev
   .gitignore
   AGENTS.md
   alire-dev.toml
   alire.toml
   LICENSE
   Makefile
   pyproject.toml
   README.md
   run_download.py
   setup.sh
   uv.lock
```
<!-- AGENTS:TREE-END -->
