# AGENTS.md - q3as repository guide

q3as fine-tunes Qwen3-8B on Ada and SPARK code with Unsloth QLoRA, then
evaluates the result with the ada-eval framework. This file maps the
repository for coding agents: what lives where, how the pieces connect, and
which commands to use. Regenerate the tree below with `make agents-tree`.

## Pipeline at a glance

1. `make download` fetches the official `Qwen/Qwen3-8B` to `models/qwen3-8b/`
   (Unsloth is the training framework only, never the weight source).
2. `make parse-data` runs the parser modules (doc chunking, AST extraction)
   into `data/processed/*.jsonl`.
3. `make build-dataset` walks Ada source trees of the sibling repos, ingests
   the parser outputs, and emits `data/processed/dataset.jsonl` plus the
   train/val/test split files (chat-format turns: code, doc-QA, defect
   pairs, contract-writing, toolchain QA).
4. `make train` runs QLoRA fine-tuning (Unsloth) with seeded early stopping
   and writes checkpoints to `outputs/q3as/`.
5. `make generate` produces Ada solutions with both models into
   `outputs/generated_solutions/<label>/`.
6. `make eval` and `make eval-pipeline` score them (BLEU, compilation, unit
   tests, SPARK proofs) and write reports to `outputs/`; `make eval-report`
   writes the versioned summary to `docs/results/result-vX.Y.Z.md` (version
   from `alire.toml`, bumped with `make bump-version`).

End-user and developer documentation lives in `docs/` (architecture,
datasets and training, data provenance, toolchain setup, evaluation);
README links to it. Keep those pages current when behavior changes.

## Data provenance and eval integrity

**Full documentation: `docs/data-provenance.md` - keep it in sync.** When a
new data source is added (new `--input-dir`/`--extra-input-dir`, new parser
source, new sibling repo), update that page's source table and license
notes, then run `make check-integrity` before building. The same applies
when the guard, the defect families, or the split logic change: update
`docs/datasets-and-training.md`.

The contract in short: everything under `../ada-eval/data` is eval-proper
(the 19 benchmark samples and their canonical solutions); the guard
(`data/processing_scripts/eval_guard.py`) hashes every eval subprogram and
prompt in normalized and alpha-renamed structural forms and
`build_dataset` drops any matching record group before splitting.
`make check-integrity` must exit 0 after any dataset change. Never train
on `canonical_solution`, `base/`, `tests/`, `prompt.md`, or compacted
records from ada-eval.

## Where the important files live

- `data/processing_scripts/build_dataset.py` - the dataset builder. Discovers
  and pairs Ada sources, sanitizes prose into Simplified Technical English
  (STE), injects correct-vs-wrong defect pairs (five families), distills
  agent-skill guidance into system prompts, and writes JSONL plus metadata.
- `data/processing_scripts/eval_guard.py` - eval-integrity guard. Hashes every
  Ada subprogram and prompt in `../ada-eval/data` (normalized and
  alpha-renamed structural forms) and drops any training record matching
  them; `make check-integrity` fails if a split file contains eval content.
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
  toolchain deps (gnatprove, gnatformat) live in the dev manifest.
- `q3as-local-index/` - vendored Alire index (mirrors the crates q3as needs
  from the community index branch `stable-1.4.0`). setup.sh registers it
  ahead of the community index when the installed `alr` is older than the
  latest release, so modern binary crates (gnatprove 16.x, gnatformat 26.x)
  install on old distro alr packages (e.g. alr 1.2.1 on Debian).
- `setup.sh` - one-shot bootstrap: shallow-clones the sibling repos, creates
  `.env` from `.env.dev` (never overwrites), registers the vendored Alire
  index when `alr` is outdated, runs `uv sync`, extracts local Python
  headers for Triton.
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
by tooling. On old alr the vendored `q3as-local-index/` (registered by
`setup.sh`) supplies the modern binary crates the pinned `stable-1.2.1`
community index lacks.

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
           ada_ast_units.jsonl
           contract_mutations.jsonl
           dataset.jsonl
           dataset_metadata.json
           dataset_test.jsonl
           dataset_train.jsonl
           dataset_val.jsonl
           docs_chunks.jsonl
       processing_scripts/
           build_dataset.py
           code_variants.py
           eval_guard.py
           parse_ada_ast.py
           parse_docs.py
       raw/
   deploy/
       Modelfile
   docs/
       results/
           README.md
           result-data-v0.1.0.json
           result-v0.1.0.md
       architecture.md
       data-provenance.md
       datasets-and-training.md
       evaluation.md
       toolchain-setup.md
   eval/
       results/
       baseline_eval.py
       eval_pipeline.py
       generate.py
       system_prompt_spark.txt
   q3as-local-index/
       index/
           gn/
               gnatformat_bin/
                   gnatformat_bin-26.0.0.toml
               gnatprove/
                   gnatprove-16.1.0.toml
           index.toml
   scripts/
       ada_env.sh
       alire_env.py
       bump_version.py
       gen_agents_tree.py
       gen_contract_mutations.py
       gen_eval_report.py
       validate_defects.py
   tests/
       test_build_dataset.py
       test_code_variants.py
       test_defect_families.py
       test_eval_guard.py
       test_generate.py
       test_parsers.py
       test_reporting.py
   tools/
       check-links.py
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
