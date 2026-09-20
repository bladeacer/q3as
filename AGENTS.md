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

## Data provenance and eval integrity

Training data comes from these sources, and none of them may contain
ada-eval evaluation content:

- Sibling repos (`../adacovex`, `../Ada_CRDT`, `../Ada-83-TLALOC`, and the
  docs in `../learn`): spec/body pairs, doc-QA, AST-derived turns.
- `../ada-spark`, `../SimpleEnglish`, `../skills`: guidance and toolchain QA
  (system prompts and prose only, never eval content).
- `../ada-eval`: **eval-proper only.** Everything under
  `../ada-eval/data/base/{expanded,compacted}` are the same 19 samples q3as
  is scored on, and `canonical_solution` there is the literal answer key.
  The parsers do walk that tree for AST/contract turns, so every record
  derived from it is guarded (below). The sample-authoring recipe in the
  ada-eval README ("Adding a new Sample") is documentation, not data.

**The eval guard (`data/processing_scripts/eval_guard.py`).** It hashes all
19 eval samples' base projects, canonical solutions, tests, prompts, and any
`data/generated`/`data/evaluated` completions in two forms: normalized text
(case-folded, comments stripped, whitespace collapsed) and a structural form
(user identifiers alpha-renamed by first occurrence; numbers and logic kept,
so a changed bound changes the hash). Every chat record's code fences are
extracted with the same parser the corpus uses and compared against the
blocklist; eval prompt text is also caught as a substring. `build_dataset`
drops the whole group of any record that matches (one bad turn implies its
siblings are paraphrases of the same eval content) and records the count in
`dataset_metadata.json` under `splits.eval_guard`.

This catches more than copy-paste: `make check-integrity` has confirmed
sibling-repo code that is structurally identical to an eval sample after
renaming (for example a `STARS` procedure), and `../learn` doc examples that
are verbatim eval subprograms. Detection is content-based, so it works no
matter which repo a turn claims as its source.

**Rules for future changes:**

1. Never train on `canonical_solution`, `base/`, `tests/`, `prompt.md`, or
   compacted records from ada-eval. `--input-dir ../ada-eval` in parser
   targets is allowed only because the guard removes derived matches.
2. After any dataset change run `make check-integrity` (exit 0 required)
   alongside `make validate-defects`.
3. Adding new eval samples to ada-eval automatically grows the blocklist;
   rebuild the dataset afterward. No new dataset record may match them.
4. If the guard reports `degraded` (ada-eval missing), builds proceed but
   `make check-integrity` exits 2: integrity is unverified, not proven.

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
           dataset.jsonl
           dataset_metadata.json
           dataset_test.jsonl
           dataset_train.jsonl
           dataset_val.jsonl
           docs_chunks.jsonl
       processing_scripts/
           build_dataset.py
           eval_guard.py
           parse_ada_ast.py
           parse_docs.py
       raw/
   deploy/
       Modelfile
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
       gen_agents_tree.py
       validate_defects.py
   tests/
       test_build_dataset.py
       test_defect_families.py
       test_eval_guard.py
       test_generate.py
       test_parsers.py
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
