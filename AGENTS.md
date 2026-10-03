# AGENTS.md - q3as repository guide

q3as fine-tunes Qwen3-8B on Ada and SPARK code with Unsloth QLoRA, then
evaluates the result with the ada-eval framework. This file maps the
repository for coding agents: what lives where, how the pieces connect, and
which commands to use. Regenerate the tree below with `make agents-tree`.

## Pipeline at a glance

1. `make download` fetches the official `Qwen/Qwen3-8B` to `models/qwen3-8b/`
   (Unsloth is the training framework only, never the weight source).
2. `make fetch-sources` fills the archive cache `data/raw_repos/<owner>/<repo>`
   (HTTP tarballs, no git, cached; core sources plus the RobertBoettcherSF
   `Ada-Algorithms` monorepo).
3. `make parse-data` runs the parser modules (doc chunking, AST extraction)
   into `data/processed/*.jsonl`. AST extraction uses libadalang when
   `make ast-deps` has installed it, and the structural scanner otherwise.
4. `make gen-contracts` and `make gen-verified-spark` are the two gnatprove
   stages: the first proves ten synthetic contract-template families, the
   second proves real SPARK2 subprograms from the Ada-Algorithms monorepo
   and keeps only the ones the prover discharges. Both write
   `data/processed/*.jsonl` in the chat-record shape the builder ingests, and
   both skip cleanly (keeping what they wrote) when `gnatprove` is absent.
5. `make build-dataset` walks the cached Ada source trees, ingests
   the parser outputs, and emits `data/processed/dataset.jsonl` plus the
   train/val/test split files (chat-format turns: code, doc-QA, defect
   pairs, contract-writing, toolchain QA, prover-verified SPARK bodies).
6. `make train` runs QLoRA fine-tuning (Unsloth) with seeded early stopping
   and writes checkpoints plus a training summary (train/val/test loss
   histories) to `outputs/q3as/`.
7. `make generate` produces Ada solutions with both models into
   `outputs/generated_solutions/<label>/`, writing a `generation_meta.json`
   sidecar per model (reply shape, files actually changed, spec/body routing)
   that `make eval` reports next to the scores.
8. `make eval-pipeline` scores them with ada-eval (compilation, unit tests,
   SPARK proofs) into `outputs/eval_results/`, then `make eval` adds BLEU,
   exact match and compliance from `outputs/eval_results.json` and prints the
   pipeline tallies; that order matters, because `eval` reads the tallies
   `eval-pipeline` writes and the other order reports the previous run's
   numbers. `make eval-data` (a prerequisite of `make eval`) rebuilds the
   derived benchmark JSONL it scores against. `make eval-report`
   writes the versioned summary to `docs/results/result-vX.Y.Z.md` (version from
   [`alire.toml`](alire.toml), bumped with `make bump-version`), including the
   train/val/test loss table and a training-trend verdict.

Steps 3 to 5 skip themselves when nothing they read has changed (see
[`stage_state.py`](data/processing_scripts/stage_state.py) below), so a second
`make all` reaches training in seconds. `FORCE=1` rebuilds anyway. Workers
default to one per core; `DATASET_WORKERS=1` forces serial.

Upstream sources move: each cache entry records the commit it was fetched at,
`make check-sources` compares those with the current heads (no download, non-zero
exit when anything moved), and `make update-sources` re-fetches only what moved
and rebuilds the dataset. Editing `scripts/fetch_repos.py` invalidates the
dataset stages once, because it is an input to them.

End-user and developer documentation lives in [`docs/`](docs) (architecture,
datasets and training, data provenance, toolchain setup, evaluation); README
links to it. Keep those pages current when behavior changes.

## Data provenance and eval integrity

**Full documentation: [`docs/data-provenance.md`](docs/data-provenance.md) -
keep it in sync.** When a new data source is added (new URL in `CORE_REPOS` of
[`scripts/fetch_repos.py`](scripts/fetch_repos.py), new hub, new parser source),
update that page's source table and license notes, then run `make
check-integrity` before building. The same applies when the guard, the defect
families, or the split logic change: update
[`docs/datasets-and-training.md`](docs/datasets-and-training.md).

The contract in short: everything under the cached ada-eval's `data/`
directory (formerly the `../ada-eval` sibling) is eval-proper (the 19 benchmark
samples and their canonical solutions); the guard
([`data/processing_scripts/eval_guard.py`](data/processing_scripts/eval_guard.py))
hashes every eval subprogram and prompt in normalized and alpha-renamed
structural forms and `build_dataset` does not pass ada-eval as a training source
and still drops any matching record group before splitting. `make
check-integrity` must exit 0 after any dataset change. Never train on
`solution/` (a compacted record's `canonical_solution`), `base/`, `tests/`,
`prompt.md`, `comments.md`, or any compacted, generated, or evaluated record
from ada-eval.

## Where the important files live

- [`data/processing_scripts/build_dataset.py`](data/processing_scripts/build_dataset.py)
  - the dataset builder. Discovers and pairs Ada sources, sanitizes prose into
  Simplified Technical English (STE), injects correct-vs-wrong defect pairs
  (seventeen families), distills agent-skill guidance into system prompts,
  dedups (verbatim plus a cap of `AST_STRUCTURAL_CAP = 10` on alpha-renamed
  AST-record shapes, chosen by a controlled cap-tuning experiment, see
  docs/datasets-and-training.md; a defect turn is capped per defect, not per
  code shape, so a family is not thinned away by plain copies of the code it
  corrects; prose turns have no signature and are never capped), drops
  empty-assistant records, and writes JSONL plus metadata with per-source
  provenance. It also builds the turn kinds that no parser module emits:
  `lab_pair` completion and diff-derived explanation twins from the
  training-material labs (`discover_lab_pairs`, `build_lab_pair_turns`) and
  `assurance_qa` from the SPARK Platinum repos' compliance docs
  (`load_spark_assurance_docs`, `build_spark_assurance_turns`).
  `pair_files` keys a body on `(directory, stem)`, so a spec only ever pairs
  with the body beside it (a stem-only key paired 138 specs with a foreign
  body across the monorepo), and `_SKILL_DOC_PATTERNS` matches each
  guidance repo's real layout (`skills/<name>/...`), which is how the
  ada-spark proof references stopped being silently dropped (two of seven
  documents were loaded before the fix; five of six now are).
- [`data/processing_scripts/source_paths.py`](data/processing_scripts/source_paths.py)
  - resolves every source repo to its cache directory
  (`data/raw_repos/<owner>/<repo>`, with the legacy sibling location as
  fallback); no pipeline file hard-codes paths.
- [`data/processing_scripts/eval_guard.py`](data/processing_scripts/eval_guard.py)
  - eval-integrity guard. Hashes every Ada subprogram and prompt in the cached
  ada-eval `data/` tree (normalized and alpha-renamed structural forms)
  and drops any training record matching them; `make check-integrity` fails if a
  split file contains eval content.
- [`data/processing_scripts/stage_state.py`](data/processing_scripts/stage_state.py)
  - staleness check for the dataset stages. Each stage declares its input
  trees (with the suffixes it consumes), input files, scripts, and
  output-changing parameters; the fingerprint is stored in
  `data/processed/.stages/<stage>.json` and the stage is skipped when it still
  matches. Content hashes, not mtimes (the cache is re-extracted from
  tarballs). `--workers` is never part of a fingerprint. `FORCE=1` overrides.
- [`data/processing_scripts/progress.py`](data/processing_scripts/progress.py)
  - `phase()` and `Progress` for the long stages: one log line per named step
  with its wall time, plus an item count with rate and ETA inside long loops,
  so a multi-minute build does not look hung.
- [`data/processing_scripts/`](data/processing_scripts) - dataset helpers live
  beside the builder.
- [`training/download_model.py`](training/download_model.py) - HF model download
  plus terminating sanity checks (light by default; deep GPU check runs in a
  child process with a timeout so the pipeline can never hang).
- [`training/train_unsloth.py`](training/train_unsloth.py) - QLoRA training
  script (Unsloth, 1024-token window, single-process dataset tokenization, LoRA
  adapters on Qwen3-8B). The train split streams from JSONL into an Arrow
  table of int32 `input_ids` under `data/processed/.tokenized/` (151 MB for
  72,105 records against 468 MB of JSONL (measured on the v0.7.0 dataset), and
  trl's own 4-minute tokenization pass is skipped because the column marks the
  dataset processed); the eval splits stay text for the chunked eval callback
  and are sampled by `--eval-sample` / `--test-sample` so a run's evaluation
  fits `--eval-budget-min`.   Each eval pass logs an item count with a rate and
  an ETA (a multi-minute pass that logs nothing reads as a hang) and the
  eval-loss line projects the evaluation time still to come. A run interrupted
  halfway continues from its last checkpoint by default (`--no-resume` opts
  out): checkpoints keep the optimizer, scheduler, and RNG state, and the eval
  points already scored ride along in the trainer state, so early stopping and
  the loss curve survive the interruption. The 500-step default is 0.06 of one
  epoch on purpose, so a full generate-and-evaluate loop comes back in hours;
  see the step budget section of docs/datasets-and-training.md before changing
  it. Table keys cover split content,
  chat template, tokenizer vocabulary, truncation length, and a pipeline
  version; tables a run did not touch are pruned. The no-val-file
  fallback carve keeps records in memory and says so in the summary. It
  configures its own logger, because importing unsloth makes
  `logging.basicConfig` a no-op.
- [`eval/generate.py`](eval/generate.py) - batch generation for the fine-tuned
  and base models. A reply becomes project files through
  `parse_generated_files`: `File: <path>` entries when the model follows the
  prompt's contract, otherwise a single fenced block at the sample's target
  path. Every ada-eval target is a spec (`.ads`), so a fallback block that
  opens a `package body` is routed to the sibling `.adb` (`body_sibling`),
  and only when the base tree already carries that mirror file; an explicit
  `File:` entry still wins. Per-sample reply shape, changed-file count and
  whether the routing fired go to `generation_meta.json` beside the packed
  JSONL, because none of it fits the ada-eval sample schema and all of it
  changes how a build number reads.
- [`eval/baseline_eval.py`](eval/baseline_eval.py) - reference-based scoring:
  joins the `make generate` output to the ada-eval canonical solutions and
  reports BLEU-4, exact match, file-set match, and standard compliance per
  model, plus the ada-eval BUILD/TEST/PROVE tallies
  (`outputs/eval_results.json`) and the reply-format shape read from
  `generation_meta.json`. Refuses reference records with no solution files: a
  hollow pack reports BLEU 0.0 with every standard `Unknown` and reads like a
  model result. Exits non-zero when there is nothing to score;
  it never scores the training corpus.
- [`eval/ada_eval_common.py`](eval/ada_eval_common.py) - the single
  build/test/prove tally shared by both eval modules and the report, plus the
  packed-dataset naming rules. Do not reimplement it: a third divergent copy is
  what made the published prove counts disagree with the comparison report.
- [`eval/eval_pipeline.py`](eval/eval_pipeline.py) - BUILD/TEST/PROVE comparison
  report between base and fine-tuned models via ada-eval. It is the only writer
  of `outputs/eval_results/`, so it must run before `make eval` prints its
  tallies.
- [`scripts/validate_defects.py`](scripts/validate_defects.py) - compiles the
  dataset's defect pairs with the real GNAT and checks the claimed compiler
  messages. Exit code is the contract: any "compiles clean" defect is a failure.
- [`scripts/gen_contract_mutations.py`](scripts/gen_contract_mutations.py) -
  gnatprove-verified synthetic contract curriculum (ten template families
  aimed at the eval proof blockers, `--limit` instances per family), emitted
  as `contract-synth:` groups that are exempt from the AST-structural cap.
- [`scripts/gen_verified_spark.py`](scripts/gen_verified_spark.py) - proves the
  real SPARK2 units of all 16 `Ada-Algorithms/*/SPARK2` trees and keeps only
  the ones every check of which comes out proved. A batch that hits the
  timeout is dropped whole: a killed run's partial summary cannot be trusted.
  `--limit 0` (the default) takes every discovered unit; a full run is about
  30 minutes on 16 cores and is paid once per source change.
- [`scripts/fetch_repos.py`](scripts/fetch_repos.py) - archive-cache fetcher
  (HTTP tarballs, parallel, cached; every repo in `CORE_REPOS` including the
  Ada-Algorithms monorepo). Each cache entry records the upstream commit it
  came from, so `--check` can report what moved (exit 1 when something did) and
  `--update` can re-fetch only that. A head it cannot read is `unknown`, never
  "changed", so a failed check cannot destroy a good cache. `make fetch-sources`
  runs the default (reuse), `make check-sources` and `make update-sources` the
  two comparison modes.
- [`scripts/pack_eval_data.py`](scripts/pack_eval_data.py) - rebuilds ada-eval's
  `data/base/compacted/*.jsonl`, the derived JSONL that `baseline_eval` scores
  against and that no fresh cache contains. Sets
  `GIT_CEILING_DIRECTORIES` to the ada-eval checkout, because the packer is
  git-aware and the cache sits inside this repository where `data/raw_repos/`
  is ignored, so an unguarded pack succeeds while writing records with no
  solution files; passes `force=True` for the same reason; then verifies that
  every record carries files. `make eval-data` runs it and `make eval`
  depends on it.
- [`scripts/ada_env.sh`](scripts/ada_env.sh) - runs any command inside the Alire
  toolchain environment (`alr exec`); all Ada tool invocations go through it.
- [`scripts/alire_env.py`](scripts/alire_env.py) - Python side of the same:
  `find_tool("gnatprove")` resolves binaries through the Alire environment for
  subprocess calls.
- [`scripts/build_libadalang.py`](scripts/build_libadalang.py) - builds and
  installs the shared `libadalang.so` that the AST parser's ctypes wrapper
  dlopens (`make ast-deps`). Resolves [`alire-ast.toml`](alire-ast.toml) in the
  gitignored `.alire-ast/` workspace and builds [`ast.gpr`](ast.gpr) there, so
  the SPARK toolchain in `.alire-dev/` is untouched. Optional: without it
  [`parse_ada_ast.py`](data/processing_scripts/parse_ada_ast.py) uses its
  structural scanner.
- [`scripts/toolshims/unzip`](scripts/toolshims/unzip) - a Python
  [`unzip`](scripts/toolshims/unzip) stand-in that
  [`build_libadalang.py`](scripts/build_libadalang.py) puts on PATH, because the
  libadalang chain ships `.zip` source archives and `alr` would otherwise offer
  to install [`unzip`](scripts/toolshims/unzip) with sudo.
- [`scripts/gen_agents_tree.py`](scripts/gen_agents_tree.py) - rewrites the file
  tree section below.
- [`alire.toml`](alire.toml) / [`alire-dev.toml`](alire-dev.toml) /
  [`alire-ast.toml`](alire-ast.toml) - publishing, dev, and AST manifests.
  Dev-only toolchain deps (gnatprove, gnatformat) live in the dev manifest;
  `libadalang` lives in the AST manifest because only the dataset parser needs
  it. `make bump-version` keeps all three in step.
- [`q3as-local-index/`](q3as-local-index) - vendored Alire index (mirrors the
  crates q3as needs from the community index branch `stable-1.4.0`, plus a
  patched `gnatcoll_gmp` and the pinned `libadalang`). setup.sh registers it
  ahead of the community index when the installed `alr` is older than the latest
  release, so modern binary crates (gnatprove 16.x, gnatformat 26.x) install on
  old distro alr packages (e.g. alr 1.2.1 on Debian).
- [`setup.sh`](setup.sh) - one-shot bootstrap: fetches the data/guidance
  repositories into the archive cache with
  [`scripts/fetch_repos.py`](scripts/fetch_repos.py) (HTTP tarballs, no git),
  creates `.env` from [`.env.dev`](.env.dev) (never overwrites), registers the
  vendored Alire index when `alr` is outdated, runs `uv sync`, extracts local
  Python headers for Triton.
- [`scripts/python_env.sh`](scripts/python_env.sh) - prints the `CPATH` Triton
  needs to compile its CUDA driver shim, resolved from the interpreter that will
  run training (the venv, else `python3`) instead of a hardcoded version. The
  single place both the [`Makefile`](Makefile) (`train`) and
  [`scripts/run_cap_experiment.sh`](scripts/run_cap_experiment.sh) ask; it warns
  on stderr when the headers are missing. [`setup.sh`](setup.sh) writes the
  cache it reads.
- [`Makefile`](Makefile) - entry points for every step; `make help` lists them.
- [`deploy/Modelfile`](deploy/Modelfile) - Ollama deployment definition for the
  fine-tuned model.
- [`tests/`](tests) - pytest unit tests for the dataset builder, the eval
  scorers, the Alire toolchain resolver, and the reporting helpers (`make
  test`).
- [`docs/changelogs/`](docs/changelogs) - one changelog file per version
  (`vX.Y.Z.md`) plus [`index.md`](docs/changelogs/index.md), the index that
  links every version. It shares the repo's versioned-artifact convention with
  [`docs/results/`](docs/results): the version comes from
  [`alire.toml`](alire.toml) via `make bump-version`, and each release links its
  own `docs/results/result-vX.Y.Z.md` when one exists. Add a `vX.Y.Z.md` and
  index it when behaviour, metrics, or documented claims change - not for every
  commit. Link a results file only if it is present; a withdrawn run is
  described in prose instead of linked. There is no root `CHANGELOG.md`.
- [`docs/README.md`](docs/README.md) - the documentation index, linked from the
  project README. Every page under [`docs/`](docs) ends with the same navigation
  footer (project README, docs index, changelog index, results index); keep it
  when adding a page. Run `make agents-tree` after adding a file, and `make
  lint` (which checks every markdown link and anchor).

## Ada toolchain through Alire

q3as resolves every Ada tool through the Alire environment, never a bare
`PATH` lookup. The dev manifest declares the toolchain; commands run
through `alr exec`:

- Shell: `scripts/ada_env.sh <cmd>` (used by Makefile targets).
- Python: `from alire_env import find_tool` then pass
  `env=alire_env_path()` to `subprocess.run`.

`make prove` syncs the toolchain. Because alr 1.2.1 has no `--manifest` option,
it copies [`alire-dev.toml`](alire-dev.toml) into the gitignored `.alire-dev/`
workspace and runs `alr update` there; the real manifests are never modified by
tooling. On old alr the vendored [`q3as-local-index/`](q3as-local-index)
(registered by [`setup.sh`](setup.sh)) supplies the modern binary crates the
pinned `stable-1.2.1` community index lacks.

The AST parser has a second, independent Alire workspace: `make ast-deps` copies
[`alire-ast.toml`](alire-ast.toml) into `.alire-ast/` and resolves `libadalang`
there, so touching the parser toolchain never disturbs the prover. It stays
optional: without it
[`parse_ada_ast.py`](data/processing_scripts/parse_ada_ast.py) logs `libadalang
available: False` and uses its structural scanner.

## Conventions

- Dataset prose follows STE: short active sentences, no em-dashes, no
  hedging; code, flags, and quoted compiler output are exempt.
- Expected compiler messages in defect pairs are GNAT-verified strings.
  Change them only with `make validate-defects` evidence.
- Python: `uv` for env/deps, ruff + mypy clean for all touched files,
  pytest for the dataset logic.
- The source repositories (see the table in
  [`docs/data-provenance.md`](docs/data-provenance.md): adacovex, Ada_CRDT,
  Ada-83-TLALOC, ada-eval, learn, training_material, ada-spark, SimpleEnglish,
  skills, plus the RobertBoettcherSF `Ada-Algorithms` monorepo) are inputs only;
  [`scripts/fetch_repos.py`](scripts/fetch_repos.py) fetches them into the
  gitignored `data/raw_repos/` cache. Their licenses are credited in
  [`README.md`](README.md).

## Project tree

<!-- AGENTS:TREE-BEGIN -->
```text
q3as/
   .freebuff/
       project-id
   data/
       processed/
           .stages/
               ada_ast_units.json
               contract_mutations.json
               dataset.json
               docs_chunks.json
               verified_spark.json
           ada_ast_units.jsonl
           contract_mutations.jsonl
           dataset.jsonl
           dataset_metadata.json
           dataset_test.jsonl
           dataset_train.jsonl
           dataset_val.jsonl
           docs_chunks.jsonl
           verified_spark.jsonl
       processing_scripts/
           build_dataset.py
           code_variants.py
           eval_guard.py
           parse_ada_ast.py
           parse_docs.py
           progress.py
           source_paths.py
           stage_state.py
       raw/
   deploy/
       Modelfile
   docs/
       changelogs/
           index.md
           v0.1.0.md
           v0.2.0.md
           v0.3.0.md
           v0.4.0.md
           v0.4.1.md
           v0.5.0.md
           v0.6.0.md
           v0.7.0.md
           v0.8.0.md
       results/
           README.md
           result-data-v0.1.0.json
           result-data-v0.3.0.json
           result-data-v0.4.1.json
           result-data-v0.7.0.json
           result-v0.1.0.md
           result-v0.3.0.md
           result-v0.4.1.md
           result-v0.7.0.md
       architecture.md
       data-provenance.md
       datasets-and-training.md
       evaluation.md
       README.md
       toolchain-setup.md
   eval/
       ada_eval_common.py
       baseline_eval.py
       eval_pipeline.py
       generate.py
       system_prompt_spark.txt
   q3as-local-index/
       index/
           gn/
               gnatcoll_gmp/
                   gnatcoll_gmp-24.0.0.toml
               gnatformat_bin/
                   gnatformat_bin-26.0.0.toml
               gnatprove/
                   gnatprove-16.1.0.toml
           li/
               libadalang/
                   libadalang-24.0.0.toml
           index.toml
   scripts/
       toolshims/
           unzip
       ada_env.sh
       alire_env.py
       build_cap_variants.sh
       build_libadalang.py
       bump_version.py
       collect_cap_results.py
       fetch_repos.py
       gen_agents_tree.py
       gen_contract_mutations.py
       gen_eval_report.py
       gen_verified_spark.py
       make_probe_splits.py
       pack_eval_data.py
       python_env.sh
       run_cap_experiment.sh
       validate_defects.py
   tests/
       test_alire_env.py
       test_build_dataset.py
       test_check_links.py
       test_code_variants.py
       test_defect_families.py
       test_eval_guard.py
       test_eval_pipeline.py
       test_eval_report_training.py
       test_eval_scoring.py
       test_gen_contract_mutations.py
       test_generate.py
       test_lab_extraction.py
       test_makefile.py
       test_pack_eval_data.py
       test_parsers.py
       test_reporting.py
       test_stage_state.py
       test_train_data.py
   tools/
       check-links.py
   trainer_output/
   training/
       download_model.py
       train_unsloth.py
   .env.dev
   .gitignore
   AGENTS.md
   alire-ast.toml
   alire-dev.toml
   alire.toml
   ast.gpr
   LICENSE
   Makefile
   plan.md
   pyproject.toml
   README.md
   run_download.py
   setup.sh
   uv.lock
```
<!-- AGENTS:TREE-END -->
