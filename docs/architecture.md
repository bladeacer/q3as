# Architecture

What each part of q3as does and how the pieces connect.

## Pipeline at a glance

1. `make download` fetches the official [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) to `models/qwen3-8b/`. Unsloth is the training framework only (patched kernels, QLoRA); the weights are always Qwen's original release.
2. `make fetch-sources` fills the archive cache `data/raw_repos/<owner>/<repo>/`
   with HTTP tarballs of every source repository (core sources only;
   ada-eval, learn, skills, the Ada-Algorithms algorithm monorepo, etc.). No git metadata, cached
   by repository identity: re-runs fetch nothing already on disk.
3. `make parse-data` runs the parser modules (heading-aware doc chunking,
   AST extraction) into standalone JSONL under `data/processed/`.
4. `make build-dataset` walks the cached Ada source trees, ingests the
   parser outputs, and emits `data/processed/dataset.jsonl` plus
   `dataset_{train,val,test}.jsonl` (chat-format turns).
5. `make train` runs QLoRA fine-tuning (Unsloth, 2048-token window, LoRA
   adapters on Qwen3-8B) with seeded, val-monitored training and early
   stopping; checkpoints and a training summary (train/val/test loss
   histories) land in `outputs/q3as/`.
6. `make generate` produces Ada solutions with both models into
   `outputs/generated_solutions/<label>/`.
7. `make eval` and `make eval-pipeline` score them (BLEU, compilation,
   unit tests, SPARK proofs) and write reports to `outputs/`.

## Module map

| Path | Responsibility |
|---|---|
| `data/processing_scripts/build_dataset.py` | Dataset builder: discovery, spec/body pairing, standard detection, STE sanitization, defect injection (17 GNAT-verified families), system-prompt composition, dedup, guard, splits |
| `data/processing_scripts/parse_docs.py` | Heading-aware Markdown/RST chunking into STE-cleaned doc-section turns |
| `data/processing_scripts/parse_ada_ast.py` | Ada semantic-unit extraction (libadalang when available, structural fallback): specs, bodies, aspects, constrained types; builds contract-writing turns |
| `data/processing_scripts/eval_guard.py` | Eval-integrity guard (see [Data provenance](data-provenance.md)) |
| `data/processing_scripts/source_paths.py` | Central source-path resolution: repo name to cache directory (`data/raw_repos/<owner>/<repo>`), legacy sibling fallback |
| `scripts/fetch_repos.py` | Archive-cache fetcher: parallel HTTP tarballs, per-repo metadata (URL, license SPDX), all repos in `CORE_REPOS` (including the Ada-Algorithms monorepo) |
| `scripts/validate_defects.py` | GNAT-compiles defect pairs; exit code fails any family that "compiles clean" |
| `scripts/ada_env.sh` / `scripts/alire_env.py` | Alire toolchain entry points (shell / Python) |
| `training/download_model.py` | HF download with terminating sanity checks (deep GPU check runs in a child process with a timeout) |
| `training/train_unsloth.py` | QLoRA training: seed 42, train-split default, eval every 50 steps, early stopping (patience 10), test-split perplexity; writes `training_summary.json` with train/eval loss histories for the report |
| `eval/generate.py` | Batch generation for both models; prompts embed the full project tree; replies are parsed into restricted multi-file overlays (target dir only, no project-file overwrites, identical echoes skipped) |
| `eval/baseline_eval.py` | BLEU/compliance metrics and aggregation |
| `eval/eval_pipeline.py` | ada-eval BUILD/TEST/PROVE comparison (see [Evaluation](evaluation.md)) |
| `q3as-local-index/` | Vendored Alire index for outdated-alr distributions (see [Toolchain setup](toolchain-setup.md)) |

## Data integrity rules (summary)

The full provenance and integrity contract lives in
[Data provenance](data-provenance.md). Short version for agents working in
this repo:

- ada-eval data is eval-proper; the guard removes any training record that
  matches it (verbatim, reformatted, or identifier-renamed).
- After any dataset or data-source change, run `make check-integrity` and
  update `docs/data-provenance.md` with the new source before building.
