# Architecture

What each part of q3as does and how the pieces connect.

## Pipeline at a glance

1. `make download` fetches the official [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) to `models/qwen3-8b/`. Unsloth is the training framework only (patched kernels, QLoRA); the weights are always Qwen's original release.
2. `make parse-data` runs the parser modules (heading-aware doc chunking,
   AST extraction) into standalone JSONL under `data/processed/`.
3. `make build-dataset` walks the Ada source trees of the sibling repos,
   ingests the parser outputs, and emits `data/processed/dataset.jsonl`
   plus `dataset_{train,val,test}.jsonl` (chat-format turns).
4. `make train` runs QLoRA fine-tuning (Unsloth, 2048-token packing, LoRA
   adapters on Qwen3-8B) with seeded, val-monitored training and early
   stopping; checkpoints land in `outputs/q3as/`.
5. `make generate` produces Ada solutions with both models into
   `outputs/generated_solutions/<label>/`.
6. `make eval` and `make eval-pipeline` score them (BLEU, compilation,
   unit tests, SPARK proofs) and write reports to `outputs/`.

## Module map

| Path | Responsibility |
|---|---|
| `data/processing_scripts/build_dataset.py` | Dataset builder: discovery, spec/body pairing, standard detection, STE sanitization, defect injection (11 GNAT-verified families), system-prompt composition, dedup, guard, splits |
| `data/processing_scripts/parse_docs.py` | Heading-aware Markdown/RST chunking into STE-cleaned doc-section turns |
| `data/processing_scripts/parse_ada_ast.py` | Ada semantic-unit extraction (libadalang when available, structural fallback): specs, bodies, aspects, constrained types; builds contract-writing turns |
| `data/processing_scripts/eval_guard.py` | Eval-integrity guard (see [Data provenance](data-provenance.md)) |
| `scripts/validate_defects.py` | GNAT-compiles defect pairs; exit code fails any family that "compiles clean" |
| `scripts/ada_env.sh` / `scripts/alire_env.py` | Alire toolchain entry points (shell / Python) |
| `training/download_model.py` | HF download with terminating sanity checks (deep GPU check runs in a child process with a timeout) |
| `training/train_unsloth.py` | QLoRA training: seed 42, train-split default, eval every 50 steps, early stopping (patience 10), test-split perplexity |
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
