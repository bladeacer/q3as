# Changelog

Notable changes to q3as, newest first.

The version lives in `alire.toml` and is mirrored into `alire-dev.toml`,
`alire-ast.toml`, and `pyproject.toml`; `make bump-version` keeps all four in
step, and two tests in `tests/test_reporting.py` fail if they ever disagree.
Each release's evaluation summary is written to
`docs/results/result-vX.Y.Z.md` by `make eval-report`, using the same version
string, and is listed in the [results index](docs/results/README.md).

## [0.3.0]

Results: [result-v0.3.0.md](docs/results/result-v0.3.0.md)

Two themes: the numbers in `docs/results/` now measure the models, and both
the dataset and the evaluation now cover the whole benchmark.

### Measured on the full benchmark

All 19 benchmark samples were generated for both models, so these rates
describe the benchmark rather than a subset:

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Build | 8/19 (42.1%) | 15/19 (78.9%) | +36.8 pts |
| Unit tests | 5/19 (26.3%) | 11/19 (57.9%) | +31.6 pts |
| SPARK proved | 0 | 0 | - |
| File-set match | 94.7% | 100.0% | +5.3 pts |
| Exact match vs canonical | 26.3% | 26.3% | 0.0 pts |
| BLEU-4 vs canonical | 0.770 | 0.659 | -0.111 |
| Standard compliance | 0.351 | 0.281 | -0.070 |

Fine-tuning roughly doubles the build and test rates, but the two models are
level on exact match and the base model is *ahead* on BLEU-4 and standard
compliance. The reading is that the fine-tune taught structure rather than
recall: it emits the reference's file layout every time and compiles far more
often, while a general 8B model writes Ada that is textually closer to the
reference but less often well formed. Textual similarity is the weaker signal
here, which is why the build and test columns are the headline.

No sample proved, for either model. The remaining blockers are overflow
checks (8), postconditions (3), uninitialized reads (3), raise checks (2) and
discriminant checks (2). The base model additionally produced 11 prover errors
against the fine-tuned model's 2.

### Evaluation

- `make eval` scores the models' actual generations. It previously read
  `data/processed/dataset.jsonl` and compared each training prompt against
  its own gold answer, so the reported BLEU and compliance were corpus
  statistics carrying model labels, identical for the fine-tuned and base
  blocks. It now joins `make generate` output to the ada-eval canonical
  solutions by dataset and sample name, and scores the decoded Ada source.
- `compute_bleu` was a character 3-gram *precision* over the first 200
  characters, with no brevity penalty. Replaced with real BLEU-4: clipped
  n-gram precisions, brevity penalty, add-one smoothing for the high-order
  precisions that short Ada files zero out, and an Ada-aware tokenizer that
  keeps operators separate from operands.
- Ada standard detection was a substring scan over the system prompt, whose
  fixed boilerplate contains "including SPARK 2014". Every record was
  therefore labelled SPARK 2014 and compliance was scored against the wrong
  keyword set. `make eval` now reuses `build_dataset.detect_ada_standard` and
  runs it on the *canonical* solution rather than on the model's own guess.
- The versioned report could never include the baseline aggregate:
  `scripts/gen_eval_report.py` read `bleu`, `compliance`, `per_model`,
  `generated` and `timestamp` while `eval/baseline_eval.py` wrote `avg_bleu`,
  `avg_compliance` and neither of the others. No key overlapped, so the
  section never rendered.
- The build/test/prove tally existed in three copies that had drifted. The
  report counted `proved_incorrectly` and `subprogram_not_found` as harness
  errors while both eval modules counted them as unproved, so the published
  prove counts disagreed with `outputs/comparison_report.txt` for the same
  run. Collapsed into `eval/ada_eval_common.py`.
- "No data" is no longer rendered as a measured `0.0%`. `rate_pct` returns
  `None` and `has_results` separates an empty tally from a measured zero.
- Partial coverage is stated explicitly, in the log and on screen, because a
  rate over the generated subset must not read like a rate over the whole
  benchmark.
- `--evals` in `make eval` selects the tools each eval needs instead of always
  requiring all three. `--max-samples` was removed from
  `eval/eval_pipeline.py`, where it never did anything.
- `eval/results/`, an empty leftover from an earlier layout, was removed.

### Failure reporting

- `make generate` and `make eval-pipeline` exited 0 after a crashed,
  OOM-killed or unparseable worker, leaving stale per-sample files behind and
  reporting success. Workers now fail loudly, and a generation run that
  produced nothing for either model exits non-zero.
- `--verbose` was a no-op in all five entry points: `setLevel(DEBUG)` ran
  before `basicConfig(level=INFO)`, which reset the root level. All five
  honour it now.
- `make generate` gained `--worker-timeout` (default 4 h) so a wedged worker
  cannot hang the pipeline.

### Dataset

- Rebuilt end to end. The previous `dataset.jsonl` was truncated mid-record
  (9,955 valid lines, then a JSON error) because the last build never
  finished, so every downstream number came from a partial corpus.
- 73,080 turns; 65,809 train / 3,709 val / 3,562 test. 58,058 duplicates
  removed (21,410 verbatim, 36,648 AST-structural over the cap), 2,413
  empty-assistant records dropped. The eval guard blocked 479 signatures and
  dropped 695 contaminated groups. `make check-integrity` passes.
- The guard's drop is a *group* count, but it was stored and logged as
  `dropped_records`. Metadata now records `dropped_groups` and
  `dropped_records` separately.
- The corpus roughly doubled in size, because the AST units are now extracted
  with libadalang rather than the regex scanner, which feeds the defect and
  variant generators.

### AST extraction

- `make ast-deps` builds and installs the shared `libadalang.so` that the
  ctypes bindings `dlopen`. The Alire crate ships a static library, so
  `scripts/build_libadalang.py` drives `ast.gpr` twice: the stack as
  `static-pic`, then libadalang alone as `relocatable`.
- libadalang 24.0.0 is vendored in `q3as-local-index/` so an outdated `alr`
  resolves a pinned, reproducible set. The `gnatcoll_gmp` entry is patched to
  drop the `libgmp` dependency, which would otherwise make `alr` shell out to
  `sudo apt-get install libgmp-dev`.
- The extraction code had never been executed and none of it matched the real
  API. `source_text` is `text`; `parents` is a method; `PackageBody` is not a
  `BasePackageDecl` subclass, so every body record had an empty package name;
  aspect clauses live on the declaration rather than the spec, and are now
  parsed by the same helper the scanner uses; and an anonymous access type's
  unnamed dereference function no longer aborts its whole file. Result:
  44,531 to 47,069 AST turns, with 6,514 files and zero fallbacks.
- mypy now sees the real bindings and type-checks that path.

### Training

- A missing validation split used to overwrite a successfully loaded held-out
  test split with a carve from the training data, then record `test_source` as
  the test file regardless. The held-out split is kept now, and the provenance
  says `fallback-carve` when it is a carve, so a reported test perplexity can
  no longer be a training-data number presented as held-out.
- `bfloat16` support was probed and logged, then ignored while the config
  hard-coded `"bf16": True`. The probe now drives the config, with an fp16
  fallback.
- The `setup_training_environment` docstring promised best-checkpoint restore
  that the same function hard-disables. Corrected, and a `save_steps` that is
  not a multiple of `eval_steps` is now reported instead of silently rounded.
- The deep model sanity check returned success when it timed out, printing
  "model is ready for fine-tuning" for a check that never completed. It now
  reports INCONCLUSIVE without failing the run.
- `base_model` is no longer hardcoded as `base_qwen3-8b` in the scorer; the
  label is derived from the model directory the same way `generate.py` does it,
  so pointing `--base-model` elsewhere no longer silently matches nothing.

### Toolchain and packaging

- New `scripts/python_env.sh` is the single place that resolves Triton's
  `CPATH` from the interpreter that will run training. The `Makefile` and
  `scripts/run_cap_experiment.sh` each had it hardcoded to `python3.13`, so
  the header export stopped silently on any other version.
- `setup.sh` never actually installed `Python.h`: it ships in
  `libpythonX.Y-dev`, and `pythonX.Y-dev` contains nothing under
  `usr/include`. The old glob caught the right package by accident, in the
  directory `apt-get download` had been scattering `.deb` files into the
  repository root.
- `setup.sh` also failed on every run at the index-registration step, because
  `alr index --list` prints a priority number before the name and the
  membership test anchored at the start of the line. It exited 1 without ever
  reaching the header step.
- `scripts/alire_env.py` gained the `__main__` block the documentation
  already told users to run, and it exits non-zero when a tool the pipeline
  invokes is missing. `gnatdoc` is reported but never fails the check, since
  `alire-dev.toml` does not depend on it.
- `find_tool` decided whether to warn about a system-installed tool based on
  which of its two searches found the binary. With no Alire environment the
  two searches coincide, so a distribution-provided `gnatprove` was silently
  reported as the managed one. It now keys on whether the hit is inside the
  Alire prefix.

### Tests and versioning

- `make test` was rewriting the repository's own `pyproject.toml`.
  `bump_version.EXTRA_VERSION_FILES` points at the real file and `set_version`
  writes it unconditionally, while the tests monkeypatched `MANIFESTS` but not
  `EXTRA_VERSION_FILES`; `test_bump_major_resets` baked `1.0.0` into
  `pyproject.toml` on every run. That is where the `1.0.0`-versus-`0.2.0`
  version drift came from. Isolated with an autouse fixture, the version
  realigned, and two new guards fail if the manifests and `pyproject.toml`
  ever disagree again.
- ruff's `target-version = "py311"` contradicted `requires-python = ">=3.12"`.
  Removed, so ruff infers it from `requires-python` and the two cannot drift.
- `make help` omitted five targets, including `check-integrity`, which the
  documentation cites nine times. All 26 targets are listed now, and the
  `make eval` description matches what it does.

### Documentation

- `docs/evaluation.md` documented RUN and BASE evals and a 13/13 canonical
  sanity check that exist in no Make target, and contradicted itself, since
  the benchmark has 19 samples.
- `docs/data-provenance.md`'s "current shipped dataset" numbers matched
  nothing. Corrected to the rebuilt corpus, and a toolchain-inputs table was
  added so the non-training inputs are accounted for too.
- The Ada-Algorithms training-use approval was cited to the repository README,
  which contains no LLM-usage disclosure; the MIT text is in `LICENSE`. The
  page now records that the approval is not documented in the repository
  rather than pointing at a citation that is not there.
- `AGENTS.md` still described `setup.sh` as shallow-cloning sibling
  repositories; it fetches HTTP tarballs and has for some time.
- The turn-kind table listed `ast_impl`, `ast_contract` and friends, but the
  builder buckets every `ast_*` kind into `ast_qa`, which is the largest kind
  in the corpus and appeared in no table.
- `eval_guard.py`'s docstring claimed numbers are collapsed to one symbol;
  `structural_text()` keeps them verbatim on purpose, because a changed bound
  is a logic change.

## [0.2.0]

Monorepo migration (the `RobertBoettcherSF/Ada-Algorithms` source was added to
the archive cache), OOM fixes for the 8 GB training budget, and the AST
structural-cap experiments.

Its evaluation run was removed from `docs/results/`. It covered only 7 of the
19 benchmark samples, so its rates did not describe the benchmark, and the
evaluation code that produced it scored the training corpus rather than the
models. The source changes remain; only the results files were withdrawn.

## [0.1.0]

Results: [result-v0.1.0.md](docs/results/result-v0.1.0.md)

The first versioned evaluation, 19 samples, reporting build 16/19 fine-tuned
against 7/19 base.

Those build and test columns came from ada-eval and are unaffected by the
evaluation bug fixed in 0.3.0. The BLEU and compliance figures in that file
are corpus statistics, not model metrics, and must not be compared with the
ones 0.3.0 introduces.
