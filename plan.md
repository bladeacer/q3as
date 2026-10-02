# plan.md - remaining work for the extraction-quality track

State of the 2026-10-02 session: the data track is finished. What is left is
the measurement, which needs a GPU and is out of scope here.

## Done and verified (no action needed)

- Contract curriculum: 10 template families, 708 gnatprove-verified turns,
  exempt from the AST-structural cap (`contract-synth:` group prefix).
- `make update-sources` fixed (forced fetch + git-protocol commit
  resolution); all 10 cached repos record upstream commits, `--check` exits 0.
- Two defect-injector false claims fixed (`typo` multi-line profile guard,
  `nonexistent_call` declaration/collision guard); `make validate-defects`
  clean.
- `lab_pair` extraction: prompt/answer twins -> completion turns plus
  diff-derived STE explanation twins (129 completions + 123 explanations =
  252 turns, 47 labs).
- `assurance_qa`: 5 turns from the Ada_CRDT/adacovex compliance docs
  (ladder criteria, zero-justification doctrine, skip taxonomy, ledger
  format, CI gating). Loader: `load_spark_assurance_docs` /
  `build_spark_assurance_turns` in `build_dataset.py`.
- `gen_verified_spark.py`: proves real SPARK2 units with local gnatprove
  and emits completion turns (`spark_verified` kind). Widened to all 16
  `Ada-Algorithms/*/SPARK2` trees: 577 units discovered, **568 proved**
  (400 with contracts), 9 dropped. Source labels carry the topic tree.
- Fixed while widening: `subprocess.TimeoutExpired` is a `SubprocessError`,
  not an `OSError`, so the batch guard never caught it and one heavy batch
  aborted a 20-minute run. A timed-out batch is now dropped whole.
- Fixed answer chunking (`_chunk_start`): a table anchor no longer leaves a
  `ion | Status |` header fragment in front of the assurance ladder answer,
  a heading anchor no longer leaks its tail (`— Choose Your Path`), and a
  prose anchor no longer starts mid-sentence. 7 new unit tests.
- Dataset rebuilt (v0.7.0): **80,233 turns** (train 72,124 / val 4,083 /
  test 4,026). `make check-integrity` clean, 406 tests pass, `make lint`
  clean, `make validate-defects` clean.

## Remaining work

### 1. Measure it (needs a GPU slot)

The only thing left on this track. Retrain on the v0.7.0 dataset and run
`make eval-pipeline`, then compare the prove-blocker histogram and the
per-dataset build/test table against v0.4.1. The hypothesis this data work
exists to test: 568 real proved SPARK2 subprograms plus the assurance
doctrine reduce the blocker count. `make eval-report` writes the summary to
`docs/results/result-v0.7.0.md` and links it from the results index.

### 2. Optional follow-ups

- Consider raising `gen_contract_mutations --limit` now that the cap
  exemption holds (diminishing returns; only if the next eval still shows
  contract-shaped blockers).
- `misc/SPARK2` supplies 403 of the 568 verified turns, and many are one-line
  utilities (`Add (Left, Right : Number) return Sum`). The AST-structural cap
  thins the duplicates; if the next eval shows no benefit from them, cap the
  tree rather than the kind.

## Files touched across the two sessions (for the eventual commit)

- `scripts/gen_contract_mutations.py` (10 families, body turns, limit 240)
- `scripts/gen_verified_spark.py` (new; all 16 SPARK2 trees, timeout fix)
- `scripts/fetch_repos.py` (forced fetch, git-protocol commit resolution)
- `data/processing_scripts/build_dataset.py` (lab extraction + explanations,
  assurance QA, contract-synth dedup exemption, defect injector guards,
  `spark_verified` kind mapping, answer-chunk start fix)
- `tests/test_gen_contract_mutations.py` (new), `tests/test_lab_extraction.py`
  (new), `tests/test_build_dataset.py` (answer-chunk tests)
- `Makefile` (gen-verified-spark target, lint list, extra-turns list)
- `docs/` (datasets-and-training, data-provenance, AGENTS.md file map and
  pipeline steps, changelogs v0.5.0 + v0.6.0 + v0.7.0 + index), versions
  bumped to 0.7.0 in the four manifests
- `data/processed/contract_mutations.jsonl`, `verified_spark.jsonl`
  (regenerated; gitignored)