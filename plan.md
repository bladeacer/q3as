# plan.md - remaining work for the extraction-quality track

State of the 2026-09-30 dev session and exactly what is left to finish it.
Pick up here next session.

## Done and verified (no action needed)

- Contract curriculum: 10 template families, 708 gnatprove-verified turns,
  exempt from the AST-structural cap (`contract-synth:` group prefix).
- `make update-sources` fixed (forced fetch + git-protocol commit
  resolution); all 10 cached repos record upstream commits, `--check` exits 0.
- Two defect-injector false claims fixed (`typo` multi-line profile guard,
  `nonexistent_call` declaration/collision guard); `make validate-defects`
  clean.
- `lab_pair` extraction: prompt/answer twins -> completion turns, now with
  diff-derived STE explanation twins (129 completions + 123 explanations =
  252 turns, 47 labs).
- `assurance_qa`: 5 turns from the Ada_CRDT/adacovex compliance docs
  (ladder criteria, zero-justification doctrine, skip taxonomy, ledger
  format, CI gating). Loader: `load_spark_assurance_docs` /
  `build_spark_assurance_turns` in `build_dataset.py`.
- `gen_verified_spark.py`: proves real SPARK2 units with local gnatprove
  and emits completion turns (`spark_verified` kind). Currently sources
  only `sorting/SPARK2` + `graphs/SPARK2`: 41/42 units proved (35 with
  contracts).
- Dataset rebuilt: 79,340 turns. Tests 399 pass. Ruff/mypy clean on all
  touched files.

## Remaining work

### 1. Rebuild the dataset after the last code edits (10 min)

The last edits (lab-explanation quote fix in `build_dataset.py`, test
updates) postdate the last build, so the dataset stage is stale and the
current `data/processed/dataset.jsonl` predates the quote fix. Run:

```bash
make build-dataset        # stages skip where unchanged
make check-integrity      # must stay clean
make test && make lint    # lint regenerates AGENTS.md tree first if needed
```

Expect lab_pair=252 (the previous 79,340-turn build already had 252; the
delta is only the explanation wording fix, so counts should not move much).

### 2. Widen `gen_verified_spark` to all 16 SPARK2 trees (~1-2 h)

`DEFAULT_SOURCES` in `scripts/gen_verified_spark.py` lists only
`sorting/SPARK2` and `graphs/SPARK2`. The monorepo has 16 SPARK2 trees
(clustering, compression, concurrency, cryptography, geometry, graphs,
hashing, matrices, misc, ml, numerical, parsing, searching, sorting,
strings, trees). Replace `DEFAULT_SOURCES` with a glob over
`data/raw_repos/RobertBoettcherSF/Ada-Algorithms/*/SPARK2`, bump `--limit`
(according to prover budget; 64 units took ~3 min at `--level=1`), and
re-run `make gen-verified-spark FORCE=1`.

Watch-outs:

- `--mode=prove` is already set; concurrency trees may need
  `--no-annotation-check` or will simply drop (fine - the gate decides).
- Expect a lower kept-ratio outside sorting; the dropped units are exactly
  the unproved-claim turns we must not train on.
- The stage is fingerprinted, so this cost is one-time per source change.

### 3. Smoke-check the two new turn kinds end-to-end (~15 min)

After the rebuild in step 1, sample and eyeball:

```bash
uv run python - <<'EOF'
import json
for want in ('lab_pair', 'assurance_qa', 'spark_verified'):
    pass
EOF
```

Concretely: one `assurance_qa` answer quotes the ladder table (not a
fallback keyword hit), one `spark_verified` assistant fence ends at
`end <Name>;` with no package wrapper or trailing next-declaration text
(the `_body_only` overshoot fix landed late; the current file was generated
before it - **regenerate with `make gen-verified-spark FORCE=1` during
step 2**), and one lab explanation quote carries no leading `+`.

### 4. Docs + changelog for this batch (~30 min)

Follow the repo convention (behaviour changed -> document + version):

- `make bump-version VERSION=0.7.0` (or PART=minor).
- New `docs/changelogs/v0.7.0.md`: lab explanation twins, assurance_qa
  from the Platinum repos' compliance docs, `spark_verified` turns from
  the SPARK2 trees, per-kind counts from the step-1 build, and the
  `verified_spark` stage in `make gen-verified-spark`.
- `docs/datasets-and-training.md`: add `lab_pair` explanation twins and
  the `spark_verified` kind to the turn-kind table; mention
  `assurance_qa`.
- `docs/changelogs/index.md`: add the 0.7.0 row.
- `make agents-tree` (new test file if any) and `make lint`.

### 5. Optional follow-ups (not needed to close the session)

- `AGENTS.md` tree: already regenerated during lint; re-run after any new
  files.
- Consider raising `gen_contract_mutations --limit` now that the cap
  exemption holds (diminishing returns; only if the next eval still shows
  contract-shaped blockers).
- The eval-side work (retrain + `make eval-pipeline` to measure the
  prove-blocker delta vs v0.4.1) is the actual success metric for
  everything above; it needs a GPU slot and is out of scope for the data
  track.

## Files touched this session (for the eventual commit)

- `scripts/gen_contract_mutations.py` (10 families, body turns, limit 240)
- `scripts/gen_verified_spark.py` (new)
- `scripts/fetch_repos.py` (forced fetch, git-protocol commit resolution)
- `data/processing_scripts/build_dataset.py` (lab extraction +
  explanations, assurance QA, contract-synth dedup exemption, defect
  injector guards, `spark_verified` kind mapping)
- `tests/test_gen_contract_mutations.py` (new), `tests/test_lab_extraction.py` (new)
- `Makefile` (gen-verified-spark target, lint list, extra-turns list)
- `docs/` (datasets-and-training, data-provenance, changelogs v0.5.0 +
  v0.6.0 + index), versions bumped to 0.6.0 in the four manifests
- `data/processed/contract_mutations.jsonl`, `verified_spark.jsonl`
  (regenerated; gitignored)
