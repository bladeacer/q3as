# Data Provenance and Eval Integrity

Where q3as training data comes from, what licenses apply, and how the
pipeline guarantees that evaluation content never leaks into training.

## Data sources

All sources are fetched into a local archive cache by
[`scripts/fetch_repos.py`](../scripts/fetch_repos.py) (`make fetch-sources`).
The cache lives at `data/raw_repos/<owner>/<repo>/` (gitignored), holds HTTP
tarballs (no git metadata), and is content-addressed by repository identity: a
re-run re-fetches nothing already on disk. Per-repository metadata
(`.q3as-source.json`) records the URL, fetch time, the license SPDX when the
GitHub API answers, and the commit the tarball is the head of. That commit is
resolved through the git smart-HTTP endpoint (`/info/refs?service=git-upload-pack`,
the `git ls-remote` URL), which has no hourly quota; the REST API stays as a
fallback. The legacy `../<repo>` sibling layout still resolves as a fallback,
but nothing creates it anymore.

## Keeping the sources current

The cache is a tarball snapshot, not a clone, so nothing pulls on its own.
Because each entry records the commit it came from, the snapshot can be
compared with upstream:

| Command | What it does |
|---|---|
| `make fetch-sources` | Reuse the cache. No network, no rebuild. This is what the pipeline calls. |
| `make check-sources` | Ask GitHub for each repository's head commit and print cached vs upstream. Downloads nothing. Exits non-zero when anything moved, so it can gate a scheduled run. |
| `make update-sources` | Re-fetch only the repositories whose commit moved (or was never recorded), then rebuild the dataset from the refreshed cache and print the `make check-integrity` reminder. |

There is no separate "rebuild" step to remember: the dataset stages fingerprint
their input trees by content, so a refreshed repository changes the digest and
`make build-dataset` re-parses and rebuilds. A repository whose head cannot be
read (network, quota on the API fallback) is reported as `unknown` and is
**not** re-fetched, so a failed check never destroys a good cache. Entries
fetched before commits were recorded show as `untracked`; `--update` re-fetches
them once so they start being tracked, and says so, because that is the one
case where a cache that may be perfectly fine is touched.

When a source changes, review it before rebuilding: a new upstream revision can
add license-relevant material, or add code whose text collides with the
evaluation suite (the guard drops it, but the drop is worth reading in the
build log rather than taking on faith).

| Source | Cache path | Used for | License |
|---|---|---|---|
| [adacovex](https://github.com/bladeacer/adacovex) | `data/raw_repos/bladeacer/adacovex` | Ada/SPARK source with contract specs (code pairs, defect pairs, AST turns); `docs/usage` and `docs/archive` for `assurance_qa` (CI gating on an assurance level) | Apache-2.0 |
| [Ada_CRDT](https://github.com/bladeacer/Ada_CRDT) | `data/raw_repos/bladeacer/Ada_CRDT` | Spec/body pairs for diversity; `docs/proof` and `docs/compliance` for `assurance_qa` (assurance ladder, zero-justification doctrine, skip taxonomy, proof ledger) | MIT |
| [Ada-83-TLALOC](https://github.com/ViMoBr/Ada-83-TLALOC) | `data/raw_repos/ViMoBr/Ada-83-TLALOC` | Ada 83-era source (legacy patterns). Training use explicitly permitted by the author ([forum post](https://forum.ada-lang.io/t/fine-tuning-8b-ai-model-on-ada-spark/4746/3)) | GPL-3.0-or-later w/ GCC runtime exception; tests CC-BY-SA-4.0 |
| [ada-eval](https://github.com/AdaCore/ada-eval) | `data/raw_repos/AdaCore/ada-eval` | **Eval-proper only** (see below). Used for benchmark generation/evaluation and the guard; the training pipeline never passes it as an input; also the uv path dependency for eval tooling | Apache-2.0 |
| [AdaCore/learn](https://github.com/AdaCore/learn) | `data/raw_repos/AdaCore/learn` | Course material: doc-QA and heading-chunked doc sections | CC-BY-4.0 |
| [AdaCore/training_material](https://github.com/AdaCore/training_material) | `data/raw_repos/AdaCore/training_material` | AdaCore training courses (RST): doc-QA and heading-chunked doc sections. Description follows the repo README: collection of Ada/SPARK teaching courses in ReStructured Text. The labs ship a `prompt/` tree beside an `answer/` tree, which is where `lab_pair` completion and diff-derived explanation turns come from | CC-BY-4.0 |
| [agent-sh/ada-spark](https://github.com/agent-sh/ada-spark) | `data/raw_repos/agent-sh/ada-spark` | Current-toolchain guidance in system prompts | MIT |
| [AminBlg/SimpleEnglish](https://github.com/AminBlg/SimpleEnglish) | `data/raw_repos/AminBlg/SimpleEnglish` | STE writing rules and word map (paraphrased, no spec text) | MIT |
| [AdaCore/skills](https://github.com/AdaCore/skills) | `data/raw_repos/AdaCore/skills` | Toolchain QA (gnatprove, alire, gnatdoc, gnattest, gnatfuzz) | Apache-2.0 |
| [RobertBoettcherSF/Ada-Algorithms](https://github.com/RobertBoettcherSF/Ada-Algorithms) | `data/raw_repos/RobertBoettcherSF/Ada-Algorithms` | Ada/SPARK algorithm implementations in a single monorepo (distributed systems, graph algorithms, image processing, compression, SPARK-verified sheets, parsers): thousands of files across category directories. The 16 `SPARK2` topic trees are also the source of the prover-verified `spark_verified` turns. Fetched as one archive. [The author approved training use outside the repository](https://forum.ada-lang.io/t/fine-tuning-8b-ai-model-on-ada-spark/4746/6), so that approval is not recorded in the cached tree (its README carries no LLM-usage disclosure). The MIT text is the repository's `LICENSE` | MIT |

Licensing summary: Apache-2.0 and MIT code is redistributable with attribution;
CC-BY-4.0 course material is used with attribution; the GPL-licensed
Ada-83-TLALOC code is used for model training only (weights are not source-code
redistribution) and is covered by the author's explicit permission. The
RobertBoettcherSF Ada-Algorithms monorepo is MIT (its `LICENSE` file, in the
cached copy under `data/raw_repos/`) with the author's green light
for training use ([forum post](https://forum.ada-lang.io/t/fine-tuning-8b-ai-model-on-ada-spark/4746/6)),
a permission granted outside the repository.

Each license above was read from the cached repository's own license file, not
from the GitHub API: the fetcher records a `license_spdx` field when the API
answers, and it is `null` for AdaCore/learn, AdaCore/skills,
AdaCore/training_material, agent-sh/ada-spark, bladeacer/Ada_CRDT,
RobertBoettcherSF/Ada-Algorithms, and ViMoBr/Ada-83-TLALOC. The evidence sits
next to the code in the cache: a `LICENSE` file in each of those trees
(`LICENSES/` with GPL-3.0-or-later, the GCC runtime exception 3.1, and
CC-BY-SA-4.0 for Ada-83-TLALOC). The project credits in the
[README](../README.md#project-credits) repeat this table; keep the two in step.

## Toolchain inputs (not training data)

The table above is training data. The Ada toolchain q3as parses and proves with
is a separate set of external inputs, resolved through Alire rather than
[`scripts/fetch_repos.py`](../scripts/fetch_repos.py), and none of it is model
input:

| Input | Origin | License |
|---|---|---|
| `libadalang` crate (24.0.0) | [AdaCore/libadalang](https://github.com/AdaCore/libadalang) release archive, via the vendored [`q3as-local-index`](../q3as-local-index) | Apache-2.0 WITH LLVM-exception |
| libadalang Python bindings (24.0.0) | the same release archive, `python/` subdirectory, installed by `uv sync` as the `ast` group | Apache-2.0 WITH LLVM-exception |
| `gnat`, `gnatcoll*`, `libgpr2`, `langkit_support`, `adasat`, `xmlada` | community Alire index, pulled in by the `libadalang` crate | GPL-3.0-or-later w/ GCC runtime exception (tools only, not redistributed) |
| GMP | the distribution's `libgmp-dev`, unpacked to the user cache without sudo | LGPL-3.0-only (linked, not redistributed) |

The bindings change what the dataset contains, so it is worth being precise
about what they are for:
[`parse_ada_ast.py`](../data/processing_scripts/parse_ada_ast.py) uses
libadalang for exact extraction of specs, bodies, types, and aspect clauses, and
falls back to its regex scanner when the shared library is absent. The scanner's
output is not discarded or down-weighted; libadalang simply removes the class of
mistakes a regex cannot make, and it reports `valid` per unit so syntactically
broken sources are visible. Neither path can introduce eval-proper content: both
read the same cached Ada trees, and `make check-integrity` still gates the
splits.

## ada-eval is eval-proper

Everything under the cached ada-eval's `data/base/expanded` tree is the same
19 samples q3as is scored on (`spark_learn` 13, `spark_custom` 2,
`spark_human_eval_silver` 4), and each sample's `solution/` project is the
literal answer key. A fresh cache ships only that directory. Running ada-eval
locally adds `data/base/compacted` (the same samples as JSONL) plus
`data/generated` and `data/evaluated` (completions and scores for those very
prompts); the guard hashes those too when they exist. Training on any of it
would let the model memorize the benchmark.

The sample-authoring recipe in the ada-eval README ("Adding a new Sample")
is documentation we follow when extending the benchmark, not data.

Default parser targets do not include the cached ada-eval tree. The guard
still scans its data directory before records are deduplicated and split.

## The eval guard

[`data/processing_scripts/eval_guard.py`](../data/processing_scripts/eval_guard.py)
blocks eval content from reaching any training split:

1. **Blocklist.** It hashes all eval samples' base projects, canonical
   solutions, tests, prompts, and any `data/generated`/`data/evaluated`
   completions in two forms:
   - *normalized text*: case-folded (Ada is case-insensitive), comments
     stripped, whitespace collapsed. Catches verbatim and reformatted
     copies.
   - *structural*: user identifiers alpha-renamed by first occurrence.
     Numbers and logic stay significant, so a changed bound changes the
     hash. Catches renamed copies.
2. **Detection.** Every chat record's code fences are extracted with the
   same parser the corpus uses and compared against the blocklist. Eval
   prompt text is also caught as a substring of any message.
3. **Enforcement.** `build_dataset` runs guard, then dedup, then split.
   The whole group of a matching record is dropped (sibling turns are
   paraphrases of the same eval content), and the count is recorded in
   `dataset_metadata.json` under `splits.eval_guard`.

This catches more than copy-paste: it has caught sibling-repo code that is
structurally identical to an eval sample after renaming, and `learn` doc
examples that are verbatim eval subprograms (eval samples were authored
from the same corpus we mine). Detection is content-based, so it works no
matter which repo a turn claims as its source. It fires on the Ada-Algorithms
monorepo too: the first rebuild after adding it dropped records in
groups whose code matched eval content structurally, and the current
build (see below) still drops a similar volume.

Current shipped dataset (AST_STRUCTURAL_CAP = 10), as recorded in
`data/processed/dataset_metadata.json` at build time:
- 80,262 turns; group-aware splits 72,105 train / 4,071 val / 4,086 test,
- eval guard: 425 blocked signatures (394 subprograms: 207 exact,
  187 structural; 31 prompts: 17 exact, 14 structural), 795 contaminated
  groups dropped,
- dedup before split: 52,535 duplicates removed (23,593 verbatim, 28,942
  AST-structural over the cap),
- 2,369 empty-assistant records dropped (mostly spec-only units from the Ada-
  Algorithms monorepo).

Those totals move with the source cache: a rebuild after `make
update-sources` changes the turn, split, dedup, and empty-assistant lines
here. The guard blocklist is derived from the ada-eval samples alone, so it
only changes when ada-eval itself moves, and `make check-integrity`
re-verifies the whole pipeline from content rather than trusting this page.
Read the numbers out of `dataset_metadata.json` when the two disagree.

The guard drops whole *groups*, not individual records: one contaminated
turn removes every sibling turn derived from the same unit, so a group count
is what the guard reports. `dataset_metadata.json` records it as
`eval_guard.dropped_groups` alongside `dropped_records`.

Known limit: hash-based blocking identifies content up to renaming and
reformatting. A semantically equivalent but restructured algorithm is not
caught; the split discipline (val/test are never trained on) is the second
layer of defense.

## Rules for contributors

1. Never train on ada-eval's `solution/` (the `canonical_solution` of a
   compacted record), `base/`, `tests/`, `prompt.md`, `comments.md`, or any
   compacted, generated, or evaluated record.
2. After any change to data sources or the dataset, run
   `make check-integrity` (must exit 0) alongside `make validate-defects`.
   3. **When adding a new data source:** add its URL to `CORE_REPOS` in
   [`scripts/fetch_repos.py`](../scripts/fetch_repos.py), update the table above
   and the [`Makefile`](../Makefile) source list, and re-run `make
   check-integrity` before building the dataset. AGENTS.md reminds agents of
   this obligation.
4. Adding new eval samples to ada-eval automatically grows the blocklist;
   rebuild the dataset afterward.
5. If the guard reports `degraded` (ada-eval missing), builds proceed but
   `make check-integrity` exits 2: integrity is unverified, not proven.

Navigation: [project README](../README.md) · [docs index](README.md) · [changelog index](changelogs/index.md) · [results index](results/README.md)
