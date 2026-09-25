#!/usr/bin/env bash
# Build cap-variant datasets for the AST_STRUCTURAL_CAP tuning experiment.
# Each variant differs ONLY in --ast-structural-cap; everything else is the
# exact `make build-dataset` command, so the only moving part is the cap.
# cap0 keeps one record per structural family, cap-1 is unlimited.
set -euo pipefail
cd "$(dirname "$0")/.."

CAPS=(1 2 3 5 10 -1)
OUT_ROOT=outputs/capexp/datasets
mkdir -p "$OUT_ROOT"

for cap in "${CAPS[@]}"; do
  label=$( [ "$cap" == "-1" ] && echo "inf" || echo "$cap" )
  dir="$OUT_ROOT/cap$label"
  mkdir -p "$dir"
  echo "=== cap=$label -> $dir ==="
  uv run python data/processing_scripts/build_dataset.py \
    --input-dir data/raw/ \
    --extra-input-dir data/raw_repos/bladeacer/adacovex \
    --extra-input-dir data/raw_repos/bladeacer/Ada_CRDT \
     --extra-input-dir data/raw_repos/ViMoBr/Ada-83-TLALOC \
     --extra-input-dir data/raw_repos/RobertBoettcherSF/Ada-Algorithms \
    --doc-dir data/raw_repos/AdaCore/learn \
    --doc-dir data/raw_repos/AdaCore/training_material \
    --guidance-dir data/raw_repos/agent-sh/ada-spark \
    --guidance-dir data/raw_repos/AminBlg/SimpleEnglish \
    --guidance-dir data/raw_repos/AdaCore/skills \
    --extra-turns data/processed/docs_chunks.jsonl \
    --extra-turns data/processed/ada_ast_units.jsonl \
    --extra-turns data/processed/contract_mutations.jsonl \
    --ast-structural-cap "$cap" \
    --output-dir "$dir"
done

echo "All cap variants built under $OUT_ROOT:"
du -sh "$OUT_ROOT"/cap*/dataset_train.jsonl
