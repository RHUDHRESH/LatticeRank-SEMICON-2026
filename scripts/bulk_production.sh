#!/usr/bin/env bash
# Bulk Phase 2 production: resumable sharded generation.
#
# Each shard is an independent, complete split directory (images/, manifest.jsonl,
# coverage_report.json, DATASET_INFO.json) with globally continuous indices, so
# shards merge into one dataset and interrupted runs resume by skipping any
# shard whose manifest already holds its full count.
#
# Usage:
#   bash scripts/bulk_production.sh [TARGET_SHARDS] [ROOT] [SHARD_SIZE]
#   bash scripts/bulk_production.sh            # 7 shards  = 35,000 pairs (~41 GB)
#   bash scripts/bulk_production.sh 720        # full 3.6M pairs (~4.9 TB - cluster scale)
#
# Layout: every 7th shard is RGB (~14%), the rest grayscale. Seeds are
# 10000000 + global_index; indices are globally continuous across shards.
set -u
TARGET=${1:-7}
ROOT=${2:-data/phase2_bulk}
SHARD_SIZE=${3:-5000}
WORKERS=${WORKERS:-7}
BASE=10000000

cd "$(dirname "$0")/.."
mkdir -p "$ROOT"
echo "bulk production: $TARGET shards x $SHARD_SIZE pairs -> $ROOT (workers=$WORKERS)"
START_TIME=$(date +%s)

for i in $(seq 0 $((TARGET - 1))); do
  SHARD=$(printf "shard_%05d" "$i")
  MANIFEST="$ROOT/$SHARD/manifest.jsonl"
  if [ -f "$MANIFEST" ] && [ "$(wc -l < "$MANIFEST" | tr -d ' ')" -eq "$SHARD_SIZE" ]; then
    echo "[skip] $SHARD already complete"
    continue
  fi
  START=$((i * SHARD_SIZE))
  if [ $((i % 7)) -eq 6 ]; then MODALITY=rgb; else MODALITY=gray; fi
  echo "[run ] $SHARD indices $START..$((START + SHARD_SIZE - 1)) modality=$MODALITY"
  python scripts/generate_dataset.py --phase 2 \
    --split p2_bulk \
    --count "$SHARD_SIZE" \
    --start-index "$START" \
    --output-dir "$ROOT/$SHARD" \
    --modality "$MODALITY" \
    --workers "$WORKERS" \
    --fast-png
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "done in $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
