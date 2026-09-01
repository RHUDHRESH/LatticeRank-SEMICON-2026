# Bulk semiconductor pair corpus (Phase 2)

Large-scale production of the Phase 2 pair generator: unknown zoom s ∈ [8, 12]
by field of view, unknown stage rotation, 20% absent pairs, decoys, a
four-level severity ladder, and an RGB optical mode. Every pair is produced by
the same measured-ground-truth pipeline validated by the twelve gates in
`scripts/validate_phase2.py` — nothing about the data contract changes at
scale.

## Production layout

    data/phase2_bulk/
      shard_00000/          # 5,000 pairs, grayscale
        images/0000000_ref.png
        images/0000000_search.png
        ...
        manifest.jsonl      # one record per pair, §6 schema + diagnostics flag
        coverage_report.json
        citations.json
        DATASET_INFO.json
      shard_00001/
      ...
      shard_00006/          # every 7th shard (~14%) is the RGB optical mode

Shards are *independent and resumable*: indices, seeds and filenames are
globally continuous (`seed = 10_000_000 + global_index`), so a shard that
finished is never touched again and an interrupted run skips it. Shards merge
into one dataset by concatenating manifests.

## Running it

```bash
# tranche 1: 7 shards = 35,000 pairs (~41 GB, ~3 h on 8 cores)
bash scripts/bulk_production.sh

# any size: the first argument is the shard count
bash scripts/bulk_production.sh 40          # 200k pairs (~230 GB)
bash scripts/bulk_production.sh 720         # 3.6M pairs (~4.9 TB)
```

`WORKERS=14 bash scripts/bulk_production.sh ...` overrides worker count.

## The 3.6M-pair scale-out (read before committing)

| Resource | 35k tranche (this machine) | 3.6M pairs |
|---|---|---|
| Storage | ~41 GB | **~4.9 TB** of PNGs |
| Compute | ~3 h on 8 cores at ~3.6 pairs/s | **~11.6 days** of continuous 8-core time |

3.6M pairs is a cluster job, not a desktop job. The pipeline is already shaped
for it: shards are independent processes with disjoint seed ranges, so the
same command distributes across machines by giving each worker a shard range
(`--start-index` and `--count` in `scripts/generate_dataset.py`), and shards
need no cross-communication. Practical routes:

1. **One beefy node**: 64 cores ≈ 22 pairs/s ≈ 2 days for 3.6M; one 6 TB
   volume.
2. **N machines**: split `720 / N` shards each; merge manifests afterwards.
3. **Storage relief**: if PNG size matters more than portability, switch the
   writer to raw `uint8` `.npy` shards (2.0 MB/pair, no encode cost) or
   webp/JPEG-XL lossless (~0.6 MB/pair) — a one-function change in
   `scripts/generate_dataset.py::_png_bytes`; keep G12's encoder flag in sync.

## Quality invariants (checked, not assumed)

- Byte-identical regeneration from seeds (gate G12, encoder-aware).
- Class/label decorrelation: the present/absent and severity coins use
  dedicated RNG streams (gate G4 stays at chance even at n=1600).
- Measured ground truth: position from the warp-tracked mask, rotation and
  zoom from converged ZNCC readouts at the known location; sign convention
  empirically calibrated (`scripts/verify_conventions.py`).
- Stratification per shard: severity 30/30/25/15, four preset families,
  30/30/25/15 scene-profile mix, ≥2% zoom/rotation edge cases, ~10% near-edge
  placements, 40% decoy rate, 20% absent rate.
- Shard-level gates: run `scripts/validate_phase2.py --data-root
  data/phase2_bulk --splits shard_00000 ...` (sampled oracles) after each
  tranche; G9-style seed disjointness holds by construction (continuous
  global indices over one base).

## Flagship splits vs bulk corpus

`data/phase2/` (train/val/test/holdout/stress/rgb, 3,000 pairs) remains the
*evaluation-grade* reference set with the full 12-gate treatment. The bulk
corpus is for training at scale; gates are run per tranche on sampled shards.
Both come from the identical generator code path — only the seed base
(10M+ vs 1.1M-2.1M) and the PNG encoder flag differ.
