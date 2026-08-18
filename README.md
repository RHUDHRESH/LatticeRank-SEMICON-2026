# LatticeRank — Periodic-Aware Localization for Wafer Inspection

LatticeRank locates a 1,000 × 1,000 high-magnification Reference field inside a
1,000 × 1,000 Search image covering ten times the physical field of view. It
returns `(x, y)` in Search pixels.

![Cross-scale localization task](docs/images/01_localization_task.png)

## Measured result

On 80 leak-free, scene-disjoint synthetic validation pairs (39 DRAM,
41 FinFET), the shipped pipeline achieves **41.25% localization accuracy within
5 pixels**. Its catastrophic error rate (>25 px) is **58.75%**; periodic alias
selection remains the main limitation.

Keep these measurements separate:

- **Shipped δ=0.10:** 90.0% candidate-pool recall within 5 px, followed by
  **41.25% final localization accuracy**.
- **Diagnostic δ=0.15:** **92.5% candidate-pool recall** overall
  (**97.4% DRAM / 87.8% FinFET**). This wider-pool diagnostic is not the
  shipped setting and is not localization accuracy.

Evidence: [metrics](results/validation_metrics.json),
[predictions](results/validation_predictions.csv),
[candidate recall](results/candidate_recall.json), and
[claim provenance](results/claim_provenance.json).

## Pipeline

1. Reduce the full Reference to the known 10:1 Search scale.
2. Harvest local maxima from raw, mid-band, and directionality ZNCC maps.
3. Describe candidates with scene-relative and spatial correspondence
   features.
4. Rank with a packaged `HistGradientBoostingClassifier`.
5. Blend periodic-background-cancellation residual evidence.
6. Apply the nearest-Search-centre rule only inside a narrow measured
   equivalence set.

Distance to Search centre is not a learned feature. Details are in
[Method](docs/METHOD.md).

## Quick start

Python 3.12+ is required.

POSIX:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/inference.py examples/dram/reference.png examples/dram/search.png
```

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/inference.py examples\dram\reference.png examples\dram\search.png
```

Normal stdout is one parenthesized coordinate, for example `(644.50, 283.50)`.
Use `--json` for diagnostics.

## Generate deterministic pairs

```bash
python scripts/generate_dataset.py --architecture DRAM --count 30 --output-dir generated/dram
python scripts/generate_dataset.py --architecture FinFET --count 30 --output-dir generated/finfet
```

A small mixed smoke set:

```bash
python scripts/generate_dataset.py --count 4 --architecture both \
  --seed-start 910000 --output-dir generated/smoke
```

The generator renders DRAM-like and FinFET-like structures from one
world-coordinate scene, then applies independent Reference/Search acquisition
noise and geometry. It is **SEM-like, not physically exact**. See
[Data generator](docs/DATA_GENERATOR.md) and
[References](docs/REFERENCES.md).

## Regenerate reviewer figures

```bash
python scripts/make_figures.py
```

This rebuilds two compact example pairs and ten PNG figures from committed
metrics and deterministic generated samples. No decorative or AI-generated
imagery is used.

## Runtime

A one-pair smoke of the cleaned tree on `examples/dram` took **8.2 seconds**
and printed `(644.50, 283.50)`. This is a single-process, no-network
measurement, not an 80-pair mean. See [runtime provenance](results/runtime.json).

## Limitations

- Validation is synthetic-only; no real sponsor SEM pairs were available.
- 47/80 validation pairs are catastrophic failures, usually distant periodic
  aliases.
- FinFET is harder than DRAM (34.1% versus 48.7% within 5 px).
- The 80-pair validation split is modest.
- A separate 30+ randomized compliance evaluation was not run during cleanup;
  the published 41.25% result is the fixed 80-pair validation measurement.

Read [Results](docs/RESULTS.md) and
[Failure analysis](docs/FAILURE_ANALYSIS.md) before citing performance.

## Repository map

| Path | Role |
|---|---|
| `scripts/inference.py` | Localization CLI: two image paths, one `(x, y)` |
| `scripts/generate_dataset.py` | Standalone DRAM / FinFET pair generator |
| `scripts/train_ranker.py` | Retrain the packaged HGB ranker |
| `scripts/evaluate.py` | Reproduce validation metrics |
| `driftforge/models/hgb_r2.joblib` | Auto-loaded ranker weights |
| `docs/` | Method, generator, results, failures, references |
| `results/` | Measured claims and per-pair predictions |
| `examples/` | One DRAM pair and one FinFET pair |
| `tests/` | Generator, leakage, inference, and packaging tests |
