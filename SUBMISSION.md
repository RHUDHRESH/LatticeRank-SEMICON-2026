# LatticeRank submission sheet

**Challenge:** SEMICON India Hackathon 2026, Track 2 — Drift-Sense

**Deliverable:** LatticeRank 1.0 (`driftforge` Python package)

**Public repository:** https://github.com/RHUDHRESH/LatticeRank-SEMICON-2026

**Output contract:** one Search-image centre coordinate, `(x, y)`

This page is the reviewer handoff. It maps the official repository contract to
the exact file or command that satisfies it and separates measured evidence
from interpretation.

## Fastest judge path

Use Python 3.12 or newer. A CPU is sufficient; no network, dataset download, or
weight download is required after dependency installation.

```bash
git clone https://github.com/RHUDHRESH/LatticeRank-SEMICON-2026.git
cd LatticeRank-SEMICON-2026
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on POSIX or
`.\.venv\Scripts\Activate.ps1` in PowerShell, then run:

```bash
python -m pip install --disable-pip-version-check -r requirements.txt
python -m pip check
python scripts/judge_check.py
python scripts/verify_evidence.py
```

`judge_check.py` deliberately launches inference from outside the repository
directory. It verifies automatic model discovery, both architectures, image
bounds, and the one-line stdout contract. `verify_evidence.py` recomputes the
published rates from final coordinates.

## Official requirement matrix

| Official requirement | Submission evidence | Status |
|---|---|---:|
| Public, self-contained repository | This repository and [`README.md`](README.md) | Ready |
| Clone-to-inference instructions | [`README.md`](README.md#run-a-real-inference) and [`docs/FIVE_MINUTE_DEMO.md`](docs/FIVE_MINUTE_DEMO.md) | Ready |
| Standalone DRAM/FinFET generator | [`scripts/generate_dataset.py`](scripts/generate_dataset.py); accepts `--architecture`, `--count`, and `--output-dir` | Ready |
| Recorded true centre coordinates | Generator writes per-pair JSON plus dataset-level CSV/JSON records | Ready |
| Independent Reference/Search noise | Separate acquisition streams; tested in [`tests/test_generator_integrity.py`](tests/test_generator_integrity.py) | Ready |
| Edge brightening, blur, rotation, scale, noise | Implemented in `driftforge/`; documented in [`docs/DATA_GENERATOR.md`](docs/DATA_GENERATOR.md) | Ready |
| Standalone localization script | [`scripts/inference.py`](scripts/inference.py); two image paths in, one `(x, y)` line out | Ready |
| Automatically loaded weights | [`driftforge/models/hgb_r2.joblib`](driftforge/models/hgb_r2.joblib), with hash/provenance metadata | Ready |
| Reproducible training entry point | [`scripts/train_ranker.py`](scripts/train_ranker.py) with scene-disjoint manifests | Ready |
| Complete environment freeze | [`requirements.txt`](requirements.txt); runtime/dev roles are separately documented | Ready |
| 2–3 sources per augmentation family | Parameter-to-source traceability in [`docs/REFERENCES.md`](docs/REFERENCES.md) | Ready |
| At least 30 randomized pairs | 40-pair internal compliance run and 30-pair untouched external confirmation | Ready |
| Runtime per pair | [`results/runtime.json`](results/runtime.json) and [`results/validation_metrics.json`](results/validation_metrics.json) | Ready |
| Success and honest failure | [`README.md`](README.md#one-success-and-one-honest-failure) and [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md) | Ready |

## Direct contract checks

Inference:

```bash
python scripts/inference.py examples/dram/reference.png examples/dram/search.png
python scripts/inference.py examples/finfet/reference.png examples/finfet/search.png
```

Expected stdout is exactly one coordinate line per invocation, for example
`(644.50, 283.50)`. Diagnostic JSON is opt-in with `--json` and does not alter
the default contract.

Generation:

```bash
python scripts/generate_dataset.py --architecture DRAM --count 2 --output-dir generated/dram
python scripts/generate_dataset.py --architecture FinFET --count 2 --output-dir generated/finfet
python scripts/validate_dataset.py generated/dram
python scripts/validate_dataset.py generated/finfet
```

Full release gates:

```bash
python -m pytest -q
python scripts/verify_evidence.py
```

## Results that may be cited

| Distribution | Pair protocol | Accuracy within 5 px | Catastrophic error >25 px |
|---|---:|---:|---:|
| Pinned external development | 4 seeds × 30 | **93.33% (112/120)** | 6.67% |
| Pinned external untouched confirmation | 1 held-back seed × 30 | **100.00% (30/30)** | 0.00% |
| Internal fixed stress | 80 scene-disjoint pairs | 48.75% (39/80) | 51.25% |
| Internal randomized compliance | seed 2026 × 40 | 55.00% (22/40) | 45.00% |

The external source is the public reference-style
[`FlankerDev12/drift-sense-ref`](https://github.com/FlankerDev12/drift-sense-ref)
generator pinned at commit `59376381`. It is not described as the official
hidden evaluator. All 150 external coordinates, split labels, source hashes,
and the frozen selection equation are in
[`results/external_starter_benchmark.json`](results/external_starter_benchmark.json)
and [`results/external_starter_predictions.csv`](results/external_starter_predictions.csv).

The 90%+ result must not be merged with the broader internal stress result.
Distribution names and denominators should accompany every cited percentage.

## Evidence integrity

- Final accuracy is recomputed from `pred_x`, `pred_y`, `gt_x`, and `gt_y`.
- Ground truth is used only after inference for scoring and figure labels.
- Distance to Search centre is not a learned feature. It is used only for the
  organiser's tie convention after evidence-equivalence or exact-wallpaper
  detection.
- Figures 12–14 are generated from measured pair `validation-000240`; they are
  not conceptual mock-ups.
- [`results/claim_provenance.json`](results/claim_provenance.json) maps each
  claim to its row evidence, environment, revision, and reproduction command.
- Limitations, negative results, and distribution shifts are retained in the
  README and results documentation.

## Reviewer reading order

1. [`README.md`](README.md) — problem, result, method, and runnable commands.
2. [`docs/FIVE_MINUTE_DEMO.md`](docs/FIVE_MINUTE_DEMO.md) — narrated execution.
3. [`docs/METHOD.md`](docs/METHOD.md) — equations and step-by-step inference.
4. [`docs/RESULTS.md`](docs/RESULTS.md) — benchmark protocols and uncertainty.
5. [`docs/REFERENCES.md`](docs/REFERENCES.md) — physical parameter traceability.
6. [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md) — what still fails and why.

## Portal handoff

The repository component is complete. The separate i4C portal component
requires the registered team's PDF and team/contact fields; those identity
details are intentionally not invented or embedded here. The slide-deck result
must use the distribution-qualified values above and link this public
repository.
