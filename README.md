# LatticeRank

### Periodic-aware localization for semiconductor wafer inspection

LatticeRank locates a 1,000 × 1,000 high-magnification **Reference** field
inside a 1,000 × 1,000 **Search** image that covers ten times the physical
area. It returns exactly one Search coordinate: `(x, y)`.

The hard part is not finding a similar patch. DRAM and FinFET structures repeat,
so hundreds of locations can look correct. LatticeRank generates those
hypotheses, removes the repeating lattice, and ranks the non-periodic evidence
that remains.

![Reference-to-Search localization task](docs/images/01_localization_task.png)

**Start here:** [60-second judge check](#run-it-in-60-seconds) ·
[measured results](#measured-results) · [method](#how-it-works) ·
[scientific traceability](docs/REFERENCES.md) · [submission checklist](SUBMISSION.md)

## Run it in 60 seconds

Python 3.12+ and a CPU are sufficient. The model is bundled; no weights or
dataset are downloaded at inference time.

```bash
git clone https://github.com/RHUDHRESH/LatticeRank-SEMICON-2026.git
cd LatticeRank-SEMICON-2026
python -m venv .venv
```

Activate with `source .venv/bin/activate` on POSIX or
`.\.venv\Scripts\Activate.ps1` in PowerShell, then run:

```bash
python -m pip install --disable-pip-version-check -r requirements.txt
python scripts/judge_check.py
python scripts/verify_evidence.py
```

These commands test both architectures from outside the repository and
recompute every published rate from final coordinates.

Direct inference is simply:

```bash
python scripts/inference.py examples/dram/reference.png examples/dram/search.png
```

```text
(644.50, 283.50)
```

Add `--json` for diagnostics.

## Measured results

| Benchmark | Protocol | Within 5 px | Over 25 px | Median error |
|---|---|---:|---:|---:|
| External development | 4 seeds × 30 pairs | **93.33% (112/120)** | 6.67% | 1.44 px |
| External untouched confirmation | held-back seed × 30 pairs | **100.00% (30/30)** | 0.00% | 1.46 px |
| Internal fixed stress | 80 scene-disjoint pairs | 48.75% (39/80) | 51.25% | 62.57 px |
| Internal randomized compliance | seed 2026 × 40 pairs | 55.00% (22/40) | 45.00% | 4.36 px |

![Accuracy and catastrophic errors across named benchmarks](docs/images/06_pipeline_ablation.png)

Evidence: [external summary](results/external_starter_benchmark.json) ·
[150 external predictions](results/external_starter_predictions.csv) ·
[internal metrics](results/validation_metrics.json) ·
[80 internal predictions](results/validation_predictions.csv) ·
[claim provenance](results/claim_provenance.json)

### What the numbers reveal

The correct site is a raw local maximum in **100%** of fixed scenes and enters
the candidate pool in **90.0%**, but final selection reaches **48.75%**.

![Candidate recall as the adaptive pool widens](docs/images/03_candidate_recall.png)

## How it works

```text
Reference + Search
        │
        ▼
10:1 anti-aliased scale normalization
        │
        ▼
raw + mid-band + directionality correlation maps
        │
        ▼
adaptive local-maximum candidate pool
        │
        ▼
periodic cancellation + structural ranking
        │
        ▼
evidence-equivalent tie rule
        │
        ▼
      (x, y)
```

### 1. Normalize the physical scale

Anti-alias and reduce the complete Reference by the known 10:1 pixel ratio.

### 2. Preserve multiple plausible sites

Evaluate zero-mean normalized cross-correlation for each channel:

```text
ρc(x,y) = <Sxy − μxy, Tc − μT> / (||Sxy − μxy|| ||Tc − μT||)
```

Preserve an adaptive union of local maxima:

```text
C = ⋃c localmax(ρc ≥ max(ρc) − 0.10)
```

### 3. Cancel what repeats

Estimate the lattice and subtract the median of eight neighboring lattice
translations, leaving site-specific structure.

```text
periodic(I) = median of neighboring lattice translations
residual(I) = I − periodic(I)
```

![Measured periodic background, residual, and uniqueness mask](docs/images/13_periodic_residual_explainer.png)

### 4. Rank independent evidence

Inside the validated device-pitch envelope:

```text
score = z(periodic residual) + 0.05 z(raw ZNCC) + 0.05 z(mid-band ZNCC)
```

Broader geometry uses the packaged 77-feature HGB ranker plus residual evidence.

![Five real candidates and every term in the frozen score](docs/images/14_candidate_evidence.png)

### 5. Handle exact wallpaper

Low-context wallpaper uses the challenge’s centre convention. The seven fixed
exact-wallpaper cases improve from 0/7 to **7/7** within 5 px.

![Step-by-step measured inference on validation-000240](docs/images/12_inference_walkthrough.png)

## Measured examples

Measured DRAM success: **0.06 px** error.

![Measured successful localization](docs/images/07_success_example.png)

Measured FinFET alias failure: **256.75 px** error.

![Measured periodic-alias failure](docs/images/08_periodic_alias_failure.png)

## Synthetic data

DriftForge generates labeled DRAM and FinFET pairs with independent
Reference/Search acquisition effects.

![Generated DRAM and FinFET examples](docs/images/02_generated_pairs.png)

```bash
python scripts/generate_dataset.py --architecture DRAM --count 30 --output-dir generated/dram
python scripts/generate_dataset.py --architecture FinFET --count 30 --output-dir generated/finfet
python scripts/validate_dataset.py generated/dram
```

[Parameter ranges and citations](docs/REFERENCES.md)

## Reproduce the evidence

```bash
# Full release gates
python -m pytest -q
python scripts/verify_evidence.py

# Fixed and randomized evaluations
python scripts/evaluate.py validation --output-dir reproduced-validation
python scripts/evaluate.py randomized --count 40 --seed 2026 --output-dir reproduced-randomized

# Candidate-recall diagnostic and figures
python scripts/evaluate_candidate_recall.py --output reproduced-candidate-recall.json
python scripts/make_figures.py
```

Runtime: **2.86 s median**, **6.15 s mean**, **30.32 s P95**.
[Evidence](results/runtime.json)

## Scientific integrity

- Metrics are recomputed from emitted coordinates.
- Ground truth and generator metadata are not inference features.
- Splits remain separate.
- [Rejected experiments](results/optimization_experiments.json) remain public.

## Limitations

- All scored evidence is synthetic; no sponsor SEM test pairs were available.
- Internal remote-alias selection is 48.75%.
- Candidate coverage, robustness, and runtime are weaker for FinFET.
- Runtime can exceed 30 seconds on highly repetitive scenes.
- DRAM and FinFET still share one orthogonal-line rendering primitive; a more
  device-specific generator needs a generator-family holdout and retraining.

## Repository map

| Path | Purpose |
|---|---|
| `scripts/inference.py` | two images in, one coordinate out |
| `scripts/judge_check.py` | cross-platform evaluator smoke check |
| `scripts/generate_dataset.py` | deterministic DRAM/FinFET generator |
| `scripts/evaluate.py` | fixed and randomized evaluation |
| `scripts/verify_evidence.py` | recompute claims from coordinates |
| `driftforge/` | generator, matching pipeline, model, and features |
| `examples/` | one compact DRAM pair and one FinFET pair |
| `results/` | metrics, provenance, and row-level predictions |
| `docs/REFERENCES.md` | parameter-to-literature traceability |
| `SUBMISSION.md` | one-page compliance handoff |
| `tests/` | 49 generator, inference, integrity, and packaging tests |

License: [MIT](LICENSE).
