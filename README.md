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

`judge_check.py` launches inference from outside the repository directory,
loads the packaged 77-feature model automatically, checks DRAM and FinFET, and
verifies that stdout is one coordinate. `verify_evidence.py` recomputes every
headline rate from the final emitted coordinates.

Direct inference is simply:

```bash
python scripts/inference.py examples/dram/reference.png examples/dram/search.png
```

```text
(644.50, 283.50)
```

Use `--json` only when diagnostics are wanted. Normal stdout remains exactly
one `(x, y)` line.

## Measured results

The strongest result is on a pinned public reference-style generator. The
broader internal stress generator is substantially harder. They are reported
separately because pooling them would be misleading.

| Benchmark | Protocol | Within 5 px | Over 25 px | Median error |
|---|---|---:|---:|---:|
| External development | 4 seeds × 30 pairs | **93.33% (112/120)** | 6.67% | 1.44 px |
| External untouched confirmation | held-back seed × 30 pairs | **100.00% (30/30)** | 0.00% | 1.46 px |
| Internal fixed stress | 80 scene-disjoint pairs | 48.75% (39/80) | 51.25% | 62.57 px |
| Internal randomized compliance | seed 2026 × 40 pairs | 55.00% (22/40) | 45.00% | 4.36 px |

![Accuracy and catastrophic errors across named benchmarks](docs/images/06_pipeline_ablation.png)

The external source is
[`FlankerDev12/drift-sense-ref`](https://github.com/FlankerDev12/drift-sense-ref)
at commit `59376381eb284cdeb48cc727b1b75ca29c842437`. Seeds 4200–4600
were used while freezing the selection equation. Seed 4700 was held back and
then localized 30/30 pairs within 5 px. This is a public reference-style
generator, not the organizer’s hidden evaluator.

Evidence: [external summary](results/external_starter_benchmark.json) ·
[150 external predictions](results/external_starter_predictions.csv) ·
[internal metrics](results/validation_metrics.json) ·
[80 internal predictions](results/validation_predictions.csv) ·
[claim provenance](results/claim_provenance.json)

### What the numbers reveal

The correct site is a raw-correlation local maximum within 5 px in **100%** of
the fixed 80 scenes, but the raw global maximum is correct only **38.75%** of
the time. The shipped candidate pool contains a correct site in **90.0%** of
scenes. Final accuracy is **48.75%**.

That gap is the project’s central scientific finding: candidate generation is
usually successful; selecting the correct periodic copy remains difficult.

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

The complete Reference is anti-aliased and reduced by the known 10:1 pixel
ratio. It is never cropped or pasted into Search.

### 2. Preserve multiple plausible sites

For channel `c`, LatticeRank evaluates zero-mean normalized cross-correlation:

```text
ρc(x,y) = <Sxy − μxy, Tc − μT> / (||Sxy − μxy|| ||Tc − μT||)
```

Instead of trusting one maximum, it forms an adaptive union:

```text
C = ⋃c localmax(ρc ≥ max(ρc) − 0.10)
```

### 3. Cancel what repeats

The lattice basis is estimated from the Search image. Eight neighboring
lattice translations estimate the periodic background; subtracting their
median leaves missing contacts, roughness, defects, and other site-specific
structure.

```text
periodic(I) = median of neighboring lattice translations
residual(I) = I − periodic(I)
```

![Measured periodic background, residual, and uniqueness mask](docs/images/13_periodic_residual_explainer.png)

### 4. Rank independent evidence

Inside the validated device-pitch envelope, the frozen score is:

```text
score = z(periodic residual) + 0.05 z(raw ZNCC) + 0.05 z(mid-band ZNCC)
```

Broader geometry uses the packaged HGB candidate ranker, 77 structural and
scene-relative features, and residual evidence. Distance to Search centre is
not a learned feature.

![Five real candidates and every term in the frozen score](docs/images/14_candidate_evidence.png)

### 5. Handle true wallpaper honestly

When thousands of candidates coexist with almost no coarse context, the image
does not identify an absolute copy. LatticeRank detects that regime from image
statistics and uses the challenge’s centre convention directly. The seven
fixed exact-wallpaper cases improve from 0/7 to **7/7** within 5 px.

![Step-by-step measured inference on validation-000240](docs/images/12_inference_walkthrough.png)

Every panel above is generated from measured arrays. Ground truth appears only
after inference as an evaluation label.

## One success and one honest failure

When the correct alias wins, localization is subpixel. The DRAM hard-profile
case below has **0.06 px** error.

![Measured successful localization](docs/images/07_success_example.png)

The FinFET case below is the unresolved failure mode: a plausible remote copy
outranks the true site, producing **256.75 px** error.

![Measured periodic-alias failure](docs/images/08_periodic_alias_failure.png)

## Synthetic data, with explicit boundaries

DriftForge is LatticeRank’s dataset subsystem. It generates DRAM and FinFET
pairs with separate Reference/Search acquisition streams, recorded ground
truth, blur, dose/read noise, edge response, gain/gamma, scan distortion,
rotation, scale, charging, defects, and structural variation.

![Generated DRAM and FinFET examples](docs/images/02_generated_pairs.png)

```bash
python scripts/generate_dataset.py --architecture DRAM --count 30 --output-dir generated/dram
python scripts/generate_dataset.py --architecture FinFET --count 30 --output-dir generated/finfet
python scripts/validate_dataset.py generated/dram
```

The generator is deterministic for a fixed seed and physically motivated, but
it is not a proprietary microscope simulator. Every implemented parameter,
code range, rationale, and supporting literature is mapped in the
[citation matrix](docs/REFERENCES.md).

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

External confirmation, from a checkout of the pinned source:

```bash
python scripts/evaluate_external_starter.py /path/to/drift-sense-ref \
  --count 30 --seed 4700 --output reproduced-external-4700.json
```

Measured full-pipeline runtime on the 80-pair internal run was **2.86 s
median**, **6.15 s mean**, and **30.32 s P95** per pair. Repetitive scenes
produce the long candidate-count tail. See [runtime evidence](results/runtime.json).

## Scientific integrity

- Accuracy is recomputed from `pred_x`, `pred_y`, `gt_x`, and `gt_y`.
- Training, validation, external development, and confirmation identities are
  explicit and never pooled into one headline.
- Ground truth, filenames, generator metadata, and centre distance are not
  inference features.
- Failed experiments remain in the
  [optimization ledger](results/optimization_experiments.json).
- A nonlinear fusion reached 92.5% on its tuning half and collapsed to 27.5%
  on untouched scenes; it was rejected.
- Six ideas learned from strong public submissions were tested and rejected
  when they failed locked holdouts. See the optional
  [public-field review](docs/COMPETITIVE_REVIEW.md).

## Limitations

- All scored evidence is synthetic; no sponsor SEM test pairs were available.
- Internal remote-alias selection remains 48.75% and is not finalist-grade.
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

LatticeRank is the product and release. `driftforge` is the Python package and
synthetic-data subsystem. License: [MIT](LICENSE).
