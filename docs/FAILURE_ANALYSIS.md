# Failure analysis

The internal fixed benchmark has **41 catastrophic failures in 80 pairs
(51.25%)**, where catastrophic means error greater than 25 Search pixels.
Successful predictions are usually subpixel; failures jump by whole lattice
periods and often land hundreds of pixels away.

![Internal error distribution](images/05_error_cdf.png)

## Dominant mechanism: remote periodic alias selection

The shipped candidate pool contains a ≤5 px point in **72/80 = 90.0%** of
fixed pairs, but final localization succeeds in only **39/80 = 48.75%**. This
gap isolates selection as the bottleneck.

![Measured periodic-alias failure](images/08_periodic_alias_failure.png)

In `validation-000256`, the current production system selects a plausible
FinFET alias 256.75 pixels from ground truth. This is not a near-miss and not a
coordinate-convention error.

## The aliases are often physically different

A latent-scene audit of 30 catastrophic cases, before independent acquisition
noise, found:

- median NCC between true and selected latent patches: 0.859;
- only 6.7% with latent NCC ≥0.95;
- 76.7% with latent NCC ≤0.90.

Most aliases are distinguishable in the generated physical scene. The hard
part is preserving their weak non-periodic differences after 10× decimation,
blur, Poisson/read noise, gamma change, shear, jitter, and independent capture.

![Local structural correspondence](images/09_structural_comparison.png)

## Why the external benchmark is different

On the pinned public reference-style generator, the image-derived pitch gate
selects the periodic-residual consensus. It achieves **112/120 = 93.33%** on
development seeds and **30/30 = 100%** on the untouched confirmation seed.
The external development set still contains eight catastrophic aliases and a
557.97 px maximum error.

The broader DriftForge distribution includes larger relative acquisition
transform, hard low-dose noise, exact wallpaper, boundary forcing, multiple
pitch families, and out-of-distribution ranges. Its fallback ranker/residual
path does not generalize to 90%. These distributions are intentionally not
pooled.

![Named benchmark comparison](images/06_pipeline_ablation.png)

## Architecture and proposal gaps

Internal final ≤5 px accuracy is 51.28% for DRAM and 46.34% for FinFET. The
proposal stage already shows the same asymmetry: 94.87% DRAM candidate recall
and 85.37% FinFET candidate recall at `δ=0.10`.

The diagnostic `δ=0.15` pool reaches 97.44% DRAM and 87.80% FinFET recall, but
it increases the median pool from 119.5 to 546.5 candidates and is not final
localization accuracy.

## Experiments rejected by evidence

- **Nonlinear fusion:** 37/40 on the tuning half, 11/40 on the untouched half.
- **Random-forest fusion:** 37/40 tuning, 14/40 untouched.
- **Tiny Siamese encoder:** 5% alone, 33.75% in ensemble; global pooling erased
  the spatial arrangement needed to identify one cell.
- **Line-sequence fingerprint:** 15% on a locked 20-pair slice, even with oracle
  affine rectification.
- **Local keypoint consensus:** 15% on the same slice.
- **Affine residual alignment:** improved an oracle diagnostic but remained far
  below 90%.
- **256/1,024 candidate shortlists:** reduced runtime by deleting correct sites.
- **Higher synthetic defect density:** did not improve true-site rank and made
  generation much slower; reverted.

These negatives are important: a 92.5% tuning score existed and was not
shipped because its untouched result collapsed. The compact ledger is
[optimization_experiments.json](../results/optimization_experiments.json).

## Runtime failure mode

The internal fixed run measures 2.86 s median but 30.32 s P95 and 45.20 s
maximum. Runtime grows with thousands of phase-equivalent candidates. Exact
wallpaper is fast because it exits before the 77-feature descriptor; the
external pitch-gated consensus also avoids that descriptor. Broad fallback
geometry still needs a stronger cheap discriminator before an under-256
shortlist is safe.

## What is solved and what is not

Solved:

- exact output contract and packaged model loading;
- 7/7 internal exact-wallpaper cases through the centre convention;
- 90%+ on the pinned public reference-style distribution;
- provenance from generator commit through final coordinate rows.

Not solved:

- 90% on DriftForge's broad hard-noise distribution;
- worst-case remote aliases;
- safe sub-256 candidate pruning;
- real-instrument calibration without sponsor SEM pairs.

See [Results](RESULTS.md), [Method](METHOD.md), and the
[parameter traceability matrix](REFERENCES.md).
