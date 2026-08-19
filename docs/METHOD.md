# Method

LatticeRank returns the centre of a 1,000 × 1,000 high-magnification Reference
inside a 1,000 × 1,000 Search image covering ten times the physical field.
Coordinates are `(x, y)`: column, row, top-left origin, Search pixels.

![Cross-scale localization task](images/01_localization_task.png)

## Why ordinary correlation fails

At the correct scale, every lattice copy can correlate strongly. The useful
signal is the small part that does **not** repeat: mat boundaries, missing
contacts, particles, line variation, and local acquisition-consistent context.
LatticeRank therefore separates proposal from identity:

```text
periodic carrier → proposes phase-valid locations
non-periodic residual → identifies the physical copy
evidence equivalence → invokes the centre rule only for a real tie
```

## Production path

1. **Normalize scale.** Gaussian anti-aliasing precedes the known 10:1
   Reference reduction. The complete Reference becomes an approximately
   100 × 100 Search-scale template.
2. **Build independent evidence maps.** ZNCC is computed on robust intensity,
   a sigma-3 minus sigma-15 mid-band, and a polarity-invariant doubled-angle
   gradient field.
3. **Harvest phase hypotheses.** For each channel, local maxima within
   `δ=0.10` of that channel's maximum enter a union pool. The cap is 8,000.
4. **Resolve exact wallpaper before ranking.** At least 2,500 candidates, low
   20 px coarse context, and low additional decay at 60 px identify a
   non-observable wallpaper regime. LatticeRank returns Search centre as the
   challenge specifies and skips expensive features.
5. **Estimate the Search lattice.** Spectral and autocorrelation evidence
   recover two basis vectors, their pitch envelope, orientation, confidence,
   and phase residuals.
6. **Use residual consensus in its validated envelope.** Eight lattice-shifted
   copies estimate the periodic background. Their median is subtracted from
   Search and template. The production score is

   ```text
   score = z(residual) + 0.05 z(raw) + 0.05 z(mid-band)
   ```

   Candidates within 0.025 of the maximum are evidence-equivalent. The one
   nearest Search centre wins only inside this set.
7. **Fall back for broader geometry.** Outside the validated pitch envelope,
   77 features describe scene-relative peak strength, lattice phase, noise,
   tiny-shift alignment, blockwise correspondence, gradients, profiles, and
   local spectra. A balanced `HistGradientBoostingClassifier` supplies a
   scene-normalized score, which is fused with the periodic residual. The
   fallback equivalence margin is 0.05.
8. **Return one in-bounds coordinate.** Inference never abstains and normal
   stdout contains exactly one `(x, y)` line.

Distance to Search centre is prohibited as a learned feature. It is used only
after image evidence defines an equivalence set or when exact wallpaper makes
the physical copy non-identifiable.

![Measured inference walkthrough](images/12_inference_walkthrough.png)

Every panel above is regenerated from `validation-000240`: full Reference,
Search-scale template, Search coordinate, raw response/candidates, residual
map, and final selected neighbourhood.

## Residual derivation

For lattice basis vectors `v1`, `v2`, form eight shifted images at
`±v1`, `±v2`, `±(v1+v2)`, and `±(v1−v2)`. Then:

```text
periodic(I) = median(shifted copies)
residual(I) = I − periodic(I)
uniqueness(T) = std(shifted template copies)
```

Weighted ZNCC gives high influence to template pixels whose translated copies
disagree. Repeated fins and lines cancel; local structural identity survives.

## What the gate means

The residual-only fast path is not selected from filenames, generator metadata,
or ground truth. It uses image-derived Search pitches. On the pinned external
reference-style benchmark it gives 93.33% development accuracy and 100% on the
untouched 30-pair confirmation seed. On DriftForge's wider transformations and
noise, the fallback remains only 48.75% on 80 pairs. See [Results](RESULTS.md).

## Reproduce one inference

```bash
python scripts/inference.py examples/dram/reference.png examples/dram/search.png
python scripts/inference.py --json examples/dram/reference.png examples/dram/search.png
```

The JSON form exposes candidate count, selection mode, wallpaper diagnostic,
lattice compatibility, evidence weights, and model provenance.
