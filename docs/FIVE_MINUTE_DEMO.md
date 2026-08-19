# Five-minute judge demonstration

This path exercises the release exactly as submitted: pinned environment,
packaged weights, two architectures, the one-coordinate contract, and
coordinate-derived benchmark math. It does not download data or weights at
runtime.

## 0:00–1:30 · Create the clean environment

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on POSIX or
`.\.venv\Scripts\Activate.ps1` in PowerShell, then install the complete
competition freeze:

```bash
python -m pip install --disable-pip-version-check -r requirements.txt
python -m pip check
```

Python 3.12 or newer is required. No GPU is required.

## 1:30–3:00 · Run the release gate

```bash
python scripts/judge_check.py
```

The check loads the bundled 77-feature model, prints its SHA-256 prefix, runs
both packaged DRAM and FinFET examples from a directory outside the repository,
and verifies that stdout is exactly one in-bounds `(x, y)` coordinate.

## 3:00–3:30 · Inspect one inference

```bash
python scripts/inference.py examples/dram/reference.png examples/dram/search.png
python scripts/inference.py examples/dram/reference.png examples/dram/search.png --json
```

The first command emits only `(644.50, 283.50)`. The JSON form exposes the
candidate count, selection mode, equivalence-set size, residual use, model
checksum, and coordinate convention.

![Measured inference walkthrough](images/12_inference_walkthrough.png)

## 3:30–4:00 · Recompute the claims from coordinates

```bash
python scripts/verify_evidence.py
```

This script ignores prose claims. It recomputes Euclidean error from every
`pred_x`, `pred_y`, `gt_x`, and `gt_y`, then checks the fixed, randomized, and
external development/confirmation rates against their JSON records.

## 4:00–5:00 · Follow one result back to its source

Open these in order:

1. [`results/validation_predictions.csv`](../results/validation_predictions.csv)
   — the final emitted coordinates, one row per scene.
2. [`results/validation_metrics.json`](../results/validation_metrics.json)
   — protocol, model hash, manifest hash, code-content hash, metrics, runtime.
3. [`results/external_starter_benchmark.json`](../results/external_starter_benchmark.json)
   and [`external_starter_predictions.csv`](../results/external_starter_predictions.csv)
   — pinned source revision, frozen equation, and 150 final external rows.
4. [`results/claim_provenance.json`](../results/claim_provenance.json)
   — claim-to-row-field mapping.
5. [`REFERENCES.md`](REFERENCES.md) — parameter-by-parameter physical rationale
   and two or three sources per implemented augmentation family.

For the full check after the demonstration:

```bash
python -m pytest -q
python scripts/evaluate.py randomized --count 40 --seed 2026 --output-dir reproduced-randomized
```

The benchmark is intentionally slower than the smoke demonstration because
highly periodic scenes can create thousands of candidate aliases.
