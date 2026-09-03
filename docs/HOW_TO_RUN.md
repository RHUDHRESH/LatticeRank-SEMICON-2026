# How to run LatticeRank (Phase 2)

This is the jury-facing run sheet. The scored entry point is a single file
with a frozen signature. Nothing downloads at run time.

## 1. Environment

| Item | Required |
|---|---|
| Python | 3.11 (reference machine) |
| CPU / RAM | 4-core x86, 8 GB; no GPU |
| Network | none during the scored run |
| Weights | already inside `driftforge/models/` |

```bash
python -m pip install --disable-pip-version-check -r requirements.txt
```

`requirements.txt` is a `pip freeze` pin list. The solver uses NumPy, SciPy,
Pillow, scikit-learn, and joblib. Matplotlib and pytest are included so the
documented figures and tests can be regenerated, not because inference needs
them.

## 2. Scored command

```bash
python register.py --input pairs.csv --output predictions.csv
```

That is the only entry point. Not a notebook. Not an interactive prompt.

`pairs.csv` uses the organizer header:

```csv
pair_id,search_path,reference_path
p001,search/p001.png,reference/p001.png
```

Relative image paths are resolved against, in order:

1. the path as written
2. the directory that holds `pairs.csv`
3. the directory that holds `register.py`

Absolute paths are used as given. The working directory of the evaluator does
not have to be the package root.

RGB Set D images are accepted and converted to a single `uint8` grayscale
plane before matching.

## 3. Output contract

`predictions.csv` has exactly these columns, in this order:

```csv
pair_id,x,y,theta,scale,found,score
```

| Column | Meaning |
|---|---|
| `pair_id` | copied from the input; every input id appears once |
| `x`, `y` | match centre in search-image pixels, origin top-left, subpixel allowed |
| `theta` | rotation in degrees, counter-clockwise positive, about the match centre |
| `scale` | recovered down-scaling factor, reported in `[8, 12]` |
| `found` | `1` or `0`. When `0`, `x`, `y`, `theta`, and `scale` are `0` |
| `score` | calibrated probability that the reported coordinate is correct, in `[0, 1]` |

Rules that are enforced in code, not by convention:

- a missing row scores zero, so decode failures, solver exceptions, and
  deadline expiry still emit a row
- `found = 0` zeroes the pose columns but keeps a real `score`
- an internal crash is **not** written as a confident rejection; it uses the
  bottom of the score range (`1e-6`) and prints a diagnostic on stderr

![How to read one output row](images/v2_how_to_read.svg)

![V1 versus V2 columns](images/v2_output_contract.svg)

## 4. What `found` and `score` mean

They are different models.

- `found` answers: does the reference occur in this search image at all?
- `score` answers: is the coordinate I am about to report the correct site?

A present pair localized to the wrong lattice copy is `found = 1` with a low
`score`. That is coherent. Do not threshold `score` to invent a second
presence flag.

![Presence versus coordinate trust](images/v2_presence_vs_score.svg)

## 5. Packaged generator

The addendum also requires a documented generator in the zip. From the zip
root:

```bash
python generate_dataset.py --phase 2 --split p2_val --count 20 \
    --output-dir generated/phase2 --modality gray --seed-base 20260827
```

Equivalent:

```bash
python scripts/generate_dataset.py --phase 2 --split p2_val --count 20 \
    --output-dir generated/phase2 --modality gray --seed-base 20260827
```

Phase 1 (fixed 10x, always present):

```bash
python generate_dataset.py --architecture DRAM --count 30 --output-dir generated/dram
python generate_dataset.py --architecture FinFET --count 30 --output-dir generated/finfet
```

Cited parameter ranges: [REFERENCES.md](REFERENCES.md).
Validation gates: `python scripts/validate_phase2.py --data-root data/phase2 --splits p2_val --quick`.

## 6. Self-checks (not part of the scored run)

```bash
python -m pytest -q
python scripts/verify_evidence.py
python scripts/verify_offline.py --entry register.py --args "--input examples/pairs.csv --output predictions.csv"
python scripts/build_submission.py
python scripts/audit_phase2_contract.py dist/LatticeRank_Phase2.zip
```

Single-pair smoke using the shipped examples:

```bash
python register.py --input examples/pairs.csv --output predictions.csv
```

If `examples/pairs.csv` is not present, any three-column CSV pointing at
`examples/dram/` and `examples/finfet/` is enough.

## 7. What is in the zip

| File | Role |
|---|---|
| `register.py` | scored entry point |
| `generate_dataset.py` | documented generator (zip-root wrapper) |
| `requirements.txt` | `pip freeze` environment |
| `docs/failure_analysis.pdf` | two-page judged write-up |
| `docs/HOW_TO_RUN.md` | this sheet |
| `docs/V1_VS_V2.md` | architecture, V1 vs V2, charts |
| `docs/REFERENCES.md` | generator citations |
| `driftforge/models/*.pkl`, `*.joblib` | shipped weights, no download |

The method is the Phase 1 periodic-aware matcher extended over the disclosed
scale and rotation ranges, plus presence and confidence models. That is the
extension the addendum lists as allowed.
