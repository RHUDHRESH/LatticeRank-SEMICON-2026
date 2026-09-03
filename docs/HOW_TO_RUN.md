# How to run LatticeRank

Jury sheet for the scored zip. One entry point. Nothing downloads.

```bash
python -m pip install --disable-pip-version-check -r requirements.txt
python register.py --input pairs.csv --output predictions.csv
```

Smoke, using only files inside this package:

```bash
python register.py --input examples/pairs.csv --output predictions.csv
```

| | |
|---|---|
| Python | 3.11 |
| Machine | 4-core x86, 8 GB RAM, no GPU |
| Network | none |
| Weights | `driftforge/models/` already in the zip |

`requirements.txt` is a `pip freeze` pin list. Inference needs NumPy, SciPy,
Pillow, scikit-learn, and joblib. Matplotlib and pytest are there so figures
and tests can be regenerated.

---

## Input

Organizer header, one row per pair:

```csv
pair_id,search_path,reference_path
p001,search/p001.png,reference/p001.png
```

Other plausible column names are accepted (`reference`, `ref_path`, `search`,
`id`, …). Relative image paths are tried in this order:

1. The path as written.
2. Beside `pairs.csv`.
3. Beside `register.py`.

Absolute paths are used as given. The process working directory does not have
to be the package root. RGB, RGBA, palette, and numeric single-frame images
are converted to `uint8` grayscale.

If a path or decode fails, that pair still produces a row with `found = 0`
and `score = 1e-6`, plus a line on stderr.

---

## Output

Exact header, this order, one row per unique `pair_id`:

```csv
pair_id,x,y,theta,scale,found,score
```

| Column | Contract |
|---|---|
| `pair_id` | Copied from the input. Every input id appears once. |
| `x`, `y` | Search-image centre, origin top-left, subpixel allowed. |
| `theta` | Degrees, **CCW positive**, about the match centre. |
| `scale` | Down-scaling factor, reported in `[8, 12]`. Not `1/s`. |
| `found` | `1` or `0`. When `0`, `x y theta scale` are `0`. |
| `score` | P(coordinate is correct), monotone in `[0, 1]`. |

Illustrative rows (not organizer data):

```csv
pair_id,x,y,theta,scale,found,score
present-ok,512.25,384.10,-1.40,9.50,1,0.81
present-uncertain,640.00,200.50,0.20,10.00,1,0.22
absent,0.00,0.00,0.00,0.00,0,0.07
unusable,0.00,0.00,0.00,0.00,0,0.000001
```

How to act on a row:

| `found` | `score` | Meaning |
|---|---|---|
| 1 | high (~0.5+) | Use the coordinate. |
| 1 | low | Reference is likely in the image; confirm the selected lattice site. |
| 0 | very low | Absence, or the pair could not be processed. Pose is zero by contract. |
| 0 | not tiny | Rare disagreement: review. Do not invent pose from this row. |

![Operational reading of one row](images/v2_how_to_read.svg)

*Use `found` for the presence workflow. Use `score` for coordinate trust.*

![Output columns](images/v2_output_contract.svg)

*A missing row scores zero. Failures still emit a valid row.*

![Presence and score are different axes](images/v2_presence_vs_score.svg)

*`found` is not a rounded `score`. A wrong lattice copy is present-but-incorrect.*

---

## Generator

Required by the addendum. Zip-root wrapper and implementation are equivalent:

```bash
python generate_dataset.py --phase 2 --split p2_val --count 20 \
    --output-dir generated/phase2 --modality gray --seed-base 20260827
```

Phase 1 (fixed 10×, always present):

```bash
python generate_dataset.py --architecture DRAM --count 30 --output-dir generated/dram
python generate_dataset.py --architecture FinFET --count 30 --output-dir generated/finfet
```

Cited parameter ranges: [REFERENCES.md](REFERENCES.md).
Gates: `python scripts/validate_phase2.py --data-root data/phase2 --splits p2_val --quick`.

---

## Self-checks (not scored)

```bash
python -m pytest -q
python scripts/verify_evidence.py
python scripts/verify_offline.py --entry register.py --args "--input examples/pairs.csv --output predictions.csv"
python scripts/build_submission.py
python scripts/audit_phase2_contract.py dist/LatticeRank_Phase2.zip
```

| File | Role |
|---|---|
| `register.py` | Scored entry point |
| `generate_dataset.py` | Documented generator |
| `requirements.txt` | Frozen environment |
| `docs/README.md` | Reading order |
| `docs/HOW_TO_RUN.md` | This sheet |
| `docs/V1_VS_V2.md` | Architecture and charts |
| `docs/failure_analysis.pdf` | Judged write-up, ≤ 2 pages |
| `driftforge/models/` | Shipped weights |

The method is the Phase 1 periodic-aware matcher extended over the disclosed
`[8, 12]` zoom and ±5° rotation ranges, plus presence and confidence models.
That extension is listed as allowed.
