# LatticeRank — judge handoff

**Challenge:** SEMICON India Hackathon 2026, Drift-Sense<br>
**Release:** LatticeRank 1.0<br>
**Contract:** two image paths in, exactly one `(x, y)` Search coordinate out<br>
**Repository:** https://github.com/RHUDHRESH/LatticeRank-SEMICON-2026

## Fast path

Python 3.12+ and a CPU are sufficient. The model is included.

```bash
git clone https://github.com/RHUDHRESH/LatticeRank-SEMICON-2026.git
cd LatticeRank-SEMICON-2026
python -m venv .venv
```

Activate the environment, then:

```bash
python -m pip install --disable-pip-version-check -r requirements.txt
python -m pip check
python scripts/judge_check.py
python scripts/verify_evidence.py
```

The first script runs both architectures from outside the repository working
directory. The second recomputes all published rates from final coordinates.

## Compliance map

| Requirement | Evidence |
|---|---|
| Standalone inference | [`scripts/inference.py`](scripts/inference.py) |
| Automatic model loading | [`driftforge/models/hgb_r2.joblib`](driftforge/models/hgb_r2.joblib) |
| DRAM and FinFET generation | [`scripts/generate_dataset.py`](scripts/generate_dataset.py) |
| Recorded ground truth and independent acquisitions | [`tests/test_generator_integrity.py`](tests/test_generator_integrity.py) |
| Reproducible training | [`scripts/train_ranker.py`](scripts/train_ranker.py) and scene-disjoint manifests |
| Complete dependency freeze | [`requirements.txt`](requirements.txt) |
| Parameter citations | [`docs/REFERENCES.md`](docs/REFERENCES.md) |
| 30+ randomized evaluation | [`results/evaluation_30plus.json`](results/evaluation_30plus.json) |
| Runtime | [`results/runtime.json`](results/runtime.json) |
| Final-coordinate evidence | [`results/validation_predictions.csv`](results/validation_predictions.csv) |
| Claim provenance | [`results/claim_provenance.json`](results/claim_provenance.json) |
| Tests and packaging | [`tests/`](tests/) |

## Results that may be cited

| Distribution | Protocol | Within 5 px | Over 25 px |
|---|---:|---:|---:|
| Pinned external development | 120 pairs | **93.33%** | 6.67% |
| Pinned external confirmation | untouched 30 pairs | **100.00%** | 0.00% |
| Internal fixed stress | 80 pairs | 48.75% | 51.25% |
| Internal randomized compliance | 40 pairs | 55.00% | 45.00% |

Always name the distribution and denominator. The external source is the
public reference-style `FlankerDev12/drift-sense-ref` generator at commit
`59376381`; it is not the hidden official evaluator. External and internal
rates must not be pooled.

## Direct commands

```bash
python scripts/inference.py examples/dram/reference.png examples/dram/search.png
python scripts/inference.py examples/finfet/reference.png examples/finfet/search.png
python scripts/generate_dataset.py --architecture DRAM --count 2 --output-dir generated/dram
python scripts/generate_dataset.py --architecture FinFET --count 2 --output-dir generated/finfet
python -m pytest -q
```

For the complete visual explanation, method, results, failure analysis, and
reproduction commands, read the [README](README.md). The separate portal/PDF
submission still requires the registered team’s identity and contact fields;
those details are intentionally not invented here.
