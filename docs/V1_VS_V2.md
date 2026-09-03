# LatticeRank V1 → V2

## Visual architecture guide, migration notes, and measured evidence

V2 keeps the central V1 idea—match semiconductor structure while respecting
periodicity—but expands a fixed-pose locator into a complete registration
system. It now searches scale and rotation, decides whether the reference is
present, reports calibrated confidence, accepts grayscale or RGB inputs, stays
inside a time budget, and guarantees one valid output row for every input row.

> **Short version:** V1 answers **“where is this known-scale reference?”** V2
> answers **“is it present, where is it, what is its pose, and how trustworthy
> is the answer?”**

![V2 official sample scorecard](images/v2_official_scorecard.svg)

## 1. One-screen comparison

| Dimension | V1 | V2 |
|---|---|---|
| Primary job | Translation-only localization | Presence-aware 4-DoF registration |
| Required output | `(x, y)` | `(x, y, theta, scale, found, score)` |
| Reference scale | Fixed 10×, known in advance | Unknown down-scaling factor in `[8, 12]` |
| Rotation | Treated as acquisition noise | Searched and reported, nominally ±5° |
| Reference presence | Always present | May be absent |
| Input channels | Grayscale workflow | Grayscale and RGB-compatible decoding |
| Core similarity | Multi-channel ZNCC with periodic reasoning | Band-passed ZNCC across a pose grid |
| Candidate handling | Adaptive local maxima, residual ranking, tie rule | Dense pose sweep, best surface peak, local refinement |
| Confidence | Diagnostic scores | Calibrated probability of coordinate correctness |
| Failure behavior | Coordinate result | Valid zero-pose row plus low score |
| Runtime control | Measured but uncapped | Per-pair deadline with staged degradation |
| Entry point | `scripts/inference.py` | `register.py` |

### The contract grew in three directions

```mermaid
flowchart LR
    V1["V1: locate"] --> T["Translation<br/>x, y"]
    V1 --> K["Known pose<br/>10× scale"]

    V2["V2: register + decide"] --> T2["Translation<br/>x, y"]
    V2 --> P2["Pose<br/>theta, scale"]
    V2 --> D2["Decision<br/>found"]
    V2 --> C2["Trust<br/>score"]

    classDef old fill:#eaf2ff,stroke:#2f6fed,color:#172033
    classDef new fill:#e9faf3,stroke:#1c9b67,color:#172033
    class V1,T,K old
    class V2,T2,P2,D2,C2 new
```

## 2. Architecture: what stayed, what expanded, what was added

```mermaid
flowchart TB
    subgraph Shared["Shared lineage"]
        A["Anti-aliased template construction"]
        B["Normalized correlation"]
        C["Periodic-structure awareness"]
        D["Subpixel peak estimation"]
    end

    subgraph V1Only["V1 deployment path"]
        E["Known 10× normalization"]
        F["Multi-channel candidate harvest"]
        G["Periodic residual ranking"]
        H["Evidence-equivalent centre tie rule"]
        I["Return x, y"]
        E --> F --> G --> H --> I
    end

    subgraph V2Only["V2 deployment path"]
        J["Scale × rotation sweep"]
        K["Band-pass each search signal"]
        L["Dense full-image ZNCC"]
        M["Three-pass local pose refinement"]
        N["Presence decision"]
        O["Correctness probability"]
        P["Enforce output contract"]
        J --> K --> L --> M --> N --> O --> P
    end

    Shared --> V1Only
    Shared --> V2Only
```

The important design choice is extension rather than replacement. The V2 code
reuses the same physical scale normalization and normalized-correlation family,
then searches the dimensions that are no longer supplied by the task.

### Change map

```mermaid
flowchart LR
    A["Fixed scale"] -- "expanded" --> B["9 scale hypotheses<br/>8.0 to 12.0"]
    C["Rotation ignored"] -- "expanded" --> D["5 coarse angles<br/>−6° to +6°"]
    E["Always present"] -- "added" --> F["Presence classifier<br/>found = 0 or 1"]
    G["Coordinate only"] -- "added" --> H["Pose + confidence"]
    I["Best effort"] -- "hardened" --> J["One row per pair<br/>even on failure"]
```

## 3. V2 end-to-end data flow

```mermaid
flowchart TD
    CSV["pairs.csv"] --> PATHS["Resolve paths relative to CSV and package"]
    PATHS --> LOAD["Decode reference + search"]
    LOAD --> GRAY["Normalize supported image modes to uint8 grayscale"]
    GRAY --> BASE["Build anti-aliased template once per scale"]
    BASE --> ROT["Rotate the small template for each angle"]
    ROT --> BP["Difference-of-Gaussians band-pass"]
    BP --> ZNCC["Dense full-image ZNCC"]
    ZNCC --> BEST["Best x, y, scale, theta across 45 poses"]
    BEST --> REFINE["Alternate translation, scale, rotation<br/>for three passes"]
    REFINE --> FEATURES["Scene + peak + refinement features"]
    FEATURES --> FOUND["Presence probability → found"]
    FEATURES --> SCORE["Correctness probability → score"]
    FOUND --> CONTRACT["Clamp pose, zero absent pose, emit row"]
    SCORE --> CONTRACT
    CONTRACT --> OUT["predictions.csv"]
```

### Processing sequence for one pair

```mermaid
sequenceDiagram
    participant R as register.py
    participant D as Dense sweep
    participant F as Refiner
    participant P as Presence model
    participant C as Correctness model
    participant O as Output writer

    R->>R: Decode and normalize images
    R->>D: reference, search, deadline
    loop 9 scales × 5 rotations
        D->>D: build/rotate template and score surface
    end
    D-->>R: coarse x, y, scale, theta, peak
    R->>F: refine selected pose
    F-->>R: subpixel position and pose
    R->>P: scene-level evidence
    P-->>R: found decision
    R->>C: peak and refinement evidence
    C-->>R: correctness probability
    R->>O: contract-safe row
```

## 4. How V2 works, stage by stage

### 4.1 Input and path safety

`register.py` accepts several sensible header spellings for the pair identifier,
reference path, and search path. Relative image paths are tried against the CSV
directory and the submission directory, so the evaluator can launch the program
from an unrelated working directory.

Supported single-frame inputs include common grayscale, palette, RGB, RGBA,
CMYK, and numeric image modes. They are converted deterministically to a single
`uint8` plane before matching.

```mermaid
flowchart LR
    A["CSV value"] --> B{"Absolute path?"}
    B -- Yes --> C["Use it"]
    B -- No --> D{"Exists as written?"}
    D -- Yes --> C
    D -- No --> E{"Exists beside pairs.csv?"}
    E -- Yes --> C
    E -- No --> F{"Exists beside register.py?"}
    F -- Yes --> C
    F -- No --> G["Emit a failure row<br/>and diagnostic"]
```

### 4.2 Pose search

V1 receives the scale. V2 must estimate it, so `dense_pose_search` evaluates
nine scales and five rotations. Anti-aliasing and down-scaling are computed once
per scale; only the much smaller template is rotated inside the angle loop.

```text
scale grid    = 8.0, 8.5, 9.0, …, 11.5, 12.0
rotation grid = −6°, −3°, 0°, +3°, +6°
pose count    = 9 × 5 = 45
```

For every pose, V2 computes a full valid ZNCC surface and retains the global
best finite response. That dense search avoids a proposal cap silently dropping
the correct site.

```mermaid
flowchart TB
    S["Scale 8.0"] --> R1["−6°"] & R2["−3°"] & R3["0°"] & R4["+3°"] & R5["+6°"]
    S2["Scale 8.5"] --> Q1["−6°"] & Q2["−3°"] & Q3["0°"] & Q4["+3°"] & Q5["+6°"]
    DOT["⋮"] --> SN["Scale 12.0 × five angles"]
    R1 & R2 & R3 & R4 & R5 & Q1 & Q2 & Q3 & Q4 & Q5 & SN --> MAX["Best finite peak across all surfaces"]
```

### 4.3 Band-passed matching

Low-frequency charging and brightness drift can dominate raw intensity while
carrying little positional information. V2 applies a Difference-of-Gaussians
response before ZNCC:

```text
band_pass(I) = Gaussian(I, σ=2) − Gaussian(I, σ=8)
```

The measured gain is largest at the hardest severity level.

![Raw versus band-passed localization by severity](images/v2_filter_ablation.svg)

### 4.4 Full-resolution refinement

The pose grid is intentionally coarse. After site selection, V2 performs three
alternating passes:

```mermaid
flowchart LR
    A["Coarse pose"] --> B["Local translation<br/>±32 px window"]
    B --> C["Scale oracle<br/>±1.0"]
    C --> D["Rotation oracle<br/>±3°"]
    D --> E{"Three passes?"}
    E -- No --> B
    E -- Yes --> F["Subpixel x, y,<br/>scale, theta"]
```

Translation uses a local full-resolution correlation and a parabolic peak fit.
When several periodic peaks are nearly equivalent, the refiner prefers the one
nearest the incoming estimate rather than jumping by a lattice period.

### 4.5 Presence and correctness are different questions

V2 deliberately does not collapse `found` and `score` into one value:

- `found` estimates whether the reference occurs anywhere in the search image.
- `score` estimates whether the reported coordinate is correct.

A reference can be present while the selected lattice copy is wrong. In that
case `found = 1` and a low `score` is coherent and useful.

```mermaid
quadrantChart
    title Presence and coordinate trust
    x-axis Low presence evidence --> High presence evidence
    y-axis Low coordinate trust --> High coordinate trust
    quadrant-1 Present and localized
    quadrant-2 Unusual: verify decision
    quadrant-3 Absent or unusable
    quadrant-4 Present but likely mislocalized
```

### 4.6 Output contract

Every input identifier must appear exactly once in the output.

| Column | Meaning | V2 enforcement |
|---|---|---|
| `pair_id` | Input identity | One output row per unique identifier |
| `x`, `y` | Search-image centre | Finite and clipped to image bounds |
| `theta` | CCW rotation in degrees | Clipped to the disclosed reportable range |
| `scale` | Reference-to-search down-scaling factor | Clipped to `[8, 12]` |
| `found` | Presence flag | Exactly `0` or `1` |
| `score` | Probability coordinate is correct | Finite, on the probability scale |

```mermaid
flowchart TD
    A{"Pair processed?"}
    A -- No --> B["found = 0<br/>pose = 0,0,0,0<br/>bottom score"]
    A -- Yes --> C{"Presence decision"}
    C -- Absent --> D["found = 0<br/>pose = 0,0,0,0<br/>retain real score"]
    C -- Present --> E["found = 1<br/>clamp finite pose<br/>retain real score"]
    B --> F["Write exactly one row"]
    D --> F
    E --> F
```

### 4.7 Runtime and graceful degradation

The dense sweep is the expensive stage. Later stages consult a deadline and can
be skipped when the remaining budget is too small. A valid row is still emitted.

```mermaid
flowchart LR
    A["Dense search<br/>required"] --> B{"Time for refinement?"}
    B -- Yes --> C["Refine pose"]
    B -- No --> D["Keep coarse pose"]
    C --> E{"Time for presence features?"}
    D --> E
    E -- Yes --> F["Presence model"]
    E -- No --> G["Peak-threshold fallback"]
    F --> H["Contract-safe row"]
    G --> H
```

![V2 runtime distribution](images/v2_runtime.svg)

## 5. Measured results

### 5.1 V2 official sample

The tracked aggregate evaluation contains 20 organizer-generated pairs: eight
nominal, six degraded, four absent, and two RGB. It was run once after the
solver was frozen; the file records aggregate metrics and does not reproduce
organizer imagery or coordinates.

![Official localization credit by set](images/v2_official_sets.svg)

| Block | Measurement | Points |
|---|---:|---:|
| Localization | weighted credit `0.9704` | **38.82 / 40** |
| Pose | gated mean credit `0.8656` | **17.31 / 20** |
| Rejection | F1 `0.9412` | **14.12 / 15** |
| Confidence | AUC `0.8889` | **8.89 / 10** |
| RGB | Set D `1.000`, A–C `0.9704` | **+6 bonus** |

Scored subtotal: **79.14 / 85**, before the six-point RGB bonus.

Source: [`results/phase2_experiments/official_sample_evaluation.json`](../results/phase2_experiments/official_sample_evaluation.json)

### 5.2 V1 benchmarks

V1 results come from four named protocols. These are not direct V2 comparisons:
V1 uses known pose and always-present pairs, while V2 solves more outputs under
a different evaluation protocol.

![V1 localization benchmarks](images/v1_benchmarks.svg)

| V1 protocol | Within 5 px | Median error |
|---|---:|---:|
| External development, 120 pairs | 93.33% | 1.44 px |
| External untouched confirmation, 30 pairs | 100.00% | 1.46 px |
| Internal fixed stress, 80 pairs | 48.75% | 62.57 px |
| Internal randomized compliance, 40 pairs | 55.00% | 4.36 px |

Sources: [`external_starter_benchmark.json`](../results/external_starter_benchmark.json),
[`validation_metrics.json`](../results/validation_metrics.json), and
[`evaluation_30plus.json`](../results/evaluation_30plus.json).

### 5.3 Do not mix official and internal V2 numbers

The internal generator renders the reference and search as independent
acquisitions. The organizer sample uses a different acquisition relationship.
The paired experiment below isolates how much that one design choice changes
the internal localization problem.

![Independent acquisition versus shared-canvas reference](images/v2_acquisition_gap.svg)

```mermaid
flowchart LR
    W["One latent wafer scene"] --> A1["Independent reference acquisition"]
    W --> A2["Independent search acquisition"]
    A1 --> HARD["Noise does not identify the true copy<br/>30% within 1 px"]
    A2 --> HARD

    S["One rendered search canvas"] --> CROP["Reference derived from same canvas"]
    S --> CROP
    CROP --> EASY["Shared detail helps identify the copy<br/>65% within 1 px"]
```

This is why the roughly 30% internal stress result must not be presented as a
forecast of the 97% official localization credit. The protocols answer
different questions.

Source: [`samecanvas_bound.json`](../results/phase2_experiments/samecanvas_bound.json).

## 6. Operational reading of V2 output

```mermaid
flowchart TD
    A["Read one output row"] --> B{"found = 0?"}
    B -- Yes --> C["Treat pose as intentionally zeroed"]
    C --> D{"score also very low?"}
    D -- Yes --> E["Consistent absence or unusable input"]
    D -- No --> F["Review: absence decision and coordinate trust differ"]
    B -- No --> G{"score high?"}
    G -- Yes --> H["Coordinate is suitable for downstream use"]
    G -- No --> I["Reference likely exists,<br/>but confirm the selected site"]
```

The output should be consumed as two layers:

1. Use `found` for presence/absence workflow.
2. Use `score` to prioritize manual review or downstream automation.

Never infer pose from a `found = 0` row; its pose fields are zero by contract.

## 7. Failure modes and safeguards

| Failure mode | V1 exposure | V2 safeguard |
|---|---|---|
| Wrong working directory | Paths may fail globally | Resolve relative to CSV and package |
| Unsupported or corrupt image | Run may stop | Emit a valid failure row and diagnostic |
| Constant image / invalid ZNCC | Non-finite peak | Reject non-finite evidence |
| Correct site omitted by proposal cap | Candidate recall ceiling | Dense full-image sweep |
| Periodic peak jump during refinement | Whole-period error | Prefer evidence-equivalent peak nearest seed |
| Unknown pose | Outside V1 contract | Search 45 coarse pose hypotheses, then refine |
| Absent reference | Forced false localization | Independent presence decision |
| Low-confidence present pair | Hidden behind a coordinate | Separate correctness score |
| Slow machine | Tail can exceed budget | Deadline-aware stage shedding |
| Internal exception | Missing or malformed output | One contract-safe row per pair |

## 8. Run and verify V2

```bash
python -m pip install --disable-pip-version-check -r requirements.txt
python register.py --input pairs.csv --output predictions.csv
python scripts/audit_phase2_contract.py dist/LatticeRank_Phase2.zip
```

The expected output header is:

```csv
pair_id,x,y,theta,scale,found,score
```

Full repository verification:

```bash
python -m pytest -q
python scripts/verify_evidence.py
python scripts/build_submission.py
```

## 9. Rebuild every chart in this guide

The SVG files are not hand-entered. One standard-library script loads the
tracked evidence JSON and regenerates all six charts:

```bash
python scripts/build_v2_visuals.py
```

Generated files:

- `docs/images/v2_official_scorecard.svg`
- `docs/images/v2_official_sets.svg`
- `docs/images/v1_benchmarks.svg`
- `docs/images/v2_filter_ablation.svg`
- `docs/images/v2_acquisition_gap.svg`
- `docs/images/v2_runtime.svg`

## 10. Documentation map

```mermaid
flowchart TB
    README["README.md<br/>quick start + headline results"]
    GUIDE["V1_VS_V2.md<br/>architecture + migration + visuals"]
    FAIL["failure_analysis.md<br/>measured V2 limits and negative results"]
    FIND["PHASE2_FINDINGS.md<br/>experiment ledger"]
    REF["REFERENCES.md<br/>parameter provenance"]
    DATA["BULK_DATASET.md<br/>large corpus workflow"]

    README --> GUIDE
    GUIDE --> FAIL
    GUIDE --> FIND
    GUIDE --> REF
    GUIDE --> DATA
```

## 11. File-level implementation map

| Area | Main files |
|---|---|
| V2 command-line and contract | `register.py` |
| Deadline and degradation | `driftforge/budget.py` |
| Dense pose sweep | `driftforge/dense.py` |
| Pose conventions and template construction | `driftforge/pose.py` |
| Local pose refinement | `driftforge/refine.py` |
| Presence decision | `driftforge/presence_model.py` |
| Correctness probability | `driftforge/correctness_model.py` |
| V2 synthetic generator | `driftforge/generator.py`, `driftforge/phase2.py` |
| Contract audit | `scripts/audit_phase2_contract.py` |
| Submission packaging | `scripts/build_submission.py` |
| Visual regeneration | `scripts/build_v2_visuals.py` |

## 12. Bottom line

V1 proves the periodic-localization core. V2 turns that core into a bounded,
presence-aware registration product:

```text
V1 = locate(x, y | fixed pose, present)

V2 = decide presence
   + search translation, scale, rotation
   + refine the selected pose
   + estimate coordinate correctness
   + guarantee a valid row under every failure mode
```

The official sample result—**79.14 / 85 scored points plus the RGB bonus**—shows
that the expanded contract works end to end. The internal stress results remain
useful for engineering because they expose the specific acquisition condition
under which periodic site identity becomes the limiting problem.
