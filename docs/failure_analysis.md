# LatticeRank Phase 2 — Failure Analysis

All numbers below are measured on our own generator (`data/phase2/p2_val` +
`p2_stress`), which is harder than the disclosed blind-set spec. We measured
*why* rather than assuming: the 40% decoy rate and the severity ladder — the
two things it was built to stress — turn out to have **no measurable effect**
on localization (§1b). What does is that we render the reference and the search
as independent acquisitions, worth **35 points of ≤1 px rate**. These are
diagnostic numbers, not predictions for the organizers' Set A.

## 1. Diagnosis: localization error is binary, and selection is the bottleneck

At n=180 the pairs landing within 1, 2, 3 and 5 px are the **same 33 pairs** —
not one pair fell between 1 and 5 px. When the pipeline picks the right
lattice site the answer is already sub-pixel; when it does not, the error is
hundreds of pixels. Consequence: **mean tiered credit equals site-selection
accuracy**, so subpixel refinement cannot move this score block at all.

Three independent measurements pin down where the error comes from
(`results/gt_pose_ceiling.json`, n=316; `PHASE2_FINDINGS.md` §3, §5):

| stage | rate | note |
|---|---:|---|
| Detection (true site is *a* local max) | ~99–100% | not the problem |
| Pool (true site survives harvest, GT pose) | 90–95% | not the problem |
| **Selection** (ranker picks it) | **15–43%** | **the bottleneck** |

Pool-high, selection-poor is an *identity-evidence* gap, not a recall, cap or
NMS problem. This sharpens Phase 1's own flagged weakness — internal
selection at 48.75% against a 90% pool (`README.md`) — Phase 2 confirms the
same shape and shows it did not improve under unknown pose: pool recall held
at 90–95% while selection fell to 15–43%. Root cause, measured directly: on
67,370 held-out candidates the current lattice-relative features
(`disp_periods_x`, `disp_periods_y`, `lat_phase_res`, `parity_even`) score
**AUC 0.38–0.62** against the true-site label — near chance
(`results/phys_eval_full.json`). Seven hand-designed selectors and two
retrained rankers all landed in the same 0.18–0.28 band for the same reason:
re-weighting existing evidence cannot fix a feature set that does not carry
the signal.

## 1b. Why our corpus is hard — measured, not assumed

Set A is specified as "noise comparable to the Phase 1 sample prompt" and decoys
are never mentioned, so the obvious worry was that we had made the task
artificially hard. Three arms of the shipped pipeline, no code changes
(`results/phase2_experiments/set_a_calibration.json`):

| arm | n | ≤1 px | 95% CI | pts/40 |
|---|---:|---:|---|---:|
| severity 0–1, **no decoys** ("Set A-like") | 180 | 30.0% | [23.8, 37.1] | 12.3 |
| severity 0–1, **with decoys** | 110 | 31.8% | [23.9, 41.0] | 14.8 |
| full severity mix | 110 | 30.0% | [22.2, 39.1] | 12.8 |

Indistinguishable. **Neither decoys nor severity move localization**, so Set A
conditions should not be expected to rescue the block.

What does move it is a design choice in the generator. We render the reference
and the search as **independent acquisitions** of one latent scene — separate
RNG streams, so the noise realizations differ. Generators that instead crop the
reference from the rendered search pixels leave the noise pattern as a unique
fingerprint of the true site. Paired, n=60, identical pairs, only the reference
construction differing (`samecanvas_bound.json`):

| arm | ≤1 px | 95% CI | pts/40 | median err |
|---|---:|---|---:|---:|
| independent acquisition (ours) | 30.0% | [19.9, 42.5] | 12.9 | 203.62 px |
| reference cropped from search pixels | **65.0%** | [52.4, 75.8] | **33.5** | **0.87 px** |

Exact McNemar 26 vs 5, **p = 1.9e-4**, Δ **+35.0 pp**. By severity: +11.8 /
+33.3 / +18.2 / **+90.9** pp at levels 0–3 — at severity 3 the pipeline goes
1/11 → 11/11. That is the noise-fingerprint signature: severe noise is
catastrophic when independent and a strong unique cue when shared.

This is an **upper bound, not a prediction** — a real same-canvas generator
still applies its own reference-side PSF and noise, so an unknown generator
lands inside [12.9, 33.5]/40. The point stands regardless: the 20-point spread
on the largest scored block is set by a generator design choice we cannot
observe, not by anything reachable in the solver. The pipeline is sub-pixel
(0.87 px median) the moment a shared-noise cue exists.

## 1c. The ceiling, measured with an oracle

The decisive test is not "can we rank better" but "is the true site separable at
all". Hand the matcher the **ground-truth pose** — the exact scale and rotation —
and remove search from the problem entirely
(`results/phase2_experiments/localization_ceiling.json`, n=80;
`dense_residual_ceiling.json`, n=60):

| arm | ≤1 px | 95% CI |
|---|---:|---|
| shipped (global argmax over 45 poses) | 25.0% | [16.8, 35.5] |
| **oracle: ground-truth pose, band-passed ZNCC** | **31.2%** | [22.2, 42.1] |
| **oracle: ground-truth pose, dense periodic residual** | **36.7%** | [25.6, 49.3] |

Perfect pose knowledge is worth **~6 pp, not 70**. At the true pose the true site
is the global maximum in only **14–16 of 60–80** pairs, with a median of 10–22
locations scoring higher. Dense residual matching — periodic cancellation, the
mechanism at the heart of the Phase 1 method — is statistically
indistinguishable from band-passed ZNCC (paired McNemar 5 v 4, **p = 1.00**).

Two consequences follow, and they are the honest boundary of this work:

1. **Pose search is not the bottleneck**, so no amount of grid refinement,
   coarse-to-fine strategy or pose modelling can help. This is consistent with
   the earlier null result at 45→99→187 poses.
2. **Any rule computed over these surfaces is bounded above by ~35%**, because
   the ranking cannot promote a site the similarity places 20th. Candidate
   ranking, feature re-weighting and threshold tuning are all such rules — which
   is why ten families of them all returned the same 0.18–0.28 band.

The cross-pose information the sweep discards was tested and is **not** usable:
summing across poses scores 2.5%, vote counts 1.2%, peakiness 18.8%, against
25.0% for the shipped rule. Neighbouring poses are too correlated for
accumulation to reinforce anything; it smears the surface instead.

### Seven targeted attacks on the diagnosed defects

The loss splits into disjoint buckets at the estimated pose: **22.5%** where the
truth is not among the candidates at all, **45.0%** where it is a candidate but
is not picked. Each was attacked directly
(`results/phase2_experiments/localization_targeted_attacks.json`):

| diagnosed defect | attack | result |
|---|---|---|
| pose error | oracle ground-truth pose | +6 pp only |
| discarded cross-pose evidence | sum / votes / peakiness | 2.5 / 1.2 / 18.8% vs 25.0% |
| wrong similarity | dense periodic residual | p = 1.00, no change |
| ranking among near-ties | 5 rules on identical shortlists | all worse than peak (35.0%) |
| truth suppressed by NMS | widen the shortlist | argmax is NMS-independent — cannot help |
| true site scores too low | trimmed/robust ZNCC ×4 | monotonically worse |
| DoG band never tuned for this | 12-config sweep, 3 regimes | shipped (2,8) is best |
| missing Phase 1 ranking stage | residual re-rank of shortlist | +1/−1, p = 1.00 |

The band sweep is the other informative negative. The lattice pitch is 3.55–11.6 px and
the shipped band passes 2–8 px, sitting directly on it — apparently the worst possible
choice, since the lattice is identical at every candidate. Suppressing it should have
exposed the aperiodic content. Instead accuracy **collapses to 0%** above the pitch
(12.5 / 7.5 / 0.0 / 0.0 / 0.0% at σ_lo = 4/6/8/12/16). The lattice is what lets the
correlation lock onto a position at all; without it there is nothing to register against.
Registration and disambiguation want opposite filters, and the shipped (2.0, 8.0) is the
best of twelve configurations tested — accidentally near-optimal.

The last one is the informative negative. Trimming degrades accuracy smoothly as
the discarded fraction rises (35.0 → 33.3 → 31.7 → 30.0% at 0/10/25/40%). If
damage were **localized**, dropping the worst-agreeing pixels would rescue the
true site; that it never does means the disagreement is spread **uniformly**
across the template — the signature of independent acquisition noise, not
occlusion. This is positive corroboration of the mechanism in §1b, arrived at
from the opposite direction.

The substantive finding behind the 22.5% bucket is that the truth's median
percentile on the surface is **80.2%** among failures: a fifth of the entire
surface outscores it. The defect is the true site's **score**, not its rank —
which is why every re-ranking attack was bound to fail.

Within the extension space the addendum allows — Phase 1's periodic-aware ZNCC
family, searched over the disclosed ranges — **the localization block is capped
near 13/40 and the cap is a property of the evidence, not of the estimator.**

## 2. What worked: DoG band-pass, confirmed at n=280

`results/phase2_experiments/exp01/summary.json`, identical pose sweep and
selection rule, only the ZNCC input differing:

| arm | ≤1 px | rate | 95% CI | mean credit | pts/40 | median s/pair |
|---|---:|---:|---|---:|---:|---:|
| raw | 52/280 | 18.6% | [14.5, 23.5] | 0.189 | 7.5 | 3.49 |
| **DoG** | **83/280** | **29.6%** | [24.6, 35.2] | **0.334** | **13.3** | 3.54 |

Paired: fixed 49, broke 18, net **+31**, McNemar **p = 0.0002**. Gain
concentrates where Set B's 0.55 weight sits — it is a low-frequency
charging-drift suppression effect, not generic contrast enhancement
(`robust_contrast`, an affine map, provably cannot move ZNCC and did not):

| severity | raw | DoG | Δ (pp) |
|---|---:|---:|---:|
| 0 | 26.4% | 35.8% | +9.4 |
| 1 | 27.6% | 28.9% | +1.3 |
| 2 | 15.9% | 26.1% | +10.1 |
| **3** | **7.3%** | **29.3%** | **+22.0** |

Severity 3 quadruples but remains the floor in absolute terms (29.3% vs
35.8% at severity 0) — the hardest regime is improved, not solved.

**Real example**, `results/phase2_failures/rank003_p2_val-000123_err466.7px.png`
(row in `exp01/per_pair.csv`): true site (238.29, 496.96), FinFET, severity 3.
Raw-input error 465.7 px, DoG-input error 497.7 px — one of the 18 pairs DoG
*breaks*, illustrating that the net +31 gain is not uniform. A hit on the same
file set, `rank012_...err0.7px.png`: true site recovered to 0.4–0.7 px, the
typical shape of a correct pick. Half of measured misses land near an
integer lattice translate of the truth, with scale and rotation still
recovered correctly — the matcher finds *a* period-consistent site, just not
the right one.

### Compact scoreboard

| block | metric | value | evidence |
|---|---|---:|---|
| Localization (n=280) | mean credit, DoG vs raw | 0.334 / 0.189 | `exp01/summary.json` |
| Pose \| localized (n=25) | scale / rotation credit | 1.000 / 0.956–1.000 | `PHASE2_FINDINGS.md` §1 |
| Rejection (lockbox n=51) | F1, AUC, CI90 | 0.9024, 0.8707, [0.838, 0.953] | `driftforge/models/presence_hgb.metadata.json` |
| Confidence (lockbox n=74) | AUC, vs raw ZNCC | 0.892 vs 0.760 | `results/phase2_experiments/exp18_correctness.json` |
| Runtime (n=60, uncontended) | median / P95 / max | 4.33 / 4.51 / 4.62 s | `results/phase2_experiments/uncontended_runtime.json` |

## 3. What we killed, and why (do not re-run)

| hypothesis | verdict | evidence |
|---|---|---|
| Finer pose grid (45→99→187 poses) | no effect | identical credit to 3 decimals |
| 5 anti-alias filters (Gaussian/box/area/cubic/Lanczos) | no effect | byte-identical site on 24/25 pairs |
| Candidate pool caps (400, 32) | harmful | each silently deleted the true site |
| Lattice-normalized DoG bandwidth | harmful | −16 pp; measured pitches (3.55–11.6 px) make the band 1.8–22.5× too narrow |
| Nearest-centre rule, applied to full surface | harmful | 0/60 pairs fixed at any margin; equivalence sets are singletons, no ties exist to arbitrate. Likely an implementation artifact — Phase 1 applies the rule inside a ranked pool, this test applied it to ~800k raw positions |
| Ranker retraining on Phase 2 corpus | no gain | −5.8 pp selection-given-pool (p=0.51), 8.5× cost |
| Lattice-sibling hard negatives | no gain | −3.8 pp vs incumbent |
| Bias/gain photometric compensation | harmful | AUC inverts to 0.429 at severity 3 (ablation sweep, this session) |

## 4. Final shipped architecture

Phase 1's periodic-aware ZNCC pipeline, extended (not replaced) per the
addendum's allowed changes: dense full-resolution pose sweep (45 poses) over
the disclosed scale/rotation ranges, DoG-band-passed ZNCC as the matching
signal, refinement to convert coarse pose into subpixel/sub-degree credit,
and **two separate fitted models**: a 17-feature HGB presence model for
`found` (F1 0.90), and a 20-feature logistic correctness model for `score`
(AUC 0.892 vs 0.760 for raw ZNCC). They answer different questions -- *is the
reference here* versus *is the coordinate I am reporting correct* -- and a
present-but-mislocalized pair must score low on the second while being 1 on
the first, so one probability cannot serve both. See `README.md` "Phase 1 to
Phase 2: what changed" for the stage-by-stage table.

## 5. Honest limitations

- **Selection is the only bottleneck that matters**: pool recall 90–95%,
  selection 15–43%. Further harvest/pooling engineering will not move the
  score; new discriminating evidence is required and does not yet exist.
- **FinFET is not weaker than DRAM** under Phase 2 conditions — the Phase 1
  README claim does not reproduce (25.0% vs 34.6% after band-pass, overlapping
  intervals on smaller per-architecture samples). This document supersedes
  that claim.
- **Rejection lockbox is small** (51 and 55 scenes across two slices). F1 =
  0.9024 point estimate, but the 90% CI lower tail is **0.838**, so the F1 ≥
  0.90 bonus gate is at the boundary, not cleared.
- **Confidence AUC (0.892, n=74) beats raw ZNCC (0.760)** on the same lockbox,
  but n=74 is a small sample; the run is checkpointed to
  `results/phase2_experiments/exp18_correctness.json`.
- **Runtime is fine for the shipped path** — median **4.33 s**, P95 4.51 s,
  max 4.62 s over 60 pairs measured uncontended with threads pinned to 4, so
  4.3× headroom on the 20 s cap and 0/60 over 5 s. The figure rose from 3.54 s
  when the correctness model was added. Forcing the budget down confirms the
  `Deadline` path sheds refinement and presence rather than overrunning, and
  the output contract holds at every budget tested down to 0.5 s. Separately,
  the *heavier* recall-diagnostic
  harvester used to measure 90–95% pool recall is not what ships — at that
  configuration 60/316 pairs (19%) exceeded the 20 s cap
  (`results/gt_pose_ceiling.json`). Pool recall has not been separately
  re-measured on the exact shipped, faster configuration.
- **Pose credit (n=25) is a small sample**; treat 1.000/0.956 as encouraging,
  not certain.
- All scored evidence is synthetic and self-generated; no organizer sample
  pairs were available at analysis time.

## 6. Next steps, ranked by evidence

1. **Do not spend further effort on candidate ranking.** §1c bounds every such
   rule at ~35% even with perfect pose. New *features* over the same surfaces
   are the same class of change and inherit the same bound. The only escape is a
   different similarity measure, and a materially different method is a
   no-appeal disqualification — so this is closed by rule, not just by budget.
2. **Re-test the nearest-centre tie rule inside the ranked candidate pool**
   (not the full correlation surface) — the harmful result above is suspected
   to be a scope bug, not evidence against the rule itself.
3. **Grow the rejection/confidence lockbox** past 51–74 scenes before trusting
   the F1 ≥ 0.90 bonus gate or the 0.892 AUC point estimate.
4. **Persist the confidence-AUC and bias/gain ablation runs** to `results/`
   so every number in this document has a durable file, not a session log.
5. **Confirm the organizer-baseline gap**: a sponsor baseline reportedly
   scores 0.50 on `p2_val` against our 0.28–0.33; the control run was started
   twice and stalled both times and needs one clean execution.

## 7. Reproducibility

Python 3.11, CPU-only, no network at run time (`scripts/verify_offline.py`,
audit-hook based). Weights ship inside the package
(`driftforge/models/presence_hgb.pkl`, `correctness_lr.pkl`, `hgb_r2.joblib`,
SHA256-pinned in their metadata files) and are loaded script-relative.
`scripts/build_submission.py` builds the zip from an explicit allow-list, fails
if any required file is absent, then extracts the finished archive to a scratch
directory and runs the documented entry point inside it. Validation is
deterministic: 12 gates in
`scripts/validate_phase2.py`, byte-identical regeneration, and
`results/phase2_experiments/exp01/per_pair.csv` reproduces the DoG-vs-raw
comparison in Section 2 row-for-row.
