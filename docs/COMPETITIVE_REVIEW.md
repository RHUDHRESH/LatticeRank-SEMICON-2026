# Public-field review: what LatticeRank learned

This is an engineering review of public Drift-Sense repositories, not an
official leaderboard. Repositories use different generators, tolerances,
hardware, and split discipline, so their headline percentages are **not
directly rankable**. The useful comparison is method, evidence quality,
failure behavior, and judge reproducibility.

## Scope and audit boundary

On 19 August 2026, a name- and description-based GitHub search produced 208
candidate SEMICON / Drift-Sense repositories. A structural presence/absence
audit checked the judge-facing assets below. It did not execute every project
and did not treat a README claim as an independently reproduced result.

| Repository evidence checked | Points |
|---|---:|
| Clear task story and judge path | 10 |
| Exact two-images-to-one-coordinate inference contract | 12 |
| Dataset generator and ground truth | 10 |
| Machine-readable metrics and row evidence | 12 |
| Claim provenance and split identity | 10 |
| Method and failure analysis | 10 |
| Parameter-to-source traceability | 10 |
| Tests and CI | 10 |
| Packaging and dependency instructions | 10 |
| Visual walkthrough and short demonstration | 7 |
| License and repository metadata | 10 |
| **Total** | **101** |

LatticeRank was the only repository in that snapshot with all 101 checklist
points present. That means it was the most complete *submission artifact* in
the audit; it does not mean it had the best localization algorithm. Search is
not exhaustive, repositories can change after the snapshot, and the score is
not endorsed by the organizers.

## The strongest public methods inspected

| Project | Publicly reported evidence | Main idea | What transferred to LatticeRank |
|---|---|---|---|
| [Metralign](https://github.com/Achxy/metralign) | 1,398/1,400 synthetic pairs within 1 px; 239 ms mean; 97/100 on a separately implemented renderer | Reciprocal/phase geometry, axis-separated periodic differences, ambiguity-aware confidence | Separate x/y residuals and phase geometry were prototyped. Neither improved the locked LatticeRank slice, so they were rejected. Its artifact sealing and real-image claim boundaries are documentation models worth copying. |
| [Techtonics Drift-Sense](https://github.com/DK-A/Techtonics_Drift-Sense_Wafer_Inspection_PS2) | 118/120 within 5 px on its held-out generator; 672 ms reported | Multi-scale/angle NCC, geometry gate, fine search, phase refinement, Siamese reranking | LatticeRank had already tested the same families: 25-transform matching reached 30%, and its Siamese prototype reached 5% alone. More transform search selected more aliases on DriftForge. |
| [Ashish6312 Drift-Sense](https://github.com/Ashish6312/drift-sense) | 54.0% on 200 test pairs; 52.8% on a 1,000-pair robustness set; 72.4 ms mean on the larger run | Valid-region phase correlation plus periodic residual verification | This was the most informative negative-result record. LatticeRank reproduced the phase-only idea: 35% alone and 45% in its best fusion versus 50% current on the same locked 20 scenes. Rejected. |
| [FabIndica](https://github.com/GalacticVraj/FabIndica) | 0.681 correct-match rate within 1 px on 135 held-out pairs; 2.65 s/pair | Scale sweep, raw/residual modes, uncertainty and agreement | The “require independent evidence to agree” lesson directly supports LatticeRank's accepted residual/raw/mid-band consensus. The repository also reports the unidentifiable uniform-array stratum instead of hiding it. |
| [Maniteja8883 Drift-Sense](https://github.com/Maniteja8883/Drift-Sense) | 30-pair artifacts and a deterministic FFT-ZNCC geometry pipeline | Bounded transform search, subpixel refinement, explicit ambiguity diagnostics | Strong evaluator packaging, but an independent adversarial periodic check showed that its apparent accuracy was dominated by a near-centre synthetic placement regime. LatticeRank therefore did not adopt centre distance as ordinary ranking evidence. |

The reported numbers above belong to each project's own protocol. They must
not be read as one common benchmark table.

## What was implemented and what survived

The public review produced six concrete experiments on LatticeRank's locked
data:

1. Axis-separated lattice residuals: 40% alone; no fusion gain.
2. Valid-region phase-only correlation: 35% alone; best fusion 45% versus 50% current.
3. Image-derived scale/rotation correction: even the latent oracle failed to
   raise candidate recall or top-1 accuracy.
4. Scene-grouped positive-versus-hard-alias ranking: 30% versus 37.5% for the
   packaged ranker on the identical capped pool.
5. Eighteen robust image channels plus conservative map consensus: best 40%
   versus 48.75% production across 80 scenes.
6. A broader wallpaper gate: no threshold improved either locked 40-scene half.

None was promoted. Full measured summaries are preserved in the
[experiment ledger](../results/optimization_experiments.json).

One competitor-aligned idea had already survived a stricter gate: periodic
residual evidence combined with small raw and mid-band support. On the pinned
public reference-style generator it reached 112/120 during development and
30/30 on the untouched confirmation seed. That same method remains 39/80 on
the broader DriftForge stress set. The split is the scientific conclusion:
the method is strong on the sponsor-like public family and not yet universal.

## Where LatticeRank is better—and where it is not

LatticeRank's strongest competitive advantages are submission integrity:

- a one-command judge smoke test that also runs outside the repository;
- a complete, pinned environment and wheel-install test;
- final-coordinate-derived metrics with row-level evidence;
- content hashes, split labels, and source-revision provenance;
- a parameter-to-citation matrix;
- explicit negative experiments and visible catastrophic failures;
- separate external, internal, and randomized results with no pooling.

Its weakness is equally clear: internal remote-alias ranking remains 48.75%,
runtime has a long tail, and FinFET remains harder. Metralign is the stronger
algorithmic reference today; Ashish's project has the cleaner large-sample
robustness study; Techtonics reports much lower latency. LatticeRank should be
judged as the most auditable artifact in this review, not as a universal
accuracy winner.

## Next technically credible experiment

The common lesson is not “add another channel.” It is to train a nonlinear
groupwise ranker on *sets of aliases from the same scene*, while holding out an
entire generator family. That requires more scene-diverse training evidence
than is available in this release. Until that experiment is performed, the
honest release keeps the proven external consensus and reports the internal
ceiling unchanged.
