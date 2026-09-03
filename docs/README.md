# Documentation

Read in this order. Each file has one job.

| # | File | Job | Audience |
|---|---|---|---|
| 1 | [README.md](../README.md) | What this is, the one scored command, official-sample result | Jury, first open |
| 2 | [HOW_TO_RUN.md](HOW_TO_RUN.md) | Exact CLI, CSV contract, how to read a row | Jury executing the zip |
| 3 | [JOURNEY.md](JOURNEY.md) | How it was built: what worked, what failed, what we retracted | Jury judging engineering |
| 4 | [V1_VS_V2.md](V1_VS_V2.md) | Architecture, migration, every chart | Jury scoring method |
| 5 | [failure_analysis.md](failure_analysis.md) / [PDF](failure_analysis.pdf) | Limits, leftover points, honest negatives | Jury scoring the 10-pt write-up |
| 6 | [REFERENCES.md](REFERENCES.md) | Parameter-to-source citations | Jury scoring generator / citations |

No figure in these pages is drawn by hand. Two commands regenerate all of them:

```bash
python scripts/build_v2_visuals.py            # the diagrams and scorecards
python scripts/build_inference_gallery.py     # runs the solver, plots what it did
```

The second one is the honest one: it calls the same `register.process` the scored
entry point calls, over a stratified slice of `data/phase2`, and plots the
result against ground truth pair by pair — misses included. It writes
`results/phase2_experiments/inference_gallery.json`, and
`--reuse` redraws from that record without solving again.

Two number families appear in these pages and must not be mixed:

- **Official sample** — 20 organizer pairs, scored with the published rubric. This is the only organizer-data result in the repository.
- **Internal stress** — our generator, independent reference/search acquisitions. Harder than the scored task. Use it to understand failure modes, not to forecast the blind 200.

The dated engineering notebook `PHASE2_FINDINGS.md` is **not** a scored document;
[JOURNEY.md](JOURNEY.md) is the distilled, checked version of it.

Two videos exist, neither needed to score the entry:

- [Terminal demo, 0:26](demo/latticerank_demo.mp4) — a real `register.py` run on
  `examples/pairs.csv`. Tracked here, and shipped inside the zip.
- A 3:56 explainer built from the slides in `demo/source_slides/` and the
  narration in [demo/voiceover.md](demo/voiceover.md). The MP4 is **not** tracked
  — it is 9.5 MB that `python scripts/render_explainer_video.py` reproduces, with
  `python scripts/check_voiceover_timing.py` checking the script against the
  slide clock and `python scripts/build_narrated_video.py` laying a recorded
  voiceover onto it.
