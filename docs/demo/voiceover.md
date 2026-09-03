# Explainer voiceover script

Narration for `latticerank_explainer.mp4` (3:56, 236.0 s, 32 segments).

This file is the source of truth. Each line is tagged with the segment it plays
over, so the pacing can be checked instead of guessed:

```bash
python scripts/check_voiceover_timing.py
```

That reads the segment table straight out of `scripts/render_explainer_video.py`,
counts words, prints the words-per-minute each line demands, flags anything that
would have to be rushed, and writes the paste-ready chunks to
`docs/demo/voiceover_sarvam.txt`.

## Why the word counts look small

A 6-second slide is not 6 seconds of speech. Read at a comfortable 140 wpm, it
holds about 13 words, and the last word should land before the crossfade rather
than on top of it. Every line below is written to sit between roughly 100 and
145 wpm: under 100 the slide feels dead, over 150 the numbers stop landing.

Total: about 490 words across 236 seconds, which averages near 125 wpm. That is
deliberately below 140 — the gaps between lines are where a slide gets read.

## Sarvam settings

| Setting | Value | Why |
|---|---|---|
| Model | `bulbul:v3` | 2,500 characters per request, best quality. `bulbul:v2` also works, at 1,500. |
| Language | `en-IN` | The script is English with technical terms. |
| Speaker | `shubh`, `amelia` or `sophia` (v3); `abhilash` or `anushka` (v2) | Pick one and use it for every chunk, or the joins will be audible. |
| `pace` | `1.0`, then adjust | If a chunk runs long against its slides, try `1.05`–`1.1`. Do not go past `1.2`; the digits start slurring. |
| `speech_sample_rate` | `24000` | Matches the video's delivery; 44.1 kHz is available on REST if you want it. |
| `temperature` (v3) | `0.5` | Lower than default, so takes are consistent between chunks. |

Numbers are spelled out in words below ("seventy-nine point one four", not
"79.14"). Text-to-speech front-ends disagree about decimals, versions and
hyphens, and this removes the argument.

## Script

Chunk boundaries fall on chapter boundaries, so each chunk is one audio file.

```voiceover
[00] LatticeRank. Four minutes: how we register a wafer image, and when we refuse.
[01] One reference image. One search image. The structure repeats, so hundreds of places look like the answer.
[02] Four candidates here. Their correlation scores are almost identical, but only one is the true site.
[03] Choose wrong and the defect location, the overlay number, and the tool decision move with it.
[04] In a lattice, similarity is cheap. What matters is the evidence that breaks the repetition.
[05] Wrong cell, and the error is a whole period. Not a blur.
[06] So: keep every plausible candidate, and commit only on evidence.
[07] Phase one handed us the scale, promised the reference was present, and asked only for x and y.
[08] That version one matcher still picks the site. Phase two extends it.
[09] Phase two takes the training wheels off. Zoom, rotation and presence are all unknown now.
[10] Zoom lands anywhere from eight to twelve times. Rotation, within five degrees.
[11] One pair in five contains no true instance. Two hundred pairs, four sets.
[12] So the output row grows: x, y, theta, scale, found, and a calibrated score.
[13] Report a pose, reject absences, and fail out loud.
[14] One pipeline. Seven stages carry a raw pair through to a single decided row.
[15] Decode safely. Band-pass out the drift and charging. Sweep poses, correlate, rank, refine, then decide.
[16] Version two's new work is the last two stages: presence and confidence.
[17] Here are the same stages again, this time with the image each one actually sees.
[18] Decode, band-pass, then nine scales crossed with five rotations. That is forty-five full-image correlations per pair.
[19] The best peak is refined to subpixel. Then: is it present, and how far do we trust it?
[20] Forty-five hypotheses on a grid. Hard-coding the disclosed bounds is allowed, so we do.
[21] Found and score are different questions. A present pair can be wrong.
[22] Our generator is a controlled lab: known poses, audited labels, deliberate damage.
[23] Twenty audited sample pairs: sixteen present, four with no true instance.
[24] Exactly one row per pair. A missing row scores zero, so failures still write one.
[25] Read found first, then score. One triggers a re-scan; the other decides action.
[26] Install, then one command. Python three eleven, four CPU cores, no GPU, no network, weights in the zip.
[27] This is the real thing, not a mock-up. The smoke input ships inside the zip: two pairs, DRAM and FinFET. It loads the models, sweeps the pose grid for each pair, refines the winner, then writes the CSV. Two pairs, six point four seconds, and both were found.
[28] Both rows present, both localized. And score is a probability, not a raw correlation.
[29] On the official twenty-pair sample: seventy-nine point one four out of eighty-five. Frozen solver, nothing tuned.
[30] The published baseline averages zero point eight. We average zero point nine seven.
[31] Register, decide, and report how much to trust it.
```

## Getting the audio onto the video

Generate one file per chunk, in order, then join and mux. `imageio-ffmpeg`
already ships an ffmpeg binary, so nothing new is needed:

```bash
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

Using that path as `FFMPEG`, with `chunks.txt` listing `file 'chunk_1.wav'` and
so on:

```bash
FFMPEG -f concat -safe 0 -i chunks.txt -c:a pcm_s16le voice.wav
FFMPEG -i docs/demo/latticerank_explainer.mp4 -i voice.wav \
       -c:v copy -c:a aac -b:a 128k -shortest docs/demo/latticerank_explainer_vo.mp4
```

Check the join: `voice.wav` should land near 236 seconds. Shorter is fine, the
tail is just quiet. If it overruns, re-generate with `pace` at `1.05` rather than
cutting words, because the word counts are already sized to the slides.

For frame-accurate narration, generate one clip per `[NN]` line instead of per
chunk and pad each to its segment length before concatenating. Thirty-two
requests instead of six, and every sentence lands on its own slide.
