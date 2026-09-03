#!/usr/bin/env python3
"""Lay a single narration recording onto the explainer, line by line.

Documentation helper, not a runtime dependency.

The recording is one continuous read of ``docs/demo/voiceover.md`` -- about 170
words per minute -- while the video runs 3:56. Muxing it as-is would leave the
last 69 seconds silent and every line would drift further from its slide.

So the recording is cut at its own pauses and each line is pinned to the start of
the slide it was written for. The slack becomes what it was budgeted to be: quiet
time to read the slide. Nothing is re-rendered; the video file is untouched and
only its audio track is replaced.

Cut points come from aligning structure, not from arithmetic on the clock. Two
simpler methods were tried and both failed: splitting on each line's share of
elapsed time gives a 14-word line under a second, because the pauses between
sentences are uneven; splitting on each line's share of *voiced* time cuts
mid-word, because a loudness floor high enough to find pauses also discards
quiet syllables, and how many it discards varies by sentence.

What is reliable is that the reader pauses where the script has a full stop,
colon or semicolon. So the script's clause breaks and the pauses ffmpeg found
are aligned as two sequences -- Needleman-Wunsch, with a penalty for a break the
reader ran through and for a pause the script does not explain -- and each line
ends at the pause its last clause break matched. Every cut then lands inside real
silence. Counts are close (a few dozen of each), so the alignment is nearly
diagonal and a local mistake costs one clause, not the rest of the video.

    python scripts/build_narrated_video.py --audio path/to/narration.wav
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
VIDEO = OUT / "latticerank_explainer.mp4"
SCRIPT = OUT / "voiceover.md"

LEAD_IN = 0.30      # seconds after a slide appears before its line starts
TAIL_GUARD = 0.12   # keep a line off the crossfade into the next slide
FADE = 0.015        # seconds of fade at each clip edge, to kill clicks
LINE = re.compile(r"^\[(\d+)\]\s+(.+?)\s*$")


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def load_segments() -> tuple[list, int]:
    spec = importlib.util.spec_from_file_location(
        "explainer", ROOT / "scripts" / "render_explainer_video.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    segments = module.build_segments()
    module.allocate(segments, module.TARGET_SECONDS * module.FPS)
    return segments, module.FPS


def load_lines() -> dict[int, str]:
    lines = {}
    for raw in SCRIPT.read_text(encoding="utf-8").splitlines():
        match = LINE.match(raw)
        if match:
            lines[int(match.group(1))] = match.group(2)
    return lines


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit("expected 16-bit PCM WAV")
        rate, channels = handle.getframerate(), handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def write_wav(path: Path, data: np.ndarray, rate: int) -> None:
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def detect_pauses(path: Path, noise: str, hold: float) -> list[tuple[float, float]]:
    proc = subprocess.run(
        [ffmpeg(), "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"silencedetect=noise={noise}:d={hold}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(v) for v in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(v) for v in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return list(zip(starts, ends))[:len(ends)]


def voiced_clock(voice: np.ndarray, rate: int, floor_db: float,
                 hop: float = 0.010) -> tuple[np.ndarray, float]:
    """Cumulative seconds of speech, sampled every ``hop`` seconds."""
    step = max(1, int(hop * rate))
    usable = len(voice) - len(voice) % step
    frames = voice[:usable].reshape(-1, step)
    rms = np.sqrt(np.maximum((frames ** 2).mean(axis=1), 1e-12))
    voiced = rms > 10 ** (floor_db / 20)
    return np.cumsum(voiced) * hop, hop


CLAUSE = re.compile(r"[.?!:;]+(?:\s+|$)")


def clause_breaks(lines: dict[int, str]) -> list[tuple[int, int, bool]]:
    """Every place the reader is likely to pause, as (line, words so far, line end)."""
    breaks: list[tuple[int, int, bool]] = []
    spoken = 0
    for i in range(len(lines)):
        pieces = [p for p in CLAUSE.split(lines[i]) if p.strip()]
        for k, piece in enumerate(pieces):
            spoken += len(piece.split())
            breaks.append((i, spoken, k == len(pieces) - 1))
    return breaks


def needleman_wunsch(expected: list[float], found: list[float],
                     skip: float) -> list[int | None]:
    """Monotonic alignment of expected break times to detected pause times."""
    m, n = len(expected), len(found)
    INF = float("inf")
    cost = [[INF] * (n + 1) for _ in range(m + 1)]
    move = [[""] * (n + 1) for _ in range(m + 1)]
    cost[0][0] = 0.0
    for i in range(1, m + 1):
        cost[i][0], move[i][0] = cost[i - 1][0] + skip, "e"
    for j in range(1, n + 1):
        cost[0][j], move[0][j] = cost[0][j - 1] + skip, "c"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            options = (
                (cost[i - 1][j - 1] + (expected[i - 1] - found[j - 1]) ** 2, "m"),
                (cost[i - 1][j] + skip, "e"),
                (cost[i][j - 1] + skip, "c"),
            )
            cost[i][j], move[i][j] = min(options)

    matches: list[int | None] = [None] * m
    i, j = m, n
    while i > 0 or j > 0:
        step = move[i][j]
        if step == "m":
            matches[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif step == "e":
            i -= 1
        else:
            j -= 1
    return matches


def align(lines: dict[int, str], words: list[int],
          pauses: list[tuple[float, float]], duration: float, skip: float,
          cum_voiced: np.ndarray, hop: float) -> list[float]:
    breaks = clause_breaks(lines)
    total_words = sum(words)
    total_voiced = float(cum_voiced[-1])
    mids = [(s + e) / 2 for s, e in pauses]

    # Both sequences are compared in seconds of *speech*, not seconds of tape.
    # Elapsed time drifts against the script because the pauses are uneven, and
    # that drift accumulates until correct matches look expensive. Voiced time
    # tracks words almost linearly, so the comparison stays honest end to end.
    expected = [total_voiced * spoken / total_words for _line, spoken, _end in breaks]
    found = [float(cum_voiced[min(int(m / hop), len(cum_voiced) - 1)]) for m in mids]

    matches = needleman_wunsch(expected, found, skip)
    paired = [(k, j) for k, j in enumerate(matches) if j is not None]
    error = [abs(expected[k] - found[j]) for k, j in paired]
    print(f"breaks {len(breaks)} clause breaks in the script, {len(mids)} pauses "
          f"heard, {len(paired)} matched")
    print(f"       match error {sum(error) / len(error):.2f}s of speech average, "
          f"{max(error):.2f}s worst")

    # Cut only where a line's last clause break matched a real pause. Where the
    # reader ran two lines together there is no safe cut, so they stay one clip
    # and play from the earlier slide -- looser sync, but never a clipped word.
    ends = {i: k for k, (i, _spoken, is_end) in enumerate(breaks) if is_end}
    cuts: dict[int, float] = {}
    previous = 0.0
    for i in range(len(words) - 1):
        j = matches[ends[i]]
        if j is None:
            continue
        cut = mids[j]
        if cut <= previous + 0.25:
            continue
        cuts[i] = cut
        previous = cut
    print(f"cuts   {len(cuts)} of {len(words) - 1} line ends fall on a pause; "
          f"the rest stay joined to the next line")
    return cuts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path,
                        default=Path.home() / "Downloads" / "multi-speaker.wav")
    parser.add_argument("--output", type=Path,
                        default=OUT / "latticerank_explainer_narrated.mp4")
    parser.add_argument("--noise", default="-30dB", help="silencedetect threshold")
    parser.add_argument("--hold", type=float, default=0.20,
                        help="shortest pause treated as a possible cut, seconds")
    parser.add_argument("--floor-db", type=float, default=-40.0,
                        help="loudness floor that counts as speech, for the rate check")
    parser.add_argument("--skip-penalty", type=float, default=1.0,
                        help="alignment cost of an unmatched break or pause")
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"no audio at {args.audio}")

    segments, fps = load_segments()
    lines = load_lines()
    missing = [i for i in range(len(segments)) if i not in lines]
    if missing:
        raise SystemExit(f"no narration for segments {missing}")

    voice, rate = read_wav(args.audio)
    duration = len(voice) / rate
    words = [len(lines[i].split()) for i in range(len(segments))]
    total_words = sum(words)
    print(f"audio  {duration:.2f}s, {rate} Hz, {total_words} words "
          f"({total_words / duration * 60:.0f} wpm)")

    starts = np.cumsum([0] + [s.frames for s in segments])[:-1] / fps
    video_seconds = sum(s.frames for s in segments) / fps
    print(f"video  {video_seconds:.2f}s, {len(segments)} segments, "
          f"{video_seconds - duration:+.1f}s of read time to distribute")

    pauses = detect_pauses(args.audio, args.noise, args.hold)
    cum_voiced, hop = voiced_clock(voice, rate, args.floor_db)
    speech = float(cum_voiced[-1])
    print(f"speech {speech:.2f}s of it is voice "
          f"({total_words / speech * 60:.0f} wpm while speaking)")
    cuts = align(lines, words, pauses, duration, args.skip_penalty,
                 cum_voiced, hop)

    groups, first, t0 = [], 0, 0.0
    for i in range(len(segments)):
        if i in cuts or i == len(segments) - 1:
            t1 = cuts.get(i, duration)
            groups.append((first, i, t0, t1))
            first, t0 = i + 1, t1
    print(f"clips  {len(groups)} clips for {len(segments)} slides\n")

    track = np.zeros(int(round(video_seconds * rate)), dtype=np.float32)
    fade = max(1, int(FADE * rate))
    window = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    def voiced_between(a: float, b: float) -> float:
        lo = min(int(a / hop), len(cum_voiced) - 1)
        hi = min(int(b / hop), len(cum_voiced) - 1)
        return float(cum_voiced[hi] - cum_voiced[lo])

    reference = total_words / speech * 60
    print(f"{'lines':>7} {'at':>7} {'window':>7} {'clip':>6} {'wpm':>5}  fit")
    overflow, suspect = [], []
    for a, b, t0, t1 in groups:
        held = starts[b] + segments[b].frames / fps - starts[a]
        clip = voice[int(t0 * rate):int(t1 * rate)].copy()
        if len(clip) > 2 * fade:
            clip[:fade] *= window
            clip[-fade:] *= window[::-1]
        clip_seconds = len(clip) / rate
        room = held - LEAD_IN - TAIL_GUARD
        offset = LEAD_IN if clip_seconds <= room else max(0.05, held - TAIL_GUARD - clip_seconds)
        at = int(round((starts[a] + offset) * rate))
        end = min(len(track), at + len(clip))
        track[at:end] += clip[:end - at]

        label = f"{a:02d}" if a == b else f"{a:02d}-{b:02d}"
        spill = clip_seconds - room
        note = "ok" if spill <= 0 else f"bleeds {spill:.2f}s"
        if spill > 0:
            overflow.append((label, spill))
        # Words per voiced second is the check on the split: it should sit close
        # to the whole recording's rate. Far off means this clip holds the wrong
        # sentences, however well it happens to fit the slides.
        spoken = max(voiced_between(t0, t1), 0.05)
        said = sum(words[a:b + 1])
        rate_i = said / spoken * 60
        if not 0.8 * reference <= rate_i <= 1.25 * reference:
            note += f"  ({rate_i:.0f} wpm spoken, expected ~{reference:.0f})"
            suspect.append((label, rate_i))
        print(f"{label:>7} {starts[a]:7.2f} {held:7.2f} {clip_seconds:6.2f} "
              f"{rate_i:5.0f}  {note}")

    wav = OUT / "explainer_narration_timed.wav"
    write_wav(wav, track, rate)
    print(f"\nwrote {wav.relative_to(ROOT)} ({len(track) / rate:.2f}s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(VIDEO), "-i", str(wav),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest",
         str(args.output)],
        check=True,
    )
    print(f"wrote {args.output.relative_to(ROOT)} "
          f"({args.output.stat().st_size / 1e6:.1f} MB)")
    for label, spill in overflow:
        print(f"  clip [{label}] runs {spill:.2f}s past its slides")
    for label, rate_i in suspect:
        print(f"  clip [{label}] split looks wrong: {rate_i:.0f} wpm spoken")
    if not suspect and not overflow:
        print("  every clip sits inside its slides and within 25% of the "
              "recording's speaking rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
