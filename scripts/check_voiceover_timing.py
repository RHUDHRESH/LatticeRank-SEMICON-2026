#!/usr/bin/env python3
"""Check the explainer voiceover against the video's real segment clock.

Documentation helper, not a runtime dependency.

A narration script written against guessed slide lengths drifts: one long
sentence over a 5-second slide pushes every later line onto the wrong visual.
So the segment table is read out of :mod:`render_explainer_video` -- the same
code that decides how many frames each slide gets -- and every tagged line in
``docs/demo/voiceover.md`` is measured against the slide it plays over.

    python scripts/check_voiceover_timing.py
    python scripts/check_voiceover_timing.py --wpm 150 --max-wpm 160

Exits non-zero if any line would have to be rushed, or if a tag is missing,
duplicated or points at a segment that does not exist.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "demo" / "voiceover.md"
CHUNKS = ROOT / "docs" / "demo" / "voiceover_sarvam.txt"

#: Sarvam's REST ceiling is 2500 characters for bulbul:v3 and 1500 for v2.
#: Chunk to the smaller one so the same file works on either model.
CHUNK_LIMIT = 1400

LINE = re.compile(r"^\[(\d+)\]\s+(.+?)\s*$")


def load_segments():
    spec = importlib.util.spec_from_file_location(
        "explainer", ROOT / "scripts" / "render_explainer_video.py")
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    segments = module.build_segments()
    module.allocate(segments, module.TARGET_SECONDS * module.FPS)
    return segments, module.FPS


def load_lines(path: Path) -> dict[int, str]:
    lines: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = LINE.match(raw)
        if not match:
            continue
        index = int(match.group(1))
        if index in lines:
            raise SystemExit(f"segment [{index:02d}] is tagged twice")
        lines[index] = match.group(2)
    if not lines:
        raise SystemExit(f"no [NN] tagged lines found in {path}")
    return lines


def clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{seconds % 60:05.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wpm", type=float, default=140.0,
                        help="comfortable narration speed, used for the word budget")
    parser.add_argument("--max-wpm", type=float, default=150.0,
                        help="fail above this; the numbers stop landing")
    parser.add_argument("--min-wpm", type=float, default=85.0,
                        help="warn below this; the slide feels dead")
    parser.add_argument("--pad", type=float, default=0.4,
                        help="seconds left at the end of a slide for the crossfade")
    args = parser.parse_args()

    segments, fps = load_segments()
    lines = load_lines(SCRIPT)

    missing = [i for i in range(len(segments)) if i not in lines]
    extra = [i for i in lines if i >= len(segments)]
    if missing:
        raise SystemExit(f"no narration for segments: {missing}")
    if extra:
        raise SystemExit(f"narration tagged for segments that do not exist: {extra}")

    print(f"{'#':>2} {'start':>8} {'secs':>6} {'words':>5} {'budget':>6} {'wpm':>5}  line")
    start, total_words, failures, slow = 0.0, 0, [], []
    for i, segment in enumerate(segments):
        seconds = segment.frames / fps
        text = lines[i]
        words = len(text.split())
        speak = max(0.5, seconds - args.pad)
        budget = int(speak * args.wpm / 60)
        wpm = words / speak * 60
        total_words += words
        flag = " "
        if wpm > args.max_wpm:
            flag, _ = "!", failures.append((i, wpm))
        elif wpm < args.min_wpm:
            flag, _ = "~", slow.append((i, wpm))
        head = text if len(text) <= 62 else text[:59] + "..."
        print(f"{i:>2} {clock(start):>8} {seconds:6.2f} {words:5d} {budget:6d} "
              f"{wpm:5.0f}{flag} {head}")
        start += seconds

    runtime = start
    print()
    print(f"{len(segments)} segments, {clock(runtime)} of video, {total_words} words, "
          f"{total_words / runtime * 60:.0f} wpm average")

    chunks = write_chunks(segments, lines)
    print(f"wrote {CHUNKS.relative_to(ROOT)}: {len(chunks)} chunks, "
          f"longest {max(len(c) for _, c in chunks)} chars (limit {CHUNK_LIMIT})")

    for i, wpm in slow:
        print(f"  slow  [{i:02d}] {wpm:.0f} wpm: room for another clause")
    for i, wpm in failures:
        print(f"  RUSHED [{i:02d}] {wpm:.0f} wpm exceeds {args.max_wpm:.0f}: cut words")
    if failures:
        return 1
    print("every line fits its slide")
    return 0


def write_chunks(segments, lines: dict[int, str]) -> list[tuple[str, str]]:
    """Group narration into per-chapter blocks that fit one Sarvam request."""
    chunks: list[tuple[str, str]] = []
    label, buffer = None, []

    def flush() -> None:
        if buffer:
            chunks.append((label, " ".join(buffer)))
            buffer.clear()

    for i, segment in enumerate(segments):
        chapter = segment.chapter
        candidate = " ".join(buffer + [lines[i]])
        if label is not None and (chapter != label or len(candidate) > CHUNK_LIMIT):
            flush()
        label = chapter
        buffer.append(lines[i])
    flush()

    # A one-line chapter (the title card) is not worth its own generation.
    merged: list[tuple[str, str]] = []
    for chapter, text in chunks:
        if merged and len(merged[-1][1]) < 200 \
                and len(merged[-1][1]) + len(text) + 1 <= CHUNK_LIMIT:
            merged[-1] = (merged[-1][0], f"{merged[-1][1]} {text}")
        else:
            merged.append((chapter, text))
    chunks = merged

    body = [
        "# Paste one block at a time into Sarvam. Same model, speaker and pace",
        "# for every block, or the joins will be audible. Generated by",
        "# scripts/check_voiceover_timing.py -- edit docs/demo/voiceover.md instead.",
        "",
    ]
    for n, (chapter, text) in enumerate(chunks, start=1):
        body.append(f"----- chunk {n} of {len(chunks)} · {chapter} · {len(text)} chars -----")
        body.append(text)
        body.append("")
    CHUNKS.write_text("\n".join(body), encoding="utf-8")
    return chunks


if __name__ == "__main__":
    raise SystemExit(main())
