#!/usr/bin/env python3
"""Render the 4-minute LatticeRank Phase 2 explainer.

Documentation helper, not a runtime dependency. Sources, in order of trust:

* ``docs/demo/source_slides/deck_*.jpg`` -- the LatticeRank presentation slides,
  shown whole and then cropped into the region under discussion.
* ``docs/images/v2_*.svg`` -- generated evidence charts, rasterised through Edge
  so the detail beats stay sharp instead of upscaling a 1024 px slide.
* ``docs/demo/latticerank_demo.mp4`` -- the recorded smoke run, spliced in whole.
* ``docs/demo/predictions.csv`` -- the rows that run actually wrote.

Runtime is pinned: :data:`TARGET_SECONDS` frames are emitted no matter how the
segment table below is edited, so the deliverable cannot drift out of the
3:50-4:00 window. Segment lengths are weights, not seconds.

    python scripts/render_explainer_video.py
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
DECK = OUT / "source_slides"
CACHE = OUT / "slides"
IMAGES = ROOT / "docs" / "images"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

W, H = 1280, 720
BAR = 46
CONTENT_H = H - BAR
FPS = 24
TARGET_SECONDS = 236  # 3:56, inside the required 3:50-4:00 window
XFADE = 10

CREAM = (247, 243, 236)
NAVY = (16, 32, 58)
INK = (22, 34, 54)
ORANGE = (224, 122, 47)
GREEN = (46, 125, 74)
DIM = (150, 164, 186)
SLATE = (96, 112, 134)
TERMINAL_BG = (13, 17, 23)

FONT = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_B = Path(r"C:\Windows\Fonts\segoeuib.ttf")
MONO = Path(r"C:\Windows\Fonts\consola.ttf")
MONO_B = Path(r"C:\Windows\Fonts\consolab.ttf")


def fnt(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        path = MONO_B if bold and MONO_B.exists() else MONO
    else:
        path = FONT_B if bold and FONT_B.exists() else FONT
    return ImageFont.truetype(str(path), size)


# --------------------------------------------------------------------------- #
# image plumbing
# --------------------------------------------------------------------------- #

def load(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def crop_whitespace(img: Image.Image, thresh: int = 248, pad: int = 16) -> Image.Image:
    arr = np.asarray(img)
    mask = arr.min(axis=2) < thresh
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    box = (
        max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
        min(img.width, int(xs.max()) + 1 + pad), min(img.height, int(ys.max()) + 1 + pad),
    )
    if box[2] - box[0] < 120 or box[3] - box[1] < 120:
        return img
    return img.crop(box)


def rasterize_svg(name: str) -> Image.Image:
    """Screenshot an SVG through Edge at 2x the video width, then trim margins."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{Path(name).stem}.png"
    if not dest.exists():
        subprocess.run(
            [str(EDGE), "--headless", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=2", "--window-size=1280,760",
             f"--screenshot={dest}", (IMAGES / name).resolve().as_uri()],
            check=True, capture_output=True,
        )
        crop_whitespace(load(dest)).save(dest)
    return load(dest)


def frame_for(src: Image.Image, *, bg: tuple[int, int, int] | None = None,
              sharpen: bool = False, heading: str | None = None,
              note: str | None = None) -> Image.Image:
    """Fit ``src`` into the content area, letterboxed in its own background colour.

    A wide crop -- a single band lifted out of a slide -- leaves most of the frame
    empty. When that happens and a heading is supplied, the band is pushed down
    and titled, so the frame reads as a deliberate slide instead of a stray strip.
    """
    if bg is None:
        bg = src.getpixel((2, 2))
        if not isinstance(bg, tuple):
            bg = CREAM
    top, foot = 0, 0
    if heading is not None:
        scale = min(W / src.width, CONTENT_H / src.height)
        if CONTENT_H - src.height * scale > 150:
            top, foot = 156, 96 if note else 24
    avail = CONTENT_H - top - foot
    scale = min(W / src.width, avail / src.height)
    size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
    body = src.resize(size, Image.Resampling.LANCZOS)
    if sharpen and scale > 1.05:
        body = body.filter(ImageFilter.UnsharpMask(radius=1.5, percent=115, threshold=3))
    canvas = Image.new("RGB", (W, H), bg)
    canvas.paste(body, ((W - size[0]) // 2, top + (avail - size[1]) // 2))
    draw = ImageDraw.Draw(canvas)
    if top:
        draw.text((72, 66), heading, font=fnt(42, bold=True), fill=INK)
        draw.rectangle((72, 128, 200, 132), fill=ORANGE)
    if top and note:
        draw.text((72, CONTENT_H - 66), note, font=fnt(24), fill=SLATE)
    draw.rectangle((0, CONTENT_H, W, H), fill=NAVY)
    return canvas


def sub(src: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    x0, y0, x1, y1 = box
    return src.crop((round(x0 * src.width), round(y0 * src.height),
                     round(x1 * src.width), round(y1 * src.height)))


# --------------------------------------------------------------------------- #
# hand-built cards, in the deck's cream-and-navy palette
# --------------------------------------------------------------------------- #

def card() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), CREAM)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, CONTENT_H, W, H), fill=NAVY)
    return img, draw


def title_card() -> Image.Image:
    img, draw = card()
    draw.rectangle((0, 0, 12, CONTENT_H), fill=NAVY)
    draw.text((72, 150), "SEMICON INDIA 2026  ·  DRIFT-SENSE PHASE 2",
              font=fnt(18, bold=True), fill=ORANGE)
    draw.text((72, 196), "LatticeRank", font=fnt(88, bold=True), fill=INK)
    draw.text((72, 312), "Periodic-aware registration for semiconductor inspection",
              font=fnt(30), fill=SLATE)
    draw.rounded_rectangle((72, 380, 1000, 444), 10, fill=(236, 231, 222))
    draw.text((96, 412), "python register.py --input pairs.csv --output predictions.csv",
              font=fnt(24, mono=True), fill=INK, anchor="lm")
    draw.text((72, 494), "The problem  ·  V1 to V2  ·  how it works  ·  how to run  ·  results",
              font=fnt(22, bold=True), fill=NAVY)
    draw.text((72, 534), "Four minutes, in that order.", font=fnt(22), fill=SLATE)
    return img


def run_card() -> Image.Image:
    img, draw = card()
    draw.text((72, 56), "05  ·  HOW TO RUN", font=fnt(18, bold=True), fill=ORANGE)
    draw.text((72, 92), "One frozen command", font=fnt(52, bold=True), fill=INK)
    rows = [
        ("1", "Install", "python -m pip install -r requirements.txt"),
        ("2", "Score the blind set", "python register.py --input pairs.csv --output predictions.csv"),
        ("3", "Smoke test, files in the zip", "python register.py --input examples/pairs.csv --output predictions.csv"),
    ]
    y = 190
    for num, label, cmd in rows:
        draw.rounded_rectangle((72, y, 1208, y + 96), 12, fill=(238, 233, 224))
        draw.ellipse((96, y + 30, 132, y + 66), fill=NAVY)
        draw.text((114, y + 48), num, font=fnt(20, bold=True), fill=CREAM, anchor="mm")
        draw.text((152, y + 26), label, font=fnt(21, bold=True), fill=NAVY)
        draw.text((152, y + 58), cmd, font=fnt(20, mono=True), fill=INK)
        y += 112
    draw.text((72, y + 12),
              "Python 3.11  ·  4-core CPU  ·  no GPU  ·  no network  ·  weights already in the zip",
              font=fnt(21, bold=True), fill=NAVY)
    draw.text((72, y + 46), "Not a notebook, not an interactive prompt, no downloads at run time.",
              font=fnt(20), fill=SLATE)
    return img


def predictions_card() -> Image.Image:
    with (OUT / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    img, draw = card()
    draw.text((72, 56), "05  ·  HOW TO RUN", font=fnt(18, bold=True), fill=ORANGE)
    draw.text((72, 92), "What that run wrote", font=fnt(52, bold=True), fill=INK)
    cols = (96, 260, 420, 580, 730, 890, 1010)
    draw.rounded_rectangle((72, 190, 1208, 400), 12, fill=(238, 233, 224))
    for x, name in zip(cols, header):
        draw.text((x, 214), name, font=fnt(20, bold=True, mono=True), fill=NAVY)
    draw.line((96, 248, 1184, 248), fill=(206, 199, 188), width=2)
    y = 274
    for row in body:
        cells = [row[0]] + [f"{float(v):.2f}" for v in row[1:5]] + [row[5], f"{float(row[6]):.3f}"]
        for x, value in zip(cols, cells):
            draw.text((x, y), value, font=fnt(20, mono=True), fill=INK)
        y += 44
    draw.text((72, 434), "Two pairs, 6.4 s wall clock, both present, both localized.",
              font=fnt(22, bold=True), fill=NAVY)
    y = 476
    for line, colour in (
        ("found = 1  ·  reference present, so the pose columns carry a real answer", GREEN),
        ("score = P(this coordinate is the correct site), not a raw correlation", ORANGE),
        ("scale is the down-scaling factor in [8, 12]  ·  theta is degrees, CCW positive", SLATE),
    ):
        draw.ellipse((74, y + 8, 86, y + 20), fill=colour)
        draw.text((100, y), line, font=fnt(21), fill=INK)
        y += 36
    return img


def end_card() -> Image.Image:
    img, draw = card()
    draw.rectangle((0, 0, 12, CONTENT_H), fill=NAVY)
    draw.text((72, 140), "LATTICERANK  ·  PHASE 2", font=fnt(18, bold=True), fill=ORANGE)
    draw.text((72, 186), "Register. Decide.", font=fnt(72, bold=True), fill=INK)
    draw.text((72, 268), "Report trust.", font=fnt(72, bold=True), fill=INK)
    draw.text((72, 388), "79.14 / 85 on the official 20-pair sample",
              font=fnt(30, bold=True), fill=NAVY)
    draw.text((72, 430), "plus the RGB and rejection-F1 bonuses  ·  frozen solver, nothing tuned on it",
              font=fnt(22), fill=SLATE)
    draw.rounded_rectangle((72, 494, 1000, 552), 10, fill=(236, 231, 222))
    draw.text((96, 523), "python register.py --input pairs.csv --output predictions.csv",
              font=fnt(23, mono=True), fill=INK, anchor="lm")
    return img


# --------------------------------------------------------------------------- #
# segment table
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    chapter: str
    caption: str
    weight: float
    provider: Callable[[int], Image.Image]
    fixed_frames: int | None = None
    frames: int = field(default=0, init=False)


def static(image: Image.Image) -> Callable[[int], Image.Image]:
    return lambda _i: image


def demo_provider(path: Path) -> Callable[[int], Image.Image]:
    """Stream the recorded terminal run, letterboxed on its own background."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    stream = iter(reader)
    cache: dict[int, Image.Image] = {}
    state = {"last": None}

    def provider(i: int) -> Image.Image:
        if i in cache:
            return cache[i]
        try:
            raw = Image.fromarray(next(stream))
            state["last"] = frame_for(raw, bg=TERMINAL_BG)
        except StopIteration:
            pass
        cache.clear()
        cache[i] = state["last"]
        return state["last"]

    return provider


def build_segments() -> list[Segment]:
    problem = load(DECK / "deck_01_problem.jpg")
    hardest = load(DECK / "deck_02_hardest_place.jpg")
    v1 = load(DECK / "deck_03_v1_solved.jpg")
    v2 = load(DECK / "deck_04_v2_changed.jpg")
    onepage = load(DECK / "deck_05_v1_vs_v2.jpg")
    overview = load(DECK / "deck_06_overview.jpg")
    process = load(DECK / "deck_07_process.jpg")
    generator = load(DECK / "deck_08_generator.jpg")

    def whole(src: Image.Image) -> Image.Image:
        return frame_for(src)

    def zoom(src: Image.Image, box: tuple[float, float, float, float],
             heading: str | None = None, note: str | None = None) -> Image.Image:
        return frame_for(sub(src, box), sharpen=True, heading=heading, note=note)

    chart = {name: frame_for(rasterize_svg(f"{name}.svg")) for name in (
        "v2_pose_grid", "v2_presence_vs_score", "v2_output_contract",
        "v2_how_to_read", "v2_official_scorecard", "v2_vs_baseline",
    )}

    c1, c2 = "01 · The problem", "02 · V1 to V2"
    c3, c4 = "03 · How it works", "04 · Data and contract"
    c5, c6 = "05 · How to run", "06 · Results"

    return [
        Segment("LatticeRank · Phase 2", "A four-minute walkthrough of the scored entry",
                6, static(title_card())),
        Segment(c1, "One reference, one search image, hundreds of convincing copies",
                8, static(whole(problem))),
        Segment(c1, "Four candidates score almost the same. One of them is the site",
                8, static(zoom(problem, (0.33, 0.135, 1.00, 0.60)))),
        Segment(c1, "Pick the wrong copy and every downstream measurement moves with it",
                7, static(zoom(problem, (0.015, 0.60, 0.99, 0.93),
                               "What a wrong copy costs",
                               "One contract: x, y, theta, scale, found, score. One truth. One decision."))),
        Segment(c1, "Similarity is everywhere, so similarity is cheap",
                7, static(whole(hardest))),
        Segment(c1, "The failure is a whole lattice period, not a couple of soft pixels",
                6, static(zoom(hardest, (0.47, 0.615, 0.99, 0.86),
                               "Why this is hard",
                               "Choose the wrong cell and the reported point is a full period out, not a blur."))),
        Segment(c1, "Keep the candidates, then commit only when the evidence is strong",
                5, static(zoom(hardest, (0.02, 0.858, 0.99, 0.985),
                               "The approach in one line",
                               "Every stage that follows exists to make the last clause true."))),

        Segment(c2, "Phase 1 gave us scale, presence, and only x and y to return",
                8, static(whole(v1))),
        Segment(c2, "The V1 matcher still does the site selection in V2",
                6, static(zoom(v1, (0.47, 0.45, 0.97, 0.86)))),
        Segment(c2, "Phase 2 removes the training wheels", 8, static(whole(v2))),
        Segment(c2, "Zoom, rotation, and presence all become unknowns",
                6, static(zoom(v2, (0.02, 0.278, 0.99, 0.47),
                               "Three assumptions are gone",
                               "Scale, rotation, and presence become things to search for, not things given."))),
        Segment(c2, "What stays fixed, and the 200-pair set behind every number",
                6, static(zoom(v2, (0.02, 0.46, 0.99, 0.84)))),
        Segment(c2, "Same core idea, harder contract", 7, static(whole(onepage))),
        Segment(c2, "V2 must report a pose, reject absences, and fail out loud",
                5, static(zoom(onepage, (0.50, 0.19, 0.97, 0.752)))),

        Segment(c3, "One pipeline: seven stages from raw pair to a decided row",
                7, static(whole(overview))),
        Segment(c3, "Decode, band-pass, pose hypotheses, correlation, ranking, refine",
                8, static(zoom(overview, (0.02, 0.552, 0.68, 0.86),
                               "The seven stages",
                               "Stages 1-5 come from V1. Stage 3 grows to 45 hypotheses; 6-7 decide."))),
        Segment(c3, "What V2 added on top of the Phase 1 locator",
                6, static(zoom(overview, (0.68, 0.586, 1.00, 0.87)))),
        Segment(c3, "The same eight steps, with the images each stage actually sees",
                7, static(whole(process))),
        Segment(c3, "Input, safe decode, band-pass, then a 45-hypothesis pose grid",
                8, static(zoom(process, (0.29, 0.185, 1.00, 0.525),
                               "From a pair to 45 hypotheses",
                               "Decode safely, band-pass away drift and charging, then sweep 9 scales x 5 rotations."))),
        Segment(c3, "Correlation surfaces, refinement, presence, then the output row",
                8, static(zoom(process, (0.29, 0.525, 1.00, 0.875),
                               "From surfaces to a decided row",
                               "Refine the winner subpixel, ask whether it is present, then calibrate the score."))),
        Segment(c3, "Nine scales by five rotations, each cell a full-image correlation",
                7, static(chart["v2_pose_grid"])),
        Segment(c3, "found and score answer different questions, so they are separate",
                6, static(chart["v2_presence_vs_score"])),

        Segment(c4, "The generator is a controlled lab, not a picture maker",
                6, static(whole(generator))),
        Segment(c4, "Twenty audited sample pairs, sixteen present and four absent",
                6, static(zoom(generator, (0.015, 0.137, 0.72, 0.945)))),
        Segment(c4, "Exactly one row per pair; a missing row scores zero",
                7, static(chart["v2_output_contract"])),
        Segment(c4, "How a process engineer reads a row: presence first, then trust",
                6, static(chart["v2_how_to_read"])),

        Segment(c5, "Install, score, smoke: three commands, no notebook",
                8, static(run_card())),
        Segment(c5, "The recorded smoke run on the example pairs shipped in the zip",
                0, demo_provider(OUT / "latticerank_demo.mp4"), fixed_frames=631),
        Segment(c5, "Two rows, both present, both localized, 6.4 s wall clock",
                7, static(predictions_card())),

        Segment(c6, "Official 20-pair sample, frozen solver, published rubric",
                8, static(chart["v2_official_scorecard"])),
        Segment(c6, "Against the organizer's brute-force baseline on the same pairs",
                6, static(chart["v2_vs_baseline"])),
        Segment(c6, "Register. Decide. Report trust.", 5, static(end_card())),
    ]


def allocate(segments: list[Segment], total_frames: int) -> None:
    """Turn weights into frame counts so the sum is exactly ``total_frames``."""
    fixed = sum(s.fixed_frames or 0 for s in segments)
    flexible = [s for s in segments if s.fixed_frames is None]
    budget = total_frames - fixed
    if budget <= 0:
        raise SystemExit("fixed segments already exceed the target runtime")
    weight = sum(s.weight for s in flexible)
    for s in segments:
        if s.fixed_frames is not None:
            s.frames = s.fixed_frames
        else:
            s.frames = max(XFADE + 6, round(budget * s.weight / weight))
    drift = total_frames - sum(s.frames for s in segments)
    longest = max(flexible, key=lambda s: s.frames)
    longest.frames += drift
    if sum(s.frames for s in segments) != total_frames:
        raise SystemExit("frame allocation failed")


# --------------------------------------------------------------------------- #
# overlay and encode
# --------------------------------------------------------------------------- #

def timecode(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def overlay(base: Image.Image, segment: Segment, index: int, total: int) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, CONTENT_H, W, H), fill=NAVY)
    label = segment.chapter.upper()
    label_font = fnt(15, bold=True)
    draw.text((24, CONTENT_H + 22), label, font=label_font, fill=ORANGE, anchor="lm")
    edge = 24 + draw.textlength(label, font=label_font) + 18
    draw.line((edge, CONTENT_H + 12, edge, H - 12), fill=(46, 66, 98), width=1)
    draw.text((edge + 18, CONTENT_H + 22), segment.caption, font=fnt(18), fill=CREAM,
              anchor="lm")
    clock = f"{timecode(index / FPS)} / {timecode(total / FPS)}"
    draw.text((W - 22, CONTENT_H + 22), clock, font=fnt(15, mono=True), fill=DIM,
              anchor="rm")
    draw.rectangle((0, H - 4, W, H), fill=(30, 48, 76))
    draw.rectangle((0, H - 4, int(W * (index + 1) / total), H), fill=ORANGE)
    return img


def frames(segments: list[Segment], total: int) -> Iterator[Image.Image]:
    index = 0
    previous: Image.Image | None = None
    for segment in segments:
        for i in range(segment.frames):
            base = segment.provider(i)
            if previous is not None and i < XFADE:
                base = Image.blend(previous, base, (i + 1) / (XFADE + 1))
            yield overlay(base, segment, index, total)
            index += 1
        previous = segment.provider(segment.frames - 1)


def write_mp4(segments: list[Segment], total: int, path: Path) -> None:
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(path), fps=FPS, codec="libx264", quality=6,
        pixelformat="yuv420p", macro_block_size=1,
    )
    try:
        for i, frame in enumerate(frames(segments, total)):
            writer.append_data(np.asarray(frame))
            if i % 480 == 0:
                print(f"  {i}/{total} frames  ({i / FPS:5.1f}s)")
    finally:
        writer.close()


def write_poster(total: int, path: Path) -> None:
    """A still for the README: an animated GIF of a 4-minute deck is dead weight."""
    img = title_card()
    draw = ImageDraw.Draw(img)
    cx, cy, r = 1082, 236, 62
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=NAVY)
    draw.polygon([(cx - 18, cy - 28), (cx - 18, cy + 28), (cx + 28, cy)], fill=CREAM)
    draw.text((cx, cy + r + 30), f"{timecode(total / FPS)}  walkthrough",
              font=fnt(22, bold=True), fill=NAVY, anchor="mm")
    draw.rectangle((0, CONTENT_H, W, H), fill=NAVY)
    draw.text((24, CONTENT_H + 22), "LATTICERANK · PHASE 2", font=fnt(15, bold=True),
              fill=ORANGE, anchor="lm")
    draw.text((228, CONTENT_H + 22),
              "79.14 / 85 on the official 20-pair sample, frozen solver",
              font=fnt(18), fill=CREAM, anchor="lm")
    draw.rectangle((0, H - 4, W, H), fill=ORANGE)
    img.save(path)


def main() -> int:
    total = TARGET_SECONDS * FPS
    segments = build_segments()
    allocate(segments, total)
    print(f"{len(segments)} segments, {total} frames, {total / FPS:.1f}s "
          f"({timecode(total / FPS)})")

    CACHE.mkdir(parents=True, exist_ok=True)
    for i, segment in enumerate(segments):
        overlay(segment.provider(0), segment, 0, total).save(CACHE / f"preview_{i:02d}.png")

    mp4 = OUT / "latticerank_explainer.mp4"
    poster = OUT / "explainer_poster.png"
    write_mp4(segments, total, mp4)
    write_poster(total, poster)
    print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {poster} ({poster.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
