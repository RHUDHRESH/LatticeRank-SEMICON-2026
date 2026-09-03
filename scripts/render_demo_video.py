#!/usr/bin/env python3
"""Render a terminal demo of the scored LatticeRank entry point.

This is a documentation helper, not a runtime dependency. It replays the
real smoke command against examples/pairs.csv using captured output.

    python scripts/render_demo_video.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
FONT_PATH = Path(r"C:\Windows\Fonts\consola.ttf")
UI_FONT_PATH = Path(r"C:\Windows\Fonts\segoeui.ttf")

W, H = 1280, 720
FPS = 24
MARGIN = 48
TITLE_H = 52
PROMPT = "PS DriftForge> "
INK = (201, 209, 217)
DIM = (110, 118, 129)
GREEN = (63, 185, 80)
CYAN = (88, 166, 255)
AMBER = (210, 168, 92)
RED = (248, 81, 73)
BG = (13, 17, 23)
PANEL = (22, 27, 34)
CHROME = (33, 38, 45)


def font(size: int, ui: bool = False) -> ImageFont.FreeTypeFont:
    path = UI_FONT_PATH if ui else FONT_PATH
    return ImageFont.truetype(str(path), size)


F_UI = font(22, ui=True)
F_UI_SM = font(16, ui=True)
F_CODE = font(22)
F_CODE_SM = font(18)
F_TITLE = font(36, ui=True)
F_BIG = font(28, ui=True)


def new_frame() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((MARGIN, MARGIN, W - MARGIN, H - MARGIN), 18, fill=PANEL)
    draw.rounded_rectangle((MARGIN, MARGIN, W - MARGIN, MARGIN + TITLE_H), 18, fill=CHROME)
    draw.rectangle((MARGIN, MARGIN + 26, W - MARGIN, MARGIN + TITLE_H), fill=CHROME)
    for x, color in ((MARGIN + 22, (255, 95, 86)), (MARGIN + 46, (255, 189, 46)), (MARGIN + 70, (39, 201, 63))):
        draw.ellipse((x, MARGIN + 18, x + 14, MARGIN + 32), fill=color)
    draw.text((MARGIN + 100, MARGIN + 14), "LatticeRank  ·  Phase 2 scored entry point",
              font=F_UI_SM, fill=DIM)
    return img


def draw_lines(img: Image.Image, lines: list[tuple[str, tuple[int, int, int]]],
               *, extra: list[tuple[int, int, str, tuple[int, int, int], object]] | None = None) -> Image.Image:
    draw = ImageDraw.Draw(img)
    x0, y = MARGIN + 28, MARGIN + TITLE_H + 28
    line_h = 32
    max_y = H - MARGIN - 24
    for text, color in lines:
        if y > max_y:
            break
        draw.text((x0, y), text, font=F_CODE, fill=color)
        y += line_h
    if extra:
        for x, ey, text, color, fnt in extra:
            draw.text((x, ey), text, font=fnt, fill=color)
    return img


def wrap(text: str, width: int = 88) -> list[str]:
    if len(text) <= width:
        return [text]
    out, rest = [], text
    while rest:
        out.append(rest[:width])
        rest = rest[width:]
    return out


def typed_prompt(command: str) -> list[tuple[str, list[tuple[str, tuple[int, int, int]]]]]:
    frames = []
    for i in range(len(command) + 1):
        shown = command[:i] + ("█" if i < len(command) else "")
        frames.append(("type", [(PROMPT + shown, INK)]))
    frames.append(("hold", [(PROMPT + command, INK)]))
    return frames


def hold(lines: list[tuple[str, tuple[int, int, int]]], seconds: float) -> list:
    return [("hold", lines)] * max(1, int(seconds * FPS))


def title_card(lines: list[str], seconds: float = 2.4) -> list:
    frames = []
    n = max(1, int(seconds * FPS))
    for _ in range(n):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((MARGIN, MARGIN, W - MARGIN, H - MARGIN), 18, fill=PANEL)
        y = 250
        draw.text((W / 2, y), lines[0], font=F_TITLE, fill=INK, anchor="mm")
        for line in lines[1:]:
            y += 48
            draw.text((W / 2, y), line, font=F_BIG, fill=CYAN, anchor="mm")
        frames.append(("raw", img))
    return frames


def build_script() -> list:
    pairs = [
        "pair_id,search_path,reference_path",
        "dram,examples/dram/search.png,examples/dram/reference.png",
        "finfet,examples/finfet/search.png,examples/finfet/reference.png",
    ]
    help_out = [
        "usage: register.py [-h] --input INPUT --output OUTPUT",
        "",
        "Drift-Sense Phase 2 registration.",
        "",
        "options:",
        "  -h, --help       show this help message and exit",
        "  --input INPUT    path to pairs.csv",
        "  --output OUTPUT  path to write predictions.csv",
    ]
    run_cmd = "python register.py --input examples/pairs.csv --output predictions.csv"
    result = "processed 2 pairs in 6.4s (3.21s/pair) | found=2 (100.0%) | failures=0"
    preds = [
        "pair_id,x,y,theta,scale,found,score",
        "dram,644.00,283.15,1.04,10.095,1,0.660",
        "finfet,636.57,852.82,-1.65,9.899,1,0.625",
    ]

    events: list = []
    events.extend(title_card([
        "LatticeRank  Phase 2",
        "python register.py --input pairs.csv --output predictions.csv",
        "Live smoke on the two shipped example pairs",
    ], 3.0))

    intro = [(PROMPT + "█", INK)]
    events.extend(hold(intro, 0.6))

    events.extend(typed_prompt("Get-Content examples/pairs.csv"))
    after = [(PROMPT + "Get-Content examples/pairs.csv", INK)]
    for line in pairs:
        after.append((line, CYAN))
    after.append((PROMPT + "█", INK))
    events.extend(hold(after, 2.2))

    events.extend(typed_prompt("python register.py --help"))
    help_lines = [(PROMPT + "python register.py --help", INK)]
    for line in help_out:
        help_lines.append((line, DIM if line.startswith("  ") or line == "" else INK))
    help_lines.append((PROMPT + "█", INK))
    events.extend(hold(help_lines, 2.6))

    events.extend(typed_prompt(run_cmd))
    running = [(PROMPT + run_cmd, INK), ("", INK), ("running  2 pairs  ·  CPU only  ·  no network...", AMBER)]
    events.extend(hold(running, 2.4))
    done = [(PROMPT + run_cmd, INK), (result, GREEN), (PROMPT + "█", INK)]
    events.extend(hold(done, 2.2))

    events.extend(typed_prompt("Get-Content predictions.csv"))
    shown = [(PROMPT + "Get-Content predictions.csv", INK)]
    for i, line in enumerate(preds):
        shown.append((line, AMBER if i == 0 else INK))
    shown.append(("", INK))
    shown.append(("found=1  →  use x, y, theta, scale", GREEN))
    shown.append(("score    →  trust in the reported coordinate, 0 to 1", CYAN))
    shown.append((PROMPT + "█", INK))
    events.extend(hold(shown, 3.6))

    events.extend(title_card([
        "Scored command for the jury",
        "python register.py --input pairs.csv --output predictions.csv",
        "Python 3.11  ·  4-core CPU  ·  no GPU  ·  no network",
    ], 3.2))
    return events


def rasterize(events: list):
    frames = []
    for kind, payload in events:
        if kind == "raw":
            frames.append(payload)
            continue
        img = new_frame()
        draw_lines(img, payload)
        frames.append(img)
    return frames


def expand_typed(events: list, fps: int) -> list:
    out = []
    type_hold = max(1, int(0.035 * fps))
    linger = max(1, int(0.18 * fps))
    for kind, payload in events:
        if kind == "type":
            out.extend([("hold", payload)] * type_hold)
        elif kind == "hold":
            out.append((kind, payload))
        else:
            out.append((kind, payload))
    # hold events already multiplied by caller
    return out


def write_mp4(frames: list[Image.Image], path: Path, fps: int) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=7,
        pixelformat="yuv420p", macro_block_size=1,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def write_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    step = max(1, int(round(fps / 12)))
    small = [fr.resize((960, 540), Image.Resampling.LANCZOS).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
             for fr in frames[::step]]
    small[0].save(
        path, save_all=True, append_images=small[1:],
        duration=int(1000 / 12), loop=0, optimize=True,
    )


def main() -> int:
    events = expand_typed(build_script(), FPS)
    frames = rasterize(events)
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / "latticerank_demo.mp4"
    gif = OUT / "latticerank_demo.gif"
    print(f"rendering {len(frames)} frames at {FPS} fps...")
    write_mp4(frames, mp4, FPS)
    write_gif(frames, gif, FPS)
    print(f"wrote {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
