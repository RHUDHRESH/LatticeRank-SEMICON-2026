from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import PRESETS, REFERENCE_FOV_NM, WORLD_FOV_NM, SceneSpec


#: Radius range of the distinctive dark-core/bright-rim fabrication anomaly. Also
#: used as the world-edge margin so the cluster renders whole wherever it lands.
SIGNATURE_RADIUS_NM = (72.0, 135.0)


@dataclass(frozen=True)
class DecoySpec:
    """A near-duplicate of the reference motif placed elsewhere in the world.

    Decoys are part of the latent layout: they are drawn inside a single
    ``render_scene`` call (layout-domain compositing), never pasted into a
    rendered array. Pitch and orientation always equal the host scene's; only
    contact occupancy, line widths and internal defect texture are perturbed,
    which is what makes a decoy a 60-90% structural near-duplicate rather than
    an unrelated field.
    """

    cx_nm: float
    cy_nm: float
    half_nm: float
    phase_x_nm: float
    phase_y_nm: float
    width_x_nm: float
    width_y_nm: float
    contact_x_nm: float
    contact_y_nm: float
    occupancy: float
    seed: int


def signature_placement(
    spec: SceneSpec,
) -> tuple[float, float, float, np.random.Generator]:
    """World position and radius of the distinctive fabrication anomaly.

    Seeded **only** by scene identity. Nothing here reads the target centre,
    the Reference crop, the label or any target-derived quantity - that
    independence is the whole point of E0, and exposing it as one function
    means the diagnostic verifies the code the renderer actually runs rather
    than a copy of it.

    The live generator is returned alongside so the caller continues the same
    stream for the brightness draws, preserving v1 draw order exactly.
    """
    rng = np.random.default_rng(spec.seed * 104_729 + 43)
    margin = SIGNATURE_RADIUS_NM[1]
    sx = float(rng.uniform(margin, WORLD_FOV_NM - margin))
    sy = float(rng.uniform(margin, WORLD_FOV_NM - margin))
    radius = float(rng.uniform(*SIGNATURE_RADIUS_NM))
    return sx, sy, radius, rng


def _uniform(rng: np.random.Generator, pair) -> float:
    return float(rng.uniform(pair[0], pair[1]))


def _nearest_equivalent_to_center(value: float, pitch: float) -> float:
    center = WORLD_FOV_NM / 2.0
    return value + round((center - value) / pitch) * pitch


def make_scene_spec(seed: int, architecture: str, profile: str, preset: str | None = None) -> SceneSpec:
    rng = np.random.default_rng(seed)
    if architecture not in {"dram", "finfet"}:
        raise ValueError("architecture must be 'dram' or 'finfet'")

    if preset is None:
        preset_name = f"{architecture}_{'dense' if rng.random() < 0.65 else 'open'}"
    else:
        # Phase 2 needs the same preset family for paired realizations (absent
        # references, decoys). The random draw is still consumed so the rest of
        # the spec keeps the exact draw order of the un-overridden call.
        preset_name = preset
        rng.random()
    if preset_name not in PRESETS:
        raise ValueError(f"unknown preset family {preset_name!r}")
    if not preset_name.startswith(architecture):
        raise ValueError(f"preset family {preset_name!r} does not belong to architecture {architecture!r}")
    p = PRESETS[preset_name]
    pitch_x = _uniform(rng, p["pitch_x"])
    pitch_y = _uniform(rng, p["pitch_y"])
    phase_x = float(rng.uniform(0, pitch_x))
    phase_y = float(rng.uniform(0, pitch_y))

    margin = REFERENCE_FOV_NM / 2.0 + 20.0
    cx = float(rng.uniform(margin, WORLD_FOV_NM - margin))
    cy = float(rng.uniform(margin, WORLD_FOV_NM - margin))
    mat_period = float(rng.uniform(2200, 3600))
    strip_width = float(rng.uniform(220, 480))
    zone_x = float(rng.uniform(0, mat_period))
    zone_y = float(rng.uniform(0, mat_period))

    if profile == "boundary":
        # Force the high-magnification crop to intersect a mat/periphery seam.
        kx = int(rng.integers(1, max(2, int(WORLD_FOV_NM // mat_period))))
        cx = float(np.clip(kx * mat_period - zone_x + rng.uniform(-120, 120), margin, WORLD_FOV_NM - margin))
    elif profile == "ambiguous":
        # A perfectly periodic reference has multiple valid candidates. Put the
        # phase-equivalent target closest to image centre, matching the rule.
        cx = _nearest_equivalent_to_center(cx, pitch_x)
        cy = _nearest_equivalent_to_center(cy, pitch_y)
        cx = float(np.clip(cx, margin, WORLD_FOV_NM - margin))
        cy = float(np.clip(cy, margin, WORLD_FOV_NM - margin))

    roughness = float(rng.uniform(0.4, 2.0))
    defect_density = 0.0 if profile == "ambiguous" else float(rng.uniform(0.4, 1.4))
    if profile in {"hard", "ood"}:
        roughness *= 1.5
        defect_density *= 1.4

    spec = SceneSpec(
        architecture=architecture,
        preset=preset_name,
        seed=seed,
        phase_x_nm=phase_x,
        phase_y_nm=phase_y,
        pitch_x_nm=pitch_x,
        pitch_y_nm=pitch_y,
        width_x_nm=_uniform(rng, p["width_x"]),
        width_y_nm=_uniform(rng, p["width_y"]),
        contact_x_nm=_uniform(rng, p["contact"]),
        contact_y_nm=_uniform(rng, p["contact"]),
        # The explicit ambiguity profile is an intentionally exact wallpaper.
        # All other profiles retain fabrication-like line variation.
        line_jitter_nm=0.0 if profile == "ambiguous" else float(rng.uniform(0.25, 1.8)),
        width_jitter_fraction=0.0 if profile == "ambiguous" else float(rng.uniform(0.02, 0.12)),
        roughness_nm=0.0 if profile == "ambiguous" else roughness,
        roughness_lambda_nm=float(rng.uniform(90, 420)),
        mat_period_nm=mat_period,
        strip_width_nm=strip_width,
        zone_offset_x_nm=zone_x,
        zone_offset_y_nm=zone_y,
        target_cx_nm=cx,
        target_cy_nm=cy,
        defect_density=defect_density,
        profile=profile,
    )
    return spec


def _hash_wave(index: np.ndarray, a: float, phase: float) -> np.ndarray:
    return np.sin(index * a + phase)


def _line_distance(
    coord: np.ndarray,
    axial: np.ndarray,
    phase: float,
    pitch: float,
    width: float,
    jitter_nm: float,
    width_jitter: float,
    roughness_nm: float,
    roughness_lambda: float,
    seed_phase: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.rint((coord - phase) / pitch).astype(np.int32)
    jitter = jitter_nm * _hash_wave(idx, 1.6180339, seed_phase)
    local_width = width * (1.0 + width_jitter * _hash_wave(idx, 2.4142136, seed_phase * 0.73 + 1.0))
    line_phase = idx * 0.754877666 + seed_phase
    rough = roughness_nm * (
        0.68 * np.sin((2 * np.pi / roughness_lambda) * axial + line_phase)
        + 0.32 * np.sin((2 * np.pi / (roughness_lambda * 0.43)) * axial + line_phase * 1.7)
    )
    center = phase + idx * pitch + jitter + rough
    return coord - center, idx, local_width


def render_scene(spec: SceneSpec, origin_x_nm: float, origin_y_nm: float, fov_nm: float, size_px: int) -> np.ndarray:
    """Render a shared physical scene at any FOV/resolution.

    The geometry is evaluated in world nanometres rather than cropped from a
    bitmap, so reference and search stay physically consistent without a
    10,000 x 10,000 floating-point allocation.
    """
    return render_scene_with_layers(spec, origin_x_nm, origin_y_nm, fov_nm, size_px)[0]


#: Layer ids emitted by :func:`render_scene_with_layers` for modality mapping.
LAYER_SUBSTRATE = 0
LAYER_VLINE = 1
LAYER_HLINE = 2
LAYER_CONTACT = 3
LAYER_ROUTE = 4

#: Per-layer reflectance triples for the RGB optical mode: substrate /
#: vertical metal lines / horizontal poly lines / contacts / routing strips.
#: The layer map is the material identity, so both architectures share one
#: table.
RGB_REFLECTANCE = np.array(
    [
        (0.10, 0.11, 0.13),
        (0.33, 0.58, 0.50),
        (0.52, 0.46, 0.30),
        (0.85, 0.82, 0.70),
        (0.15, 0.16, 0.19),
    ],
    dtype=np.float32,
)


def render_scene_with_layers(
    spec: SceneSpec,
    origin_x_nm: float,
    origin_y_nm: float,
    fov_nm: float,
    size_px: int,
    cd_bias_frac: float = 0.0,
    decoys: tuple[DecoySpec, ...] = (),
    rgb_reflectance: np.ndarray | None = None,
    cell_dropout: float = 0.0,
    cell_size_variation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a window of the latent world and its per-pixel material layer.

    ``cd_bias_frac`` grows/shrinks every drawn polygon's *width* (lines,
    contacts, routing traces) by that fraction while leaving every pitch,
    phase and zone period untouched - the layout-domain model of a CD/etch
    drift between two acquisitions.

    ``cell_dropout`` drops that fraction of checkerboard contacts via a
    deterministic per-cell hash of the scene seed, and ``cell_size_variation``
    modulates each contact's semiaxes by up to +/- that fraction per cell -
    per-cell process variation (missing and oversized/undersized vias) that
    is identical in both acquisitions of the same world and gives every
    window its own signature against pure lattice translates. Zeroes (the
    Phase 1 defaults) render the exact Phase 1 layout.

    ``decoys`` are near-duplicate motifs composited here, in the layout
    domain, before any rendering. Nothing is ever blended after rendering.

    ``rgb_reflectance``, when given as an ``(n_layers, 3)`` table, maps the
    layer ids to per-channel reflectance and returns an ``(H, W, 3)`` physical
    image; structural defects are painted with the same rng stream and masks
    as the grayscale path so both modalities describe one latent world.
    """
    px_nm = fov_nm / size_px
    xs = origin_x_nm + (np.arange(size_px, dtype=np.float32) + 0.5) * px_nm
    ys = origin_y_nm + (np.arange(size_px, dtype=np.float32) + 0.5) * px_nm
    x = xs[None, :]
    y = ys[:, None]
    phase = (spec.seed % 10_007) / 10_007.0 * 2 * np.pi
    cd = 1.0 + cd_bias_frac

    dx, ix, wx = _line_distance(
        x, y, spec.phase_x_nm, spec.pitch_x_nm, spec.width_x_nm * cd,
        spec.line_jitter_nm, spec.width_jitter_fraction, spec.roughness_nm,
        spec.roughness_lambda_nm, phase,
    )
    dy, iy, wy = _line_distance(
        y, x, spec.phase_y_nm, spec.pitch_y_nm, spec.width_y_nm * cd,
        spec.line_jitter_nm, spec.width_jitter_fraction, spec.roughness_nm,
        spec.roughness_lambda_nm * 1.19, phase + 1.37,
    )

    vertical = np.abs(dx) <= wx / 2.0
    horizontal = np.abs(dy) <= wy / 2.0
    parity = ((ix + iy) & 1) == 0
    contact_ax = max(spec.contact_x_nm * cd / 2.0, 1.0)
    contact_ay = max(spec.contact_y_nm * cd / 2.0, 1.0)
    if cell_dropout > 0 or cell_size_variation > 0:
        # Per-cell process variation, hashed from the scene seed and the cell
        # indices: identical in both acquisitions of the same world, never
        # correlated with the target, and absent from the Phase 1 path.
        keep = _cell_hash(ix, iy, spec.seed ^ 0xC0DAC7)
        if cell_dropout > 0:
            parity = parity & (keep >= cell_dropout)
        if cell_size_variation > 0:
            mod_x = 1.0 + cell_size_variation * 2.0 * (_cell_hash(ix, iy, spec.seed ^ 0x51CE) - 0.5)
            mod_y = 1.0 + cell_size_variation * 2.0 * (_cell_hash(ix, iy, spec.seed ^ 0x51CE2) - 0.5)
            contact_ax = np.maximum(contact_ax * mod_x, 1.0)
            contact_ay = np.maximum(contact_ay * mod_y, 1.0)
    contact = ((dx / contact_ax) ** 2 + (dy / contact_ay) ** 2 <= 1.0) & parity
    del contact_ax, contact_ay

    img = np.full((size_px, size_px), 0.10, dtype=np.float32)
    layers = np.zeros((size_px, size_px), dtype=np.uint8)
    if spec.architecture == "dram":
        img = np.where(horizontal, 0.52, img)
        layers[horizontal] = LAYER_HLINE
        img = np.where(vertical, np.maximum(img, 0.66), img)
        layers[vertical] = LAYER_VLINE
        img = np.where(contact, 0.92, img)
        layers[contact] = LAYER_CONTACT
    else:
        img = np.where(vertical, 0.58, img)       # fins
        layers[vertical] = LAYER_VLINE
        img = np.where(horizontal, np.maximum(img, 0.73), img)  # gates
        layers[horizontal] = LAYER_HLINE
        img = np.where(contact, 0.94, img)
        layers[contact] = LAYER_CONTACT

    # Array mats separated by peripheral/routing strips. These provide the
    # larger-scale context that real dense arrays possess and defeat the toy
    # assumption that the entire die is one infinite wallpaper.
    zx = np.mod(x + spec.zone_offset_x_nm, spec.mat_period_nm)
    zy = np.mod(y + spec.zone_offset_y_nm, spec.mat_period_nm)
    strip = (zx < spec.strip_width_nm) | (zy < spec.strip_width_nm)
    route_x = np.abs(np.mod(x + 37.0, 310.0) - 155.0) < 24.0 * cd
    route_y = np.abs(np.mod(y + 83.0, 370.0) - 185.0) < 28.0 * cd
    route = np.full_like(img, 0.15)
    route = np.where(route_x, 0.57, route)
    route = np.where(route_y, np.maximum(route, 0.68), route)
    route = np.where(route_x & route_y, 0.88, route)
    if spec.profile != "ambiguous":
        img = np.where(strip, route, img)
        layers[strip] = LAYER_ROUTE

    if rgb_reflectance is not None:
        img = np.clip(np.asarray(rgb_reflectance, dtype=np.float32)[layers], 0.0, 1.0).astype(np.float32)

    for decoy in decoys:
        img, layers = _composite_decoy(img, layers, decoy, spec, xs, ys, px_nm)

    if spec.defect_density > 0:
        _paint_defects(img, spec, xs, ys)
    return np.clip(img, 0.0, 1.0).astype(np.float32), layers


def _composite_decoy(
    img: np.ndarray,
    layers: np.ndarray,
    decoy: DecoySpec,
    spec: SceneSpec,
    xs: np.ndarray,
    ys: np.ndarray,
    px_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one near-duplicate motif over the base layout inside its window.

    Compositing happens on the layout stack (single render, no rendered-array
    blending). The motif keeps the host pitch and orientation; widths,
    contact sizes, occupancy and internal defect texture are perturbed.
    """
    rgb_mode = img.ndim == 3
    pad_nm = 60.0
    half_ext = decoy.half_nm + pad_nm
    i0 = max(int((decoy.cy_nm - half_ext - ys[0]) / px_nm), 0)
    i1 = min(int((decoy.cy_nm + half_ext - ys[0]) / px_nm) + 1, img.shape[-2])
    j0 = max(int((decoy.cx_nm - half_ext - xs[0]) / px_nm), 0)
    j1 = min(int((decoy.cx_nm + half_ext - xs[0]) / px_nm) + 1, img.shape[-1])
    if i1 <= i0 or j1 <= j0:
        return img, layers
    xs_sub = xs[j0:j1]
    ys_sub = ys[i0:i1]
    x = xs_sub[None, :]
    y = ys_sub[:, None]
    rng = np.random.default_rng(decoy.seed)

    # Width perturbation is decoy-internal: a per-decoy offset on top of the
    # shared width-jitter texture, so decoy linewidths disagree with the true
    # site without changing pitch.
    line_wobble = float(rng.uniform(-0.15, 0.15))
    parity_phase = int(rng.integers(0, 2))
    width_x = decoy.width_x_nm * (1.0 + line_wobble)
    width_y = decoy.width_y_nm * (1.0 + line_wobble)
    dx, ix, wx = _line_distance(
        x, y, decoy.phase_x_nm, spec.pitch_x_nm, width_x,
        spec.line_jitter_nm, spec.width_jitter_fraction,
        spec.roughness_nm, spec.roughness_lambda_nm,
        (decoy.seed % 10_007) / 10_007.0 * 2 * np.pi,
    )
    dy, iy, wy = _line_distance(
        y, x, decoy.phase_y_nm, spec.pitch_y_nm, width_y,
        spec.line_jitter_nm, spec.width_jitter_fraction,
        spec.roughness_nm, spec.roughness_lambda_nm * 1.19,
        (decoy.seed % 10_007) / 10_007.0 * 2 * np.pi + 1.37,
    )
    vertical = np.abs(dx) <= wx / 2.0
    horizontal = np.abs(dy) <= wy / 2.0
    parity = ((ix + iy) & 1) == parity_phase
    # Occupancy perturbation drops a deterministic subset of contacts so the
    # decoy cell grid is not a bit-level copy of the true motif's.
    keep = _cell_hash(ix, iy, decoy.seed) < decoy.occupancy
    contact = (
        (dx / max(decoy.contact_x_nm / 2.0, 1.0)) ** 2
        + (dy / max(decoy.contact_y_nm / 2.0, 1.0)) ** 2 <= 1.0
    ) & parity & keep

    if spec.architecture == "dram":
        motif = np.where(horizontal, 0.52, 0.10).astype(np.float32)
        motif = np.where(vertical, np.maximum(motif, 0.66), motif)
        motif = np.where(contact, 0.92, motif)
    else:
        motif = np.where(vertical, 0.58, 0.10).astype(np.float32)
        motif = np.where(horizontal, np.maximum(motif, 0.73), motif)
        motif = np.where(contact, 0.94, motif)
    motif_layers = np.zeros(motif.shape, dtype=np.uint8)
    motif_layers[horizontal] = LAYER_HLINE
    motif_layers[vertical] = LAYER_VLINE
    motif_layers[contact] = LAYER_CONTACT

    dist_x = np.abs(xs_sub - decoy.cx_nm)
    dist_y = np.abs(ys_sub - decoy.cy_nm)
    edge = np.clip((half_ext - np.maximum(dist_x[None, :], dist_y[:, None])) / pad_nm, 0.0, 1.0)
    alpha = (edge * edge * (3.0 - 2.0 * edge)).astype(np.float32)
    if rgb_mode:
        # RGB mode composites the motif's *materials*: the layer map decides
        # what the decoy is made of, so intensity-level defect specks (a
        # grayscale-mode texture perturbation) are skipped here.
        motif_rgb = RGB_REFLECTANCE[motif_layers]
        alpha3 = alpha[..., None]
        img[i0:i1, j0:j1] = img[i0:i1, j0:j1] * (1.0 - alpha3) + motif_rgb * alpha3
        layers[i0:i1, j0:j1] = np.where(alpha > 0.5, motif_layers, layers[i0:i1, j0:j1])
        return img, layers

    # A few bright/dark specks inside the decoy footprint so its defect
    # texture differs from the true site's.
    for _ in range(int(rng.integers(1, 4))):
        sx = float(rng.uniform(decoy.cx_nm - decoy.half_nm, decoy.cx_nm + decoy.half_nm))
        sy = float(rng.uniform(decoy.cy_nm - decoy.half_nm, decoy.cy_nm + decoy.half_nm))
        radius = float(rng.uniform(8.0, 26.0))
        rr = ((x - sx) ** 2 + (y - sy) ** 2) / max(radius**2, 1.0)
        if rng.random() < 0.5:
            motif = np.maximum(motif, (0.30 + 0.55 * np.exp(-2.2 * rr)).astype(np.float32))
        else:
            dark = rr <= 1.0
            motif[dark] *= float(rng.uniform(0.3, 0.6))

    img[i0:i1, j0:j1] = img[i0:i1, j0:j1] * (1.0 - alpha) + motif * alpha
    layers[i0:i1, j0:j1] = np.where(alpha > 0.5, motif_layers, layers[i0:i1, j0:j1])
    return img, layers


def _cell_hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic per-cell uniform in [0, 1) for occupancy decisions."""
    return np.mod(np.sin(ix * 12.9898 + iy * 78.233 + seed) * 43758.5453, 1.0)


def _apply_structural_defects(img: np.ndarray, spec: SceneSpec, xs: np.ndarray, ys: np.ndarray) -> None:
    _paint_defects(img, spec, xs, ys)


def _paint_defects(img: np.ndarray, spec: SceneSpec, xs: np.ndarray, ys: np.ndarray) -> None:
    """Paint structural defects; works on (H, W) or (H, W, C) stacks.

    For channel stacks the same masks and rng draws are shared across
    channels, so a defect is one physical object seen by every channel.

    Every operator is applied inside its defect's bounding box only. The
    float32 bump of a bright particle rounds to exactly 0.25 beyond
    rr ~ 17 (box corners reach rr <= 12.5), so the boxed composition
    ``max(max(img, 0.25), bumps)`` is bit-identical to the original
    full-array ``max(img, bump)`` per particle while touching a few percent
    of the pixels.
    """
    rng = np.random.default_rng(spec.seed * 65_537 + 19)
    n = int(round(10 * spec.defect_density))
    x = xs[None, :]
    y = ys[:, None]
    xmin, xmax = float(xs[0]), float(xs[-1])
    ymin, ymax = float(ys[0]), float(ys[-1])

    def apply_box(i0: int, i1: int, j0: int, j1: int) -> tuple[np.ndarray, np.ndarray]:
        """In-box coordinate grids; values are identical to the full-grid slices."""
        return x[0, j0:j1][None, :], y[i0:i1, 0][:, None]

    h, w = img.shape[:2]
    px_nm = float(xs[1] - xs[0]) if len(xs) > 1 else float(fov) if (fov := 0.0) else 1.0
    slack = 2.0 * px_nm + 1e-6  # int-truncation guard, in nanometres

    def box(i_axis: bool, center: float, half_nm: float) -> tuple[int, int]:
        if i_axis:
            lo = ys[0] + 0.5 * px_nm
            size = h
        else:
            lo = xs[0] + 0.5 * px_nm
            size = w
        half_px = int(np.ceil((half_nm + slack) / px_nm))
        c = int((center - lo) / px_nm)
        return max(c - half_px, 0), min(c + half_px + 1, size)

    for _ in range(n):
        cx = float(rng.uniform(0, WORLD_FOV_NM))
        cy = float(rng.uniform(0, WORLD_FOV_NM))
        radius = float(rng.uniform(18, 85))
        if cx + radius < xmin or cx - radius > xmax or cy + radius < ymin or cy - radius > ymax:
            continue
        kind = int(rng.integers(0, 4))
        if kind == 0:  # missing/etched patch (ellipse: |x-cx| <= r, |y-cy| <= 0.7r)
            i0, i1 = box(True, cy, 0.7 * radius)
            j0, j1 = box(False, cx, 1.0 * radius)
            if i1 <= i0 or j1 <= j0:
                continue
            bx, by = apply_box(i0, i1, j0, j1)
            mask = ((bx - cx) / radius) ** 2 + ((by - cy) / (radius * 0.7)) ** 2 <= 1
            img[i0:i1, j0:j1][mask] *= float(rng.uniform(0.15, 0.45))
        elif kind == 1:  # bright particle/residue - floor+bump in defect order
            # max(img, bump) with bump = 0.25 + 0.72*exp(-2.4*rr): outside the
            # box the bump equals exactly 0.25 in float32 (rr >= 7.5 at the box
            # edge), so a single floor pass plus the in-box bumps compose
            # bit-identically to the original full-array maximum, while later
            # dark defects keep their original order relative to it.
            np.maximum(img, np.float32(0.25), out=img)
            i0, i1 = box(True, cy, 7.5 * radius)
            j0, j1 = box(False, cx, 7.5 * radius)
            if i1 > i0 and j1 > j0:
                bx, by = apply_box(i0, i1, j0, j1)
                rr = ((bx - cx) ** 2 + (by - cy) ** 2) / max(radius ** 2, 1)
                region = img[i0:i1, j0:j1]
                bump = (0.25 + 0.72 * np.exp(-2.4 * rr)).astype(np.float32)
                if img.ndim == 3:
                    bump = bump[..., None]
                np.maximum(region, bump, out=region)
        elif kind == 2:  # bridge (|x-cx| < r, |y-cy| < 0.45r)
            i0, i1 = box(True, cy, 0.45 * radius)
            j0, j1 = box(False, cx, 1.0 * radius)
            if i1 <= i0 or j1 <= j0:
                continue
            bx, by = apply_box(i0, i1, j0, j1)
            mask = (np.abs(bx - cx) < radius) & (np.abs(by - cy) < radius * 0.45)
            region = img[i0:i1, j0:j1]
            region[mask] = np.maximum(region[mask], 0.78)
        else:  # scratch (|x-cx| < 2.8r, |y-cy| bounded by the line distance term)
            i0, i1 = box(True, cy, 2.8 * radius + max(0.12 * radius, 3.0))
            j0, j1 = box(False, cx, 2.8 * radius)
            if i1 <= i0 or j1 <= j0:
                continue
            bx, by = apply_box(i0, i1, j0, j1)
            slope = float(rng.uniform(-0.6, 0.6))
            dist = np.abs((by - cy) - slope * (bx - cx)) / math.sqrt(1 + slope ** 2)
            local = (np.abs(bx - cx) < radius * 2.8) & (dist < max(radius * 0.12, 3.0))
            region = img[i0:i1, j0:j1]
            region[local] = np.maximum(region[local], 0.82)

    # A larger missing-contact/residue cluster, distinguished from the four
    # random defect kinds above by its dark-core / bright-rim morphology.
    #
    # This is an ordinary fabrication anomaly drawn uniformly in world coordinates
    # from a stream seeded only by scene identity. Its placement never reads
    # the target centre, Reference crop, or ground-truth coordinate.
    sx, sy, radius, signature_rng = signature_placement(spec)
    if not (sx + radius < xmin or sx - radius > xmax or sy + radius < ymin or sy - radius > ymax):
        i0, i1 = box(True, sy, 0.72 * 1.45 * radius)
        j0, j1 = box(False, sx, 1.45 * radius)
        if i1 > i0 and j1 > j0:
            bx, by = apply_box(i0, i1, j0, j1)
            rr = ((bx - sx) / radius) ** 2 + ((by - sy) / (radius * 0.72)) ** 2
            missing = rr <= 1.0
            rim = (rr > 1.0) & (rr <= 1.45)
            region = img[i0:i1, j0:j1]
            region[missing] *= float(signature_rng.uniform(0.08, 0.28))
            region[rim] = np.maximum(region[rim], np.float32(float(signature_rng.uniform(0.72, 0.90))))
