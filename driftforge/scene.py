from __future__ import annotations

import math

import numpy as np

from .config import PRESETS, REFERENCE_FOV_NM, WORLD_FOV_NM, SceneSpec


#: Radius range of the distinctive dark-core/bright-rim fabrication anomaly. Also
#: used as the world-edge margin so the cluster renders whole wherever it lands.
SIGNATURE_RADIUS_NM = (72.0, 135.0)


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


def make_scene_spec(seed: int, architecture: str, profile: str) -> SceneSpec:
    rng = np.random.default_rng(seed)
    if architecture not in {"dram", "finfet"}:
        raise ValueError("architecture must be 'dram' or 'finfet'")

    preset_name = f"{architecture}_{'dense' if rng.random() < 0.65 else 'open'}"
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
    px_nm = fov_nm / size_px
    xs = origin_x_nm + (np.arange(size_px, dtype=np.float32) + 0.5) * px_nm
    ys = origin_y_nm + (np.arange(size_px, dtype=np.float32) + 0.5) * px_nm
    x = xs[None, :]
    y = ys[:, None]
    phase = (spec.seed % 10_007) / 10_007.0 * 2 * np.pi

    dx, ix, wx = _line_distance(
        x, y, spec.phase_x_nm, spec.pitch_x_nm, spec.width_x_nm,
        spec.line_jitter_nm, spec.width_jitter_fraction, spec.roughness_nm,
        spec.roughness_lambda_nm, phase,
    )
    dy, iy, wy = _line_distance(
        y, x, spec.phase_y_nm, spec.pitch_y_nm, spec.width_y_nm,
        spec.line_jitter_nm, spec.width_jitter_fraction, spec.roughness_nm,
        spec.roughness_lambda_nm * 1.19, phase + 1.37,
    )

    vertical = np.abs(dx) <= wx / 2.0
    horizontal = np.abs(dy) <= wy / 2.0
    parity = ((ix + iy) & 1) == 0
    contact = ((dx / max(spec.contact_x_nm / 2.0, 1.0)) ** 2 + (dy / max(spec.contact_y_nm / 2.0, 1.0)) ** 2 <= 1.0) & parity

    img = np.full((size_px, size_px), 0.10, dtype=np.float32)
    if spec.architecture == "dram":
        img = np.where(horizontal, 0.52, img)
        img = np.where(vertical, np.maximum(img, 0.66), img)
        img = np.where(contact, 0.92, img)
    else:
        img = np.where(vertical, 0.58, img)       # fins
        img = np.where(horizontal, np.maximum(img, 0.73), img)  # gates
        img = np.where(contact, 0.94, img)

    # Array mats separated by peripheral/routing strips. These provide the
    # larger-scale context that real dense arrays possess and defeat the toy
    # assumption that the entire die is one infinite wallpaper.
    zx = np.mod(x + spec.zone_offset_x_nm, spec.mat_period_nm)
    zy = np.mod(y + spec.zone_offset_y_nm, spec.mat_period_nm)
    strip = (zx < spec.strip_width_nm) | (zy < spec.strip_width_nm)
    route_x = np.abs(np.mod(x + 37.0, 310.0) - 155.0) < 24.0
    route_y = np.abs(np.mod(y + 83.0, 370.0) - 185.0) < 28.0
    route = np.full_like(img, 0.15)
    route = np.where(route_x, 0.57, route)
    route = np.where(route_y, np.maximum(route, 0.68), route)
    route = np.where(route_x & route_y, 0.88, route)
    if spec.profile != "ambiguous":
        img = np.where(strip, route, img)

    if spec.defect_density > 0:
        _apply_structural_defects(img, spec, xs, ys)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _apply_structural_defects(img: np.ndarray, spec: SceneSpec, xs: np.ndarray, ys: np.ndarray) -> None:
    rng = np.random.default_rng(spec.seed * 65_537 + 19)
    n = int(round(10 * spec.defect_density))
    x = xs[None, :]
    y = ys[:, None]
    xmin, xmax = float(xs[0]), float(xs[-1])
    ymin, ymax = float(ys[0]), float(ys[-1])
    for _ in range(n):
        cx = float(rng.uniform(0, WORLD_FOV_NM))
        cy = float(rng.uniform(0, WORLD_FOV_NM))
        radius = float(rng.uniform(18, 85))
        if cx + radius < xmin or cx - radius > xmax or cy + radius < ymin or cy - radius > ymax:
            continue
        kind = int(rng.integers(0, 4))
        if kind == 0:  # missing/etched patch
            mask = ((x - cx) / radius) ** 2 + ((y - cy) / (radius * 0.7)) ** 2 <= 1
            img[mask] *= float(rng.uniform(0.15, 0.45))
        elif kind == 1:  # bright particle/residue
            rr = ((x - cx) ** 2 + (y - cy) ** 2) / max(radius ** 2, 1)
            img[:] = np.maximum(img, (0.25 + 0.72 * np.exp(-2.4 * rr)).astype(np.float32))
        elif kind == 2:  # bridge
            w = radius * 0.45
            mask = (np.abs(x - cx) < radius) & (np.abs(y - cy) < w)
            img[mask] = np.maximum(img[mask], 0.78)
        else:  # scratch
            slope = float(rng.uniform(-0.6, 0.6))
            dist = np.abs((y - cy) - slope * (x - cx)) / math.sqrt(1 + slope ** 2)
            local = (np.abs(x - cx) < radius * 2.8) & (dist < max(radius * 0.12, 3.0))
            img[local] = np.maximum(img[local], 0.82)

    # A larger missing-contact/residue cluster, distinguished from the four
    # random defect kinds above by its dark-core / bright-rim morphology.
    #
    # This is an ordinary fabrication anomaly drawn uniformly in world coordinates
    # from a stream seeded only by scene identity. Its placement never reads
    # the target centre, Reference crop, or ground-truth coordinate.
    sx, sy, radius, signature_rng = signature_placement(spec)
    if not (sx + radius < xmin or sx - radius > xmax or sy + radius < ymin or sy - radius > ymax):
        rr = ((x - sx) / radius) ** 2 + ((y - sy) / (radius * 0.72)) ** 2
        missing = rr <= 1.0
        rim = (rr > 1.0) & (rr <= 1.45)
        img[missing] *= float(signature_rng.uniform(0.08, 0.28))
        img[rim] = np.maximum(img[rim], float(signature_rng.uniform(0.72, 0.90)))
