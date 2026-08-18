from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import AcquisitionSpec


def sample_acquisition(rng: np.random.Generator, profile: str, reference: bool) -> AcquisitionSpec:
    hard = profile in {"hard", "ood", "stress"}
    ood = profile == "ood"
    if reference:
        dose = float(10 ** rng.uniform(np.log10(900), np.log10(4200)))
        read = float(rng.uniform(0.003, 0.016))
        rot = float(rng.uniform(-2.2, 2.2) * (1.5 if ood else 1.0))
        scale = float(rng.uniform(0.965, 1.035) if not ood else rng.uniform(0.93, 1.07))
        shear = float(rng.uniform(-0.25, 0.25))
        jitter = float(rng.uniform(0.0, 0.15))
    else:
        dose = float(10 ** rng.uniform(np.log10(70 if hard else 140), np.log10(700)))
        read = float(rng.uniform(0.010, 0.045 if hard else 0.032))
        rot = float(rng.uniform(-0.35, 0.35) * (2.2 if ood else 1.0))
        scale = float(rng.uniform(0.992, 1.008) if not ood else rng.uniform(0.975, 1.025))
        shear = float(rng.uniform(-4.5 if hard else -2.2, 4.5 if hard else 2.2))
        jitter = float(rng.uniform(0.05, 1.7 if hard else 0.75))
    return AcquisitionSpec(
        edge_strength=float(rng.uniform(0.14, 0.42)),
        edge_sigma_nm=float(rng.uniform(1.0, 4.5)),
        psf_sigma_x_nm=float(rng.uniform(2.0, 7.0 if not ood else 11.0)),
        psf_sigma_y_nm=float(rng.uniform(2.0, 8.5 if not ood else 13.0)),
        rotation_deg=rot,
        scale=scale,
        translate_x_px=float(rng.uniform(-1.0, 1.0) if reference else rng.uniform(-3.0, 3.0)),
        translate_y_px=float(rng.uniform(-1.0, 1.0) if reference else rng.uniform(-3.0, 3.0)),
        scan_shear_px=shear,
        scan_jitter_px=jitter,
        radial_k=float(rng.uniform(-0.006, 0.006) if not ood else rng.uniform(-0.018, 0.018)),
        dose=dose,
        read_noise_sigma=read,
        gain=float(rng.uniform(0.82, 1.20)),
        offset=float(rng.uniform(-0.08, 0.08)),
        gamma=float(rng.uniform(0.82, 1.22) if not ood else rng.uniform(0.65, 1.45)),
        vignette=float(rng.uniform(0.0, 0.20 if not hard else 0.34)),
        charging_strength=float(rng.uniform(0.0, 0.10 if not hard else 0.20)),
        charging_streaks=int(rng.integers(0, 3 if not hard else 6)),
        hot_pixel_rate=float(rng.uniform(0.0, 0.00015 if not hard else 0.0007)),
    )


def edge_brighten(img: np.ndarray, pixel_nm: float, strength: float, sigma_nm: float) -> np.ndarray:
    sigma_px = max(sigma_nm / pixel_nm, 0.35)
    smoothed = ndimage.gaussian_filter(img, sigma=sigma_px, mode="reflect")
    gx = ndimage.sobel(smoothed, axis=1, mode="reflect")
    gy = ndimage.sobel(smoothed, axis=0, mode="reflect")
    edge = np.hypot(gx, gy)
    scale = float(np.percentile(edge, 99.5)) + 1e-7
    return np.clip(img + strength * np.clip(edge / scale, 0, 1), 0, 1).astype(np.float32)


def apply_psf(img: np.ndarray, pixel_nm: float, spec: AcquisitionSpec) -> np.ndarray:
    sy = max(spec.psf_sigma_y_nm / pixel_nm, 0.15)
    sx = max(spec.psf_sigma_x_nm / pixel_nm, 0.15)
    return ndimage.gaussian_filter(img, sigma=(sy, sx), mode="reflect").astype(np.float32)


def area_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return img.astype(np.float32, copy=False)
    h, w = img.shape
    if h % factor or w % factor:
        raise ValueError("image dimensions must be divisible by factor")
    return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3), dtype=np.float32)


def _affine_warp(img: np.ndarray, spec: AcquisitionSpec, order: int) -> np.ndarray:
    theta = np.deg2rad(spec.rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    forward = spec.scale * np.array([[c, -s], [s, c]], dtype=np.float64)
    inv = np.linalg.inv(forward)
    # scipy coordinates are (y, x), while forward above is (x, y).
    inv_yx = inv[[1, 0]][:, [1, 0]]
    center = (np.array(img.shape, dtype=np.float64) - 1.0) / 2.0
    translation_yx = np.array([spec.translate_y_px, spec.translate_x_px])
    offset = center - inv_yx @ (center + translation_yx)
    return ndimage.affine_transform(img, inv_yx, offset=offset, order=order, mode="reflect", prefilter=order > 1)


def _scan_and_radial_warp(img: np.ndarray, spec: AcquisitionSpec, rng: np.random.Generator, order: int, row_jitter: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    nx, ny = (xx - cx) / max(cx, 1), (yy - cy) / max(cy, 1)
    r2 = nx * nx + ny * ny
    factor = 1.0 + spec.radial_k * r2
    map_x = cx + (xx - cx) * factor
    map_y = cy + (yy - cy) * factor
    if row_jitter is None:
        row_jitter = rng.normal(0, spec.scan_jitter_px, size=h).astype(np.float32)
    shear = spec.scan_shear_px * ((np.arange(h, dtype=np.float32) - cy) / max(h - 1, 1))
    map_x += (shear + row_jitter)[:, None]
    warped = ndimage.map_coordinates(img, [map_y, map_x], order=order, mode="reflect", prefilter=order > 1)
    return warped.astype(np.float32), row_jitter


def geometric_warp(img: np.ndarray, spec: AcquisitionSpec, rng: np.random.Generator, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    warped = _affine_warp(img, spec, order=1)
    warped, row_jitter = _scan_and_radial_warp(warped, spec, rng, order=1)
    warped_mask = None
    if mask is not None:
        warped_mask = _affine_warp(mask, spec, order=1)
        warped_mask, _ = _scan_and_radial_warp(warped_mask, spec, rng, order=1, row_jitter=row_jitter)
    return warped, warped_mask


def photometric_capture(img: np.ndarray, spec: AcquisitionSpec, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2
    out = img * np.clip(1.0 - spec.vignette * r2 / 2.0, 0.55, 1.0)

    if spec.charging_strength > 0:
        field = ndimage.gaussian_filter(rng.normal(0, 1, size=(max(8, h // 32), max(8, w // 32))), 1.2)
        field = ndimage.zoom(field, (h / field.shape[0], w / field.shape[1]), order=1)[:h, :w]
        field /= float(np.std(field)) + 1e-6
        out += spec.charging_strength * field
    for _ in range(spec.charging_streaks):
        row = int(rng.integers(0, h))
        band = int(rng.integers(1, 5))
        out[max(0, row - band):min(h, row + band + 1)] += float(rng.uniform(0.03, 0.13))

    out = np.clip(spec.gain * out + spec.offset, 0, 1)
    out = np.power(out, spec.gamma)
    counts = np.clip(out * spec.dose, 0, None)
    out = rng.poisson(counts).astype(np.float32) / max(spec.dose, 1e-6)
    out += rng.normal(0, spec.read_noise_sigma, size=out.shape).astype(np.float32)

    if spec.hot_pixel_rate > 0:
        hit = rng.random(out.shape) < spec.hot_pixel_rate
        hot = rng.random(out.shape) < 0.5
        out[hit & hot] = 1.0
        out[hit & ~hot] = 0.0
    return np.clip(out, 0, 1).astype(np.float32)


def acquire_reference(physical: np.ndarray, pixel_nm: float, spec: AcquisitionSpec, rng: np.random.Generator) -> np.ndarray:
    out = edge_brighten(physical, pixel_nm, spec.edge_strength, spec.edge_sigma_nm)
    out = apply_psf(out, pixel_nm, spec)
    out, _ = geometric_warp(out, spec, rng)
    return photometric_capture(out, spec, rng)


def acquire_search(physical_super: np.ndarray, pixel_nm_super: float, supersample: int, spec: AcquisitionSpec, rng: np.random.Generator, target_mask_super: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = edge_brighten(physical_super, pixel_nm_super, spec.edge_strength, spec.edge_sigma_nm)
    out = apply_psf(out, pixel_nm_super, spec)
    out = area_downsample(out, supersample)
    mask = area_downsample(target_mask_super, supersample)
    out, mask = geometric_warp(out, spec, rng, mask=mask)
    out = photometric_capture(out, spec, rng)
    return out, mask


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(img, 0, 1) * 255.0).astype(np.uint8)
