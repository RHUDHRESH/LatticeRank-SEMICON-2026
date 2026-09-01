"""Average physical unit-cell model -- departures from nominal identify the true site.

The semiconductor lattice repeats with period (v1, v2), but the *structure*
within that cell varies. Two nearby locations that look identical under one
pose differ sharply when we warp the patch into canonical unit-cell coordinates
and compare against the nominal (median) cell.

We infer the nominal unit cell by sampling 8-24 nearby lattice periods,
warping each into a fixed (24, 24) pixel grid via the basis (v1, v2), and
taking the pixel-wise median and MAD-based variance. Then at the candidate:

    mu(u,v)    = median over cells of C_k(u,v)
    sigma(u,v) = 1.4826 * MAD over cells of C_k(u,v)
    Z(u,v)     = (C(u,v) - mu(u,v)) / (sigma(u,v) + eps)

The reference and candidate Z-field are compared: agreement on structure
identifies the true site; random departures (aliases) do not.

This reuses the periodic-cancellation machinery (lattice basis from context,
no distance-to-centre). The physical intuition: periodic aliases share the
lattice structure but differ in local defects and variations. Nominal
cancellation lets those variations speak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Feature names, matching the required output
FEATURES = [
    "uc_z_corr",
    "uc_z_sign_agreement",
    "uc_outlier_overlap",
    "uc_top5pct_overlap",
    "uc_top10pct_overlap",
    "uc_defect_centroid_error",
    "uc_n_cells",
    "uc_sigma_median",
]

# Nominal unit-cell grid size, in pixels
UC_GRID = 24

# Number of nearby lattice periods to sample for nominal model
N_CELLS_NOMINAL = 16

# Outlier threshold (number of sigma) for defect detection
OUTLIER_SIGMA = 3.0

# Small epsilon to avoid division by zero
EPS = 1e-6


@dataclass
class UnitCellState:
    """Per-scene nominal unit cell, built from lattice-warped samples.

    Attributes:
        mu: (UC_GRID, UC_GRID) float32, median pixel value per position
        sigma: (UC_GRID, UC_GRID) float32, MAD-based sigma per position
        n_cells: int, number of cells successfully sampled
        template_z: (UC_GRID, UC_GRID) float32, Z-field of the template
        template_c: (UC_GRID, UC_GRID) float32, warped template cell
        search_f: (H, W) float32, robust-contrast search image for warping
        v1: (2,) float64, lattice basis vector 1
        v2: (2,) float64, lattice basis vector 2
    """
    mu: np.ndarray
    sigma: np.ndarray
    n_cells: int
    template_z: np.ndarray
    template_c: np.ndarray
    search_f: np.ndarray
    v1: np.ndarray
    v2: np.ndarray


def _warp_to_canonical(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    v1: np.ndarray,
    v2: np.ndarray,
    grid_size: int = UC_GRID,
) -> np.ndarray | None:
    """Warp one unit cell around (center_x, center_y) into canonical grid.

    The canonical grid is (grid_size, grid_size) pixels, sampled uniformly
    in the (v1, v2) basis. Coordinates are in search-image space. Returns None
    if the patch extends outside the image.

    Args:
        image: (H, W) float32 search image
        center_x, center_y: centre of the cell in image pixels
        v1, v2: (2,) float64 lattice basis vectors in pixels
        grid_size: output grid size (default 24)

    Returns:
        (grid_size, grid_size) float32 warped cell, or None if clipped
    """
    h, w = image.shape[:2]
    # Canonical grid: unit square in (a, b) parameter space, sampled on a
    # grid_size x grid_size mesh, then transformed to (x, y) via
    #   (x, y) = center + a*v1 + b*v2
    grid = np.linspace(0, 1, grid_size, endpoint=False)
    a_grid, b_grid = np.meshgrid(grid, grid, indexing="ij")

    # Transformed coordinates: (x, y) = center + a*v1 + b*v2
    x_coords = center_x + a_grid * v1[0] + b_grid * v2[0]
    y_coords = center_y + a_grid * v1[1] + b_grid * v2[1]

    # Check bounds
    if (x_coords < 0).any() or (x_coords >= w).any() or \
       (y_coords < 0).any() or (y_coords >= h).any():
        return None

    # Bilinear interpolation from image
    img_f = image if image.dtype == np.float32 else image.astype(np.float32)
    output = ndimage.map_coordinates(
        img_f,
        [y_coords, x_coords],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=True,
    )
    # If any NaN (clipped region), reject the whole cell
    if np.isnan(output).any():
        return None
    return output.astype(np.float32)


def _robust_sigma(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """MAD-based robust standard deviation along an axis.

    sigma = 1.4826 * MAD(stack, axis=axis)
    """
    median = np.median(stack, axis=axis, keepdims=True)
    mad = np.median(np.abs(stack - median), axis=axis)
    return 1.4826 * mad


def build(ctx) -> UnitCellState | None:
    """Build the nominal unit cell from the template, sampled at nearby cells.

    Returns None if:
    - ctx.lattice_ok is False
    - Fewer than 3 cells can be warped successfully
    """
    if not ctx.lattice_ok:
        return None

    h, w = ctx.template.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    # Sample cells around the centre of the template, in the lattice basis.
    # We sample rings around (0, 0) in (m, n) integer coordinates.
    cells_list = []
    cell_coords = []

    rings = 2  # Sample rings 0, 1, 2 around the centre: (0,0), (±1, 0), (0, ±1), (±1, ±1), (±2, ±1), etc.
    for m in range(-rings, rings + 1):
        for n in range(-rings, rings + 1):
            if m == 0 and n == 0:
                continue  # Skip the origin; we use the centre of the template itself
            # Offset in template coordinates
            dy_offset = m * ctx.v1[1] + n * ctx.v2[1]
            dx_offset = m * ctx.v1[0] + n * ctx.v2[0]
            cell_y = cy + dy_offset
            cell_x = cx + dx_offset

            # Warp this cell into canonical coordinates
            warped = _warp_to_canonical(
                ctx.template, cell_x, cell_y, ctx.v1, ctx.v2, UC_GRID
            )
            if warped is not None:
                cells_list.append(warped)
                cell_coords.append((m, n))

    if len(cells_list) < 3:
        return None

    # Stack all cells and compute per-pixel statistics
    cells_stack = np.stack(cells_list, axis=0)  # (n_cells, UC_GRID, UC_GRID)
    n_cells = len(cells_list)

    # Nominal unit cell: median and robust sigma
    mu = np.median(cells_stack, axis=0).astype(np.float32)
    sigma = _robust_sigma(cells_stack, axis=0).astype(np.float32)
    sigma = np.maximum(sigma, 1e-6)  # Ensure non-zero

    # Compute Z-field of the template (centre cell)
    template_c = _warp_to_canonical(
        ctx.template, cx, cy, ctx.v1, ctx.v2, UC_GRID
    )
    if template_c is None:
        return None

    z_ref = (template_c - mu) / (sigma + EPS)

    return UnitCellState(
        mu=mu,
        sigma=sigma,
        n_cells=n_cells,
        template_z=z_ref.astype(np.float32),
        template_c=template_c.astype(np.float32),
        search_f=ctx.search_f.astype(np.float32),
        v1=ctx.v1.copy(),
        v2=ctx.v2.copy(),
    )


def score(state, x: float, y: float) -> dict[str, float]:
    """Score a candidate location against the nominal unit cell.

    Args:
        state: UnitCellState from build(), or None if abstaining
        x, y: candidate centre in search-image pixels

    Returns:
        dict with exactly the FEATURES keys, floats or NaN where unmeasurable
    """
    out = {k: float("nan") for k in FEATURES}

    if state is None:
        return out

    # Warp the candidate into canonical coordinates
    candidate_c = _warp_to_canonical(
        state.search_f, x, y, state.v1, state.v2, UC_GRID
    )
    if candidate_c is None:
        return out

    # Z-field of the candidate
    z_candidate = (candidate_c - state.mu) / (state.sigma + EPS)

    # Feature 1: Z correlation
    z_ref_flat = state.template_z.ravel()
    z_cand_flat = z_candidate.ravel()
    valid = np.isfinite(z_ref_flat) & np.isfinite(z_cand_flat)
    if valid.sum() > 3:
        z_ref_norm = z_ref_flat[valid] - z_ref_flat[valid].mean()
        z_cand_norm = z_cand_flat[valid] - z_cand_flat[valid].mean()
        r_ref = np.linalg.norm(z_ref_norm)
        r_cand = np.linalg.norm(z_cand_norm)
        if r_ref > 1e-6 and r_cand > 1e-6:
            corr = float(np.dot(z_ref_norm, z_cand_norm) / (r_ref * r_cand))
        else:
            corr = float("nan")
    else:
        corr = float("nan")
    out["uc_z_corr"] = corr

    # Feature 2: Z sign agreement
    if valid.sum() > 1:
        sign_agr = float(np.mean((z_ref_flat[valid] * z_cand_flat[valid]) > 0))
    else:
        sign_agr = float("nan")
    out["uc_z_sign_agreement"] = sign_agr

    # Feature 3: Outlier overlap (both exceed OUTLIER_SIGMA)
    z_ref_outlier = np.abs(state.template_z) > OUTLIER_SIGMA
    z_cand_outlier = np.abs(z_candidate) > OUTLIER_SIGMA
    if z_ref_outlier.sum() > 0 or z_cand_outlier.sum() > 0:
        overlap = float(np.logical_and(z_ref_outlier, z_cand_outlier).sum()
                       / np.logical_or(z_ref_outlier, z_cand_outlier).sum())
    else:
        overlap = 1.0  # No outliers in either; perfect agreement
    out["uc_outlier_overlap"] = overlap

    # Feature 4: Top 5% overlap
    thresh_ref_5 = np.percentile(np.abs(state.template_z), 95)
    thresh_cand_5 = np.percentile(np.abs(z_candidate), 95)
    top_ref_5 = np.abs(state.template_z) > thresh_ref_5
    top_cand_5 = np.abs(z_candidate) > thresh_cand_5
    if top_ref_5.sum() > 0 or top_cand_5.sum() > 0:
        overlap_5 = float(np.logical_and(top_ref_5, top_cand_5).sum()
                         / np.logical_or(top_ref_5, top_cand_5).sum())
    else:
        overlap_5 = 1.0
    out["uc_top5pct_overlap"] = overlap_5

    # Feature 5: Top 10% overlap
    thresh_ref_10 = np.percentile(np.abs(state.template_z), 90)
    thresh_cand_10 = np.percentile(np.abs(z_candidate), 90)
    top_ref_10 = np.abs(state.template_z) > thresh_ref_10
    top_cand_10 = np.abs(z_candidate) > thresh_cand_10
    if top_ref_10.sum() > 0 or top_cand_10.sum() > 0:
        overlap_10 = float(np.logical_and(top_ref_10, top_cand_10).sum()
                          / np.logical_or(top_ref_10, top_cand_10).sum())
    else:
        overlap_10 = 1.0
    out["uc_top10pct_overlap"] = overlap_10

    # Feature 6: Defect centroid error
    # Find outlier pixels and compute centroid displacement
    z_ref_outlier_mask = np.abs(state.template_z) > OUTLIER_SIGMA
    z_cand_outlier_mask = np.abs(z_candidate) > OUTLIER_SIGMA
    if z_ref_outlier_mask.sum() > 0 and z_cand_outlier_mask.sum() > 0:
        ref_centroid = np.array(np.unravel_index(
            np.argmax(z_ref_outlier_mask), z_ref_outlier_mask.shape
        ), dtype=np.float32)
        cand_centroid = np.array(np.unravel_index(
            np.argmax(z_cand_outlier_mask), z_cand_outlier_mask.shape
        ), dtype=np.float32)
        centroid_error = float(np.linalg.norm(ref_centroid - cand_centroid))
    else:
        centroid_error = float("nan")
    out["uc_defect_centroid_error"] = centroid_error

    # Feature 7: Number of cells used
    out["uc_n_cells"] = float(state.n_cells)

    # Feature 8: Median sigma (variability across the nominal cell)
    out["uc_sigma_median"] = float(np.median(state.sigma))

    return out
