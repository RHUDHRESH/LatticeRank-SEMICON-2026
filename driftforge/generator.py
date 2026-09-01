from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import OUTPUT_SIZE, REFERENCE_FOV_NM, WORLD_FOV_NM, AcquisitionSpec, SceneSpec
from .phase2 import (
    ABSENT_FRACTION,
    CD_BIAS_FRAC,
    CONTACT_DROPOUT_RANGE,
    CONTACT_SIZE_VARIATION_RANGE,
    DECOY_PROBABILITY,
    LOCAL_DEFECT_LAMBDA,
    OCCLUSION_FRAC_RANGE,
    _apply_channel_shifts,
    acquire_rgb_reference,
    acquire_rgb_search,
    apply_reference_damage,
    make_decoy_specs,
    modulated_reflectance,
    net_rotation_label,
    sample_architecture,
    sample_edge_case,
    sample_family,
    sample_phase2_acquisition,
    sample_profile,
    sample_rgb_capture,
    sample_rgb_chroma,
    sample_severity,
    sample_stage_theta,
    sample_target_site,
    sample_zoom,
    severity_scene,
)
from .scene import make_scene_spec, render_scene, render_scene_with_layers
from .sem import acquire_reference, acquire_search, sample_acquisition, to_uint8

ARCHITECTURES = ("dram", "finfet")
PROFILES = ("standard", "hard", "boundary", "ambiguous", "ood")


def normalize_architecture(value: str) -> str:
    """Return the canonical architecture name, accepting display-case forms."""
    if not isinstance(value, str):
        raise TypeError("architecture must be a string")
    architecture = value.strip().casefold()
    if architecture not in ARCHITECTURES:
        raise ValueError("architecture must be DRAM or FinFET")
    return architecture


def normalize_profile(value: str) -> str:
    """Return a supported canonical generation profile."""
    if not isinstance(value, str):
        raise TypeError("profile must be a string")
    profile = value.strip().casefold()
    if profile not in PROFILES:
        raise ValueError(
            "profile must be one of: " + ", ".join(PROFILES)
        )
    return profile


@dataclass
class Sample:
    reference: np.ndarray
    search: np.ndarray
    gt_x: float
    gt_y: float
    nominal_gt_x: float
    nominal_gt_y: float
    architecture: str
    profile: str
    seed: int
    scene: SceneSpec
    reference_acquisition: AcquisitionSpec
    search_acquisition: AcquisitionSpec
    target_mask: np.ndarray | None = None

    def metadata(self) -> dict:
        data = {
            "seed": self.seed,
            "architecture": self.architecture,
            "profile": self.profile,
            "gt_x": self.gt_x,
            "gt_y": self.gt_y,
            "nominal_gt_x": self.nominal_gt_x,
            "nominal_gt_y": self.nominal_gt_y,
        }
        data.update(self.scene.to_dict())
        data.update(self.reference_acquisition.to_dict("ref_"))
        data.update(self.search_acquisition.to_dict("search_"))
        return data


def _target_mask(size: int, cx_px: float, cy_px: float, target_size_px: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    half = target_size_px / 2.0
    # Soft box makes centroid recovery stable under interpolation and nonlinear
    # scan warps while still representing the complete matching region.
    dx = np.maximum(np.abs(xx - cx_px) - half, 0)
    dy = np.maximum(np.abs(yy - cy_px) - half, 0)
    return np.exp(-(dx * dx + dy * dy) / 2.0).astype(np.float32)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    weights = np.clip(mask.astype(np.float64), 0, None)
    total = float(weights.sum())
    if total <= 1e-9:
        raise RuntimeError("target mask disappeared during geometric warp")
    yy, xx = np.indices(mask.shape)
    return float((weights * xx).sum() / total), float((weights * yy).sum() / total)


def _mask_area_stretch(warped_mask: np.ndarray, nominal_mask: np.ndarray) -> float:
    """Local area stretch of the geometric warp, from mask second moments.

    The tracked mask is a soft square, so its covariance transforms as
    ``C = A C0 A^T`` under the local affine part of the warp and
    ``sqrt(det C / det C0) = |det A|`` is the area stretch - invariant to the
    rotation and insensitive to pure shear. The apparent zoom a matcher must
    undo is the nominal FOV zoom divided by that stretch, which keeps the
    scale label *measured* (§2.2) instead of assuming the warp away.
    """
    nominal = _mask_moments(nominal_mask)
    warped = _mask_moments(warped_mask)
    det_ratio = (warped[2] / nominal[2]) if nominal[2] > 1e-9 else 1.0
    return float(np.sqrt(max(det_ratio, 1e-12)))


def _mask_moments(mask: np.ndarray) -> tuple[float, float, float]:
    """(mu_x, mu_y, determinant of the covariance) of a soft mask."""
    weights = np.clip(mask.astype(np.float64), 0, None)
    total = float(weights.sum())
    if total <= 1e-9:
        return 0.0, 0.0, 1.0
    yy, xx = np.indices(mask.shape, dtype=np.float64)
    mx = float((weights * xx).sum() / total)
    my = float((weights * yy).sum() / total)
    vxx = float((weights * (xx - mx) ** 2).sum() / total)
    vyy = float((weights * (yy - my) ** 2).sum() / total)
    vxy = float((weights * (xx - mx) * (yy - my)).sum() / total)
    return mx, my, vxx * vyy - vxy * vxy


def generate_sample(
    seed: int,
    architecture: str | None = None,
    profile: str = "standard",
    search_supersample: int = 2,
    return_debug: bool = False,
) -> Sample:
    if search_supersample not in {1, 2, 4}:
        raise ValueError("search_supersample must be 1, 2, or 4")
    master = np.random.default_rng(seed)
    architecture = (
        normalize_architecture(architecture)
        if architecture is not None
        else ("dram" if master.random() < 0.5 else "finfet")
    )
    profile = normalize_profile(profile)
    scene = make_scene_spec(seed * 17 + 3, architecture, profile)

    ref_rng = np.random.default_rng(seed * 7_919 + 101)
    search_rng = np.random.default_rng(seed * 1_000_003 + 313)
    ref_acq = sample_acquisition(ref_rng, profile, reference=True)
    search_acq = sample_acquisition(search_rng, profile, reference=False)

    ref_origin_x = scene.target_cx_nm - REFERENCE_FOV_NM / 2.0
    ref_origin_y = scene.target_cy_nm - REFERENCE_FOV_NM / 2.0
    physical_ref = render_scene(scene, ref_origin_x, ref_origin_y, REFERENCE_FOV_NM, OUTPUT_SIZE)
    reference = acquire_reference(physical_ref, REFERENCE_FOV_NM / OUTPUT_SIZE, ref_acq, ref_rng)

    super_size = OUTPUT_SIZE * search_supersample
    physical_search = render_scene(scene, 0.0, 0.0, WORLD_FOV_NM, super_size)
    nominal_x = scene.target_cx_nm / (WORLD_FOV_NM / OUTPUT_SIZE)
    nominal_y = scene.target_cy_nm / (WORLD_FOV_NM / OUTPUT_SIZE)
    target_size_super = REFERENCE_FOV_NM / (WORLD_FOV_NM / super_size)
    mask_super = _target_mask(super_size, nominal_x * search_supersample, nominal_y * search_supersample, target_size_super)
    search, target_mask = acquire_search(
        physical_search,
        WORLD_FOV_NM / super_size,
        search_supersample,
        search_acq,
        search_rng,
        mask_super,
    )
    gt_x, gt_y = _mask_centroid(target_mask)

    return Sample(
        reference=to_uint8(reference),
        search=to_uint8(search),
        gt_x=gt_x,
        gt_y=gt_y,
        nominal_gt_x=nominal_x,
        nominal_gt_y=nominal_y,
        architecture=architecture,
        profile=profile,
        seed=seed,
        scene=scene,
        reference_acquisition=ref_acq,
        search_acquisition=search_acq,
        target_mask=target_mask if return_debug else None,
    )


# ---------------------------------------------------------------------------
# Phase 2: unknown zoom, unknown rotation, absent pairs, severity ladder,
# decoys, RGB optical mode. See docs/REFERENCES.md for the parameter-level
# citation of every range introduced below.
# ---------------------------------------------------------------------------

ABSENT_SCENE_XOR = 0x5EED
PHASE2_SPLITS = (
    "p2_train", "p2_val", "p2_test", "p2_holdout_fam", "p2_stress", "p2_val_rgb",
    "p2_bulk",
)


@dataclass
class Phase2Sample:
    """One Phase 2 pair. All arrays are uint8: (1000, 1000) or (1000, 1000, 3)."""

    reference: np.ndarray
    search: np.ndarray
    present: bool
    gt_x: float | None
    gt_y: float | None
    gt_theta: float | None
    gt_scale: float
    severity: int
    modality: str
    architecture: str
    preset_family: str
    n_decoys: int
    metadata: dict


def generate_phase2_sample(
    seed: int,
    *,
    architecture: str | None = None,
    preset_family: str | None = None,
    severity: int | None = None,
    present: bool | None = None,
    modality: str = "gray",
    n_decoys: int | None = None,
    search_supersample: int = 2,
    split: str = "p2_train",
    profile: str | None = None,
    edge_case: str | None = None,
    present_frac: float = 1.0 - ABSENT_FRACTION,
) -> Phase2Sample:
    """Generate one Phase 2 pair, deterministically from ``seed``.

    The pair is produced by two independent acquisitions of the latent world:

    - **Search** renders the whole 10 um world at ``WORLD_FOV_NM`` with the
      severity ladder's degradation, the pair's CD bias (§3.3), and any
      decoys (§3.5), then acquires it with its own AcquisitionSpec.
    - **Reference** renders ``WORLD_FOV_NM / s`` nanometres of the same world
      at the sampled site - or, for absent pairs (§3.4), of a different
      structural realization of the same architecture and preset family -
      and acquires it with the stage orientation applied on top of the
      reference acquisition jitter (§3.1).

    Zoom is produced **only** by the reference field of view (§2.1); nothing
    is ever resized after rendering. Ground truth is measured from the target
    mask tracked through the identical search warp (§2.2), and the rotation
    label uses the convention verified by ``scripts/verify_conventions.py``
    (§2.3).

    Draw order (fixed, so a seed fully determines the sample): severity and
    the present/absent coin (each from its own dedicated stream, decorrelated
    from the imagery by construction), then architecture, preset family,
    profile, edge case, zoom s, stage theta, CD bias, occlusion fraction,
    target site, decoy coin and decoy layouts; the reference and search
    acquisition streams are seeded independently by ``ref_seed`` /
    ``search_seed``.
    """
    if search_supersample not in {1, 2, 4}:
        raise ValueError("search_supersample must be 1, 2, or 4")
    if modality not in {"gray", "rgb"}:
        raise ValueError("modality must be 'gray' or 'rgb'")
    if split not in PHASE2_SPLITS:
        raise ValueError("split must be one of: " + ", ".join(PHASE2_SPLITS))
    if not 0 <= int(severity if severity is not None else 0) <= 3:
        raise ValueError("severity must be in 0..3")
    if not 0.0 <= present_frac <= 1.0:
        raise ValueError("present_frac must be within [0, 1]")

    rng = np.random.default_rng(seed)
    # Severity draws from a dedicated stream as well: a master-stream draw
    # position showed the same learnable cross-draw association that gate G4
    # exposed for the present/absent coin, and the severity ladder should be
    # image-visible only through its physical parameters, not through stream
    # structure.
    sev = (
        int(severity)
        if severity is not None
        else sample_severity(split, np.random.default_rng(seed ^ 0x5E7))
    )
    # The present/absent coin comes from a dedicated stream. Drawing it from
    # the master stream left a small but learnable statistical association
    # between the label and the imagery across the split (gate G4 measured
    # AUC 0.60 at n=1600 despite verified per-seed independence), so the
    # label is decorrelated from every other draw by construction.
    present = bool(np.random.default_rng(seed ^ 0xC1A55).random() < present_frac) if present is None else bool(present)
    if architecture is not None:
        arch = normalize_architecture(architecture)
    elif split == "p2_holdout_fam":
        from .phase2 import HOLDOUT_FAMILY

        arch = HOLDOUT_FAMILY.split("_")[0]
    else:
        arch = sample_architecture(rng)
    family = preset_family if preset_family is not None else sample_family(rng, split, arch)
    prof = profile if profile is not None else sample_profile(rng)
    edge = edge_case if edge_case is not None else sample_edge_case(rng)
    s = sample_zoom(rng, edge)
    stage_theta = sample_stage_theta(rng, edge)
    cd_band = CD_BIAS_FRAC[sev]
    cd_half = float(rng.uniform(cd_band[0], cd_band[1]))
    cd_bias = float(rng.uniform(-cd_half, cd_half))
    occlusion_target = float(rng.uniform(*OCCLUSION_FRAC_RANGE[sev]))
    cell_dropout = float(rng.uniform(*CONTACT_DROPOUT_RANGE))
    cell_variation = float(rng.uniform(*CONTACT_SIZE_VARIATION_RANGE))
    ref_fov_nm = WORLD_FOV_NM / s

    scene = severity_scene(make_scene_spec(seed, arch, prof, preset=family), sev, rng)
    # Phase 2 raises the sparse-defect population ~3x over the Phase 1 draw:
    # particles/residue give every reference window its own fingerprint
    # against pure lattice translates (real arrays carry process residue),
    # which is what makes the true site beat its periodic aliases.
    scene = replace(scene, defect_density=scene.defect_density * 3.0)
    target_cx, target_cy = sample_target_site(rng, scene, ref_fov_nm, edge)
    scene = replace(scene, target_cx_nm=target_cx, target_cy_nm=target_cy)

    # Decoys are drawn BEFORE the present/absent branch and anchor at this
    # scene's own motif (§3.4): decoy presence must not correlate with
    # `present`, or the rejection head learns absence from the wrong cue.
    # For absent pairs the near-duplicates mirror the search scene's motif,
    # not the (foreign) reference - the true instance is still absent.
    want_decoys = bool(n_decoys is not None) or rng.random() < DECOY_PROBABILITY
    decoys: tuple = ()
    if want_decoys:
        decoys = make_decoy_specs(
            rng, scene, int(n_decoys) if n_decoys is not None else 1 + sev,
            ref_fov_nm, target_cx, target_cy,
        )

    scene_seed = seed
    ref_seed = seed * 7_919 + 101
    search_seed = seed * 1_000_003 + 313
    ref_rng = np.random.default_rng(ref_seed)
    search_rng = np.random.default_rng(search_seed)

    rgb = modality == "rgb"
    chroma = sample_rgb_chroma(rng) if rgb else None
    reflectance = modulated_reflectance(chroma["chroma"], chroma["tint"]) if rgb else None

    # ---- SEARCH: whole world, own acquisition, CD bias applied here ----
    super_size = OUTPUT_SIZE * search_supersample
    search_phys, _ = render_scene_with_layers(
        scene, 0.0, 0.0, WORLD_FOV_NM, super_size,
        cd_bias_frac=cd_bias, decoys=decoys, rgb_reflectance=reflectance,
        cell_dropout=cell_dropout, cell_size_variation=cell_variation,
    )
    search_capture = (
        sample_rgb_capture(search_rng, gray_pair=chroma["gray_pair"]) if rgb else None
    )
    if rgb:
        _apply_channel_shifts(search_phys, search_capture["channel_shifts_px"])

    pixel_nm_super = WORLD_FOV_NM / super_size
    target_size_super = ref_fov_nm / pixel_nm_super
    mask_super = _target_mask(
        super_size, target_cx / pixel_nm_super, target_cy / pixel_nm_super, target_size_super
    )
    search_acq = sample_phase2_acquisition(search_rng, sev, reference=False)
    if rgb:
        search_f, warped_mask = acquire_rgb_search(
            search_phys, pixel_nm_super, search_supersample, search_acq, search_rng,
            mask_super, search_capture,
        )
    else:
        search_f, warped_mask = acquire_search(
            search_phys, pixel_nm_super, search_supersample, search_acq, search_rng, mask_super
        )

    # ---- REFERENCE: same world if present, different realization if absent ----
    absent_scene_seed = None
    if present:
        ref_scene, ref_cx, ref_cy = scene, target_cx, target_cy
    else:
        absent_scene_seed = seed ^ ABSENT_SCENE_XOR
        ref_scene = severity_scene(
            make_scene_spec(absent_scene_seed, arch, prof, preset=family), sev, rng
        )
        ref_cx, ref_cy = sample_target_site(rng, ref_scene, ref_fov_nm, edge)

    ref_origin_x = ref_cx - ref_fov_nm / 2.0
    ref_origin_y = ref_cy - ref_fov_nm / 2.0
    ref_phys, _ = render_scene_with_layers(
        ref_scene, ref_origin_x, ref_origin_y, ref_fov_nm, OUTPUT_SIZE,
        cd_bias_frac=0.0, rgb_reflectance=reflectance,
        cell_dropout=cell_dropout, cell_size_variation=cell_variation,
    )
    ref_capture = (
        sample_rgb_capture(ref_rng, gray_pair=chroma["gray_pair"]) if rgb else None
    )
    if rgb:
        _apply_channel_shifts(ref_phys, ref_capture["channel_shifts_px"])

    pixel_nm_ref = ref_fov_nm / OUTPUT_SIZE
    defect_lambda = float(rng.uniform(*LOCAL_DEFECT_LAMBDA[sev]))
    realized_occlusion = apply_reference_damage(
        ref_phys, pixel_nm_ref, ref_rng, defect_lambda, occlusion_target
    )
    ref_acq = sample_phase2_acquisition(ref_rng, sev, reference=True)
    ref_acq = replace(ref_acq, rotation_deg=ref_acq.rotation_deg + stage_theta)
    if rgb:
        reference_f = acquire_rgb_reference(ref_phys, pixel_nm_ref, ref_acq, ref_rng, ref_capture)
    else:
        reference_f = acquire_reference(ref_phys, pixel_nm_ref, ref_acq, ref_rng)

    # ---- LABELS ----
    # Ground truth is measured, not computed (§2.2/§2.3). The centroid of the
    # mask tracked through the identical warp gives the position; the
    # rotation and zoom labels are brute-force ZNCC readouts at that known
    # location, in a narrow window around the analytic prior. The analytic
    # prior (net_rotation_label) documents the stage convention whose SIGN
    # was calibrated by scripts/verify_conventions.py.
    predicted_theta = None
    if present:
        from .pose import rotation_oracle, scale_oracle
        from .sem import area_downsample

        nominal_search_mask = area_downsample(mask_super, search_supersample)
        gt_x, gt_y = _mask_centroid(warped_mask)
        predicted_theta = net_rotation_label(ref_acq, search_acq)
        # Iterate (theta, scale) to convergence; each scale pass pins its
        # template shape at the current estimate, and a final scale pass at
        # the converged theta removes the one-step lag so the label is the
        # fixed point of exactly the procedure the gates re-run.
        gt_theta, _ = rotation_oracle(
            reference_f, search_f, gt_x, gt_y, s,
            lo=predicted_theta - 1.5, hi=predicted_theta + 1.5, step=0.05,
        )
        if not np.isfinite(gt_theta):
            gt_theta = predicted_theta
        gt_scale = s
        for _ in range(2):
            measured_scale, _ = scale_oracle(
                reference_f, search_f, gt_x, gt_y, gt_theta,
                lo=max(7.5, gt_scale - 0.6), hi=min(12.5, gt_scale + 0.6),
                coarse_step=0.05, fine_step=0.01, shape_scale=gt_scale,
            )
            if np.isfinite(measured_scale):
                gt_scale = measured_scale
            refined_theta, _ = rotation_oracle(
                reference_f, search_f, gt_x, gt_y, gt_scale,
                lo=gt_theta - 0.75, hi=gt_theta + 0.75, step=0.05,
            )
            if np.isfinite(refined_theta):
                gt_theta = refined_theta
        final_scale, _ = scale_oracle(
            reference_f, search_f, gt_x, gt_y, gt_theta,
            lo=max(7.5, gt_scale - 0.3), hi=min(12.5, gt_scale + 0.3),
            coarse_step=0.05, fine_step=0.01, shape_scale=gt_scale,
        )
        if np.isfinite(final_scale):
            gt_scale = final_scale
        stretch = s / gt_scale if gt_scale else 1.0
    else:
        gt_x = gt_y = gt_theta = None
        gt_scale = s
        stretch = 1.0

    reference = to_uint8(reference_f)
    search = to_uint8(search_f)

    metadata = {
        "scene_seed": scene_seed,
        "ref_seed": ref_seed,
        "search_seed": search_seed,
        "preset_family": family,
        "profile": prof,
        "severity": sev,
        "modality": modality,
        "present": int(present),
        "gt_scale": gt_scale,
        "n_decoys": len(decoys),
        "decoy_sites": [
            [round(d.cx_nm / 10.0, 4), round(d.cy_nm / 10.0, 4)] for d in decoys
        ],
        "occlusion_frac": round(realized_occlusion, 6),
        "occlusion_target_frac": occlusion_target,
        "cd_bias_pct": round(cd_bias * 100.0, 4),
        "cell_dropout_frac": round(cell_dropout, 4),
        "cell_size_variation_frac": round(cell_variation, 4),
        "edge_case": edge,
        "absent_scene_seed": absent_scene_seed,
        "ref_fov_nm": ref_fov_nm,
        "nominal_zoom": s,
        "scale_area_stretch": round(stretch, 6),
        "predicted_theta_deg": None if predicted_theta is None else round(predicted_theta, 4),
        "stage_theta_deg": stage_theta,
        "scene": scene.to_dict(),
        "ref_acquisition": ref_acq.to_dict("ref_"),
        "search_acquisition": search_acq.to_dict("search_"),
    }
    if rgb:
        from .phase2 import channel_spread

        metadata["rgb_gray_pair"] = chroma["gray_pair"]
        metadata["rgb_chroma"] = chroma["chroma"]
        metadata["rgb_channel_spread_ref"] = round(channel_spread(reference), 4)
        metadata["rgb_channel_spread_search"] = round(channel_spread(search), 4)

    return Phase2Sample(
        reference=reference,
        search=search,
        present=present,
        gt_x=gt_x,
        gt_y=gt_y,
        gt_theta=gt_theta,
        gt_scale=gt_scale,
        severity=sev,
        modality=modality,
        architecture=arch,
        preset_family=family,
        n_decoys=len(decoys),
        metadata=metadata,
    )

