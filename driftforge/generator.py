from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OUTPUT_SIZE, REFERENCE_FOV_NM, WORLD_FOV_NM, AcquisitionSpec, SceneSpec
from .scene import make_scene_spec, render_scene
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

