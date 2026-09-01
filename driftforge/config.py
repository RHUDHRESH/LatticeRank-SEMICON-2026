from __future__ import annotations

from dataclasses import asdict, dataclass


WORLD_FOV_NM = 10_000.0
REFERENCE_FOV_NM = 1_000.0
OUTPUT_SIZE = 1_000


@dataclass(frozen=True)
class SceneSpec:
    architecture: str
    preset: str
    seed: int
    phase_x_nm: float
    phase_y_nm: float
    pitch_x_nm: float
    pitch_y_nm: float
    width_x_nm: float
    width_y_nm: float
    contact_x_nm: float
    contact_y_nm: float
    line_jitter_nm: float
    width_jitter_fraction: float
    roughness_nm: float
    roughness_lambda_nm: float
    mat_period_nm: float
    strip_width_nm: float
    zone_offset_x_nm: float
    zone_offset_y_nm: float
    target_cx_nm: float
    target_cy_nm: float
    defect_density: float
    profile: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AcquisitionSpec:
    edge_strength: float
    edge_sigma_nm: float
    psf_sigma_x_nm: float
    psf_sigma_y_nm: float
    rotation_deg: float
    scale: float
    translate_x_px: float
    translate_y_px: float
    scan_shear_px: float
    scan_jitter_px: float
    radial_k: float
    dose: float
    read_noise_sigma: float
    gain: float
    offset: float
    gamma: float
    vignette: float
    charging_strength: float
    charging_streaks: int
    hot_pixel_rate: float

    def to_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


# Geometry ranges are deliberately families, not claims that every fab uses one
# exact pitch. Sources and the intended interpretation are documented in
# docs/REFERENCES.md.
PRESETS = {
    "dram_dense": dict(pitch_x=(48, 72), pitch_y=(72, 110), width_x=(15, 28), width_y=(20, 38), contact=(18, 34)),
    "dram_open": dict(pitch_x=(75, 120), pitch_y=(100, 170), width_x=(22, 42), width_y=(28, 55), contact=(25, 48)),
    "finfet_dense": dict(pitch_x=(28, 50), pitch_y=(48, 78), width_x=(7, 18), width_y=(16, 32), contact=(15, 28)),
    "finfet_open": dict(pitch_x=(50, 90), pitch_y=(75, 130), width_x=(12, 28), width_y=(22, 45), contact=(20, 40)),
}


PROFILE_MIX = {
    "train": [("standard", 0.55), ("hard", 0.25), ("boundary", 0.12), ("ambiguous", 0.08)],
    "validation": [("standard", 0.50), ("hard", 0.25), ("boundary", 0.15), ("ambiguous", 0.10)],
    "test": [("standard", 0.35), ("hard", 0.30), ("boundary", 0.15), ("ambiguous", 0.10), ("ood", 0.10)],
    "stress": [("hard", 0.25), ("boundary", 0.20), ("ambiguous", 0.25), ("ood", 0.30)],
    # Extended leak-free validation set. It mirrors the validation mix but is
    # a separate, scene-disjoint split rather than a re-slice.
    "validation_benchmark": [("standard", 0.50), ("hard", 0.25), ("boundary", 0.15), ("ambiguous", 0.10)],
    # Phase 2 scene-structure mixes (driftforge.phase2.PROFILE_MIX_P2); the
    # severity ladder, not the profile, owns acquisition difficulty there.
    "p2_train": [("standard", 0.62), ("hard", 0.18), ("boundary", 0.10), ("ambiguous", 0.10)],
    "p2_val": [("standard", 0.62), ("hard", 0.18), ("boundary", 0.10), ("ambiguous", 0.10)],
    "p2_test": [("standard", 0.62), ("hard", 0.18), ("boundary", 0.10), ("ambiguous", 0.10)],
    "p2_holdout_fam": [("standard", 0.62), ("hard", 0.18), ("boundary", 0.10), ("ambiguous", 0.10)],
    "p2_stress": [("standard", 0.62), ("hard", 0.18), ("boundary", 0.10), ("ambiguous", 0.10)],
    "p2_val_rgb": [("standard", 0.62), ("hard", 0.18), ("boundary", 0.10), ("ambiguous", 0.10)],
    "p2_bulk": [("standard", 0.62), ("hard", 0.18), ("boundary", 0.10), ("ambiguous", 0.10)],
}

