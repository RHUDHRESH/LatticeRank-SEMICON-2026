from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .generator import generate_sample
from .splits import read_manifest


def gaussian_heatmap(x: float, y: float, size: int = 250, stride: int = 4, sigma_px: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = x / stride
    cy = y / stride
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_px ** 2)).astype(np.float32)


def model_arrays(sample) -> dict[str, np.ndarray]:
    reference = sample.reference.astype(np.float32) / 255.0
    search = sample.search.astype(np.float32) / 255.0
    # This 100x100 view is the correct physical scale for correlation heads;
    # the full-resolution reference is retained for a learnable encoder.
    reference_at_search_scale = ndimage.zoom(reference, 0.1, order=1, prefilter=False).astype(np.float32)
    return {
        "reference": reference[None],
        "reference_at_search_scale": reference_at_search_scale[None],
        "search": search[None],
        "target_xy": np.asarray([sample.gt_x, sample.gt_y], dtype=np.float32),
        "target_xy_normalized": np.asarray([sample.gt_x / 999.0, sample.gt_y / 999.0], dtype=np.float32),
        "target_heatmap_stride4": gaussian_heatmap(sample.gt_x, sample.gt_y)[None],
    }


class OnTheFlyPairDataset:
    """Framework-neutral lazy dataset; optionally converts arrays to Torch.

    Rendering on access avoids a multi-gigabyte checked-in dataset while the
    immutable manifest makes every pair reproducible. Use num_workers and a
    local cache for training-scale throughput.
    """

    def __init__(self, manifest: str | Path, supersample: int = 1, as_torch: bool = False):
        self.records = read_manifest(manifest)
        self.supersample = supersample
        self.as_torch = as_torch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        sample = generate_sample(
            seed=int(row["seed"]), architecture=row["architecture"], profile=row["profile"],
            search_supersample=self.supersample,
        )
        result = model_arrays(sample)
        result["id"] = row["id"]
        result["architecture"] = row["architecture"]
        result["profile"] = row["profile"]
        if self.as_torch:
            try:
                import torch
            except ImportError as exc:
                raise ImportError("Install PyTorch separately to use as_torch=True") from exc
            result = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in result.items()}
        return result

