"""The shared per-scene context every physical-evidence module reads.

Building this is the expensive part of a scene -- the pose-correct template,
the lattice basis, the periodic residuals of both sides. Fifteen experiments
that each rebuilt it would pay fifteen times for one result, so it is built
once per pair and passed to every extractor.

Extractor modules conform to a three-part contract::

    FEATURES: list[str]                  # column names, stable and prefixed
    def build(ctx) -> state | None       # per-scene precompute; None = abstain
    def score(state, x, y) -> dict       # per-candidate, keys exactly FEATURES

``build`` returning ``None`` means the module cannot speak about this scene
(no usable lattice, no detectable edges); the harness then fills its columns
with NaN rather than a fabricated value, because the learned ranker handles
missing evidence natively and a zero would be a lie about the measurement.

Coordinates are always **search-image pixels**, and ``(x, y)`` is the centre of
the candidate patch, matching the convention ``harvest`` produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SceneContext:
    """Everything an extractor may need, computed once per (reference, search)."""

    reference: np.ndarray          # original high-magnification reference, uint8
    search: np.ndarray             # original search image, uint8
    search_f: np.ndarray           # robust-contrast search, float32
    template: np.ndarray           # pose-correct template at (scale, rotation)
    scale: float                   # down-scaling factor s in [8, 12]
    rotation: float                # template rotation, degrees
    v1: np.ndarray                 # lattice basis vector 1, search pixels
    v2: np.ndarray                 # lattice basis vector 2, search pixels
    lattice_ok: bool
    t_res: np.ndarray              # periodic residual of the template
    t_unique: np.ndarray           # per-pixel uniqueness (std of shift stack)
    s_res: np.ndarray              # periodic residual of the search image
    margin: int                    # template border the shift stack cannot fill
    extras: dict = field(default_factory=dict)

    @property
    def half_w(self) -> float:
        return (self.template.shape[1] - 1) / 2.0

    @property
    def half_h(self) -> float:
        return (self.template.shape[0] - 1) / 2.0

    def window(self, image: np.ndarray, x: float, y: float,
               shape: tuple[int, int] | None = None) -> np.ndarray | None:
        """The patch of ``image`` centred on ``(x, y)``, or None if clipped.

        Returning None rather than a reflected or zero-padded patch is
        deliberate: an edge candidate scored against invented pixels produces a
        confident number about data that was never acquired.
        """
        h, w = (self.template.shape if shape is None else shape)
        top = int(round(y)) - h // 2
        left = int(round(x)) - w // 2
        if top < 0 or left < 0 or top + h > image.shape[0] or left + w > image.shape[1]:
            return None
        return image[top:top + h, left:left + w]

    def lattice_offsets(self, rings: int = 1) -> list[np.ndarray]:
        """Displacement vectors to the lattice siblings, in search pixels."""
        out: list[np.ndarray] = []
        for m in range(-rings, rings + 1):
            for n in range(-rings, rings + 1):
                if m == 0 and n == 0:
                    continue
                if max(abs(m), abs(n)) > rings:
                    continue
                out.append(m * self.v1 + n * self.v2)
        return out


def build_context(reference: np.ndarray, search: np.ndarray, *,
                  scale: float, rotation: float) -> SceneContext:
    """Assemble the context. ``scale`` is the down-scaling factor, not a zoom."""
    import math

    from ..baseline import _robust_contrast, _template_from_reference
    from ..lattice import estimate_lattice
    from ..residual import SHIFT_SET, periodic_residual

    search_f = _robust_contrast(search)
    # template_scale convention: internal zoom is 0.1 * template_scale, so a
    # down-scaling factor s needs template_scale = 10 / s.
    template = _template_from_reference(reference, 10.0 / scale, rotation)

    lat = estimate_lattice(search)
    B = lat.basis
    v1, v2 = B[:, 0].copy(), B[:, 1].copy()
    ok = bool(abs(float(np.linalg.det(B))) > 1.0
              and np.linalg.norm(v1) >= 2.0 and np.linalg.norm(v2) >= 2.0)

    if ok:
        t_res, t_unique = periodic_residual(template, v1, v2)
        s_res, _ = periodic_residual(search_f, v1, v2)
        margin = int(math.ceil(max(abs(m * v1[i] + n * v2[i])
                                   for m, n in SHIFT_SET for i in (0, 1)))) + 1
    else:
        t_res = np.zeros_like(template, dtype=np.float32)
        t_unique = np.zeros_like(template, dtype=np.float32)
        s_res = np.zeros_like(search_f, dtype=np.float32)
        margin = 0

    return SceneContext(
        reference=reference, search=search, search_f=search_f,
        template=template, scale=scale, rotation=rotation,
        v1=v1, v2=v2, lattice_ok=ok,
        t_res=t_res, t_unique=t_unique, s_res=s_res, margin=margin,
    )
