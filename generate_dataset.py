#!/usr/bin/env python3
"""Documented dataset generator required by the Phase 2 submission zip.

The addendum lists ``generate_dataset.py`` alongside ``register.py``,
``requirements.txt``, and ``failure_analysis.pdf``. This file is the zip-root
entry point so a judge can run it without knowing the internal layout.

The implementation lives in ``scripts/generate_dataset.py``. Both of these
are equivalent:

    python generate_dataset.py --phase 2 --split p2_val --count 20 \\
        --output-dir generated/phase2 --modality gray --seed-base 20260827

    python scripts/generate_dataset.py --phase 2 --split p2_val --count 20 \\
        --output-dir generated/phase2 --modality gray --seed-base 20260827

See README.md and docs/HOW_TO_RUN.md for the full command set, and
docs/REFERENCES.md for the cited SEM and layout sources behind every
parameter family.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "generate_dataset.py"
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
