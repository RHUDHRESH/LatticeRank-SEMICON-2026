#!/usr/bin/env python3
"""Read-only, cross-platform release smoke check for a judge's fresh clone."""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.model import load_model_bundle, model_file_provenance, package_versions

COORDINATE = re.compile(r"^\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)$")


def main() -> int:
    print("LatticeRank 1.0 | judge smoke check")
    print(f"Python: {sys.version.split()[0]} | executable: {sys.executable}")
    if sys.version_info < (3, 12):
        raise RuntimeError("Python 3.12 or newer is required by the pinned release")

    bundle = load_model_bundle()
    provenance = model_file_provenance()
    print(
        f"Model: {provenance['path']} | {len(bundle['features'])} features | "
        f"sha256 {provenance['sha256'][:16]}..."
    )
    versions = package_versions()
    print(
        "Runtime: "
        + " | ".join(
            f"{name} {versions[name]}"
            for name in ("numpy", "scipy", "scikit-learn", "joblib", "Pillow")
        )
    )

    inference = PROJECT / "scripts" / "inference.py"
    for architecture in ("dram", "finfet"):
        folder = PROJECT / "examples" / architecture
        truth = json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))
        completed = subprocess.run(
            [
                sys.executable,
                str(inference),
                str(folder / "reference.png"),
                str(folder / "search.png"),
            ],
            cwd=tempfile.gettempdir(),
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = completed.stdout.strip()
        match = COORDINATE.fullmatch(output)
        if match is None:
            raise RuntimeError(
                f"{architecture} output contract failed; received {output!r}"
            )
        x, y = map(float, match.groups())
        if not (0.0 <= x <= 999.0 and 0.0 <= y <= 999.0):
            raise RuntimeError(f"{architecture} returned an out-of-bounds coordinate")
        error = math.hypot(x - float(truth["x"]), y - float(truth["y"]))
        print(
            f"{architecture.upper()}: {output} | packaged-example error {error:.2f} px"
        )

    print("PASS | model loads, both architectures run outside the repository cwd, and stdout is one coordinate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
