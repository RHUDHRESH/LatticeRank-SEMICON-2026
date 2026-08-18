from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from driftforge.model import MODEL_PATH


def test_built_wheel_installs_with_loadable_model(tmp_path: Path) -> None:
    """Build outside the source tree, inspect package data, then install it."""
    project = Path(__file__).resolve().parents[1]
    staging = tmp_path / "source"
    staging.mkdir()
    shutil.copy2(project / "pyproject.toml", staging / "pyproject.toml")
    shutil.copytree(
        project / "driftforge",
        staging / "driftforge",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(staging),
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    member = "driftforge/models/hgb_r2.joblib"
    with zipfile.ZipFile(wheel) as archive:
        assert member in archive.namelist()
        wheel_digest = hashlib.sha256(archive.read(member)).hexdigest()
    assert wheel_digest == hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    installed = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(wheel),
            "--no-deps",
            "--target",
            str(installed),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    check_code = (
        "import sys; "
        f"sys.path.insert(0, {str(installed)!r}); "
        "from driftforge.model import MODEL_PATH, load_model_bundle; "
        "assert MODEL_PATH.is_file(); "
        "bundle = load_model_bundle(); "
        "assert bundle['model'].n_features_in_ == len(bundle['features'])"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", check_code],
        check=True,
        capture_output=True,
        text=True,
    )
