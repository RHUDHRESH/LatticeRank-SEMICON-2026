from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
from PIL import Image

from driftforge.model import MODEL_FEATURES, ModelCompatibilityError
from driftforge.pipeline import LocateV2Result
from scripts import inference


def _write_pair(directory: Path) -> tuple[Path, Path]:
    reference = directory / "reference.png"
    search = directory / "search.png"
    Image.fromarray(np.full((1000, 1000, 3), 96, dtype=np.uint8), "RGB").save(
        reference
    )
    Image.fromarray(np.full((1000, 1000), 112, dtype=np.uint8), "L").save(search)
    return reference, search


def _mock_pipeline(monkeypatch, *, x: float = 12.345, y: float = 67.891) -> None:
    monkeypatch.setattr(
        inference,
        "load_model_bundle",
        lambda _path: {
            "model": object(),
            "features": list(MODEL_FEATURES),
            "metadata": {"format_version": 1},
        },
    )
    monkeypatch.setattr(
        inference,
        "locate_v2",
        lambda *_args, **_kwargs: LocateV2Result(
            x=x,
            y=y,
            probability=0.75,
            eq_set_size=2,
            n_candidates=12,
            used_residual=True,
            diagnostics={"score_max": 1.25},
        ),
    )


def test_normal_stdout_is_exact_coordinate_tuple(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    reference, search = _write_pair(tmp_path)
    _mock_pipeline(monkeypatch)
    assert inference.main([str(reference), str(search)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "(12.35, 67.89)\n"
    assert captured.err == ""


def test_json_mode_retains_diagnostics(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    reference, search = _write_pair(tmp_path)
    _mock_pipeline(monkeypatch)
    assert inference.main([str(reference), str(search), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["x"] == 12.345
    assert payload["score_max"] == 1.25
    assert payload["model"]["feature_count"] == len(MODEL_FEATURES)
    assert payload["model"]["path"] == "driftforge/models/hgb_r2.joblib"
    assert len(payload["model"]["sha256"]) == 64
    assert captured.err == ""


def test_input_and_model_errors_use_stderr_only(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    missing = tmp_path / "missing.png"
    assert inference.main([str(missing), str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "not found" in captured.err

    reference, search = _write_pair(tmp_path)
    monkeypatch.setattr(
        inference,
        "load_model_bundle",
        lambda _path: (_ for _ in ()).throw(
            ModelCompatibilityError("incompatible feature schema")
        ),
    )
    assert inference.main([str(reference), str(search)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "incompatible feature schema" in captured.err


def test_inference_never_reads_ground_truth_files(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    reference, search = _write_pair(tmp_path)
    (tmp_path / "labels.csv").write_text("secret\n", encoding="utf-8")
    (tmp_path / "ground_truth.json").write_text("{}\n", encoding="utf-8")
    _mock_pipeline(monkeypatch)

    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if Path(file).name in {"labels.csv", "ground_truth.json"}:
            raise AssertionError("inference attempted to read ground truth")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    assert inference.main([str(reference), str(search)]) == 0
    assert capsys.readouterr().out == "(12.35, 67.89)\n"


def test_invalid_dimensions_are_rejected_before_model_load(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    reference = tmp_path / "small.png"
    search = tmp_path / "search.png"
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(reference)
    Image.fromarray(np.zeros((1000, 1000), dtype=np.uint8)).save(search)
    called = False

    def unexpected_model_load(_path):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(inference, "load_model_bundle", unexpected_model_load)
    assert inference.main([str(reference), str(search)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "must be 1000x1000" in captured.err
    assert called is False
