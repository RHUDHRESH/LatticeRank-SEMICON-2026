#!/usr/bin/env python3
"""Materialize paired synthetic SEM images and their coordinate labels.

This is the documented generator required by the submission rules. It writes
image pairs, a ground-truth manifest, a coverage report and a citation record
into one output directory, and it is deterministic: the same seed produces
byte-identical images on any machine and at any worker count.

Phase 1 (``--phase 1``, the default) produces the original fixed-10x,
always-present pairs. **Phase 2** (``--phase 2``, implied by ``--split``) adds
the three assumptions the Phase 2 addendum removes:

* **Unknown zoom** ``s`` drawn uniformly from ``[8, 12]``. The zoom is produced
  by the reference *field of view* -- the scene is re-rendered at a narrower
  FOV -- and never by resizing a 10x render, so the reference carries the
  detail a real high-magnification acquisition would have.
* **Unknown stage rotation** of up to +/-5 degrees, stacked on the ordinary
  acquisition jitter. The sign convention is verified empirically by
  ``scripts/verify_conventions.py``; the naive ``search - reference`` formula
  has the wrong sign under this codebase's template conventions.
* **Absent pairs** (20% by default, ``--present-frac``): the reference comes
  from a different structural realization of the same architecture and preset
  family, so it is periodically similar and plausible but genuinely not
  present. The correct answer for these is ``found = 0``.

On top of those, Phase 2 applies a four-level severity ladder (dose, noise,
PSF, charging, scan geometry, photometry, roughness, CD bias and reference
damage), same-architecture decoys on 40% of present pairs, and an RGB optical
mode (``--modality rgb``) with per-channel gain, colour cast and 0.3-1.2 px
chromatic misregistration.

Ground truth is **measured, not derived**: position comes from the target mask
tracked through the identical search warp, and rotation and zoom are
brute-force ZNCC readouts at that known location. Twelve validation gates in
``scripts/validate_phase2.py`` cover oracle recovery, leakage probes,
marginal KS tests and byte-identical regeneration.

Examples
--------
Phase 1, 30 DRAM pairs::

    python scripts/generate_dataset.py --architecture DRAM --count 30 \\
        --output-dir generated/dram

Phase 2 validation split, 400 grayscale pairs::

    python scripts/generate_dataset.py --phase 2 --split p2_val --count 400 \\
        --output-dir data/phase2/p2_val --modality gray --seed-base 1300000

Phase 2 RGB optical split::

    python scripts/generate_dataset.py --phase 2 --split p2_val_rgb --count 200 \\
        --output-dir data/phase2/p2_val_rgb --modality rgb

A resumable bulk shard (see ``docs/BULK_DATASET.md``)::

    python scripts/generate_dataset.py --phase 2 --split p2_bulk --count 5000 \\
        --start-index 30000 --output-dir data/phase2_bulk/shard_00006 \\
        --modality rgb --workers 7 --fast-png

Notes
-----
``--fast-png`` trades roughly 15% more disk for a substantially faster encode
and is recorded in ``DATASET_INFO.json`` so the regeneration gate compares
like with like. ``--hide-labels`` writes a public split with no ground truth.
Parameter ranges and their sources are documented in ``docs/REFERENCES.md``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge import __version__
from driftforge.generator import (
    PHASE2_SPLITS,
    PROFILES,
    generate_phase2_sample,
    generate_sample,
    normalize_architecture,
    normalize_profile,
)
from driftforge.phase2 import build_citations, build_coverage_report
from driftforge.splits import SPLIT_SEED_BASE, read_manifest


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _save_png_atomic(array, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(array).save(temporary, format="PNG", optimize=True)
    # Decode/CRC verification happens before the atomic rename. A partial file
    # can therefore never masquerade as a completed dataset item.
    with Image.open(temporary) as image:
        image.verify()
    temporary.replace(path)


def _architecture_arg(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized == "both":
        return normalized
    try:
        return normalize_architecture(normalized)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("choose DRAM, FinFET, or both") from exc


def _profile_arg(value: str) -> str:
    try:
        return normalize_profile(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "choose one of: " + ", ".join(PROFILES)
        ) from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _validate_records(records: list[dict]) -> list[dict]:
    required = ("id", "split", "seed", "scene_id", "architecture", "profile")
    normalized: list[dict] = []
    ids: set[str] = set()
    scenes: set[str] = set()
    for index, source in enumerate(records):
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(
                f"record {index} is missing required fields: {', '.join(missing)}"
            )
        record = dict(source)
        sample_id = str(record["id"])
        if (
            not sample_id
            or sample_id in {".", ".."}
            or Path(sample_id).name != sample_id
        ):
            raise ValueError(f"record {index} has unsafe id {sample_id!r}")
        if sample_id in ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        scene_id = str(record["scene_id"])
        if not scene_id or scene_id in scenes:
            raise ValueError(f"duplicate or empty scene id: {scene_id!r}")
        try:
            record["seed"] = int(record["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record {sample_id} has an invalid seed") from exc
        if record["seed"] < 0:
            raise ValueError(f"record {sample_id} has a negative seed")
        record["id"] = sample_id
        record["scene_id"] = scene_id
        record["architecture"] = normalize_architecture(record["architecture"])
        record["profile"] = normalize_profile(record["profile"])
        record["split"] = str(record["split"])
        ids.add(sample_id)
        scenes.add(scene_id)
        normalized.append(record)
    return normalized


def _records_from_args(args: argparse.Namespace) -> list[dict]:
    if args.manifest:
        return _validate_records(read_manifest(args.manifest))
    architectures = [args.architecture] if args.architecture != "both" else ["dram", "finfet"]
    return _validate_records([
        {
            "id": f"custom-{i:06d}",
            "split": "custom",
            "seed": args.seed_start + i,
            "scene_id": f"scene-{args.seed_start + i}",
            "architecture": architectures[i % len(architectures)],
            "profile": args.profile,
        }
        for i in range(args.count)
    ])


def _seed_provenance(args: argparse.Namespace, records: list[dict]) -> dict:
    if args.manifest:
        digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        source = {
            "kind": "manifest",
            "manifest": args.manifest.name,
            "manifest_sha256": digest,
        }
    else:
        source = {
            "kind": "contiguous_seed_range",
            "seed_start": args.seed_start,
            "requested_count": args.count,
        }
    source.update(
        {
            "slice_start": args.start,
            "slice_limit": args.limit,
            "selected_ids": [record["id"] for record in records],
            "selected_seeds": [int(record["seed"]) for record in records],
        }
    )
    return source


def _ensure_output_is_safe(
    output_dir: Path,
    records: list[dict],
    *,
    hide_labels: bool,
    overwrite: bool,
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    if hide_labels and output_dir.exists():
        leaked = [
            path
            for path in (
                output_dir / "labels.csv",
                output_dir / "ground_truth.json",
                output_dir / "metadata",
            )
            if path.exists()
        ]
        if leaked:
            raise ValueError(
                "refusing to create a hidden-label split over existing ground truth; "
                "use an empty output directory"
            )
    output_ids = [
        f"public-{index:06d}" if hide_labels else record["id"]
        for index, record in enumerate(records)
    ]
    if overwrite:
        selected = set(output_ids)
        generated_files = (
            list((output_dir / "reference").glob("*.png"))
            + list((output_dir / "search").glob("*.png"))
            + list((output_dir / "metadata").glob("*.json"))
        )
        stale = [
            path
            for path in generated_files
            if path.stem not in selected
        ]
        if stale:
            raise ValueError(
                "refusing to leave stale samples while overwriting; use an empty "
                f"output directory (first stale file: {stale[0]})"
            )
        return
    targets = [output_dir / "DATASET_INFO.json"]
    if not hide_labels:
        targets.extend((output_dir / "labels.csv", output_dir / "ground_truth.json"))
    for sample_id in output_ids:
        targets.extend(
            (
                output_dir / "reference" / f"{sample_id}.png",
                output_dir / "search" / f"{sample_id}.png",
            )
        )
        if not hide_labels:
            targets.append(output_dir / "metadata" / f"{sample_id}.json")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace them: "
            + ", ".join(existing[:3])
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize paired synthetic SEM images and coordinate labels.")
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 = Phase 1 fixed-10x pairs (default), 2 = Phase 2 unknown-zoom/rotation pairs",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="JSONL manifest made by make_manifests.py")
    source.add_argument(
        "--count",
        type=_positive_int,
        default=30,
        help="Number of custom pairs when no manifest is supplied",
    )
    parser.add_argument(
        "--architecture",
        type=_architecture_arg,
        default="both",
        metavar="{DRAM,FinFET,both}",
    )
    parser.add_argument(
        "--profile",
        type=_profile_arg,
        default="standard",
        metavar="{standard,hard,boundary,ambiguous,ood}",
    )
    parser.add_argument("--seed-start", type=_nonnegative_int, default=900_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-supersample", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--start", type=_nonnegative_int, default=0)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--hide-labels", action="store_true", help="Create a public test folder without ground-truth files")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace matching files in an existing labelled output directory",
    )
    # ---- Phase 2 options ----
    parser.add_argument("--split", choices=PHASE2_SPLITS, help="Phase 2 split name (implies --phase 2)")
    parser.add_argument("--modality", choices=("gray", "rgb"), default="gray")
    parser.add_argument("--severity", type=int, choices=(0, 1, 2, 3), default=None, help="Fix the severity level instead of sampling the split mix")
    parser.add_argument("--present-frac", type=float, default=0.8, help="Fraction of pairs that contain the true instance")
    parser.add_argument("--seed-base", type=_nonnegative_int, default=None, help="Phase 2 scene-seed base (defaults to the split's registered base)")
    parser.add_argument("--export-debug", action="store_true", help="Store per-pair acquisition diagnostics alongside the manifest")
    parser.add_argument("--workers", type=_positive_int, default=1, help="Parallel worker processes (output is deterministic regardless)")
    parser.add_argument("--start-index", type=_nonnegative_int, default=0, help="Phase 2: global index of the first pair (resumable shards; seeds and filenames offset by it)")
    parser.add_argument("--fast-png", action="store_true", help="Phase 2: fast PNG encoding (compress_level=1 instead of optimize) for bulk production")
    return parser


def run(args: argparse.Namespace) -> None:
    if args.manifest and not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")

    records = _records_from_args(args)[args.start:]
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise ValueError("the requested manifest slice contains no records")
    _ensure_output_is_safe(
        args.output_dir,
        records,
        hide_labels=args.hide_labels,
        overwrite=args.overwrite,
    )
    refs = args.output_dir / "reference"
    searches = args.output_dir / "search"
    metadata_dir = args.output_dir / "metadata"
    refs.mkdir(parents=True, exist_ok=True)
    searches.mkdir(parents=True, exist_ok=True)
    if not args.hide_labels:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict] = []
    ground_truth: list[dict] = []
    for index, record in enumerate(records, start=1):
        sample = generate_sample(
            seed=int(record["seed"]),
            architecture=record["architecture"],
            profile=record["profile"],
            search_supersample=args.search_supersample,
        )
        sample_id = (
            f"public-{index - 1:06d}" if args.hide_labels else record["id"]
        )
        _save_png_atomic(sample.reference, refs / f"{sample_id}.png")
        _save_png_atomic(sample.search, searches / f"{sample_id}.png")
        if not args.hide_labels:
            meta = sample.metadata()
            meta["id"] = sample_id
            meta["split"] = record["split"]
            meta["scene_id"] = record["scene_id"]
            _write_text_atomic(
                metadata_dir / f"{sample_id}.json",
                json.dumps(meta, indent=2, sort_keys=True) + "\n",
            )
            labels.append({
                "id": sample_id,
                "reference": f"reference/{sample_id}.png",
                "search": f"search/{sample_id}.png",
                "x": f"{sample.gt_x:.6f}",
                "y": f"{sample.gt_y:.6f}",
                "architecture": sample.architecture,
                "profile": sample.profile,
                "scene_id": record["scene_id"],
                "seed": str(sample.seed),
            })
            ground_truth.append(
                {
                    "id": sample_id,
                    "x": sample.gt_x,
                    "y": sample.gt_y,
                    "architecture": sample.architecture,
                    "profile": sample.profile,
                    "scene_id": record["scene_id"],
                    "seed": sample.seed,
                }
            )
        print(f"[{index}/{len(records)}] {sample_id}", file=sys.stderr, flush=True)

    if not args.hide_labels:
        labels_temporary = args.output_dir / "labels.csv.tmp"
        with labels_temporary.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "id",
                "reference",
                "search",
                "x",
                "y",
                "architecture",
                "profile",
                "scene_id",
                "seed",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(labels)
        labels_temporary.replace(args.output_dir / "labels.csv")
        _write_text_atomic(
            args.output_dir / "ground_truth.json",
            json.dumps(
                {
                    "coordinate_convention": (
                        "x=column, y=row, origin at top-left, units=Search pixels"
                    ),
                    "samples": ground_truth,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    architectures = Counter(record["architecture"] for record in records)
    dataset_info = {
            "schema_version": 1,
            "generator_version": __version__,
            "count": len(records),
            "labels_included": not args.hide_labels,
            "search_supersample": args.search_supersample,
            "coordinate_convention": "x=column, y=row, origin at top-left, units=search pixels",
            "architectures": dict(sorted(architectures.items())),
        }
    if not args.hide_labels:
        dataset_info["seed_provenance"] = _seed_provenance(args, records)
    _write_text_atomic(
        args.output_dir / "DATASET_INFO.json",
        json.dumps(dataset_info, indent=2, sort_keys=True) + "\n",
    )


# ---------------------------------------------------------------------------
# Phase 2 dataset materialization
# ---------------------------------------------------------------------------

PHASE2_MANIFEST_FIELDS = (
    "id", "split", "scene_seed", "ref_seed", "search_seed", "architecture",
    "preset_family", "severity", "modality", "present", "gt_x", "gt_y",
    "gt_theta", "gt_scale", "n_decoys", "decoy_sites", "occlusion_frac",
    "cd_bias_pct", "edge_case", "ref_image", "search_image",
)


def _png_bytes(array: np.ndarray, fast: bool = False) -> bytes:
    buffer = io.BytesIO()
    if fast:
        Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="PNG", compress_level=1)
    else:
        Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _phase2_record(index: int, split: str, sample, export_debug: bool, id_width: int = 6) -> dict:
    meta = sample.metadata
    # Flagship splits keep the §4 filename rule ({index:05d}); bulk shards
    # with global indices use 7 digits in both ids and filenames.
    path_width = 7 if id_width == 7 else 5
    record = {
        "id": f"{split}-{index:0{id_width}d}",
        "split": split,
        "scene_seed": int(meta["scene_seed"]),
        "ref_seed": int(meta["ref_seed"]),
        "search_seed": int(meta["search_seed"]),
        "architecture": sample.architecture,
        "preset_family": sample.preset_family,
        "severity": int(sample.severity),
        "modality": sample.modality,
        "present": int(sample.present),
        "gt_x": None if sample.gt_x is None else round(float(sample.gt_x), 4),
        "gt_y": None if sample.gt_y is None else round(float(sample.gt_y), 4),
        "gt_theta": None if sample.gt_theta is None else round(float(sample.gt_theta), 4),
        "gt_scale": round(float(sample.gt_scale), 4),
        "n_decoys": int(sample.n_decoys),
        "decoy_sites": meta["decoy_sites"],
        "occlusion_frac": meta["occlusion_frac"],
        "cd_bias_pct": meta["cd_bias_pct"],
        "edge_case": meta["edge_case"],
        "ref_image": f"images/{index:0{path_width}d}_ref.png",
        "search_image": f"images/{index:0{path_width}d}_search.png",
    }
    if export_debug:
        record["diagnostics"] = meta
    return record


def _phase2_worker(task: tuple) -> tuple:
    """Generate one pair; returns (index, ref_png, search_png, record).

    Runs in a worker process when --workers > 1. Every input is picklable and
    the pair depends only on its seed, so output is byte-identical to the
    single-process path.
    """
    (index, seed, split, modality, severity, present_frac,
     search_supersample, export_debug, fast_png, id_width) = task
    sample = generate_phase2_sample(
        seed,
        split=split,
        modality=modality,
        severity=severity,
        present_frac=present_frac,
        search_supersample=search_supersample,
    )
    return (
        index,
        _png_bytes(sample.reference, fast=fast_png),
        _png_bytes(sample.search, fast=fast_png),
        _phase2_record(index, split, sample, export_debug, id_width=id_width),
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def run_phase2(args: argparse.Namespace) -> None:
    seed_base = args.seed_base if args.seed_base is not None else SPLIT_SEED_BASE[args.split]
    start_index = args.start_index
    id_width = 7 if args.start_index > 0 or args.split == "p2_bulk" else 6
    output_dir: Path = args.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def image_stem(index: int) -> str:
        return f"{index:07d}" if id_width == 7 else f"{index:05d}"

    tasks = [
        (
            start_index + index,
            seed_base + start_index + index,
            args.split,
            args.modality,
            args.severity,
            args.present_frac,
            args.search_supersample,
            args.export_debug,
            args.fast_png,
            id_width,
        )
        for index in range(args.count)
    ]

    records: list[dict] = [None] * args.count  # type: ignore[list-item]
    if args.workers > 1:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=args.workers) as pool:
            for done, (index, ref_png, search_png, record) in enumerate(
                pool.imap_unordered(_phase2_worker, tasks, chunksize=2), start=1
            ):
                records[index - start_index] = record
                _write_bytes_atomic(images_dir / f"{image_stem(index)}_ref.png", ref_png)
                _write_bytes_atomic(images_dir / f"{image_stem(index)}_search.png", search_png)
                if done % 25 == 0 or done == args.count:
                    print(f"[{done}/{args.count}] {args.split}", file=sys.stderr, flush=True)
    else:
        for done, (index, ref_png, search_png, record) in enumerate(map(_phase2_worker, tasks), start=1):
            records[index - start_index] = record
            _write_bytes_atomic(images_dir / f"{image_stem(index)}_ref.png", ref_png)
            _write_bytes_atomic(images_dir / f"{image_stem(index)}_search.png", search_png)
            if done % 25 == 0 or done == args.count:
                print(f"[{done}/{args.count}] {args.split}", file=sys.stderr, flush=True)

    manifest_path = output_dir / "manifest.jsonl"
    _write_text_atomic(
        manifest_path,
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )
    _write_text_atomic(
        output_dir / "citations.json",
        json.dumps(build_citations(), indent=2, sort_keys=True) + "\n",
    )
    coverage = build_coverage_report(records, args.split)
    _write_text_atomic(
        output_dir / "coverage_report.json",
        json.dumps(coverage, indent=2, sort_keys=True) + "\n",
    )

    present_count = sum(int(record["present"]) for record in records)
    severities = Counter(record["severity"] for record in records)
    dataset_info = {
        "schema_version": 2,
        "generator_version": __version__,
        "phase": 2,
        "split": args.split,
        "count": len(records),
        "modality": args.modality,
        "present_frac_realized": present_count / len(records),
        "severities": dict(sorted(severities.items())),
        "seed_base": seed_base,
        "png_encoder": "fast" if args.fast_png else "optimize",
        "start_index": start_index,
        "overrides": {
            "severity": args.severity,
            "present_frac": args.present_frac,
            "modality": args.modality,
        },
        "seed_provenance": {
            "kind": "contiguous_scene_seed_range",
            "seed_base": seed_base,
            "count": args.count,
            "ref_seed_formula": "scene_seed * 7919 + 101",
            "search_seed_formula": "scene_seed * 1000003 + 313",
        },
    }
    _write_text_atomic(
        output_dir / "DATASET_INFO.json",
        json.dumps(dataset_info, indent=2, sort_keys=True) + "\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.split is not None:
        if args.phase != 2:
            parser.error("--split is a Phase 2 option; pass --phase 2")
        args.phase = 2
    if args.phase == 2 and args.split is None:
        parser.error("--phase 2 requires --split")
    if args.phase == 2 and args.manifest:
        parser.error("--manifest is a Phase 1 option")
    try:
        if args.phase == 2:
            run_phase2(args)
        else:
            run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
