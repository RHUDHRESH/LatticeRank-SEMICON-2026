#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge import __version__
from driftforge.generator import (
    PROFILES,
    generate_sample,
    normalize_architecture,
    normalize_profile,
)
from driftforge.splits import read_manifest


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
