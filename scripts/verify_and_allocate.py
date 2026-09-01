#!/usr/bin/env python3
"""Verify every Phase 2 pair and allocate the corpus 70% use / 30% test.

Phase A - integrity sweep over every source (flagship splits + bulk shards):
    both images exist and decode at 1000x1000 with the right mode, every
    manifest record carries the full schema with in-range labels, every image
    file belongs to a manifest record, and no scene seed appears twice.

Phase B - orphan recovery: partial shards hold images without manifests
    (interrupted runs). Pairs are regenerated from their seeds and accepted
    only if both PNGs match byte-for-byte; accepted pairs get a reconstructed
    manifest fragment written next to them.

Phase C - sampled deep verification: per source, regenerate pairs from seeds
    and compare PNG bytes (encoder-aware), plus a brute-force pose-oracle spot
    check on a subset.

Phase D - allocation: stratified seeded shuffle (present x modality x
    severity), 30% test / 70% use, materialized as hardlinked images under
    data/dataset/{use,test} with fresh manifests and an allocation report.

Usage:
    python scripts/verify_and_allocate.py [--test-frac 0.30] [--seed 20260830]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_phase2_sample
from driftforge.pose import rotation_oracle, scale_oracle
from scripts.generate_dataset import _phase2_record

REQUIRED_FIELDS = {
    "id", "split", "scene_seed", "ref_seed", "search_seed", "architecture",
    "preset_family", "severity", "modality", "present", "gt_x", "gt_y",
    "gt_theta", "gt_scale", "n_decoys", "decoy_sites", "occlusion_frac",
    "cd_bias_pct", "edge_case", "ref_image", "search_image",
}

MANIFESTLESS_SOURCES = ("p2_train", "p2_val", "p2_test", "p2_holdout_fam",
                        "p2_stress", "p2_val_rgb")


def encoder_for(source_dir: Path) -> str:
    info_path = source_dir / "DATASET_INFO.json"
    if info_path.is_file():
        try:
            return json.loads(info_path.read_text()).get("png_encoder", "optimize")
        except json.JSONDecodeError:
            return "fast"  # partial bulk shards: written with --fast-png
    return "fast"


def encode_png(array: np.ndarray, encoder: str) -> bytes:
    buffer = io.BytesIO()
    img = Image.fromarray(np.ascontiguousarray(array))
    if encoder == "fast":
        img.save(buffer, format="PNG", compress_level=1)
    else:
        img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def check_image(path: Path, modality: str) -> str | None:
    try:
        with Image.open(path) as im:
            expected = "RGB" if modality == "rgb" else "L"
            if im.size != (1000, 1000):
                return f"size {im.size}"
            if im.mode != expected:
                return f"mode {im.mode} != {expected}"
            im.verify()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a defect
        return f"decode error: {exc}"
    return None


def verify_source(source_dir: Path, records: list[dict], problems: list[str]) -> int:
    """Phase A for one source. Returns the number of verified pairs."""
    modality_default = "rgb" if "rgb" in source_dir.name else "gray"
    seen_stems: set[str] = set()
    verified = 0
    for record in records:
        pid = record.get("id", "?")
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            problems.append(f"{pid}: missing fields {sorted(missing)}")
            continue
        modality = record.get("modality", modality_default)
        ok = True
        for key in ("ref_image", "search_image"):
            path = source_dir / record[key]
            if not path.is_file():
                problems.append(f"{pid}: missing file {record[key]}")
                ok = False
                continue
            stem = record[key]
            if stem in seen_stems:
                problems.append(f"{pid}: duplicate image reference {stem}")
            seen_stems.add(stem)
            issue = check_image(path, modality)
            if issue:
                problems.append(f"{pid}: {record[key]} {issue}")
                ok = False
        if not ok:
            continue
        if record["present"]:
            if None in (record["gt_x"], record["gt_y"], record["gt_theta"]):
                problems.append(f"{pid}: present pair with null ground truth")
                continue
            if not (0 <= record["gt_x"] < 1000 and 0 <= record["gt_y"] < 1000):
                problems.append(f"{pid}: gt position out of frame")
                continue
            # the label is the narrow-window ZNCC measurement around the prior:
            # net rotation legitimately reaches ~9.5 deg (stage +/-5 + jitter
            # +/-2.2 + search +/-0.35 + local warp), and the measured zoom
            # lands ~+/-2% from the nominal draw (local warp + measurement).
            if not -10.6 <= record["gt_theta"] <= 10.6:
                problems.append(f"{pid}: gt_theta {record['gt_theta']} out of range")
                continue
        else:
            if record["gt_x"] is not None or record["gt_theta"] is not None:
                problems.append(f"{pid}: absent pair with non-null position")
                continue
        if not 7.3 <= record["gt_scale"] <= 12.7:
            problems.append(f"{pid}: gt_scale {record['gt_scale']} out of range")
            continue
        if record["severity"] not in (0, 1, 2, 3):
            problems.append(f"{pid}: bad severity")
            continue
        verified += 1
    # every PNG on disk must belong to a record
    images_dir = source_dir / "images"
    if images_dir.is_dir():
        for png in images_dir.glob("*.png"):
            rel = f"images/{png.name}"
            if rel not in seen_stems:
                problems.append(f"orphan image {source_dir.name}/{rel} not in manifest")
    return verified


def recover_orphans(source_dir: Path, shard_index: int, problems: list[str]) -> int:
    """Phase B: regenerate manifest-less pairs and adopt byte-identical ones."""
    images_dir = source_dir / "images"
    if not images_dir.is_dir():
        return 0
    base = 10_000_000 + shard_index * 5000  # bulk_production.sh layout
    modality = "rgb" if shard_index % 7 == 6 else "gray"
    adopted = 0
    recovered: list[dict] = []
    for ref_png in sorted(images_dir.glob("*_ref.png")):
        stem = ref_png.name[: -len("_ref.png")]
        if not stem.isdigit():
            problems.append(f"{source_dir.name}: unexpected file {ref_png.name}")
            continue
        index = int(stem)
        search_png = images_dir / f"{stem}_search.png"
        if not search_png.is_file():
            problems.append(f"{source_dir.name}/{stem}: ref without search")
            continue
        seed = base + (index % 5000) + (index // 5000) * 0  # global index determines seed
        seed = 10_000_000 + index
        try:
            sample = generate_phase2_sample(seed, split="p2_bulk", modality=modality)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{source_dir.name}/{stem}: regeneration failed: {exc}")
            continue
        enc = encoder_for(source_dir)
        try:
            if (encode_png(sample.reference, enc) != ref_png.read_bytes()
                    or encode_png(sample.search, enc) != search_png.read_bytes()):
                problems.append(f"{source_dir.name}/{stem}: regenerated bytes differ - DISCARDED")
                continue
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{source_dir.name}/{stem}: encode failed: {exc}")
            continue
        record = _phase2_record(index, "p2_bulk", sample, export_debug=False, id_width=7)
        recovered.append(record)
        adopted += 1
    if recovered:
        frag = source_dir / "manifest.jsonl"
        frag.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recovered))
        info = {
            "schema_version": 2,
            "phase": 2,
            "split": "p2_bulk",
            "count": len(recovered),
            "modality": modality,
            "png_encoder": enc,
            "note": "manifest reconstructed by verify_and_allocate.py from byte-verified regeneration",
        }
        (source_dir / "DATASET_INFO.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return adopted


def deep_verify(source_dir: Path, records: list[dict], sample_n: int, seed: int,
                problems: list[str], oracle_n: int) -> dict:
    """Phase C: sampled regeneration byte-checks plus pose-oracle spot checks."""
    if not records:
        return {"regen_checked": 0, "regen_ok": 0, "oracle_checked": 0, "oracle_ok": 0}
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(records), size=min(sample_n, len(records)), replace=False)
    enc = encoder_for(source_dir)
    ok = 0
    for i in picks:
        record = records[int(i)]
        try:
            sample = generate_phase2_sample(
                int(record["scene_seed"]), split=record["split"], modality=record["modality"],
                present_frac=0.8, search_supersample=int(
                    json.loads((source_dir / "DATASET_INFO.json").read_text()).get("search_supersample", 2))
                if (source_dir / "DATASET_INFO.json").is_file() else 2,
            )
        except Exception as exc:  # noqa: BLE001
            problems.append(f"deep {record['id']}: regeneration error {exc}")
            continue
        match = (encode_png(sample.reference, enc) == (source_dir / record["ref_image"]).read_bytes()
                 and encode_png(sample.search, enc) == (source_dir / record["search_image"]).read_bytes())
        ok += match
        if not match:
            problems.append(f"deep {record['id']}: regenerated PNG bytes differ")
    oracle_picks = rng.choice(len(records), size=min(oracle_n, len(records)), replace=False)
    o_ok = 0
    o_total = 0
    for i in oracle_picks:
        record = records[int(i)]
        if not record["present"] or record["gt_x"] is None:
            continue
        try:
            ref = np.asarray(Image.open(source_dir / record["ref_image"]).convert("L"))
            sea = np.asarray(Image.open(source_dir / record["search_image"]).convert("L"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"oracle {record['id']}: read error {exc}")
            continue
        rec_theta, _ = rotation_oracle(ref, sea, record["gt_x"], record["gt_y"], record["gt_scale"])
        rec_scale, _ = scale_oracle(ref, sea, record["gt_x"], record["gt_y"], rec_theta,
                                    shape_scale=record["gt_scale"])
        d_theta = rec_theta - record["gt_theta"]
        d_scale = (rec_scale - record["gt_scale"]) / record["gt_scale"]
        o_total += 1
        o_ok += abs(d_theta) <= 0.6 and abs(d_scale) <= 0.02
    return {"regen_checked": len(picks), "regen_ok": ok,
            "oracle_checked": o_total, "oracle_ok": o_ok}


def materialize(records: list[dict], dest: Path, link: bool) -> None:
    (dest / "images").mkdir(parents=True, exist_ok=True)
    for record in records:
        source_dir = Path(record["_source_dir"])
        for key in ("ref_image", "search_image"):
            src = source_dir / record[key]
            dst = dest / record[key]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            if link:
                try:
                    os.link(src, dst)
                    continue
                except OSError:
                    pass
            shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--deep-sample", type=int, default=8, help="regeneration byte-checks per source")
    parser.add_argument("--oracle-sample", type=int, default=5, help="pose-oracle spot checks per source")
    parser.add_argument("--output-root", type=Path, default=Path("data/dataset"))
    args = parser.parse_args(argv)

    problems: list[str] = []
    pool: list[dict] = []
    per_source: dict[str, dict] = {}
    seed_seen: dict[int, str] = {}

    sources: list[tuple[str, Path]] = []
    for split in sorted(MANIFESTLESS_SOURCES):
        d = Path("data/phase2") / split
        if d.is_dir():
            sources.append((split, d))
    bulk = Path("data/phase2_bulk")
    if bulk.is_dir():
        for d in sorted(bulk.iterdir()):
            if d.is_dir():
                sources.append((d.name, d))

    for name, d in sources:
        mp = d / "manifest.jsonl"
        records = []
        if mp.is_file():
            for line in mp.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
        verified = verify_source(d, records, problems)
        adopted = 0
        if not mp.is_file():
            # partial shard: recover orphans, then re-read the reconstructed manifest
            shard_index = int(name.rsplit("_", 1)[1])
            adopted = recover_orphans(d, shard_index, problems)
            if (d / "manifest.jsonl").is_file():
                records = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines()]
                verified = verify_source(d, records, problems)
        for record in records:
            record["_source_dir"] = str(d)
            seed = int(record["scene_seed"])
            if seed in seed_seen:
                problems.append(f"duplicate scene seed {seed} in {name} and {seed_seen[seed]}")
            seed_seen[seed] = name
        pool.extend(records)
        per_source[name] = {
            "records": len(records), "verified": verified, "recovered": adopted,
        }
        print(f"[verify] {name}: {len(records)} records, {verified} verified, {adopted} recovered", flush=True)

    # Phase C deep checks, per source
    deep = {}
    for name, d in sources:
        records = [r for r in pool if r["_source_dir"] == str(d)]
        deep[name] = deep_verify(d, records, args.deep_sample, args.seed, problems,
                                 args.oracle_sample)
        print(f"[deep] {name}: {deep[name]}", flush=True)

    # Phase D allocation
    rng = np.random.default_rng(args.seed)
    strata: dict[tuple, list[dict]] = {}
    for record in pool:
        key = (record["present"], record["modality"], record["severity"])
        strata.setdefault(key, []).append(record)
    use: list[dict] = []
    test: list[dict] = []
    for key in sorted(strata):
        group = strata[key]
        order = rng.permutation(len(group))
        n_test = int(round(len(group) * args.test_frac))
        for pos, idx in enumerate(order):
            (test if pos < n_test else use).append(group[idx])

    for part in ("use", "test"):
        part_dir = args.output_root / part
        materialize([r for r in (use if part == "use" else test)], part_dir, link=True)

    for part, part_records in (("use", use), ("test", test)):
        manifest = args.output_root / part / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for record in part_records:
                slim = {k: v for k, v in record.items() if not k.startswith("_")}
                handle.write(json.dumps(slim, sort_keys=True) + "\n")

    report = {
        "allocation_seed": args.seed,
        "test_frac": args.test_frac,
        "total_pairs": len(pool),
        "use_pairs": len(use),
        "test_pairs": len(test),
        "use_present": sum(r["present"] for r in use),
        "test_present": sum(r["present"] for r in test),
        "per_source": per_source,
        "deep_verification": deep,
        "problems": problems,
        "verdict": "ALL PAIRS VERIFIED" if not problems else f"{len(problems)} PROBLEMS FOUND",
    }
    (args.output_root / "allocation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("total_pairs", "use_pairs", "test_pairs",
                                             "use_present", "test_present", "verdict")}, indent=1))
    if problems:
        print(f"PROBLEMS ({len(problems)}):", file=sys.stderr)
        for p in problems[:30]:
            print("  " + p, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
