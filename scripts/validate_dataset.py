#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed on corrupt, missing or malformed materialized pairs.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    labels_path = args.dataset / "labels.csv"
    if not labels_path.exists():
        raise SystemExit("labels.csv is missing (or this is an intentionally unlabeled public split)")
    with labels_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = set()
    scenes = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"duplicate id: {row['id']}")
        if row["scene_id"] in scenes:
            raise ValueError(f"duplicate scene: {row['scene_id']}")
        ids.add(row["id"])
        scenes.add(row["scene_id"])
        x, y = float(row["x"]), float(row["y"])
        if not (0 <= x < 1000 and 0 <= y < 1000):
            raise ValueError(f"out-of-bounds label for {row['id']}: {(x, y)}")
        for field in ("reference", "search"):
            path = args.dataset / row[field]
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                if image.mode != "L" or image.size != (1000, 1000):
                    raise ValueError(f"bad image contract for {path}: mode={image.mode}, size={image.size}")
    print(f"OK: {len(rows)} complete, checksum-valid, uniquely labelled pairs")


if __name__ == "__main__":
    main()

