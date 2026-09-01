"""Build and self-verify the Phase 2 submission zip.

Packaging from git state would ship a **Phase 1** submission: ``register.py``,
the whole Phase 2 solver and both Phase 2 model pickles are untracked, so
``git archive`` silently omits the entry point and the weights while still
producing a plausible-looking zip. Nothing in the archive would announce the
loss -- the pipeline degrades to a raw-correlation threshold and quietly
forfeits the rejection and confidence blocks.

So the manifest is an explicit allow-list, checked against :data:`REQUIRED`
before the zip is written and again by extracting the finished archive to a
scratch directory and running the documented entry point inside it. A missing
weight file fails the build instead of the scored run.

    python scripts/build_submission.py
    python scripts/build_submission.py --output dist/LatticeRank_Phase2.zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Every file matching these globs is packaged, if it exists.
INCLUDE = (
    "register.py",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "driftforge/**/*.py",
    "driftforge/models/*",
    "scripts/*.py",
    "tests/*.py",
    "docs/*.md",
    "docs/failure_analysis.pdf",
    "docs/images/*",
    "examples/**/*",
    "manifests/*",
    "results/*.json",
    "results/*.csv",
    # Every evidence file the judged failure analysis cites must travel with it,
    # or its references are broken in the shipped zip. Enforced by check_citations.
    "results/phase2_experiments/**/*.json",
    "results/phase2_experiments/**/*.csv",
    "results/phase2_failures/rank003_*.png",
)

#: Never packaged, whatever INCLUDE matched. ``data/`` alone is 71 GB.
EXCLUDE_PARTS = ("__pycache__", ".git", ".pytest_cache", ".venv", "data",
                 "generated", "experiments")
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".npz", ".log", ".tmp")

#: The build fails if any of these is absent. The addendum names the first four
#: explicitly; the rest are what makes the run score above a degraded fallback.
REQUIRED = (
    "register.py",
    "requirements.txt",
    "scripts/generate_dataset.py",
    "docs/failure_analysis.pdf",
    "driftforge/models/presence_hgb.pkl",
    "driftforge/models/correctness_lr.pkl",
    "driftforge/models/hgb_r2.joblib",
    "driftforge/dense.py",
    "driftforge/refine.py",
    "driftforge/presence_model.py",
    "driftforge/correctness_model.py",
    "driftforge/budget.py",
)

#: The exact output header, in order, from the addendum's output contract.
EXPECTED_HEADER = ["pair_id", "x", "y", "theta", "scale", "found", "score"]


def collect() -> list[Path]:
    seen: set[Path] = set()
    for pattern in INCLUDE:
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDE_PARTS for part in rel.parts):
                continue
            if path.suffix.lower() in EXCLUDE_SUFFIX:
                continue
            seen.add(rel)
    return sorted(seen)


def check_citations(present: set[str]) -> int:
    """Fail the build if the judged failure analysis cites a file we do not ship.

    ``failure_analysis.pdf`` is read by a judge who cannot see this repository.
    Every ``results/...`` path it names has to be inside the zip, or the evidence
    trail dead-ends exactly where it is being checked. The markdown is the
    machine-readable twin of the PDF, so it is what gets scanned.
    """
    source = ROOT / "docs" / "failure_analysis.md"
    if not source.is_file():
        return 0
    cited = set(re.findall(r"results/[A-Za-z0-9_./-]+", source.read_text(encoding="utf-8")))
    cited = {c.rstrip(".,);") for c in cited}
    missing = sorted(c for c in cited if c not in present)
    if missing:
        print("FAIL: failure_analysis cites files that are not in the zip:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        raise SystemExit(1)
    return len(cited)


def check_pdf_pages(limit: int = 2) -> int | None:
    """Return the page count, or None if pypdf is unavailable."""
    try:
        import pypdf
    except ImportError:
        return None
    pages = len(pypdf.PdfReader(ROOT / "docs/failure_analysis.pdf").pages)
    if pages > limit:
        raise SystemExit(f"FAIL: failure_analysis.pdf is {pages} pages, limit {limit}")
    return pages


def smoke_test(zip_path: Path) -> None:
    """Extract the finished zip elsewhere entirely and run the entry point.

    This is the only check that proves the archive is self-contained: it runs
    from a scratch directory with no access to the source tree, so a weight
    loaded by an accidental repo-relative path fails here rather than silently
    during the scored run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(work)
        pkg = work / zip_path.stem
        if not (pkg / "register.py").is_file():
            raise SystemExit(f"FAIL: register.py missing from extracted {pkg}")

        pairs = work / "pairs.csv"
        n_in = 0
        with pairs.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pair_id", "reference_path", "search_path"])
            for name in ("dram", "finfet"):
                ref = pkg / "examples" / name / "reference.png"
                search = pkg / "examples" / name / "search.png"
                if ref.is_file() and search.is_file():
                    writer.writerow([name, ref, search])
                    n_in += 1

        out = work / "predictions.csv"
        proc = subprocess.run(
            [sys.executable, "register.py", "--input", str(pairs), "--output", str(out)],
            cwd=str(pkg), capture_output=True, text=True, timeout=600, check=False,
        )
        if not out.is_file():
            raise SystemExit(f"FAIL: no predictions.csv\n{proc.stdout}\n{proc.stderr}")
        with out.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        header = list(rows[0].keys()) if rows else []
        if header != EXPECTED_HEADER:
            raise SystemExit(f"FAIL: header {header} != {EXPECTED_HEADER}")
        if len(rows) != n_in:
            raise SystemExit(f"FAIL: {len(rows)} rows for {n_in} pairs")
        for row in rows:
            if row["found"] == "0" and any(float(row[c]) != 0.0
                                           for c in ("x", "y", "theta", "scale")):
                raise SystemExit(f"FAIL: found=0 with non-zero pose: {row}")
        print(f"  smoke: {len(rows)}/{n_in} rows, header exact, zeroed-pose contract held")
        for row in rows:
            print(f"    {row['pair_id']:8s} found={row['found']} "
                  f"x={float(row['x']):.2f} y={float(row['y']):.2f} "
                  f"scale={float(row['scale']):.3f} score={float(row['score']):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 2 submission zip.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "dist" / "LatticeRank_Phase2.zip")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="skip the extract-and-run check (fast, much weaker)")
    args = parser.parse_args()

    files = collect()
    present = {str(rel).replace("\\", "/") for rel in files}
    missing = [name for name in REQUIRED if name not in present]
    if missing:
        print("FAIL: required files are not in the manifest:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        return 1

    n_cited = check_citations(present)
    pages = check_pdf_pages()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.stem
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in files:
            archive.write(ROOT / rel, arcname=str(Path(stem) / rel))

    size_mb = args.output.stat().st_size / 1e6
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"built {args.output}")
    print(f"  {len(files)} files, {size_mb:.1f} MB, sha256 {digest[:16]}...")
    print(f"  failure_analysis.pdf: {pages if pages else '?'} pages")
    print(f"  required files: all {len(REQUIRED)} present")
    print(f"  cited evidence: all {n_cited} paths in failure_analysis resolve")

    if not args.skip_smoke:
        smoke_test(args.output)

    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "zip": args.output.name,
        "sha256": digest,
        "n_files": len(files),
        "size_bytes": args.output.stat().st_size,
        "failure_analysis_pages": pages,
        "files": sorted(present),
    }, indent=1), encoding="utf-8")
    print(f"  manifest {manifest.name}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
