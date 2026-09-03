#!/usr/bin/env python3
"""Audit the built Phase 2 submission against 21 release clauses.

This is deliberately a submission audit, not a scorer.  It never reads the
organizer sample pack or its ground truth.  The only images exercised are the
two examples already shipped inside the submission plus one intentionally
missing input used to verify failure-row semantics.

Usage::

    python scripts/audit_phase2_contract.py dist/LatticeRank_Phase2.zip
    python scripts/audit_phase2_contract.py dist/LatticeRank_Phase2.zip \
        --json dist/LatticeRank_Phase2.audit.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


EXPECTED_HEADER = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
EXPECTED_IDS = ["dram", "finfet", "missing"]
REQUIRED = {
    "register.py",
    "generate_dataset.py",
    "requirements.txt",
    "pyproject.toml",
    "scripts/generate_dataset.py",
    "docs/HOW_TO_RUN.md",
    "docs/failure_analysis.pdf",
    "driftforge/dense.py",
    "driftforge/refine.py",
    "driftforge/presence_model.py",
    "driftforge/correctness_model.py",
    "driftforge/budget.py",
    "driftforge/models/presence_hgb.pkl",
    "driftforge/models/correctness_lr.pkl",
    "driftforge/models/hgb_r2.joblib",
    "results/phase2_experiments/uncontended_runtime.json",
}
FORBIDDEN_ARCHIVE_PARTS = {"data", "generated", "__pycache__", ".git", ".venv"}
FORBIDDEN_NETWORK_IMPORTS = {
    "aiohttp", "ftplib", "http.client", "httpx", "requests", "socket",
    "urllib", "urllib3", "webbrowser",
}


@dataclass(frozen=True)
class Check:
    number: int
    name: str
    passed: bool
    evidence: str


def _result(number: int, name: str, condition: bool, evidence: str) -> Check:
    return Check(number, name, bool(condition), evidence)


def _relative_members(archive: zipfile.ZipFile) -> tuple[str, list[str]]:
    names = [name.replace("\\", "/") for name in archive.namelist() if not name.endswith("/")]
    roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    root = next(iter(roots)) if len(roots) == 1 else ""
    relative = [str(PurePosixPath(name).relative_to(root)) for name in names] if root else []
    return root, relative


def _safe_archive_names(names: list[str]) -> bool:
    for raw in names:
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            return False
    return True


def _network_imports(source: str) -> set[str]:
    found: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return {
        module for module in found
        if any(module == bad or module.startswith(bad + ".") for bad in FORBIDDEN_NETWORK_IMPORTS)
    }


def _sitecustomize(allowed_roots: list[Path]) -> str:
    # Network events are blocked dynamically. File reads are not globally
    # blocked because Python and native wheels legitimately read the installed
    # runtime; path confinement is instead proved by running from an unrelated
    # CWD with only the CSV/image paths and packaged model paths available.
    roots = [str(path.resolve()) for path in allowed_roots]
    return f'''\
import sys

_BLOCKED = {{
    "socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
    "urllib.Request",
}}
_ALLOWED_ROOTS = {roots!r}

def _hook(event, args):
    if event in _BLOCKED:
        raise RuntimeError("NETWORK BLOCKED: " + event)

sys.addaudithook(_hook)
'''


def _run_entry(package: Path, dataset: Path, site: Path, output: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(site)
    env["PYTHONHASHSEED"] = "0"
    return subprocess.run(
        [sys.executable, str(package / "register.py"),
         "--input", str(dataset / "pairs.csv"), "--output", str(output)],
        cwd=str(dataset.parent), env=env, capture_output=True, text=True,
        timeout=120,
    )


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def audit(zip_path: Path) -> tuple[list[Check], dict[str, object]]:
    checks: list[Check] = []
    with zipfile.ZipFile(zip_path) as archive:
        raw_names = [name.replace("\\", "/") for name in archive.namelist()]
        root, relative = _relative_members(archive)
        present = set(relative)

        checks.append(_result(1, "single archive root", bool(root), root or "multiple roots"))
        checks.append(_result(2, "safe archive paths", _safe_archive_names(raw_names),
                              f"{len(raw_names)} members; no absolute/traversal paths"))
        missing = sorted(REQUIRED - present)
        checks.append(_result(3, "required files and weights", not missing,
                              "all present" if not missing else "missing: " + ", ".join(missing)))

        req_text = archive.read(f"{root}/requirements.txt").decode("utf-8") if root else ""
        req_lines = [line.strip() for line in req_text.splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
        pinned = bool(req_lines) and all("==" in line and not line.startswith(("-e", "git+"))
                                         for line in req_lines)
        checks.append(_result(4, "frozen requirements", pinned,
                              f"{len(req_lines)} exact pins"))

        pyproject = archive.read(f"{root}/pyproject.toml").decode("utf-8") if root else ""
        checks.append(_result(5, "Python 3.11 declared", 'requires-python = ">=3.11"' in pyproject,
                              "pyproject requires Python >=3.11"))

        forbidden = sorted(name for name in relative
                           if FORBIDDEN_ARCHIVE_PARTS.intersection(PurePosixPath(name).parts)
                           or "ground_truth.csv" in name.lower()
                           or "manifest_jury" in name.lower())
        checks.append(_result(6, "no sponsor/test data packaged", not forbidden,
                              "no data, organizer key, or jury manifest" if not forbidden
                              else "forbidden: " + ", ".join(forbidden[:5])))

        try:
            from pypdf import PdfReader
            with archive.open(f"{root}/docs/failure_analysis.pdf") as stream:
                page_count = len(PdfReader(stream).pages)
        except Exception as exc:  # pragma: no cover - actionable environment failure
            page_count = -1
            pdf_evidence = f"unreadable: {exc}"
        else:
            pdf_evidence = f"{page_count} page(s), limit 2"
        checks.append(_result(7, "failure analysis page limit", 0 < page_count <= 2,
                              pdf_evidence))

        register_source = archive.read(f"{root}/register.py").decode("utf-8") if root else ""
        forbidden_imports = sorted(_network_imports(register_source)) if register_source else ["missing"]
        checks.append(_result(8, "entry point has no network imports", not forbidden_imports,
                              "static import scan clean" if not forbidden_imports
                              else "imports: " + ", ".join(forbidden_imports)))

        runtime = json.loads(archive.read(
            f"{root}/results/phase2_experiments/uncontended_runtime.json"
        )) if root else {}
        measured = runtime.get("measured", {})
        median = float(measured.get("median_s", math.inf))
        maximum = float(measured.get("max_s", math.inf))
        sample_n = int(measured.get("n", 0))
        checks.append(_result(9, "recorded runtime budgets", sample_n >= 20 and median <= 5.0 and maximum < 20.0,
                              f"n={sample_n}, median={median:.3f}s, max={maximum:.3f}s"))

    with tempfile.TemporaryDirectory(prefix="latticerank-contract-") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(work)
        package = work / root
        dataset = work / "dataset"
        images = dataset / "images"
        images.mkdir(parents=True)
        for architecture in ("dram", "finfet"):
            for role in ("reference", "search"):
                source = package / "examples" / architecture / f"{role}.png"
                destination = images / f"{architecture}_{role}.png"
                destination.write_bytes(source.read_bytes())

        with (dataset / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            # Search-before-reference mirrors the supplied organizer format.
            writer.writerow(["pair_id", "search_path", "reference_path"])
            writer.writerow(["dram", "images/dram_search.png", "images/dram_reference.png"])
            writer.writerow(["finfet", "images/finfet_search.png", "images/finfet_reference.png"])
            writer.writerow(["missing", "images/no-search.png", "images/no-reference.png"])

        site = work / "audit_site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(
            _sitecustomize([package, dataset]), encoding="utf-8"
        )

        # Prove the hook is active before trusting it.
        probe_env = os.environ.copy()
        probe_env["PYTHONPATH"] = str(site)
        probe = subprocess.run(
            [sys.executable, "-c",
             "import urllib.request; urllib.request.urlopen('http://example.invalid', timeout=1)"],
            cwd=str(work), env=probe_env, capture_output=True, text=True, timeout=10,
        )
        hook_live = "NETWORK BLOCKED" in (probe.stdout + probe.stderr)

        first = dataset / "predictions-1.csv"
        second = dataset / "predictions-2.csv"
        run1 = _run_entry(package, dataset, site, first)
        run2 = _run_entry(package, dataset, site, second)
        completed = (hook_live and run1.returncode == 0 and run2.returncode == 0
                     and first.is_file() and second.is_file())
        run_evidence = (f"rc={run1.returncode}/{run2.returncode}; network hook "
                        f"{'active' if hook_live else 'FAILED'}")
        checks.append(_result(10, "exact --input/--output CLI", completed, run_evidence))
        checks.append(_result(11, "runs outside submission CWD", completed,
                              f"cwd={work.name}; package={root}"))
        checks.append(_result(12, "relative paths resolve from pairs.csv", completed,
                              "images supplied only beside pairs.csv"))
        checks.append(_result(13, "offline execution", completed,
                              "network audit hook active; both runs completed"))

        header, rows = _read_rows(first) if first.is_file() else ([], [])
        checks.append(_result(14, "exact output header", header == EXPECTED_HEADER,
                              ",".join(header)))
        checks.append(_result(15, "one row per pair", len(rows) == len(EXPECTED_IDS),
                              f"{len(rows)}/{len(EXPECTED_IDS)} rows"))
        ids = [row.get("pair_id", "") for row in rows]
        checks.append(_result(16, "pair IDs preserved once and in order", ids == EXPECTED_IDS,
                              repr(ids)))

        numeric_columns = EXPECTED_HEADER[1:]
        finite = True
        for row in rows:
            try:
                finite &= all(math.isfinite(float(row[column])) for column in numeric_columns)
            except (KeyError, TypeError, ValueError):
                finite = False
        checks.append(_result(17, "all numeric outputs finite", finite,
                              f"checked {len(rows)} rows x {len(numeric_columns)} fields"))

        found_binary = all(row.get("found") in {"0", "1"} for row in rows)
        checks.append(_result(18, "found is binary", found_binary,
                              repr([row.get("found") for row in rows])))
        rejected = [row for row in rows if row.get("found") == "0"]
        zeroed = bool(rejected) and all(
            all(float(row[column]) == 0.0 for column in ("x", "y", "theta", "scale"))
            for row in rejected
        )
        checks.append(_result(19, "found=0 zeroes all pose columns", zeroed,
                              f"{len(rejected)} rejection/failure row(s) checked"))

        accepted = [row for row in rows if row.get("found") == "1"]
        locations_ok = bool(accepted) and all(
            0.0 <= float(row["x"]) <= 999.0 and 0.0 <= float(row["y"]) <= 999.0
            for row in accepted
        )
        pose_ok = bool(accepted) and all(
            8.0 <= float(row["scale"]) <= 12.0
            and -5.0 <= float(row["theta"]) <= 5.0
            for row in accepted
        )
        checks.append(_result(20, "accepted locations are in bounds", locations_ok,
                              f"{len(accepted)} accepted row(s) checked"))
        checks.append(_result(21, "accepted poses are in disclosed bounds and deterministic",
                              pose_ok and first.read_bytes() == second.read_bytes(),
                              f"{len(accepted)} pose(s); byte-identical rerun="
                              f"{first.read_bytes() == second.read_bytes()}"))

    metadata = {
        "zip": str(zip_path.resolve()),
        "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "checks_passed": sum(check.passed for check in checks),
        "checks_total": len(checks),
        "sponsor_data_used": False,
        "checks": [asdict(check) for check in checks],
    }
    return checks, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip", type=Path)
    parser.add_argument("--json", type=Path, default=None,
                        help="optional path for a machine-readable audit report")
    args = parser.parse_args(argv)
    if not args.zip.is_file():
        parser.error(f"submission zip not found: {args.zip}")

    checks, report = audit(args.zip)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.number:02d}/21 {check.name}: {check.evidence}")
    print(f"\n{report['checks_passed']}/{report['checks_total']} contract checks passed")
    print(f"zip sha256 {report['sha256']}")
    print("Sponsor/organizer data used by audit: NO")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report {args.json}")
    return 0 if report["checks_passed"] == report["checks_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
