#!/usr/bin/env python3
"""Prove the scored run needs no network and no repository working directory.

Phase 2 lists "any network access during the scored run" and "reading outside
the supplied paths" as disqualifications with no appeal. Both are claims about
behaviour, so both need a mechanical check rather than an assurance.

    python scripts/verify_offline.py                     # judge_check.py
    python scripts/verify_offline.py --entry register.py --args "--input pairs.csv --output out.csv"

The check runs the target in a subprocess that has an **audit hook** installed
via a generated ``sitecustomize``. Audit hooks (``sys.addaudithook``, 3.8+)
observe events without replacing objects, so any genuine connection attempt
raises while pickle, joblib and numpy keep working normally.

Do not be tempted to monkeypatch ``socket.socket`` instead: it is a *class*,
and replacing it with a function corrupts unrelated runtime machinery. That
approach makes ``joblib.load`` fail with a misleading
``function() argument 'code' must be code, not str`` -- a false positive that
looks exactly like a real model-loading defect.

The subprocess also runs from a directory outside the repository, so any path
that silently resolves relative to the repo root fails here rather than on the
evaluator's machine.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

#: Audit events that indicate real network egress. ``socket.connect`` covers
#: raw sockets; the name-resolution events catch a lookup that precedes it;
#: ``urllib.Request`` catches the common high-level path.
BLOCKED_EVENTS = (
    "socket.connect",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "urllib.Request",
)

SITECUSTOMIZE = '''\
import sys

_BLOCKED = {blocked!r}


def _hook(event, args):
    if event in _BLOCKED:
        raise RuntimeError("NETWORK BLOCKED: " + event)


sys.addaudithook(_hook)
'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--entry",
        default="scripts/judge_check.py",
        help="script to run, relative to the repository root",
    )
    parser.add_argument(
        "--args",
        default="",
        help="arguments passed to the entry script, as one quoted string",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="working directory for the run (default: a fresh temp directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entry = (PROJECT / args.entry).resolve()
    if not entry.is_file():
        print(f"error: entry script not found: {entry}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="driftforge-offline-") as tmp:
        site = Path(tmp) / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(
            SITECUSTOMIZE.format(blocked=BLOCKED_EVENTS), encoding="utf-8"
        )
        run_cwd = Path(args.cwd).resolve() if args.cwd else Path(tmp)

        # Self-test: if the hook does not actually block, the whole check is
        # worthless and must fail loudly rather than report a false pass.
        probe = subprocess.run(
            [sys.executable, "-c",
             "import urllib.request;"
             "urllib.request.urlopen('http://example.invalid', timeout=2)"],
            cwd=str(run_cwd),
            env={"PYTHONPATH": str(site), "SYSTEMROOT": _systemroot()},
            capture_output=True,
            text=True,
        )
        if "NETWORK BLOCKED" not in (probe.stderr + probe.stdout):
            print("error: the audit hook did not block a probe request; "
                  "this check cannot prove anything", file=sys.stderr)
            print(probe.stderr[-500:], file=sys.stderr)
            return 2

        print(f"entry     : {entry.relative_to(PROJECT)}")
        print(f"cwd       : {run_cwd}")
        print(f"python    : {sys.version.split()[0]}")
        print(f"blocking  : {', '.join(BLOCKED_EVENTS)}")
        print("-" * 60)

        result = subprocess.run(
            [sys.executable, str(entry), *shlex.split(args.args)],
            cwd=str(run_cwd),
            env={"PYTHONPATH": str(site), "SYSTEMROOT": _systemroot()},
            capture_output=True,
            text=True,
        )

    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        blocked = "NETWORK BLOCKED" in result.stderr
        print("-" * 60)
        print(
            "FAIL | the run attempted network access"
            if blocked
            else "FAIL | the run did not complete offline outside the repository",
            file=sys.stderr,
        )
        return 1

    print("-" * 60)
    print("PASS | completed with no network access, outside the repository cwd")
    return 0


def _systemroot() -> str:
    import os

    return os.environ.get("SYSTEMROOT", r"C:\Windows")


if __name__ == "__main__":
    raise SystemExit(main())
