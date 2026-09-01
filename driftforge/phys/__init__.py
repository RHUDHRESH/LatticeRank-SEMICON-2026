"""Physically-motivated candidate evidence, measured inside LatticeRank.

Every module here answers one question about a candidate that is justified by
SEM image formation, registration geometry or lattice mechanics. None of them
generates candidates, reorders the proposal stage, or replaces any part of the
declared Phase 1 chain -- they add columns to the candidate feature table that
the existing ranker already consumes.

The doctrine these modules exist under, arrived at by measurement:

    New physical evidence never vetoes baseline candidate generation unless it
    independently proves higher recall.

Response-space lattice cancellation, RCC-as-pre-ranker and uniqueness-weighted
proposals each reordered candidates correctly and each lost true sites when
allowed to replace baseline evidence. Evidence that reorders belongs in the
feature table; only evidence that raises recall belongs in the proposal stage.
"""
