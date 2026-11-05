"""Empirical coverage audit for the yardage quantile models.

Input: one row per (market, season, week, player) with the realised stat in
`actual` and the out-of-sample quantile predictions in q05..q95 (the columns
`evaluate/walk_forward.py` must emit for this audit; it currently emits only
q10/q50/q90). Nominal central intervals:
    50% -> [q25, q75]    80% -> [q10, q90]    90% -> [q05, q95]

Usage:
    python scripts/quantile_coverage_audit.py <walk_forward.csv> <out_stem>

Writes <out_stem>.json (raw numbers). With no input file it writes a JSON
recording that the data is insufficient, and exits 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = {0.50: ("q25", "q75"), 0.80: ("q10", "q90"), 0.90: ("q05", "q95")}
PINBALL_QUANTILES = {"q05": 0.05, "q10": 0.10, "q25": 0.25, "q50": 0.50,
                     "q75": 0.75, "q90": 0.90, "q95": 0.95}


def coverage(actual, lo, hi) -> float:
    """Share of outcomes inside the closed interval [lo, hi]."""
    actual, lo, hi = (np.asarray(x, dtype=float) for x in (actual, lo, hi))
    inside = (actual >= lo) & (actual <= hi)
    return float(inside.mean()) if inside.size else float("nan")


def pinball(actual, pred, q: float) -> float:
    diff = np.asarray(actual, float) - np.asarray(pred, float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _terciles(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="first")
    return pd.cut(ranks, 3, labels=["low", "mid", "high"])


def _periods(seasons: pd.Series) -> pd.Series:
    lo, hi = seasons.min(), seasons.max()
    edges = np.linspace(lo, hi + 1, 4)
    return pd.cut(seasons, edges, labels=["early", "mid", "late"], right=False)


def audit(df: pd.DataFrame) -> dict:
    out: dict = {"status": "ok", "markets": {}}
    for market, g in df.groupby("market"):
        g = g.dropna(subset=["actual", "q50"]).copy()
        entry: dict = {"n": int(len(g))}
        entry["coverage"] = {}
        entry["mean_width"] = {}
        for nominal, (lo_c, hi_c) in LEVELS.items():
            key = f"{nominal:.2f}"
            entry["coverage"][key] = coverage(g["actual"], g[lo_c], g[hi_c])
            entry["mean_width"][key] = float((g[hi_c] - g[lo_c]).mean())
        entry["pinball"] = {k: pinball(g["actual"], g[k], q) for k, q in PINBALL_QUANTILES.items()}
        naive = g["actual"].quantile
        entry["naive_pinball"] = {
            k: pinball(g["actual"], np.full(len(g), naive(q)), q) for k, q in PINBALL_QUANTILES.items()}
        g["tercile"] = _terciles(g["q50"])
        entry["by_tercile"] = {}
        for t, sub in g.groupby("tercile", observed=True):
            entry["by_tercile"][str(t)] = {
                "n": int(len(sub)),
                "median_projection": float(sub["q50"].median()),
                **{f"{n:.2f}": coverage(sub["actual"], sub[lo], sub[hi])
                   for n, (lo, hi) in LEVELS.items()},
            }
        g["period"] = _periods(g["season"])
        entry["by_period"] = {}
        for p, sub in g.groupby("period", observed=True):
            entry["by_period"][str(p)] = {
                "n": int(len(sub)),
                **{f"{n:.2f}": coverage(sub["actual"], sub[lo], sub[hi])
                   for n, (lo, hi) in LEVELS.items()},
            }
        out["markets"][market] = entry
    return out


def insufficient(reason: str) -> dict:
    return {"status": "insufficient_data", "reason": reason, "markets": {}}


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    src, stem = Path(argv[1]), Path(argv[2])
    if not src.exists():
        result = insufficient(f"no walk-forward quantile output at {src}")
    else:
        result = audit(pd.read_csv(src))
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
