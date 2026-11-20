#!/usr/bin/env python
"""CFB in-season total drift correction (R1), walk-forward test.

Implements docs/2026-11-20-cfb-total-drift-prereg.md exactly: the served total
candidate (ridge on DEFAULT_BLOCKS + adj_epa) is walked forward over the
2014-2025 seasons, and within each held-out season a game in week w gets

    delta = mean(predicted_total - actual_total) over that season's games in
            weeks < w, or 0 when fewer than 100 such games
    total' = predicted_total - delta

Descriptive cells (not gating): the cached sportsbook closing totals for the
2026 snapshot, and the 2026 public snapshot itself. No API calls, no model
reimplementation: the folds, fit and data loader are the repo's own.

Run: PYTHONPATH=src uv run python scripts/cfb_total_drift.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfb_predictor.config import PUBLIC_SNAPSHOT_PATH, SPORTSBOOK_CACHE_DIR
from cfb_predictor.evaluate import walk_forward as wf
from cfb_predictor.models.manifest import TOTAL_SPEC
from cfb_predictor.tools import block_eval as be
from cfb_predictor.tools.total_eval import bias_gap

CANDIDATE = TOTAL_SPEC["kind"]
BLOCKS = tuple(TOTAL_SPEC["blocks"])
MIN_PRIOR = 100
BOOT_N = 4000
SEED = 20261017
ROWS_PATH = Path(__file__).resolve().parents[1] / "docs" / "2026-11-20-cfb-total-drift-rows.csv"


def r1_deltas(season, week, pred, actual, min_prior: int = MIN_PRIOR) -> np.ndarray:
    """delta per game: mean(pred - actual) over the same season's earlier weeks.

    A game's own week is never in its history, and a season with fewer than
    `min_prior` earlier games gets delta = 0.
    """
    season = np.asarray(season)
    week = np.asarray(week)
    resid = np.asarray(pred, float) - np.asarray(actual, float)
    out = np.zeros(len(week), float)
    for s in np.unique(season):
        in_season = season == s
        season_weeks = week[in_season]
        season_resid = resid[in_season]
        season_pos = np.flatnonzero(in_season)
        for w in np.unique(season_weeks):
            prior = season_weeks < w
            if prior.sum() >= min_prior:
                out[season_pos[season_weeks == w]] = season_resid[prior].mean()
    return out


def add_r1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta"] = r1_deltas(df["season"], df["week"], df["pred_total"], df["actual_total"])
    df["pred_r1"] = df["pred_total"] - df["delta"]
    return df


def walk_forward_rows(games_df, aux) -> pd.DataFrame:
    folds = wf.prepare_folds(games_df, blocks=BLOCKS, aux=aux)
    pieces = []
    for fold in folds:
        val = fold["val_df"]
        pred = wf._fit_predict(
            CANDIDATE, fold["train_df"], val, fold["feature_cols"], "total_points"
        )
        pieces.append(
            pd.DataFrame(
                {
                    "season": val["season"].to_numpy(),
                    "week": val["week"].to_numpy(),
                    "game_id": val["game_id"].astype(str).to_numpy(),
                    "pred_total": np.asarray(pred, float),
                    "actual_total": val["total_points"].to_numpy(float),
                }
            )
        )
    rows = pd.concat(pieces, ignore_index=True)
    return add_r1(rows).sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def snapshot_rows() -> tuple[pd.DataFrame, float | None]:
    snap = json.loads(PUBLIC_SNAPSHOT_PATH.read_text())
    rows = []
    sigma = None
    for week_block in snap["weeks"].values():
        predictions = week_block.get("predictions") or {}
        for game in week_block.get("games") or []:
            if game.get("home_score") is None or game.get("away_score") is None:
                continue
            pred = predictions.get(str(game["game_id"]))
            if not pred or pred.get("predicted_total") is None:
                continue
            if sigma is None and pred.get("total_sigma") is not None:
                sigma = float(pred["total_sigma"])
            rows.append(
                {
                    "season": int(game["season"]),
                    "week": int(game["week"]),
                    "game_id": str(game["game_id"]),
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "pred_total": float(pred["predicted_total"]),
                    "actual_total": float(game["home_score"]) + float(game["away_score"]),
                }
            )
    return add_r1(pd.DataFrame(rows)).sort_values(["week", "game_id"]).reset_index(drop=True), sigma


def _norm_team(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "".join(ch for ch in ascii_name.lower() if ch.isalnum())


def sportsbook_totals(cache_dir: Path) -> pd.DataFrame:
    """One median full-match total line per cached event, across all books."""
    rows = []
    for path in sorted(Path(cache_dir).glob("event_*.json")):
        event = json.loads(path.read_text())
        markets = [
            m
            for m in event.get("markets") or []
            if isinstance(m, dict) and m.get("type") == "POINT_TOTAL" and m.get("segment") == "FULL_MATCH"
        ]
        lines = []
        for market in markets:
            for outcomes in (market.get("outcomes") or {}).values():
                for outcome in outcomes or []:
                    if outcome.get("type") == "OVER" and outcome.get("modifier") is not None:
                        lines.append(float(outcome["modifier"]))
        if not lines:
            continue
        away, home = [part.strip() for part in event["name"].split(" @ ")]
        rows.append(
            {
                "home_key": _norm_team(home),
                "away_key": _norm_team(away),
                "line": float(statistics.median(lines)),
                "n_books": len(lines),
            }
        )
    return pd.DataFrame(rows)


def log_loss(prob_over, over) -> float:
    p = np.clip(np.asarray(prob_over, float), 1e-12, 1 - 1e-12)
    y = np.asarray(over, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def paired_ci(base_err, r1_err) -> tuple[float, float, float]:
    delta = float(np.mean(base_err - r1_err))
    lo, hi = be.paired_bootstrap(base_err, r1_err, n=BOOT_N, seed=SEED)
    return delta, lo, hi


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def print_report(rows: pd.DataFrame) -> dict:
    base_err = np.abs(rows["pred_total"] - rows["actual_total"])
    r1_err = np.abs(rows["pred_r1"] - rows["actual_total"])
    delta, lo, hi = paired_ci(base_err, r1_err)

    print("CFB in-season total drift (R1) — walk-forward")
    print(f"candidate={CANDIDATE}  blocks={BLOCKS}")
    print(f"held-out seasons: {int(rows['season'].min())}-{int(rows['season'].max())}  games={len(rows)}")
    print()

    print("PRIMARY (pooled paired per-game |base error| - |R1 error|)")
    print(f"  base MAE = {base_err.mean():.4f}")
    print(f"  R1 MAE   = {r1_err.mean():.4f}")
    print(f"  delta    = {delta:+.4f}  95% CI {_fmt_ci((lo, hi))}  (bootstrap {BOOT_N}, seed {SEED})")
    primary_pass = lo > 0
    print(f"  rule 1 (low95 > 0): {'PASS' if primary_pass else 'FAIL'}")
    print()

    print("PER-SEASON (same paired delta, same bootstrap)")
    print(f"{'season':>6} {'n':>5} {'base_mae':>9} {'r1_mae':>9} {'delta':>9} {'95% CI':>22} {'upper95<0':>9}")
    any_worse = False
    for season, sub in rows.groupby("season"):
        sb = np.abs(sub["pred_total"] - sub["actual_total"])
        sr = np.abs(sub["pred_r1"] - sub["actual_total"])
        sd, slo, shi = paired_ci(sb, sr)
        worse = shi < 0
        any_worse = any_worse or worse
        print(
            f"{int(season):>6} {len(sub):>5} {sb.mean():>9.4f} {sr.mean():>9.4f} "
            f"{sd:>+9.4f} {_fmt_ci((slo, shi)):>22} {str(worse):>9}"
        )
    print(f"  rule 2 (no season upper95 < 0): {'PASS' if not any_worse else 'FAIL'}")
    print()

    gap_base = bias_gap(rows["pred_total"].to_numpy(float), rows["actual_total"].to_numpy(float))
    gap_r1 = bias_gap(rows["pred_r1"].to_numpy(float), rows["actual_total"].to_numpy(float))
    gap_ok = gap_r1 <= gap_base
    print("TOTAL-BIAS GAP (tools/total_eval.bias_gap, 5 quintiles)")
    print(f"  base = {gap_base:.4f}")
    print(f"  R1   = {gap_r1:.4f}")
    print(f"  change (R1 - base) = {gap_r1 - gap_base:+.4f}")
    print(f"  rule 3 (does not widen): {'PASS' if gap_ok else 'FAIL'}")
    print()

    print("DELTA AT SEASON END")
    print(f"{'season':>6} {'final_week':>10} {'final_week_delta':>17} {'n_prior_final':>14} {'full_season_delta':>18}")
    for season, sub in rows.groupby("season"):
        final_week = int(sub["week"].max())
        final = sub[sub["week"] == final_week]
        n_prior = int((sub["week"] < final_week).sum())
        full = float(np.mean(sub["pred_total"] - sub["actual_total"]))
        print(f"{int(season):>6} {final_week:>10} {final['delta'].iloc[0]:>+17.4f} {n_prior:>14} {full:>+18.4f}")
    print()

    print("COUNTS")
    print(f"  total held-out games: {len(rows)}")
    print(f"  seasons: {rows['season'].nunique()} ({int(rows['season'].min())}-{int(rows['season'].max())})")
    print(f"  games with delta != 0: {int((rows['delta'] != 0).sum())}")
    print(f"  games with delta == 0: {int((rows['delta'] == 0).sum())}")
    print("  per-season games / first week with delta != 0:")
    for season, sub in rows.groupby("season"):
        nz = sub[sub["delta"] != 0]
        first = int(nz["week"].min()) if len(nz) else None
        print(f"    {int(season)}: {len(sub)} games, first corrected week {first}")
    print()

    print("DESCRIPTIVE — 2026 public snapshot (retrospective rescoring, not served pre-game values)")
    snap, sigma = snapshot_rows()
    if snap.empty:
        print("  no snapshot rows with predictions and final scores")
    else:
        sb_err = np.abs(snap["pred_total"] - snap["actual_total"])
        sr_err = np.abs(snap["pred_r1"] - snap["actual_total"])
        sd, slo, shi = paired_ci(sb_err, sr_err)
        print(f"  games={len(snap)}  weeks={int(snap['week'].min())}-{int(snap['week'].max())}")
        print(f"  base MAE = {sb_err.mean():.4f}")
        print(f"  R1 MAE   = {sr_err.mean():.4f}")
        print(f"  delta    = {sd:+.4f}  95% CI {_fmt_ci((slo, shi))}  (bootstrap {BOOT_N}, seed {SEED})")
        print("  week-by-week applied delta / MAE:")
        print(f"  {'week':>5} {'n':>4} {'delta':>8} {'base_mae':>9} {'r1_mae':>9}")
        for week, sub in snap.groupby("week"):
            wb = np.abs(sub["pred_total"] - sub["actual_total"]).mean()
            wr = np.abs(sub["pred_r1"] - sub["actual_total"]).mean()
            print(f"  {int(week):>5} {len(sub):>4} {sub['delta'].iloc[0]:>+8.4f} {wb:>9.4f} {wr:>9.4f}")
        print(f"  note: snapshot predictions may be retrospective rescoring, not served pre-game values.")
    print()

    print("DESCRIPTIVE — cached sportsbook closing totals (2026 snapshot)")
    totals = sportsbook_totals(SPORTSBOOK_CACHE_DIR)
    if totals.empty:
        print("  no cached sportsbook total lines found")
    elif snap.empty:
        print("  sportsbook totals exist, but no snapshot games to join")
    else:
        keyed = snap.copy()
        keyed["home_key"] = keyed["home_team"].map(_norm_team)
        keyed["away_key"] = keyed["away_team"].map(_norm_team)
        joined = keyed.merge(totals, on=["home_key", "away_key"], how="inner")
        joined = joined[joined["actual_total"] != joined["line"]]
        if joined.empty:
            print("  no sportsbook totals matched a played snapshot game")
        else:
            sigma_used = float(sigma if sigma is not None else 16.227224426131006)
            over = (joined["actual_total"] > joined["line"]).astype(int).to_numpy()
            p_base = 1.0 - norm.cdf(joined["line"], loc=joined["pred_total"], scale=sigma_used)
            p_r1 = 1.0 - norm.cdf(joined["line"], loc=joined["pred_r1"], scale=sigma_used)
            ll_base = log_loss(p_base, over)
            ll_r1 = log_loss(p_r1, over)
            print(f"  matched games={len(joined)}  sigma={sigma_used:.6f}  pushes excluded={len(snap) - len(joined)} unmatched/push")
            print(f"  over rate (actual) = {over.mean():.4f}")
            print(f"  over/under log loss base = {ll_base:.4f}")
            print(f"  over/under log loss R1   = {ll_r1:.4f}")
            print(f"  delta (base - R1) = {ll_base - ll_r1:+.4f}  (positive means R1 better)")
            print(f"  line source: median full-match POINT_TOTAL across cached books; last pre-game snapshot, not a verified close")
    print()

    verdict = "PASS" if (primary_pass and not any_worse and gap_ok) else "fails"
    print(f"VERDICT (prereg ship rule, all must hold): {verdict}")
    return {
        "primary_pass": primary_pass,
        "any_worse": any_worse,
        "gap_ok": gap_ok,
        "delta": delta,
        "lo": lo,
        "hi": hi,
    }


def selfcheck() -> int:
    # 99 prior games: not enough history, delta must stay 0.
    season = np.ones(100, int)
    week = np.array([1] * 99 + [2])
    pred = np.array([50.0] * 99 + [50.0])
    actual = np.array([48.0] * 99 + [48.0])
    d = r1_deltas(season, week, pred, actual)
    assert d[0] == 0.0, "a game in week 1 has no earlier week"
    assert d[-1] == 0.0, f"99 prior games must not trigger the correction, got {d[-1]}"

    # 100 prior games: delta equals their mean residual.
    season = np.ones(102, int)
    week = np.array([1] * 100 + [2] * 2)
    pred = np.array([50.0] * 102)
    actual = np.array([48.0] * 100 + [10.0] * 2)
    d = r1_deltas(season, week, pred, actual)
    assert d[0] == 0.0, "same-week games are not in a week-1 game's history"
    assert d[100] == 2.0 and d[101] == 2.0, f"100 prior games must give mean +2.0, got {d[100:]}"
    assert d[100] != np.mean([2.0] * 100 + [40.0] * 2), "same-week games must be excluded"

    # A week-2 game does not see other week-2 games, only week 1.
    season = np.ones(101, int)
    week = np.array([1] * 100 + [2])
    pred = np.array([50.0] * 101)
    actual = np.array([48.0] * 100 + [0.0])
    d = r1_deltas(season, week, pred, actual)
    assert d[-1] == 2.0, f"week-2 delta must ignore the week-2 game itself, got {d[-1]}"

    print("selfcheck OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CFB in-season total drift (R1) walk-forward")
    parser.add_argument("--selfcheck", action="store_true", help="run the delta-rule asserts and exit")
    args = parser.parse_args()
    if args.selfcheck:
        return selfcheck()

    games_df, aux = be.load_recent_cfb_data(2014, 2025)
    games_df["game_id"] = games_df["game_id"].astype(str)
    rows = walk_forward_rows(games_df, aux)
    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows[["season", "week", "game_id", "pred_total", "actual_total", "delta", "pred_r1"]].to_csv(
        ROWS_PATH, index=False
    )
    print_report(rows)
    print(f"rows written: {ROWS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
