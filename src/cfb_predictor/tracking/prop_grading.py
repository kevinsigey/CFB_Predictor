"""Player-prop grading: how close each counted projection landed.

The per-pick counterpart of the track record's per-market MAE
(`store._summarize_player_props`), and the CFB mirror of NBA_Predictor's
`tracking/prop_grading.py` (PR #163): every counted player projection for
the most recent completed game days, laid against the player's actual box
score, and graded in units of the model's own error.

**Which pick is graded.** The COUNTED pick, by the track record's own
definition -- the earliest recorded projection per (game, player, market),
read through `store._earliest_recorded` so the two can never disagree
about which row counts. Only a counted pick whose own stamp proves it was
made before its kickoff (`store._label_pick_timing`, failing closed) is
graded: a post-kickoff row is excluded from the graded sets, and every
served row carries `made_before_kickoff` as its proof.

**The grade.** A miss is measured against the prop model's own holdout
error for that stat, read from `models/manifest.json`'s `yardage_metrics`
(the `holdout_mae`/`mae` the training run publishes for the market, when
it publishes one). Bands, identical to NBA's:

* ``On the number`` -- |actual - projected| <= 0.5 x MAE
* ``Close``         -- |actual - projected| <= 1.0 x MAE
* ``Off``           -- |actual - projected| <= 2.0 x MAE
* ``Way off``       -- beyond that

Training publishes a walk-forward `holdout_mae` (and the naive baseline's
`naive_mae`) per yardage market into the manifest -- fitted on earlier
seasons, scored on a season the fit never saw
(`evaluate/player_walk_forward.py`, wired into `train_all` and backfilled
onto an existing manifest by `publish_yardage_holdout_metrics`) -- so the
band is a number actually measured out of sample. A stat whose manifest
entry carries no such figure (a manifest written before the measurement
existed) has no grade: the row is served with its delta and `grade=None`,
never judged against an invented band (and never against the track
record's live MAE, which is measured over the very picks it would be
judging).

**Most confident.** The section ranks a day's projections by how far the
model moved from the player's own rolling average -- |projected -
baseline|, the model's biggest departures from what the player usually
does. The baseline is the backend's own feature discipline
(`features/player_usage.py`: `shift(1)`-then-rolling with
`DEFAULT_ROLL_WINDOW`), reconstructed from the RECORDED actuals in the
tracking store: the mean of the player's previous `DEFAULT_ROLL_WINDOW`
recorded games for that stat, ordered by kickoff, the graded game never
in its own baseline. A player with no recorded prior game has no baseline
in stored data; his rows are graded but cannot be ranked here.

**Delta direction.** ``delta = actual - projected``: positive is the
player beating the projection, negative is the player falling short.

**Anytime TD is not graded here.** It is a probability, not a counting
stat; it stays with the track record's confidence-bucket block. Only the
stat-column markets in `store._YARDAGE_MARKETS` are graded.

**Days.** Completed game days (the kickoff's UTC date) that hold at least
one graded pick, newest first, bounded to `DAYS_LIMIT` days so the
payload stays the size of a few game days rather than the whole record.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import MODELS_DIR
from ..features.player_usage import DEFAULT_ROLL_WINDOW
from . import store

#: The graded counting stats, in the track record's own market order --
#: every stat-column market the prop models project. Derived from
#: `store._YARDAGE_MARKETS` so the graded set and the summarised set are
#: the same set by construction; anytime_td is absent because it is not a
#: stat column (see the module docstring).
STATS = tuple(market for market in store._YARDAGE_MARKETS)

GRADE_ON_THE_NUMBER = "On the number"
GRADE_CLOSE = "Close"
GRADE_OFF = "Off"
GRADE_WAY_OFF = "Way off"

#: Grade bands as multiples of the stat's holdout MAE, tightest first. The
#: comparison carries the same epsilon discipline as NBA's grading (a miss
#: of exactly the band is inside the band).
_BANDS = ((0.5, GRADE_ON_THE_NUMBER), (1.0, GRADE_CLOSE), (2.0, GRADE_OFF))
_EPSILON = 1e-9

#: How many completed game days the payload carries, and how many rows
#: each ranked section shows per day.
DAYS_LIMIT = 4
SECTION_LIMIT = 8

#: The rolling-average window, the backend's own (`features/player_usage.py`
#: declares it once for training and serving alike; the baseline here is
#: that same window over recorded actuals).
BASELINE_WINDOW = DEFAULT_ROLL_WINDOW


def grade_for(abs_delta: float, mae: float | None) -> str | None:
    """The grade for one miss, or None when there is no measured error to
    judge it against (no band is ever invented)."""
    if mae is None or mae <= 0:
        return None
    ratio = abs_delta / mae
    for multiple, label in _BANDS:
        if ratio <= multiple + _EPSILON:
            return label
    return GRADE_WAY_OFF


def _manifest_metric(figures: dict, *keys: str) -> float | None:
    for key in keys:
        value = (figures or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def prop_bands(models_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """The prop models' own published holdout MAE and naive MAE per stat.

    Read from `models/manifest.json`'s `yardage_metrics` -- the artifact
    `models/manifest.py::train_all` writes at fit time. The band is what a
    projection has to miss by before it counts as wrong, so it must be a
    number a training run actually measured on held-out data: the
    walk-forward `holdout_mae`/`mae` (and `naive_mae`) figures are picked
    up verbatim, per stat, with no other change. A manifest entry without
    them (written before the measurement existed) contributes nothing, so
    that stat's rows grade None rather than borrow another stat's scale.
    """
    path = Path(models_dir) / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}, {}
    bands: dict[str, float] = {}
    naive: dict[str, float] = {}
    for stat, figures in (manifest.get("yardage_metrics") or {}).items():
        mae = _manifest_metric(figures or {}, "holdout_mae", "mae")
        if mae is not None:
            bands[str(stat)] = mae
        naive_mae = _manifest_metric(figures or {}, "naive_mae", "naive_holdout_mae")
        if naive_mae is not None:
            naive[str(stat)] = naive_mae
    return bands, naive


def _round1(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)


def _kickoff_iso(value) -> str:
    """The stored commence_time normalised to an ISO instant (the store
    writes both `...T...+00:00` and `... ...+00:00` shapes; the payload
    speaks one)."""
    return store._parse_utc(value).isoformat()


def _rolling_baselines(resolved: pd.DataFrame) -> dict[tuple[str, str], list[tuple[datetime, str, float]]]:
    """{(player_id, market): [(kickoff, game_id, actual), ...] sorted by kickoff}.

    Every recorded actual in the store for a graded stat, dated by its
    game's kickoff. A row whose game the store cannot date cannot be
    ordered against a graded game, so it is left out rather than placed
    by guess.
    """
    series: dict[tuple[str, str], list[tuple[datetime, str, float]]] = {}
    for row in resolved.itertuples():
        if row.market not in STATS or row.game_id is None or pd.isna(row.actual_value):
            continue
        try:
            kickoff = store._parse_utc(row.commence_time)
        except (TypeError, ValueError):
            continue
        series.setdefault((row.player_id, row.market), []).append(
            (kickoff, row.game_id, float(row.actual_value))
        )
    for entries in series.values():
        entries.sort(key=lambda e: (e[0], e[1]))
    return series


def _baseline_for(
    series: dict[tuple[str, str], list[tuple[datetime, str, float]]],
    player_id: str,
    market: str,
    game_id: str,
    kickoff: datetime,
) -> float | None:
    """The player's rolling average going INTO this game: the mean of his
    previous `BASELINE_WINDOW` recorded games for this stat, or None when
    the stored record holds no prior game for him. Only games kicked off
    strictly before this one count, and the graded game is excluded by
    identity, so a row can never leak into its own baseline."""
    entries = series.get((player_id, market))
    if not entries:
        return None
    prior = [value for when, gid, value in entries if when < kickoff and gid != game_id]
    if not prior:
        return None
    window = prior[-BASELINE_WINDOW:]
    return sum(window) / len(window)


def build_prop_grading(
    player_teams: dict[str, str] | None = None,
    *,
    models_dir: Path | None = None,
    days_limit: int = DAYS_LIMIT,
    section_limit: int = SECTION_LIMIT,
) -> dict:
    """The graded player props for the most recent completed game days.

    Computed from stored data only: recorded predictions, their recorded
    outcomes, the games' kickoffs, and the manifest's published MAEs.
    Nothing is recomputed from model files and nothing is fetched; the
    caller (the `/prop-grading` route) caches the result. A store that
    cannot be read -- a deployment whose tracking DB was never created,
    a store written before props existed -- grades EMPTY rather than
    raising: no recorded props is a state, not an error.

    `player_teams` maps player_id -> the player's team (the caller reads
    it from the public snapshot's prop rows, the one place a player's
    team is published); it only places a graded row in its game -- a
    player the map cannot place in the game's two teams is served with
    `team`/`opponent` None rather than a guessed side.
    """
    bands, naive = prop_bands(MODELS_DIR if models_dir is None else models_dir)
    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            stat: {"holdout_mae": bands.get(stat), "naive_mae": naive.get(stat)}
            for stat in STATS
        },
        "days": [],
    }
    try:
        with contextlib.closing(store._connect()) as conn:
            props = pd.read_sql(
                "SELECT * FROM player_prop_predictions WHERE resolved = 1", conn
            )
            games = pd.read_sql(
                "SELECT game_id, home_team, away_team, commence_time FROM game_predictions",
                conn,
            )
    except sqlite3.Error:
        return payload
    if props.empty or games.empty:
        return payload

    labelled = store._label_pick_timing(
        props.merge(games, on="game_id", how="left")
    )
    counted = store._earliest_recorded(labelled, ("game_id", "player_id", "market"))
    graded = counted[
        counted["made_before_kickoff"] & counted["market"].isin(STATS)
    ]
    if graded.empty:
        return payload

    series = _rolling_baselines(labelled)
    teams = player_teams or {}
    stat_order = {stat: i for i, stat in enumerate(STATS)}

    rows: list[dict] = []
    for pick in graded.itertuples():
        kickoff = store._parse_utc(pick.commence_time)
        predicted = float(pick.predicted_value)
        actual = float(pick.actual_value)
        delta = actual - predicted
        team = teams.get(pick.player_id)
        if team == pick.home_team:
            opponent = pick.away_team
        elif team == pick.away_team:
            opponent = pick.home_team
        else:
            # The team map cannot place this player in this game (no
            # snapshot row for him, or a stale side): serve neither side
            # rather than guess one.
            team, opponent = None, None
        baseline = _baseline_for(series, pick.player_id, pick.market, pick.game_id, kickoff)
        rows.append(
            {
                "player": pick.player_name,
                "team": team,
                "opponent": opponent,
                "stat": pick.market,
                "projected": _round1(predicted),
                "actual": _round1(actual),
                "delta": _round1(delta),
                "grade": grade_for(abs(delta), bands.get(pick.market)),
                "made_before_kickoff": True,
                "kickoff": _kickoff_iso(pick.commence_time),
                "_game_id": pick.game_id,
                "_label": f"{pick.away_team} @ {pick.home_team}",
                "_date": kickoff.date().isoformat(),
                "_kickoff": kickoff,
                "_abs_delta": abs(delta),
                "_abs_departure": (
                    abs(predicted - baseline) if baseline is not None else None
                ),
                "_stat_order": stat_order.get(pick.market, 99),
            }
        )
    if not rows:
        return payload

    def publish(row_list: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in row_list
        ]

    days_out = []
    for day in sorted({r["_date"] for r in rows}, reverse=True)[:days_limit]:
        day_rows = [r for r in rows if r["_date"] == day]

        by_game: dict[str, dict] = {}
        for r in day_rows:
            entry = by_game.setdefault(
                r["_game_id"],
                {"game_id": r["_game_id"], "label": r["_label"], "rows": []},
            )
            entry["rows"].append(r)
        games_out = []
        for entry in sorted(
            by_game.values(),
            key=lambda e: (min(r["_kickoff"] for r in e["rows"]), e["game_id"]),
        ):
            entry = dict(entry)
            entry["rows"] = publish(
                sorted(
                    entry["rows"],
                    key=lambda r: (r["player"] or "", r["_stat_order"]),
                )
            )
            games_out.append(entry)

        ranked = [r for r in day_rows if r["_abs_departure"] is not None]
        most_confident = sorted(
            ranked,
            key=lambda r: (-r["_abs_departure"], -r["_abs_delta"], r["player"] or ""),
        )[:section_limit]
        surprises = sorted(
            day_rows, key=lambda r: (-r["_abs_delta"], r["player"] or "")
        )[:section_limit]

        days_out.append(
            {
                "date": day,
                "most_confident": publish(most_confident),
                "biggest_surprises": publish(surprises),
                "games": games_out,
            }
        )
    payload["days"] = days_out
    return payload
