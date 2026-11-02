"""routes.py — game/player prop/track-record endpoints. Thin HTTP layer
over data/features/models/tracking, near-verbatim port of nfl_predictor's
own api/routes.py. Two CFB-specific adaptations: (1) schedules -> games
(data.games); (2) CFBD's Game model carries no spread_line/total_line, so
_lines_for_game pulls them from data/sportsbook_api.py's own spreads/totals
markets instead, matched by team name.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException

from ..config import (
    CURRENT_SEASON,
    DATA_DIR,
    PUBLIC_MODE,
    PUBLIC_SNAPSHOT_PATH,
    PUBLIC_SNAPSHOT_REFRESH_URL,
)
from ..data import games as games_data
from ..data import player_stats, sportsbook_api
from ..data import advanced_stats, cfbd_advanced
from ..features import build as feature_build
from ..features.build import DEFAULT_BLOCKS
from ..features import player_usage, power_ratings
from ..models import game_outcome, manifest, player_props, qb_passing_td, season_projection
from ..odds import value_bets
from ..tracking import prop_grading, store

router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)

_public_snapshot_cache: dict | None = None
_public_snapshot_etag: str | None = None


def _public_snapshot() -> dict:
    """The precomputed data public_snapshot.py generates. Cold-start value
    is whatever was baked into this image at build time; a running process
    then keeps it current via refresh_public_snapshot_from_remote below,
    polled on a timer (see api/main.py's lifespan). Empty dict if none has
    been generated yet, so every PUBLIC_MODE branch below just falls
    through to a live compute instead of a 500."""
    global _public_snapshot_cache
    if _public_snapshot_cache is None:
        _public_snapshot_cache = (
            json.loads(PUBLIC_SNAPSHOT_PATH.read_text()) if PUBLIC_SNAPSHOT_PATH.exists() else {}
        )
    return _public_snapshot_cache


def refresh_public_snapshot_from_remote() -> bool:
    """Re-fetch the precomputed public snapshot from its GitHub raw URL and
    swap it in -- lets the scheduled snapshot-refresh GitHub Action reach
    this running process without a Docker rebuild+redeploy. ETag-conditional
    so an unchanged snapshot costs one small request. Any fetch/parse
    failure is swallowed and leaves the previous snapshot serving -- a
    transient network hiccup here must never blank an already-working
    public site."""
    global _public_snapshot_cache, _public_snapshot_etag
    try:
        headers = {"If-None-Match": _public_snapshot_etag} if _public_snapshot_etag else {}
        resp = requests.get(PUBLIC_SNAPSHOT_REFRESH_URL, headers=headers, timeout=30)
        if resp.status_code == 304:
            return False
        resp.raise_for_status()
        snapshot = resp.json()
    except Exception as exc:
        logger.info("public snapshot remote refresh skipped: %s", exc)
        return False
    _public_snapshot_cache = snapshot
    _public_snapshot_etag = resp.headers.get("ETag")
    return True


def _snapshot_week(season: int, week: int) -> dict | None:
    snap = _public_snapshot()
    if snap.get("season") != season:
        return None
    return snap.get("weeks", {}).get(str(week))


def current_season_and_week() -> tuple[int, int]:
    """Current CFB season/week, anchored to the real schedule's own week-1
    kickoff date rather than a hardcoded month/day guess. A fixed
    `date(season, 8, 20)` anchor drifts every year the season's actual
    opening week doesn't start exactly then (confirmed live: it was a full
    week ahead of the real current week).

    **The week is derived from the schedule, not from 7-day arithmetic.**
    Measuring `((today - anchor).days // 7) + 1` put the rollover on whatever
    weekday week 1 happened to kick off -- a Friday -- so the whole site moved
    to the next gameweek on Friday night with Saturday's games still unplayed.

    The obvious repair, rolling over on Sunday, is also wrong, and the schedule
    says so: measured from the committed snapshot, **week 1 has games on Thu,
    Fri, Sat, Sun AND Mon**, and week 6 has a Wednesday. A Sunday rollover would
    hide week 1's Monday game for two days -- the same defect, two days later.

    So the rule is: **the current week is the first week whose last scheduled
    game has not yet been played.** Friday and Saturday stay in the week whose
    Monday game is still ahead; the week turns over once that game is behind us,
    and a week whose games are all in the past advances to the next one. No
    scheduled game is ever off the board, which is the only property that
    actually matters here.
    """
    today = date.today()
    season = today.year if today.month >= 2 else today.year - 1
    week = None
    try:
        schedule = games_data.fetch_schedules([season])
        gamedays = pd.DataFrame(
            {"week": schedule["week"], "gameday": pd.to_datetime(schedule["gameday"], errors="coerce")}
        ).dropna()
        last_by_week = gamedays.groupby("week")["gameday"].max().sort_index()
        remaining = [int(w) for w, gd in last_by_week.items() if gd.date() >= today]
        if last_by_week.empty:
            raise ValueError("schedule carried no usable gamedays")
        # Past the final week of the season, roll to the one after it rather
        # than clamping to the last week with games, which would go stale.
        week = remaining[0] if remaining else int(last_by_week.index[-1]) + 1
    except Exception:
        # No schedule (cold cache, network, a fetch that raises). Fall back to
        # Sunday-anchored arithmetic: worse than the schedule-derived rule
        # because it cannot see a Monday or Wednesday game, but it still refuses
        # to advance mid-week, which the old 7-day-from-anchor form did.
        anchor = date(season, 8, 20)
        first_close = anchor + timedelta(days=(6 - anchor.weekday()) % 7)
        week = 1 + (today - first_close).days // 7 + (1 if today >= first_close else 0)
    return season, max(1, min(20, week))


@router.get("/current-week")
def get_current_week():
    season, week = current_season_and_week()
    return {"season": season, "week": week}


@router.get("/snapshot-meta")
def get_snapshot_meta() -> dict:
    """When the data this site serves was actually produced.

    The snapshot carries generated_at at its top level, but every route that
    reads it returns a nested block (a week, a standings table) and drops it,
    so no other endpoint can answer "how old are these numbers?". The hub
    needs to say that on the card, and this is also the honest answer for
    anyone tracking how fresh a deployment is.

    Empty dict when no snapshot has been generated yet -- a public deploy
    before its first snapshot -- so this reports source "live" rather than
    raising. That is the real state, not a failure.
    """
    snap = _public_snapshot()
    return {
        "generated_at": snap.get("generated_at"),
        "source": "public_snapshot" if snap else "live",
    }

# Re-fetch the current season's weekly player stats at most this often,
# same TTL rationale as data/games.py's _CURRENT_SEASON_TTL_SECONDS --
# unconditionally force-refreshing here cost ~15-20 CFBD calls per single
# Player Props page view once Task 17 changed the fetch to one call per
# week (see this plan's final-review fix, Task 23, findings C1+C2).
_CURRENT_SEASON_PLAYER_STATS_TTL_SECONDS = 6 * 60 * 60


def _player_stats_cache_missing_a_finished_teams_stats(season: int, games_df: pd.DataFrame) -> bool:
    """True when the cached player-stats parquet exists but has zero rows
    for a team whose game already has a final score -- confirmed live as
    the actual SMU-returns-0-stats bug: CFBD hadn't finished posting one
    side's box score at the moment this cache was written (a fetch that
    landed kickoff-adjacent caught one team's stats but not the other's),
    so the incomplete snapshot got cached and nothing re-checks it until
    the blanket TTL lapses. Checked in *addition* to the TTL below, never
    instead of it -- this only adds a faster trigger, it never skips a
    refresh the TTL would still want."""
    path = player_stats._season_cache_path(season)
    if not path.exists():
        return False
    season_games = games_df[games_df["season"] == season]
    finished = season_games[season_games["home_score"].notna() & season_games["away_score"].notna()]
    if finished.empty:
        return False
    finished_teams = set(finished["home_team"]) | set(finished["away_team"])
    cached_teams = set(pd.read_parquet(path)["recent_team"])
    return not finished_teams.issubset(cached_teams)


def _player_stats_needs_refresh(season: int, games_df: pd.DataFrame) -> bool:
    path = player_stats._season_cache_path(season)
    if not path.exists():
        return True
    if (time.time() - path.stat().st_mtime) > _CURRENT_SEASON_PLAYER_STATS_TTL_SECONDS:
        return True
    return _player_stats_cache_missing_a_finished_teams_stats(season, games_df)


@lru_cache(maxsize=1)
def _load_models_cached() -> dict:
    return manifest.load_models()


# A player on a team with no current-season box score yet -- week 1, or a
# bye-to-opener gap -- has no usage history to roll. That case previously got a
# placeholder dict keyed `passing_yards`, `rushing_yards`, `receiving_yards`,
# `targets`, `carries`, `receptions`. But `predict_props` does
# `feature_row.reindex(feature_cols)`, and `feature_cols` is
# `passing_yards_roll, rushing_yards_roll, ...`. None of those keys matched, so
# every one reindexed to NaN and `.fillna(0)` produced the all-zero origin -- a
# point in feature space the model was never trained on. What came back was the
# intercept, not a prediction: on the 2026-09-27 Azure build, 71% of 2112 rows
# shared a single `anytime_td_prob` as a result.
#
# Two things are wrong and both are fixed below. The keys must be derived from
# PLAYER_FEATURE_COLUMNS so they cannot drift again. And the values must be a
# plausible in-season rate rather than zeros -- an all-zero row is the origin
# whatever it happens to be keyed with.
#
# `_ROSTER_FALLBACK_RATES` is a deliberately unremarkable CFB per-game rate. It
# is a placeholder for a player we know nothing about, not a claim about that
# player, and the two leagues' sibling route skips these players entirely
# instead. Prefer skipping if the branch ever grows a way to do so honestly.
_ROSTER_FALLBACK_RATES: dict[str, float] = {
    "passing_yards_roll": 180.0,
    "rushing_yards_roll": 40.0,
    "receiving_yards_roll": 30.0,
    "targets_roll": 4.0,
    "carries_roll": 8.0,
    "receptions_roll": 3.0,
}


def _roster_fallback_feature_row() -> pd.Series:
    """Feature row for a roster player with no usage history.

    Keyed off `PLAYER_FEATURE_COLUMNS` so the key set cannot drift from what
    `predict_props` reads. Any column without an explicit placeholder rate gets a
    neutral non-zero default, because a zero row is the origin the comment above
    is about.
    """
    return pd.Series(
        {column: _ROSTER_FALLBACK_RATES.get(column, 1.0) for column in player_usage.PLAYER_FEATURE_COLUMNS}
    )


def _load_models_or_503() -> dict:
    """manifest.load_models() raises FileNotFoundError when no manifest has
    ever been trained -- turn that into a friendly 503 instead of letting it
    propagate as a bare unhandled 500. Does not address the larger "ship
    models in the image" / persistent-disk deploy question -- that's out of
    scope for this fix (see this plan's final-review fix, Task 23, finding
    C3).

    It also raises `manifest.StaleArtifactError` when the committed models no
    longer match the code serving them -- a model fitted against a different
    `anytime_td` definition, or against a different feature list. That is the
    same class of "not ready to predict" as an untrained model, so it becomes a
    503 too, carrying the verifier's own message: the whole point of that check
    is that it names both sides, and swallowing the reason into a generic 503
    would throw away the only actionable part. It stays a 503 rather than a
    200-with-empty-predictions because a stale model must not serve predictions
    at all.

    That type, not a bare `ValueError`: `load_manifest` parses JSON, so a
    corrupt `manifest.json` raises `json.JSONDecodeError`, itself a
    `ValueError`. Catching `ValueError` here would report a corrupt file as
    "stale trained model" and point an operator at a retrain instead of at the
    file."""
    try:
        return _load_models_cached()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="No trained model yet — run POST /api/retrain first"
        ) from exc
    except manifest.StaleArtifactError as exc:
        raise HTTPException(status_code=503, detail=f"Stale trained model: {exc}") from exc


def _team_name_matches(cfbd_name: str, odds_name: str) -> bool:
    """Best-effort match between CFBD's short school name (e.g. "Texas")
    and the odds provider's own team name (e.g. "Texas Longhorns" from The
    Odds API, or "Texas" from data/sportsbook_api.py -- close enough to
    CFBD's own naming that this rarely needs its full tolerance for that
    source) -- these are different naming conventions with no shared id
    space. Matches when the CFBD name's words are a prefix of the odds
    name's words (case-insensitive), which covers the common case without
    a full manual mapping table. Real gap acknowledged in the plan/ledger:
    this will still miss genuinely divergent names (e.g. abbreviations,
    "St." vs "State") -- degrades gracefully to no line found, same as
    a missing key or empty odds_df already does."""
    cfbd_words = cfbd_name.lower().split()
    odds_words = odds_name.lower().split()
    return odds_words[: len(cfbd_words)] == cfbd_words


def _lines_for_game(odds_df: pd.DataFrame, home_team: str, away_team: str) -> tuple[float | None, float | None]:
    """CFBD's Game model (unlike nflverse's schedule frame) carries no
    market spread/total -- pull them from data/sportsbook_api.py's own
    spreads/totals markets instead (odds_df's shape is provider-agnostic;
    data/odds_api.py, the previous source, produces the exact same shape),
    matched by team name (best-effort -- see _team_name_matches).
    Best-effort: returns (None, None) for either line the odds feed
    doesn't have for this matchup (a common case outside Power-conference
    games, per the design spec)."""
    if odds_df is None or odds_df.empty:
        return None, None
    matches = odds_df[
        odds_df["home_team"].map(lambda t: _team_name_matches(home_team, t))
        & odds_df["away_team"].map(lambda t: _team_name_matches(away_team, t))
    ]
    if matches.empty:
        return None, None
    # _team_name_matches is a prefix match, not an exact one -- two distinct
    # FBS teams can share a short-name prefix (e.g. "Miami" matches both
    # "Miami Hurricanes" and "Miami (OH) RedHawks"). If the matched rows
    # disagree on which real (home_team, away_team) pair they belong to,
    # trusting the first row would silently attach a DIFFERENT game's line
    # to this one -- worse than the graceful (None, None) this function is
    # documented to return on a genuine miss. Refuse to guess: treat an
    # ambiguous match the same as no match.
    if matches[["home_team", "away_team"]].drop_duplicates().shape[0] > 1:
        return None, None

    spread_line = None
    spread_rows = matches[
        (matches["market"] == "spreads")
        & matches["outcome_name"].map(lambda t: _team_name_matches(home_team, t))
    ]
    if not spread_rows.empty:
        # odds_df's `point` is the raw handicap (negative when the home
        # team is favored) regardless of provider -- confirmed against real
        # data/sportsbook_api.py data too (e.g. a -4.5 favorite). Negate it
        # so spread_line means "home expected margin" -- the same
        # convention nflverse uses and that game_outcome.margin_to_probabilities
        # documents and is pinned against (see this plan's final review,
        # finding B2).
        spread_line = -float(spread_rows.iloc[0]["point"])

    total_line = None
    total_rows = matches[(matches["market"] == "totals") & (matches["outcome_name"] == "Over")]
    if not total_rows.empty:
        total_line = float(total_rows.iloc[0]["point"])

    return spread_line, total_line


#: Committed + current-season efficiency for the total model's epa-family block, rebuilt at most hourly. The snapshot job is
#: where the current-season pull (1 CFBD call per ~6 days, disk-cached) happens; a request-time caller only ever reads files.
_EFFICIENCY_TTL_SECONDS = 3600
_efficiency_cache: dict[tuple, tuple[float, feature_build.Aux]] = {}


def _efficiency_aux(games_df: pd.DataFrame) -> feature_build.Aux | None:
    """The efficiency aux for the seasons in `games_df`, or None when there is none. Callers must treat None as
    "cannot price this", never as zeros."""
    seasons = tuple(sorted(int(s) for s in games_df["season"].dropna().unique())) if "season" in games_df.columns else ()
    hit = _efficiency_cache.get(seasons)
    if hit and time.time() - hit[0] < _EFFICIENCY_TTL_SECONDS:
        return hit[1]
    try:
        frames = [cfbd_advanced.efficiency_frame_from_disk(years=list(seasons))]
        if seasons and seasons[-1] == CURRENT_SEASON:
            frames.append(cfbd_advanced.current_season_efficiency(CURRENT_SEASON, DATA_DIR / "cache" / "cfbd"))
        frames = [f for f in frames if len(f)]
        if not frames:
            return None
        frame = pd.concat(frames, ignore_index=True).assign(game_id=lambda d: d["game_id"].astype(str))
        aux = feature_build.Aux(efficiency=frame.drop_duplicates(["game_id", "team"], keep="last"))
    except Exception:  # noqa: BLE001 - fail closed: no aux means no total, never a guess
        logger.exception("efficiency aux unavailable; the total prediction is withheld")
        return None
    _efficiency_cache[seasons] = (time.time(), aux)
    return aux


def _predict_total(models: dict, home: str, away: str, games_df: pd.DataFrame, gameday, conference_game) -> float | None:
    """The total-points prediction, or None when it cannot be priced honestly (spec needs efficiency aux that is
    missing). Models with no `total_spec` use the shared feature row, exactly as before."""
    spec = models.get("total_spec")
    if not spec:
        return None
    aux = _efficiency_aux(games_df)
    if aux is None:
        return None
    row = feature_build.build_features_for_game(
        home, away, games_df, gameday=gameday, conference_game=conference_game,
        blocks=tuple(spec["blocks"]), aux=aux,
    )
    X = row.reindex(spec["feature_cols"]).fillna(0)
    return float(models["total_model"].predict(X.to_frame().T)[0])  # named columns: the model was fitted on a frame


def _predict_game_from_models(
    models: dict, home: str, away: str, games_df: pd.DataFrame,
    spread_line: float | None = None, total_line: float | None = None,
    gameday=None, conference_game=None,
) -> dict:
    # Whatever blocks the fitted margin/win model was trained on, in the same order it recorded them. `aux` stays
    # None for them: `_assemble` refuses a block with no aux rather than zero-filling it. The TOTAL model has its own
    # spec (see manifest.TOTAL_SPEC) and gets its own aux in `_predict_total`.
    blocks = tuple(models.get("feature_blocks", DEFAULT_BLOCKS))
    feature_row = feature_build.build_features_for_game(
        home, away, games_df, gameday=gameday, conference_game=conference_game,
        blocks=blocks, aux=None,
    )
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0)

    if models["chosen_candidate"] == "elo":
        predicted_margin = game_outcome.predict_margin_elo(
            {"points_per_rating_point": game_outcome.ELO_POINTS_PER_RATING_POINT},
            feature_row["rating_diff"], feature_row["home_rest_days"], feature_row["away_rest_days"],
        )
    else:
        predicted_margin = float(models["game_outcome_model"].predict(X.to_numpy().reshape(1, -1))[0])

    if models.get("total_spec"):
        predicted_total = _predict_total(models, home, away, games_df, gameday, conference_game)
    else:
        predicted_total = float(models["total_model"].predict(X.to_numpy().reshape(1, -1))[0])
    total_sigma = models["total_sigma"] if predicted_total is not None else None

    result = game_outcome.margin_to_probabilities(
        predicted_margin, models["sigma"],
        spread_line=spread_line, total_line=total_line,
        predicted_total=predicted_total, total_sigma=total_sigma,
    )
    # Not part of the public API response shape, just handed straight to
    # whichever caller wants it (season_projection.py sums this across a
    # team's remaining games for a projected point differential) -- extra
    # dict keys are harmless since this project doesn't validate the
    # /prediction endpoint's response against a schema.
    result["predicted_margin"] = predicted_margin
    result["predicted_total"] = predicted_total
    result["sigma"] = models["sigma"]
    result["total_sigma"] = total_sigma
    result["model_version"] = models.get("model_version")
    return result


def _load_game_history(season: int) -> pd.DataFrame:
    """Historical + requested-season game data for feature building. A
    still-in-progress CURRENT_SEASON needs the always-refetch
    fetch_current_season_partial() instead of load_training_data()'s
    per-season cache, or every request that season would return the same
    stale games-so-far snapshot forever."""
    history_seasons = games_data.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_df = games_data.load_training_data(history_seasons)
        current_df = games_data.fetch_current_season_partial()
        return pd.concat([historical_df, current_df], ignore_index=True)
    return games_data.load_training_data(history_seasons + [season])


def _load_player_history(season: int) -> pd.DataFrame:
    """Historical + requested-season player stats for feature building.
    player_stats.fetch_weekly_player_stats needs each season's games_df to
    attach week numbers to CFBD's raw per-game box scores (see
    data/player_stats.py), so this loads game history first."""
    history_seasons = games_data.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_games = games_data.load_training_data(history_seasons)
        current_games = games_data.fetch_current_season_partial()
        games_df = pd.concat([historical_games, current_games], ignore_index=True)
        historical_df = player_stats.fetch_weekly_player_stats(history_seasons, games_df)
        current_df = player_stats.fetch_weekly_player_stats(
            [season], games_df, force_refresh=_player_stats_needs_refresh(season, games_df)
        )
        # Read-side guard. player_stats drops these rows at ingest, but the
        # cache on the VPS was written before that guard existed and nothing
        # revisits it, so the live hub kept serving them after the ingest fix
        # landed. This also covers /players/{season}/{week}/props, which reads
        # the same frame, and player_hub's leaderboards, built from it too.
        return player_stats.drop_team_rows(
            pd.concat([historical_df, current_df], ignore_index=True))
    all_seasons = history_seasons + [season]
    games_df = games_data.load_training_data(all_seasons)
    # Read-side guard, same reason as the branch above.
    return player_stats.drop_team_rows(
        player_stats.fetch_weekly_player_stats(all_seasons, games_df))


@router.get("/games")
def get_games(season: int, week: int):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap["games"]
    return _get_games_live(season, week)


def _get_games_live(season: int, week: int):
    games = games_data.fetch_week_games(season, week)
    games = games.astype(object).where(pd.notna(games), None)
    records = games.to_dict("records")

    # CFBD's Game model carries no spread_line/total_line (see module
    # docstring), unlike nfl_data_py's schedule rows -- without this, the
    # frontend's list-view GameCard (which reads game.spread_line directly,
    # the same field NFL's /games rows already have) never has anything to
    # show, even once a real market line exists via
    # _get_game_prediction_live's per-game fetch. Only for the current week,
    # same quota-safety gate as everywhere else this project calls
    # sportsbook_api -- and doing it once here for the whole week (instead
    # of once per game) is what /prediction's later per-game calls then
    # reuse from sportsbook_api's own on-disk cache, at no extra request
    # cost.
    if records and (season, week) == current_season_and_week():
        odds_df = sportsbook_api.fetch_game_odds(games[["home_team", "away_team"]])
        for record in records:
            spread_line, total_line = _lines_for_game(odds_df, record["home_team"], record["away_team"])
            record["spread_line"] = spread_line
            record["total_line"] = total_line

    return records


@router.get("/games/{season}/{week}/{game_id}/prediction")
def get_game_prediction(season: int, week: int, game_id: str):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            pred = snap["predictions"].get(game_id)
            if pred is not None:
                return pred
    return _get_game_prediction_live(season, week, game_id)


def _get_game_prediction_live(season: int, week: int, game_id: str):
    games = games_data.fetch_week_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0]

    models = _load_models_or_503()
    # Exclude the game's own row from its feature history -- fetch_week_games
    # (unlike the old fetch_upcoming_games) makes already-finished games
    # reachable here too, and without this exclusion a finished game's
    # prediction would leak its own result into its own features.
    history = _load_game_history(season)
    history = history[history["game_id"] != game_id]
    # Only ever fetch real odds for the actual current week -- sportsbooks
    # don't post real lines many weeks out anyway (confirmed live: a game
    # 4 months away came back with zero priced markets), and this function
    # is called once per game across public_snapshot.py's whole multi-week
    # rebuild window (~350 games/run) -- fetching odds for every one of
    # those would be ~350 Sportsbook API requests against its 150/day cap,
    # in a single snapshot run alone.
    if (season, week) == current_season_and_week():
        odds_df = sportsbook_api.fetch_game_odds(matches[["home_team", "away_team"]])
    else:
        odds_df = pd.DataFrame()
    spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
    prediction = _predict_game_from_models(
        models, game["home_team"], game["away_team"], history,
        spread_line=spread_line, total_line=total_line,
        gameday=game.get("gameday"), conference_game=game.get("conference_game"),
    )
    return prediction


@router.get("/predictions/{season}/{week}/batch")
def get_predictions_batch(season: int, week: int):
    """Bare dict[game_id, GamePrediction] for the whole week -- a distinct
    path from GET /predictions/{season}/{week} (which already returns a
    different shape: a list of {game_id, status, verdict} tracking
    objects). Lets the frontend preload every game's prediction in one
    request instead of one request per game."""
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap["predictions"]
    return _get_predictions_batch_live(season, week)


def _get_predictions_batch_live(season: int, week: int) -> dict:
    games = games_data.fetch_week_games(season, week)
    if games.empty:
        return {}

    models = _load_models_or_503()
    history = _load_game_history(season)

    # Fetch odds ONCE for the whole week, not once per game inside the loop
    # below -- CFB's Sportsbook API has a hard 150 requests/day quota
    # shared across projects (already a source of real incidents this
    # session), same reasoning and same pattern as _get_games_live above.
    if (season, week) == current_season_and_week():
        odds_df = sportsbook_api.fetch_game_odds(games[["home_team", "away_team"]])
    else:
        odds_df = pd.DataFrame()

    predictions: dict[str, dict] = {}
    for _, game in games.iterrows():
        try:
            # Exclude the game's own row from its feature history, same as
            # _get_game_prediction_live -- otherwise a finished game's
            # prediction leaks its own result into its own features.
            game_history = history[history["game_id"] != game["game_id"]]
            spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
            predictions[game["game_id"]] = _predict_game_from_models(
                models, game["home_team"], game["away_team"], game_history,
                spread_line=spread_line, total_line=total_line,
                gameday=game.get("gameday"), conference_game=game.get("conference_game"),
            )
        except Exception:
            # One bad game must not fail the whole batch -- mirrors
            # public_snapshot.py's _build_week discipline exactly.
            logger.exception("batch prediction failed for game_id=%s", game.get("game_id"))
            continue
    return predictions


@router.get("/players/{season}/{week}/props")
def get_player_props(season: int, week: int):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap["player_props"]
    return _get_player_props_live(season, week)


def _get_player_props_live(season: int, week: int):
    try:
        models = _load_models_cached()
        player_history = _load_player_history(season)

        # Standardize team name variants to prevent mismatches
        player_history["recent_team"] = player_history["recent_team"].replace({
            "Southern Methodist": "SMU"
        })

        # 1. Fetch upcoming games for this specific week to find active teams (e.g., FSU vs SMU)
        upcoming_games = games_data.fetch_upcoming_games(season, week)
        if upcoming_games.empty:
            upcoming_games = games_data.fetch_current_season_partial()
            upcoming_games = upcoming_games[upcoming_games["week"] == week]

        if upcoming_games.empty:
            return []

        active_teams = set(upcoming_games["home_team"]).union(set(upcoming_games["away_team"]))

        # 2. Extract players present in historical/current stats for active teams
        latest_players = (
            player_history[
                (player_history["season"] == season) & 
                (player_history["recent_team"].isin(active_teams))
            ]
            [["player_id", "player_name", "position", "recent_team"]]
            .drop_duplicates("player_id")
        )

        found_teams = set(latest_players["recent_team"].unique())
        missing_teams = active_teams - found_teams

        # 3. Fallback: a team with no current-season stats yet (week 1, or a
        # bye-to-opener gap) pulls from the cached full-FBS roster instead —
        # fetched at most once a week (data/player_stats.py's
        # fetch_season_roster), not once per request per missing team like
        # the old CFBD TeamsApi.get_roster fallback did.
        if missing_teams:
            try:
                roster = player_stats.fetch_season_roster(season)
                fallback = roster[
                    roster["recent_team"].isin(missing_teams) & roster["position"].isin({"WR", "TE", "RB", "QB"})
                ]
                if not fallback.empty:
                    latest_players = pd.concat([latest_players, fallback], ignore_index=True).drop_duplicates("player_id")
            except Exception as roster_err:
                logger.warning("Failed to load roster fallback for teams=%s: %s", missing_teams, roster_err)

        results = []
        for _, player in latest_players.iterrows():
            try:
                if "roster_" in str(player["player_id"]) or (player["player_id"].isdigit() and not (player_history["player_id"] == player["player_id"]).any()):
                    feature_row = _roster_fallback_feature_row()
                else:
                    feature_row = player_usage.build_features_for_player(player["player_id"], player_history)
                
                if feature_row is None:
                    continue
                
                props = player_props.predict_props(models["player_models"], feature_row, position=player["position"])

                # QB passing TDs. Its OWN key rather than a member of
                # `predict_props`'s flat `{market: float}` result, because a call
                # carries a line, a side, both tail probabilities and the
                # distribution it chose -- a shape that does not fit beside
                # `{"passing_yards": 241.0}` without loosening every consumer.
                #
                # QBs only, and the gate is the POSITION, not the fit: `passing_tds`
                # is 0 for every non-QB, so a fitted model will happily produce a
                # real-looking over/under for a running back who never threw.
                #
                # Contained: a market that raises costs this one field, not the
                # player's yardage and anytime-TD rows, which are what the page has
                # today. Every other exception handler in this loop is per-PLAYER
                # and `continue`s, so without this the whole prop row would vanish.
                qb_call = None
                if player["position"] == "QB":
                    try:
                        qb_call = qb_passing_td.qb_passing_td_call(
                            models["player_models"].get(qb_passing_td.PASSING_TD_MARKET), feature_row
                        )
                    except Exception as qb_err:
                        logger.warning(
                            "QB passing-TD call unavailable for player_id=%s: %s",
                            player.get("player_id"), qb_err,
                        )
                results.append({
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "recent_team": player["recent_team"],
                    "position": player["position"],
                    # Always None, and that is the correct answer, not a
                    # placeholder waiting to be filled in. CFBD's
                    # game-player-stats endpoint has no starter field at all --
                    # its athlete objects carry exactly `id`, `name`, `stat`
                    # (verified three ways on 2026-09-28; the full evidence and
                    # the two refuted heuristics are recorded at the athlete
                    # flatten in data/player_stats.py, which is where a future
                    # session would come to "fix" this).
                    #
                    # `None` means "this sport has no depth-chart data" and the UI
                    # renders a visible "Projected order -- no depth-chart feed"
                    # line. `False` means "this player is known to be on the
                    # bench" and renders a bench row -- a claim about data this
                    # project does not have, which is why a failed depth-chart
                    # fetch in the NFL sibling is None for the same reason.
                    # Never "fix" this to False, and never derive it from list
                    # position in the athletes array or from mere appearance in
                    # the box score: both are refuted, with numbers, in the
                    # comment named above.
                    #
                    # The pair is emitted together so a shared frontend can read
                    # `is_starter` and `depth_slot` off either sport without a
                    # per-sport branch.
                    "is_starter": None,
                    "depth_slot": None,
                    # Always present, and `None` means "no call could be made" --
                    # no fitted model, or a QB the artefact refused to score. A
                    # frontend reading this key must treat None as "not offered",
                    # never as a 0.0 line.
                    "qb_passing_td": qb_call,
                    **props,
                })
            except Exception as player_err:
                logger.warning("Failed to predict props for player_id=%s: %s", player.get("player_id"), player_err)
                continue

        # 4. Strict position filter to ensure only offensive skill players make it into the final API output
        valid_positions = {"WR", "TE", "RB", "QB"}
        results = [p for p in results if p["position"] in valid_positions]

        return results
    except Exception as e:
        logger.exception("Failed to load player props for season=%s week=%s", season, week)
        return []

def _full_track_record() -> dict:
    result = store.get_track_record()
    result["calibration"] = store.get_calibration()
    return result


def _drop_per_pick_rows(result: dict) -> dict:
    """Strip the row-level pick lists from a track-record payload, in place.

    Only `games.per_pick` and `player_props.per_pick` go; every summary field
    stays exactly what the full payload carries. Those rows are the bulk of
    the payload -- this service's full record ran to ~5.8 MB, essentially
    all of it `player_props.per_pick` rows no surface renders -- and are
    served from `/track-record/picks` for the one reader action that needs
    them: opening the every-pick fold.
    """
    for section in ("games", "player_props"):
        block = result.get(section)
        if isinstance(block, dict):
            block.pop("per_pick", None)
    return result


@router.get("/track-record")
def get_track_record(picks: str | None = None):
    result = _full_track_record()
    if picks == "deferred":
        # Payload-split contract with the Sports frontend (Sports_Predictor
        # PR #81): the summary without the row-level picks. Any other value --
        # or none -- keeps the historical full payload, so a consumer that
        # predates the split sees no change at all.
        _drop_per_pick_rows(result)
    return result


@router.get("/track-record/picks")
def get_track_record_picks():
    """The row-level game picks the full `/track-record` payload inlines as
    `games.per_pick`, served on their own so the summary can defer them."""
    games = _full_track_record().get("games") or {}
    return {"per_pick": games.get("per_pick") or []}


_prop_grading_cache: dict | None = None
_prop_grading_cached_at: float = 0.0
_PROP_GRADING_TTL_SECONDS = 300


def _player_team_map() -> dict[str, str]:
    """player_id -> team, from the public snapshot's prop rows.

    The tracking store never recorded a player's team (the snapshot tick
    mapped team -> game at record time and stored only the game), so the
    snapshot -- the one published place a prop row carries its player's
    team -- is the source. Later weeks win: `recent_team` is the player's
    current side, and a graded row is only placed when that side is one
    of its game's two teams (see tracking/prop_grading.py).
    """
    teams: dict[str, str] = {}
    snap = _public_snapshot()
    for week in (snap.get("weeks") or {}).values():
        for row in (week or {}).get("player_props") or []:
            player_id, team = row.get("player_id"), row.get("recent_team")
            if player_id and team:
                teams[str(player_id)] = team
    return teams


@router.get("/prop-grading")
def get_prop_grading():
    """Every counted, pre-kickoff player projection for the most recent
    completed game days, graded against the actual box score -- the
    per-pick counterpart of the track record's per-market MAE, mirroring
    NBA's prop grading. Computed from the tracking store only (never a
    live pipeline, never a fetch) and cached on a short TTL: the payload
    changes only when a game reconciles, and a Saturday slate is ~1,600
    graded rows the reader should not pay to rebuild per visit."""
    global _prop_grading_cache, _prop_grading_cached_at
    now = time.time()
    if _prop_grading_cache is not None and now - _prop_grading_cached_at < _PROP_GRADING_TTL_SECONDS:
        return _prop_grading_cache
    payload = prop_grading.build_prop_grading(_player_team_map())
    _prop_grading_cache = payload
    _prop_grading_cached_at = now
    return payload


@router.get("/kalshi-feed")
def get_kalshi_feed():
    """Read-only feed for the Algo Trade Hub: frozen pre-game snapshots for games that have not
    kicked off, plus reliability buckets over graded pre-game snapshots.

    Served from the tracking database only, so it never recomputes a prediction: a live forecast
    is a different number from the snapshot the hub graded, and recomputing would swap the series
    out from under the calibration check with no error anywhere.
    """
    return {
        "sport": "cfb",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lead_hours": SNAPSHOT_LEAD_HOURS,
        "games": store.get_feed_predictions(),
        "calibration": store.get_calibration(),
    }


@router.get("/games/{game_id}/verdict")
def get_game_verdict(game_id: str):
    verdict = store.get_game_verdict(game_id)
    if verdict is None:
        raise HTTPException(status_code=404, detail="Game not tracked or not yet resolved")
    return verdict


@router.get("/predictions/{season}/{week}")
def get_predictions_for_week(season: int, week: int):
    # A union, not an if/else fallback: a week in progress has both
    # already-final games and still-upcoming ones, and the old if/else
    # dropped every finished game (and its verdict) whenever any game in
    # the week was still unplayed (see this plan's final review, finding B3).
    upcoming = games_data.fetch_upcoming_games(season, week)
    finished = games_data.load_training_data(seasons=[season])
    finished = finished[finished["week"] == week]
    games = pd.concat([finished, upcoming], ignore_index=True).drop_duplicates(subset="game_id")
    return store.get_predictions_for_week(season, week, games)


@router.get("/standings")
def get_standings(season: int = CURRENT_SEASON):
    if PUBLIC_MODE:
        snap = _public_snapshot()
        if snap.get("season") == season and "standings" in snap:
            return snap["standings"]
    return _get_standings_live(season)


def _get_standings_live(season: int):
    schedule = games_data.fetch_schedules([season], force_refresh=(season == CURRENT_SEASON))
    played = schedule[schedule["home_score"].notna() & schedule["away_score"].notna()]
    remaining = schedule[schedule["home_score"].isna()]

    current_records = season_projection.compute_current_records(played)
    team_conferences = season_projection.build_team_conferences(schedule)

    models = _load_models_cached()
    history = _load_game_history(season)

    def predict_fn(home: str, away: str) -> dict:
        return _predict_game_from_models(models, home, away, history)

    return season_projection.project_standings(remaining, current_records, team_conferences, predict_fn)


@router.get("/power-rankings")
def get_power_rankings(season: int = CURRENT_SEASON):
    if PUBLIC_MODE:
        snap = _public_snapshot()
        if snap.get("season") == season and "power_rankings" in snap:
            return snap["power_rankings"]
    return _get_power_rankings_live(season)


def _get_power_rankings_live(season: int) -> dict:
    """Elo-style power ratings for every FBS team, exposed for the first
    time via this endpoint -- features/power_ratings.py's final_ratings()
    was previously only ever consumed internally as a training feature.
    Ratings are computed off the full multi-season history (so a team's
    rating carries continuity across season boundaries, same as it does as
    a training feature); win/loss/tie record and conference are scoped to
    just this season's own games, same as _get_standings_live already does.
    CFB has no divisions (only conferences realign; see
    models/season_projection.py's own docstring) -- unlike NFL's power
    rankings, no `division` field is included here."""
    history = _load_game_history(season)
    ratings = power_ratings.final_ratings(history)

    season_games = history[history["season"] == season]
    played = season_games[season_games["home_score"].notna() & season_games["away_score"].notna()]
    records = season_projection.compute_current_records(played)
    team_conferences = season_projection.build_team_conferences(season_games)

    rows = []
    for team, conference in team_conferences.items():
        record = records.get(team, {"wins": 0, "losses": 0, "ties": 0})
        rows.append({
            "team": team,
            "rating": ratings.get(team, power_ratings.DEFAULT_START_RATING),
            "wins": record["wins"],
            "losses": record["losses"],
            "ties": record["ties"],
            "conference": conference,
        })
    rows.sort(key=lambda r: r["rating"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    return {"season": season, "rankings": rows}


@router.get("/teams/{team}/form")
def get_team_form(team: str, season: int = CURRENT_SEASON, n: int = 5):
    history = _load_game_history(season)
    return _team_form(team, history, n, season)


def _team_form(team: str, history: pd.DataFrame, n: int, season: int) -> dict:
    """Last n played games' W/L/T + score for `team`, oldest-to-newest --
    pure filter/reshape of the multi-season history every other endpoint
    already loads via _load_game_history, no new data fetch. Restricted to
    `season` -- history spans multiple seasons for power-rating continuity,
    but recent form should only ever reflect the current season."""
    involved = history[
        (history["season"] == season)
        & ((history["home_team"] == team) | (history["away_team"] == team))
        & history["home_score"].notna() & history["away_score"].notna()
    ].sort_values("gameday")
    recent = involved.tail(n)

    rows = []
    for _, g in recent.iterrows():
        is_home = g["home_team"] == team
        team_score = g["home_score"] if is_home else g["away_score"]
        opponent_score = g["away_score"] if is_home else g["home_score"]
        opponent = g["away_team"] if is_home else g["home_team"]
        result = "W" if team_score > opponent_score else "L" if team_score < opponent_score else "T"
        rows.append({
            "game_id": g["game_id"],
            "opponent": opponent,
            "is_home": bool(is_home),
            "result": result,
            "team_score": int(team_score),
            "opponent_score": int(opponent_score),
            "gameday": str(g["gameday"]),
        })
    return {"team": team, "recent_form": rows}


@router.get("/games/{game_id}/head-to-head")
def get_head_to_head(game_id: str, season: int = CURRENT_SEASON, week: int = 1, n_seasons: int = 8):
    return _get_head_to_head_live(game_id, season, week, n_seasons)


def _get_head_to_head_live(game_id: str, season: int, week: int, n_seasons: int) -> dict:
    """Meetings between game_id's two teams across the last n_seasons --
    resolves the two teams from the given season/week's slate, then filters
    the same multi-season history _load_game_history already loads for
    every other endpoint (no new data fetch)."""
    games = games_data.fetch_week_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0]
    home, away = game["home_team"], game["away_team"]

    history = _load_game_history(season)
    min_season = season - n_seasons + 1
    history = history[history["season"] >= min_season]

    pair_mask = (
        ((history["home_team"] == home) & (history["away_team"] == away))
        | ((history["home_team"] == away) & (history["away_team"] == home))
    )
    meetings_df = history[
        pair_mask
        & history["home_score"].notna() & history["away_score"].notna()
        & (history["game_id"] != game_id)
    ].sort_values("gameday")

    meetings = [
        {
            "game_id": g["game_id"],
            "season": int(g["season"]),
            "gameday": str(g["gameday"]),
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_score": int(g["home_score"]),
            "away_score": int(g["away_score"]),
        }
        for _, g in meetings_df.iterrows()
    ]
    return {"game_id": game_id, "meetings": meetings}


@router.get("/hub/teams")
def get_hub_teams(season: int = CURRENT_SEASON):
    if PUBLIC_MODE:
        snap = _public_snapshot()
        if snap.get("season") == season and snap.get("hub_teams"):
            return snap["hub_teams"]
    return _get_hub_teams_live(season)


def _get_hub_teams_live(season: int) -> dict:
    advanced = advanced_stats.fetch_advanced(season)
    games = _load_game_history(season)
    return {"season": season, "teams": advanced_stats.team_hub(advanced, games, season),
            "advanced_available": bool(advanced)}


@router.get("/hub/players")
def get_hub_players(season: int = CURRENT_SEASON):
    if PUBLIC_MODE:
        snap = _public_snapshot()
        if snap.get("season") == season and snap.get("hub_players"):
            return snap["hub_players"]
    return _get_hub_players_live(season)


def _get_hub_players_live(season: int) -> dict:
    weekly = _load_player_history(season)
    ppa = advanced_stats.fetch_player_ppa(season)
    return {"season": season, **advanced_stats.player_hub(weekly, ppa, season)}


@router.post("/retrain")
def retrain():
    if PUBLIC_MODE:
        raise HTTPException(status_code=403, detail="Retraining is disabled in public mode")
    result = manifest.train_all()
    _load_models_cached.cache_clear()
    return {"trained_at": result["trained_at"], "chosen_candidate": result["chosen_candidate"]}


def _attach_game_id(stats_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """Join weekly player stats to the game_id of the game each row's
    player actually played in, matched by season/week/team -- weekly
    stats carry a team and week but no game_id of their own -- unless the data
    layer already supplied one (`player_stats.KEEP_COLUMNS` includes `game_id`
    now), in which case that exact id is kept: merging a second `game_id` in
    would produce `game_id_x`/`game_id_y` and no `game_id` at all."""
    if "game_id" in stats_df.columns:
        return stats_df
    if stats_df.empty or games_df.empty:
        return stats_df.iloc[0:0]
    home = games_df[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "recent_team"})
    away = games_df[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "recent_team"})
    team_game = pd.concat([home, away], ignore_index=True)
    return stats_df.merge(team_game, on=["season", "week", "recent_team"], how="inner")


# How close to kickoff a game's prediction is frozen. The first snapshot
# inside this window is kept forever (INSERT OR IGNORE), so it is the one the
# track record grades and the Kalshi feed serves. 48h puts Saturday games'
# snapshots on Thursday, after the previous Saturday's results are in.
SNAPSHOT_LEAD_HOURS = float(os.getenv("SNAPSHOT_LEAD_HOURS", "48"))


def _games_to_snapshot(season: int, week: int, now: datetime, lead_hours: float | None = None) -> pd.DataFrame:
    """Upcoming games from this week and next whose kickoff is within the
    lead window. Next week is included because current_season_and_week()
    rolls over on the date of week 1's first kickoff, not on a fixed
    football-week boundary.

    The window filters on the UPPER bound only: past games still reach
    `record_game_predictions`, which rejects them itself, and the tick's own
    test depends on that.
    """
    lead = timedelta(hours=SNAPSHOT_LEAD_HOURS if lead_hours is None else lead_hours)
    frames = [f for f in (games_data.fetch_upcoming_games(season, wk) for wk in (week, week + 1)) if not f.empty]
    if not frames:
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True)
    kickoff = pd.to_datetime(games["gameday"], utc=True, errors="coerce")
    horizon = pd.Timestamp(now + lead)
    # A NaT kickoff cannot be placed relative to the window, and snapshotting it would freeze a
    # prediction at an unknown distance from kickoff.
    return games[kickoff.notna() & (kickoff <= horizon)].drop_duplicates("game_id").reset_index(drop=True)


def background_tracking_tick(season: int, week: int) -> None:
    """Snapshot upcoming-game (and player-prop) predictions inside the lead
    window, then reconcile anything now resolved. Called on a timer from
    api/main.py's lifespan."""
    try:
        models = _load_models_cached()
    except Exception as exc:
        logger.warning("background_tracking_tick skipped: %s", exc)
        return

    games = _games_to_snapshot(season, week, datetime.now(timezone.utc))
    if not games.empty:
        history = _load_game_history(season)
        odds_df = sportsbook_api.fetch_game_odds(games[["home_team", "away_team"]])
        predictions = []
        for _, game in games.iterrows():
            try:
                spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
                pred = _predict_game_from_models(
                    models, game["home_team"], game["away_team"], history,
                    spread_line=spread_line, total_line=total_line,
                    gameday=game.get("gameday"), conference_game=game.get("conference_game"),
                )
                predictions.append(
                    {
                        "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                        "commence_time": str(game["gameday"]),
                        "home_spread_line": spread_line, "total_line": total_line,
                        "season": season, "week": int(game["week"]) if pd.notna(game.get("week")) else week,
                        **pred,
                    }
                )
            except Exception:
                logger.exception("prediction failed for game_id=%s", game.get("game_id"))
                continue
        try:
            store.record_game_predictions(predictions)
        except Exception:
            logger.exception("record_game_predictions failed")

        try:
            # THIS week's games only. `_get_player_props_live(season, week)` is this week's prop feed
            # (it can fall back to the current week), while `games` also holds next week's -- so a
            # team playing in both had its prop stored under next week's game_id, and
            # `record_player_prop_predictions` is INSERT OR IGNORE, which would freeze that wrong
            # snapshot forever.
            team_to_game = {}
            for _, g in games[games["week"] == week].iterrows():
                team_to_game[g["home_team"]] = g["game_id"]
                team_to_game[g["away_team"]] = g["game_id"]
            prop_rows = []
            for prop in _get_player_props_live(season, week):
                game_id = team_to_game.get(prop["recent_team"])
                if game_id is None:
                    continue
                prop_rows.append({
                    "game_id": game_id, "player_id": prop["player_id"], "player_name": prop["player_name"],
                    "market": "anytime_td", "predicted_value": prop["anytime_td_prob"],
                    "position": prop["position"],
                })
                for market in player_props.POSITION_MARKETS.get(prop["position"], []):
                    value = prop.get(market)
                    if value is not None:
                        prop_rows.append({
                            "game_id": game_id, "player_id": prop["player_id"], "player_name": prop["player_name"],
                            "market": market, "predicted_value": value,
                            "position": prop["position"],
                        })
            store.record_player_prop_predictions(prop_rows)
        except Exception:
            logger.exception("player prop snapshot failed for season=%s week=%s", season, week)

    try:
        completed = games_data.fetch_current_season_partial()
        store.reconcile_game_predictions(completed[["game_id", "home_score", "away_score"]])
    except Exception:
        logger.exception("reconcile_game_predictions failed")
        completed = pd.DataFrame()

    # Backfill games that finished before this tracker ever ran once
    # (e.g. background_tracking_tick wasn't wired up / wasn't running yet)
    # -- reconcile only ever updates an EXISTING snapshot, so a game with
    # no snapshot at all would otherwise never get a verdict. Excludes
    # each game from its own history so the prediction still reflects
    # strictly pre-game information, not this game's own result.
    try:
        if not completed.empty:
            untracked_ids = store.get_untracked_game_ids(list(completed["game_id"]))
            if untracked_ids:
                models = _load_models_cached()
                history_all = _load_game_history(season)
                # No odds fetch here (unlike the two live/upcoming call
                # sites): backfill can touch an unpredictable, potentially
                # large batch of already-finished games at once (e.g. after
                # the tracker was down for a while), and a real spread
                # line matters far less for games that already happened --
                # ATS backfill just comes back unset (None), same as any
                # other genuinely-no-line game.
                odds_df = pd.DataFrame()
                backfill_games = []
                for _, game in completed[completed["game_id"].isin(untracked_ids)].iterrows():
                    try:
                        spread_line, total_line = _lines_for_game(odds_df, game["home_team"], game["away_team"])
                        history_excl = history_all[history_all["game_id"] != game["game_id"]]
                        pred = _predict_game_from_models(
                            models, game["home_team"], game["away_team"], history_excl,
                            spread_line=spread_line, total_line=total_line,
                            gameday=game.get("gameday"), conference_game=game.get("conference_game"),
                        )
                        backfill_games.append({
                            "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                            "commence_time": str(game["gameday"]), "season": season,
                            "week": int(game["week"]) if pd.notna(game.get("week")) else None,
                            "home_spread_line": spread_line, "total_line": total_line,
                            "actual_home_score": game["home_score"], "actual_away_score": game["away_score"],
                            **pred,
                        })
                    except Exception:
                        logger.exception("backfill prediction failed for game_id=%s", game.get("game_id"))
                        continue
                store.record_resolved_game_predictions(backfill_games)
    except Exception:
        logger.exception("backfill of untracked finished games failed")

    try:
        if not completed.empty:
            actual_stats = player_stats.fetch_weekly_player_stats([season], completed)
            store.reconcile_player_prop_predictions(_attach_game_id(actual_stats, completed))
    except Exception:
        logger.exception("player prop reconciliation failed")

    try:
        store.backfill_unresolved_games(games_data)
    except Exception:
        logger.exception("backfill_unresolved_games failed")