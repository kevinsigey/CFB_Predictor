"""SQLite persistence for the live prediction track record.

Snapshots each game's core-market predictions and each tracked player prop
before kickoff, then reconciles them against actual results once games are
played. Snapshot rows are immutable (``INSERT OR IGNORE``); reconciliation
only fills outcome columns on existing, unresolved rows. Verbatim port of
nfl_predictor/tracking/store.py — this schema has zero sport-specific logic.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import TRACKING_DB_PATH
from ..features import player_usage


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRACKING_DB_PATH), timeout=15)
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS game_predictions (
            game_id TEXT PRIMARY KEY,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            home_cover_prob REAL,
            away_cover_prob REAL,
            over_prob REAL,
            under_prob REAL,
            home_spread_line REAL,
            total_line REAL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            moneyline_hit INTEGER,
            ats_hit INTEGER,
            total_hit INTEGER
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(game_predictions)")}
    for column in ("home_spread_line", "total_line", "ats_hit", "total_hit", "season", "week"):
        if column not in existing_cols:
            conn.execute(f"ALTER TABLE game_predictions ADD COLUMN {column} REAL" if column in ("home_spread_line", "total_line")
                         else f"ALTER TABLE game_predictions ADD COLUMN {column} INTEGER")
    # Kalshi feed: the predicted distribution behind each frozen snapshot, so the trade hub can
    # price a strike from the model's own margin/total distribution instead of a point estimate.
    #
    # Deliberately NOT a `backfilled` column. Pre-game-ness is already derived, live, by
    # `_snapshotted_after_kickoff` (snapshotted_at >= commence_time, failing closed): it cannot
    # drift out of step with the timestamps, and a database written before this change needs no
    # migration pass over its rows.
    for column, sql_type in _FEED_COLUMNS.items():
        if column not in existing_cols:
            conn.execute(f"ALTER TABLE game_predictions ADD COLUMN {column} {sql_type}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_prop_predictions (
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            market TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            snapshotted_at TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_value REAL,
            PRIMARY KEY (game_id, player_id, market)
        )
        """
    )
    existing_prop_cols = {row[1] for row in conn.execute("PRAGMA table_info(player_prop_predictions)")}
    if "position" not in existing_prop_cols:
        # Phase 9: position enables the per-position MAE breakdown. Nullable
        # so rows recorded before this migration keep working (they surface
        # under "unknown" in the summary).
        conn.execute("ALTER TABLE player_prop_predictions ADD COLUMN position TEXT")
    return conn


_FEED_COLUMNS = {
    "predicted_margin": "REAL",
    "sigma": "REAL",
    "predicted_total": "REAL",
    "total_sigma": "REAL",
    "model_version": "TEXT",
}


def _parse_utc(value: str) -> datetime:
    """ISO timestamp -> aware UTC datetime. Naive values are UTC (that is how both
    commence_time and snapshotted_at are written)."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_float(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _require_pre_kickoff(commence_time: str) -> None:
    kickoff = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    if kickoff <= datetime.now(timezone.utc):
        raise ValueError("Game predictions must be snapshotted before kickoff")


def record_game_predictions(games: list[dict]) -> int:
    """Snapshots only the games that are still pre-kickoff. A single
    already-kicked-off (or malformed) game in the batch used to raise for
    the whole call, silently dropping every other valid game in the same
    tick along with it -- skip just the invalid ones instead (see this
    plan's final-review fix, Task 23, finding I4)."""
    if not games:
        return 0
    valid_games = []
    for game in games:
        try:
            _require_pre_kickoff(game["commence_time"])
            valid_games.append(game)
        except ValueError:
            continue
    if not valid_games:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            game["game_id"], game["home_team"], game["away_team"], game["commence_time"], now,
            float(game["home_win_prob"]), float(game["away_win_prob"]),
            game.get("home_cover_prob"), game.get("away_cover_prob"),
            game.get("over_prob"), game.get("under_prob"),
            game.get("home_spread_line"), game.get("total_line"), game.get("season"), game.get("week"),
            game.get("predicted_margin"), game.get("sigma"), game.get("predicted_total"),
            game.get("total_sigma"), game.get("model_version"),
        )
        for game in valid_games
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob,
                 home_spread_line, total_line, season, week,
                 predicted_margin, sigma, predicted_total, total_sigma, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


def _present(*values) -> bool:
    """Whether every value is a real number rather than missing.

    `None or 0` is the wrong instrument for "this probability was never computed".
    It converts an absent input into a confident zero, and a zero is a *legible*
    probability: on a two-sided market, `0.0 >= 0.0` resolves to the home side by
    the accident of `>=`. So a game snapshotted with a spread line but no cover
    probabilities -- which is what the odds feed yields whenever it has a spread
    without a matching market -- was recorded as a graded ATS call the model never
    made, in both directions:

        home 31-24, spread -3.5, cover probs None  -> ats_hit = 1  (fabricated hit)
        home 20-24, spread -3.5, cover probs None  -> ats_hit = 0  (fabricated miss)

    That is worse here than elsewhere in this codebase, because the output is the
    track record. Half a market is equally unusable: `0.6` against `None` is not
    obviously the home side, but it is not a call either, and the same expression
    reports it as one.

    So presence is required on *every* value a comparison depends on, and a
    missing one yields no grade rather than a wrong one.
    """
    return all(value is not None and not pd.isna(value) for value in values)


def _compute_hits(
    home_score: float, away_score: float, home_win_prob: float, away_win_prob: float,
    home_spread_line: float | None, home_cover_prob: float | None, away_cover_prob: float | None,
    total_line: float | None, over_prob: float | None, under_prob: float | None,
) -> tuple[int, int | None, int | None]:
    home_win = home_score > away_score
    predicted_home_win = home_win_prob >= away_win_prob
    moneyline_hit = int(predicted_home_win == home_win)

    # Grade a market only when the line *and* both sides of the probability pair
    # are present. See `_present` for why a missing probability must not be
    # coerced to zero rather than merely being unusual.
    ats_hit = None
    if _present(home_spread_line) and _present(home_cover_prob, away_cover_prob):
        home_margin = home_score - away_score
        home_covered = home_margin > home_spread_line
        predicted_home_cover = home_cover_prob >= away_cover_prob
        ats_hit = int(predicted_home_cover == home_covered)

    total_hit = None
    if _present(total_line) and _present(over_prob, under_prob):
        actual_total = home_score + away_score
        went_over = actual_total > total_line
        predicted_over = over_prob >= under_prob
        total_hit = int(predicted_over == went_over)

    return moneyline_hit, ats_hit, total_hit


def reconcile_game_predictions(results_df: pd.DataFrame) -> int:
    if results_df.empty:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        unresolved = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(results_df, on="game_id", how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            moneyline_hit, ats_hit, total_hit = _compute_hits(
                row["home_score"], row["away_score"], row["home_win_prob"], row["away_win_prob"],
                row.get("home_spread_line"), row.get("home_cover_prob"), row.get("away_cover_prob"),
                row.get("total_line"), row.get("over_prob"), row.get("under_prob"),
            )
            cursor = conn.execute(
                """
                UPDATE game_predictions
                SET resolved = 1, actual_home_score = ?, actual_away_score = ?,
                    moneyline_hit = ?, ats_hit = ?, total_hit = ?
                WHERE game_id = ? AND resolved = 0
                """,
                (int(row["home_score"]), int(row["away_score"]), moneyline_hit, ats_hit, total_hit, row["game_id"]),
            )
            resolved_count += cursor.rowcount
        return resolved_count


def get_untracked_game_ids(game_ids: list[str]) -> set[str]:
    """Which of these game_ids have no row in game_predictions at all yet
    -- never snapshotted before kickoff (distinct from "pending", which
    means snapshotted but not yet resolved). Used to backfill a game the
    tracker missed entirely, e.g. because the tracking loop wasn't
    running yet when it kicked off."""
    if not game_ids:
        return set()
    with contextlib.closing(_connect()) as conn:
        placeholders = ",".join("?" * len(game_ids))
        tracked = pd.read_sql(
            f"SELECT game_id FROM game_predictions WHERE game_id IN ({placeholders})", conn, params=game_ids
        )
    return set(game_ids) - set(tracked["game_id"])


def record_resolved_game_predictions(games: list[dict]) -> int:
    """Backfill a prediction snapshot for an already-finished game that was
    never tracked before kickoff -- records the prediction AND its known
    outcome in one shot, skipping the pre-kickoff requirement since
    there's no live pregame moment left to protect. Each game dict's
    prediction fields must already come from strictly pre-game
    information (the caller's job); this function only persists it."""
    if not games:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for game in games:
        moneyline_hit, ats_hit, total_hit = _compute_hits(
            game["actual_home_score"], game["actual_away_score"], game["home_win_prob"], game["away_win_prob"],
            game.get("home_spread_line"), game.get("home_cover_prob"), game.get("away_cover_prob"),
            game.get("total_line"), game.get("over_prob"), game.get("under_prob"),
        )
        rows.append((
            game["game_id"], game["home_team"], game["away_team"], game["commence_time"], now,
            float(game["home_win_prob"]), float(game["away_win_prob"]),
            game.get("home_cover_prob"), game.get("away_cover_prob"),
            game.get("over_prob"), game.get("under_prob"),
            game.get("home_spread_line"), game.get("total_line"), game.get("season"), game.get("week"),
            1, int(game["actual_home_score"]), int(game["actual_away_score"]), moneyline_hit, ats_hit, total_hit,
            game.get("predicted_margin"), game.get("sigma"), game.get("predicted_total"),
            game.get("total_sigma"), game.get("model_version"),
        ))
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob,
                 home_spread_line, total_line, season, week,
                 resolved, actual_home_score, actual_away_score, moneyline_hit, ats_hit, total_hit,
                 predicted_margin, sigma, predicted_total, total_sigma, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


def backfill_unresolved_games(games_data_module) -> int:
    """Targeted backfill: reconciles any still-unresolved snapshot whose
    season isn't CURRENT_SEASON, which the normal tick's
    fetch_current_season_partial() never looks at again. Takes the
    games_data module (not a season list) so it can look up whichever
    season each unresolved row actually belongs to."""
    with contextlib.closing(_connect()) as conn:
        unresolved = pd.read_sql("SELECT season FROM game_predictions WHERE resolved = 0", conn)
    if unresolved.empty:
        return 0

    seasons = sorted({int(s) for s in unresolved["season"].dropna()} | {pd.Timestamp.now().year})
    finished = games_data_module.load_training_data(seasons=seasons)
    if finished.empty:
        return 0
    return reconcile_game_predictions(finished[["game_id", "home_score", "away_score"]])


def _db_version() -> tuple:
    """Changes on every write to the tracking DB: the main file and its WAL."""
    db = Path(TRACKING_DB_PATH)
    stamps = []
    for path in (db, db.with_name(db.name + "-wal")):
        try:
            st = path.stat()
            stamps.append((st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            stamps.append(None)
    return tuple(stamps)


_memo: dict[str, tuple[tuple, object]] = {}


def _memoised_on_db_version(name: str, build):
    """Return `build()`, reusing the last result while the DB version is unchanged.

    The version is read BEFORE building, so a write landing mid-build leaves a
    stale key and the next call simply rebuilds. Callers must not mutate the
    result; the record is shared between calls.
    """
    version = _db_version()
    hit = _memo.get(name)
    if hit is not None and hit[0] == version:
        return hit[1]
    value = build()
    _memo[name] = (version, value)
    return value


def get_track_record() -> dict:
    """Memoised on the DB version; see `_build_track_record` for what it returns."""
    return _memoised_on_db_version("track_record", _build_track_record)


def _build_track_record() -> dict:
    """Aggregate accuracy summary across every reconciled game and player prop -- not a
    per-game list (the frontend already has that in the game detail modal's own verdict
    section; this is the "how good is the model overall" view, same shape as PL_Predictor's
    Data Hub track record).

    **Two figures, and the headline is the bigger one.** Kevin, 2026-10-01
    (`predictor-hub` docs/superpowers/specs/2026-10-01-track-record-counts-every-pick.md,
    merged as predictor-hub #66), verbatim: "i dont really care about picks made after kickoff
    because im always re running the models ... with every model change it will stop tracking
    ... make it that whats recorded remains recorded and then just use every prediction we make
    for the track record stuff." CFB #26 had it the other way round, excluding post-kickoff
    rows from `n_resolved` and reporting them as `n_rebuilt`; that exclusion is gone and the
    key with it, because "resolved rows refused as look-forward" is now false of every row.

    So each half returns:
      * the headline over COUNTED picks -- one per (game, market), the earliest recorded,
        whenever it was made. `n_resolved` keeps its name and its meaning as "resolved games
        in the record"; what changed is which resolved games it counts, so a site reading
        `n_resolved` reads the fuller record and nothing has to be renamed.
      * `pre_kickoff`, the same keys over the subset made before its own kickoff, with its
        own `n_resolved`, plus `n_pre_kickoff` at the top for a one-number read.
      * `per_pick`, one row per recorded pick carrying `made_before_kickoff` and its own
        `snapshotted_at`, so disclosure is per pick and not only in aggregate.
    """
    with contextlib.closing(_connect()) as conn, conn:
        resolved_games = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 1", conn)
        resolved_props = pd.read_sql("SELECT * FROM player_prop_predictions WHERE resolved = 1", conn)

    labelled_games = _label_pick_timing(resolved_games)
    counted_games = _earliest_recorded(labelled_games, ("game_id",))
    pre_kickoff_games = counted_games[counted_games["made_before_kickoff"]]
    return {
        "games": {
            **_summarize_games(counted_games),
            "pre_kickoff": _summarize_games(pre_kickoff_games),
            "n_pre_kickoff": int(len(pre_kickoff_games)),
            "per_pick": _game_pick_rows(labelled_games, set(counted_games.index)),
        },
        "player_props": _summarize_player_props(resolved_props),
    }


def _label_pick_timing(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds `made_before_kickoff`, derived live from this row's own stamp.

    Never a stored flag: the column does not exist in either table and must not be added,
    because a flag could drift out of step with the timestamps it claims to describe and
    nothing would notice. `test_grading_never_restamps_the_snapshot_timestamp_the_guard_reads`
    is what makes the derivation trustworthy -- grading cannot rewrite `snapshotted_at`, so
    what is read here is the row's original provenance however it arrived.
    """
    if frame.empty:
        labelled = frame.copy()
        labelled["made_before_kickoff"] = pd.Series(dtype=bool)
        return labelled
    labelled = frame.copy()
    labelled["made_before_kickoff"] = [
        _made_before_kickoff(stamp, commence)
        for stamp, commence in zip(frame["snapshotted_at"], frame["commence_time"])
    ]
    return labelled


def _utc_instants(values: pd.Series) -> pd.Series:
    """A sortable UTC-instant series for a column of ISO strings.

    `na_position="last"` is the whole point: a row whose stamp cannot be parsed must never
    win a "which was recorded first" comparison, so it sorts behind everything provable. If
    a key has ONLY unparseable stamps the row is still kept (the pick exists) -- it simply
    cannot claim to be the earliest.
    """
    return pd.to_datetime(values, format="ISO8601", utc=True, errors="coerce")


def _earliest_recorded(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    """One counted row per key: the EARLIEST recorded one, by UTC instant.

    Rule 2 of the 2026-10-01 spec. A later rerun of the model on the same game and market is
    kept as history -- it is still in `frame`, and it still appears in `per_pick` with
    `counted: false` -- but it neither replaces the counted pick nor counts a second time.
    Without that, re-running the model until it is right would be free.

    Reversible in one place: count the LATEST row per key instead (drop the
    `sort_values`/`drop_duplicates` pair and take `.tail()` per key) if that is ever the
    better answer. Both tables already make duplicates unreachable (`INSERT OR IGNORE` on
    their primary keys), so this is the belt to that braces: it is what makes the rule a
    property of the read side rather than of today's write path.
    """
    if frame.empty:
        return frame
    ordered = frame.assign(_instant=_utc_instants(frame["snapshotted_at"]))
    ordered = ordered.sort_values("_instant", kind="stable", na_position="last")
    counted = ordered.drop_duplicates(subset=list(keys), keep="first").drop(columns="_instant")
    return counted.sort_index()


def _game_pick_rows(labelled: pd.DataFrame, counted_index: set) -> list[dict]:
    """One row per recorded (game, market) pick, hit and miss alike, never filtered.

    Same shape as NFL_Predictor's `_per_pick_rows`, so the two sports publish the same
    disclosure. `made_before_kickoff` and `snapshotted_at` are on the row rather than only
    in the aggregate, and an ungraded market is omitted rather than listed as a fabricated
    miss -- an ungraded market is not a pick the model made.
    """
    rows: list[dict] = []
    for index, game in labelled.iterrows():
        before = bool(game["made_before_kickoff"])
        for market, column in (("moneyline", "moneyline_hit"), ("ats", "ats_hit"), ("totals", "total_hit")):
            hit = game[column]
            if hit is None or pd.isna(hit):
                continue
            rows.append({
                "game_id": game["game_id"],
                "gameday": game["commence_time"],
                "market": market,
                "pick": _pick_words(game, market),
                "actual": _actual_words(game, market),
                "hit": bool(hit),
                "made_before_kickoff": before,
                "snapshotted_at": game["snapshotted_at"],
                "counted": index in counted_index,
            })
    return rows


def _pick_words(game: pd.Series, market: str) -> str:
    """What the model backed, in words a reader can check against the result."""
    if market == "moneyline":
        return game["home_team"] if game["home_win_prob"] >= game["away_win_prob"] else game["away_team"]
    if market == "ats":
        line = float(game["home_spread_line"])
        if game["home_cover_prob"] >= game["away_cover_prob"]:
            return f"{game['home_team']} {line:+.1f}"
        return f"{game['away_team']} {-line:+.1f}"
    return "Over" if game["over_prob"] >= game["under_prob"] else "Under"


def _actual_words(game: pd.Series, market: str) -> str:
    """What actually happened, in the same words."""
    if market == "moneyline":
        return game["home_team"] if game["actual_home_score"] > game["actual_away_score"] else game["away_team"]
    margin = game["actual_home_score"] - game["actual_away_score"]
    line = float(game["home_spread_line"])
    if market == "ats":
        if margin > line:
            return f"{game['home_team']} {line:+.1f}"
        return f"{game['away_team']} {-line:+.1f}"
    total = game["actual_home_score"] + game["actual_away_score"]
    return "Over" if total > float(game["total_line"]) else "Under"


def get_feed_predictions(now: datetime | None = None) -> list[dict]:
    """Frozen pre-game snapshots for games that have not started yet: the rows the trade hub may
    compare against Kalshi prices.

    Two exclusions, and they are the same question asked twice. `resolved = 0` drops games that
    have already been graded; `_snapshotted_after_kickoff` drops rows written at or after kickoff
    (PR #1's `rebuilt` rule), which is the leakage guard from spec 5a. A row that fails either is
    never served, and an unparseable timestamp is treated as rebuilt, so the filter fails closed.
    """
    now = now or datetime.now(timezone.utc)
    with contextlib.closing(_connect()) as conn:
        rows = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 0", conn)
    feed = []
    for _, row in rows.iterrows():
        if _snapshotted_after_kickoff(row["snapshotted_at"], row["commence_time"]):
            continue
        try:
            start = _parse_utc(row["commence_time"])
            snapshotted = _parse_utc(row["snapshotted_at"])
        except (TypeError, ValueError):
            continue
        if start <= now:
            continue
        feed.append({
            "game_id": row["game_id"],
            "season": None if pd.isna(row["season"]) else int(row["season"]),
            "week": None if pd.isna(row["week"]) else int(row["week"]),
            "home": row["home_team"],
            "away": row["away_team"],
            "start_utc": start.isoformat(),
            "p_home": float(row["home_win_prob"]),
            "margin_mu": _optional_float(row["predicted_margin"]),
            "sigma": _optional_float(row["sigma"]),
            "total_mu": _optional_float(row["predicted_total"]),
            "total_sigma": _optional_float(row["total_sigma"]),
            "home_spread_line": _optional_float(row["home_spread_line"]),
            "total_line": _optional_float(row["total_line"]),
            "model_version": row["model_version"] if isinstance(row["model_version"], str) else None,
            "snapshotted_at": snapshotted.isoformat(),
            # Always False here -- that is what the rebuilt filter above guarantees. The key stays
            # because the hub's parser rejects a truthy value, and False is the honest statement.
            "backfilled": False,
        })
    return sorted(feed, key=lambda r: (r["start_utc"], r["game_id"]))


def _calibration_buckets(pairs: list[tuple[float, int]], n_buckets: int) -> list[dict]:
    buckets = []
    for i in range(n_buckets):
        lo, hi = i / n_buckets, (i + 1) / n_buckets
        last = i == n_buckets - 1
        inside = [(p, y) for p, y in pairs if lo <= p < hi or (last and p == 1.0)]
        buckets.append({
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "n": len(inside),
            "mean_prob": sum(p for p, _ in inside) / len(inside) if inside else None,
            "hit_rate": sum(y for _, y in inside) / len(inside) if inside else None,
        })
    return buckets


# 4 buckets, not 10, ruled 2026-09-27. Same change, same reasoning, and the same commit shape as
# NFL_Predictor -- the two feeds must publish the same bucket layout or the hub's gate arithmetic
# differs by sport, which would be worse than either value on its own.
#
# The trade hub gates an edge on `n >= calibration_min_n` in the bucket the edge's own probability
# falls in, and it needs EVERY bucket to clear that bar before admitting anything. So the settled
# contracts needed are `n_buckets x calibration_min_n`: at 10 x 20 that is 200, while the hub's
# reviewer renders a verdict at 100. The product was twice as strict about admitting an edge as it
# was about judging one, and on the real distribution (42 settled for CFB winner) the winner gate
# admitted nothing at all at 100 settled.
#
# At 4 x 20 that is 80, under the reviewer's bar, and the same measurement shows 4 buckets admitting
# 95% / 82% / 97% of edge mass at 100 settled where 10 admitted 0% / 46% / 66%.
CALIBRATION_N_BUCKETS = 4


def get_calibration(n_buckets: int = CALIBRATION_N_BUCKETS) -> dict:
    """Reliability buckets over resolved, genuinely pre-game snapshots: home win probability vs
    home won, home cover probability vs covered (at the recorded line), over probability vs went
    over. Ties and pushes are left out.

    Rebuilt rows are excluded by the same predicate the feed uses, so the buckets the hub
    calibrates against and the rows it prices come from the same population.
    """
    with contextlib.closing(_connect()) as conn:
        rows = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 1", conn)
    winner: list[tuple[float, int]] = []
    spread: list[tuple[float, int]] = []
    total: list[tuple[float, int]] = []
    for _, row in rows.iterrows():
        if _snapshotted_after_kickoff(row["snapshotted_at"], row["commence_time"]):
            continue
        home, away = row["actual_home_score"], row["actual_away_score"]
        if pd.isna(home) or pd.isna(away):
            continue
        margin, points = float(home - away), float(home + away)
        if margin != 0:
            winner.append((float(row["home_win_prob"]), int(margin > 0)))
        line, prob = row["home_spread_line"], row["home_cover_prob"]
        if pd.notna(line) and pd.notna(prob) and margin != float(line):
            spread.append((float(prob), int(margin > float(line))))
        line, prob = row["total_line"], row["over_prob"]
        if pd.notna(line) and pd.notna(prob) and points != float(line):
            total.append((float(prob), int(points > float(line))))
    return {
        "n_buckets": n_buckets,
        "winner": _calibration_buckets(winner, n_buckets),
        "spread": _calibration_buckets(spread, n_buckets),
        "total": _calibration_buckets(total, n_buckets),
    }


def _summarize_games(resolved: pd.DataFrame) -> dict:
    if resolved.empty:
        return {
            "n_resolved": 0, "pct_moneyline_correct": None, "pct_ats_correct": None,
            "pct_totals_correct": None, "weekly_trend": [],
        }
    # A stored hit flag is not on its own evidence of a graded market. Rows written
    # by the pre-`_present` build carry an `ats_hit` for games where the cover
    # probabilities were never computed, and filtering on `notna()` alone counts
    # those fabrications in `pct_ats_correct` -- the headline number a reader judges
    # the model by. `get_game_verdict` already refuses to report a `predicted` side
    # for exactly these rows, so without this the per-game view and the aggregate
    # give opposite answers about the same game.
    #
    # The data migration that would repair the stored flags is separate; until it
    # runs, excluding the rows here is what keeps the aggregate honest.
    def _pair_present(*columns: str) -> pd.Series:
        """Row-wise `all(_present(...))`, for filtering a frame.

        `_present` is scalar; iterating it with Series arguments would make
        `pd.isna` return a Series and blow up on truthiness. This is the same rule
        expressed over columns.
        """
        mask = pd.Series(True, index=resolved.index)
        for column in columns:
            mask &= resolved[column].notna()
        return mask

    ats = resolved[resolved["ats_hit"].notna() & _pair_present("home_cover_prob", "away_cover_prob")]
    totals = resolved[resolved["total_hit"].notna() & _pair_present("over_prob", "under_prob")]

    weekly_trend = []
    with_week = resolved[resolved["week"].notna()]
    if not with_week.empty:
        grouped = with_week.groupby("week")["moneyline_hit"].agg(["mean", "size"]).reset_index()
        weekly_trend = [
            {"week": int(r["week"]), "pct_moneyline_correct": float(r["mean"]), "n_games": int(r["size"])}
            for _, r in grouped.sort_values("week").iterrows()
        ]

    return {
        "n_resolved": int(len(resolved)),
        "pct_moneyline_correct": float(resolved["moneyline_hit"].mean()),
        "pct_ats_correct": float(ats["ats_hit"].mean()) if not ats.empty else None,
        "pct_totals_correct": float(totals["total_hit"].mean()) if not totals.empty else None,
        "weekly_trend": weekly_trend,
    }


_TD_CONFIDENCE_BUCKETS = (("50-60%", 0.50, 0.60), ("60-70%", 0.60, 0.70), ("70%+", 0.70, 1.01))


def _td_confidence_buckets(anytime_td: pd.DataFrame) -> list[dict]:
    """Hit rate per predicted-probability bucket for anytime-TD props."""
    buckets = []
    for label, lo, hi in _TD_CONFIDENCE_BUCKETS:
        in_bucket = anytime_td[(anytime_td["predicted_value"] >= lo) & (anytime_td["predicted_value"] < hi)]
        buckets.append({
            "label": label,
            "n": int(len(in_bucket)),
            "hit_rate": float(in_bucket["actual_value"].mean()) if not in_bucket.empty else None,
        })
    return buckets


def _summarize_player_props(resolved: pd.DataFrame) -> dict:
    """Per-market metrics over graded props, in two figures.

    The headline is every COUNTED pick -- one per (game, player, market), the earliest
    recorded, whenever it was made -- and `pre_kickoff` beside it is the subset made before
    its own kickoff, with its own n. `n_pre_kickoff` gives that n at the top and `per_pick`
    puts `made_before_kickoff` and `snapshotted_at` on every row, so honesty is disclosure
    rather than exclusion. CFB #26 excluded post-kickoff rows and reported them as
    `n_rebuilt`; Kevin's 2026-10-01 decision reversed that ("just use every prediction we
    make for the track record stuff"), so `n_rebuilt` is removed rather than renamed into a
    lie -- nothing is refused any more.

    The counted key is (game, PLAYER, market), not (game, market): a yardage pick is one pick
    per player per game, and collapsing a game's five rushers into a single pick would delete
    the record rather than deduplicate it.

    `player_prop_predictions` carries no `commence_time`, so this joins to `game_predictions`
    on `game_id` (that table's primary key, so the join is many-to-one and safe). A row whose
    game is absent, or whose stamp cannot be parsed, still COUNTS -- it is a recorded pick --
    but its `made_before_kickoff` is False, because the timestamps do not prove otherwise and
    a `true` is never backfilled on their behalf.
    """
    if not resolved.empty:
        with contextlib.closing(_connect()) as conn:
            kickoff_times = pd.read_sql("SELECT game_id, commence_time FROM game_predictions", conn)
        labelled = _label_pick_timing(resolved.merge(kickoff_times, on="game_id", how="left"))
    else:
        labelled = _label_pick_timing(resolved)

    counted = _earliest_recorded(labelled, ("game_id", "player_id", "market"))
    pre_kickoff = counted[counted["made_before_kickoff"]]
    return {
        **_prop_market_metrics(counted),
        "pre_kickoff": _prop_market_metrics(pre_kickoff),
        "n_pre_kickoff": int(len(pre_kickoff)),
        "per_pick": _prop_pick_rows(labelled, set(counted.index)),
    }


def _prop_market_metrics(counted: pd.DataFrame) -> dict:
    """The per-market metrics, over whatever frame it is handed.

    Split out of `_summarize_player_props` so the headline and the pre-kickoff subset are
    computed by the same code over two frames. They cannot then disagree about a market
    family, a `mae_by_position` grouping or a confidence bucket -- only about which rows.
    """
    result: dict[str, dict] = {}

    anytime_td = counted[counted["market"] == "anytime_td"]
    if anytime_td.empty:
        result["anytime_td"] = {
            "n_resolved": 0, "hit_rate_when_called": None, "brier_score": None,
            "confidence_buckets": _td_confidence_buckets(anytime_td),
        }
    else:
        called = anytime_td[anytime_td["predicted_value"] >= 0.5]
        result["anytime_td"] = {
            "n_resolved": int(len(anytime_td)),
            "n_called": int(len(called)),
            "hit_rate_when_called": float(called["actual_value"].mean()) if not called.empty else None,
            "brier_score": float(((anytime_td["predicted_value"] - anytime_td["actual_value"]) ** 2).mean()),
            "confidence_buckets": _td_confidence_buckets(anytime_td),
        }

    for market in _YARDAGE_MARKETS:
        rows = counted[counted["market"] == market]
        if rows.empty:
            result[market] = {
                "n_resolved": 0, "mean_absolute_error": None,
                "mean_signed_error": None, "mae_by_position": {},
            }
        else:
            signed = rows["predicted_value"] - rows["actual_value"]
            by_position = rows.assign(position=rows["position"].fillna("unknown")).groupby("position")
            result[market] = {
                "n_resolved": int(len(rows)),
                "mean_absolute_error": float(signed.abs().mean()),
                # Positive = systematic over-prediction, negative = under-prediction.
                "mean_signed_error": float(signed.mean()),
                "mae_by_position": {
                    position: float((group["predicted_value"] - group["actual_value"]).abs().mean())
                    for position, group in by_position
                },
            }
    return result


def _prop_pick_rows(labelled: pd.DataFrame, counted_index: set) -> list[dict]:
    """Every recorded prop row, counted or not, with its timing disclosed.

    A row the rule did not count is still published: rule 2 keeps a rerun as history, and
    hiding it would defeat the point of an append-only record. `counted` says which one the
    headline scored, `made_before_kickoff` says when it was made, `snapshotted_at` is its own
    stamp so a reader can check that for themselves.
    """
    rows = []
    for index, prop in labelled.iterrows():
        rows.append({
            "game_id": prop["game_id"],
            "player_id": prop["player_id"],
            "player_name": prop["player_name"],
            "position": None if pd.isna(prop["position"]) else prop["position"],
            "market": prop["market"],
            "predicted_value": float(prop["predicted_value"]),
            "actual_value": None if pd.isna(prop["actual_value"]) else float(prop["actual_value"]),
            "made_before_kickoff": bool(prop["made_before_kickoff"]),
            "snapshotted_at": prop["snapshotted_at"],
            "gameday": None if pd.isna(prop["commence_time"]) else prop["commence_time"],
            "counted": index in counted_index,
        })
    return sorted(rows, key=lambda r: (r["snapshotted_at"], r["game_id"], r["player_id"], r["market"]))


def get_game_verdict(game_id: str) -> dict | None:
    with contextlib.closing(_connect()) as conn:
        rows = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = ?", conn, params=(game_id,))
    if rows.empty or not rows.iloc[0]["resolved"]:
        return None
    row = rows.iloc[0]

    predicted_home_win = row["home_win_prob"] >= row["away_win_prob"]
    actual_home_win = row["actual_home_score"] > row["actual_away_score"]
    verdict = {
        "game_id": game_id,
        "resolved": True,
        "moneyline": {
            "hit": bool(row["moneyline_hit"]),
            "predicted": "home_win" if predicted_home_win else "away_win",
            "actual": "home_win" if actual_home_win else "away_win",
        },
        "ats": None,
        "totals": None,
        "actual_home_score": int(row["actual_home_score"]),
        "actual_away_score": int(row["actual_away_score"]),
        "home_spread_line": float(row["home_spread_line"]) if pd.notna(row["home_spread_line"]) else None,
        "total_line": float(row["total_line"]) if pd.notna(row["total_line"]) else None,
    }
    # The `hit` flags were written by `_compute_hits`, which already refuses to
    # grade a market whose probabilities are missing. Guard the re-derivation the
    # same way, so a row written by an older build cannot report a `predicted`
    # side that was fabricated here. The two must agree: `hit` is a stored fact,
    # `predicted` is recomputed, and a disagreement would be a silent contradiction
    # inside a single verdict object.
    if pd.notna(row["ats_hit"]) and _present(row["home_cover_prob"], row["away_cover_prob"]):
        predicted_home_cover = row["home_cover_prob"] >= row["away_cover_prob"]
        verdict["ats"] = {"hit": bool(row["ats_hit"]), "predicted": "home_cover" if predicted_home_cover else "away_cover"}
    if pd.notna(row["total_hit"]) and _present(row["over_prob"], row["under_prob"]):
        predicted_over = row["over_prob"] >= row["under_prob"]
        verdict["totals"] = {"hit": bool(row["total_hit"]), "predicted": "over" if predicted_over else "under"}
    return verdict


def record_player_prop_predictions(props: list[dict]) -> int:
    if not props:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (prop["game_id"], prop["player_id"], prop["player_name"], prop["market"],
         float(prop["predicted_value"]), prop.get("position"), now)
        for prop in props
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value, position, snapshotted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


_MARKET_TO_STAT_COLUMN = {
    "anytime_td": None,
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "carries": "carries",
}

# Every market except anytime_td resolves against a stat column -- these are
# the "yardage-style" markets in the track-record summary. Derived from
# _MARKET_TO_STAT_COLUMN so adding a market only ever needs one edit. anytime_td
# is `None` here BECAUSE it is not a stat column: it is a boolean over two of
# them, so it resolves through player_usage.anytime_td_actual below rather than
# by picking one column out of the row.
_YARDAGE_MARKETS = tuple(market for market, col in _MARKET_TO_STAT_COLUMN.items() if col is not None)


def reconcile_player_prop_predictions(player_stats_df: pd.DataFrame) -> int:
    if player_stats_df.empty:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        unresolved = pd.read_sql("SELECT * FROM player_prop_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(player_stats_df, on=["game_id", "player_id"], how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            if row["market"] == "anytime_td":
                # Not `(rushing + receiving + passing) > 0` written out again:
                # this used to be an inline copy of the sum and it is the
                # reason NFL's PR #26 had to chase a second implementation in
                # two repos. `anytime_td_actual` is the one definition, and
                # this grader scores the market exactly as the classifier was
                # fitted on it.
                actual = player_usage.anytime_td_actual(
                    row.get("rushing_tds"), row.get("receiving_tds")
                )
            else:
                stat_col = _MARKET_TO_STAT_COLUMN[row["market"]]
                if stat_col not in row or pd.isna(row[stat_col]):
                    continue
                actual = float(row[stat_col])
            cursor = conn.execute(
                """
                UPDATE player_prop_predictions
                SET resolved = 1, actual_value = ?
                WHERE game_id = ? AND player_id = ? AND market = ? AND resolved = 0
                """,
                (actual, row["game_id"], row["player_id"], row["market"]),
            )
            resolved_count += cursor.rowcount
        return resolved_count


def _made_before_kickoff(snapshotted_at, commence_time) -> bool:
    """Whether this row's OWN stamp proves the pick was made before the game started.

    Both values are parsed to aware UTC and compared as INSTANTS (`_parse_utc`), never as
    strings and never as naive wall clocks. That distinction is the whole risk here: a
    `commence_time` handed over by an upstream feed can carry any offset, so
    `2026-09-12T23:30:00+09:00` (14:30Z) is half an hour BEFORE a `15:00:00+00:00`
    kickoff even though the wall clocks read "23:30" against "15:00" and sort the other way.
    A comparison that dropped the offset -- or read the host's local zone -- would label
    that pick post-kickoff. `test_made_before_kickoff_is_derived_from_utc_instants_not_
    wall_clock_strings` is that pair, and the mirror case where a wall clock says "before"
    and the instant says after.

    Naive values are UTC, which is how both columns are written (`now(timezone.utc)` in the
    writers, `UTC` in the feed's kickoffs), and `astimezone` normalises whatever offset is
    present, so the answer does not depend on the machine this runs on.

    Fails CLOSED, and that is the rule's one remaining prohibition: an unparseable stamp, or
    a row whose game never reached `game_predictions` so there is no kickoff time to compare
    against, gives False. Such a row still counts as a recorded pick; it is simply never
    presented as a pre-game one, because nothing here can prove that it was.
    """
    try:
        return _parse_utc(snapshotted_at) < _parse_utc(commence_time)
    except (TypeError, ValueError):
        return False


def _snapshotted_after_kickoff(snapshotted_at: str, commence_time: str) -> bool:
    """Whether the row was stamped at or after its kickoff: the exact complement.

    Still used where the question is "may this row be served/treated as a pre-game call"
    (`get_feed_predictions`, `get_predictions_for_week`), and kept as one call into the
    derivation above rather than a second copy of it, so the two can never disagree about
    where the boundary is.
    """
    return not _made_before_kickoff(snapshotted_at, commence_time)


def get_predictions_for_week(season: int, week: int, games_df: pd.DataFrame) -> list[dict]:
    if games_df.empty:
        return []
    game_ids = tuple(games_df["game_id"])
    placeholders = ",".join("?" * len(game_ids))
    with contextlib.closing(_connect()) as conn:
        tracked = pd.read_sql(
            f"SELECT * FROM game_predictions WHERE game_id IN ({placeholders})", conn, params=game_ids
        )
    tracked_by_id = {row["game_id"]: row for _, row in tracked.iterrows()}

    results = []
    for _, game in games_df.iterrows():
        row = tracked_by_id.get(game["game_id"])
        if row is None:
            results.append({"game_id": game["game_id"], "status": "untracked", "verdict": None})
            continue
        resolved = bool(row["resolved"])
        results.append({
            "game_id": game["game_id"],
            "status": "resolved" if resolved else "pending",
            # Snapshotted at or after kickoff means rebuilt after the fact
            # (record_resolved_game_predictions): shown, never counted.
            "rebuilt": _snapshotted_after_kickoff(row["snapshotted_at"], row["commence_time"]),
            "home_win_prob": row["home_win_prob"],
            "away_win_prob": row["away_win_prob"],
            "verdict": get_game_verdict(game["game_id"]) if resolved else None,
        })
    return results
