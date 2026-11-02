"""Prop grading: counted pre-kickoff projections vs the box score.

Mirrors NBA_Predictor's prop grading (PR #163) on CFB's store. The three
properties under test are the ones the feature's honesty rests on:

- the graded pick is the COUNTED one (earliest recorded per game/player/
  market, the track record's own rule) and only when its own stamp proves
  it was made before kickoff -- a post-kickoff row is excluded, not
  relabelled;
- a grade exists only where the manifest publishes a real holdout MAE for
  the stat, and the bands are multiples of that number;
- an unrecorded/empty store grades empty, never crashes.
"""
import contextlib
import json

import pytest
from fastapi.testclient import TestClient

from cfb_predictor import config
from cfb_predictor.api import routes
from cfb_predictor.api.main import app
from cfb_predictor.tracking import prop_grading, store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(routes, "_prop_grading_cache", None)
    monkeypatch.setattr(routes, "_prop_grading_cached_at", 0.0)
    yield


@pytest.fixture
def models_dir(tmp_path):
    """A manifest publishing a real holdout MAE for passing_yards only."""
    path = tmp_path / "models"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({
        "yardage_metrics": {
            "passing_yards": {"n_train": 100, "holdout_mae": 40.0, "naive_mae": 55.0},
            "rushing_yards": {"n_train": 100},
        }
    }))
    return path


def _game(game_id, home, away, commence):
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, resolved)
            VALUES (?, ?, ?, ?, ?, 0.5, 0.5, 1)
            """,
            (game_id, home, away, commence, "2026-08-01T00:00:00+00:00"),
        )


def _prop(game_id, player_id, name, market, predicted, actual, snapped, position="QB"):
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value,
                 snapshotted_at, resolved, actual_value, position)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (game_id, player_id, name, market, predicted, snapped, actual, position),
        )


KICKOFF = "2026-10-31T17:00:00+00:00"
BEFORE = "2026-10-31T12:00:00+00:00"
AFTER = "2026-10-31T18:30:00+00:00"


def _slate():
    """One completed game day (plus a later one) with a known answer key."""
    _game("g1", "Georgia", "Alabama", KICKOFF)
    _prop("g1", "101", "QB One", "passing_yards", 300.0, 340.0, BEFORE)
    _prop("g1", "101", "QB One", "rushing_yards", 20.0, 5.0, BEFORE)
    _prop("g1", "102", "RB Two", "rushing_yards", 80.0, 150.0, BEFORE, position="RB")
    _prop("g1", "105", "QB Five", "passing_yards", 200.0, 210.0, BEFORE)
    _prop("g1", "106", "QB Six", "passing_yards", 200.0, 270.0, BEFORE)
    _prop("g1", "107", "QB Seven", "passing_yards", 200.0, 295.0, BEFORE)
    # Post-kickoff: recorded, resolved, and excluded from the graded sets.
    _prop("g1", "103", "Late Three", "passing_yards", 200.0, 400.0, AFTER)
    # A probability market: never graded here.
    _prop("g1", "101", "QB One", "anytime_td", 0.4, 1.0, BEFORE)
    # A game the store never tracked: no kickoff to prove timing against.
    _prop("ghost", "104", "Ghost Four", "passing_yards", 200.0, 200.0, BEFORE)
    # A later day, so day ordering is observable.
    _game("g2", "Ohio State", "Michigan", "2026-11-01T17:00:00+00:00")
    _prop("g2", "201", "QB Eight", "passing_yards", 250.0, 240.0, "2026-11-01T12:00:00+00:00")
    # QB One's recorded history: six prior Saturdays, passing actuals
    # 200..300 -- the causal last-5 baseline going into g1 is 260.
    for i, (day, actual) in enumerate(zip(
        ["2026-09-05", "2026-09-12", "2026-09-19", "2026-09-26", "2026-10-03", "2026-10-10"],
        [200.0, 220.0, 240.0, 260.0, 280.0, 300.0],
    )):
        gid = f"p{i}"
        _game(gid, "Georgia", "Florida", f"{day}T17:00:00+00:00")
        _prop(gid, "101", "QB One", "passing_yards", 250.0, actual, f"{day}T12:00:00+00:00")
    # RB Two's single prior game: baseline 60, departure |80 - 60| = 20.
    _game("p9", "Georgia", "Florida", "2026-10-24T17:00:00+00:00")
    _prop("p9", "102", "RB Two", "rushing_yards", 70.0, 60.0, "2026-10-24T12:00:00+00:00", position="RB")


def _day(payload, date):
    return next(d for d in payload["days"] if d["date"] == date)


def _rows(day):
    return [row for game in day["games"] for row in game["rows"]]


def test_only_counted_pre_kickoff_counting_stats_are_graded(models_dir):
    _slate()
    payload = prop_grading.build_prop_grading(models_dir=models_dir)
    rows = _rows(_day(payload, "2026-10-31"))
    players = {r["player"] for r in rows}
    assert players == {"QB One", "RB Two", "QB Five", "QB Six", "QB Seven"}
    assert all(r["made_before_kickoff"] is True for r in rows)
    assert all(r["stat"] != "anytime_td" for r in rows)
    by_key = {(r["player"], r["stat"]): r for r in rows}
    assert by_key[("QB One", "passing_yards")]["delta"] == 40.0
    assert by_key[("QB One", "rushing_yards")]["delta"] == -15.0
    assert by_key[("RB Two", "rushing_yards")]["actual"] == 150.0


def test_bands_are_multiples_of_the_manifests_real_holdout_mae(models_dir):
    _slate()
    payload = prop_grading.build_prop_grading(models_dir=models_dir)
    assert payload["stats"]["passing_yards"] == {"holdout_mae": 40.0, "naive_mae": 55.0}
    assert payload["stats"]["rushing_yards"] == {"holdout_mae": None, "naive_mae": None}
    by_player = {r["player"]: r for r in _rows(_day(payload, "2026-10-31")) if r["stat"] == "passing_yards"}
    assert by_player["QB Five"]["grade"] == "On the number"   # |10| <= 0.5 x 40
    assert by_player["QB One"]["grade"] == "Close"            # |40| <= 1.0 x 40
    assert by_player["QB Six"]["grade"] == "Off"              # |70| <= 2.0 x 40
    assert by_player["QB Seven"]["grade"] == "Way off"        # |95| > 2.0 x 40


def test_stat_without_a_published_mae_is_ungraded_never_invented(models_dir):
    _slate()
    payload = prop_grading.build_prop_grading(models_dir=models_dir)
    rushing = [r for r in _rows(_day(payload, "2026-10-31")) if r["stat"] == "rushing_yards"]
    assert rushing and all(r["grade"] is None for r in rushing)


def test_committed_manifest_publishes_no_holdout_mae_so_no_grades():
    """The repo's real manifest records only n_train per yardage market:
    every stat must come back unjudged rather than banded by a number
    nobody measured."""
    _slate()
    payload = prop_grading.build_prop_grading()
    assert all(figures == {"holdout_mae": None, "naive_mae": None}
               for figures in payload["stats"].values())
    assert all(r["grade"] is None for r in _rows(_day(payload, "2026-10-31")))


def test_most_confident_ranks_by_departure_from_the_causal_rolling_baseline(models_dir):
    _slate()
    payload = prop_grading.build_prop_grading(models_dir=models_dir)
    confident = _day(payload, "2026-10-31")["most_confident"]
    # QB One passing: |300 projected - 260 last-5 baseline| = 40, the
    # day's biggest departure; RB Two rushing: |80 - 60| = 20. The other
    # QBs have no recorded history, so they cannot be ranked at all.
    assert [(r["player"], r["stat"]) for r in confident] == [
        ("QB One", "passing_yards"), ("RB Two", "rushing_yards"),
    ]


def test_biggest_surprises_rank_by_absolute_delta_both_directions(models_dir):
    _slate()
    payload = prop_grading.build_prop_grading(models_dir=models_dir)
    surprises = _day(payload, "2026-10-31")["biggest_surprises"]
    assert [(r["player"], r["delta"]) for r in surprises[:3]] == [
        ("QB Seven", 95.0), ("QB Six", 70.0), ("RB Two", 70.0),
    ]
    assert ("QB One", -15.0) in [(r["player"], r["delta"]) for r in surprises]


def test_days_are_newest_first_with_labelled_games(models_dir):
    _slate()
    payload = prop_grading.build_prop_grading(
        {"101": "Georgia", "102": "Georgia", "201": "Michigan"}, models_dir=models_dir
    )
    # Newest first, bounded at the 4-day window: the slate's own prior
    # games are graded days too and fill the window behind the two latest.
    dates = [d["date"] for d in payload["days"]]
    assert dates == sorted(dates, reverse=True)
    assert dates[:2] == ["2026-11-01", "2026-10-31"] and len(dates) == 4
    day = _day(payload, "2026-10-31")
    game = next(g for g in day["games"] if g["game_id"] == "g1")
    assert game["label"] == "Alabama @ Georgia"
    row = next(r for r in game["rows"] if r["player"] == "QB One" and r["stat"] == "passing_yards")
    assert row["team"] == "Georgia" and row["opponent"] == "Alabama"
    assert row["kickoff"] == KICKOFF
    later = next(r for r in _rows(_day(payload, "2026-11-01")) if r["player"] == "QB Eight")
    assert later["team"] == "Michigan" and later["opponent"] == "Ohio State"


def test_row_shape_is_exactly_the_contract(models_dir):
    _slate()
    payload = prop_grading.build_prop_grading(models_dir=models_dir)
    assert set(payload) == {"generated_at", "stats", "days"}
    row = _rows(_day(payload, "2026-10-31"))[0]
    assert set(row) == {
        "player", "team", "opponent", "stat", "projected", "actual",
        "delta", "grade", "made_before_kickoff", "kickoff",
    }


def test_empty_store_grades_empty_and_never_crashes(models_dir):
    payload = prop_grading.build_prop_grading(models_dir=models_dir)
    assert payload["days"] == []
    assert payload["stats"]["passing_yards"] == {"holdout_mae": 40.0, "naive_mae": 55.0}


def test_grade_for_band_edges():
    assert prop_grading.grade_for(20.0, 40.0) == "On the number"
    assert prop_grading.grade_for(40.0, 40.0) == "Close"
    assert prop_grading.grade_for(80.0, 40.0) == "Off"
    assert prop_grading.grade_for(80.1, 40.0) == "Way off"
    assert prop_grading.grade_for(10.0, None) is None
    assert prop_grading.grade_for(10.0, 0.0) is None


def test_route_serves_the_payload(models_dir, monkeypatch):
    _slate()
    monkeypatch.setattr(routes, "_public_snapshot", lambda: {})
    monkeypatch.setattr(prop_grading, "MODELS_DIR", models_dir)
    body = TestClient(app).get("/api/prop-grading").json()
    assert [d["date"] for d in body["days"]][:2] == ["2026-11-01", "2026-10-31"]
    assert body["stats"]["passing_yards"]["holdout_mae"] == 40.0
