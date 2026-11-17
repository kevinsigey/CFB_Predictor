"""Regression contract for GET /api/track-record payload size and speed.

Fixture: a tracking DB shaped like production's, 566 resolved games and 21,000
resolved player-prop rows. The prop rows are what made the payload ~6.8 MB and
the request ~14.6 s live; `player_props.per_pick` is read by no frontend, so it
is opt-in (`?detail=per_pick`), and the default must stay under BUDGET_BYTES.
"""

import contextlib
import time

import pytest
from fastapi.testclient import TestClient

from cfb_predictor.api.main import app
from cfb_predictor.tracking import store

# Measured on this fixture: default ~354 KB uncompressed (almost all of it
# games.per_pick, which the frontend reads; ~6 KB gzipped), the full record
# ~6.8 MB before the split. 400 KB keeps the prop rows out without flagging
# normal game-row growth.
BUDGET_BYTES = 400_000
N_GAMES = 566
N_PROPS = 21_000
MARKETS = ("anytime_td", "passing_yards", "rushing_yards", "receiving_yards", "receptions", "carries")


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    store._memo.clear()
    yield
    store._memo.clear()


@pytest.fixture
def realistic_db():
    with contextlib.closing(store._connect()) as conn:
        games = [
            (
                f"g{i}", "Home", "Away", "2025-09-06T17:00:00+00:00",
                "2025-09-05T12:00:00+00:00", 0.6, 0.4, 0.55, 0.45, 0.5, 0.5,
                -3.5, 52.5, 1, 27 + i % 7, 20 + i % 5, 1, 1, 0,
            )
            for i in range(N_GAMES)
        ]
        conn.executemany(
            "INSERT INTO game_predictions (game_id, home_team, away_team, commence_time, snapshotted_at,"
            " home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob,"
            " home_spread_line, total_line, resolved, actual_home_score, actual_away_score,"
            " moneyline_hit, ats_hit, total_hit) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            games,
        )
        props = [
            (
                f"g{i % N_GAMES}", f"p{i}", f"Player {i}", MARKETS[i % len(MARKETS)],
                0.3 + (i % 50) / 100, "2025-09-05T13:00:00+00:00", 1, float(i % 3), "RB",
            )
            for i in range(N_PROPS)
        ]
        conn.executemany(
            "INSERT INTO player_prop_predictions (game_id, player_id, player_name, market, predicted_value,"
            " snapshotted_at, resolved, actual_value, position) VALUES (?,?,?,?,?,?,?,?,?)",
            props,
        )
        conn.commit()


@pytest.fixture
def client():
    return TestClient(app)


def test_default_payload_fits_budget_and_keeps_game_rows(realistic_db, client):
    response = client.get("/api/track-record")

    assert response.status_code == 200
    assert len(response.content) < BUDGET_BYTES, len(response.content)
    body = response.json()
    # One row per recorded (game, market): moneyline, ats and totals for each game.
    assert len(body["games"]["per_pick"]) == 3 * N_GAMES
    assert "per_pick" not in body["player_props"]
    assert body["player_props"]["anytime_td"]["n_resolved"] > 0


def test_default_is_the_full_record_minus_prop_rows(realistic_db, client):
    default = client.get("/api/track-record").json()
    full = client.get("/api/track-record?detail=per_pick").json()

    full["player_props"].pop("per_pick")
    assert default == full


def test_detail_per_pick_serves_every_prop_row(realistic_db, client):
    body = client.get("/api/track-record?detail=per_pick").json()

    assert len(body["player_props"]["per_pick"]) == N_PROPS


def test_gzip_is_applied_when_the_client_accepts_it(realistic_db, client):
    response = client.get("/api/track-record?detail=per_pick", headers={"Accept-Encoding": "gzip"})

    assert response.headers.get("content-encoding") == "gzip"


def test_second_call_is_served_from_cache_and_the_record_is_not_mutated(realistic_db, client, monkeypatch):
    builds = []
    original = store._prop_pick_rows

    def counting(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_prop_pick_rows", counting)

    t0 = time.perf_counter()
    client.get("/api/track-record")
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    client.get("/api/track-record")
    second = time.perf_counter() - t0

    assert len(builds) == 1
    assert second < first * 0.5, (first, second)
    # The default route copies; the memoised record still carries the prop rows.
    assert len(store.get_track_record()["player_props"]["per_pick"]) == N_PROPS


def test_a_write_to_the_tracking_db_invalidates_the_cache(realistic_db, client, monkeypatch):
    builds = []
    original = store._prop_pick_rows

    def counting(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_prop_pick_rows", counting)
    client.get("/api/track-record")
    with contextlib.closing(store._connect()) as conn:
        conn.execute("UPDATE player_prop_predictions SET actual_value = 2.0 WHERE player_id = 'p0'")
        conn.commit()

    client.get("/api/track-record")

    assert len(builds) == 2
