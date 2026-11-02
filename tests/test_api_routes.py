import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cfb_predictor.api.main import app
from cfb_predictor.api import routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "401520145", "season": season, "week": week,
              "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
              "home_score": None, "away_score": None}]
        ),
    )
    monkeypatch.setattr(
        routes.games_data, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "401520145", "season": season, "week": week,
              "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
              "home_score": None, "away_score": None}]
        ),
    )
    monkeypatch.setattr(
        routes.games_data, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "Texas", "away_team": "Ohio State", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(routes.sportsbook_api, "fetch_game_odds", lambda *a, **k: pd.DataFrame())
    # _get_game_prediction_live only bothers fetching odds for the actual
    # current week (see its own docstring) -- pin this away from any
    # season/week a test might use so it deterministically takes the skip
    # branch instead of calling the real current_season_and_week(), which
    # would otherwise hit games_data.fetch_schedules for real (network/
    # cache dependent, not what this fixture's mocks are for).
    monkeypatch.setattr(routes, "current_season_and_week", lambda: (9999, 99))
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {
            "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
            "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
            "feature_cols": [], "player_feature_cols": [],
        },
    )
    monkeypatch.setattr(
        routes, "_predict_game_from_models",
        lambda models, home, away, games_df, spread_line=None, total_line=None, gameday=None, conference_game=None: {
            "home_win_prob": 0.4, "away_win_prob": 0.6, "home_cover_prob": 0.45, "away_cover_prob": 0.55,
            "over_prob": 0.52, "under_prob": 0.48,
        },
    )
    monkeypatch.setattr(routes.store, "get_track_record", lambda: {"n_resolved_games": 0, "pct_moneyline_correct": None})
    return TestClient(app)


def test_get_games_returns_week_slate(client):
    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["game_id"] == "401520145"


def test_get_games_attaches_real_spread_and_total_for_the_current_week(client, monkeypatch):
    # CFBD's Game model carries no spread_line/total_line of its own (see
    # routes.py's module docstring) -- the frontend's list-view GameCard
    # reads game.spread_line directly (the same field NFL's own /games rows
    # already carry), so without this the CFB games list never has
    # anything to show even once sportsbook_api returns a real line.
    monkeypatch.setattr(routes, "current_season_and_week", lambda: (2025, 1))
    monkeypatch.setattr(
        routes.sportsbook_api, "fetch_game_odds",
        lambda *a, **k: pd.DataFrame([
            {"home_team": "Texas", "away_team": "Ohio State", "market": "spreads",
             "outcome_name": "Texas", "point": -4.5},
            {"home_team": "Texas", "away_team": "Ohio State", "market": "totals",
             "outcome_name": "Over", "point": 48.5},
        ]),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["spread_line"] == 4.5
    assert body[0]["total_line"] == 48.5


def test_get_game_prediction(client):
    response = client.get("/api/games/2025/1/401520145/prediction")

    assert response.status_code == 200
    body = response.json()
    assert body["home_win_prob"] == 0.4


def test_predict_game_from_models_includes_sigma_and_total_sigma(monkeypatch):
    class _FakeTotalModel:
        def predict(self, _X):
            return [45.0]

    monkeypatch.setattr(
        routes.feature_build, "build_features_for_game",
        lambda home, away, games_df, gameday=None, conference_game=None, **_: pd.Series({"rating_diff": 50.0, "home_rest_days": 7.0, "away_rest_days": 7.0}),
    )
    models = {
        "feature_cols": ["rating_diff", "home_rest_days", "away_rest_days"],
        "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
        "total_model": _FakeTotalModel(),
    }

    result = routes._predict_game_from_models(models, "Texas", "Oklahoma", pd.DataFrame())

    assert result["sigma"] == 12.0
    assert result["total_sigma"] == 10.0
    assert "predicted_margin" in result
    assert "predicted_total" in result


def test_get_game_prediction_404s_for_unknown_game(client):
    response = client.get("/api/games/2025/1/nonexistent/prediction")

    assert response.status_code == 404


def test_get_track_record(client):
    response = client.get("/api/track-record")

    assert response.status_code == 200
    assert response.json()["n_resolved_games"] == 0


def test_get_game_verdict_404s_when_not_resolved(client, monkeypatch):
    monkeypatch.setattr(routes.store, "get_game_verdict", lambda game_id: None)

    response = client.get("/api/games/nope/verdict")

    assert response.status_code == 404


def test_get_games_handles_nan_scores_for_unplayed_games(client, monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "401520145", "season": season, "week": week,
              "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
              "home_score": float("nan"), "away_score": float("nan")}]
        ),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["home_score"] is None
    assert body[0]["away_score"] is None


def test_get_games_includes_finished_games(client, monkeypatch):
    # A week with a mix of finished and still-upcoming games must return
    # both -- fetch_upcoming_games alone would silently drop the finished
    # one, which is exactly the bug this route must not have.
    monkeypatch.setattr(
        routes.games_data, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [
                {"game_id": "401520145", "season": season, "week": week,
                 "gameday": "2025-08-30", "home_team": "Texas", "away_team": "Ohio State",
                 "home_score": 24, "away_score": 17},
                {"game_id": "401520200", "season": season, "week": week,
                 "gameday": "2025-08-30", "home_team": "Alabama", "away_team": "Western Carolina",
                 "home_score": None, "away_score": None},
            ]
        ),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    game_ids = {g["game_id"] for g in response.json()}
    assert game_ids == {"401520145", "401520200"}


def test_lines_for_game_reads_spreads_and_totals_from_odds_api():
    odds_df = pd.DataFrame(
        [
            {"home_team": "Texas", "away_team": "Ohio State", "market": "spreads", "outcome_name": "Texas", "point": -3.5},
            {"home_team": "Texas", "away_team": "Ohio State", "market": "totals", "outcome_name": "Over", "point": 51.5},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Texas", "Ohio State")

    # The Odds API's raw point is -3.5 (home favored); _lines_for_game
    # negates it so spread_line means "home expected margin" (home
    # favored by 3.5), matching game_outcome's documented convention.
    assert spread_line == 3.5
    assert total_line == 51.5


def test_lines_for_game_returns_none_when_odds_missing():
    spread_line, total_line = routes._lines_for_game(pd.DataFrame(), "Texas", "Ohio State")

    assert spread_line is None
    assert total_line is None


def test_lines_for_game_matches_cfbd_school_name_to_odds_api_full_name():
    """CFBD's Game model carries short school names ("Texas") while The
    Odds API uses full team names ("Texas Longhorns") -- exact-string
    matching between the two never matches in practice (see this plan's
    final-review fix, Task 23, finding I5)."""
    odds_df = pd.DataFrame(
        [
            {"home_team": "Texas Longhorns", "away_team": "Ohio State Buckeyes", "market": "spreads",
             "outcome_name": "Texas Longhorns", "point": -3.5},
            {"home_team": "Texas Longhorns", "away_team": "Ohio State Buckeyes", "market": "totals",
             "outcome_name": "Over", "point": 51.5},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Texas", "Ohio State")

    assert spread_line == 3.5
    assert total_line == 51.5


def test_lines_for_game_returns_none_for_genuinely_non_matching_teams():
    assert routes._team_name_matches("Texas", "Oklahoma Sooners") is False

    # Odds feed only has a different matchup on file -- no row should match
    # "Texas" vs "Oklahoma", so this should degrade to (None, None) same as
    # a missing key or empty odds_df.
    odds_df = pd.DataFrame(
        [
            {"home_team": "Oklahoma Sooners", "away_team": "Alabama Crimson Tide", "market": "spreads",
             "outcome_name": "Oklahoma Sooners", "point": -3.5},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Texas", "Oklahoma")

    assert spread_line is None
    assert total_line is None


def test_lines_for_game_refuses_to_guess_between_ambiguous_prefix_matches():
    """_team_name_matches is a prefix match, not an exact one -- two
    distinct FBS teams can share a short-name prefix (e.g. CFBD's "Miami"
    matches both "Miami Hurricanes" and "Miami (OH) RedHawks"). If the
    odds feed's matched rows disagree on which real (home, away) pair
    they belong to, _lines_for_game must refuse to guess rather than
    silently attach a different game's line to this one."""
    odds_df = pd.DataFrame(
        [
            {"home_team": "Miami Hurricanes", "away_team": "Clemson Tigers", "market": "spreads",
             "outcome_name": "Miami Hurricanes", "point": -3.5},
            {"home_team": "Miami (OH) RedHawks", "away_team": "Clemson Tigers", "market": "spreads",
             "outcome_name": "Miami (OH) RedHawks", "point": 10.0},
        ]
    )

    spread_line, total_line = routes._lines_for_game(odds_df, "Miami", "Clemson")

    assert spread_line is None
    assert total_line is None

def test_get_player_props_includes_recent_team_and_position(client, monkeypatch):
    monkeypatch.setattr(
        routes, "_load_player_history",
        lambda season: pd.DataFrame(
            [{"player_id": "cfb-001", "player_name": "Quinn Ewers", "position": "QB",
              "recent_team": "Texas", "season": season}]
        ),
    )
    monkeypatch.setattr(
        routes.player_usage, "build_features_for_player",
        lambda player_id, history: pd.Series({"dummy_feature": 1.0}),
    )
    monkeypatch.setattr(
        routes.player_props, "predict_props",
        lambda player_models, feature_row, position: {"anytime_td_prob": 0.37, "passing_yards": 260.0},
    )

    response = client.get("/api/players/2025/1/props")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["recent_team"] == "Texas"
    assert body[0]["position"] == "QB"


def _games_df_with_final_scores(season=2026):
    return pd.DataFrame([
        {"season": season, "home_team": "Florida State", "away_team": "SMU", "home_score": 24, "away_score": 27},
    ])


def test_player_stats_needs_refresh_when_cache_missing_a_finished_teams_stats(monkeypatch, tmp_path):
    """The SMU class of bug: the cache was written before CFBD finished
    posting one side's box score, so it has rows for Florida State but none
    for SMU even though SMU's game already has a final score. This must
    force a refresh regardless of how fresh the cache file's mtime is."""
    monkeypatch.setattr(routes.player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    path = routes.player_stats._season_cache_path(2026)
    pd.DataFrame([{"recent_team": "Florida State", "season": 2026, "week": 1}]).to_parquet(path)

    assert routes._player_stats_needs_refresh(2026, _games_df_with_final_scores()) is True


def test_player_stats_does_not_need_refresh_when_cache_has_every_finished_teams_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    path = routes.player_stats._season_cache_path(2026)
    pd.DataFrame([
        {"recent_team": "Florida State", "season": 2026, "week": 1},
        {"recent_team": "SMU", "season": 2026, "week": 1},
    ]).to_parquet(path)

    assert routes._player_stats_needs_refresh(2026, _games_df_with_final_scores()) is False


def test_player_stats_refresh_check_ignores_games_with_no_final_score_yet(monkeypatch, tmp_path):
    """A team with no rows in the cache is expected and fine as long as
    their game hasn't finished yet -- only a *finished* game missing stats
    is the stale-cache signal."""
    monkeypatch.setattr(routes.player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    path = routes.player_stats._season_cache_path(2026)
    pd.DataFrame([{"recent_team": "Texas", "season": 2026, "week": 1}]).to_parquet(path)
    upcoming_game = pd.DataFrame([
        {"season": 2026, "home_team": "Texas", "away_team": "Oklahoma", "home_score": None, "away_score": None},
    ])

    assert routes._player_stats_needs_refresh(2026, upcoming_game) is False


def test_player_stats_needs_refresh_when_no_cache_exists_yet(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    assert routes._player_stats_needs_refresh(2026, _games_df_with_final_scores()) is True


def test_get_team_form_returns_last_n_games_with_result(client, monkeypatch):
    monkeypatch.setattr(
        routes, "_load_game_history",
        lambda season: pd.DataFrame([
            {"game_id": "g1", "season": season, "week": 1, "gameday": "2025-08-30",
             "home_team": "Texas", "away_team": "Ohio State", "home_score": 24, "away_score": 17},
            {"game_id": "g2", "season": season, "week": 2, "gameday": "2025-09-06",
             "home_team": "Oklahoma", "away_team": "Texas", "home_score": 30, "away_score": 10},
            {"game_id": "g3", "season": season, "week": 3, "gameday": "2025-09-13",
             "home_team": "Texas", "away_team": "Baylor", "home_score": 20, "away_score": 20},
        ]),
    )

    response = client.get("/api/teams/Texas/form?season=2025&n=5")

    assert response.status_code == 200
    body = response.json()
    assert body["team"] == "Texas"
    results = [g["result"] for g in body["recent_form"]]
    assert results == ["W", "L", "T"]
    assert body["recent_form"][1]["is_home"] is False
    assert body["recent_form"][1]["opponent"] == "Oklahoma"


def test_get_team_form_excludes_unplayed_games(client, monkeypatch):
    monkeypatch.setattr(
        routes, "_load_game_history",
        lambda season: pd.DataFrame([
            {"game_id": "g1", "season": season, "week": 1, "gameday": "2025-08-30",
             "home_team": "Texas", "away_team": "Ohio State", "home_score": 24, "away_score": 17},
            {"game_id": "g2", "season": season, "week": 5, "gameday": "2025-10-04",
             "home_team": "Texas", "away_team": "TCU", "home_score": None, "away_score": None},
        ]),
    )

    response = client.get("/api/teams/Texas/form?season=2025")

    assert response.status_code == 200
    body = response.json()
    assert len(body["recent_form"]) == 1
    assert body["recent_form"][0]["game_id"] == "g1"


def test_get_team_form_excludes_games_from_other_seasons(client, monkeypatch):
    """_load_game_history spans multiple seasons for power-rating
    continuity -- recent form must not bleed in last season's results."""
    monkeypatch.setattr(
        routes, "_load_game_history",
        lambda season: pd.DataFrame([
            {"game_id": "g_old", "season": 2024, "week": 14, "gameday": "2024-11-30",
             "home_team": "Texas", "away_team": "Texas A&M", "home_score": 17, "away_score": 7},
            {"game_id": "g_new", "season": 2025, "week": 1, "gameday": "2025-08-30",
             "home_team": "Texas", "away_team": "Ohio State", "home_score": 24, "away_score": 17},
        ]),
    )

    response = client.get("/api/teams/Texas/form?season=2025&n=5")

    assert response.status_code == 200
    form = response.json()["recent_form"]
    assert len(form) == 1
    assert form[0]["game_id"] == "g_new"


def test_get_head_to_head_returns_past_meetings_between_the_two_teams(client, monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_week_games",
        lambda season, week: pd.DataFrame([
            {"game_id": "current_game", "season": season, "week": week, "gameday": "2025-11-01",
             "home_team": "Texas", "away_team": "Oklahoma", "home_score": None, "away_score": None},
        ]),
    )
    monkeypatch.setattr(
        routes, "_load_game_history",
        lambda season: pd.DataFrame([
            {"game_id": "past1", "season": season - 1, "week": 6, "gameday": "2024-10-12",
             "home_team": "Oklahoma", "away_team": "Texas", "home_score": 10, "away_score": 34},
            {"game_id": "unrelated", "season": season, "week": 3, "gameday": "2025-09-13",
             "home_team": "Texas", "away_team": "Baylor", "home_score": 20, "away_score": 20},
        ]),
    )

    response = client.get("/api/games/current_game/head-to-head?season=2025&week=9&n_seasons=8")

    assert response.status_code == 200
    body = response.json()
    assert body["game_id"] == "current_game"
    assert len(body["meetings"]) == 1
    assert body["meetings"][0]["game_id"] == "past1"
    assert body["meetings"][0]["home_team"] == "Oklahoma"


def test_get_head_to_head_404s_for_unknown_game_id(client):
    response = client.get("/api/games/nonexistent/head-to-head?season=2025&week=1")

    assert response.status_code == 404


def test_get_predictions_batch_returns_dict_keyed_by_game_id(client):
    response = client.get("/api/predictions/2025/1/batch")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert body["401520145"]["home_win_prob"] == 0.4


def test_get_predictions_batch_fetches_odds_once_for_the_whole_week(client, monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_week_games",
        lambda season, week: pd.DataFrame([
            {"game_id": "g1", "season": season, "week": week, "gameday": "2025-08-30",
             "home_team": "Texas", "away_team": "Ohio State", "home_score": None, "away_score": None},
            {"game_id": "g2", "season": season, "week": week, "gameday": "2025-08-30",
             "home_team": "Alabama", "away_team": "Georgia", "home_score": None, "away_score": None},
        ]),
    )
    monkeypatch.setattr(routes, "current_season_and_week", lambda: (2025, 1))
    calls = []
    monkeypatch.setattr(
        routes.sportsbook_api, "fetch_game_odds",
        lambda *a, **k: (calls.append(1), pd.DataFrame())[1],
    )

    response = client.get("/api/predictions/2025/1/batch")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"g1", "g2"}
    # Exactly one call for the whole week's games, never once per game --
    # CFB's Sportsbook API quota is a hard 150/day cap shared across
    # projects.
    assert len(calls) == 1


def test_get_predictions_batch_one_bad_game_does_not_fail_the_rest(client, monkeypatch):
    monkeypatch.setattr(
        routes.games_data, "fetch_week_games",
        lambda season, week: pd.DataFrame([
            {"game_id": "good", "season": season, "week": week, "gameday": "2025-08-30",
             "home_team": "Texas", "away_team": "Ohio State", "home_score": None, "away_score": None},
            {"game_id": "bad", "season": season, "week": week, "gameday": "2025-08-30",
             "home_team": "Broken", "away_team": "Team", "home_score": None, "away_score": None},
        ]),
    )

    def flaky_predict(models, home, away, games_df, spread_line=None, total_line=None, gameday=None, conference_game=None):
        if home == "Broken":
            raise ValueError("feature build blew up for this one game")
        return {"home_win_prob": 0.4, "away_win_prob": 0.6}

    monkeypatch.setattr(routes, "_predict_game_from_models", flaky_predict)

    response = client.get("/api/predictions/2025/1/batch")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"good"}


def test_get_predictions_for_week_returns_list_of_predictions(client, monkeypatch):
    monkeypatch.setattr(
        routes.store, "get_predictions_for_week",
        lambda season, week, games: [
            {"game_id": "401520145", "status": "pending", "home_win_prob": 0.4, "away_win_prob": 0.6, "verdict": None},
        ]
    )

    response = client.get("/api/predictions/2025/1")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["game_id"] == "401520145"
    assert body[0]["status"] == "pending"


def test_get_power_rankings_sorts_descending_by_rating(client, monkeypatch):
    monkeypatch.setattr(
        routes, "_load_game_history",
        lambda season: pd.DataFrame([
            {"game_id": "g1", "season": season, "week": 1, "gameday": "2025-08-30",
             "home_team": "Texas", "away_team": "Ohio State", "home_score": 40, "away_score": 10,
             "home_conference": "Big 12", "away_conference": "Big Ten",
             "home_division": "fbs", "away_division": "fbs"},
            {"game_id": "g2", "season": season, "week": 2, "gameday": "2025-09-06",
             "home_team": "Ohio State", "away_team": "Texas", "home_score": 10, "away_score": 40,
             "home_conference": "Big Ten", "away_conference": "Big 12",
             "home_division": "fbs", "away_division": "fbs"},
        ]),
    )

    response = client.get("/api/power-rankings?season=2025")

    assert response.status_code == 200
    body = response.json()
    assert body["season"] == 2025
    ratings = [row["rating"] for row in body["rankings"]]
    assert ratings == sorted(ratings, reverse=True)
    assert body["rankings"][0]["rank"] == 1
    # Texas won both meetings decisively -- should be ranked above Ohio State.
    assert body["rankings"][0]["team"] == "Texas"


def test_get_power_rankings_cfb_response_has_no_division_field(client, monkeypatch):
    """CFB has no divisions (only conferences realign, see
    models/season_projection.py) -- unlike NFL's power rankings, the CFB
    response must never include a `division` field."""
    monkeypatch.setattr(
        routes, "_load_game_history",
        lambda season: pd.DataFrame([
            {"game_id": "g1", "season": season, "week": 1, "gameday": "2025-08-30",
             "home_team": "Texas", "away_team": "Ohio State", "home_score": 40, "away_score": 10,
             "home_conference": "Big 12", "away_conference": "Big Ten",
             "home_division": "fbs", "away_division": "fbs"},
        ]),
    )

    response = client.get("/api/power-rankings?season=2025")

    assert response.status_code == 200
    body = response.json()
    assert body["rankings"], "expected at least one ranking row"
    for row in body["rankings"]:
        assert "division" not in row
        assert row["conference"] in {"Big 12", "Big Ten"}


def test_get_predictions_for_week_includes_both_resolved_and_pending_games(client, monkeypatch):
    """A week in progress has both finished games (from load_training_data)
    and still-upcoming ones (from fetch_upcoming_games). The old if/else
    fallback dropped the finished game whenever any game that week was
    still unplayed (see this plan's final review, finding B3); the route
    must union both sources instead. The client fixture's default mocks
    already return one pending game ("401520145", week=1) from
    fetch_upcoming_games and one finished game ("g0", week=1) from
    load_training_data."""
    captured = {}

    def fake_get_predictions_for_week(season, week, games):
        captured["game_ids"] = set(games["game_id"])
        return [
            {"game_id": gid, "status": "resolved" if gid == "g0" else "pending", "verdict": None}
            for gid in games["game_id"]
        ]

    monkeypatch.setattr(routes.store, "get_predictions_for_week", fake_get_predictions_for_week)

    response = client.get("/api/predictions/2025/1")

    assert response.status_code == 200
    assert captured["game_ids"] == {"401520145", "g0"}
    body = response.json()
    statuses = {row["game_id"]: row["status"] for row in body}
    assert statuses["401520145"] == "pending"
    assert statuses["g0"] == "resolved"


# --- Track-record payload split (Sports_Predictor PR #81 contract) ---------
# `?picks=deferred` returns the summary with NO row-level picks; the game rows
# move to `/track-record/picks`; a plain request keeps the historical payload.
# On this service the split is the difference between ~5.8 MB and ~10 KB: the
# full record's bulk is `player_props.per_pick` rows no surface renders.

_SPLIT_RECORD = {
    "games": {
        "n_resolved": 2,
        "pct_moneyline_correct": 0.5,
        "per_pick": [
            {"game_id": "401520145", "pick": "Texas"},
            {"game_id": "401520146", "pick": "Ohio State"},
        ],
    },
    "player_props": {
        "n_resolved": 3,
        "per_pick": [{"player_id": "4688386", "market": "passing_yards"}],
    },
    "n_resolved_games": 2,
}


@pytest.fixture
def split_client(client, monkeypatch):
    """The shared `client` fixture's store stub carries no pick rows; this one
    serves a record that does, so the split has something to split. Deep-copied
    per call because the deferred path strips keys off the dict it is handed."""
    monkeypatch.setattr(
        routes.store, "get_track_record",
        lambda: copy.deepcopy(_SPLIT_RECORD),
    )
    monkeypatch.setattr(routes.store, "get_calibration", lambda: {"winner": []})
    return client


def test_track_record_default_still_inlines_per_pick_rows(split_client):
    response = split_client.get("/api/track-record")

    assert response.status_code == 200
    body = response.json()
    assert body["games"]["per_pick"] == _SPLIT_RECORD["games"]["per_pick"]
    assert body["player_props"]["per_pick"] == _SPLIT_RECORD["player_props"]["per_pick"]
    assert body["games"]["n_resolved"] == 2
    assert body["calibration"] == {"winner": []}


def test_track_record_deferred_omits_rows_and_keeps_summary_identical(split_client):
    response = split_client.get("/api/track-record?picks=deferred")

    assert response.status_code == 200
    body = response.json()
    assert "per_pick" not in body["games"]
    assert "per_pick" not in body["player_props"]
    expected = copy.deepcopy(_SPLIT_RECORD)
    del expected["games"]["per_pick"]
    del expected["player_props"]["per_pick"]
    expected["calibration"] = {"winner": []}
    assert body == expected


def test_track_record_picks_endpoint_returns_the_game_rows(split_client):
    response = split_client.get("/api/track-record/picks")

    assert response.status_code == 200
    assert response.json() == {"per_pick": _SPLIT_RECORD["games"]["per_pick"]}


def test_track_record_unknown_picks_value_keeps_full_payload(split_client):
    body = split_client.get("/api/track-record?picks=inline").json()

    assert body["games"]["per_pick"] == _SPLIT_RECORD["games"]["per_pick"]
    assert body["player_props"]["per_pick"] == _SPLIT_RECORD["player_props"]["per_pick"]
