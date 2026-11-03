import json

import numpy as np
import pandas as pd
import pytest

from cfb_predictor.models import manifest


def _fake_games(seasons):
    rng = np.random.default_rng(3)
    rows = []
    teams = [f"T{i}" for i in range(8)]
    for season in seasons:
        for week in range(1, 6):
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                rows.append(
                    {
                        "game_id": f"{season}_{week}_{home}_{away}", "season": season, "week": week,
                        "gameday": pd.Timestamp(f"{season}-08-25") + pd.Timedelta(days=7 * (week - 1)),
                        "home_team": home, "away_team": away,
                        "home_score": int(rng.integers(10, 45)), "away_score": int(rng.integers(10, 45)),
                        "home_division": "fbs", "away_division": "fbs", "conference_game": False,
                    }
                )
    return pd.DataFrame(rows)


def _fake_fbs_teams(season):
    return pd.DataFrame({"team": [f"T{i}" for i in range(8)], "conference": ["X"] * 8})


def _fake_player_stats(seasons, games_df=None):
    rng = np.random.default_rng(4)
    rows = []
    for season in seasons:
        for week in range(1, 6):
            rows.append(
                {
                    "player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "T0",
                    "season": season, "week": week,
                    "passing_yards": 0, "passing_tds": 0,
                    "rushing_yards": int(rng.integers(50, 140)), "rushing_tds": int(rng.integers(0, 2)),
                    "receiving_yards": int(rng.integers(0, 30)), "receiving_tds": 0,
                    "receptions": 2, "targets": float("nan"), "carries": 18,
                }
            )
    return pd.DataFrame(rows)


def test_train_all_writes_a_manifest_with_chosen_candidate(monkeypatch, tmp_path):
    from cfb_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", _fake_fbs_teams)
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", _fake_player_stats)

    result = manifest.train_all(seasons=seasons, total_spec=None)

    assert result["chosen_candidate"] in ("elo", "ridge", "xgb")
    assert (tmp_path / "manifest.json").exists()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["chosen_candidate"] == result["chosen_candidate"]
    assert (tmp_path / "game_outcome_model.pkl").exists()
    assert (tmp_path / "total_points_model.pkl").exists()
    assert (tmp_path / "anytime_td_model.pkl").exists()
    assert (tmp_path / "rushing_yards_model.pkl").exists()


def test_train_all_publishes_yardage_holdout_figures(monkeypatch, tmp_path):
    """The yardage entries carry the walk-forward holdout figures next to
    n_train: fit on earlier seasons only, scored on the last two — the
    honest error scale the prop grader bands by."""
    from cfb_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", _fake_fbs_teams)
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", _fake_player_stats)

    result = manifest.train_all(seasons=seasons, total_spec=None)

    figures = result["yardage_metrics"]["rushing_yards"]
    assert figures["n_train"] > 0
    assert figures["n_holdout"] == 10  # the fake RB's 2023 + 2024 rows, five weeks each
    assert figures["holdout_mae"] >= 0
    assert figures["naive_mae"] is not None and figures["naive_mae"] >= 0
    assert figures["holdout"]["val_seasons"] == [2023, 2024]


def test_load_models_round_trips_after_train_all(monkeypatch, tmp_path):
    from cfb_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", _fake_fbs_teams)
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", _fake_player_stats)

    manifest.train_all(seasons=seasons, total_spec=None)
    models = manifest.load_models()

    assert "game_outcome_model" in models
    assert "player_models" in models
    assert "anytime_td" in models["player_models"]


def test_train_all_rejects_training_data_without_walk_forward_fold(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest.games_data, "load_training_data", lambda s: _fake_games([2024]))
    monkeypatch.setattr(manifest.games_data, "fetch_fbs_teams", _fake_fbs_teams)

    with pytest.raises(ValueError, match="walk-forward validation fold"):
        manifest.train_all(seasons=[2024])


def test_committed_yardage_artifacts_match_their_declared_objectives():
    """The pickles shipped in models/ were fitted with the objective the code says each market gets."""
    import json
    from cfb_predictor.config import MODELS_DIR
    from cfb_predictor.models import manifest, player_props

    declared = json.loads((MODELS_DIR / "manifest.json").read_text())["yardage_objectives"]
    assert set(declared) == set(player_props.ABSOLUTE_ERROR_MARKETS)
    for market, objective in declared.items():
        fitted = manifest._load_pickle(manifest._yardage_model_path(market))
        assert fitted.get_params()["objective"] == objective == "reg:absoluteerror"
