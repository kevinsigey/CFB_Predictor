"""Yardage walk-forward holdout: the published MAE must be out-of-sample.

The prop grader bands live picks in units of these numbers, so the three
properties under test are the ones the measurement's honesty rests on:

- folds are strictly chronological -- a fold's fit stands on earlier
  seasons only, and the injected-fitter test would fail if a single
  validation-season row reached a fit;
- the naive MAE is the player's own causal rolling mean of the target on
  the same held-out rows, with baseline-less rows counted, not filled;
- residuals pool by row across folds, so the published figure is the
  per-row error over the whole holdout, not a mean of per-season means.
"""
import numpy as np
import pandas as pd

from cfb_predictor.evaluate import player_walk_forward

ROLL_COLS = [
    "passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll",
    "targets_roll", "carries_roll", "receptions_roll",
]


def _frame(seasons, rows_per_season=3):
    """A minimal built player frame: every market's target positive except
    receiving/receptions (structurally zero here, so those markets must
    never be scored)."""
    rows = []
    for season in seasons:
        for i in range(rows_per_season):
            rows.append({
                "player_id": f"p{i}", "season": season, "week": i + 1,
                "passing_yards": 100.0, "rushing_yards": 50.0,
                "receiving_yards": 0.0, "carries": 10.0, "receptions": 0.0,
                "passing_yards_roll": 90.0, "rushing_yards_roll": 40.0,
                "receiving_yards_roll": 0.0, "targets_roll": 1.0,
                "carries_roll": 9.0, "receptions_roll": 0.0,
            })
    return pd.DataFrame(rows)


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _constant_fit(value):
    def fit(X, y, market):
        return _ConstantModel(value)

    return fit


def test_folds_are_strictly_chronological():
    folds = player_walk_forward.prepare_player_folds(
        _frame(range(2018, 2026)), min_train_seasons=6
    )
    assert [f["val_season"] for f in folds] == [2024, 2025]
    for fold in folds:
        assert fold["train_df"]["season"].max() < fold["val_season"]
        assert set(fold["val_df"]["season"].unique()) == {fold["val_season"]}


def test_fit_never_sees_a_validation_season_row():
    """The leakage tripwire: the injected fitter records the seasons of
    every frame it is fitted on. Fitting on all seasons (or a shuffled
    split) would put the validation season in the record and fail this."""
    recorded = []

    def recording_fit(X, y, market):
        recorded.append(sorted(int(s) for s in X["season"].unique()))
        return _ConstantModel(0.0)

    frame = _frame(range(2018, 2026))
    metrics = player_walk_forward.yardage_holdout_metrics(
        frame, ["season"], min_train_seasons=6, fit_fn=recording_fit
    )
    # Three scored markets (passing, rushing, carries) x two folds, in
    # fold order within each market: fits stand on 2018-2023, then 2018-2024.
    expected = [list(range(2018, 2024)), list(range(2018, 2025))]
    assert recorded == expected * 3
    assert metrics["passing_yards"]["holdout"]["val_seasons"] == [2024, 2025]


def test_naive_mae_is_the_players_own_rolling_mean_on_the_same_rows():
    frame = pd.DataFrame([
        # Training season: the fit stands on these and scores none of them.
        {"player_id": "p1", "season": 2023, "week": 1, "passing_yards": 100.0,
         "rushing_yards": 0.0, "receiving_yards": 0.0, "carries": 0.0, "receptions": 0.0,
         "passing_yards_roll": 90.0},
        # Validation season: known answer key.
        {"player_id": "p1", "season": 2024, "week": 1, "passing_yards": 100.0,
         "rushing_yards": 0.0, "receiving_yards": 0.0, "carries": 0.0, "receptions": 0.0,
         "passing_yards_roll": 90.0},   # naive error |90 - 100| = 10
        {"player_id": "p2", "season": 2024, "week": 1, "passing_yards": 60.0,
         "rushing_yards": 0.0, "receiving_yards": 0.0, "carries": 0.0, "receptions": 0.0,
         "passing_yards_roll": 80.0},   # naive error |80 - 60| = 20
        {"player_id": "p3", "season": 2024, "week": 1, "passing_yards": 50.0,
         "rushing_yards": 0.0, "receiving_yards": 0.0, "carries": 0.0, "receptions": 0.0,
         "passing_yards_roll": np.nan},  # no prior game: no baseline exists
    ])
    metrics = player_walk_forward.yardage_holdout_metrics(
        frame, ["passing_yards_roll"], min_train_seasons=1, fit_fn=_constant_fit(0.0)
    )
    passing = metrics["passing_yards"]
    # The constant-zero model is scored on all three held-out rows...
    assert passing["n_holdout"] == 3
    assert passing["holdout_mae"] == (100.0 + 60.0 + 50.0) / 3
    # ...the naive baseline only on the two rows that have one.
    assert passing["n_naive"] == 2
    assert passing["naive_mae"] == (10.0 + 20.0) / 2


def test_residuals_pool_by_row_not_by_fold():
    """One fold with a single row and one with three: the pooled MAE is
    total error over total rows (17.5), not the mean of fold MAEs (15)."""
    rows = []
    for season, actuals in ((2022, [999.0]), (2023, [10.0]), (2024, [20.0, 20.0, 20.0])):
        for i, actual in enumerate(actuals):
            rows.append({
                "player_id": f"p{season}_{i}", "season": season, "week": 1,
                "passing_yards": actual, "rushing_yards": 0.0, "receiving_yards": 0.0,
                "carries": 0.0, "receptions": 0.0, "passing_yards_roll": 0.0,
            })
    metrics = player_walk_forward.yardage_holdout_metrics(
        pd.DataFrame(rows), ["passing_yards_roll"], min_train_seasons=1,
        fit_fn=_constant_fit(0.0),
    )
    assert metrics["passing_yards"]["n_holdout"] == 4
    assert metrics["passing_yards"]["holdout_mae"] == (10.0 + 20.0 * 3) / 4


def test_market_with_no_positive_holdout_rows_is_absent_not_zero():
    metrics = player_walk_forward.yardage_holdout_metrics(
        _frame(range(2018, 2026)), ROLL_COLS, min_train_seasons=6,
        fit_fn=_constant_fit(0.0),
    )
    assert set(metrics) == {"passing_yards", "rushing_yards", "carries"}
