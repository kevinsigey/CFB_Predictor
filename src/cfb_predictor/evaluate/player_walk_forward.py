"""player_walk_forward.py — season-by-season walk-forward holdout for the
yardage prop models (`models/player_props.py`), the player-level counterpart
of `evaluate/walk_forward.py`'s game folds.

Why this exists: `models/manifest.py::train_all` fits each yardage regressor
on every row it has and recorded only `n_train` per market. The prop grader
(`tracking/prop_grading.py`) judges live picks in units of the model's own
holdout error, and no holdout was ever measured for these models — so every
grade came back None rather than be banded by an in-sample number (an error
measured on the rows the fit was trained on is optimistic by construction,
and a band built from it would call misses "Close" that an honest scale
calls "Off").

The discipline is the game folds', applied to the player frame:

* **Features are built once, then sliced by season.** Every player feature
  is a `shift(1)` rolling mean (`features/player_usage.py`), so a row's
  features are already causal — they see only games before the row's own.
  Building the frame over all seasons and slicing afterwards leaks nothing,
  the same argument `walk_forward.py` makes for the game frame.
* **A fold's fit sees earlier seasons only.** For validation season Y the
  regressor is fitted on rows from seasons < Y and scored on season Y —
  the production fitter itself (`fit_yardage_regressor`, market objective
  and all), on the production row subset (the market's target > 0), so the
  number describes the model family that actually ships, not a stand-in.
* **The naive baseline is the player's own rolling mean of the target** —
  the `{stat}_roll` column, i.e. the baseline the repo's features already
  imply and the same one NBA's prop holdout uses. It is scored on the same
  validation rows wherever that baseline exists; a row whose player has no
  prior recorded game has no baseline and is counted (`n_naive`), never
  silently filled.

Residuals are pooled across folds (summed absolute error over summed rows),
so the published MAE is the per-row error over every held-out row, weighted
by volume rather than by season.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..models import player_props


def prepare_player_folds(player_df: pd.DataFrame, min_train_seasons: int = 2) -> list[dict]:
    """Walk-forward folds over a built player frame, sliced by season.

    Mirrors `walk_forward.prepare_folds`: fold i validates on season
    `seasons[i]` against a fit on every earlier season in the frame, and a
    fold exists only once at least `min_train_seasons` earlier seasons do.
    """
    seasons = sorted(int(s) for s in player_df["season"].unique())
    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_df = player_df[player_df["season"].isin(seasons[:i])]
        val_df = player_df[player_df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        folds.append({"val_season": val_season, "train_df": train_df, "val_df": val_df})
    return folds


def _production_fit(X: pd.DataFrame, y: pd.Series, market: str):
    """The shipped fitter, so the holdout measures the shipped model family."""
    return player_props.fit_yardage_regressor(X, y, market=market)


def yardage_holdout_metrics(
    player_df: pd.DataFrame,
    feature_cols: list[str],
    min_train_seasons: int = 2,
    fit_fn: Callable | None = None,
) -> dict[str, dict]:
    """Pooled walk-forward holdout MAE and naive MAE per yardage market.

    `player_df` is the frame `train_all` fits on (built by
    `features.player_usage.build_player_training_frame`, all-NaN-feature
    rows already dropped, exactly as the production fit receives it).
    Returns one entry per market that at least one fold could score; a
    market no fold scored is ABSENT, so a caller can never publish a number
    nobody measured. `fit_fn(X, y, market) -> fitted model` is injectable
    for tests; production uses the shipped regressor.
    """
    fit = fit_fn or _production_fit
    folds = prepare_player_folds(player_df, min_train_seasons=min_train_seasons)

    metrics: dict[str, dict] = {}
    for market, target_col in player_props.YARDAGE_TARGETS.items():
        roll_col = f"{target_col}_roll"
        abs_err = 0.0
        n_holdout = 0
        naive_abs_err = 0.0
        n_naive = 0
        val_seasons: list[int] = []
        for fold in folds:
            train_subset = fold["train_df"][fold["train_df"][target_col] > 0]
            val_subset = fold["val_df"][fold["val_df"][target_col] > 0]
            if train_subset.empty or val_subset.empty:
                continue
            model = fit(
                train_subset[feature_cols].fillna(0), train_subset[target_col], market
            )
            preds = np.asarray(model.predict(val_subset[feature_cols].fillna(0)), dtype=float)
            actual = val_subset[target_col].to_numpy(dtype=float)
            abs_err += float(np.abs(preds - actual).sum())
            n_holdout += int(len(val_subset))
            val_seasons.append(int(fold["val_season"]))

            if roll_col in val_subset.columns:
                baseline = val_subset[roll_col]
                has_baseline = baseline.notna()
                if has_baseline.any():
                    naive_abs_err += float(
                        np.abs(
                            baseline[has_baseline].to_numpy(dtype=float)
                            - actual[has_baseline.to_numpy()]
                        ).sum()
                    )
                    n_naive += int(has_baseline.sum())
        if not n_holdout:
            continue
        metrics[market] = {
            "holdout_mae": abs_err / n_holdout,
            "naive_mae": (naive_abs_err / n_naive) if n_naive else None,
            "n_holdout": n_holdout,
            "n_naive": n_naive,
            "holdout": {
                "kind": "walk_forward_seasons",
                "val_seasons": val_seasons,
                "min_train_seasons": int(min_train_seasons),
            },
        }
    return metrics
