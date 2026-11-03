"""Train, save, and load the game-outcome and player-prop models."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import pandas as pd

from ..config import MODELS_DIR
from ..data import games as games_data
from ..data import player_stats
from ..evaluate import player_walk_forward, walk_forward
from ..data import cfbd_advanced
from ..features import build as feature_build
from ..features.build import DEFAULT_BLOCKS
from ..features import player_usage
from . import game_outcome, player_props, qb_passing_td

MANIFEST_PATH = MODELS_DIR / "manifest.json"
GAME_MODEL_FILENAME = "game_outcome_model.pkl"
TOTAL_MODEL_FILENAME = "total_points_model.pkl"
ANYTIME_TD_MODEL_FILENAME = "anytime_td_model.pkl"

#: Its OWN file, not `f"{market}_model.pkl"`, for the same reason the yardage
#: models get one: the QB passing-TD fit is a different SHAPE (a fitted dict
#: holding a distribution, an alpha and both log losses) from a bare regressor,
#: and a name that looks like the others invites loading it as one.
QB_PASSING_TD_MODEL_FILENAME = "qb_passing_td_model.pkl"

DEFAULT_TRAIN_SEASONS = 8


class StaleArtifactError(ValueError):
    """The committed models no longer match the code that would serve them.

    Its own type, rather than a bare `ValueError`, because callers need to tell
    this apart from every other `ValueError` on the path: `load_manifest`
    parses JSON, and a truncated or corrupt `manifest.json` raises
    `json.JSONDecodeError`, which IS a `ValueError`. Catching `ValueError` at
    the API edge would then report a corrupt file as "stale trained model" and
    send an operator to retrain when the actual fix is to restore the file.
    Subclasses `ValueError` so existing `pytest.raises(ValueError)` assertions
    keep holding.
    """


def _save_pickle(obj, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _artifact_path(filename: str):
    return MODELS_DIR / filename


def _yardage_model_path(market: str):
    return _artifact_path(f"{market}_model.pkl")


def _fit_qb_passing_td(player_train_df: pd.DataFrame) -> dict | None:
    """Fit and persist the QB passing-TD model, or return None if there is none.

    **None, and the artefact REMOVED, when no QB has usable history** -- the same
    discipline the yardage loop in `train_all` already uses. A model fitted on
    nothing raises inside `fit_qb_passing_td_model`, and the alternative (an
    artefact that predicts a flat distribution for every QB forever, from a market
    that was never really trained) is worse than no market: it looks fitted in a
    manifest and scores every QB.

    The manifest entry carries the fit's OWN numbers -- which distribution won,
    both log losses, alpha, the variance ratio, `n_train` and the exact columns
    read -- because NFL's docstring once described its artefact as it was *before*
    a retrain onto anytime-TD label v2 features and nothing noticed. Prose is not
    asserted by default; a manifest key is read by whatever retrains next.

    Only QBs are fitted. `passing_tds` is 0 for every non-QB, so including them
    would add a mass point at zero and make the variance-to-mean ratio -- the
    quantity the Poisson-vs-negative-binomial choice reacts to -- a statement
    about how many running backs there are.
    """
    path = _artifact_path(QB_PASSING_TD_MODEL_FILENAME)
    has_position = "position" in player_train_df.columns
    qb = player_train_df[player_train_df["position"] == "QB"] if has_position else player_train_df.iloc[0:0]
    if qb.empty or "passing_tds" not in qb.columns:
        path.unlink(missing_ok=True)
        return None

    # `passing_tds_roll` is NaN on each QB's first week -- there is no prior week
    # to roll -- and those rows carry no information about a next week's
    # expectation. Dropping on that column ALONE is deliberate: `dropna` over every
    # feature would silently discard rows where an unrelated stat was missing, and
    # the count of what survived is the `n_train` recorded below.
    cols = list(qb_passing_td.MU_FEATURE_COLUMNS) + [player_usage.PASSING_TDS_ROLL_COLUMN]
    usable = qb.dropna(subset=[player_usage.PASSING_TDS_ROLL_COLUMN])
    if usable.empty:
        path.unlink(missing_ok=True)
        return None

    fitted = qb_passing_td.fit_qb_passing_td_model(
        usable[cols].fillna(0), usable["passing_tds"].astype(float)
    )
    _save_pickle(fitted, path)
    return {
        "distribution": fitted["distribution"],
        "alpha": fitted["alpha"],
        "log_loss": fitted["log_loss"],
        "variance_ratio": fitted["variance_ratio"],
        "n_train": fitted["n_train"],
        "feature_cols": fitted["feature_cols"],
    }


def _fbs_teams_by_season(seasons: list[int]) -> dict[int, set[str]]:
    """One data.games.fetch_fbs_teams() call per season (each cached after
    first fetch) -- the authoritative FBS team universe build_training_frame
    and evaluate.walk_forward.prepare_folds use to exclude FCS-opponent
    games from every candidate's training rows."""
    return {season: set(games_data.fetch_fbs_teams(season)["team"]) for season in seasons}


#: The TOTAL model's own spec. Margin and win share one feature set (DEFAULT_BLOCKS); the total does not: ridge on those
#: columns plus the opponent-adjusted EPA block beat the XGBoost total on the repo's walk-forward (MAE 13.673 -> 13.398,
#: paired-bootstrap CI of the gain [+0.19, +0.37]; total-bias gap 2.78 -> 0.51). `python -m cfb_predictor.tools.total_eval`
#: re-runs that comparison. `total_spec=None` keeps the legacy XGBoost total on the shared columns.
TOTAL_SPEC = {"kind": "ridge", "blocks": ["adj_epa"]}
_TOTAL_FITTERS = {"ridge": game_outcome.fit_margin_regression, "xgb": game_outcome.fit_xgb_margin}
_BLOCKS_NEEDING_EFFICIENCY = {"epa", "adj_epa"}


def total_spec_columns(spec: dict) -> list[str]:
    """The columns a total spec's model reads, derived from the CODE (never from a manifest)."""
    return feature_build.feature_columns(tuple(spec["blocks"]))


def fit_sigmas(folds: list[dict], chosen: str) -> tuple[float, float]:
    """(margin sigma, total sigma) from pooled out-of-fold residuals: the chosen game model for the margin and the
    legacy XGBoost total model for the total, each predicted on seasons it was never trained on."""
    sigma = game_outcome.sigma_from_residuals(walk_forward.oof_residuals(folds, chosen, "margin"))
    total_sigma = game_outcome.sigma_from_residuals(walk_forward.oof_residuals(folds, "xgb", "total_points"))
    return sigma, total_sigma


def fit_total(games_df: pd.DataFrame, fbs_teams: dict, seasons: list[int], min_train_seasons: int, spec: dict) -> tuple:
    """(model, total_sigma) for a total spec. Sigma is from the out-of-fold residuals of THIS model on THESE features.

    A spec that needs the efficiency aux raises when the committed data does not cover the training seasons: a total
    model fitted on a zero-filled block is the fabricated-features defect, not a degraded mode."""
    blocks = tuple(spec["blocks"])
    aux = None
    if _BLOCKS_NEEDING_EFFICIENCY & set(blocks):
        efficiency = cfbd_advanced.efficiency_frame_from_disk(years=seasons)
        covered = set(efficiency["season"].dropna().astype(int)) if len(efficiency) else set()
        if not set(seasons) <= covered:
            raise ValueError(f"no committed CFBD advanced stats for seasons {sorted(set(seasons) - covered)}; "
                             "the total spec's epa block cannot be fitted")
        aux = feature_build.Aux(efficiency=efficiency)
    frame, cols = feature_build.build_training_frame(games_df, fbs_teams=fbs_teams, blocks=blocks, aux=aux)
    folds = walk_forward.prepare_folds(games_df, fbs_teams=fbs_teams, min_train_seasons=min_train_seasons, blocks=blocks, aux=aux)
    if not folds:
        raise ValueError("Training the total model requires at least one walk-forward validation fold.")
    kind = spec["kind"]
    sigma = game_outcome.sigma_from_residuals(walk_forward.oof_residuals(folds, kind, "total_points"))
    return _TOTAL_FITTERS[kind](frame[cols], frame["total_points"]), sigma


def train_all(
    seasons: list[int] | None = None, blocks: tuple[str, ...] = DEFAULT_BLOCKS, total_spec: dict | None = TOTAL_SPEC,
) -> dict:
    """Fit all models, persist their artifacts, and return their manifest."""
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    seasons = seasons or games_data.default_completed_seasons(n=DEFAULT_TRAIN_SEASONS)

    games_df = games_data.load_training_data(seasons)
    fbs_teams = _fbs_teams_by_season(seasons)
    train_df, feature_cols = feature_build.build_training_frame(games_df, fbs_teams=fbs_teams, blocks=blocks)

    folds = walk_forward.prepare_folds(games_df, fbs_teams=fbs_teams, min_train_seasons=max(1, len(seasons) - 2))
    if not folds:
        raise ValueError("Training requires at least one walk-forward validation fold.")
    candidate_scores = {}
    for candidate in ("elo", "ridge", "xgb"):
        scored = walk_forward.evaluate_candidate(folds, candidate) if folds else pd.DataFrame()
        candidate_scores[candidate] = float(scored["log_loss"].mean()) if not scored.empty else float("inf")
    chosen = min(candidate_scores, key=candidate_scores.get)

    X_train = train_df[feature_cols]
    y_margin = train_df["margin"]

    if chosen == "elo":
        game_model = game_outcome.fit_elo_candidate(train_df)
    elif chosen == "ridge":
        game_model = game_outcome.fit_margin_regression(X_train, y_margin)
    else:
        game_model = game_outcome.fit_xgb_margin(X_train, y_margin)

    # Sigma from OUT-OF-SAMPLE residuals (the pooled walk-forward folds), never from the training rows the model
    # was fitted on: an in-sample sigma is optimistic and made every published probability overconfident.
    sigma, total_sigma = fit_sigmas(folds, chosen)
    if total_spec is None:
        total_model = game_outcome.fit_xgb_margin(X_train, train_df["total_points"])
    else:
        total_model, total_sigma = fit_total(
            games_df, fbs_teams, seasons, max(1, len(seasons) - 2), total_spec)

    _save_pickle(game_model, _artifact_path(GAME_MODEL_FILENAME))
    _save_pickle(total_model, _artifact_path(TOTAL_MODEL_FILENAME))

    player_df_raw = player_stats.fetch_weekly_player_stats(seasons, games_df)
    player_train_df, player_feature_cols = player_usage.build_player_training_frame(player_df_raw)
    player_train_df = player_train_df.dropna(subset=player_feature_cols, how="all")
    X_player = player_train_df[player_feature_cols].fillna(0)

    anytime_td_model = player_props.fit_anytime_td_classifier(X_player, player_train_df["anytime_td"])
    _save_pickle(anytime_td_model, _artifact_path(ANYTIME_TD_MODEL_FILENAME))

    # The QB passing-TD market. `MU_FEATURE_COLUMNS` plus `passing_tds_roll`,
    # which `player_usage` deliberately keeps OUT of `PLAYER_FEATURE_COLUMNS` so
    # widening that list would not change the feature count of every already-
    # committed artefact. So the columns this model reads are assembled HERE,
    # from the model module's own declaration and the feature module's own column
    # name -- never a literal list of either.
    qb_metric = _fit_qb_passing_td(player_train_df)

    yardage_metrics = {}
    for market, target_col in player_props.YARDAGE_TARGETS.items():
        path = _yardage_model_path(market)
        subset = player_train_df[player_train_df[target_col] > 0]
        if subset.empty:
            path.unlink(missing_ok=True)
            continue
        model = player_props.fit_yardage_regressor(
            subset[player_feature_cols].fillna(0), subset[target_col], market=market
        )
        _save_pickle(model, path)
        yardage_metrics[market] = {"n_train": int(len(subset))}

    # Holdout figures for the same markets, measured AFTER the production
    # fits and without touching them: each fold refits the production
    # regressor on earlier seasons only and scores a season it never saw
    # (`evaluate/player_walk_forward.py`), so the manifest publishes an
    # honest out-of-sample error scale per market — the scale the prop
    # grader bands live picks by. A market no fold could score keeps its
    # bare `n_train` entry and stays unbanded: a missing number, never an
    # invented one.
    holdout_metrics = _yardage_holdout(player_train_df, player_feature_cols, len(seasons))
    for market, figures in holdout_metrics.items():
        if market in yardage_metrics:
            yardage_metrics[market] = {**yardage_metrics[market], **figures}

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(int(s) for s in seasons),
        "n_train": int(len(train_df)),
        "feature_cols": feature_cols,
        "player_feature_cols": player_feature_cols,
        "chosen_candidate": chosen,
        "candidate_scores": candidate_scores,
        "sigma": sigma,
        "total_sigma": total_sigma,
        **({"total_spec": _stored_total_spec(total_spec)} if total_spec else {}),
        "yardage_metrics": yardage_metrics,
        **({"qb_passing_td": qb_metric} if qb_metric else {}),
        "feature_blocks": list(blocks),
        "artifact_fingerprint": artifact_fingerprint(player_feature_cols),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def _stored_total_spec(spec: dict) -> dict:
    return {"kind": spec["kind"], "blocks": list(spec["blocks"]), "feature_cols": total_spec_columns(spec)}


def _yardage_holdout(player_train_df: pd.DataFrame, player_feature_cols: list[str], n_seasons: int) -> dict:
    """Walk-forward holdout figures for the yardage markets.

    `min_train_seasons` matches this module's game folds (`train_all` and
    `fit_total` both use `max(1, len(seasons) - 2)`): with the usual eight
    training seasons the measured holdout is the last two, each fitted on
    six-plus earlier seasons — the same evidentiary standard the game
    model's published sigma already stands on.
    """
    return player_walk_forward.yardage_holdout_metrics(
        player_train_df, player_feature_cols, min_train_seasons=max(1, n_seasons - 2)
    )


def publish_yardage_holdout_metrics() -> dict:
    """Measure the yardage holdout and patch ONLY `yardage_metrics` in the manifest.

    The measurement counterpart of `retrain_total_only`'s patch discipline,
    one step further: no model is refitted for shipping at all — the
    walk-forward fits exist only to be scored and are discarded — so every
    pickle, the `trained_at` stamp (and with it `model_version`) and every
    other manifest key are left exactly as committed. The data is loaded on
    the manifest's own seasons through the same calls `train_all` makes,
    and the frame gets `train_all`'s exact row handling, so the published
    figures describe the training universe the shipped models were fitted
    on, measured the way `train_all` itself now measures it.
    """
    manifest = load_manifest()
    seasons = manifest["seasons"]
    games_df = games_data.load_training_data(seasons)
    player_df_raw = player_stats.fetch_weekly_player_stats(seasons, games_df)
    player_train_df, player_feature_cols = player_usage.build_player_training_frame(player_df_raw)
    player_train_df = player_train_df.dropna(subset=player_feature_cols, how="all")
    holdout_metrics = _yardage_holdout(player_train_df, player_feature_cols, len(seasons))
    for market, figures in holdout_metrics.items():
        if market in manifest["yardage_metrics"]:
            manifest["yardage_metrics"][market] = {**manifest["yardage_metrics"][market], **figures}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def retrain_total_only() -> dict:
    """Refit ONLY the total model on the manifest's own seasons and patch the manifest; the game-outcome pickle, the
    sigma and every player model are left byte-for-byte as committed. `trained_at` moves so `model_version` changes and
    the snapshot job rebuilds every week instead of reusing weeks priced by the old total model."""
    manifest = load_manifest()
    seasons = manifest["seasons"]
    games_df = games_data.load_training_data(seasons)
    total_model, total_sigma = fit_total(
        games_df, _fbs_teams_by_season(seasons), seasons, max(1, len(seasons) - 2), TOTAL_SPEC)
    _save_pickle(total_model, _artifact_path(TOTAL_MODEL_FILENAME))
    manifest.update(total_sigma=total_sigma, total_spec=_stored_total_spec(TOTAL_SPEC),
                    trained_at=datetime.now(timezone.utc).isoformat())
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def artifact_fingerprint(player_feature_cols) -> dict:
    """What these committed pickles were fitted AGAINST, in a form the code can
    check itself against at load time.

    Two keys, and both are about meaning rather than shape:

    * `player_feature_cols` -- the exact list `train_all` fitted the player
      models on. Checked as an exact list, not a subset: a subset test cannot
      see an artefact that is missing a feature.
    * `anytime_td_label_version` from `player_usage`. Bump the version whenever
      the `anytime_td` definition changes and every artefact fitted before that
      bump stops loading loudly instead of quietly.

    The silent-degradation case this exists to close is measured, not
    hypothetical: on the artefact this branch started from,
    `models/manifest.json` listed 5 `player_feature_cols` while the code had 6
    (the 6th, `receptions_roll`, had been added to `ROLL_STATS` without a
    retrain), and `anytime_td_model.pkl` was fitted on 5. `load_models()`
    returned that payload without complaint and every anytime-TD prediction came
    out of a model that no longer matched the code.

    A subset check would have missed that, because the stale 5 columns ARE a
    subset of the 6 servable ones. The label half is the second reason: the
    label is never recomputed at serving -- `predict_props` scores the committed
    pickle from features alone -- so an artefact fitted against the OLD
    `anytime_td` definition serves exactly like one fitted against the new one.
    Right shape, wrong meaning, no error anywhere.
    """
    return {
        "player_feature_cols": list(player_feature_cols),
        "anytime_td_label_version": player_usage.ANYTIME_TD_LABEL_VERSION,
    }


def _verify_artifact_fingerprint(manifest: dict) -> None:
    """Raise when the committed artefacts disagree with the code they serve under.

    Called from `load_models`. A manifest written before this check existed has
    no `artifact_fingerprint` key at all, and that is treated as a FAILURE, not
    a pass -- the absence of a fingerprint is exactly the state this is
    detecting, so skipping the check when it is missing would defeat it.

    **Both halves are compared against the CODE, never against the manifest.**
    A check whose expectation comes from the artefact it is inspecting cannot
    detect that artefact drifting: NFL's first version of this built its
    expectation from `manifest["player_feature_cols"]`, so the manifest was
    compared with itself and passed for exactly the stale artefact it exists to
    catch. `player_usage.PLAYER_FEATURE_COLUMNS` is the authority instead,
    because it is the one thing training and serving both derive from:
    `train_all` fits the player models on the list `build_player_training_frame`
    returns (that constant), and `predict_props` reindexes a live row by
    `manifest["player_feature_cols"]`, which the check below pins to it.
    """
    recorded = manifest.get("artifact_fingerprint")
    if recorded is None:
        raise StaleArtifactError(
            f"{MANIFEST_PATH} has no 'artifact_fingerprint' key, so the committed "
            "models cannot be verified against the code that serves them. It was "
            "written before fingerprinting existed, and an unverified artefact is "
            "the case this check exists to catch. Re-run training "
            "(`python -m cfb_predictor.models.manifest`) to write one."
        )

    # The code's own fingerprint. Nothing here is read out of the manifest.
    current = artifact_fingerprint(player_usage.PLAYER_FEATURE_COLUMNS)
    expected_features = current["player_feature_cols"]
    fitted_features = list(recorded.get("player_feature_cols") or [])
    if fitted_features != expected_features:
        raise StaleArtifactError(
            f"{MANIFEST_PATH} records models fitted on {fitted_features} but the "
            f"code's player features are now {expected_features} "
            f"({player_usage.__name__}.PLAYER_FEATURE_COLUMNS). The committed "
            "pickles were fitted on a different feature set, so every prediction "
            "would come from a stale model. Re-run training "
            "(`python -m cfb_predictor.models.manifest`)."
        )

    # The fingerprint and the manifest's own `player_feature_cols` are written
    # from one value at fit time, so they must still agree -- and this is not
    # belt-and-braces. `load_models` hands `player_models["feature_cols"]` to
    # `player_props.predict_props`, which reindexes every live row by the
    # manifest's list, not the fingerprint's. A manifest whose top-level list had
    # drifted would score a 6-column model on a narrower feature set while the
    # check above stayed green: the same silent degradation from the other side.
    manifest_features = list(manifest.get("player_feature_cols") or [])
    if manifest_features != fitted_features:
        raise StaleArtifactError(
            f"{MANIFEST_PATH} records an artefact fingerprint fitted on "
            f"{fitted_features} but its own 'player_feature_cols' is "
            f"{manifest_features}. Serving reindexes every player row by the "
            "manifest's list, so the two disagreeing means the models would be "
            "scored on features they were not fitted on. Re-run training "
            "(`python -m cfb_predictor.models.manifest`)."
        )

    expected_version = current["anytime_td_label_version"]
    fitted_version = recorded.get("anytime_td_label_version")
    if fitted_version != expected_version:
        raise StaleArtifactError(
            f"{MANIFEST_PATH} records the anytime-TD model fitted against label "
            f"definition v{fitted_version}, but the code now defines "
            f"v{expected_version} ({player_usage.__name__}.ANYTIME_TD_LABEL_VERSION). "
            "The committed model predicts a different market than the one being "
            "graded: `anytime_td` is now rushing + receiving TDs only, excluding "
            "passing TDs. Serving it would score every prediction against a "
            "definition it was never fitted on. Re-run training "
            "(`python -m cfb_predictor.models.manifest`)."
        )


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No trained models found. Run `python -m cfb_predictor.models.manifest` first.")
    return json.loads(MANIFEST_PATH.read_text())


def model_version(manifest: dict) -> str:
    """Identifies the trained model behind a prediction: candidate + training time."""
    return f"{manifest['chosen_candidate']}@{manifest['trained_at']}"


def load_models() -> dict:
    manifest = load_manifest()
    # Before the unpickles: an artefact that cannot be verified must not reach
    # the point of being loaded, let alone scored against.
    _verify_artifact_fingerprint(manifest)
    total_spec = manifest.get("total_spec")
    if total_spec is not None and list(total_spec.get("feature_cols") or []) != total_spec_columns(total_spec):
        raise StaleArtifactError(
            f"{MANIFEST_PATH} records a total model fitted on {total_spec.get('feature_cols')} but the code's columns for "
            f"blocks {total_spec.get('blocks')} are {total_spec_columns(total_spec)}. Re-run training "
            "(`python -m cfb_predictor.models.manifest`).")
    player_models = {
        "feature_cols": manifest["player_feature_cols"],
        "anytime_td": _load_pickle(_artifact_path(ANYTIME_TD_MODEL_FILENAME)),
    }
    for market in manifest["yardage_metrics"]:
        player_models[market] = _load_pickle(_yardage_model_path(market))

    # Present only if a manifest declares it. An older manifest, or one written
    # when no QB had history, simply has no key -- which is why this is a `get`
    # and not an index: a deployment upgraded ahead of its first retrain must come
    # up and serve every OTHER market rather than 503 on an absent pickle.
    if manifest.get("qb_passing_td"):
        fitted = _load_pickle(_artifact_path(QB_PASSING_TD_MODEL_FILENAME))
        # The fit's columns are the SAME ones `load_models` hands to every other
        # player model, plus `passing_tds_roll`. Assert the relationship ONCE here,
        # for the whole payload, rather than per player at score time: a mismatch
        # means every QB would be scored from a constant-zero roll, and
        # `qb_passing_td.expected_passing_tds` raising per player turns that into
        # a season of one-log-line-per-QB exceptions instead of a failed deploy.
        served = set(player_models["feature_cols"]) | {player_usage.PASSING_TDS_ROLL_COLUMN}
        missing = [c for c in fitted.get("feature_cols", []) if c not in served]
        if missing:
            raise RuntimeError(
                f"the QB passing-TD model is fitted on {fitted.get('feature_cols')}, and "
                f"{missing} are not columns the served feature row carries. Scoring it anyway "
                f"would fill {missing} with 0.0 for every QB -- the fitted-a-feature-served-"
                f"as-constant-zero defect. Retrain, or serve a player_usage that emits them."
            )
        player_models[qb_passing_td.PASSING_TD_MARKET] = fitted

    return {
        "game_outcome_model": _load_pickle(_artifact_path(GAME_MODEL_FILENAME)),
        "total_model": _load_pickle(_artifact_path(TOTAL_MODEL_FILENAME)),
        "chosen_candidate": manifest["chosen_candidate"],
        "model_version": model_version(manifest),
        "sigma": manifest["sigma"],
        "total_sigma": manifest["total_sigma"],
        "total_spec": total_spec,
        "player_models": player_models,
        "feature_cols": manifest["feature_cols"],
        "player_feature_cols": manifest["player_feature_cols"],
        "feature_blocks": manifest.get("feature_blocks", DEFAULT_BLOCKS),
    }


if __name__ == "__main__":
    import sys

    if "--total-only" in sys.argv:
        retrain_total_only()
    elif "--yardage-holdout-only" in sys.argv:
        publish_yardage_holdout_metrics()
    else:
        train_all()
