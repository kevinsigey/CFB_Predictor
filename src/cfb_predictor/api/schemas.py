"""schemas.py — Pydantic response models for the API. Verbatim port of
nfl_predictor/api/schemas.py — every field here is sport-agnostic."""

from __future__ import annotations

from pydantic import BaseModel


class GameSummary(BaseModel):
    game_id: str
    season: int
    week: int
    gameday: str
    home_team: str
    away_team: str
    home_score: int | None = None
    away_score: int | None = None
    # Kalshi game-winner price (home win prob), latest pre-kickoff snapshot; None when none was recorded.
    market_home_win_probability: float | None = None


class GamePrediction(BaseModel):
    home_win_prob: float
    away_win_prob: float
    home_cover_prob: float | None = None
    away_cover_prob: float | None = None
    over_prob: float | None = None
    under_prob: float | None = None


class TrackRecord(BaseModel):
    n_resolved_games: int
    pct_moneyline_correct: float | None = None


class RetrainResponse(BaseModel):
    trained_at: str
    chosen_candidate: str
