export interface GameSummary {
  game_id: string;
  season: number;
  week: number;
  gameday: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  // Kalshi game-winner price (home win prob), pre-kickoff only; null when none was recorded.
  market_home_win_probability?: number | null;
}

export interface GamePrediction {
  home_win_prob: number;
  away_win_prob: number;
  home_cover_prob: number | null;
  away_cover_prob: number | null;
  over_prob: number | null;
  under_prob: number | null;
}

export interface PlayerPropPrediction {
  player_id: string;
  player_name: string;
  anytime_td_prob: number;
  passing_yards?: number;
  rushing_yards?: number;
  receiving_yards?: number;
}

export interface TrackRecord {
  n_resolved_games: number;
  pct_moneyline_correct: number | null;
}

export interface RetrainResponse {
  trained_at: string;
  chosen_candidate: string;
}

/** `GET /facts/{game_id}` — the backend's own bundle, not a re-derived one.
 *
 * Deliberately mirrors the server shape rather than picking the fields this
 * modal happens to draw. The pick rule — which number a STARTED game shows, and
 * whether it may honestly be called pre-kickoff — is decided in
 * `cfb_predictor/api/facts.py::game_pick` and must not be second-guessed here.
 * A frontend that recomputed it would be free to disagree with the record that
 * is about to be graded, which is the exact failure `pick_timing` exists to
 * prevent. */
export interface GameFacts {
  sport: string;
  id: string;
  title: string;
  starts_at: string | null;
  status: string;
  /** "pre_kickoff" | "rebuilt" | "none" — never inferred from `status`. */
  pick_timing: "pre_kickoff" | "rebuilt" | "none";
  pick: { label: string; prob: number } | null;
  markets: Array<Record<string, unknown>>;
  drivers: Array<Record<string, unknown>>;
  context: Record<string, unknown>;
  players: Array<Record<string, unknown>>;
  record: { hits: number; settled: number } | null;
  result: Record<string, unknown> | null;
}
