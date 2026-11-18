import type { GameSummary } from "../types";

/** The market's favourite and its probability, or null when no pre-kickoff price exists (never defaulted). */
export function marketFavourite(g: Pick<GameSummary, "home_team" | "away_team" | "market_home_win_probability">): { team: string; prob: number } | null {
  const m = g.market_home_win_probability;
  if (m == null) return null;
  return m >= 0.5 ? { team: g.home_team, prob: m } : { team: g.away_team, prob: 1 - m };
}
