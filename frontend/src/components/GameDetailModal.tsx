/**
 * The fixture modal: CFB's facts block, and the signal rows under it.
 *
 * CFB had no fixture modal at all — `GamesPage` listed a title and a percentage
 * — so this is the surface spec `2026-10-01-fixture-signals-design.md` §6 means
 * by "under the facts block in each modal". It is also the only place the trust
 * badge can appear for CFB, which matters because the moneyline is the one
 * market in the fleet measured good enough to ship one (390 graded rows, 3 of 4
 * bands over the floor; `docs/superpowers/plans/2026-10-04-production-measurements.md`).
 *
 * ## Two rules this component exists to keep.
 *
 * **The pick is quoted, never recomputed.** Everything about which number a game
 * shows comes from `GET /facts/{game_id}`, whose `pick_timing` says whether it may
 * honestly be called a pre-kickoff call. A modal that derived its own pick would
 * be free to disagree with the record that is about to be graded.
 *
 * **A signal is an enhancement, never the page.** `signals` is fetched beside
 * the facts, not instead of them: an empty list renders nothing at all (spec §2,
 * "no data, no row"), and a failed request leaves the facts untouched rather
 * than becoming the modal's error state. The same rule is kept on the adapter
 * and the endpoint's side; pinning it here as well is cheap and it is the side a
 * reader actually sees.
 */
import { useEffect, useState } from "react";

import { SignalRows } from "../predictor-ui";
import type { Signal } from "../predictor-ui";
import { api } from "../api/client";
import type { GameFacts } from "../types";

interface Props {
  gameId: string;
  /** The market's side and price for this game from the list row; null/absent when no pre-kickoff price exists. */
  market?: { team: string; prob: number } | null;
  onClose: () => void;
}

/** What the pick is allowed to call itself. `rebuilt` is the one that must never
 *  be hidden: it is a post-hoc number, and the record strip that judges it is not
 *  counting it. */
const TIMING_WORDS: Record<GameFacts["pick_timing"], string> = {
  pre_kickoff: "Recorded before kickoff",
  rebuilt: "Rebuilt after kickoff — not counted in the record",
  none: "No pick recorded",
};

export function GameDetailModal({ gameId, market, onClose }: Props) {
  const [facts, setFacts] = useState<GameFacts | null>(null);
  const [failed, setFailed] = useState(false);
  const [signals, setSignals] = useState<Signal[]>([]);

  useEffect(() => {
    let cancelled = false;
    setFacts(null);
    setFailed(false);
    api
      .facts(gameId)
      .then((f) => !cancelled && setFacts(f))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [gameId]);

  useEffect(() => {
    let cancelled = false;
    api
      .signals(gameId)
      .then((res) => !cancelled && setSignals(res.signals ?? []))
      // No row, and no error: the facts below are the page, and a signal is an
      // addition to it. Swallowing this is what keeps a sub-floor or absent
      // signal from taking a fixture page down with it.
      .catch(() => !cancelled && setSignals([]));
    return () => {
      cancelled = true;
    };
  }, [gameId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={facts?.title ?? "Fixture"}
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 p-4"
    >
      <div className="mt-8 w-full max-w-2xl rounded-lg bg-pr-panel p-5 text-pr-text">
        <div className="mb-4 flex items-start justify-between gap-4">
          <h2 className="text-xl font-bold">{facts?.title ?? "Loading fixture…"}</h2>
          <button type="button" onClick={onClose} aria-label="Close" className="text-sm text-pr-text-dim">
            Close
          </button>
        </div>

        {failed && (
          <p role="alert" className="text-sm">
            We couldn&apos;t load this fixture.
          </p>
        )}

        {facts && (
          <>
            {/* The pick, and the only sentence about it. The probability is the
                backend's; `pick_timing` is what makes it honest to show. */}
            {market && (
              <div className="mb-4" data-testid="market-headline">
                <div className="text-xs uppercase tracking-wide text-pr-text-dim">Market</div>
                <div className="text-3xl font-semibold">{market.team} {Math.round(market.prob * 100)}%</div>
              </div>
            )}
            <div className="mb-4">
              <p className="text-sm text-pr-text-dim">{TIMING_WORDS[facts.pick_timing]}</p>
              {facts.pick && (
                <p className="mt-1 text-lg">
                  <span className="font-semibold">{facts.pick.label}</span>{" "}
                  {`${Math.round(facts.pick.prob * 100)}%`}
                </p>
              )}
              {/* The market leads when a pre-kickoff price exists; the model is the line above, so it is named, not repeated. */}
              {facts.pick && !market && <p className="text-xs text-pr-text-dim">Model · no market line</p>}
              {facts.pick && market && <p className="text-xs text-pr-text-dim">Model</p>}
            </div>

            {facts.record && facts.record.settled > 0 && (
              <p className="mb-4 text-sm text-pr-text-dim">
                {`${facts.record.hits} of ${facts.record.settled} pre-kickoff picks settled correctly`}
              </p>
            )}

            {/* Spec §6: compact rows under the facts block, never paragraphs.
                `SignalRows` renders nothing for an empty list, and this guard
                stops the wrapper leaving an empty margin behind. */}
            {signals.length > 0 && (
              <div className="mb-4">
                <SignalRows signals={signals} />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}