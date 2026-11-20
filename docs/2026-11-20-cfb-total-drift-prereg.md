# CFB: in-season total drift correction, preregistration (2026-11-20)

**Why.**
- In the paper ledger, CFB total score rows (64 settled) put the over at 47% on average, but the over hit 28%.
  Log loss is 0.061 worse than the Kalshi mid [-0.114, -0.009].
- I backed the served predicted totals out of the ledger's rung probabilities. Over 125 games in weeks 8-12, they
  run 4.2 points above the actual total on average (sd 14.6).
- The model learns its scoring level from earlier seasons. No number for the candidate below has been computed.

**Prior test.** `docs/cfb_league_residual_correction.md` (2026-10-22) tried a trailing 300-game league residual
correction, capped at ±4, on 2026 weeks 4-7 (250 games). It made totals worse, -0.399 [-0.762, -0.033], when the
bias had the opposite sign (-2.26). That doc asked for a re-run at week 12. This test differs in three ways: it
uses all of the season's earlier weeks with no cap, it runs on 10+ held-out past seasons rather than one
snapshot, and its sign is free. The sign flip within 2026 is itself a warning that in-season drift may be noise.

## Candidate (one, fixed)

**R1, running in-season total bias.** Within each season, for a game in week w:

    delta = mean(predicted_total - actual_total) over that season's completed games in weeks < w
            (0 if fewer than 100 such games)
    total' = predicted_total - delta

P(over line) = 1 - Phi((line - total') / total_sigma). total_sigma is the same one the base uses.

## Data: out-of-sample predictions

Use the repo's season walk-forward (`evaluate/walk_forward.py`: `prepare_folds`, `_fit_predict`) with the SERVED
total candidate and features, as in the served model and `tools/total_eval.py`. Every held-out season from the
folds is predicted by a model trained on earlier seasons only. The 2026 served totals are reported separately and
are descriptive only.

## Ship rule (all must hold)

1. **Primary:** pooled over held-out seasons, the paired per-game delta = |base error| - |R1 error|. Game-level
   bootstrap, 4000 draws, seed 20261017. low95 > 0.
2. **Nothing else worse:**
   - no single held-out season has mean delta with upper95 < 0;
   - the total-bias gap from `tools/total_eval.bias_gap` does not widen.

**If it passes:** a follow-up PR applies R1 to served totals behind a flag, default off, storing the raw total
beside it. Switching it on needs Kevin's go.
**If it fails:** docs only.

## Reported, not gating

- delta at season end for each held-out season.
- Per-season MAE deltas.
- Where the repo has sportsbook closing totals (`data/cache/sportsbook` or `odds`): over/under log loss of base vs R1
  against the closing line, as a descriptive cell.
- The 2026 served descriptive.
