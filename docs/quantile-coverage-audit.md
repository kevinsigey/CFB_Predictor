# Quantile coverage audit (CFB yardage props) -- research, nothing to audit

**Status: not measured. CFB has no quantile yardage models to audit.**

* `src/cfb_predictor` defines no `QUANTILES` and no quantile regressor. The only
  "quantile" hits are `np.quantile` bucket edges in `tools/block_eval.py` and
  `tools/total_eval.py`, which are calibration bucketing, not prediction intervals.
* `evaluate/player_walk_forward.py` runs the production yardage regressor
  (point model, one `q50`-style output, no interval) season by season. Its
  output carries no quantile columns, so coverage of an interval is undefined.
* `data/cfbd/` holds team-level advanced stats only (2014-2025 JSON), not player
  game logs, so the player feature frame cannot be rebuilt from the repo.
* Kalshi CFB lines (`data/lines/kalshi_cfb_*`) are game spread/total/moneyline
  contracts, not player props.

Nothing was measured. To audit intervals, first add quantile outputs to the CFB
player model and save out-of-sample rows with `actual` and q05..q95, then run
`python scripts/quantile_coverage_audit.py <csv> docs/quantile-coverage-audit`.

Files: `scripts/quantile_coverage_audit.py` (reusable, same as the NFL PR),
`tests/test_quantile_coverage_audit.py` (planted example), and
`docs/quantile-coverage-audit.{md,json}`. No model, route or UI changes.
