# CFB in-season total drift (R1) — walk-forward result

**Date:** 2026-11-20
**Prereg:** `docs/2026-11-20-cfb-total-drift-prereg.md` (commit `efde347`)
**Verdict: fails.**

R1 (running in-season total bias) does not clear the prereg's ship rule. The pooled
improvement is +0.0146 points of MAE with a 95% interval that includes zero
[-0.0142, +0.0420], one held-out season (2018) is significantly worse, and the
total-bias gap widens from 0.5058 to 0.5610. Docs only, as the prereg specifies.

## What was run

- Data: `block_eval.load_recent_cfb_data(2014, 2025)`, read from the committed
  CFBD advanced-stats JSON and the local games cache. No API calls.
- Folds: `evaluate.walk_forward.prepare_folds` with the served total spec,
  `candidate=ridge`, `blocks=('adj_epa',)` (`models/manifest.TOTAL_SPEC`). Held-out
  seasons are 2016-2025; each season is predicted by a model trained only on
  earlier seasons.
- Predictions: `evaluate.walk_forward._fit_predict(..., target="total_points")`.
- R1: within each held-out season, for a game in week `w`,
  `delta = mean(pred_total - actual_total)` over that season's games in weeks
  `< w`, or 0 when fewer than 100 such games; `pred_r1 = pred_total - delta`.
- Primary interval: paired game bootstrap, 4000 draws, seed 20261017
  (`block_eval.paired_bootstrap`). Per-season intervals use the same bootstrap.
- Rows: `docs/2026-11-20-cfb-total-drift-rows.csv` (7,560 held-out games).

## Primary result

Pooled paired per-game `|base error| - |R1 error|` (positive means R1 better):

| metric | value |
|---|---|
| base MAE | 13.3982 |
| R1 MAE | 13.3836 |
| delta | +0.0146 |
| 95% CI | [-0.0142, +0.0420] |
| rule 1 (`low95 > 0`) | **FAIL** |

## Per-season deltas

Same paired delta and bootstrap, within each held-out season:

| season | n | base MAE | R1 MAE | delta | 95% CI | upper95 < 0 |
|---:|---:|---:|---:|---:|---|---|
| 2016 | 760 | 13.6953 | 13.5526 | +0.1427 | [-0.0234, +0.3092] | False |
| 2017 | 776 | 14.3879 | 14.3979 | -0.0099 | [-0.0285, +0.0091] | False |
| 2018 | 772 | 13.5751 | 13.6219 | -0.0467 | [-0.0894, -0.0057] | **True** |
| 2019 | 774 | 13.5044 | 13.5299 | -0.0256 | [-0.0594, +0.0077] | False |
| 2020 | 534 | 13.8705 | 13.9196 | -0.0491 | [-0.2064, +0.1181] | False |
| 2021 | 770 | 13.1163 | 13.0289 | +0.0874 | [-0.0391, +0.2137] | False |
| 2022 | 776 | 12.9493 | 12.9578 | -0.0084 | [-0.0352, +0.0168] | False |
| 2023 | 792 | 13.2787 | 13.3046 | -0.0259 | [-0.0914, +0.0427] | False |
| 2024 | 798 | 12.8895 | 12.8730 | +0.0165 | [-0.0315, +0.0636] | False |
| 2025 | 808 | 12.9042 | 12.8571 | +0.0472 | [-0.0409, +0.1350] | False |

Rule 2 (no held-out season has mean delta with upper95 < 0): **FAIL** (2018,
-0.0467 [-0.0894, -0.0057]).

## Total-bias gap

`tools/total_eval.bias_gap`, five prediction quintiles:

| metric | value |
|---|---|
| base | 0.5058 |
| R1 | 0.5610 |
| change (R1 - base) | +0.0552 |
| rule 3 (does not widen) | **FAIL** |

## Delta at season end

`final_week_delta` is the correction applied to games in the season's last week;
`full_season_delta` is the mean `pred_total - actual_total` over the whole season
(the value a next-season game would inherit if the rule did not reset).

| season | final week | final-week delta | n prior in final week | full-season delta |
|---:|---:|---:|---:|---:|
| 2016 | 15 | -3.1504 | 759 | -3.1046 |
| 2017 | 15 | -0.5617 | 775 | -0.5138 |
| 2018 | 15 | -0.6667 | 771 | -0.6198 |
| 2019 | 15 | -0.3738 | 763 | -0.3383 |
| 2020 | 16 | -1.7275 | 515 | -1.8288 |
| 2021 | 15 | +1.3744 | 769 | +1.4078 |
| 2022 | 15 | +0.1584 | 775 | +0.1785 |
| 2023 | 15 | +0.2358 | 791 | +0.2545 |
| 2024 | 16 | +0.2892 | 797 | +0.2964 |
| 2025 | 16 | +0.9667 | 807 | +0.9915 |

## Counts

- Total held-out games: 7,560.
- Seasons: 10 (2016-2025).
- Games with `delta != 0`: 6,229.
- Games with `delta == 0`: 1,331.

| season | games | first corrected week |
|---:|---:|---:|
| 2016 | 760 | 3 |
| 2017 | 776 | 3 |
| 2018 | 772 | 3 |
| 2019 | 774 | 3 |
| 2020 | 534 | 6 |
| 2021 | 770 | 3 |
| 2022 | 776 | 3 |
| 2023 | 792 | 3 |
| 2024 | 798 | 3 |
| 2025 | 808 | 3 |

## Descriptive: 2026 public snapshot

`data/public_snapshot.json` `weeks[*].predictions[game_id].predicted_total`
against `home_score + away_score`, R1 applied by week. 758 games over weeks 1-12.

| metric | value |
|---|---|
| base MAE | 13.2610 |
| R1 MAE | 13.4653 |
| delta | -0.2043 |
| 95% CI | [-0.3775, -0.0290] |

| week | n | applied delta | base MAE | R1 MAE |
|---:|---:|---:|---:|---:|
| 1 | 99 | +0.0000 | 13.3417 | 13.3417 |
| 2 | 86 | +0.0000 | 12.3264 | 12.3264 |
| 3 | 75 | -4.3789 | 13.2468 | 12.6763 |
| 4 | 71 | -4.4187 | 11.8746 | 12.5336 |
| 5 | 59 | -3.2792 | 13.8159 | 14.0083 |
| 6 | 58 | -2.8806 | 13.4733 | 14.2569 |
| 7 | 62 | -2.2575 | 12.8895 | 13.2477 |
| 8 | 56 | -1.7272 | 15.3928 | 15.4622 |
| 9 | 56 | -1.7619 | 12.4065 | 12.7324 |
| 10 | 63 | -1.4502 | 14.1638 | 14.2595 |
| 11 | 67 | -1.5359 | 13.7232 | 14.4338 |
| 12 | 6 | -0.9860 | 11.6723 | 11.0150 |

These snapshot predictions may be retrospective rescoring, not served pre-game
values. This cell is descriptive only.

## Descriptive: cached sportsbook closing totals

The cached sportsbook events contain full-match `POINT_TOTAL` markets for 84 of
the 758 played 2026 snapshot games (the cached slate is 2026-09-19, week 3). The
line used is the median across cached books; it is the last cached pre-game
snapshot, not a verified closing price. `total_sigma` is the served value carried
by the snapshot predictions (16.227224). `P(over) = 1 - Phi((line - total) / sigma)`.
Pushes are excluded; there were none among the matched games.

| metric | value |
|---|---|
| matched games | 84 |
| sigma | 16.227224 |
| actual over rate | 0.5595 |
| over/under log loss base | 0.7199 |
| over/under log loss R1 | 0.7062 |
| delta (base - R1) | +0.0137 |

The over/under cell uses the retrospective 2026 snapshot predictions, so it is
not independent evidence for R1 either.

## Ship rule

| rule | result |
|---|---|
| 1. pooled `low95 > 0` | **FAIL** ([-0.0142, +0.0420]) |
| 2. no season with `upper95 < 0` | **FAIL** (2018) |
| 3. total-bias gap does not widen | **FAIL** (0.5058 -> 0.5610) |

**Verdict: fails.** No serving change; docs only.
