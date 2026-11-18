"""/games carries the Kalshi game-winner price as `market_home_win_probability`: the latest snapshot taken BEFORE kickoff,
matched by ET game date + both CFB team names, None when there is none (never defaulted). NFL rows never match a CFB game."""
import gzip
import json

import pandas as pd

from cfb_predictor.api import routes
from cfb_predictor.data import games as games_data, kalshi_lines as kl

# Cincinnati @ Washington is a CFB game name pair AND an NFL one (CIN @ WAS). Kickoff 20:00Z Sat 2026-10-10 = 4pm ET.
KICK = "2026-10-10 20:00:00+00:00"


def _row(ts, side, bid, ask, event="KXNCAAFGAME-26OCT10CINWASH", series="KXNCAAFGAME", name=None):
    return {"series": series, "snapshot_ts": ts, "ticker": f"{event}-{side}", "event_ticker": event,
            "title": f"{name or side} wins", "yes_bid": bid, "yes_ask": ask, "status": "active"}


def _setup(monkeypatch, tmp_path, rows):
    d = tmp_path / "lines"
    d.mkdir(exist_ok=True)
    with gzip.open(d / "kalshi_cfb_lines_2026-10-10.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    frame = pd.DataFrame([{"game_id": "g1", "season": 2025, "week": 6, "gameday": pd.Timestamp(KICK), "home_team": "Washington",
                           "away_team": "Cincinnati", "home_score": None, "away_score": None}])
    monkeypatch.setattr(games_data, "fetch_week_games", lambda s, w: frame)
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)
    kl._moneyline_snapshots.cache_clear()
    return lambda: routes._get_games_live(2025, 6)[0]["market_home_win_probability"]


def test_latest_pre_kickoff_price_is_used_and_post_kickoff_ignored(monkeypatch, tmp_path):
    get = _setup(monkeypatch, tmp_path, [
        _row("2026-10-09T12:00:00+00:00", "WASH", 0.28, 0.32, name="Washington"), _row("2026-10-09T12:00:00+00:00", "CIN", 0.68, 0.72, name="Cincinnati"),
        _row("2026-10-10T19:00:00+00:00", "WASH", 0.38, 0.42, name="Washington"), _row("2026-10-10T19:00:00+00:00", "CIN", 0.58, 0.62, name="Cincinnati"),
        _row("2026-10-10T21:00:00+00:00", "WASH", 0.98, 1.0, name="Washington"), _row("2026-10-10T21:00:00+00:00", "CIN", 0.0, 0.02, name="Cincinnati"),
    ])
    assert get() == 0.4   # 0.40 / (0.40 + 0.60) from the 19:00Z snapshot; the post-kickoff 21:00Z one is ignored


def test_no_rows_or_only_post_kickoff_rows_is_none(monkeypatch, tmp_path):
    assert _setup(monkeypatch, tmp_path, [])() is None
    assert _setup(monkeypatch, tmp_path, [_row("2026-10-10T21:00:00+00:00", "WASH", 0.9, 0.92, name="Washington")])() is None


def test_a_wide_quote_is_no_price(monkeypatch, tmp_path):
    assert _setup(monkeypatch, tmp_path, [_row("2026-10-09T12:00:00+00:00", "WASH", 0.07, 0.79, name="Washington")])() is None


def test_one_priced_side_uses_its_mid(monkeypatch, tmp_path):
    assert _setup(monkeypatch, tmp_path, [_row("2026-10-09T12:00:00+00:00", "CIN", 0.68, 0.72, name="Cincinnati")])() == 0.3


def test_an_nfl_game_never_matches_a_cfb_game(monkeypatch, tmp_path):
    """NFL CIN @ WAS on the same date, same city names and codes: wrong series or a non-CFB event ticker is ignored."""
    nfl = "KXNFLGAME-26OCT10CINWAS"
    get = _setup(monkeypatch, tmp_path, [
        _row("2026-10-09T12:00:00+00:00", "WAS", 0.9, 0.92, event=nfl, series="KXNFLGAME", name="Washington"),
        _row("2026-10-09T12:00:00+00:00", "CIN", 0.07, 0.09, event=nfl, series="KXNFLGAME", name="Cincinnati"),
        # an NFL ticker mislabelled with the CFB series in the lines file, as the ~3,400 stray rows were
        _row("2026-10-09T12:00:00+00:00", "WAS", 0.9, 0.92, event=nfl, series="KXNCAAFGAME", name="Washington"),
    ])
    assert get() is None


def test_kalshi_wording_maps_to_cfbd_names():
    assert [kl._norm(n) for n in ("Appalachian St.", "Miami (FL)", "San Jose St.", "Louisiana-Monroe", "UMass")] == \
           [kl._norm(n) for n in ("App State", "Miami", "San José State", "UL Monroe", "Massachusetts")]
