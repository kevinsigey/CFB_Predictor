"""Tests for the Kalshi college-football line collector.

The row shape is not ours to invent: the paper ledger in predictor-hub reads these
rows and expects NFL's/PL's field names, so a renamed field is a silent break.
These tests pin the shape, the game-market rules and the paging.

The fixtures below are verbatim market payloads from the live Kalshi API
(api.elections.kalshi.com/trade-api/v2/markets), captured for the three NCAAF
series on 2026-10-21.
"""
import gzip
import json
from datetime import datetime, timezone

from cfb_predictor.data import kalshi_lines as kl

GAME = {
    "ticker": "KXNCAAFGAME-26NOV03BUFFMOH-MOH", "event_ticker": "KXNCAAFGAME-26NOV03BUFFMOH",
    "title": "Miami (OH) wins", "floor_strike": None, "status": "active", "result": "",
    "yes_bid_dollars": "0.5300", "yes_ask_dollars": "0.7300",
    "no_bid_dollars": "0.2700", "no_ask_dollars": "0.4700",
    "last_price_dollars": "0.0000", "volume_fp": "0.00", "open_interest_fp": "0.00",
    "occurrence_datetime": "2026-11-04T03:00:00Z", "close_time": "2026-11-06T00:00:00Z",
    "expiration_value": "",
}

SPREAD = {
    "ticker": "KXNCAAFSPREAD-26OCT24DARTCLMB-DART7", "event_ticker": "KXNCAAFSPREAD-26OCT24DARTCLMB",
    "title": "Dartmouth wins by over 6.5 points", "floor_strike": 6.5, "status": "active", "result": "",
    "yes_bid_dollars": "0.0300", "yes_ask_dollars": "0.9700",
    "no_bid_dollars": "0.0300", "no_ask_dollars": "0.9700",
    "last_price_dollars": "0.0000", "volume_fp": "0.00", "open_interest_fp": "0.00",
    "occurrence_datetime": "2026-10-24T20:30:00Z", "close_time": "2026-10-26T17:30:00Z",
    "expiration_value": "",
}

TOTAL = {
    "ticker": "KXNCAAFTOTAL-26OCT24PVETAM-67", "event_ticker": "KXNCAAFTOTAL-26OCT24PVETAM",
    "title": "Full Game: Over 66.5 points scored", "floor_strike": 66.5, "status": "active", "result": "",
    "yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.9600",
    "no_bid_dollars": "0.0400", "no_ask_dollars": "1.0000",
    "last_price_dollars": "0.0000", "volume_fp": "0.00", "open_interest_fp": "0.00",
    "occurrence_datetime": "2026-10-25T02:00:00Z", "close_time": "2026-10-26T23:00:00Z",
    "expiration_value": "",
}


def test_normalize_keeps_the_quotes_as_probabilities_and_occurrence_as_kickoff():
    row = kl.normalize(GAME, "KXNCAAFGAME", "2026-10-21T12:00:00+00:00")
    assert row["stat"] == "moneyline"
    assert (row["yes_bid"], row["yes_ask"]) == (0.53, 0.73)
    assert (row["no_bid"], row["no_ask"]) == (0.27, 0.47)
    assert row["occurrence"] == "2026-11-04T03:00:00Z"       # kickoff
    assert row["close_time"] == "2026-11-06T00:00:00Z"       # Kalshi settles later
    assert row["result"] is None and row["status"] == "active"
    assert row["volume"] == 0.0 and row["open_interest"] == 0.0


def test_a_settled_market_carries_its_outcome():
    row = kl.normalize({**GAME, "status": "finalized", "result": "yes",
                        "expiration_value": "Miami (OH)"}, "KXNCAAFGAME", "t")
    assert row["result"] == "yes" and row["settlement_value"] == "Miami (OH)"


def test_missing_prices_are_none_not_zero():
    row = kl.normalize({"title": "Miami (OH) wins", "floor_strike": None}, "KXNCAAFGAME", "t")
    assert row["yes_bid"] is None and row["yes_ask"] is None
    assert row["no_bid"] is None and row["no_ask"] is None
    assert row["last"] is None and row["volume"] is None


def test_player_is_none_for_every_cfb_series():
    """A game market is never a prop. The total's title ("Full Game: Over 66.5 points
    scored") has a colon, so a naive title split would name "Full Game" a player."""
    for series, market in (("KXNCAAFGAME", GAME), ("KXNCAAFSPREAD", SPREAD), ("KXNCAAFTOTAL", TOTAL)):
        assert kl.normalize(market, series, "t")["player"] is None


def test_a_spread_rung_and_a_total_rung_carry_their_real_line():
    spread = kl.normalize(SPREAD, "KXNCAAFSPREAD", "t")
    total = kl.normalize(TOTAL, "KXNCAAFTOTAL", "t")
    assert spread["stat"] == "spread" and spread["line"] == 6.5
    assert total["stat"] == "total" and total["line"] == 66.5
    assert kl.normalize(GAME, "KXNCAAFGAME", "t")["line"] is None       # a moneyline has no line


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b

    def raise_for_status(self):
        assert self.status_code < 400


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        if params.get("cursor") == "p2":
            return _Resp({"markets": [GAME], "cursor": ""})
        paginated = params["series_ticker"] == "KXNCAAFGAME" and params["status"] == "open"
        return _Resp({"markets": [GAME], "cursor": "p2" if paginated else ""})


def test_collect_pages_through_cursors_and_asks_for_open_and_recently_settled():
    s = _Session()
    rows = kl.collect(s, now=datetime(2026, 10, 21, 12, tzinfo=timezone.utc))
    statuses = {(c["series_ticker"], c["status"]) for c in s.calls}
    for series in kl.SERIES:
        assert (series, "open") in statuses and (series, "settled") in statuses
    assert any(c.get("cursor") == "p2" for c in s.calls)
    assert len(rows) == len(kl.SERIES) * 2 + 1        # each series x (open, settled) + the extra page
    assert all(r["snapshot_ts"].startswith("2026-10-21") for r in rows)


def test_write_snapshot_appends_one_gzip_file_per_utc_day(tmp_path):
    rows = [kl.normalize(GAME, "KXNCAAFGAME", "t")]
    now = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
    p1 = kl.write_snapshot(rows, tmp_path, now)
    p2 = kl.write_snapshot(rows, tmp_path, now)
    assert p1 == p2 and p1.name == "kalshi_cfb_lines_2026-10-21.jsonl.gz"
    lines = gzip.open(p1, "rt").read().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["ticker"] == GAME["ticker"]
    assert kl.write_snapshot([], tmp_path, now) is None


def test_normalize_records_the_size_resting_at_each_sides_ask():
    # Kalshi's public market list carries yes_ask_size_fp and yes_bid_size_fp;
    # NO's ask is the YES bid from the other side, so it rests on the YES bid size.
    market = {"title": "A B: 5+ receptions", "floor_strike": 4.5,
              "yes_ask_size_fp": "774.30", "yes_bid_size_fp": "1.00"}
    row = kl.normalize(market, "KXTEST", "2026-11-18T00:00:00+00:00")
    assert (row["yes_ask_size"], row["no_ask_size"]) == (774.3, 1.0)
    assert kl.normalize({"title": "A B: 5+ receptions"}, "KXTEST", "t")["yes_ask_size"] is None
