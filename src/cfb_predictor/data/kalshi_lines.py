"""Collect Kalshi's college football game, spread and total markets as a free, public record of what the market thought.

Kalshi lists three binary series per college football game: one market per team to win ("Miami (OH) wins"), a ladder of
spreads ("Dartmouth wins by over 6.5 points") and a ladder of totals ("Full Game: Over 66.5 points scored"). Every rung is
a binary contract whose price is the market's probability of that outcome. Reading them needs no account, key or credits.
Snapshots taken before kickoff are the "lines" we had no historical source for; the same markets settle afterwards with a
yes/no `result`, so each snapshot is its own labelled outcome and needs no box-score join to be scored.

Prices are dollars in [0, 1] (0.62 = 62%); the dollar-denominated quote fields (`yes_ask_dollars` and friends) are what
`normalize` reads, so no cent conversion is needed. A missing bid/ask means no quote, not a 0% market.

`floor_strike` is the real threshold on a spread or total: "over 6.5 points" has floor_strike 6.5. A moneyline market has
no floor_strike, so its `line` is None.

`occurrence_datetime` is the kickoff; `close_time` is when Kalshi finishes settling, which is later. The no-hindsight rule in
the paper ledger compares against occurrence, not close_time.

Volume is recorded as Kalshi reports it, including zero: most of these markets have never traded, and the ledger's own
liquidity filter is where that gets decided.
"""
from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
#: Kalshi series ticker -> the market it prices.
SERIES = {
    "KXNCAAFGAME": "moneyline",
    "KXNCAAFSPREAD": "spread",
    "KXNCAAFTOTAL": "total",
}
#: The stats that price a game rather than a player. Their titles are not "<player>: N+ <stat>",
#: so the player slot stays None -- a game market must never look like a prop.
GAME_LEVEL_STATS = ("moneyline", "spread", "total")
LINES_DIR = Path("data/lines")
PAGE_LIMIT = 1000
SETTLED_LOOKBACK_DAYS = 4


def _dollars(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v


def normalize(market: dict, series: str, snapshot_ts: str) -> dict:
    """One flat row per market rung. Pure; the unit of everything downstream."""
    title = market.get("title") or ""
    # Same rule as NFL's normalizer: derive the player from the title first, then drop it for the
    # game-level stats. A total's title ("Full Game: Over 66.5 points scored") has a colon, so the
    # split would otherwise hand the ledger a player named "Full Game".
    player = title.split(":", 1)[0].strip() if ":" in title else None
    if SERIES.get(series) in GAME_LEVEL_STATS:
        player = None
    return {
        "snapshot_ts": snapshot_ts,
        "series": series,
        "stat": SERIES.get(series),
        "ticker": market.get("ticker"),
        "event_ticker": market.get("event_ticker"),
        "player": player,
        "title": title,
        "line": _dollars(market.get("floor_strike")),
        "yes_bid": _dollars(market.get("yes_bid_dollars")),
        "yes_ask": _dollars(market.get("yes_ask_dollars")),
        "no_bid": _dollars(market.get("no_bid_dollars")),
        "no_ask": _dollars(market.get("no_ask_dollars")),
        "last": _dollars(market.get("last_price_dollars")),
        "volume": _dollars(market.get("volume_fp")),
        "open_interest": _dollars(market.get("open_interest_fp")),
        # Contracts resting at the best ask. Kalshi's NO ask is the YES bid seen
        # from the other side, so the NO ask's size is the YES bid size.
        "yes_ask_size": _dollars(market.get("yes_ask_size_fp")),
        "no_ask_size": _dollars(market.get("yes_bid_size_fp")),
        "status": market.get("status"),
        "result": market.get("result") or None,
        "settlement_value": market.get("expiration_value") or None,
        "occurrence": market.get("occurrence_datetime") or market.get("expected_expiration_time"),
        "close_time": market.get("close_time"),
    }


def _get(session, path: str, params: dict) -> dict:
    for attempt in range(4):
        r = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if r.status_code == 429:  # public API rate limit: back off, never hammer
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi rate-limited {path} after retries")


def fetch_series(session, series: str, status: str, min_close_ts: int | None = None) -> list[dict]:
    markets: list[dict] = []
    cursor = None
    while True:
        params = {"series_ticker": series, "status": status, "limit": PAGE_LIMIT}
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if cursor:
            params["cursor"] = cursor
        body = _get(session, "/markets", params)
        markets.extend(body.get("markets") or [])
        cursor = body.get("cursor")
        if not cursor:
            return markets


def collect(session=None, now: datetime | None = None) -> list[dict]:
    """Open markets (the pre-game lines) plus the last few days of settled ones (their outcomes)."""
    session = session or requests.Session()
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat(timespec="seconds")
    since = int((now - timedelta(days=SETTLED_LOOKBACK_DAYS)).timestamp())
    rows: list[dict] = []
    for series in SERIES:
        for status, min_close in (("open", None), ("settled", since)):
            for m in fetch_series(session, series, status, min_close):
                rows.append(normalize(m, series, ts))
    return rows


def write_snapshot(rows: list[dict], out_dir: Path = LINES_DIR, now: datetime | None = None) -> Path | None:
    """Append to today's gzip JSONL file (one file per UTC day, so the repo grows by one small file a day)."""
    if not rows:
        return None
    now = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"kalshi_cfb_lines_{now:%Y-%m-%d}.jsonl.gz"
    with gzip.open(path, "at", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------------------------------------------------
# The game-winner market as a win probability for one game, from the committed snapshots (never the live API).
# ---------------------------------------------------------------------------------------------------------------------
import re
import unicodedata
from functools import lru_cache
from zoneinfo import ZoneInfo

#: Same cut-off as NBA_Predictor's moneyline: a quote wider than this is not a price. Many CFB markets sit at 0.07 / 0.79.
MAX_SPREAD = 0.08
#: Kalshi's wording -> CFBD's, after `_norm`. The rest differ only by "St." vs "State", accents and punctuation.
_ALIAS = {"umass": "massachusetts", "louisianamonroe": "ulmonroe", "appalachianst": "appst", "miamifl": "miami"}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bstate\b", "st", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return _ALIAS.get(s, s)


def _mid(row: dict) -> float | None:
    bid, ask = row.get("yes_bid"), row.get("yes_ask")
    if not bid or not ask or ask < bid or ask - bid > MAX_SPREAD:
        return None
    return (bid + ask) / 2


@lru_cache(maxsize=4)
def _moneyline_snapshots(lines_dir: str, newest_mtime: float) -> dict:
    """{(ET date 'YYMMDD', event_ticker): [(snapshot_ts, {norm team name: yes mid})]}, oldest first.

    Only KXNCAAFGAME rows whose event ticker is a KXNCAAFGAME event count: the lines files have held other leagues' rows
    (NFL games share city names and team codes with CFB schools), and those must never reach a CFB game.
    """
    out: dict = {}
    for path in sorted(Path(lines_dir).glob("kalshi_cfb_lines_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if "KXNCAAFGAME" not in line:
                    continue
                r = json.loads(line)
                event, title = r.get("event_ticker") or "", r.get("title") or ""
                if r.get("series") != "KXNCAAFGAME" or not event.startswith("KXNCAAFGAME-") or not (r.get("ticker") or "").startswith(event + "-") or not title.endswith(" wins"):
                    continue
                snaps = out.setdefault((event.split("-")[1][:7], event), {})
                snaps.setdefault(r["snapshot_ts"], {})[_norm(title[: -len(" wins")])] = _mid(r)
    return {k: sorted(v.items()) for k, v in out.items()}


def game_win_probability(home: str, away: str, kickoff_utc: str, lines_dir: Path = LINES_DIR) -> float | None:
    """The market's home win probability from the latest snapshot taken BEFORE kickoff, or None (never defaulted).

    Matched by the ET game date in the KXNCAAFGAME event ticker plus the two team names (both must be in the event; an
    event that names only one team is matched on that one). Both sides quoted: each yes mid, normalised to sum to 1. One
    side quoted: that side's mid, the other as its complement. A spread wider than MAX_SPREAD is no quote.
    """
    kick = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
    if kick.tzinfo is None:
        kick = kick.replace(tzinfo=timezone.utc)
    date = kick.astimezone(ZoneInfo("America/New_York")).strftime("%y%b%d").upper()
    h, a = _norm(home), _norm(away)
    files = list(Path(lines_dir).glob("kalshi_cfb_lines_*.jsonl.gz"))
    index = _moneyline_snapshots(str(lines_dir), max((f.stat().st_mtime for f in files), default=0.0))
    for (d, _event), snaps in index.items():
        if d != date or not any(h in p or a in p for _, p in snaps):
            continue
        for ts, prices in reversed(snaps):
            if set(prices) - {h, a} or datetime.fromisoformat(ts) >= kick:
                continue
            ph, pa = prices.get(h), prices.get(a)
            if ph is not None and pa is not None:
                return round(ph / (ph + pa), 4)
            if ph is not None:
                return round(ph, 4)
            if pa is not None:
                return round(1 - pa, 4)
    return None


def main() -> int:
    rows = collect()
    path = write_snapshot(rows)
    open_rows = sum(1 for r in rows if r["status"] == "active")
    print(f"kalshi lines: {len(rows)} rows ({open_rows} open) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
