"""Binance USD-M futures public data (no API key) with a SQLite cache."""
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://fapi.binance.com"
DB_PATH = os.path.join(os.path.dirname(__file__), "state", "arena.sqlite")
TF_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
# 5m/15m only need the pre-holdout replay window (2025-09~2026-02) plus warmup.
HISTORY_START = {"5m": "2025-08-15", "15m": "2025-08-15", "1h": "2022-06-01", "4h": "2022-06-01"}
FUNDING_START = "2022-06-01"


def _get(path, **params):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    time.sleep(0.35)  # klines limit=1500 weighs 10; stay well under 2400/min
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if attempt == 5 or e.code not in (418, 429, 500, 502, 503, 504):
                raise
            time.sleep(int(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS bars (symbol TEXT, tf TEXT, t INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, PRIMARY KEY (symbol, tf, t))")
    con.execute("CREATE TABLE IF NOT EXISTS funding (symbol TEXT, t INTEGER, rate REAL, PRIMARY KEY (symbol, t))")
    return con


def _ms(date_str):
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def server_time():
    return int(_get("/fapi/v1/time")["serverTime"])


def update(symbols, tfs=("1h", "4h")):
    """Fetch closed klines and funding rates since the last cached row.

    A kline counts as closed only when its closeTime (r[6]) is before Binance
    server time, so a fast local clock can't ingest a forming bar.
    """
    con = _db()
    now = server_time()
    for sym in symbols:
        for tf in tfs:
            last = con.execute("SELECT MAX(t) FROM bars WHERE symbol=? AND tf=?", (sym, tf)).fetchone()[0]
            start = _ms(HISTORY_START[tf]) if last is None else last + TF_MS[tf]
            while start + TF_MS[tf] <= now:
                rows = _get("/fapi/v1/klines", symbol=sym, interval=tf, startTime=start, limit=1500)
                closed = [r for r in rows if r[6] < now]
                if not closed:
                    break
                con.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)",
                                [(sym, tf, r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in closed])
                start = closed[-1][0] + TF_MS[tf]
                if len(rows) < 1500:
                    break
        last = con.execute("SELECT MAX(t) FROM funding WHERE symbol=?", (sym,)).fetchone()[0]
        start = _ms(FUNDING_START) if last is None else last + 1
        while start < now:
            rows = _get("/fapi/v1/fundingRate", symbol=sym, startTime=start, limit=1000)
            if not rows:
                break
            # funding settles on the hour; round away the few-ms jitter
            con.executemany("INSERT OR REPLACE INTO funding VALUES (?,?,?)",
                            [(sym, round(r["fundingTime"] / 3_600_000) * 3_600_000, float(r["fundingRate"])) for r in rows])
            start = rows[-1]["fundingTime"] + 1
            if len(rows) < 1000:
                break
        con.commit()
    con.close()


def load(symbols, tfs=("1h", "4h")):
    """Return {(symbol, tf): DataFrame[t,o,h,l,c,v]} and {symbol: {t: rate}}."""
    con = _db()
    bars = {}
    for sym in symbols:
        for tf in tfs:
            bars[(sym, tf)] = pd.read_sql_query(
                "SELECT t,o,h,l,c,v FROM bars WHERE symbol=? AND tf=? ORDER BY t", con, params=(sym, tf))
    funding = {}
    for sym in symbols:
        rows = con.execute("SELECT t, rate FROM funding WHERE symbol=?", (sym,)).fetchall()
        funding[sym] = dict(rows)
    con.close()
    return bars, funding
