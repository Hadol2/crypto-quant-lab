"""Read-only arena dashboard: one HTML page + JSON API over arena.sqlite.

    python3 -m arena.dashboard --host 127.0.0.1 --port 8080
"""
import argparse
import json
import os
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from arena import data

START_EQUITY = 10_000.0
HTML = os.path.join(os.path.dirname(__file__), "dashboard.html")
CACHE_S = 20
_cache, _lock = {}, threading.Lock()


def ro():
    con = sqlite3.connect(f"file:{data.DB_PATH}?mode=ro", uri=True, timeout=10)
    con.execute("PRAGMA query_only=1")
    return con


def runs(con):
    return [dict(run_id=r[0], started_ms=r[1], end_ms=r[2], cycle_ms=r[3], n_bots=r[4], status=r[5])
            for r in con.execute("SELECT run_id, started_ms, end_ms, cycle_ms, n_bots, status FROM arena_runs "
                                 "ORDER BY started_ms DESC")]


def has_ticks(con):
    return con.execute("SELECT 1 FROM sqlite_master WHERE name='arena_ticks'").fetchone() is not None


def state(run_id=None):
    con = ro()
    try:
        allruns = runs(con)
        if not allruns:
            return {"runs": []}
        run = next((r for r in allruns if r["run_id"] == run_id), allruns[0])
        rid = run["run_id"]

        bots = {}
        for b, p, gen, par, s, ret_ms, why in con.execute(
                "SELECT bot_id, params, gen, parent, start_ms, retired_ms, retire_reason FROM arena_bots WHERE run_id=?", (rid,)):
            p = json.loads(p)
            bots[b] = dict(bot=b, family=p["family"], tf=p["tf"], gen=gen, parent=par, start_ms=s, retired_ms=ret_ms,
                           reason=why, params=" ".join(f"{k}={v}" for k, v in p.items() if k not in ("family", "tf")))

        # lifetime score, same formula as run_live.lifetime (L = sum(score) / (n + 2))
        scores = defaultdict(list)
        for c, b, ret, mdd, tr, sc, eq in con.execute(
                "SELECT cycle, bot_id, ret, mdd, trades, score, equity FROM arena_scores WHERE run_id=? ORDER BY cycle", (rid,)):
            scores[b].append((c, ret, mdd, tr, sc, eq))
        for b, rows in scores.items():
            if b in bots:
                bots[b].update(cycles=len(rows), L=sum(r[4] for r in rows) / (len(rows) + 2.0),
                               last_cycle_equity=rows[-1][5])

        # equity ticks
        series, latest_t = [], None
        fam_series = defaultdict(list)
        if has_ticks(con):
            by_t = defaultdict(list)
            for t, b, eq, tr, op, lq in con.execute(
                    "SELECT t, bot_id, equity, total_trades, open, liqs FROM arena_ticks WHERE run_id=? ORDER BY t", (rid,)):
                by_t[t].append((b, eq, tr, op, lq))
            fams = sorted({v["family"] for v in bots.values()})
            for t, rows in by_t.items():
                e = np.array([r[1] for r in rows])
                series.append(dict(t=t, n=len(rows), mean=e.mean(), median=float(np.median(e)),
                                   p10=float(np.percentile(e, 10)), p90=float(np.percentile(e, 90)),
                                   min=e.min(), max=e.max(), up=int((e > START_EQUITY).sum()),
                                   down=int((e < START_EQUITY).sum()), open=int(sum(r[3] for r in rows))))
                per = defaultdict(list)
                for r in rows:
                    per[bots.get(r[0], {}).get("family")].append(r[1])
                for f in fams:
                    v = per.get(f)
                    fam_series[f].append(float(np.mean(v)) if v else None)
            if by_t:
                latest_t = max(by_t)
                for b, eq, tr, op, lq in by_t[latest_t]:
                    if b in bots:
                        bots[b].update(equity=eq, trades=tr, open=op, liqs=lq, live=True)

        for v in bots.values():
            v.setdefault("live", False)
            if not v["live"] and v.get("retired_ms") and "last_cycle_equity" in v:
                v["equity"] = v["last_cycle_equity"]
            if "equity" in v:
                v["ret"] = v["equity"] / START_EQUITY - 1

        # per cycle
        cyc = defaultdict(lambda: dict(n=0, rets=[], trades=0.0, liqs=0))
        for c, b, ret, mdd, tr, sc, eq in ((row[0], b, *row[1:]) for b, rows in scores.items() for row in rows):
            d = cyc[c]
            d["n"] += 1
            d["rets"].append(ret)
            d["trades"] += tr
        cycles = []
        for c in sorted(cyc):
            d = cyc[c]
            c1 = min(run["started_ms"] + (c + 1) * run["cycle_ms"], run["end_ms"])
            retired = Counter(v["reason"] for v in bots.values() if v["retired_ms"] == c1)
            added = Counter(("mutant" if v["parent"] else "random") for v in bots.values()
                            if v["start_ms"] == c1 and v["gen"] == c + 1)
            r = np.array(d["rets"])
            cycles.append(dict(cycle=c, t0=c1 - run["cycle_ms"], t1=c1, n=d["n"], mean=float(r.mean()),
                               median=float(np.median(r)), best=float(r.max()), worst=float(r.min()),
                               pos=int((r > 0).sum()), trades=int(d["trades"]), retired=dict(retired), added=dict(added)))

        trials = [dict(cycle=c, tried=n, passed=int(p or 0)) for c, n, p in con.execute(
            "SELECT cycle, COUNT(*), SUM(passed) FROM arena_trials WHERE run_id=? GROUP BY cycle ORDER BY cycle", (rid,))]

        live = [v for v in bots.values() if v["live"]]
        grid = defaultdict(list)
        for v in live:
            grid[(v["family"], v["tf"])].append(v)
        famtf = [dict(family=f, tf=tf, n=len(vs), mean=float(np.mean([x["ret"] for x in vs])),
                      median=float(np.median([x["ret"] for x in vs])), best=float(max(x["ret"] for x in vs)),
                      trades=int(sum(x["trades"] for x in vs)), open=int(sum(x["open"] for x in vs)))
                 for (f, tf), vs in sorted(grid.items())]

        bars_to = con.execute("SELECT MAX(t) FROM bars WHERE tf='5m'").fetchone()[0]
        return dict(runs=allruns, run=run, now_ms=int(time.time() * 1000), latest_t=latest_t,
                    bars_to=(bars_to + 300_000) if bars_to else None, series=series,
                    fam_series=dict(fam_series), bots=list(bots.values()), cycles=cycles, trials=trials, famtf=famtf)
    finally:
        con.close()


def bot_series(run_id, bot_id):
    con = ro()
    try:
        return [dict(t=t, equity=e, trades=tr, open=op) for t, e, tr, op in con.execute(
            "SELECT t, equity, total_trades, open FROM arena_ticks WHERE run_id=? AND bot_id=? ORDER BY t", (run_id, bot_id))]
    finally:
        con.close()


def cached(key, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_S:
            return hit[1]
    body = json.dumps(fn(), default=float).encode()
    with _lock:
        _cache[key] = (time.time(), body)
    return body


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                with open(HTML, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif u.path == "/api/state":
                self._send(200, cached(("state", q.get("run")), lambda: state(q.get("run"))), "application/json")
            elif u.path == "/api/bot":
                self._send(200, cached(("bot", q.get("run"), q.get("id")), lambda: bot_series(q.get("run"), q.get("id"))),
                           "application/json")
            elif u.path == "/healthz":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:
            self._send(500, json.dumps({"error": repr(e)}).encode(), "application/json")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    print(f"arena dashboard on http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
