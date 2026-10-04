"""Live paper arena on Binance USD-M closed bars, evolving every N hours.

    PYTHONPATH=~/Desktop/big nohup caffeinate -dimsu python3 -m arena.run_live --hours 24 &

Live evaluation is a replay of closed bars since a fixed anchor, so a live
result is identical to a batch replay of the same span, and missed ticks
(sleep, crash) are recovered by backfilling klines on restart.
Selection rule: arena/state/selection_rule.md (written before cycle 1).
"""
import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from arena import data, engine, strategies as st
from results import ledger

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"]
TFS = st.TFS
WARMUP_BARS = 1500
_ms = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
GATE_FROM, GATE = _ms("2025-08-15"), (_ms("2025-09-01"), engine.HOLDOUT_MS[0])
STATE = os.path.join(os.path.dirname(__file__), "state")
REPORTS = os.path.join(STATE, "reports")
TICK_MS = 300_000  # dashboard equity snapshots (output only)

# pre-registered selection rule (also written to selection_rule.md)
GATE_MIN_TRADES = 20
CULL_MIN_CYCLES = 2
CULL_FRACTION = 0.30
PARENTS = 20
PARENT_MIN_TRADES = 3
MUTANT_SHARE = 0.7
MDD_PENALTY = 2.0
SHRINK = 2.0

RULE = f"""# Arena selection rule (pre-registered)

- Replay gate (window {pd.Timestamp(GATE[0], unit='ms', tz='UTC').date()} ~ {pd.Timestamp(GATE[1], unit='ms', tz='UTC').date()}, before the holdout): trades >= {GATE_MIN_TRADES}, liquidations == 0, no drawdown kill, Sharpe > 0. Every new bot must pass it.
- Gen 0: bots that pass the gate, ranked by replay Sharpe, with a timeframe quota of about 1/3 each (if a timeframe has too few, fill from the rest).
- Cycle score = cycle return - {MDD_PENALTY} x cycle MDD.
- Lifetime score = sum(cycle score) / (cycles lived + {SHRINK}), which shrinks it toward zero.
- Cull at the end of each cycle:
  - Drawdown-kill or liquidation: retired immediately.
  - Otherwise, among bots that have lived >= {CULL_MIN_CYCLES} cycles, those in the bottom {int(CULL_FRACTION*100)}% by lifetime score AND with a score < 0 are retired (idle bots with score 0 are not culled for being idle).
- Refill: {int(MUTANT_SHARE*100)}% are mutants of the lifetime-score top {PARENTS} (live cumulative trades >= {PARENT_MIN_TRADES}), the rest are random. Both must pass the replay gate. If no bot qualifies as a parent, the refill is 100% random.
- A single 5h window never decides survival.
- No bot from a 1-3 day run qualifies for real money.
"""


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def db():
    con = sqlite3.connect(data.DB_PATH)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS arena_runs (run_id TEXT PRIMARY KEY, started_ms INTEGER, end_ms INTEGER,
        cycle_ms INTEGER, n_bots INTEGER, anchors TEXT, status TEXT);
    CREATE TABLE IF NOT EXISTS arena_bots (run_id TEXT, bot_id TEXT, params TEXT, gen INTEGER, parent TEXT,
        start_ms INTEGER, retired_ms INTEGER, retire_reason TEXT, gate TEXT, PRIMARY KEY (run_id, bot_id));
    CREATE TABLE IF NOT EXISTS arena_scores (run_id TEXT, cycle INTEGER, bot_id TEXT, ret REAL, mdd REAL,
        trades REAL, score REAL, equity REAL, PRIMARY KEY (run_id, cycle, bot_id));
    CREATE TABLE IF NOT EXISTS arena_trials (run_id TEXT, cycle INTEGER, bot_id TEXT, params TEXT, passed INTEGER,
        sharpe REAL, trades REAL, ret REAL);
    CREATE TABLE IF NOT EXISTS arena_ticks (run_id TEXT, t INTEGER, bot_id TEXT, equity REAL, total_trades REAL,
        open INTEGER, liqs REAL, fees REAL, PRIMARY KEY (run_id, t, bot_id));
    """)
    return con


class Gate:
    """Replay sanity gate on the pre-holdout window."""

    def __init__(self):
        engine.assert_outside_holdout(*GATE)
        bars, fund = data.load(SYMBOLS, TFS)
        self.m = {tf: engine.Market.from_db(bars, fund, SYMBOLS, tf, GATE_FROM, GATE[1]) for tf in TFS}

    def score(self, params):
        out = [None] * len(params)
        for tf in TFS:
            idx = [i for i, p in enumerate(params) if p["tf"] == tf]
            m = self.m[tf]
            j0 = m.index_of(GATE[0])
            for c in range(0, len(idx), 256):
                chunk = idx[c:c + 256]
                bots, sigs = st.build(m, [params[i] for i in chunk])
                r = engine.run(m, bots, sigs, j0, len(m.t), marks=[j0, len(m.t)])
                mt = engine.metrics(m, r, j0, j0, len(m.t), data.TF_MS[tf])
                for n, i in enumerate(chunk):
                    out[i] = {k: float(v[n]) for k, v in mt.items()}
        return out

    @staticmethod
    def passes(g):
        return g["trades"] >= GATE_MIN_TRADES and g["liqs"] == 0 and not g["dead"] and g["sharpe"] > 0


def live_bots(con, run_id, at_ms):
    rows = con.execute("SELECT bot_id, params, start_ms FROM arena_bots WHERE run_id=? AND start_ms<=? "
                       "AND (retired_ms IS NULL OR retired_ms>?)", (run_id, at_ms, at_ms)).fetchall()
    return [(b, json.loads(p), s) for b, p, s in rows]


def evaluate(con, run_id, anchors, at_ms, window_from=None):
    """Replay every live bot from its start to at_ms. Returns {bot_id: stats}."""
    bars, fund = data.load(SYMBOLS, TFS)
    out = {}
    bots_all = live_bots(con, run_id, at_ms)
    for tf in TFS:
        sel = [b for b in bots_all if b[1]["tf"] == tf]
        if not sel:
            continue
        tf_ms = data.TF_MS[tf]
        # only bars that have closed by at_ms
        m = engine.Market.from_db(bars, fund, SYMBOLS, tf, anchors[tf], at_ms - tf_ms + 1)
        starts = np.array([m.index_of(s) for _, _, s in sel])
        i0, i1 = int(starts.min()), len(m.t)
        if i1 <= i0:
            for b, p, s in sel:
                out[b] = dict(equity=engine.START_EQUITY, ret=0.0, mdd=0.0, trades=0.0, total_trades=0.0, liqs=0.0,
                              dead=False, open=0, fees=0.0, tf=tf, family=p["family"], last_bar="")
            continue
        bots, sigs = st.build(m, [p for _, p, _ in sel])
        j0 = max(m.index_of(window_from), i0) if window_from is not None else i0
        r = engine.run(m, bots, sigs, i0, i1, bot_start=starts, marks=[j0, i1])
        mt = engine.metrics(m, r, i0, j0, i1, tf_ms) if i1 > j0 else None
        for n, (b, p, s) in enumerate(sel):
            out[b] = dict(
                equity=float(r["eq"][-1, n]),
                ret=float(mt["ret"][n]) if mt else 0.0, mdd=float(mt["mdd"][n]) if mt else 0.0,
                trades=float(mt["trades"][n]) if mt else 0.0, total_trades=float(r["cnt"]["trades"][n]),
                liqs=float(r["cnt"]["liqs"][n]), dead=bool(r["state"]["dead"][n]),
                open=int((r["state"]["qty"][n] != 0).sum()), fees=float(r["cnt"]["fees"][n]),
                tf=tf, family=p["family"], last_bar=utc(int(m.t[-1]) + tf_ms))
    return out


def record_ticks(con, run, upto_ms):
    """Output only (dashboard): every live bot's equity on the 5m grid in [start, upto_ms].
    Missing ticks are backfilled by replay, so the series doesn't depend on uptime."""
    run_id = run["run_id"]
    have = {r[0] for r in con.execute("SELECT DISTINCT t FROM arena_ticks WHERE run_id=?", (run_id,))}
    for t in range(run["started_ms"], min(upto_ms, run["end_ms"] - 1) + 1, TICK_MS):
        if t in have:
            continue
        res = evaluate(con, run_id, run["anchors"], t)
        con.executemany("INSERT OR REPLACE INTO arena_ticks VALUES (?,?,?,?,?,?,?,?)",
                        [(run_id, t, b, s["equity"], s["total_trades"], s["open"], s["liqs"], s["fees"]) for b, s in res.items()])
        con.commit()


def spawn(con, run_id, gate, n_need, cycle, start_ms, rng, parents=(), max_trials=None):
    """Draw candidates until n_need pass the gate or max_trials (default 40x) are tried."""
    have = {r[0] for r in con.execute("SELECT bot_id FROM arena_bots WHERE run_id=?", (run_id,))}
    accepted, tried = [], 0
    max_trials = max_trials or 40 * max(n_need, 1)
    while len(accepted) < n_need and tried < max_trials:
        cands = []
        for _ in range(min(1024, max_trials - tried)):
            if parents and rng.random() < MUTANT_SHARE:
                pid, pp = parents[rng.integers(len(parents))]
                cands.append((st.mutate(pp, rng), pid))
            else:
                cands.append((st.sample(rng), None))
        tried += len(cands)
        cands = [(p, par) for p, par in cands if st.bot_id(p) not in have]
        scores = gate.score([p for p, _ in cands])
        for (p, par), g in zip(cands, scores):
            bid = st.bot_id(p)
            ok = Gate.passes(g)
            con.execute("INSERT INTO arena_trials VALUES (?,?,?,?,?,?,?,?)",
                        (run_id, cycle, bid, json.dumps(p), int(ok), g["sharpe"], g["trades"], g["ret"]))
            if ok and bid not in have and len(accepted) < n_need:
                have.add(bid)
                accepted.append((bid, p, par, g))
    con.commit()
    return accepted


def pick_gen0(accepted, n):
    by_tf = {tf: sorted([a for a in accepted if a[1]["tf"] == tf], key=lambda a: -a[3]["sharpe"]) for tf in TFS}
    quota = n // len(TFS)
    chosen = [a for tf in TFS for a in by_tf[tf][:quota]]
    rest = sorted([a for a in accepted if a not in chosen], key=lambda a: -a[3]["sharpe"])
    return chosen + rest[:n - len(chosen)]


def insert_bots(con, run_id, bots, gen, start_ms):
    con.executemany("INSERT INTO arena_bots VALUES (?,?,?,?,?,?,?,?,?)",
                    [(run_id, b, json.dumps(p), gen, par, start_ms, None, None, json.dumps(g)) for b, p, par, g in bots])
    con.commit()


def lifetime(con, run_id):
    rows = con.execute("SELECT bot_id, COUNT(*), SUM(score), SUM(trades) FROM arena_scores WHERE run_id=? GROUP BY bot_id",
                       (run_id,)).fetchall()
    return {b: dict(n=n, L=s / (n + SHRINK), trades=t) for b, n, s, t in rows}


def finish_cycle(con, run, cycle, c0, c1, gate, rng):
    run_id = run["run_id"]
    engine.assert_outside_holdout(c0, c1)
    res = evaluate(con, run_id, run["anchors"], c1, window_from=c0)
    for b, s in res.items():
        score = s["ret"] - MDD_PENALTY * s["mdd"]
        con.execute("INSERT OR REPLACE INTO arena_scores VALUES (?,?,?,?,?,?,?,?)",
                    (run_id, cycle, b, s["ret"], s["mdd"], s["trades"], score, s["equity"]))
    con.commit()
    life = lifetime(con, run_id)
    params = {b: p for b, p, _ in live_bots(con, run_id, c1)}

    # cull
    retire = {b: "dd_kill" for b, s in res.items() if s["dead"]}
    retire.update({b: "liquidation" for b, s in res.items() if s["liqs"] > 0})
    elig = sorted([b for b in res if b not in retire and life.get(b, {}).get("n", 0) >= CULL_MIN_CYCLES],
                  key=lambda b: life[b]["L"])
    for b in elig[:int(len(elig) * CULL_FRACTION)]:
        if life[b]["L"] < 0:  # idle bots (L == 0) are not culled for being idle
            retire[b] = "bottom_lifetime"
    for b, why in retire.items():
        con.execute("UPDATE arena_bots SET retired_ms=?, retire_reason=? WHERE run_id=? AND bot_id=?", (c1, why, run_id, b))
    con.commit()

    # refill
    alive = [b for b in res if b not in retire]
    ranked = sorted(alive, key=lambda b: -life[b]["L"])
    parents = [(b, params[b]) for b in ranked if life[b]["trades"] >= PARENT_MIN_TRADES][:PARENTS]
    new = spawn(con, run_id, gate, run["n_bots"] - len(alive), cycle + 1, c1, rng, parents)
    insert_bots(con, run_id, new, cycle + 1, c1)
    write_report(con, run, cycle, c0, c1, res, life, retire, new)
    ledger.append(ledger.arena_row(con, run_id, cycle, SHRINK, engine.START_EQUITY))
    return res


def _fmt_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)


def write_report(con, run, cycle, c0, c1, res, life, retire, new):
    os.makedirs(REPORTS, exist_ok=True)
    params = {b: json.loads(p) for b, p in con.execute("SELECT bot_id, params FROM arena_bots WHERE run_id=?", (run["run_id"],))}
    df = pd.DataFrame([{**s, "bot": b, "L": life.get(b, {}).get("L", 0), "cycles": life.get(b, {}).get("n", 0)}
                       for b, s in res.items()])
    top = df.sort_values("L", ascending=False).head(20)
    rows = [(r.bot, r.family, r.tf, r.cycles, f"{r.L:+.4f}", f"{r.ret*100:+.2f}%", f"{r.mdd*100:.2f}%", int(r.trades),
             f"{r.equity:,.0f}", _short(params[r.bot])) for r in top.itertuples()]
    grp = df.groupby(["family", "tf"]).agg(n=("bot", "size"), mean_ret=("ret", "mean"), med_ret=("ret", "median"),
                                           trades=("trades", "sum")).reset_index()
    grows = [(g.family, g.tf, g.n, f"{g.mean_ret*100:+.3f}%", f"{g.med_ret*100:+.3f}%", int(g.trades)) for g in grp.itertuples()]
    reasons = pd.Series(list(retire.values())).value_counts().to_dict() if retire else {}
    newdesc = pd.Series([("mutant" if par else "random") + ":" + p["family"] + "/" + p["tf"] for _, p, par, _ in new]).value_counts().to_dict()
    txt = f"""# Arena {run['run_id']} cycle {cycle}

- Window: {utc(c0)} ~ {utc(c1)} UTC. Live bots: {len(res)}.
- Cycle return: mean {df.ret.mean()*100:+.3f}%, median {df.ret.median()*100:+.3f}%. Total trades {int(df.trades.sum())}. Liquidations {int(df.liqs.sum())}.
- Retired {len(retire)}: {reasons}
- Added {len(new)}: {newdesc}

## Lifetime score top 20
{_fmt_table(rows, ['bot','family','tf','cycles','L','cycle ret','cycle mdd','trades','equity','params'])}
## Family x timeframe (this cycle)
{_fmt_table(grows, ['family','tf','n','mean ret','median ret','trades'])}
> Few cycles mostly measure luck. This is not evidence for real money.
"""
    with open(os.path.join(REPORTS, f"{run['run_id']}_cycle{cycle:02d}.md"), "w") as f:
        f.write(txt)


def _short(p):
    skip = {"family", "tf"}
    return " ".join(f"{k}={v}" for k, v in p.items() if k not in skip)


def write_leaderboard(run, res, life, now_ms, cycle, c0, c1):
    df = pd.DataFrame([{**s, "bot": b} for b, s in res.items()])
    if df.empty:
        return
    df["L"] = [life.get(b, {}).get("L", 0) for b in df.bot]
    top = df.sort_values("equity", ascending=False).head(30)
    rows = [(r.bot, r.family, r.tf, f"{r.equity:,.2f}", f"{(r.equity/engine.START_EQUITY-1)*100:+.2f}%",
             int(r.total_trades), r.open, f"{r.L:+.4f}") for r in top.itertuples()]
    txt = f"""# Arena leaderboard ({run['run_id']})

Updated {utc(now_ms)} UTC. Cycle {cycle} ({utc(c0)} ~ {utc(c1)}). Live bots {len(df)}. Equity mean ${df.equity.mean():,.0f}, median ${df.equity.median():,.0f}.
Open positions {int(df.open.sum())}. Liquidations {int(df.liqs.sum())}.

{_fmt_table(rows, ['bot','family','tf','equity','since start','trades','open','lifetime L'])}"""
    with open(os.path.join(STATE, "leaderboard.md"), "w") as f:
        f.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--cycle-hours", type=float, default=5)
    ap.add_argument("--bots", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="run_id to resume (default: latest running)")
    a = ap.parse_args()
    os.makedirs(REPORTS, exist_ok=True)
    con = db()
    rng = np.random.default_rng(a.seed)
    data.update(SYMBOLS, TFS)
    gate = Gate()

    row = con.execute("SELECT run_id, started_ms, end_ms, cycle_ms, n_bots, anchors FROM arena_runs WHERE status='running' "
                      + ("AND run_id=? " if a.resume else "") + "ORDER BY started_ms DESC LIMIT 1",
                      (a.resume,) if a.resume else ()).fetchone()
    if row:
        run = dict(run_id=row[0], started_ms=row[1], end_ms=row[2], cycle_ms=row[3], n_bots=row[4], anchors=json.loads(row[5]))
        print("resume", run["run_id"], flush=True)
    else:
        now = data.server_time()
        start = (now // 3_600_000 + 1) * 3_600_000  # next full hour
        run_id = datetime.fromtimestamp(start / 1000, timezone.utc).strftime("R%Y%m%d_%H%M")
        anchors = {tf: start - WARMUP_BARS * data.TF_MS[tf] for tf in TFS}
        run = dict(run_id=run_id, started_ms=start, end_ms=start + int(a.hours * 3_600_000),
                   cycle_ms=int(a.cycle_hours * 3_600_000), n_bots=a.bots, anchors=anchors)
        with open(os.path.join(STATE, "selection_rule.md"), "w") as f:
            f.write(RULE + f"\nWritten {utc(now)} UTC, before run {run_id} (live start {utc(start)} UTC).\n")
        accepted = spawn(con, run_id, gate, a.bots, 0, start, rng, max_trials=40 * a.bots)
        gen0 = pick_gen0(accepted, a.bots)
        insert_bots(con, run_id, gen0, 0, start)
        con.execute("INSERT INTO arena_runs VALUES (?,?,?,?,?,?,?)", (run_id, start, run["end_ms"], run["cycle_ms"],
                                                                      a.bots, json.dumps(anchors), "running"))
        con.commit()
        tfc = pd.Series([p["tf"] + "/" + p["family"] for _, p, _, _ in gen0]).value_counts().to_dict()
        print(f"run {run_id}: {len(gen0)} bots pass the gate ({len(accepted)} passed / trials logged). live start {utc(start)} UTC. {tfc}", flush=True)

    while True:
        done = {r[0] for r in con.execute("SELECT DISTINCT cycle FROM arena_scores WHERE run_id=?", (run["run_id"],))}
        cycle = len(done)
        c0 = run["started_ms"] + cycle * run["cycle_ms"]
        c1 = min(c0 + run["cycle_ms"], run["end_ms"])
        if c0 >= run["end_ms"]:
            try:
                record_ticks(con, run, run["end_ms"])
            except Exception as e:
                print(f"{utc(int(time.time()*1000))} tick record error: {e!r}", flush=True)
            con.execute("UPDATE arena_runs SET status='done' WHERE run_id=?", (run["run_id"],)); con.commit()
            try:  # a crash here would let systemd (Restart=on-failure) start a new run
                ledger.append(ledger.arena_row(con, run["run_id"], None, SHRINK, engine.START_EQUITY))
            except Exception as e:
                print(f"ledger error: {e!r}", flush=True)
            print("run finished", flush=True)
            break
        try:
            synced = data.server_time()  # bars closed before this are fetched below
            data.update(SYMBOLS, TFS)
            now = data.server_time()
            if now >= c1 + 60_000:  # all bars up to c1 are closed and fetched
                res = finish_cycle(con, run, cycle, c0, c1, gate, rng)
                print(f"{utc(now)} cycle {cycle} done: {len(res)} bots, report written", flush=True)
                continue
            if now >= run["started_ms"]:
                try:  # ticks at c1 wait for the new population, so t < c1
                    record_ticks(con, run, min(synced // TICK_MS * TICK_MS, c1 - 1))
                except Exception as e:
                    print(f"{utc(int(time.time()*1000))} tick record error: {e!r}", flush=True)
                res = evaluate(con, run["run_id"], run["anchors"], now)
                write_leaderboard(run, res, lifetime(con, run["run_id"]), now, cycle, c0, c1)
        except Exception as e:  # keep the loop alive through transient API errors
            print(f"{utc(int(time.time()*1000))} tick error: {e!r}", flush=True)
        time.sleep(300 - (time.time() % 300) + 20)


if __name__ == "__main__":
    main()
