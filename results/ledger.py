"""Experiment ledger: one JSON row per result (lab round, control-variant rescore, arena cycle/run, other backtests).

Stdlib only and Python 3.6 compatible (Oracle runs 3.6). Writers call append(); rows are upserted on read by
(source, id), latest `ts` wins, so re-running a report just replaces its row.

Each host keeps its own results/ledger.jsonl. The Mac pulls the desktop and Oracle files into results/hosts/<host>.jsonl
(research/sync_desktop.sh) and merges them into its own ledger; the merged file is never pushed back.

    python3 -m results.ledger merge     # Mac: fold results/hosts/*.jsonl into results/ledger.jsonl
    python3 -m results.ledger obsidian  # Mac: write the vault note "Arena Results.md" from the merged ledger
"""
import glob
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "ledger.jsonl")
HOSTS = os.path.join(ROOT, "hosts")
VAULT_NOTE = os.path.expanduser("~/Documents/Obsidian Vault/wiki/projects/Arena Results.md")

# source: lab | lab_variant | arena_cycle | arena_run | backtest
# trials_cum counts lab trials only (configs ever scored); arena_trials counts arena gate candidates. Never sum the two.
FIELDS = ["ts", "host", "source", "id", "date", "run", "sleeve", "catalog_id", "hypothesis", "n_configs", "new_trials", "trials_cum",
          "arena_trials", "metrics", "top", "conclusion", "report"]
REQUIRED = ["source", "id", "date", "metrics", "conclusion", "report"]
# horizon sleeves (research/state/lab_timeframes.md): results are ranked only within a sleeve, never across.
# Rows written before the field existed get the default for their source.
SLEEVES = ["intraday", "swing", "position"]
DEFAULT_SLEEVE = {"lab": "swing", "lab_variant": "swing", "lab_candidate": "swing", "lab_verdict": "swing",
                  "lab_validation": "swing", "retro": "swing", "arena_cycle": "intraday", "arena_run": "intraday"}


def sleeve(row):
    return row.get("sleeve") or DEFAULT_SLEEVE.get(row["source"], "-")


def _host():
    if sys.platform == "darwin":
        return "mac"
    return "oracle" if os.path.isdir("/home/opc") else "desktop"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(row, path=PATH):
    missing = [k for k in REQUIRED if row.get(k) in (None, "")]
    if missing:
        raise ValueError("ledger row missing {}".format(missing))
    extra = set(row) - set(FIELDS)
    if extra:
        raise ValueError("ledger row has unknown fields {}".format(sorted(extra)))
    if row.get("sleeve") not in (None, "") and row["sleeve"] not in SLEEVES:
        raise ValueError("ledger row sleeve {!r} not in {}".format(row["sleeve"], SLEEVES))
    full = dict((k, row.get(k)) for k in FIELDS)
    full["sleeve"] = sleeve(full) if full["source"] in DEFAULT_SLEEVE else full["sleeve"]
    full["ts"] = full["ts"] or _now()
    full["host"] = full["host"] or _host()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(full, ensure_ascii=False, default=float) + "\n")
    return full


def load(paths=None):
    """Rows from the given jsonl files, upserted by (source, id); latest ts wins. Sorted by date, then id."""
    best = {}
    for p in paths or [PATH]:
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                k = (r["source"], r["id"])
                if k not in best or r["ts"] >= best[k]["ts"]:
                    best[k] = r
    return sorted(best.values(), key=lambda r: (r["date"], r["id"]))


def merge():
    rows = load([PATH] + sorted(glob.glob(os.path.join(HOSTS, "*.jsonl"))))
    tmp = PATH + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, PATH)
    return rows


# ---------- arena ----------

def arena_row(con, run_id, cycle=None, shrink=2.0, start_equity=10000.0):
    """Ledger row for one finished arena cycle, or for the whole run when cycle is None. Read-only on `con`.
    shrink / start_equity mirror arena/run_live.py SHRINK and engine.START_EQUITY."""
    def q(sql, *a):
        return con.execute(sql, a).fetchall()
    started, end, cycle_ms, n_bots, status = q("SELECT started_ms, end_ms, cycle_ms, n_bots, status FROM arena_runs "
                                               "WHERE run_id=?", run_id)[0]
    last_cycle = cycle if cycle is not None else q("SELECT MAX(cycle) FROM arena_scores WHERE run_id=?", run_id)[0][0]
    if cycle is not None:
        rows = q("SELECT ret, mdd, trades FROM arena_scores WHERE run_id=? AND cycle=?", run_id, cycle)
    else:  # whole run, per bot: return from its last scored equity, worst cycle mdd, total trades
        rows = q("SELECT s.equity / ? - 1, t.mdd, t.trades FROM arena_scores s JOIN (SELECT bot_id, MAX(cycle) c, "
                 "MAX(mdd) mdd, SUM(trades) trades FROM arena_scores WHERE run_id=? GROUP BY bot_id) t "
                 "ON s.bot_id=t.bot_id AND s.cycle=t.c WHERE s.run_id=?", start_equity, run_id, run_id)
    rets = sorted(r for r, _, _ in rows)
    mdds = sorted(m for _, m, _ in rows)
    pos = sum(r > 0 for r in rets)
    c1 = min(started + (last_cycle + 1) * cycle_ms, end)
    retired = dict(q("SELECT retire_reason, COUNT(*) FROM arena_bots WHERE run_id=? AND retired_ms IS NOT NULL AND "
                     "retired_ms " + ("=" if cycle is not None else "<=") + " ? GROUP BY retire_reason", run_id, c1))
    cand, passed = q("SELECT COUNT(*), SUM(passed) FROM arena_trials WHERE run_id=? AND cycle<=?", run_id, last_cycle + 1)[0]
    params = dict(q("SELECT bot_id, params FROM arena_bots WHERE run_id=?", run_id))
    top = []
    for b, s, n in q("SELECT bot_id, SUM(score), COUNT(*) FROM arena_scores WHERE run_id=? AND cycle<=? GROUP BY bot_id "
                     "ORDER BY SUM(score) / (COUNT(*) + ?) DESC LIMIT 5", run_id, last_cycle, shrink):
        p = json.loads(params.get(b, "{}"))
        top.append("{} {}/{} L={:+.4f} ({} cycles)".format(b, p.get("tf", "?"), p.get("family", "?"), s / (n + shrink), n))
    med = rets[len(rets) // 2] if rets else 0.0
    m = dict(bots=len(rets), ret_median=med, ret_mean=sum(rets) / len(rets) if rets else None,
             ret_best=rets[-1] if rets else None, ret_worst=rets[0] if rets else None, pos_share=pos / len(rets) if rets else None,
             mdd_median=mdds[len(mdds) // 2] if mdds else None, mdd_max=mdds[-1] if mdds else None,
             trades=sum(t for _, _, t in rows), retired=retired, gate_passed=passed, gate_candidates=cand)
    day = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")
    base = dict(run=run_id, arena_trials=cand, metrics=m, top=top,
                hypothesis="evolving paper arena (pre-registered selection_rule.md) finds bots with a positive live score")
    if cycle is not None:
        return dict(base, source="arena_cycle", id="{}_c{:02d}".format(run_id, cycle), date=day(c1),
                    report="arena/state/reports/{}_cycle{:02d}.md".format(run_id, cycle),
                    conclusion="cycle {}: median ret {:+.2%}, {}/{} bots positive, {} retired".format(
                        cycle, med, pos, len(rets), sum(retired.values())))
    return dict(base, source="arena_run", id=run_id, date=day(end), report="arena/state/leaderboard.md",
                conclusion="{} ({}, {} cycles): median bot total ret {:+.2%}, {}/{} positive".format(
                    run_id, status, last_cycle + 1, med, pos, len(rets)))


def backfill_arena(db_path, run_id):
    """Rows for every finished cycle plus the run, from a read-only connection."""
    con = sqlite3.connect("file:{}?mode=ro".format(db_path), uri=True)
    cycles = [c for (c,) in con.execute("SELECT DISTINCT cycle FROM arena_scores WHERE run_id=? ORDER BY cycle", (run_id,))]
    return [arena_row(con, run_id, c) for c in cycles] + ([arena_row(con, run_id)] if cycles else [])


def backfill_arena_all(db_path):
    con = sqlite3.connect("file:{}?mode=ro".format(db_path), uri=True)
    runs = [r for (r,) in con.execute("SELECT run_id FROM arena_runs ORDER BY started_ms")]
    return [row for r in runs for row in backfill_arena(db_path, r)]


# ---------- Obsidian ----------

def _fmt(v):
    if isinstance(v, float):
        return "{:.3g}".format(v)
    if isinstance(v, dict):
        return ", ".join("{} {}".format(k, _fmt(x)) for k, x in v.items())
    return str(v)


def obsidian(rows=None, path=VAULT_NOTE):
    rows = rows or load()
    lab = [r for r in rows if r["source"] in ("lab", "lab_variant")]
    trials = max([r["trials_cum"] or 0 for r in lab] or [0])
    by_sleeve = {}
    for r in rows:
        if r["source"] == "lab":
            by_sleeve[sleeve(r)] = by_sleeve.get(sleeve(r), 0) + (r.get("n_configs") or 0)
    out = ["---", "type: project-log", "tags: [trading, research, ledger]", "updated: " + _now()[:10], "---", "",
           "# Arena Results", "",
           "Generated by `python3 -m results.ledger obsidian` from the merged `results/ledger.jsonl` (Mac). Do not edit by hand; "
           "edits are overwritten. Project: [[Bentley Auto Trading]].", "",
           "- Lab cumulative trials: **{}** (arena gate candidates are counted separately); new configs by sleeve: {}.".format(
               trials, ", ".join("{} {}".format(k, v) for k, v in sorted(by_sleeve.items())) or "none"),
           "- Holdout 2026-03 ~ 2026-09 stays sealed; every lab number below is in-sample unless it says OOS.", "",
           "- Results are compared only within a sleeve (intraday / swing / position); trials stay cumulative across all.", ""]
    for sl in SLEEVES + ["-"]:
        part = [r for r in reversed(rows) if sleeve(r) == sl]
        if not part:
            continue
        out += ["## Sleeve: {}".format(sl), "", "| date | source | id | conclusion | report |", "|---|---|---|---|---|"]
        for r in part:
            out.append("| {} | {} | {} | {} | `{}` |".format(r["date"], r["source"], r["id"], r["conclusion"].replace("|", "/"),
                                                            r["report"]))
        out.append("")
    out += ["## Details", ""]
    for r in reversed(rows):
        out += ["### {} ({})".format(r["id"], r["source"]), "",
                "- date {} · sleeve {}{} · host {} · written {}".format(r["date"], sleeve(r),
                    " · catalog " + r["catalog_id"] if r.get("catalog_id") else "", r["host"], r["ts"])]
        if r.get("hypothesis"):
            out.append("- hypothesis: " + r["hypothesis"])
        counts = [(k, r.get(k)) for k in ("n_configs", "new_trials", "trials_cum", "arena_trials") if r.get(k) is not None]
        if counts:
            out.append("- " + " · ".join("{} {}".format(k, v) for k, v in counts))
        out += ["- metrics: " + "; ".join("{}: {}".format(k, _fmt(v)) for k, v in r["metrics"].items())]
        if r.get("top"):
            out += ["- top:"] + ["  - `{}`".format(t) for t in r["top"]]
        out += ["- conclusion: " + r["conclusion"], "- report: `{}`".format(r["report"]), ""]
    with open(path, "w") as f:
        f.write("\n".join(out))
    return path


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "merge":
        print(len(merge()), "rows in", PATH)
    elif cmd == "obsidian":
        print(obsidian())
    elif cmd == "arena-backfill":  # python3 results/ledger.py arena-backfill <arena.sqlite>  (all runs, prints jsonl)
        for r in backfill_arena_all(sys.argv[2]):
            print(json.dumps(append(r, path=os.devnull), ensure_ascii=False))
    else:
        sys.exit(__doc__)
