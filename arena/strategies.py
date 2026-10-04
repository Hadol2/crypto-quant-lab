"""Strategy families, parameter space, mutation and signal arrays.

All signals are evaluated at bar close using data through that bar.
"""
import hashlib
import json

import numpy as np

TFS = ("5m", "15m", "1h")

COMMON = {
    "allow_short": [0, 1],
    "stop_k": (0.8, 5.0, float),     # stop distance in ATR(14)
    "risk_pct": [2.0],               # % equity at risk per trade (fixed 2% from the run after R20260929_1200)
    "max_lev": (1, 20, int),         # isolated leverage cap (auto-lowered to keep liq beyond stop)
    "max_gross": (0.5, 5.0, float),  # total notional / equity
    "max_pos": (1, 8, int),
    "daily_kill": (0.02, 0.10, float),
    "dd_kill": (0.10, 0.40, float),
}
FAMILY = {
    # Donchian/Longinus-style breakout with ATR trail and channel exit
    "breakout": {"n_entry": (10, 200, int), "n_exit": (5, 100, int), "adx_min": [0, 15, 20, 25, 30]},
    # EMA trend: enter on cross, exit on reverse cross, ATR trail
    "ema": {"fast": (3, 60, int), "slow": (10, 300, int)},
    # RSI mean reversion with fixed ATR stop and max hold
    "meanrev": {"rsi_n": (2, 21, int), "lo": (5, 35, int), "trend_n": [0, 50, 200], "max_hold": (3, 96, int)},
    # Momentum / stop-and-reverse flavour: sign of n-bar return with vol filter
    "momentum": {"lookback": (6, 240, int), "thresh_atr": (0.0, 3.0, float)},
}
TRAILING = {"breakout": 1, "ema": 1, "meanrev": 0, "momentum": 1}


def _draw(spec, rng):
    if isinstance(spec, list):
        return spec[rng.integers(len(spec))]
    lo, hi, typ = spec
    if typ is int:
        return int(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
    return round(float(rng.uniform(lo, hi)), 3)


def _fix(p):
    if p["family"] == "ema":
        p["slow"] = max(p["slow"], int(p["fast"] * 1.5) + 1)
    return p


def sample(rng, family=None, tf=None):
    fam = family or list(FAMILY)[rng.integers(len(FAMILY))]
    p = {"family": fam, "tf": tf or TFS[rng.integers(len(TFS))]}
    for k, s in {**COMMON, **FAMILY[fam]}.items():
        p[k] = _draw(s, rng)
    return _fix(p)


def mutate(p, rng, rate=0.35):
    q = dict(p)
    for k, s in {**COMMON, **FAMILY[p["family"]]}.items():
        if rng.random() > rate:
            continue
        if isinstance(s, list):
            q[k] = s[rng.integers(len(s))]
        else:
            lo, hi, typ = s
            v = q[k] * np.exp(rng.normal(0, 0.25))
            v = min(max(v, lo), hi)
            q[k] = int(round(v)) if typ is int else round(float(v), 3)
    if rng.random() < 0.1:
        q["tf"] = TFS[rng.integers(len(TFS))]
    return _fix(q)


def bot_id(p):
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]


def sig_key(p):
    names = sorted(FAMILY[p["family"]])
    names = [n for n in names if n != "max_hold"]
    return (p["family"], p["allow_short"]) + tuple(p[n] for n in names)


def _signals(mkt, p):
    f, c = p["family"], mkt._df(mkt.C)
    if f == "breakout":
        adx = mkt.feat("adx", 14)
        ok = adx >= p["adx_min"] if p["adx_min"] else c.notna()
        el = (c > mkt.feat("hh", p["n_entry"])) & ok
        es = (c < mkt.feat("ll", p["n_entry"])) & ok
        xl = c < mkt.feat("ll", p["n_exit"])
        xs = c > mkt.feat("hh", p["n_exit"])
    elif f == "ema":
        fa, sl = mkt.feat("ema", p["fast"]), mkt.feat("ema", p["slow"])
        up = fa > sl
        el = up & ~up.shift(1, fill_value=True)
        es = ~up & up.shift(1, fill_value=False) & sl.notna()
        xl, xs = ~up & sl.notna(), up
    elif f == "meanrev":
        r = mkt.feat("rsi", p["rsi_n"])
        el, es = r < p["lo"], r > 100 - p["lo"]
        if p["trend_n"]:
            e = mkt.feat("ema", p["trend_n"])
            el, es = el & (c > e), es & (c < e)
        xl, xs = r > 50, r < 50
    elif f == "momentum":
        n = p["lookback"]
        mom = (c - c.shift(n)) / mkt.feat("atr", 14)
        up, dn = mom > p["thresh_atr"], mom < -p["thresh_atr"]
        el, es = up & ~up.shift(1, fill_value=True), dn & ~dn.shift(1, fill_value=True)
        xl, xs = mom < 0, mom > 0
    else:
        raise KeyError(f)
    if not p["allow_short"]:
        es = es & False
    return [x.fillna(False).values.astype(bool) for x in (el, es, xl, xs)]


def build(mkt, params):
    """Return (bots arrays, sigs) for a list of param dicts sharing one tf."""
    keys, kidx = [], {}
    sig = np.empty(len(params), dtype=np.int64)
    for i, p in enumerate(params):
        k = sig_key(p)
        if k not in kidx:
            kidx[k] = len(keys); keys.append(p)
        sig[i] = kidx[k]
    T, S, K = len(mkt.t), len(mkt.symbols), len(keys)
    sigs = {n: np.zeros((T, K, S), bool) for n in ("EL", "ES", "XL", "XS")}
    for j, p in enumerate(keys):
        for n, a in zip(("EL", "ES", "XL", "XS"), _signals(mkt, p)):
            sigs[n][:, j, :] = a
    g = lambda k, typ=float: np.array([p[k] for p in params], dtype=typ)
    bots = dict(
        sig=sig, stop_k=g("stop_k"), risk=g("risk_pct") / 100, max_lev=g("max_lev"),
        max_gross=g("max_gross"), max_pos=g("max_pos"), daily_kill=g("daily_kill"), dd_kill=g("dd_kill"),
        trail=np.array([TRAILING[p["family"]] for p in params], bool),
        max_hold=np.array([p.get("max_hold", 10**9) for p in params], dtype=np.int64),
    )
    return bots, sigs
