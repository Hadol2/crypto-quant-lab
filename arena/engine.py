"""Vectorized bar-close USD-M futures paper engine.

One engine for both feeds: batch replay and live are the same call over
closed bars, so live results equal a replay of the same span.

Per bar t, in this order (all causal):
  1. funding at recorded settlement timestamps (positions held at bar open)
  2. liquidation / stop checks against levels set through bar t-1
  3. signal and max-hold exits at close
  4. trail ratchet with bar t close/ATR (only for positions still open)
  5. risk layer: daily-loss kill (block until next UTC day), drawdown kill (dead)
  6. entries at close, sized by risk, capped by gross exposure / positions / free margin

Isolated margin per position. Leverage is auto-lowered so the liquidation
price sits beyond the initial stop. Liquidation uses last-price highs/lows,
which is more conservative than Binance's mark price. A liquidation loses the
position's full isolated margin.
"""
import numpy as np
import pandas as pd

START_EQUITY = 10_000.0
TAKER_FEE = 0.0005          # Binance USD-M VIP0 taker (unverified without auth; conservative)
MIN_NOTIONAL = 20.0
LIQ_BUFFER = 0.005          # liq price at least stop distance + 0.5% away at entry
# One-shot holdout: history not yet used for tuning. Ends where forward (live) data begins.
HOLDOUT = (pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-09-29", tz="UTC"))
HOLDOUT_MS = (int(HOLDOUT[0].timestamp() * 1000), int(HOLDOUT[1].timestamp() * 1000))

# conservative tier-1 maintenance margin rates / per-side slippage
MMR = {"BTCUSDT": 0.004, "ETHUSDT": 0.005}
SLIP = {"BTCUSDT": 0.0002, "ETHUSDT": 0.0002}
DEFAULT_MMR, DEFAULT_SLIP = 0.01, 0.0005


class HoldoutViolation(RuntimeError):
    pass


def assert_outside_holdout(t0_ms, t1_ms):
    """Scoring windows may not overlap the one-shot holdout."""
    if t0_ms < HOLDOUT_MS[1] and t1_ms > HOLDOUT_MS[0]:
        raise HoldoutViolation(f"scoring window {t0_ms}-{t1_ms} overlaps holdout {HOLDOUT}")


class Market:
    def __init__(self, t, O, H, L, C, FUND, symbols):
        self.t, self.O, self.H, self.L, self.C, self.FUND = t, O, H, L, C, FUND
        self.symbols = list(symbols)
        self.day = t // 86_400_000
        self.mmr = np.array([MMR.get(s, DEFAULT_MMR) for s in symbols])
        self.slip = np.array([SLIP.get(s, DEFAULT_SLIP) for s in symbols])
        self._cache = {}

    @classmethod
    def from_db(cls, bars, funding, symbols, tf, t_from=None, t_to=None):
        dfs = [bars[(s, tf)].set_index("t") for s in symbols]
        idx = dfs[0].index
        for d in dfs[1:]:
            idx = idx.intersection(d.index)
        if t_from is not None:
            idx = idx[idx >= t_from]
        if t_to is not None:
            idx = idx[idx < t_to]
        t = idx.values.astype(np.int64)
        col = lambda k: np.column_stack([d.loc[idx, k].values for d in dfs]).astype(np.float64)
        fund = np.zeros((len(t), len(symbols)))
        for j, s in enumerate(symbols):
            f = funding.get(s, {})
            fund[:, j] = [f.get(int(x), 0.0) for x in t]
        return cls(t, col("o"), col("h"), col("l"), col("c"), fund, symbols)

    def index_of(self, t_ms):
        return int(np.searchsorted(self.t, t_ms))

    # ---- cached indicators, [T, S] ----
    def _df(self, a):
        return pd.DataFrame(a)

    def feat(self, name, n=0):
        key = (name, n)
        if key in self._cache:
            return self._cache[key]
        h, l, c = self._df(self.H), self._df(self.L), self._df(self.C)
        if name == "atr":
            tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()]).groupby(level=0).max()
            v = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        elif name == "ema":
            v = c.ewm(span=n, adjust=False, min_periods=n).mean()
        elif name == "rsi":
            d = c.diff()
            up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
            dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
            v = 100 - 100 / (1 + up / dn.replace(0, np.nan))
            v = v.fillna(50.0).where(up.notna())
        elif name == "adx":
            upm, dnm = h.diff(), -l.diff()
            pdm = upm.where((upm > dnm) & (upm > 0), 0.0)
            mdm = dnm.where((dnm > upm) & (dnm > 0), 0.0)
            atr = self.feat("atr", n)
            pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
            mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr
            dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
            v = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        elif name == "hh":  # highest high of the previous n bars (excludes bar t)
            v = h.rolling(n).max().shift(1)
        elif name == "ll":
            v = l.rolling(n).min().shift(1)
        else:
            raise KeyError(name)
        self._cache[key] = v
        return v


def run(mkt, bots, sigs, i0, i1, bot_start=None, marks=()):
    """Simulate bots over bars [i0, i1).

    bots: dict of per-bot arrays (see strategies.bot_arrays)
    sigs: dict EL/ES/XL/XS bool [T, K, S] plus bots["sig"] int [B] mapping bot -> K
    bot_start: int [B] first bar index at which a bot may trade (default i0)
    marks: bar indices at which to snapshot counters (taken before bar executes)
    Returns equity [i1-i0, B], counters, snapshots and final state.
    """
    B, S = len(bots["sig"]), len(mkt.symbols)
    sig = bots["sig"]
    stop_k = bots["stop_k"][:, None]
    trail = bots["trail"][:, None]
    max_hold = bots["max_hold"][:, None]
    risk, max_lev = bots["risk"], bots["max_lev"]
    max_gross, max_pos = bots["max_gross"], bots["max_pos"]
    dkill, ddkill = bots["daily_kill"], bots["dd_kill"]
    bot_start = np.full(B, i0) if bot_start is None else np.asarray(bot_start)
    atr = mkt.feat("atr", 14).values
    mmr, slip = mkt.mmr, mkt.slip

    qty = np.zeros((B, S)); entry = np.zeros((B, S)); stop = np.zeros((B, S))
    peak = np.zeros((B, S)); margin = np.zeros((B, S)); liq = np.zeros((B, S))
    held = np.zeros((B, S), dtype=np.int64)
    wallet = np.full(B, START_EQUITY); peak_eq = wallet.copy(); day_eq = wallet.copy()
    blocked = np.zeros(B, bool); dead = np.zeros(B, bool)
    cnt = {k: np.zeros(B) for k in ("trades", "wins", "liqs", "fees", "funding",
                                    "gross_win", "gross_loss", "bars_in_mkt", "daily_kills")}
    eq_hist = np.empty((i1 - i0, B))
    snaps = {}
    marks = set(marks)
    last_day = mkt.day[i0 - 1] if i0 > 0 else -1

    def close(mask, fill):
        nonlocal wallet
        if not mask.any():
            return
        fee = np.abs(qty) * fill * TAKER_FEE
        pnl = np.where(mask, qty * (fill - entry) - fee, 0.0)
        wallet = wallet + pnl.sum(1)
        cnt["fees"] += np.where(mask, fee, 0).sum(1)
        cnt["trades"] += mask.sum(1)
        cnt["wins"] += (mask & (pnl > 0)).sum(1)
        cnt["gross_win"] += np.where(pnl > 0, pnl, 0).sum(1)
        cnt["gross_loss"] += np.where(pnl < 0, -pnl, 0).sum(1)
        for a in (qty, entry, stop, peak, margin, liq):
            a[mask] = 0.0
        held[mask] = 0

    for t in range(i0, i1):
        if t in marks:
            snaps[t] = {k: v.copy() for k, v in cnt.items()}
        o, h, l, c = mkt.O[t], mkt.H[t], mkt.L[t], mkt.C[t]
        isL, isS = qty > 0, qty < 0

        # 1. funding (long pays positive rate)
        fr = mkt.FUND[t]
        if fr.any():
            pay = (qty * o * fr).sum(1)
            wallet = wallet - pay
            cnt["funding"] += pay

        # 2. liquidation / stops (levels from bar t-1)
        liqL = isL & ((o <= liq) | ((l <= liq) & (stop <= liq)))
        liqS = isS & ((o >= liq) | ((h >= liq) & (stop >= liq)))
        lq = liqL | liqS
        if lq.any():
            wallet = wallet - np.where(lq, margin, 0).sum(1)
            cnt["liqs"] += lq.sum(1); cnt["trades"] += lq.sum(1)
            cnt["gross_loss"] += np.where(lq, margin, 0).sum(1)
            for a in (qty, entry, stop, peak, margin, liq):
                a[lq] = 0.0
            held[lq] = 0
            isL, isS = qty > 0, qty < 0
        stL = isL & (l <= stop)
        stS = isS & (h >= stop)
        close(stL, np.minimum(stop, o) * (1 - slip))
        close(stS, np.maximum(stop, o) * (1 + slip))

        # 3. signal / time exits at close
        k = sig
        held += (qty != 0)
        timeout = held >= max_hold
        xl = (qty > 0) & (sigs["XL"][t][k] | timeout)
        xs = (qty < 0) & (sigs["XS"][t][k] | timeout)
        close(xl, np.broadcast_to(c * (1 - slip), (B, S)))
        close(xs, np.broadcast_to(c * (1 + slip), (B, S)))

        # 4. trail ratchet
        a_t = atr[t]
        isL, isS = qty > 0, qty < 0
        if (isL | isS).any():
            peak = np.where(isL, np.maximum(peak, c), np.where(isS, np.minimum(peak, c), peak))
            trl = trail & ~np.isnan(a_t)
            stop = np.where(isL & trl, np.maximum(stop, peak - stop_k * a_t), stop)
            stop = np.where(isS & trl, np.minimum(stop, peak + stop_k * a_t), stop)

        # 5. risk layer
        eq = wallet + (qty * (c - entry)).sum(1)
        if mkt.day[t] != last_day:
            last_day = mkt.day[t]
            day_eq = eq.copy(); blocked[:] = False
        peak_eq = np.maximum(peak_eq, eq)
        dk = ~dead & ~blocked & (eq < day_eq * (1 - dkill))
        ddk = ~dead & (eq < peak_eq * (1 - ddkill))
        kill = dk | ddk
        if kill.any():
            km = kill[:, None] & (qty != 0)
            close(km & (qty > 0), np.broadcast_to(c * (1 - slip), (B, S)))
            close(km & (qty < 0), np.broadcast_to(c * (1 + slip), (B, S)))
            cnt["daily_kills"] += dk
            blocked |= dk; dead |= ddk
            eq = wallet + (qty * (c - entry)).sum(1)

        # 6. entries
        can = ~blocked & ~dead & (t >= bot_start)
        flat = qty == 0
        el = sigs["EL"][t][k] & flat & can[:, None]
        es = sigs["ES"][t][k] & flat & can[:, None]
        if el.any() or es.any():
            n_open = (~flat).sum(1)
            gross = (np.abs(qty) * c).sum(1)
            used = margin.sum(1)
            for s in range(S):
                want = (el[:, s] | es[:, s]) & (n_open < max_pos)
                if np.isnan(a_t[s]) or not want.any():
                    continue
                d = np.where(el[:, s], 1.0, -1.0)
                dist = bots["stop_k"] * a_t[s]
                notional = eq * risk / dist * c[s]
                notional = np.minimum(notional, max_gross * eq - gross)
                lev = np.clip(np.floor(1 / (dist / c[s] + mmr[s] + LIQ_BUFFER)), 1, max_lev)
                notional = np.minimum(notional, (eq - used) * lev * 0.95)
                want &= notional >= MIN_NOTIONAL
                if not want.any():
                    continue
                px = c[s] * (1 + d * slip[s])
                fee = notional * TAKER_FEE
                wallet = wallet - np.where(want, fee, 0)
                cnt["fees"] += np.where(want, fee, 0)
                # only touch bots that enter; others may hold open positions on s
                qty[want, s] = (d * notional / px)[want]
                entry[want, s] = px[want]
                stop[want, s] = (px - d * dist)[want]
                peak[want, s] = px[want]
                margin[want, s] = (notional / lev)[want]
                liq[want, s] = (px * (1 - d * (1 / lev - mmr[s])))[want]
                held[want, s] = 0
                n_open += want; gross += np.where(want, notional, 0); used += np.where(want, notional / lev, 0)

        cnt["bars_in_mkt"] += (qty != 0).any(1)
        eq_hist[t - i0] = wallet + (qty * (c - entry)).sum(1)

    if i1 in marks:
        snaps[i1] = {k: v.copy() for k, v in cnt.items()}
    state = dict(qty=qty, wallet=wallet, dead=dead, blocked=blocked)
    return dict(eq=eq_hist, cnt=cnt, snaps=snaps, state=state)


def metrics(mkt, res, i0, j0, j1, bar_ms):
    """Metrics for window [j0, j1) of a run that started at i0."""
    eq = res["eq"][j0 - i0:j1 - i0]
    base = res["eq"][j0 - i0 - 1] if j0 > i0 else np.full(eq.shape[1], START_EQUITY)
    full = np.vstack([base[None], eq])
    days = mkt.day[j0:j1]
    last_of_day = np.r_[np.nonzero(np.diff(days))[0], len(days) - 1] + 1  # +1: row 0 is base
    d_eq = full[np.r_[0, last_of_day]]
    r = d_eq[1:] / d_eq[:-1] - 1
    sd = r.std(0, ddof=1) if len(r) > 1 else np.full(eq.shape[1], np.nan)
    sharpe = np.where(sd > 0, r.mean(0) / np.where(sd > 0, sd, 1) * np.sqrt(365), 0.0)
    span_days = (j1 - j0) * bar_ms / 86_400_000
    tot = full[-1] / full[0] - 1
    cagr = np.where(full[-1] > 0, (full[-1] / full[0]) ** (365 / max(span_days, 1e-9)) - 1, -1.0)
    runmax = np.maximum.accumulate(full, 0)
    mdd = ((runmax - full) / runmax).max(0)
    s0 = res["snaps"].get(j0) if j0 > i0 else {k: np.zeros_like(v) for k, v in res["cnt"].items()}
    s1 = res["snaps"].get(j1, res["cnt"])
    dc = {k: s1[k] - s0[k] for k in s1}
    return dict(ret=tot, cagr=cagr, sharpe=sharpe, mdd=mdd, trades=dc["trades"], wins=dc["wins"],
                liqs=dc["liqs"], fees=dc["fees"], funding=dc["funding"],
                pf=np.where(dc["gross_loss"] > 0, dc["gross_win"] / np.maximum(dc["gross_loss"], 1e-9), np.inf),
                exposure=dc["bars_in_mkt"] / max(j1 - j0, 1), dead=res["state"]["dead"].copy())
