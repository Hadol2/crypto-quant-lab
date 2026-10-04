"""Hand-computed engine checks. Run: python3 -m arena.tests.test_engine"""
import numpy as np

from arena import engine
from arena.engine import Market, run


def _mkt(o, h, l, c, fund=None):
    T = len(o)
    t = np.arange(T, dtype=np.int64) * 3_600_000 + 1_700_000_000_000
    col = lambda a: np.asarray(a, float)[:, None]
    f = np.zeros((T, 1)) if fund is None else col(fund)
    m = Market(t, col(o), col(h), col(l), col(c), f, ["BTCUSDT"])
    m.slip = np.zeros(1)
    return m


def _bot(**kw):
    b = dict(sig=np.array([0]), stop_k=np.array([kw.get("stop_k", 1.0)]), risk=np.array([0.01]),
             max_lev=np.array([kw.get("max_lev", 1.0)]), max_gross=np.array([10.0]), max_pos=np.array([1.0]),
             daily_kill=np.array([0.99]), dd_kill=np.array([0.99]), trail=np.array([kw.get("trail", True)]),
             max_hold=np.array([10**9]))
    return b


def _sigs(T, enter_at, short=False):
    s = {n: np.zeros((T, 1, 1), bool) for n in ("EL", "ES", "XL", "XS")}
    s["ES" if short else "EL"][enter_at, 0, 0] = True
    return s


def _atr(m, v):
    m._cache[("atr", 14)] = __import__("pandas").DataFrame(np.full((len(m.t), 1), float(v)))


def test_causal_stop_not_ratcheted_before_check():
    # enter long at 100 (bar 0), ATR 10, stop_k 3 -> stop 70.
    # bar 1: low 95, close 140. Causal: 95 > 70 so hold; then stop ratchets to 140-30=110.
    m = _mkt([100, 100, 139], [100, 140, 139], [100, 95, 105], [100, 140, 106])
    _atr(m, 10)
    r = run(m, _bot(stop_k=3.0), _sigs(3, 0), 0, 3)
    assert r["cnt"]["trades"][0] == 1
    # bar 2 low 105 <= stop 110 -> fill at min(110, open 139) = 110
    qty = 10_000 * 0.01 / 30 * 100 / 100 / 1.0  # notional/px, px=100
    notional = qty * 100
    expected = 10_000 - notional * engine.TAKER_FEE + qty * (110 - 100) - qty * 110 * engine.TAKER_FEE
    assert abs(r["eq"][-1, 0] - expected) < 1e-6, (r["eq"][-1, 0], expected)


def test_gap_through_stop_fills_at_open():
    m = _mkt([100, 90], [100, 91], [100, 85], [100, 88])
    _atr(m, 5)
    r = run(m, _bot(stop_k=1.0), _sigs(2, 0), 0, 2)  # stop 95, bar1 opens 90
    qty = 10_000 * 0.01 / 5
    expected = 10_000 - qty * 100 * engine.TAKER_FEE + qty * (90 - 100) - qty * 90 * engine.TAKER_FEE
    assert abs(r["eq"][-1, 0] - expected) < 1e-6


def test_liquidation_on_gap_loses_margin():
    # stop 1 ATR=2 -> dist 2%. lev = floor(1/(0.02+0.004+0.005)) = 34 -> capped by max_lev 20.
    # liq = 100*(1 - (1/20 - 0.004)) = 95.4. bar1 opens 90 (gap past liq) -> liquidated.
    m = _mkt([100, 90], [100, 91], [100, 89], [100, 90])
    _atr(m, 2)
    r = run(m, _bot(stop_k=1.0, max_lev=20.0), _sigs(2, 0), 0, 2)
    qty = 10_000 * 0.01 / 2
    notional = qty * 100
    expected = 10_000 - notional * engine.TAKER_FEE - notional / 20
    assert r["cnt"]["liqs"][0] == 1
    assert abs(r["eq"][-1, 0] - expected) < 1e-6


def test_stop_before_liq_when_no_gap():
    # same setup, but bar1 opens 99 and trades down to 90: stop (98) triggers, not liquidation.
    m = _mkt([100, 99], [100, 99], [100, 90], [100, 91])
    _atr(m, 2)
    r = run(m, _bot(stop_k=1.0, max_lev=20.0), _sigs(2, 0), 0, 2)
    assert r["cnt"]["liqs"][0] == 0 and r["cnt"]["trades"][0] == 1


def test_funding_long_pays_short_receives():
    # hold through bar1 open where funding rate 0.001 is recorded; price flat at 100.
    for short, sign in ((False, -1), (True, +1)):
        m = _mkt([100, 100], [100, 100], [100, 100], [100, 100], fund=[0, 0.001])
        _atr(m, 5)
        r = run(m, _bot(stop_k=1.0), _sigs(2, 0, short=short), 0, 2)
        qty = 10_000 * 0.01 / 5
        fee = qty * 100 * engine.TAKER_FEE
        expected = 10_000 - fee + sign * qty * 100 * 0.001
        assert abs(r["eq"][-1, 0] - expected) < 1e-6, (short, r["eq"][-1, 0], expected)


def test_holdout_guard():
    engine.assert_outside_holdout(engine.HOLDOUT_MS[0] - 10, engine.HOLDOUT_MS[0])
    engine.assert_outside_holdout(engine.HOLDOUT_MS[1], engine.HOLDOUT_MS[1] + 10)
    try:
        engine.assert_outside_holdout(engine.HOLDOUT_MS[0] - 10, engine.HOLDOUT_MS[0] + 1)
    except engine.HoldoutViolation:
        return
    raise AssertionError("holdout overlap not rejected")


def test_entry_by_one_bot_keeps_other_bots_positions():
    # bot 0 enters at bar 0, bot 1 enters at bar 1 on the same symbol; bot 0's position must survive.
    m = _mkt([100, 100, 110], [100, 100, 110], [100, 100, 110], [100, 100, 110])
    _atr(m, 5)
    b = {k: np.concatenate([v, v]) for k, v in _bot(stop_k=1.0, trail=False).items()}
    b["sig"] = np.array([0, 1])
    s = {n: np.zeros((3, 2, 1), bool) for n in ("EL", "ES", "XL", "XS")}
    s["EL"][0, 0, 0] = True
    s["EL"][1, 1, 0] = True
    r = run(m, b, s, 0, 3)
    qty = 10_000 * 0.01 / 5
    assert abs(r["eq"][-1, 0] - (10_000 - qty * 100 * engine.TAKER_FEE + qty * 10)) < 1e-6


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    raise SystemExit(fails)
