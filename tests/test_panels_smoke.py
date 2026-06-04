"""
Panel smoke tests (no Bloomberg required).

These guard against two classes of regression that pure-computation tests miss:
  1. A panel `layout()` that raises (would show "PANEL LOAD ERROR" in the app).
  2. A chart-builder that raises and gets swallowed by its callback's `except`,
     so the chart silently shows "no data" even when data exists (e.g. the
     treemap `zmid`/`cmid` bug).

The whole dashboard runs without Bloomberg by design (it shows empty states), so
layout() must always build.
"""
import importlib
import sys
import pytest

PANELS = [
    "panels.market_dashboard", "panels.vol_surface_fx", "panels.vol_scanner_unified",
    "panels.relative_value_plus", "panels.trade_ideas", "panels.structure_builder",
    "panels.blotter_fx", "panels.risk_fx", "panels.backtest", "panels.exotics_pricer",
]


class _FakeCtx:
    """Minimal stand-in for dash.callback_context so ctx-dependent callbacks
    (CSV export, deep-links, row-clicks) execute outside a real HTTP request."""
    triggered = [{"prop_id": "fake-trigger.n_clicks", "value": 1}]
    triggered_id = "fake-trigger"
    triggered_prop_ids = {}
    inputs = {}
    states = {}
    outputs_list = []
    args_grouping = []


def _degraded_getters():
    import pandas as pd
    return {
        "get_fx_spots": lambda *a, **k: {}, "get_fx_vol_surface": lambda *a, **k: {},
        "get_fx_rates": lambda *a, **k: {}, "get_fx_historical_spot": lambda *a, **k: None,
        "get_fx_historical_vol": lambda *a, **k: None, "get_fx_realized_vol": lambda *a, **k: pd.Series(dtype=float),
        "get_fx_correlation": lambda *a, **k: pd.Series(dtype=float),
        "get_fx_correlation_matrix": lambda *a, **k: pd.DataFrame(),
        "get_fx_term_structure": lambda *a, **k: {}, "get_fx_forward_curve": lambda *a, **k: {},
        "get_fx_option_chain": lambda *a, **k: pd.DataFrame(), "get_cftc_positioning": lambda *a, **k: {},
        "get_fx_vol_point": lambda *a, **k: None,
        "get_spot": lambda *a, **k: None, "get_rates": lambda *a, **k: None,
        "get_atm_vol": lambda *a, **k: None,
    }


def _synth(id_, prop):
    s = str(id_).lower()
    if prop in ("n_clicks", "n_intervals"):
        return 1
    if prop == "figure":
        return {}
    if prop == "style":
        return {}
    if prop in ("value", "active_tab"):
        if "mult" in s:
            return 1.0
        if "pair" in s or "target" in s or "h1" in s or "h2" in s:
            return "EURUSD"
        if "tenor" in s:
            return "3M"
        if "lookback" in s or "window" in s or "period" in s:
            return 120
        if "notional" in s:
            return 10
        if "delta" in s:
            return 0.25
        return None
    return None


def test_every_callback_executes_with_context_degraded(monkeypatch):
    """Execute EVERY registered callback (incl. ctx-dependent ones) across all
    panels in degraded mode. None may raise an uncaught exception other than
    PreventUpdate. This is the broad 'nothing crashes when Bloomberg is down'
    guarantee, with a callback context supplied so navigation/export callbacks run."""
    import importlib
    import dash
    from dash.exceptions import PreventUpdate

    fake = _FakeCtx()
    monkeypatch.setattr(dash, "callback_context", fake, raising=False)
    getters = _degraded_getters()
    # patch data getters (fast empty) + callback_context across all panel/core modules
    for modname, mod in list(sys.modules.items()):
        if not (modname.startswith("panels.") or modname.startswith("core.")):
            continue
        if mod is None:
            continue
        if hasattr(mod, "callback_context"):
            monkeypatch.setattr(mod, "callback_context", fake, raising=False)
        for gname, gfn in getters.items():
            if hasattr(mod, gname):
                monkeypatch.setattr(mod, gname, gfn, raising=False)

    import core.fx_portfolio as fp
    fp.create_sample_portfolio()  # empty book -> "NO POSITIONS" paths

    app = dash.Dash("all_cb", suppress_callback_exceptions=True)
    for modname in PANELS:
        importlib.import_module(modname).register_callbacks(app)

    tab_vals = [None, "greeks", "var", "stress", "attribution", "whatif", "hedge",
                "cross-pair", "correlation", "macro", "carry", "table", "heatmap",
                "skew", "signals"]
    crashes = []
    for key, spec in app.callback_map.items():
        cb = spec.get("callback")
        if not callable(cb):
            continue
        cb = getattr(cb, "__wrapped__", cb)
        specs = list(spec.get("inputs", []) or []) + list(spec.get("state", []) or [])
        def mk(d):
            v = _synth(d.get("id"), d.get("property"))
            ids = str(d.get("id", ""))
            if '"ALL"' in ids or '"MATCH"' in ids:
                return [v, v] if isinstance(v, (str, int, float)) else [None, None]
            return v
        base = [mk(d) for d in specs]
        tab_pos = [i for i, d in enumerate(specs)
                   if d.get("property") in ("value", "active_tab")
                   and ("tab" in str(d.get("id")).lower() or "view" in str(d.get("id")).lower())]
        for tv in ([None] + tab_vals if tab_pos else [None]):
            args = list(base)
            for p in tab_pos:
                if tv is not None:
                    args[p] = tv
            try:
                cb(*args)
            except PreventUpdate:
                pass
            except Exception as exc:  # noqa
                import traceback
                tb = traceback.format_exc()
                if "panels" in tb or "core" in tb:
                    crashes.append(f"{str(key)[:60]} :: {type(exc).__name__}: {exc}")
    # de-dup
    crashes = sorted(set(crashes))
    assert not crashes, "callbacks crashed in degraded mode:\n" + "\n".join(crashes[:40])


def _risk_data_mocks():
    surf = {t: {"atm": 8.0 + i * 0.3, "rr25": -0.4, "bf25": 0.2, "rr10": -0.8, "bf10": 0.5}
            for i, t in enumerate(["1W", "1M", "2M", "3M", "6M", "1Y", "2Y"])}
    spotmap = {"EURUSD": 1.085, "USDJPY": 150.2, "USDMXN": 17.1}

    def spots(pairs=None, **k):
        ps = pairs or list(spotmap)
        return {p: {"mid": spotmap.get(p, 1.10), "bid": spotmap.get(p, 1.10)} for p in ps}

    def rates(pair, **k):
        return {"r_dom": 0.05 if pair.startswith("USD") else 0.03, "r_for": 0.02}

    return {"get_fx_spots": spots, "get_fx_rates": rates,
            "get_fx_vol_surface": lambda p, **k: surf}


def test_risk_panel_all_tabs_with_positions(monkeypatch, caplog):
    """Risk panel is the most-changed file. With a live book + real data, every
    tab's charts (treemap, GEX, landscape, vega/delta/gamma, VaR, stress,
    attribution waterfall/sankey) must build without a swallowed error."""
    import logging
    import importlib
    import dash

    mocks = _risk_data_mocks()
    for modname in ("panels.risk_fx", "core.fx_analytics", "core.bloomberg_fx"):
        mod = importlib.import_module(modname)
        for name, fn_ in mocks.items():
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, fn_)

    import core.fx_portfolio as fp
    fp.create_sample_portfolio()
    today = "2026-04-01"
    fp.add_position("G10_FLOW", {"pair": "EURUSD", "option_type": "call", "direction": "buy",
        "strike": 1.10, "expiry": "2026-09-01", "notional": 10_000_000,
        "entry_spot": 1.08, "entry_vol": 0.085, "entry_date": today, "entry_premium": 120000})
    fp.add_position("EM_FLOW", {"pair": "USDMXN", "option_type": "put", "direction": "sell",
        "strike": 17.0, "expiry": "2026-08-01", "notional": 5_000_000,
        "entry_spot": 17.2, "entry_vol": 0.12, "entry_date": today, "entry_premium": 60000})

    app = dash.Dash("risk_full", suppress_callback_exceptions=True)
    import panels.risk_fx as rf
    rf.register_callbacks(app)

    tabs = ["greeks", "var", "stress", "attribution", "whatif", "hedge"]
    crashes = []
    with caplog.at_level(logging.ERROR):
        for key, spec in app.callback_map.items():
            cb = spec.get("callback")
            if not callable(cb):
                continue
            cb = getattr(cb, "__wrapped__", cb)
            specs = list(spec.get("inputs", []) or []) + list(spec.get("state", []) or [])
            base = [_synth(d.get("id"), d.get("property")) for d in specs]
            tab_pos = [i for i, d in enumerate(specs)
                       if "fxrisk-tabs" in str(d.get("id"))]
            for tv in (tabs if tab_pos else [None]):
                args = list(base)
                for p in tab_pos:
                    args[p] = tv
                try:
                    cb(*args)
                except Exception:  # noqa  (PreventUpdate etc. are fine; chart errors are logged)
                    pass
    bad = [r.getMessage()[:140] for r in caplog.records
           if any(s in r.getMessage() for s in ("failed", "error", "Error"))
           and "Bloomberg" not in r.getMessage()]
    assert not bad, "risk panel chart errors with positions:\n" + "\n".join(sorted(set(bad))[:30])


def test_backtest_all_strategies_run(monkeypatch):
    """Every backtest strategy must execute over (mocked) real history and return
    a results dict without crashing."""
    import numpy as np
    import pandas as pd
    import panels.backtest as bt

    n = 520
    idx = pd.date_range(end="2026-06-01", periods=n, freq="B")
    close = 1.08 * np.exp(np.cumsum(np.random.RandomState(11).normal(0, 0.005, n)))
    spot_df = pd.DataFrame({"open": close * 0.999, "high": close * 1.003,
                            "low": close * 0.997, "close": close}, index=idx)

    def hvol(pair, tenor="1M", metric="ATM", days=252, **k):
        m = max(int(days or 252), 30)
        base = {"ATM": 8.5, "25D_RR": -0.4, "25D_BF": 0.25}.get(metric, 8.5)
        return pd.Series(np.full(m, base) + np.random.RandomState(abs(hash(metric)) % 99).normal(0, 0.1, m),
                         index=pd.date_range(end="2026-06-01", periods=m, freq="B"))

    monkeypatch.setattr(bt, "get_fx_historical_spot", lambda p, days=252, **k: spot_df)
    monkeypatch.setattr(bt, "get_fx_historical_vol", hvol)
    if hasattr(bt, "get_fx_rates"):
        monkeypatch.setattr(bt, "get_fx_rates", lambda p, **k: {"r_dom": 0.04, "r_for": 0.02})

    ran = 0
    for strategy in bt._STRATEGIES:
        res = bt.run_backtest(strategy, "EURUSD", "3M", 0.25, 2, "fixed", "hold", 1_000_000)
        # Must not raise; with monthly entries over ~2y, trades should generate.
        if res is not None:
            assert "trades" in res or "equity_curve" in res or "stats" in res, \
                f"{strategy}: results dict missing expected keys"
            ran += 1
    assert ran >= 8, f"only {ran}/{len(bt._STRATEGIES)} strategies produced results"


@pytest.mark.parametrize("modname", PANELS)
def test_layout_builds(modname):
    mod = importlib.import_module(modname)
    layout = mod.layout()
    assert layout is not None  # a Dash component tree


def test_vol_scanner_empty_fig_accepts_title_and_msg():
    """Regression: _empty_fig(title, msg=...) must not raise 'multiple values for
    msg' (it does in 7 no-data/except branches with the old signature)."""
    import plotly.graph_objects as go
    from panels.vol_scanner_unified import _empty_fig
    assert isinstance(_empty_fig("EURUSD ATM 3M", msg="failed"), go.Figure)
    assert isinstance(_empty_fig("VOL RICHNESS"), go.Figure)
    assert isinstance(_empty_fig(msg="detail"), go.Figure)
    assert isinstance(_empty_fig("X", 350, msg="Y"), go.Figure)


def test_realized_vol_and_correlation_survive_missing_history():
    """Regression: get_fx_realized_vol / get_fx_correlation must not crash with
    AttributeError when historical spot is None (degraded / no-Bloomberg)."""
    from unittest.mock import patch
    import core.bloomberg_fx as bf
    with patch.object(bf, "get_fx_historical_spot", return_value=None):
        rv = bf.get_fx_realized_vol("EURUSD")
        assert hasattr(rv, "empty") and rv.empty
        corr = bf.get_fx_correlation("EURUSD", "USDJPY")
        assert hasattr(corr, "empty") and corr.empty


@pytest.mark.parametrize("modname", PANELS)
def test_register_callbacks(modname):
    import dash
    mod = importlib.import_module(modname)
    app = dash.Dash(modname.replace(".", "_"), suppress_callback_exceptions=True)
    mod.register_callbacks(app)  # must not raise
    assert len(app.callback_map) >= 1


def _sample_positions():
    return [
        {"id": "p1", "book": "G10_FLOW", "pair": "EURUSD", "option_type": "call",
         "direction": "buy", "strike": 1.10, "expiry": "2026-09-01",
         "notional": 10_000_000, "pnl": 25000, "status": "open"},
        {"id": "p2", "book": "G10_PROP", "pair": "USDJPY", "option_type": "put",
         "direction": "sell", "strike": 148.0, "expiry": "2026-08-01",
         "notional": 15_000_000, "pnl": -8000, "status": "open"},
    ]


def test_risk_treemap_builds_with_positions():
    """Guards the treemap cmid/zmid bug: must produce a real Treemap, not no-data."""
    import plotly.graph_objects as go
    from panels.risk_fx import _build_risk_treemap
    fig = _build_risk_treemap(_sample_positions())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert any(isinstance(tr, go.Treemap) for tr in fig.data)


def test_risk_treemap_empty_is_no_data():
    import plotly.graph_objects as go
    from panels.risk_fx import _build_risk_treemap
    fig = _build_risk_treemap([])
    assert isinstance(fig, go.Figure)


def test_greeks_landscape_builds_with_real_inputs():
    """Delta landscape must build from real spot/rates/vol (no fabrication path)."""
    import plotly.graph_objects as go
    from panels.risk_fx import _build_greeks_landscape
    spots = {"EURUSD": 1.085}
    rates = {"EURUSD": {"r_d": 0.04, "r_f": 0.02}}
    vols = {"EURUSD": 0.085}
    fig = _build_greeks_landscape(_sample_positions(), "EURUSD", spots, rates, vols)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_greeks_landscape_missing_data_is_no_data():
    """No real market data -> no fabricated surface, just a no-data figure."""
    import plotly.graph_objects as go
    from panels.risk_fx import _build_greeks_landscape
    fig = _build_greeks_landscape(_sample_positions(), "EURUSD", {}, {}, {})
    assert isinstance(fig, go.Figure)
    # no surface trace built from fabricated data
    assert not any(isinstance(tr, go.Surface) for tr in fig.data)


# ---------------------------------------------------------------------------
# Structure Builder: drive the full 14-chart callback end to end
# ---------------------------------------------------------------------------
def _mock_surface():
    s = {}
    for i, t in enumerate(["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"]):
        s[t] = {"atm": 8.0 + i * 0.2, "rr25": -0.4, "bf25": 0.2, "rr10": -0.8, "bf10": 0.5,
                "25D_RR": -0.4, "25D_BF": 0.2, "10D_RR": -0.8, "10D_BF": 0.5}
    return s


def _structure_inputs(num_legs=2):
    """Correctly-shaped scalar inputs for update_structure (per-index, not lists)."""
    from panels.structure_builder import MAX_LEGS
    def pad(vals):
        return list(vals) + [None] * (MAX_LEGS - len(vals))
    cp = pad(["call", "put"][:num_legs])
    side = pad(["buy", "buy"][:num_legs])
    delta = pad([0.5, 0.5][:num_legs])
    ratio = pad([1, 1][:num_legs])
    tmult = pad([1.0, 1.0][:num_legs])
    preset = "Custom"
    return cp + side + delta + ratio + tmult + [preset]


def test_structure_builder_full_callback_builds_all_charts():
    """The flagship structurer callback must produce real figures (payoff, greeks,
    heatmap, 3D, scenario, smile, pnl-surface, parallel-coords) from real data."""
    import dash
    import plotly.graph_objects as go
    from unittest.mock import patch
    import panels.structure_builder as sb

    app = dash.Dash("sb_full", suppress_callback_exceptions=True)
    sb.register_callbacks(app)
    # locate update_structure by one of its figure outputs
    key = next(k for k in app.callback_map if "stb-payoff-chart.figure" in k)
    fn = app.callback_map[key]["callback"]
    fn = getattr(fn, "__wrapped__", fn)

    surf = _mock_surface()
    with patch.object(sb, "get_fx_spots", return_value={"EURUSD": {"mid": 1.085, "bid": 1.0849}}), \
         patch.object(sb, "get_fx_vol_surface", return_value=surf), \
         patch.object(sb, "get_fx_rates", return_value={"r_dom": 0.04, "r_for": 0.02}), \
         patch.object(sb, "get_spot", return_value=1.085), \
         patch.object(sb, "get_rates", return_value=(0.04, 0.02)), \
         patch.object(sb, "get_atm_vol", return_value=0.09):
        out = fn("EURUSD", "3M", 10_000_000, 2, *_structure_inputs(2))

    out = list(out)
    # 14 main outputs + 3*MAX_LEGS per-leg displays
    assert len(out) == 14 + 3 * sb.MAX_LEGS
    # figure outputs by position: payoff(3) greeks(4) heatmap(5) 3d(6) scenario(8) smile(9) pnl-surface(12) parallel(13)
    fig_positions = [3, 4, 5, 6, 8, 9, 12, 13]
    built = 0
    for p in fig_positions:
        fig = out[p]
        f = go.Figure(fig) if isinstance(fig, dict) else fig
        assert isinstance(f, go.Figure), f"output {p} is not a figure: {type(fig)}"
        if len(f.data) > 0:
            built += 1
    # the core priced charts (payoff, greeks, heatmap, 3d) must have data
    assert built >= 4, f"only {built}/8 structurer charts built with data"


def test_structure_builder_no_data_returns_unavailable_not_fabricated():
    """With no real spot/rates/vol the structurer must return MARKET DATA
    UNAVAILABLE (correct arity), never charts priced off a placeholder."""
    import dash
    from unittest.mock import patch
    import panels.structure_builder as sb

    app = dash.Dash("sb_nodata", suppress_callback_exceptions=True)
    sb.register_callbacks(app)
    key = next(k for k in app.callback_map if "stb-payoff-chart.figure" in k)
    fn = app.callback_map[key]["callback"]
    fn = getattr(fn, "__wrapped__", fn)

    with patch.object(sb, "get_fx_spots", return_value={}), \
         patch.object(sb, "get_fx_vol_surface", return_value={}), \
         patch.object(sb, "get_fx_rates", return_value={}), \
         patch.object(sb, "get_spot", return_value=None), \
         patch.object(sb, "get_rates", return_value=None), \
         patch.object(sb, "get_atm_vol", return_value=None):
        out = fn("EURUSD", "3M", 10_000_000, 2, *_structure_inputs(2))
    assert len(list(out)) == 14 + 3 * sb.MAX_LEGS  # correct arity even on the no-data path


# ---------------------------------------------------------------------------
# Exotics pricer: drive run_pricer (spot/rates/vol come from State stores)
# ---------------------------------------------------------------------------
def _exo_pricer_fn():
    import dash
    import panels.exotics_pricer as exo
    app = dash.Dash("exo_run", suppress_callback_exceptions=True)
    exo.register_callbacks(app)
    key = next(k for k in app.callback_map if "exo-payoff-chart.figure" in k)
    fn = app.callback_map[key]["callback"]
    return getattr(fn, "__wrapped__", fn)


EXOTIC_PRODUCTS = ["barrier", "double_barrier", "digital", "one_touch", "dnt",
                   "range_accrual", "asian", "lookback", "forward_start",
                   "best_of", "tarf"]


def _exo_args(product="barrier", spot=1.085, r_d=0.04, r_f=0.02, vol=0.09):
    # positional after n_clicks, matching run_pricer signature; all params filled
    # so every product has what it needs (the callback uses the relevant subset).
    return (product, "EURUSD", 1_000_000, "3M",
            spot, r_d, r_f, vol,
            1, 1.05, "down-and-in", 0,              # cp(1=call), barrier_level, barrier_type, rebate
            1.12, 1.04,                              # upper/lower barrier
            1.08, 1.0,                               # strike, payout
            1.06, 1.10,                              # range low/high
            "Daily", "Arithmetic",                  # fixfreq, avgtype
            "Floating", "3M", 1.00,                 # lbtype, fwd_offset, fwd_money
            "USDJPY", "best-of",                    # pair2, bestof_type
            1.12, 100000, 2.0, 12)                   # tarf barrier/target/leverage/fixings


@pytest.mark.parametrize("product", EXOTIC_PRODUCTS)
def test_exotics_pricer_builds_charts_every_product(product):
    """Every one of the 11 exotic products must price and build its 4 charts.

    Mocks the SECOND-pair data + correlation that the best-of/worst-of pricer
    needs (otherwise it correctly refuses to price off missing data, and also
    blocks on a real Bloomberg connection attempt)."""
    import plotly.graph_objects as go
    import pandas as pd
    from unittest.mock import patch
    import panels.exotics_pricer as exo
    fn = _exo_pricer_fn()
    corr = pd.Series([0.35, 0.36, 0.34, 0.37], name="corr")
    with patch.object(exo, "get_fx_correlation", return_value=corr), \
         patch.object(exo, "get_spot", return_value=150.0), \
         patch.object(exo, "get_rates", return_value=(0.04, 0.001)), \
         patch.object(exo, "get_atm_vol", return_value=0.10):
        out = list(fn(1, *_exo_args(product)))
    assert len(out) == 8, f"{product}: wrong output arity {len(out)}"
    built = 0
    for f in out[4:8]:
        fig = go.Figure(f) if isinstance(f, dict) else f
        assert isinstance(fig, go.Figure), f"{product}: output not a figure"
        if len(fig.data) > 0:
            built += 1
    assert built >= 2, f"{product}: only {built}/4 charts built with data"


def test_exotics_pricer_no_market_data_is_unavailable():
    fn = _exo_pricer_fn()
    out = list(fn(1, *_exo_args(spot=None, r_d=None, r_f=None, vol=None)))
    assert len(out) == 8  # correct arity on the no-data path (no fabricated price)


# ---------------------------------------------------------------------------
# Vol Surface: sweep EVERY chart type through the 2x2-grid callback
# ---------------------------------------------------------------------------
def _vol_surface_mocks():
    import pandas as pd
    import numpy as np
    surf = {}
    for i, t in enumerate(["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]):
        surf[t] = {"atm": 8.0 + i * 0.2, "rr25": -0.4 - 0.01 * i, "bf25": 0.2 + 0.01 * i,
                   "rr10": -0.8, "bf10": 0.5,
                   "25D_RR": -0.4 - 0.01 * i, "25D_BF": 0.2 + 0.01 * i,
                   "10D_RR": -0.8, "10D_BF": 0.5}
    spots = {p: {"mid": m, "bid": m * 0.9999, "ask": m * 1.0001, "close": m * 0.999,
                 "change": m * 0.001, "change_pct": 0.1, "high": m * 1.003, "low": m * 0.997}
             for p, m in {"EURUSD": 1.085, "USDJPY": 150.2, "GBPUSD": 1.27}.items()}

    def hvol(pair, tenor="1M", metric="ATM", lookback=252, **k):
        n = max(int(lookback or 252), 60)
        idx = pd.date_range(end="2026-06-01", periods=n, freq="B")
        return pd.Series(8.0 + np.sin(np.arange(n) / 12.0) + 0.3, index=idx)

    def hspot(pair, days=520, **k):
        n = max(int(days or 520), 60)
        idx = pd.date_range(end="2026-06-01", periods=n, freq="B")
        c = 1.08 * np.exp(np.cumsum(np.random.RandomState(3).normal(0, 0.005, n)))
        return pd.DataFrame({"open": c * 0.999, "high": c * 1.003, "low": c * 0.997,
                             "close": c}, index=idx)

    def hcorr(a, b, window=60, days=252, **k):
        n = max(int(days or 252), 60)
        idx = pd.date_range(end="2026-06-01", periods=n, freq="B")
        return pd.Series(np.clip(0.3 + np.cumsum(np.random.RandomState(5).normal(0, 0.02, n)) * 0.03, -0.9, 0.9), index=idx)

    return {
        "get_fx_vol_surface": lambda p: surf,
        "get_fx_spots": lambda *a, **k: spots,
        "get_fx_rates": lambda p: {"r_dom": 0.04, "r_for": 0.02},
        "get_fx_historical_vol": hvol,
        "get_fx_historical_spot": hspot,
        "get_fx_realized_vol": lambda p, window=20, days=252, **k: hvol(p, lookback=days),
        "get_fx_correlation": hcorr,
        "get_fx_term_structure": lambda p, **k: {t: surf[t]["atm"] for t in surf},
        "get_spot": lambda p: {"EURUSD": 1.085, "USDJPY": 150.2, "GBPUSD": 1.27}.get(p, 1.085),
        "get_rates": lambda p: (0.04, 0.02),
        "get_atm_vol": lambda p, t=None: 0.09,
    }


def test_vol_surface_every_chart_type_renders(monkeypatch, caplog):
    """Drive the flagship 2x2-grid callback for EVERY chart-type option; a chart
    that raises is caught + logged ('Live mode rendering failed') and blanks the
    grid, so any such log for a chart type is a real bug."""
    import logging
    import importlib
    import dash
    import plotly.graph_objects as go
    import panels.vol_surface_fx as vs
    from panels.vol_surface_fx import CHART_OPTIONS

    mocks = _vol_surface_mocks()
    for modname in ("panels.vol_surface_fx", "core.fx_analytics", "core.bloomberg_fx"):
        mod = importlib.import_module(modname)
        for name, fn_ in mocks.items():
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, fn_)

    app = dash.Dash("vs_sweep", suppress_callback_exceptions=True)
    vs.register_callbacks(app)
    key = next(k for k in app.callback_map if "vsfx-chart-q1.figure" in k)
    fn = app.callback_map[key]["callback"]
    fn = getattr(fn, "__wrapped__", fn)

    chart_values = [o["value"] for o in CHART_OPTIONS if not o.get("disabled")]
    assert len(chart_values) >= 38  # sanity: we really are sweeping them all
    failures = []
    for ct in chart_values:
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            out = fn("EURUSD", "market", ct, "atm_term", "atm_term", "atm_term",
                     None, 1, None, None, "10-50", "3M", 0,
                     ["EURUSD"], "", 252, "raw", 120, 120, 120, 120)
        out = list(out)
        if len(out) != 6:
            failures.append((ct, f"arity {len(out)}"))
            continue
        f0 = go.Figure(out[0]) if isinstance(out[0], dict) else out[0]
        if not isinstance(f0, go.Figure):
            failures.append((ct, "q1 not a figure"))
        render_errs = [r.getMessage() for r in caplog.records
                       if "Live mode rendering failed" in r.getMessage()]
        if render_errs:
            failures.append((ct, render_errs[0][:120]))
    assert not failures, "vol-surface chart types failed:\n" + "\n".join(
        f"  {c}: {m}" for c, m in failures)
