"""
Market Dashboard — DESK workspace
==================================
Replaces: Market Pulse + Morning Snapshot + Events Calendar (3 → 1).

Layout:
  ┌─────────────────────────────────────────────────────────────┐
  │ 8 KPIs: DXY | G10 Vol | EM Vol | Risk Sentiment | Biggest  │
  │ Mover | Book Vega | Book Theta | Events 48h                │
  ├──────────────────────────┬──────────────────────────────────┤
  │ MOVERS TABLE             │ G10 VOL INDEX (60d chart)        │
  │ 30 pairs sorted by       │                                  │
  │ |spot Δ|                 ├──────────────────────────────────┤
  │ Spot|Δ%|ATM|ΔVol|RR     │ SKEW MONITOR (25D RR bars)       │
  │ Click row → global pair  ├──────────────────────────────────┤
  │ Filter: G10/EM/ALL       │ TERM SHAPE (1M-1Y spread bars)   │
  ├──────────────────────────┼──────────────────────────────────┤
  │ BOOK SUMMARY             │ EVENTS + POSITIONING EXTREMES    │
  │ Net Δ by pair (bars)     │ Next 5 CB meetings + |z|>1.2     │
  └──────────────────────────┴──────────────────────────────────┘
"""

import logging
import numpy as np
from scipy.stats import percentileofscore
from dash import html, dcc, Input, Output, State, callback_context, ALL, MATCH, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, TAB_STYLE, TAB_SELECTED_STYLE,
    TABLE_HEADER_STYLE, TABLE_CELL_STYLE, clickable_stat, make_stat_style,
    GAP, SECTION_GAP, CHART_SM, CHART_MD, CHART_LG, CSV_BTN_STYLE,
    no_data_fig,
)
from core.csv_export import export_csv
from core.fx_conventions import FX_PAIR_REGISTRY, tenor_to_days

# ── Constants ────────────────────────────────────────────────────────────────

_P = "mdash"  # prefix for all IDs

_MONO = "'JetBrains Mono', monospace"
_CHART_H = CHART_MD
_SMALL_H = CHART_SM

ALL_PAIRS = sorted(FX_PAIR_REGISTRY.keys())

G10_PAIRS = [p for p in ALL_PAIRS if FX_PAIR_REGISTRY[p].group == "G10"]
EM_PAIRS  = [p for p in ALL_PAIRS if p not in G10_PAIRS]

# Central bank calendar (static dates, rotated quarterly)
_CB_BANKS = {
    "FED":  {"pairs": ["EURUSD","USDJPY","GBPUSD","USDCHF"], "impact": "HIGH"},
    "ECB":  {"pairs": ["EURUSD","EURGBP","EURJPY","EURCHF"], "impact": "HIGH"},
    "BOE":  {"pairs": ["GBPUSD","EURGBP","GBPJPY"],          "impact": "HIGH"},
    "BOJ":  {"pairs": ["USDJPY","EURJPY","GBPJPY","AUDJPY"], "impact": "HIGH"},
    "SNB":  {"pairs": ["USDCHF","EURCHF"],                    "impact": "MED"},
    "RBA":  {"pairs": ["AUDUSD","AUDJPY","AUDNZD"],           "impact": "MED"},
    "RBNZ": {"pairs": ["NZDUSD","AUDNZD","NZDJPY"],           "impact": "LOW"},
    "BOC":  {"pairs": ["USDCAD","CADCHF","CADJPY"],            "impact": "MED"},
}

GROUP_OPTIONS = [
    {"label": "ALL", "value": "ALL"},
    {"label": "G10", "value": "G10"},
    {"label": "EM",  "value": "EM"},
]

SORT_OPTIONS = [
    {"label": "|SPOT Δ|", "value": "spot"},
    {"label": "|VOL Δ|",  "value": "vol"},
    {"label": "ATM LVL",  "value": "atm"},
    {"label": "PAIR",     "value": "pair"},
    {"label": "VOL %ILE", "value": "pctile"},
    {"label": "IV-RV",    "value": "ivrv"},
]

_previous_movers = {}  # Cache previous movers data for crossing detection


# ── Safe helpers ─────────────────────────────────────────────────────────────

def _ordinal(n):
    """Return an integer as an ordinal string: 1 -> '1st', 23 -> '23rd'."""
    n = int(n)
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _sf(v, d=0.0):
    """Safe float."""
    try:
        f = float(v)
        return d if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return d


def _fmt_spot(pair: str, spot: float) -> str:
    """Format spot price using pair convention (pip precision)."""
    from core.fx_conventions import FX_PAIR_REGISTRY
    spec = FX_PAIR_REGISTRY.get(pair.upper()) if pair else None
    if spec and spec.pip == 0.01:
        return f"{spot:.3f}"  # JPY pairs: 3 decimals
    elif spot >= 100:
        return f"{spot:.2f}"  # High-value pairs (e.g., USDJPY at 148)
    else:
        return f"{spot:.5f}"  # Standard 5-decimal pairs


def _pairs_for(group):
    if group == "G10":
        return G10_PAIRS
    if group == "EM":
        return EM_PAIRS
    return ALL_PAIRS


def _sf_display(v, fmt=".2f", suffix="", prefix=""):
    """Safe format: returns '—' for NaN/inf values."""
    if v is None or not isinstance(v, (int, float)) or not np.isfinite(v):
        return "—"
    return f"{prefix}{v:{fmt}}{suffix}"


def _color_chg(v):
    if v is None:
        return COLORS["text_muted"]
    if not np.isfinite(v):
        return COLORS["text_secondary"]
    if v > 0:
        return COLORS["accent_green"]
    if v < 0:
        return COLORS["accent_red"]
    return COLORS["text_secondary"]


def _pct_color(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return COLORS["text_primary"]
    if p < 20:
        return "#1565c0"
    if p > 80:
        return COLORS["accent_red"]
    return COLORS["text_primary"]


# ── Data builders ────────────────────────────────────────────────────────────

def _build_movers(pairs):
    """Build movers data for given pairs."""
    try:
        from core.bloomberg_fx import get_fx_spots, get_fx_vol_surface
    except Exception:
        logger.warning("Cannot import bloomberg_fx for movers")
        return []

    spots = get_fx_spots() or {}
    rows = []
    for pair in pairs:
        try:
            s = spots.get(pair, {})
            spot = _sf(s.get("mid", s.get("close", 0)))
            chg = _sf(s.get("change_pct", 0))

            surf = get_fx_vol_surface(pair) or {}
            atm_1m = _sf(surf.get("1M", {}).get("atm", 0))
            rr25 = _sf(surf.get("3M", {}).get("rr25", surf.get("3M", {}).get("25D_RR", 0)))

            # vol change (1-day ATM change)
            vol_chg_raw = surf.get("1M", {}).get("atm_change", None)
            if vol_chg_raw is not None:
                vol_chg = _sf(vol_chg_raw)
            else:
                try:
                    from core.fx_analytics import vol_change
                    vc = vol_change(pair, "1M", "ATM", 1)
                    vol_chg = _sf(vc.get("abs_change", 0) if isinstance(vc, dict) else 0)
                except Exception:
                    vol_chg = 0.0

            # 3M ATM vol
            atm_3m = _sf(surf.get("3M", {}).get("atm", 0))

            # Vol percentile (3M ATM, 252-day lookback)
            try:
                from core.fx_analytics import vol_percentile
                vp = vol_percentile(pair, "3M", "ATM", 252)
                if isinstance(vp, dict):
                    pctile = _sf(vp.get("percentile", 50), 50)
                elif vp is not None:
                    pctile = _sf(vp, 50)
                else:
                    pctile = 50.0
            except Exception:
                pctile = 50.0

            # Term spread: 1M ATM - 1Y ATM
            atm_1y = _sf(surf.get("1Y", {}).get("atm", 0))
            if atm_1m > 0 and atm_1y > 0:
                term_spread = atm_1m - atm_1y
            else:
                term_spread = 0.0

            # Breakeven pips: daily implied move = (ATM_1M / 100 / sqrt(252)) * spot
            from math import sqrt
            if atm_1m > 0 and spot > 0:
                breakeven_pips = (atm_1m / 100.0 / sqrt(252)) * spot
            else:
                breakeven_pips = 0.0

            # IV-RV spread
            try:
                from core.fx_analytics import iv_rv_spread
                df = iv_rv_spread(pair, "3M", 20, 60)
                if df is not None and not df.empty and len(df) > 0 and "spread" in df.columns:
                    val = df["spread"].iloc[-1]
                    iv_rv = _sf(float(val)) if np.isfinite(val) else 0.0
                else:
                    iv_rv = 0.0
            except Exception:
                iv_rv = 0.0

            # 5-day vol momentum: current ATM vs 5d average
            vol_mom = 0.0
            try:
                from core.fx_analytics import vol_change as _vc
                vc5 = _vc(pair, "3M", "ATM", 5)
                if isinstance(vc5, dict) and vc5.get("previous", 0) > 0:
                    vol_mom = (atm_3m / vc5["previous"] - 1) * 100
            except Exception:
                pass

            rows.append({
                "pair": pair, "spot": spot, "chg_pct": chg,
                "atm_1m": atm_1m, "vol_chg": vol_chg, "rr25": rr25,
                "atm_3m": atm_3m, "pctile": pctile,
                "term_spread": term_spread, "breakeven_pips": breakeven_pips,
                "iv_rv": iv_rv, "vol_mom": vol_mom,
            })
        except Exception:
            rows.append({"pair": pair, "spot": 0, "chg_pct": 0,
                         "atm_1m": 0, "vol_chg": 0, "rr25": 0,
                         "atm_3m": 0, "pctile": 50, "term_spread": 0,
                         "breakeven_pips": 0, "iv_rv": 0, "vol_mom": 0})
    return rows


def _build_kpi_data(rows):
    """8 KPIs from movers data."""
    from datetime import datetime
    kpis = {}

    # DXY proxy — ICE methodology (geometric weighted index)
    # Weights sum to 1.0; SEK weight redistributed to AUD/NZD
    try:
        from core.bloomberg_fx import get_fx_spots
        spots = get_fx_spots() or {}
        # ICE DXY basket weights with neutral baselines (updated Q1 2026)
        _dxy_cfg = [
            # (pair,  weight, baseline, is_usd_base)
            ("EURUSD", 0.576, 1.0500, False),   # XXX/USD — lower = stronger USD
            ("USDJPY", 0.136, 148.00, True),     # USD/XXX — higher = stronger USD
            ("GBPUSD", 0.119, 1.2600, False),    # XXX/USD
            ("USDCAD", 0.091, 1.3800, True),     # USD/XXX
            ("USDCHF", 0.036, 0.8900, True),     # USD/XXX
            ("AUDUSD", 0.021, 0.6400, False),    # XXX/USD
            ("NZDUSD", 0.021, 0.5700, False),    # XXX/USD
        ]
        index = 1.0
        for pair, weight, baseline, is_usd_base in _dxy_cfg:
            s = spots.get(pair, {})
            mid = _sf(s.get("mid", s.get("close", baseline)), baseline)
            if mid <= 0:
                mid = baseline
            if is_usd_base:
                # USD/XXX: higher spot = stronger USD
                ratio = mid / baseline
            else:
                # XXX/USD: lower spot = stronger USD → invert
                ratio = baseline / mid
            index *= ratio ** weight
        kpis["dxy"] = round(index * 100.0, 2)
    except Exception:
        kpis["dxy"] = 100.0

    # G10 / EM avg vol
    g10_vols = [r["atm_1m"] for r in rows if r["pair"] in G10_PAIRS and r["atm_1m"] > 0]
    em_vols  = [r["atm_1m"] for r in rows if r["pair"] in EM_PAIRS and r["atm_1m"] > 0]
    kpis["g10_vol"] = round(np.mean(g10_vols), 2) if g10_vols else 0
    kpis["em_vol"]  = round(np.mean(em_vols), 2) if em_vols else 0

    # G10 average vol percentile
    g10_pctiles = [r.get("pctile", 50) for r in rows if r["pair"] in G10_PAIRS and r.get("pctile") is not None]
    kpis["g10_pctile"] = round(np.mean(g10_pctiles), 1) if g10_pctiles else 50.0

    # IV-RV aggregate (average G10 IV-RV spread)
    g10_ivrv = [r.get("iv_rv", 0) for r in rows if r["pair"] in G10_PAIRS and r.get("iv_rv", 0) != 0]
    kpis["ivrv_agg"] = round(np.mean(g10_ivrv), 2) if g10_ivrv else 0.0

    # Biggest mover
    if rows:
        biggest = max(rows, key=lambda r: abs(r.get("chg_pct", 0)))
        chg = biggest.get('chg_pct', 0)
        kpis["biggest"] = f"{biggest['pair']} {chg:+.2f}%" if np.isfinite(chg) else f"{biggest['pair']} —"
    else:
        kpis["biggest"] = "—"

    # Book Greeks (from portfolio)
    try:
        from core.fx_portfolio import compute_portfolio_risk
        from core.bloomberg_fx import get_fx_spots as _md_spots, get_fx_rates as _md_rates, get_fx_vol_surface as _md_volsurf
        from core.fx_conventions import FX_PAIR_REGISTRY as _md_registry
        # Build market data dicts matching compute_portfolio_risk(spots, rates, vol_surfaces) signature
        _md_pairs = list(_md_registry.keys())
        _md_spot_raw = _md_spots(_md_pairs) or {}
        _md_s = {p: d.get("mid", 1.0) if isinstance(d, dict) else float(d)
                 for p, d in _md_spot_raw.items()}
        _md_r = {}
        _md_v = {}
        for _p in _md_pairs:
            _rr = _md_rates(_p)
            _md_r[_p] = {"r_d": _rr.get("r_dom", 0.04), "r_f": _rr.get("r_for", 0.02)} if isinstance(_rr, dict) else {"r_d": 0.04, "r_f": 0.02}
            _sv = _md_volsurf(_p) or {}
            # Extract flat decimal vol from surface for portfolio pricing
            _flat = 0.10
            for _t in ("3M", "1M", "6M", "1Y"):
                if _t in _sv and isinstance(_sv[_t], dict):
                    _raw = _sv[_t].get("atm", 8.0)
                    _flat = _raw / 100.0 if _raw > 1.0 else _raw
                    break
            _md_v[_p] = max(_flat, 0.001)
        risk = compute_portfolio_risk(_md_s, _md_r, _md_v)
        totals = risk.get("totals", {})
        book_vega = totals.get("vega", 0)
        book_theta = totals.get("theta", 0)
    except Exception as _e:
        import logging as _lg
        _lg.getLogger(__name__).debug("Portfolio load fallback: %s", _e)
        book_vega, book_theta = 0, 0
    if not np.isfinite(book_vega) or not np.isfinite(book_theta) or (book_vega == 0 and book_theta == 0):
        kpis["book_vega"]  = "N/A"
        kpis["book_theta"] = "N/A"
    else:
        kpis["book_vega"]  = f"${book_vega/1000:.0f}K"
        kpis["book_theta"] = f"-${abs(book_theta)/1000:.0f}K"

    # Events 48h
    try:
        evts = _gather_events()
        n48 = sum(1 for e in evts if e["days_away"] <= 2)
        kpis["events_48h"] = n48
    except Exception as _e:
        import logging as _lg
        _lg.getLogger(__name__).debug("Events fallback: %s", _e)
        kpis["events_48h"] = 0

    return kpis


def _gather_events():
    """Get next CB events sorted by days away."""
    from datetime import datetime, timedelta
    now = datetime.now()

    # Approximate meeting cadence: day-offsets from start of year (rotates naturally)
    base_days = {
        "FED":  [14, 56, 98, 140, 182, 224, 266, 308],
        "ECB":  [21, 63, 105, 147, 189, 231, 273, 315],
        "BOE":  [28, 70, 112, 154, 196, 238, 280, 322],
        "BOJ":  [7, 49, 91, 133, 175, 217, 259, 301],
        "SNB":  [35, 77, 168, 259],
        "RBA":  [7, 42, 77, 112, 147, 182, 217, 252, 287, 322],
        "RBNZ": [14, 56, 112, 168, 224, 280, 336],
        "BOC":  [21, 63, 105, 147, 189, 231, 273, 315],
    }
    # Policy rates from Bloomberg (populated at runtime)
    _policy_rates = {}

    year_start = datetime(now.year, 1, 1)
    day_of_year = (now - year_start).days

    events = []
    for bank, info in _CB_BANKS.items():
        offsets = base_days.get(bank, [14, 56, 98])
        # Find the next meeting date that is in the future
        next_offset = None
        for d in offsets:
            if d > day_of_year:
                next_offset = d
                break
        # If none found this year, wrap to first meeting next year
        if next_offset is None:
            next_offset = offsets[0] + 365
        days_away = next_offset - day_of_year
        evt_date = now + timedelta(days=days_away)
        events.append({
            "bank": bank,
            "date": evt_date.strftime("%d-%b"),
            "days_away": days_away,
            "rate": _policy_rates.get(bank, "—"),
            "impact": info["impact"],
            "pairs": info["pairs"][:3],
        })
    events.sort(key=lambda e: e["days_away"])
    return events[:8]


def _build_positioning(pairs):
    """Find pairs with |z-score| > 0.8 (extreme > 1.5, notable > 0.8)."""
    extremes = []
    try:
        from core.fx_analytics import vol_zscore
        for pair in pairs:
            try:
                info = vol_zscore(pair, "3M", "ATM", 252)
                z = _sf(info.get("zscore", 0) if isinstance(info, dict) else info)
                if abs(z) > 0.8:
                    severity = "EXTREME" if abs(z) > 1.5 else "NOTABLE"
                    extremes.append({"pair": pair, "z": z,
                                     "direction": "RICH" if z > 0 else "CHEAP",
                                     "severity": severity})
            except Exception:
                continue
    except Exception:
        pass
    extremes.sort(key=lambda e: abs(e["z"]), reverse=True)
    return extremes[:12]


def _detect_crossings(current_rows, previous):
    """Compare current movers to previous snapshot, flag threshold crossings."""
    if not previous:
        return []
    alerts = []
    for row in current_rows:
        pair = row["pair"]
        prev = previous.get(pair, {})
        if not prev:
            continue
        pctile = row.get("pctile", 50)
        prev_pctile = prev.get("pctile", 50)
        iv_rv = row.get("iv_rv", 0)
        prev_iv_rv = prev.get("iv_rv", 0)
        term = row.get("term_spread", 0)
        prev_term = prev.get("term_spread", 0)
        vol_chg = row.get("vol_chg", 0)

        # Percentile crossed below 25th (became cheap)
        if prev_pctile >= 25 and pctile < 25:
            alerts.append({"msg": f"{pair} vol dropped to {_ordinal(pctile)} %ile — crossed below 25th (CHEAP)",
                           "color": COLORS.get("accent_green", "#00cc66")})
        # Percentile crossed above 75th (became rich)
        if prev_pctile <= 75 and pctile > 75:
            alerts.append({"msg": f"{pair} vol rose to {_ordinal(pctile)} %ile — crossed above 75th (RICH)",
                           "color": COLORS.get("accent_red", "#ff3333")})
        # IV-RV flipped positive
        if prev_iv_rv <= 0 and iv_rv > 0.5:
            alerts.append({"msg": f"{pair} IV-RV flipped positive ({iv_rv:+.1f}v — IV now RICH vs RV)",
                           "color": COLORS.get("accent_red", "#ff3333")})
        # IV-RV flipped negative
        if prev_iv_rv >= 0 and iv_rv < -0.5:
            alerts.append({"msg": f"{pair} IV-RV flipped negative ({iv_rv:+.1f}v — IV now CHEAP vs RV)",
                           "color": COLORS.get("accent_green", "#00cc66")})
        # Term structure inverted
        if prev_term <= 0.3 and term > 0.5:
            alerts.append({"msg": f"{pair} term structure inverted (1M-1Y: {term:+.1f}v)",
                           "color": COLORS.get("accent_orange", "#ff8800")})
        # Big vol move
        if abs(vol_chg) > 0.5:
            alerts.append({"msg": f"{pair} ATM vol moved {vol_chg:+.2f}v in 1 day",
                           "color": COLORS.get("accent_orange", "#ff8800")})
    # Sort by severity (bigger moves first), cap at 8
    alerts.sort(key=lambda a: a["color"] == COLORS.get("accent_red", "#ff3333"), reverse=True)
    return alerts[:8]


def _build_vol_index_chart(pairs):
    """60-day G10 vol index line chart."""
    try:
        from core.fx_analytics import vol_percentile
        from core.bloomberg_fx import get_fx_historical_vol

        # Try to get real history, fall back to synthetic
        hist_vols = []
        for pair in G10_PAIRS[:8]:
            try:
                h = get_fx_historical_vol(pair, "1M", "ATM", 60)
                if h is not None and len(h) >= 5:
                    arr = np.array(h, dtype=float) if not isinstance(h, np.ndarray) else h
                    hist_vols.append(arr[-60:])
            except Exception:
                continue

        if len(hist_vols) < 1:
            return no_data_fig(height=_CHART_H, msg="INSUFFICIENT VOL DATA")
        else:
            min_len = min(len(v) for v in hist_vols)
            trimmed = [v[-min_len:] for v in hist_vols]
            index = np.mean(trimmed, axis=0)

        mean_v = float(np.mean(index))
        std_v = float(np.std(index))
        x = list(range(len(index)))

        fig = go.Figure()
        # ±1σ band
        fig.add_trace(go.Scatter(x=x, y=[mean_v + std_v]*len(x),
                                 mode="lines", line=dict(color="#333355", width=1, dash="dot"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[mean_v - std_v]*len(x),
                                 mode="lines", line=dict(color="#333355", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(34,34,64,0.15)", showlegend=False))
        # Mean
        fig.add_trace(go.Scatter(x=x, y=[mean_v]*len(x),
                                 mode="lines", line=dict(color="#808080", width=1, dash="dash"),
                                 showlegend=False))
        # Index line
        fig.add_trace(go.Scatter(x=x, y=index.tolist(),
                                 mode="lines", line=dict(color="#ff8800", width=2.5),
                                 name="G10 Vol Index",
                                 customdata=list(zip(
                                     [f"{(v - mean_v)/max(std_v, 1e-6):+.1f}\u03c3" for v in index],
                                     [f"{percentileofscore(index, v):.0f}" for v in index],
                                 )),
                                 hovertemplate="Day %{x}<br>Vol: %{y:.2f}v<br>Z: %{customdata[0]}<br>Pctile: %{customdata[1]}%<extra></extra>"))
        # Current dot
        fig.add_trace(go.Scatter(x=[len(index)-1], y=[float(index[-1])],
                                 mode="markers", marker=dict(color="#ff8800", size=7),
                                 showlegend=False))
        fig.update_layout(**_chart_layout( height=_CHART_H,
                          margin=dict(l=50, r=15, t=35, b=28), showlegend=False,
                          title=dict(text="G10 VOL INDEX (60D)", font=dict(size=10, color="#808080")),
                          xaxis_title="Trading Days", yaxis_title="Vol (%)"))
        return fig
    except Exception:
        return _empty_fig("G10 VOL INDEX", msg="Insufficient vol history")


def _build_skew_chart(pairs):
    """25D RR horizontal bar chart."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        data = []
        for pair in pairs[:20]:
            try:
                surf = get_fx_vol_surface(pair) or {}
                rr = _sf(surf.get("3M", {}).get("rr25", surf.get("3M", {}).get("25D_RR", 0)))
                if rr != 0:
                    data.append({"pair": pair, "rr": rr})
            except Exception:
                continue

        if not data:
            return _empty_fig("SKEW MONITOR", _SMALL_H, "No RR data available")

        data.sort(key=lambda d: d["rr"])
        pairs_l = [d["pair"] for d in data]
        rrs = [d["rr"] for d in data]
        colors = [COLORS["accent_red"] if r < -0.3 else COLORS["accent_green"] if r > 0.3
                  else "#808080" for r in rrs]

        fig = go.Figure(go.Bar(y=pairs_l, x=rrs, orientation="h",
                                marker_color=colors, text=[f"{r:+.1f}" for r in rrs],
                                textposition="outside", textfont=dict(size=8, color="#d4d4d4"),
                                hovertemplate="%{y}: %{x:+.2f}v<br>Skew: %{text}<extra></extra>"))
        fig.update_layout(**_chart_layout( height=_SMALL_H,
                          margin=dict(l=55, r=10, t=25, b=10), showlegend=False,
                          title=dict(text="25D RR (SKEW)", font=dict(size=10, color="#808080")),
                          xaxis=dict(zeroline=True, zerolinecolor="#808080", zerolinewidth=1,
                                     gridcolor="#111111", tickfont=dict(size=8),
                                     title=dict(text="Risk Reversal (vol pts)", font=dict(size=9, color="#808080"))),
                          yaxis=dict(tickfont=dict(size=8, color="#d4d4d4"))))
        return fig
    except Exception:
        return _empty_fig("SKEW MONITOR", _SMALL_H, "No RR data available")


def _build_term_chart(pairs):
    """1M-1Y ATM spread horizontal bars."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        data = []
        for pair in pairs[:20]:
            try:
                surf = get_fx_vol_surface(pair) or {}
                atm_1m = _sf(surf.get("1M", {}).get("atm", 0))
                atm_1y = _sf(surf.get("1Y", {}).get("atm", 0))
                if atm_1m > 0 and atm_1y > 0:
                    spread = atm_1m - atm_1y
                    data.append({"pair": pair, "spread": spread})
            except Exception:
                continue

        if not data:
            return _empty_fig("TERM SHAPE", _SMALL_H, "No term structure data")

        data.sort(key=lambda d: d["spread"])
        pairs_l = [d["pair"] for d in data]
        spreads = [d["spread"] for d in data]
        colors = [COLORS["accent_red"] if s > 0.5 else COLORS["accent_green"] if s < -0.5
                  else "#808080" for s in spreads]

        fig = go.Figure(go.Bar(y=pairs_l, x=spreads, orientation="h",
                                marker_color=colors,
                                text=[f"{s:+.1f}" for s in spreads],
                                textposition="outside", textfont=dict(size=8, color="#d4d4d4"),
                                customdata=[[d["spread"]] for d in data],
                                hovertemplate="%{y}: %{x:+.2f}v<br>%{text}<extra>1M-1Y Spread</extra>"))
        fig.update_layout(**_chart_layout( height=_SMALL_H,
                          margin=dict(l=55, r=10, t=25, b=10), showlegend=False,
                          title=dict(text="1M-1Y SPREAD", font=dict(size=10, color="#808080")),
                          xaxis=dict(zeroline=True, zerolinecolor="#808080", zerolinewidth=1,
                                     gridcolor="#111111", tickfont=dict(size=8),
                                     title=dict(text="Spread (vol pts)", font=dict(size=9, color="#808080"))),
                          yaxis=dict(tickfont=dict(size=8, color="#d4d4d4"))))
        return fig
    except Exception:
        return _empty_fig("TERM SHAPE", _SMALL_H, "No term structure data")


def _build_vol_richness_heatmap(pairs):
    """Compact vol percentile heatmap: pairs x tenors."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        from core.fx_analytics import vol_percentile

        tenors = ["1M", "3M", "6M", "1Y"]
        pair_list = [p for p in pairs if p in G10_PAIRS][:12]

        z_vals = []
        text_vals = []
        for pair in pair_list:
            row_z = []
            row_t = []
            surf = get_fx_vol_surface(pair) or {}
            for t in tenors:
                atm = _sf(surf.get(t, {}).get("atm", 0))
                pct = 50.0
                try:
                    info = vol_percentile(pair, t, "ATM", 252)
                    if isinstance(info, dict) and info:
                        pct = _sf(info.get("percentile", 50))
                except Exception:
                    pass
                row_z.append(pct)
                row_t.append(f"{atm:.1f}\n{pct:.0f}%")
            z_vals.append(row_z)
            text_vals.append(row_t)

        if not z_vals:
            return _empty_fig("VOL RICHNESS", _SMALL_H, "No percentile data")

        fig = go.Figure(go.Heatmap(
            z=z_vals, x=tenors, y=pair_list, text=text_vals,
            texttemplate="%{text}", textfont=dict(size=9, color="#d4d4d4"),
            colorscale=[[0, "#1565c0"], [0.2, "#1565c0"], [0.4, "#1a1a2e"],
                        [0.5, "#1a1a2e"], [0.6, "#1a1a2e"], [0.8, "#ff3333"], [1, "#ff3333"]],
            zmin=0, zmax=100, showscale=False,
            hovertemplate="<b>%{y}</b> %{x}<br>Percentile: %{z:.0f}<extra></extra>",
            xgap=2, ygap=2,
        ))
        fig.update_layout(**_chart_layout(
            height=CHART_SM,
            margin=dict(l=65, r=10, t=30, b=25), showlegend=False,
            title=dict(text="VOL RICHNESS (% ILE)", font=dict(size=10, color="#808080")),
            xaxis=dict(tickfont=dict(size=9, color="#808080"),
                       title=dict(text="Tenor", font=dict(size=9, color="#808080"))),
            yaxis=dict(tickfont=dict(size=9, color="#d4d4d4"), autorange="reversed"),
        ))
        return fig
    except Exception:
        return _empty_fig("VOL RICHNESS", _SMALL_H, "No percentile data")


def _chart_layout(**overrides):
    """Merge CHART_TEMPLATE with overrides including deep-merged axes."""
    from core.theme import chart_layout
    return chart_layout(**overrides)


def _empty_fig(title="", height=_CHART_H, msg="LOADING"):
    fig = go.Figure()
    for y in [0.2, 0.4, 0.6, 0.8]:
        fig.add_shape(type="line", x0=0, x1=1, y0=y, y1=y,
                      xref="paper", yref="paper", line=dict(color="#0d0d1a", width=1))
    fig.update_layout(**_chart_layout(
        height=height, margin=dict(l=20, r=10, t=30, b=10),
        title=dict(text=title, font=dict(size=10, color="#808080")),
        annotations=[dict(text=msg, x=0.5, y=0.5, showarrow=False,
                          font=dict(color="#333355", size=10, family="'JetBrains Mono', monospace"),
                          xref="paper", yref="paper")]))
    return fig


# ── Layout ───────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Store(id=f"{_P}-selected-pair", data=None),
        dcc.Interval(id=f"{_P}-interval", interval=120_000, n_intervals=0),
        dcc.Download(id=f"{_P}-csv-download"),

        # ── Title + Controls ──
        html.Div([
            html.Div([
                html.Span("MARKET DASHBOARD", style={
                    "color": "#ffffff", "fontSize": "13px", "fontWeight": "700",
                    "letterSpacing": "2px", "fontFamily": _MONO,
                }),
                html.Span(id=f"{_P}-timestamp", style={
                    "color": "#808080", "fontSize": "9px", "marginLeft": "16px",
                    "fontFamily": _MONO,
                }),
            ]),
            html.Div([
                html.Span("GROUP", style=LABEL_STYLE),
                dcc.Dropdown(id=f"{_P}-group", options=GROUP_OPTIONS, value="ALL",
                             clearable=False, style={**DROPDOWN_STYLE, "width": "90px"}),
            ], style={"display": "flex", "alignItems": "center", "gap": GAP}),
            html.Div([
                html.Span("SORT", style=LABEL_STYLE),
                dcc.Dropdown(id=f"{_P}-sort", options=SORT_OPTIONS, value="spot",
                             clearable=False, style={**DROPDOWN_STYLE, "width": "100px"}),
            ], style={"display": "flex", "alignItems": "center", "gap": GAP}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "padding": "8px 0", "borderBottom": "1px solid #222240"}),

        # ── KPI Row ──
        html.Div(id=f"{_P}-kpis", className="stat-row", style={
            "padding": "8px 0",
        }),

        html.Div(id=f"{_P}-alerts", style={
            "padding": "6px 12px",
            "marginBottom": "8px",
            "maxHeight": "100px",
            "overflowY": "auto",
            "borderLeft": f"3px solid #ff8800",
            "backgroundColor": "#0a0a12",
            "borderRadius": "0px",
        }),

        # ── Main Grid: Movers (left) + Charts (right) ──
        html.Div([
            # Left column: movers table
            html.Div([
                html.Div(id=f"{_P}-movers", style={"overflowY": "auto", "maxHeight": "520px"}),
            ], style={"flex": "1", "minWidth": "400px", "border": "1px solid #222240",
                       "padding": GAP}),

            # Right column: charts stacked
            html.Div([
                html.Button("CSV", id=f"{_P}-csv-vol", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Loading(type="dot", color=COLORS["accent_cyan"], children=
                    dcc.Graph(id=f"{_P}-vol-index", config={"displayModeBar": False, "responsive": True},
                              style={"height": f"{_CHART_H}px"})),
                html.Button("CSV", id=f"{_P}-csv-skew", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Loading(type="dot", color=COLORS["accent_cyan"], children=
                    dcc.Graph(id=f"{_P}-skew", config={"displayModeBar": False, "responsive": True},
                              style={"height": f"{_SMALL_H}px"})),
                html.Button("CSV", id=f"{_P}-csv-term", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Loading(type="dot", color=COLORS["accent_cyan"], children=
                    dcc.Graph(id=f"{_P}-term", config={"displayModeBar": False, "responsive": True},
                              style={"height": f"{_SMALL_H}px"})),
            ], style={"flex": "1", "minWidth": "350px", "display": "flex",
                       "flexDirection": "column", "gap": GAP}),
        ], style={"display": "flex", "gap": GAP, "marginTop": SECTION_GAP}),

        # ── Bottom Row: Book Summary (left) + Events & Positioning (right) ──
        html.Div([
            # Left: vol richness heatmap
            html.Div([
                html.Div("VOL RICHNESS", style={
                    "color": "#808080", "fontSize": "10px", "fontWeight": "700",
                    "letterSpacing": "1.5px", "padding": f"{GAP} 10px",
                    "fontFamily": _MONO, "borderBottom": "1px solid #222240",
                }),
                dcc.Graph(id=f"{_P}-vol-richness", config={"displayModeBar": False, "responsive": True},
                          style={"height": f"{CHART_SM}px"}),
            ], style={"flex": "1", "border": "1px solid #222240"}),

            # Right: events + positioning
            html.Div([
                html.Div("EVENTS & POSITIONING", style={
                    "color": "#808080", "fontSize": "10px", "fontWeight": "700",
                    "letterSpacing": "1.5px", "padding": f"{GAP} 10px",
                    "fontFamily": _MONO, "borderBottom": "1px solid #222240",
                }),
                html.Div(id=f"{_P}-events", style={"padding": f"{GAP}"}),
                html.Div(id=f"{_P}-positioning", style={"padding": f"{GAP}"}),
            ], style={"flex": "1", "border": "1px solid #222240"}),
        ], style={"display": "flex", "gap": GAP, "marginTop": SECTION_GAP}),

    ], style={"fontFamily": _MONO})


# ── Component builders (called from callback) ───────────────────────────────

def _render_kpis(kpis):
    """Render 8 KPI stat boxes."""
    g10_pctile = kpis.get("g10_pctile", 50)
    ivrv_agg = kpis.get("ivrv_agg", 0)
    items = [
        ("DXY PROXY",      str(kpis.get("dxy", "—")),          "#ff8800"),
        ("G10 AVG VOL",    f"{kpis.get('g10_vol', 0):.1f}v",   "#ff8800"),
        ("EM AVG VOL",     f"{kpis.get('em_vol', 0):.1f}v",    "#ff8800"),
        ("G10 %ILE",       _ordinal(g10_pctile),              _pct_color(g10_pctile)),
        ("BIGGEST MOVER",  kpis.get("biggest", "—"),            "#d4d4d4"),
        ("IV-RV AGG",      f"{ivrv_agg:+.1f}v",                "#00cc66" if ivrv_agg > 0 else "#ff3333" if ivrv_agg < 0 else "#808080"),
        ("BOOK VEGA",      kpis.get("book_vega", "—"),          "#ff8800"),
        ("EVENTS 48H",     str(kpis.get("events_48h", 0)),      "#ff8800" if kpis.get("events_48h", 0) > 0 else "#808080"),
    ]
    boxes = []
    for label, value, color in items:
        boxes.append(html.Div([
            html.Div(value, style={"fontSize": "16px", "fontWeight": "700",
                                   "color": color, "fontFamily": _MONO}),
            html.Div(label, style={"fontSize": "10px", "color": "#808080",
                                   "letterSpacing": "1px", "fontFamily": _MONO,
                                   "marginTop": "4px"}),
        ], style={**STAT_BOX_STYLE, "borderLeft": f"3px solid {color}",
                  "minWidth": "90px"}))
    return boxes


def _render_movers_table(rows, sort_key):
    """Render the movers HTML table."""
    if sort_key == "spot":
        rows.sort(key=lambda r: (abs(r.get("chg_pct", 0)), r.get("pair", "")), reverse=True)
    elif sort_key == "vol":
        rows.sort(key=lambda r: (abs(r.get("vol_chg", 0)), r.get("pair", "")), reverse=True)
    elif sort_key == "atm":
        rows.sort(key=lambda r: (r.get("atm_1m", 0), r.get("pair", "")), reverse=True)
    elif sort_key == "pctile":
        rows.sort(key=lambda r: (r.get("pctile", 50), r.get("pair", "")), reverse=True)
    elif sort_key == "ivrv":
        rows.sort(key=lambda r: (abs(r.get("iv_rv", 0)), r.get("pair", "")), reverse=True)
    else:
        rows.sort(key=lambda r: r.get("pair", ""))

    header = html.Tr([
        html.Th(h, style=TABLE_HEADER_STYLE)
        for h in ["PAIR", "SPOT", "\u0394%", "ATM 1M", "ATM 3M", "\u0394Vol", "MOM", "RR25", "%ILE", "TERM", "IV-RV", "BEV"]
    ])

    body_rows = []
    for r in rows:
        chg_color = _color_chg(r["chg_pct"])
        vol_color = _color_chg(-r["vol_chg"])  # inverted: vol up = bad
        pctile = r.get("pctile", 50)
        term_spread = r.get("term_spread", 0)
        iv_rv = r.get("iv_rv", 0)
        breakeven_pips = r.get("breakeven_pips", 0)

        # Term spread color: red if backwardation (>0.5), green if contango (<-0.5)
        if term_spread > 0.5:
            term_color = COLORS["accent_red"]
        elif term_spread < -0.5:
            term_color = COLORS["accent_green"]
        else:
            term_color = "#808080"

        # IV-RV color: green if sell vol (>1), red if buy vol (<-1)
        if iv_rv > 1:
            ivrv_color = COLORS["accent_green"]
        elif iv_rv < -1:
            ivrv_color = COLORS["accent_red"]
        else:
            ivrv_color = "#808080"

        body_rows.append(html.Tr([
            html.Td(r["pair"], style={**TABLE_CELL_STYLE, "fontWeight": "700",
                                       "color": "#d4d4d4", "cursor": "pointer"},
                    id={"type": f"{_P}-row-click", "index": r["pair"]}),
            html.Td(_fmt_spot(r['pair'], r['spot']), style=TABLE_CELL_STYLE),
            html.Td(_sf_display(r['chg_pct'], "+.2f", "%"), style={**TABLE_CELL_STYLE, "color": chg_color}),
            html.Td(_sf_display(r['atm_1m'], ".1f", "v"), style=TABLE_CELL_STYLE),
            html.Td(_sf_display(r.get('atm_3m', 0), ".1f", "v"), style=TABLE_CELL_STYLE),
            html.Td(_sf_display(r['vol_chg'], "+.2f", "v"), style={**TABLE_CELL_STYLE, "color": vol_color}),
            html.Td(_sf_display(r.get('vol_mom', 0), "+.1f", "%"), style={**TABLE_CELL_STYLE,
                     "color": COLORS["accent_red"] if r.get("vol_mom", 0) and r.get("vol_mom", 0) > 2 else
                              COLORS["accent_green"] if r.get("vol_mom", 0) and r.get("vol_mom", 0) < -2 else "#808080"}),
            html.Td(_sf_display(r['rr25'], "+.1f", "v"), style=TABLE_CELL_STYLE),
            html.Td(_ordinal(pctile), style={**TABLE_CELL_STYLE, "color": _pct_color(pctile)}),
            html.Td(_sf_display(term_spread, "+.1f", "v"), style={**TABLE_CELL_STYLE, "color": term_color}),
            html.Td(_sf_display(iv_rv, "+.1f", "v"), style={**TABLE_CELL_STYLE, "color": ivrv_color}),
            html.Td(_sf_display(breakeven_pips, ".0f", "p"), style={**TABLE_CELL_STYLE, "color": "#1565c0"}),
        ]))

    return html.Table([html.Thead(header), html.Tbody(body_rows)],
                      style={"width": "100%", "borderCollapse": "collapse",
                             "fontFamily": _MONO, "fontSize": "11px"})


def _render_events(events):
    """Render events table."""
    if not events:
        return html.Div("No upcoming events", style={"color": "#808080", "fontSize": "10px"})

    header = html.Tr([html.Th(h, style={**TABLE_HEADER_STYLE, "fontSize": "8px"})
                       for h in ["BANK", "DATE", "DAYS", "RATE", "IMPACT"]])
    body = []
    for e in events[:5]:
        imp_color = {"HIGH": "#ff3333", "MED": "#ff8800", "LOW": "#808080"}.get(e["impact"], "#808080")
        body.append(html.Tr([
            html.Td(e["bank"], style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
            html.Td(e["date"], style=TABLE_CELL_STYLE),
            html.Td(f"{e['days_away']}d", style={**TABLE_CELL_STYLE,
                     "color": "#ff3333" if e["days_away"] <= 2 else "#d4d4d4"}),
            html.Td(e["rate"], style=TABLE_CELL_STYLE),
            html.Td(e["impact"], style={**TABLE_CELL_STYLE, "color": imp_color, "fontWeight": "600"}),
        ]))
    return html.Table([html.Thead(header), html.Tbody(body)],
                      style={"width": "100%", "borderCollapse": "collapse",
                             "fontFamily": _MONO, "fontSize": "10px", "marginBottom": "8px"})


def _render_positioning(extremes):
    """Render positioning extremes with severity distinction."""
    if not extremes:
        return html.Div("No positioning extremes",
                        style={"color": "#808080", "fontSize": "10px"})

    items = []
    for e in extremes:
        color = COLORS["accent_red"] if e["direction"] == "RICH" else "#1565c0"
        severity = e.get("severity", "NOTABLE")
        opacity = "1.0" if severity == "EXTREME" else "0.6"
        items.append(html.Div([
            html.Span(e["pair"], style={"color": "#d4d4d4", "fontWeight": "700",
                                         "marginRight": "8px", "fontSize": "10px"}),
            html.Span(f"z={e['z']:+.1f}", style={"color": color, "fontWeight": "600",
                                                    "fontSize": "10px", "marginRight": "6px",
                                                    "opacity": opacity}),
            html.Span(e["direction"], style={"color": color, "fontSize": "9px",
                                              "letterSpacing": "1px", "marginRight": "6px"}),
            html.Span(severity, style={"color": COLORS["text_muted"], "fontSize": "8px",
                                        "fontWeight": "600" if severity == "EXTREME" else "400"}),
        ], style={"padding": "2px 0"}))
    return html.Div(items)


# ── Callbacks ────────────────────────────────────────────────────────────────

def register_callbacks(app):
    # ── Fast callback: timestamp + KPIs + movers table (every interval) ──
    @app.callback(
        [
            Output(f"{_P}-timestamp", "children"),
            Output(f"{_P}-kpis", "children"),
            Output(f"{_P}-movers", "children"),
            Output(f"{_P}-alerts", "children"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-group", "value"),
            Input(f"{_P}-sort", "value"),
        ],
    )
    def update_kpis_movers(n, group, sort_key):
        global _previous_movers
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S  %d-%b-%Y")
        pairs = _pairs_for(group or "ALL")
        rows = _build_movers(pairs)
        kpis = _build_kpi_data(rows)

        alerts = _detect_crossings(rows, _previous_movers)
        _previous_movers = {r["pair"]: r for r in rows}

        alert_children = []
        if alerts:
            for a in alerts:
                alert_children.append(html.Div(a["msg"], style={
                    "color": a["color"], "fontSize": "10px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginBottom": "2px",
                }))
        else:
            alert_children = [html.Div("No threshold crossings detected",
                                       style={"color": "#808080", "fontSize": "10px",
                                              "fontFamily": "'JetBrains Mono', monospace"})]

        return ts, _render_kpis(kpis), _render_movers_table(rows, sort_key or "spot"), alert_children

    # ── Charts callback (vol index, skew, term, delta bars) ──
    @app.callback(
        [
            Output(f"{_P}-vol-index", "figure"),
            Output(f"{_P}-skew", "figure"),
            Output(f"{_P}-term", "figure"),
            Output(f"{_P}-vol-richness", "figure"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-group", "value"),
        ],
    )
    def update_charts(n, group):
        pairs = _pairs_for(group or "ALL")
        return (_build_vol_index_chart(pairs), _build_skew_chart(pairs),
                _build_term_chart(pairs), _build_vol_richness_heatmap(pairs))

    # ── Slow callback: book greeks + events + positioning ──
    @app.callback(
        [
            Output(f"{_P}-events", "children"),
            Output(f"{_P}-positioning", "children"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-group", "value"),
        ],
    )
    def update_book_events(n, group):
        try:
            pairs = _pairs_for(group or "ALL")
            events = _gather_events()
            positioning = _build_positioning(pairs)
            return _render_events(events), _render_positioning(positioning)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Book/events update failed: %s", e)
            empty = html.Div("Data unavailable", style={"color": "#808080", "fontSize": "11px"})
            return empty, empty

    # ── Row click → update store (which app.py propagates to global-pair) ──
    @app.callback(
        Output(f"{_P}-selected-pair", "data"),
        Input({"type": f"{_P}-row-click", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_row_click(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        try:
            import json
            prop_id = ctx.triggered[0]["prop_id"]
            id_dict = json.loads(prop_id.rsplit(".", 1)[0])
            return id_dict["index"]
        except Exception:
            raise PreventUpdate

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output(f"{_P}-csv-download", "data"),
        [Input(f"{_P}-csv-vol", "n_clicks"),
         Input(f"{_P}-csv-skew", "n_clicks"),
         Input(f"{_P}-csv-term", "n_clicks")],
        [State(f"{_P}-vol-index", "figure"),
         State(f"{_P}-skew", "figure"),
         State(f"{_P}-term", "figure")],
        prevent_initial_call=True,
    )
    def mdash_csv_export(n1, n2, n3, fig1, fig2, fig3):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        mapping = {
            f"{_P}-csv-vol": (fig1, "MarketDash", "VolIndex"),
            f"{_P}-csv-skew": (fig2, "MarketDash", "Skew"),
            f"{_P}-csv-term": (fig3, "MarketDash", "TermShape"),
        }
        if btn not in mapping:
            return no_update
        fig, panel, chart_type = mapping[btn]
        if not fig:
            return no_update
        try:
            return export_csv(fig, panel, chart_type)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("CSV export failed: %s", e)
            return no_update
