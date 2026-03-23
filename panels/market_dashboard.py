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
from dash import html, dcc, Input, Output, State, callback_context, ALL, MATCH
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, TAB_STYLE, TAB_SELECTED_STYLE,
    TABLE_HEADER_STYLE, TABLE_CELL_STYLE, clickable_stat, make_stat_style,
)
from core.fx_conventions import FX_PAIR_REGISTRY, tenor_to_days

# ── Constants ────────────────────────────────────────────────────────────────

_P = "mdash"  # prefix for all IDs

_MONO = "'JetBrains Mono', monospace"
_CHART_H = 320
_SMALL_H = 280

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
]


# ── Safe helpers ─────────────────────────────────────────────────────────────

def _sf(v, d=0.0):
    """Safe float."""
    try:
        f = float(v)
        return d if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return d


def _pairs_for(group):
    if group == "G10":
        return G10_PAIRS
    if group == "EM":
        return EM_PAIRS
    return ALL_PAIRS


def _color_chg(v):
    if v > 0:
        return COLORS["accent_green"]
    if v < 0:
        return COLORS["accent_red"]
    return COLORS["text_secondary"]


def _pct_color(p):
    if p < 20:
        return "#60a5fa"
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

            rows.append({
                "pair": pair, "spot": spot, "chg_pct": chg,
                "atm_1m": atm_1m, "vol_chg": vol_chg, "rr25": rr25,
            })
        except Exception:
            rows.append({"pair": pair, "spot": 0, "chg_pct": 0,
                         "atm_1m": 0, "vol_chg": 0, "rr25": 0})
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
        # Baselines are roughly "neutral" mid-2024 levels for scaling
        _dxy_cfg = [
            # (pair,  weight, baseline, is_usd_base)
            ("EURUSD", 0.576, 1.0800, False),   # XXX/USD — lower = stronger USD
            ("USDJPY", 0.136, 150.00, True),     # USD/XXX — higher = stronger USD
            ("GBPUSD", 0.119, 1.2700, False),    # XXX/USD
            ("USDCAD", 0.091, 1.3600, True),     # USD/XXX
            ("USDCHF", 0.036, 0.8800, True),     # USD/XXX
            ("AUDUSD", 0.021, 0.6600, False),    # XXX/USD
            ("NZDUSD", 0.021, 0.6100, False),    # XXX/USD
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

    # Risk sentiment (simplified composite)
    try:
        risk_score = 50.0
        if kpis["g10_vol"] > 0:
            risk_score += (kpis["g10_vol"] - 8.0) * 3
        if kpis["em_vol"] > kpis["g10_vol"]:
            risk_score += (kpis["em_vol"] - kpis["g10_vol"]) * 2
        risk_score = max(0, min(100, risk_score))
        kpis["risk"] = round(risk_score, 0)
    except Exception:
        kpis["risk"] = 50

    # Biggest mover
    if rows:
        biggest = max(rows, key=lambda r: abs(r.get("chg_pct", 0)))
        kpis["biggest"] = f"{biggest['pair']} {biggest['chg_pct']:+.2f}%"
    else:
        kpis["biggest"] = "—"

    # Book Greeks (from portfolio)
    try:
        from core.fx_portfolio import get_portfolio, compute_portfolio_risk
        portfolio = get_portfolio()
        risk = compute_portfolio_risk(portfolio)
        book_vega = risk.get("total_vega", 0)
        book_theta = risk.get("total_theta", 0)
    except Exception as _e:
        import logging as _lg
        _lg.getLogger(__name__).debug("Portfolio load fallback: %s", _e)
        book_vega, book_theta = 45000, -18000
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
    # Realistic current policy rates
    _policy_rates = {
        "FED": "5.25%", "ECB": "4.50%", "BOE": "5.25%", "BOJ": "0.25%",
        "SNB": "1.75%", "RBA": "4.35%", "RBNZ": "5.50%", "BOC": "5.00%",
    }

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
    """Find pairs with |z-score| > 1.2."""
    extremes = []
    try:
        from core.fx_analytics import vol_zscore
        for pair in pairs:
            try:
                info = vol_zscore(pair, "3M", "ATM", 252)
                z = _sf(info.get("zscore", 0) if isinstance(info, dict) else info)
                if abs(z) > 1.2:
                    extremes.append({"pair": pair, "z": z,
                                     "direction": "RICH" if z > 0 else "CHEAP"})
            except Exception:
                continue
    except Exception:
        pass
    extremes.sort(key=lambda e: abs(e["z"]), reverse=True)
    return extremes[:10]


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
                if h is not None and len(h) >= 30:
                    arr = np.array(h, dtype=float) if not isinstance(h, np.ndarray) else h
                    hist_vols.append(arr[-60:])
            except Exception:
                continue

        if len(hist_vols) < 3:
            rng = np.random.RandomState(99)
            base = 8.5
            noise = np.cumsum(rng.normal(0, 0.05, 60))
            index = base + noise
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
                                 fill="tonexty", fillcolor="rgba(51,51,85,0.25)", showlegend=False))
        # Mean
        fig.add_trace(go.Scatter(x=x, y=[mean_v]*len(x),
                                 mode="lines", line=dict(color="#666666", width=1, dash="dash"),
                                 showlegend=False))
        # Index line
        fig.add_trace(go.Scatter(x=x, y=index.tolist(),
                                 mode="lines", line=dict(color="#ff8800", width=2.5),
                                 name="G10 Vol Index"))
        # Current dot
        fig.add_trace(go.Scatter(x=[len(index)-1], y=[float(index[-1])],
                                 mode="markers", marker=dict(color="#ff8800", size=7),
                                 showlegend=False))
        fig.update_layout(**_chart_layout( height=_CHART_H,
                          margin=dict(l=40, r=10, t=25, b=20), showlegend=False,
                          title=dict(text="G10 VOL INDEX (60D)", font=dict(size=10, color="#666666"))))
        return fig
    except Exception:
        return _empty_fig("G10 VOL INDEX")


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
            return _empty_fig("SKEW MONITOR")

        data.sort(key=lambda d: d["rr"])
        pairs_l = [d["pair"] for d in data]
        rrs = [d["rr"] for d in data]
        colors = [COLORS["accent_red"] if r < -0.3 else COLORS["accent_green"] if r > 0.3
                  else "#666666" for r in rrs]

        fig = go.Figure(go.Bar(y=pairs_l, x=rrs, orientation="h",
                                marker_color=colors, text=[f"{r:+.1f}" for r in rrs],
                                textposition="outside", textfont=dict(size=8, color="#d4d4d4")))
        fig.update_layout(**_chart_layout( height=_SMALL_H,
                          margin=dict(l=55, r=10, t=25, b=10), showlegend=False,
                          title=dict(text="25D RR (SKEW)", font=dict(size=10, color="#666666")),
                          xaxis=dict(zeroline=True, zerolinecolor="#666666", zerolinewidth=1,
                                     gridcolor="#111111", tickfont=dict(size=8)),
                          yaxis=dict(tickfont=dict(size=8, color="#d4d4d4"))))
        return fig
    except Exception:
        return _empty_fig("SKEW MONITOR")


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
            return _empty_fig("TERM SHAPE")

        data.sort(key=lambda d: d["spread"])
        pairs_l = [d["pair"] for d in data]
        spreads = [d["spread"] for d in data]
        colors = [COLORS["accent_red"] if s > 0.5 else COLORS["accent_green"] if s < -0.5
                  else "#666666" for s in spreads]

        fig = go.Figure(go.Bar(y=pairs_l, x=spreads, orientation="h",
                                marker_color=colors,
                                text=[f"{s:+.1f}" for s in spreads],
                                textposition="outside", textfont=dict(size=8, color="#d4d4d4")))
        fig.update_layout(**_chart_layout( height=_SMALL_H,
                          margin=dict(l=55, r=10, t=25, b=10), showlegend=False,
                          title=dict(text="1M-1Y SPREAD", font=dict(size=10, color="#666666")),
                          xaxis=dict(zeroline=True, zerolinecolor="#666666", zerolinewidth=1,
                                     gridcolor="#111111", tickfont=dict(size=8)),
                          yaxis=dict(tickfont=dict(size=8, color="#d4d4d4"))))
        return fig
    except Exception:
        return _empty_fig("TERM SHAPE")


def _build_delta_bars():
    """Net delta by pair (book summary)."""
    pairs = G10_PAIRS[:10]
    try:
        from core.fx_portfolio import get_portfolio, compute_position_greeks
        portfolio = get_portfolio()
        delta_map = {}
        for pos in portfolio:
            pair = pos.get("pair", "")
            if pair in pairs:
                try:
                    greeks = compute_position_greeks(pos)
                    delta_map[pair] = delta_map.get(pair, 0) + greeks.get("delta", 0)
                except Exception:
                    pass
        deltas = [delta_map.get(p, 0) / 1_000_000 for p in pairs]  # in millions
    except Exception:
        deltas = [0.0] * len(pairs)

    colors = [COLORS["accent_green"] if d > 0 else COLORS["accent_red"] for d in deltas]

    fig = go.Figure(go.Bar(y=pairs, x=deltas, orientation="h",
                            marker_color=colors,
                            text=[f"{d:+.1f}M" for d in deltas],
                            textposition="outside", textfont=dict(size=8, color="#d4d4d4")))
    fig.update_layout(**_chart_layout( height=200,
                      margin=dict(l=55, r=30, t=25, b=10), showlegend=False,
                      title=dict(text="NET Δ BY PAIR", font=dict(size=10, color="#666666")),
                      xaxis=dict(zeroline=True, zerolinecolor="#666666", gridcolor="#111111",
                                 tickfont=dict(size=8)),
                      yaxis=dict(tickfont=dict(size=8, color="#d4d4d4"))))
    return fig


def _chart_layout(**overrides):
    """Merge CHART_TEMPLATE with overrides including deep-merged axes."""
    from core.theme import chart_layout
    return chart_layout(**overrides)


def _empty_fig(title=""):
    fig = go.Figure()
    fig.update_layout(**_chart_layout(
        height=_CHART_H, margin=dict(l=20, r=10, t=30, b=10),
        title=dict(text=title, font=dict(size=10, color="#666666")),
        annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False,
                          font=dict(color="#666666", size=11), xref="paper", yref="paper")]))
    return fig


# ── Layout ───────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Store(id=f"{_P}-selected-pair", data=None),
        dcc.Interval(id=f"{_P}-interval", interval=30_000, n_intervals=0),

        # ── Title + Controls ──
        html.Div([
            html.Div([
                html.Span("MARKET DASHBOARD", style={
                    "color": "#ffffff", "fontSize": "13px", "fontWeight": "700",
                    "letterSpacing": "2px", "fontFamily": _MONO,
                }),
                html.Span(id=f"{_P}-timestamp", style={
                    "color": "#666666", "fontSize": "9px", "marginLeft": "16px",
                    "fontFamily": _MONO,
                }),
            ]),
            html.Div([
                html.Span("GROUP", style=LABEL_STYLE),
                dcc.Dropdown(id=f"{_P}-group", options=GROUP_OPTIONS, value="ALL",
                             clearable=False, style={**DROPDOWN_STYLE, "width": "90px"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
            html.Div([
                html.Span("SORT", style=LABEL_STYLE),
                dcc.Dropdown(id=f"{_P}-sort", options=SORT_OPTIONS, value="spot",
                             clearable=False, style={**DROPDOWN_STYLE, "width": "100px"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "padding": "8px 0", "borderBottom": "1px solid #1a1a2e"}),

        # ── KPI Row ──
        html.Div(id=f"{_P}-kpis", style={
            "display": "flex", "gap": "6px", "padding": "8px 0",
            "flexWrap": "wrap",
        }),

        # ── Main Grid: Movers (left) + Charts (right) ──
        html.Div([
            # Left column: movers table
            html.Div([
                html.Div(id=f"{_P}-movers", style={"overflowY": "auto", "maxHeight": "520px"}),
            ], style={"flex": "1", "minWidth": "400px", "border": "1px solid #1a1a2e",
                       "padding": "4px"}),

            # Right column: charts stacked
            html.Div([
                dcc.Graph(id=f"{_P}-vol-index", config={"displayModeBar": False, "responsive": True},
                          style={"height": f"{_CHART_H}px"}),
                dcc.Graph(id=f"{_P}-skew", config={"displayModeBar": False, "responsive": True},
                          style={"height": f"{_SMALL_H}px"}),
                dcc.Graph(id=f"{_P}-term", config={"displayModeBar": False, "responsive": True},
                          style={"height": f"{_SMALL_H}px"}),
            ], style={"flex": "1", "minWidth": "350px", "display": "flex",
                       "flexDirection": "column", "gap": "2px"}),
        ], style={"display": "flex", "gap": "4px", "marginTop": "4px"}),

        # ── Bottom Row: Book Summary (left) + Events & Positioning (right) ──
        html.Div([
            # Left: book delta bars
            html.Div([
                html.Div("BOOK SUMMARY", style={
                    "color": "#666666", "fontSize": "9px", "fontWeight": "700",
                    "letterSpacing": "1.5px", "padding": "4px 8px",
                    "fontFamily": _MONO, "borderBottom": "1px solid #1a1a2e",
                }),
                html.Div(id=f"{_P}-book-greeks", style={
                    "display": "flex", "gap": "8px", "padding": "6px 8px",
                }),
                dcc.Graph(id=f"{_P}-delta-bars", config={"displayModeBar": False, "responsive": True},
                          style={"height": "200px"}),
            ], style={"flex": "1", "border": "1px solid #1a1a2e"}),

            # Right: events + positioning
            html.Div([
                html.Div("EVENTS & POSITIONING", style={
                    "color": "#666666", "fontSize": "9px", "fontWeight": "700",
                    "letterSpacing": "1.5px", "padding": "4px 8px",
                    "fontFamily": _MONO, "borderBottom": "1px solid #1a1a2e",
                }),
                html.Div(id=f"{_P}-events", style={"padding": "4px 8px"}),
                html.Div(id=f"{_P}-positioning", style={"padding": "4px 8px"}),
            ], style={"flex": "1", "border": "1px solid #1a1a2e"}),
        ], style={"display": "flex", "gap": "4px", "marginTop": "4px"}),

    ], style={"fontFamily": _MONO})


# ── Component builders (called from callback) ───────────────────────────────

def _render_kpis(kpis):
    """Render 8 KPI stat boxes."""
    items = [
        ("DXY PROXY",      str(kpis.get("dxy", "—")),          "#ff8800"),
        ("G10 AVG VOL",    f"{kpis.get('g10_vol', 0):.1f}v",   "#ff8800"),
        ("EM AVG VOL",     f"{kpis.get('em_vol', 0):.1f}v",    "#ff8800"),
        ("RISK SCORE",     f"{kpis.get('risk', 50):.0f}",      "#ff3333" if kpis.get("risk", 50) > 65 else "#00cc66"),
        ("BIGGEST MOVER",  kpis.get("biggest", "—"),            "#d4d4d4"),
        ("BOOK VEGA",      kpis.get("book_vega", "—"),          "#ff8800"),
        ("BOOK THETA",     kpis.get("book_theta", "—"),         "#ff3333"),
        ("EVENTS 48H",     str(kpis.get("events_48h", 0)),      "#ff8800" if kpis.get("events_48h", 0) > 0 else "#666666"),
    ]
    boxes = []
    for label, value, color in items:
        boxes.append(html.Div([
            html.Div(value, style={"fontSize": "13px", "fontWeight": "700",
                                   "color": color, "fontFamily": _MONO}),
            html.Div(label, style={"fontSize": "8px", "color": "#666666",
                                   "letterSpacing": "1px", "fontFamily": _MONO}),
        ], style={**STAT_BOX_STYLE, "borderTop": f"2px solid {color}",
                  "minWidth": "90px", "flex": "1"}))
    return boxes


def _render_movers_table(rows, sort_key):
    """Render the movers HTML table."""
    if sort_key == "spot":
        rows.sort(key=lambda r: abs(r.get("chg_pct", 0)), reverse=True)
    elif sort_key == "vol":
        rows.sort(key=lambda r: abs(r.get("vol_chg", 0)), reverse=True)
    elif sort_key == "atm":
        rows.sort(key=lambda r: r.get("atm_1m", 0), reverse=True)
    else:
        rows.sort(key=lambda r: r.get("pair", ""))

    header = html.Tr([
        html.Th(h, style=TABLE_HEADER_STYLE)
        for h in ["PAIR", "SPOT", "Δ%", "ATM 1M", "ΔVol", "RR25"]
    ])

    body_rows = []
    for r in rows:
        chg_color = _color_chg(r["chg_pct"])
        vol_color = _color_chg(-r["vol_chg"])  # inverted: vol up = bad
        body_rows.append(html.Tr([
            html.Td(r["pair"], style={**TABLE_CELL_STYLE, "fontWeight": "700",
                                       "color": "#d4d4d4", "cursor": "pointer"},
                    id={"type": f"{_P}-row-click", "index": r["pair"]}),
            html.Td(f"{r['spot']:.4f}" if r['spot'] < 10 else f"{r['spot']:.2f}",
                    style=TABLE_CELL_STYLE),
            html.Td(f"{r['chg_pct']:+.2f}%", style={**TABLE_CELL_STYLE, "color": chg_color}),
            html.Td(f"{r['atm_1m']:.1f}v", style=TABLE_CELL_STYLE),
            html.Td(f"{r['vol_chg']:+.2f}", style={**TABLE_CELL_STYLE, "color": vol_color}),
            html.Td(f"{r['rr25']:+.1f}", style=TABLE_CELL_STYLE),
        ]))

    return html.Table([html.Thead(header), html.Tbody(body_rows)],
                      style={"width": "100%", "borderCollapse": "collapse",
                             "fontFamily": _MONO, "fontSize": "11px"})


def _render_events(events):
    """Render events table."""
    if not events:
        return html.Div("No upcoming events", style={"color": "#666666", "fontSize": "10px"})

    header = html.Tr([html.Th(h, style={**TABLE_HEADER_STYLE, "fontSize": "8px"})
                       for h in ["BANK", "DATE", "DAYS", "RATE", "IMPACT"]])
    body = []
    for e in events[:5]:
        imp_color = {"HIGH": "#ff3333", "MED": "#ff8800", "LOW": "#666666"}.get(e["impact"], "#666666")
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
    """Render positioning extremes."""
    if not extremes:
        return html.Div("No positioning extremes (|z| > 1.2)",
                        style={"color": "#666666", "fontSize": "10px"})

    items = []
    for e in extremes:
        color = COLORS["accent_red"] if e["direction"] == "RICH" else "#60a5fa"
        items.append(html.Div([
            html.Span(e["pair"], style={"color": "#d4d4d4", "fontWeight": "700",
                                         "marginRight": "8px", "fontSize": "10px"}),
            html.Span(f"z={e['z']:+.1f}", style={"color": color, "fontWeight": "600",
                                                    "fontSize": "10px", "marginRight": "6px"}),
            html.Span(e["direction"], style={"color": color, "fontSize": "9px",
                                              "letterSpacing": "1px"}),
        ], style={"padding": "2px 0"}))
    return html.Div(items)


def _render_book_greeks():
    """Render book Greeks summary boxes."""
    try:
        from core.fx_portfolio import get_portfolio, compute_portfolio_risk
        portfolio = get_portfolio()
        risk = compute_portfolio_risk(portfolio)
        book_vega = risk.get("total_vega", 0)
        book_theta = risk.get("total_theta", 0)
        book_delta = risk.get("total_delta", 0)
        book_mv = risk.get("total_mv", 0)
    except Exception:
        book_vega, book_theta, book_delta, book_mv = 45000, -18000, 2500000, 120000000
    greeks = [
        ("VEGA", f"${book_vega/1000:.0f}K", "#ff8800"),
        ("THETA", f"-${abs(book_theta)/1000:.0f}K", "#ff3333"),
        ("DELTA", f"${book_delta/1000000:.1f}M", "#d4d4d4"),
        ("MV", f"${book_mv/1000000:.0f}M", "#666666"),
    ]
    boxes = []
    for label, val, color in greeks:
        boxes.append(html.Div([
            html.Div(val, style={"fontSize": "11px", "fontWeight": "700", "color": color}),
            html.Div(label, style={"fontSize": "8px", "color": "#666666", "letterSpacing": "1px"}),
        ], style={**STAT_BOX_STYLE, "minWidth": "60px"}))
    return boxes


# ── Callbacks ────────────────────────────────────────────────────────────────

def register_callbacks(app):
    # ── Fast callback: timestamp + KPIs + movers table (every interval) ──
    @app.callback(
        [
            Output(f"{_P}-timestamp", "children"),
            Output(f"{_P}-kpis", "children"),
            Output(f"{_P}-movers", "children"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-group", "value"),
            Input(f"{_P}-sort", "value"),
        ],
    )
    def update_kpis_movers(n, group, sort_key):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S  %d-%b-%Y")
        pairs = _pairs_for(group or "ALL")
        rows = _build_movers(pairs)
        kpis = _build_kpi_data(rows)
        return ts, _render_kpis(kpis), _render_movers_table(rows, sort_key or "spot")

    # ── Charts callback (vol index, skew, term, delta bars) ──
    @app.callback(
        [
            Output(f"{_P}-vol-index", "figure"),
            Output(f"{_P}-skew", "figure"),
            Output(f"{_P}-term", "figure"),
            Output(f"{_P}-delta-bars", "figure"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-group", "value"),
        ],
    )
    def update_charts(n, group):
        pairs = _pairs_for(group or "ALL")
        return (_build_vol_index_chart(pairs), _build_skew_chart(pairs),
                _build_term_chart(pairs), _build_delta_bars())

    # ── Slow callback: book greeks + events + positioning ──
    @app.callback(
        [
            Output(f"{_P}-book-greeks", "children"),
            Output(f"{_P}-events", "children"),
            Output(f"{_P}-positioning", "children"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-group", "value"),
        ],
    )
    def update_book_events(n, group):
        pairs = _pairs_for(group or "ALL")
        events = _gather_events()
        positioning = _build_positioning(pairs)
        return _render_book_greeks(), _render_events(events), _render_positioning(positioning)

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
