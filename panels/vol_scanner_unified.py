"""
Vol Scanner Unified — VOL workspace
====================================
Replaces: Vol Scanner + Vol Richness + Skew Lab (3 → 1).

Three view tabs inside one panel:
  [TABLE]   30-pair DataTable, signals, drill-down sparklines
  [HEATMAP] 30×6 percentile heatmap, click → drill-down (ATM, IV-RV, Vol Cone)
  [SKEW]    30×6 RR surface, smile comparison, wing analysis, butterfly, tails
"""

import json
import numpy as np
from dash import html, dcc, Input, Output, State, callback_context, ALL, MATCH, dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, TAB_STYLE, TAB_SELECTED_STYLE,
    TABLE_HEADER_STYLE, TABLE_CELL_STYLE, clickable_stat, make_stat_style,
    GAP, SECTION_GAP, CHART_SM, CHART_MD, CHART_LG,
)
from core.fx_conventions import FX_PAIR_REGISTRY, tenor_to_days

# ── Constants ────────────────────────────────────────────────────────────────

_P = "vsu"  # prefix
_MONO = "'JetBrains Mono', monospace"

ALL_PAIRS = sorted(FX_PAIR_REGISTRY.keys())
G10_PAIRS = [p for p in ALL_PAIRS if FX_PAIR_REGISTRY[p].group == "G10"]
EM_PAIRS  = [p for p in ALL_PAIRS if p not in G10_PAIRS]

HEATMAP_TENORS = ["1M", "2M", "3M", "6M", "1Y", "2Y"]
SURFACE_TENORS = ["1M", "2M", "3M", "6M", "1Y", "2Y"]

GROUP_OPTIONS = [
    {"label": "ALL", "value": "ALL"},
    {"label": "G10 MAJOR", "value": "G10_MAJOR"},
    {"label": "G10 CROSS", "value": "G10_CROSS"},
    {"label": "SCANDIE",   "value": "SCANDIE"},
    {"label": "EM",        "value": "EM"},
]

LOOKBACK_OPTIONS = [
    {"label": "60D",  "value": 60},
    {"label": "120D", "value": 120},
    {"label": "252D", "value": 252},
]

SIGNAL_OPTIONS = [
    {"label": "ALL SIGNALS", "value": "ALL"},
    {"label": "VOL CHEAP",   "value": "CHEAP"},
    {"label": "VOL RICH",    "value": "RICH"},
    {"label": "SKEW EXTREME","value": "SKEW"},
    {"label": "RV BUY",      "value": "RV_BUY"},
    {"label": "RV SELL",     "value": "RV_SELL"},
    {"label": "TERM STEEP",  "value": "TERM"},
]

METRIC_OPTIONS = [
    {"label": "ATM",     "value": "ATM"},
    {"label": "25D RR",  "value": "25D_RR"},
    {"label": "25D BF",  "value": "25D_BF"},
]

TAIL_MOVES = [1, 2, 3, 5]

RICHNESS_COLORSCALE = [
    [0.0, "#0044ff"], [0.15, "#4488ff"], [0.35, "#aaccff"],
    [0.50, "#ffffff"], [0.65, "#ffccaa"], [0.85, "#ff6644"],
    [1.0, "#ff3333"],
]

SKEW_COLORSCALE = [
    [0.0, "#ff3333"], [0.35, "#ff8888"], [0.5, "#ffffff"],
    [0.65, "#8888ff"], [1.0, "#0044ff"],
]


# ── Safe helpers ─────────────────────────────────────────────────────────────

def _sf(v, d=0.0):
    try:
        f = float(v)
        return d if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return d


def _pairs_for(group):
    if group == "G10_MAJOR":
        return [p for p in ALL_PAIRS if FX_PAIR_REGISTRY[p].group == "G10" and FX_PAIR_REGISTRY[p].subgroup == "Majors"]
    if group == "G10_CROSS":
        return [p for p in ALL_PAIRS if FX_PAIR_REGISTRY[p].group == "G10" and FX_PAIR_REGISTRY[p].subgroup == "Crosses"]
    if group == "SCANDIE":
        return [p for p in ALL_PAIRS if FX_PAIR_REGISTRY[p].group == "G10" and FX_PAIR_REGISTRY[p].subgroup == "Scandies"]
    if group == "EM":
        return EM_PAIRS
    return ALL_PAIRS


def _extract_atm(surface, tenor):
    t = surface.get(tenor, {})
    return _sf(t.get("atm", 0))


def _extract_rr(surface, tenor):
    t = surface.get(tenor, {})
    return _sf(t.get("rr25", t.get("25D_RR", 0)))


def _extract_bf(surface, tenor):
    t = surface.get(tenor, {})
    return _sf(t.get("bf25", t.get("25D_BF", 0)))


def _chart_layout(**overrides):
    """Merge CHART_TEMPLATE with overrides including deep-merged axes."""
    from core.theme import chart_layout
    return chart_layout(**overrides)


def _empty_fig(title=""):
    fig = go.Figure()
    fig.update_layout(**_chart_layout(
        height=CHART_SM, margin=dict(l=20, r=10, t=30, b=10),
        title=dict(text=title, font=dict(size=10, color="#808080")),
        annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False,
                          font=dict(color="#808080", size=11), xref="paper", yref="paper")]))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# TABLE VIEW — Data & signal logic (from vol_scanner.py)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_signal(atm_pct, rr_pct, ivrv_z, term_z):
    signals = []
    if atm_pct < 10:
        signals.append("VOL CHEAP")
    elif atm_pct < 15:
        signals.append("VOL CHEAP-ISH")
    if atm_pct > 90:
        signals.append("VOL RICH")
    if rr_pct < 10 or rr_pct > 90:
        signals.append("SKEW EXTREME")
    if abs(term_z) > 2:
        signals.append("TERM STEEP")
    if ivrv_z < -1:
        signals.append("RV BUY")
    if ivrv_z > 1:
        signals.append("RV SELL")
    return " | ".join(signals) if signals else "—"


def _build_scanner_rows(pairs, lookback):
    """Build table rows for vol scanner."""
    try:
        from core.bloomberg_fx import get_fx_spots, get_fx_vol_surface
        from core.fx_analytics import vol_percentile, vol_zscore, iv_rv_spread
    except Exception:
        return []

    spots = get_fx_spots() or {}
    rows = []
    for pair in pairs:
        try:
            s = spots.get(pair, {})
            spot = _sf(s.get("mid", s.get("close", 0)))
            chg = _sf(s.get("change_pct", 0))

            surf = get_fx_vol_surface(pair) or {}
            atm_1m = _extract_atm(surf, "1M")
            atm_3m = _extract_atm(surf, "3M")
            atm_1y = _extract_atm(surf, "1Y")
            rr25 = _extract_rr(surf, "3M")
            bf25 = _extract_bf(surf, "3M")

            # Percentiles & z-scores
            try:
                p_info = vol_percentile(pair, "3M", "ATM", lookback)
                atm3m_pct = _sf(p_info.get("percentile", 50) if isinstance(p_info, dict) else 50)
            except Exception:
                atm3m_pct = 50
            try:
                z_info = vol_zscore(pair, "3M", "ATM", lookback)
                z_atm3m = _sf(z_info.get("zscore", 0) if isinstance(z_info, dict) else z_info)
            except Exception:
                z_atm3m = 0

            # IV-RV
            try:
                ivrv_info = iv_rv_spread(pair, "3M", lookback=lookback)
                if hasattr(ivrv_info, 'iloc'):
                    ivrv_3m = _sf(ivrv_info["spread"].iloc[-1] if "spread" in ivrv_info.columns else 0)
                else:
                    ivrv_3m = 0
            except Exception:
                ivrv_3m = 0

            # RR percentile
            try:
                rr_info = vol_percentile(pair, "3M", "25D_RR", lookback)
                rr3m_pct = _sf(rr_info.get("percentile", 50) if isinstance(rr_info, dict) else 50)
            except Exception:
                rr3m_pct = 50

            term_spread = atm_1m - atm_1y if (atm_1m > 0 and atm_1y > 0) else 0
            signal = _compute_signal(atm3m_pct, rr3m_pct, ivrv_3m, term_spread)

            # Group label
            spec = FX_PAIR_REGISTRY.get(pair)
            if spec:
                grp_label = {"Majors": "G10 MAJ", "Crosses": "G10 X",
                             "Scandies": "SCAN"}.get(spec.subgroup, spec.group)
            else:
                grp_label = "EM"

            rows.append({
                "pair": pair, "group": grp_label,
                "spot": round(spot, 4), "chg_pct": round(chg, 2),
                "atm_1m": round(atm_1m, 2), "atm_3m": round(atm_3m, 2),
                "atm_1y": round(atm_1y, 2),
                "rr25_3m": round(rr25, 2), "bf25_3m": round(bf25, 2),
                "ivrv_3m": round(ivrv_3m, 2),
                "atm3m_pct": round(atm3m_pct, 0),
                "rr3m_pct": round(rr3m_pct, 0),
                "z_atm3m": round(z_atm3m, 2),
                "term_spread": round(term_spread, 2),
                "signal": signal,
            })
        except Exception:
            rows.append({
                "pair": pair, "group": "—", "spot": 0, "chg_pct": 0,
                "atm_1m": 0, "atm_3m": 0, "atm_1y": 0,
                "rr25_3m": 0, "bf25_3m": 0, "ivrv_3m": 0,
                "atm3m_pct": 50, "rr3m_pct": 50, "z_atm3m": 0,
                "term_spread": 0, "signal": "—",
            })
    return rows


def _build_top_movers(rows):
    """5 top mover stat boxes."""
    if not rows:
        return []
    cheapest = min(rows, key=lambda r: r.get("atm3m_pct", 50))
    richest = max(rows, key=lambda r: r.get("atm3m_pct", 50))
    biggest_skew = max(rows, key=lambda r: abs(r.get("rr25_3m", 0)))
    biggest_ivrv = max(rows, key=lambda r: abs(r.get("ivrv_3m", 0)))
    n_signals = sum(1 for r in rows if r.get("signal", "—") != "—")

    items = [
        ("CHEAPEST VOL", f"{cheapest['pair']} {cheapest['atm3m_pct']:.0f}%ile", "#60a5fa"),
        ("RICHEST VOL", f"{richest['pair']} {richest['atm3m_pct']:.0f}%ile", "#ff3333"),
        ("BIGGEST SKEW", f"{biggest_skew['pair']} {biggest_skew['rr25_3m']:+.1f}", "#ff8800"),
        ("IV-RV GAP", f"{biggest_ivrv['pair']} {biggest_ivrv['ivrv_3m']:+.1f}", "#ff8800"),
        ("SIGNALS", f"{n_signals}/{len(rows)}", "#00cc66" if n_signals > 0 else "#808080"),
    ]
    boxes = []
    for label, value, color in items:
        boxes.append(html.Div([
            html.Div(value, style={"fontSize": "11px", "fontWeight": "700",
                                   "color": color, "fontFamily": _MONO}),
            html.Div(label, style={"fontSize": "8px", "color": "#808080",
                                   "letterSpacing": "1px", "fontFamily": _MONO}),
        ], style={**STAT_BOX_STYLE, "borderTop": f"2px solid {color}", "flex": "1"}))
    return boxes


# ═══════════════════════════════════════════════════════════════════════════
# HEATMAP VIEW — Percentile matrix (from vol_richness.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_heatmap(metric, lookback):
    """30×6 percentile heatmap."""
    try:
        from core.fx_analytics import vol_percentile
        from core.bloomberg_fx import get_fx_vol_surface

        vol_matrix = []
        pct_matrix = []
        for pair in ALL_PAIRS:
            vol_row = []
            pct_row = []
            surf = get_fx_vol_surface(pair) or {}
            for tenor in HEATMAP_TENORS:
                try:
                    p_info = vol_percentile(pair, tenor, metric, lookback)
                    if isinstance(p_info, dict):
                        vol_row.append(_sf(p_info.get("current", 0)))
                        pct_row.append(_sf(p_info.get("percentile", 50)))
                    else:
                        val = _extract_atm(surf, tenor) if metric == "ATM" else _extract_rr(surf, tenor)
                        vol_row.append(_sf(val))
                        pct_row.append(50.0)
                except Exception:
                    vol_row.append(0)
                    pct_row.append(50)
            vol_matrix.append(vol_row)
            pct_matrix.append(pct_row)

        z = np.array(pct_matrix)
        custom = np.array(vol_matrix)
        text = [[f"{custom[i][j]:.1f}<br><sub>{z[i][j]:.0f}%</sub>"
                 for j in range(len(HEATMAP_TENORS))] for i in range(len(ALL_PAIRS))]

        fig = go.Figure(go.Heatmap(
            z=z, x=HEATMAP_TENORS, y=ALL_PAIRS, customdata=custom,
            text=text, texttemplate="%{text}", textfont=dict(size=8),
            colorscale=RICHNESS_COLORSCALE, zmin=0, zmax=100,
            hovertemplate="<b>%{y}</b> %{x}<br>Vol: %{customdata:.2f}<br>Pctl: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Pctl", tickfont=dict(size=8), len=0.5),
            xgap=1, ygap=1,
        ))
        fig.update_layout(**_chart_layout(height=CHART_LG,
                          margin=dict(l=65, r=60, t=30, b=20),
                          title=dict(text=f"VOL RICHNESS — {metric} ({lookback}D)",
                                     font=dict(size=10, color="#808080")),
                          yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
                          xaxis=dict(tickfont=dict(size=9))))
        return fig
    except Exception:
        return _empty_fig("VOL RICHNESS")


def _build_cross_bar(metric, lookback):
    """Cross-pair bar chart sorted by 3M percentile."""
    try:
        from core.fx_analytics import vol_percentile
        data = []
        for pair in ALL_PAIRS:
            try:
                info = vol_percentile(pair, "3M", metric, lookback)
                pct = _sf(info.get("percentile", 50) if isinstance(info, dict) else 50)
                data.append({"pair": pair, "pct": pct})
            except Exception:
                data.append({"pair": pair, "pct": 50})

        data.sort(key=lambda d: d["pct"])
        pairs_l = [d["pair"] for d in data]
        pcts = [d["pct"] for d in data]
        colors = ["#0044ff" if p < 20 else "#ff3333" if p > 80 else "#808080" for p in pcts]

        fig = go.Figure(go.Bar(y=pairs_l, x=pcts, orientation="h",
                                marker_color=colors,
                                text=[f"{p:.0f}%" for p in pcts],
                                textposition="outside", textfont=dict(size=7)))
        fig.update_layout(**_chart_layout(height=CHART_MD,
                          margin=dict(l=55, r=20, t=25, b=10), showlegend=False,
                          title=dict(text=f"3M {metric} PERCENTILE RANK",
                                     font=dict(size=10, color="#808080")),
                          xaxis=dict(range=[0, 105], tickfont=dict(size=8)),
                          yaxis=dict(tickfont=dict(size=7))))
        return fig
    except Exception:
        return _empty_fig("CROSS-PAIR BAR")


# ── Heatmap drill-down charts ────────────────────────────────────────────────

def _build_atm_history(pair, tenor, lookback):
    """ATM vol history line chart."""
    try:
        from core.bloomberg_fx import get_fx_historical_vol
        from core.fx_analytics import vol_percentile
        hist = get_fx_historical_vol(pair, tenor, "ATM", 252)
        if hist is None or len(hist) < 10:
            from core.fx_analytics import _synth_vol_history
            hist = _synth_vol_history(pair, tenor, "ATM", 252)

        arr = np.array(hist, dtype=float) if not isinstance(hist, np.ndarray) else hist
        mean_v = float(np.nanmean(arr))
        std_v = float(np.nanstd(arr))
        x = list(range(len(arr)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[mean_v + std_v]*len(x),
                                 mode="lines", line=dict(color="#333355", width=1, dash="dot"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[mean_v - std_v]*len(x),
                                 mode="lines", line=dict(color="#333355", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(51,51,85,0.25)", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[mean_v]*len(x),
                                 mode="lines", line=dict(color="#808080", width=1, dash="dash"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=arr.tolist(), mode="lines",
                                 line=dict(color="#ff8800", width=2.5), name="ATM"))
        fig.add_trace(go.Scatter(x=[len(arr)-1], y=[float(arr[-1])],
                                 mode="markers", marker=dict(color="#ff8800", size=7),
                                 showlegend=False))
        fig.update_layout(**_chart_layout(height=CHART_SM,
                          margin=dict(l=40, r=10, t=25, b=15), showlegend=False,
                          title=dict(text=f"{pair} ATM {tenor} HISTORY",
                                     font=dict(size=9, color="#808080"))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} ATM HISTORY")


def _build_ivrv_chart(pair, tenor, lookback):
    """IV vs RV spread area chart."""
    try:
        from core.fx_analytics import iv_rv_spread
        df = iv_rv_spread(pair, tenor, lookback=lookback)
        if df is None or (hasattr(df, '__len__') and len(df) < 10):
            return _empty_fig(f"{pair} IV-RV")

        x = list(range(len(df)))
        iv = df["iv"].values if hasattr(df, 'columns') else np.zeros(len(df))
        rv = df["rv"].values if hasattr(df, 'columns') else np.zeros(len(df))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=iv, mode="lines",
                                 line=dict(color="#ff8800", width=2.5), name="IV"))
        fig.add_trace(go.Scatter(x=x, y=rv, mode="lines",
                                 line=dict(color="#00cc66", width=1.5), name="RV"))
        fig.update_layout(**_chart_layout(height=CHART_SM,
                          margin=dict(l=40, r=10, t=25, b=15), showlegend=True,
                          legend=dict(x=0.02, y=0.98, font=dict(size=8)),
                          title=dict(text=f"{pair} IV vs RV ({tenor})",
                                     font=dict(size=9, color="#808080"))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} IV-RV")


def _build_volcone_chart(pair, lookback):
    """Vol cone fan chart."""
    try:
        from core.fx_analytics import vol_cone
        cone = vol_cone(pair, lookback=lookback)
        if cone is None or (hasattr(cone, '__len__') and len(cone) < 2):
            return _empty_fig(f"{pair} VOL CONE")

        windows = cone["window"].values if hasattr(cone, 'columns') else list(range(len(cone)))
        x = [str(w) for w in windows]

        fig = go.Figure()
        if "p10" in cone.columns and "p90" in cone.columns:
            fig.add_trace(go.Scatter(x=x, y=cone["p90"].values, mode="lines",
                                     line=dict(color="rgba(26,26,46,0.5)", width=1),
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=cone["p10"].values, mode="lines",
                                     line=dict(color="rgba(26,26,46,0.5)", width=1),
                                     fill="tonexty", fillcolor="rgba(26,26,46,0.15)",
                                     showlegend=False))
        if "p25" in cone.columns and "p75" in cone.columns:
            fig.add_trace(go.Scatter(x=x, y=cone["p75"].values, mode="lines",
                                     line=dict(color="rgba(255,136,0,0.3)", width=1),
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=cone["p25"].values, mode="lines",
                                     line=dict(color="rgba(255,136,0,0.3)", width=1),
                                     fill="tonexty", fillcolor="rgba(255,136,0,0.08)",
                                     showlegend=False))
        if "p50" in cone.columns:
            fig.add_trace(go.Scatter(x=x, y=cone["p50"].values, mode="lines",
                                     line=dict(color="#808080", width=1, dash="dot"),
                                     name="Median"))
        if "current" in cone.columns:
            fig.add_trace(go.Scatter(x=x, y=cone["current"].values, mode="lines+markers",
                                     line=dict(color="#ff8800", width=2),
                                     marker=dict(size=4), name="Current RV"))

        fig.update_layout(**_chart_layout(height=CHART_SM,
                          margin=dict(l=40, r=10, t=25, b=15), showlegend=True,
                          legend=dict(x=0.02, y=0.98, font=dict(size=8)),
                          title=dict(text=f"{pair} VOL CONE",
                                     font=dict(size=9, color="#808080"))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} VOL CONE")


def _build_drill_stats(pair, tenor, lookback):
    """5 stat boxes for heatmap drill-down."""
    try:
        from core.fx_analytics import vol_percentile, iv_rv_spread, breakeven_vol, vol_regime_detect

        pct_info = vol_percentile(pair, tenor, "ATM", lookback)
        pct = _sf(pct_info.get("percentile", 50) if isinstance(pct_info, dict) else 50)
        pct_color = "#ff3333" if pct > 80 else "#60a5fa" if pct < 20 else "#ff8800"

        try:
            from core.fx_analytics import iv_rv_percentile
            ivrv_info = iv_rv_percentile(pair, tenor, lookback=lookback)
            ivrv_pct = _sf(ivrv_info.get("percentile", 50) if isinstance(ivrv_info, dict) else 50)
        except Exception:
            ivrv_pct = 50

        try:
            be = breakeven_vol(pair, tenor, tenor_to_days(tenor))
            be_rv = _sf(be.get("breakeven_rv", 0) if isinstance(be, dict) else 0)
            cushion = _sf(be.get("iv_rv_cushion", 0) if isinstance(be, dict) else 0)
        except Exception:
            be_rv = 0
            cushion = 0

        try:
            regime = vol_regime_detect(pair)
            regime_label = regime.get("regime", "NORMAL") if isinstance(regime, dict) else "NORMAL"
            regime_color = regime.get("color", "#d4d4d4") if isinstance(regime, dict) else "#d4d4d4"
        except Exception:
            regime_label = "NORMAL"
            regime_color = "#d4d4d4"

        items = [
            ("IV %ILE", f"{pct:.0f}%", pct_color),
            ("IV-RV %ILE", f"{ivrv_pct:.0f}%", "#ff8800"),
            ("BRKEVN RV", f"{be_rv:.1f}v", "#00cc66" if cushion > 0 else "#ff3333"),
            ("CARRY/DAY", f"{cushion:.2f}v", "#ff8800"),
            ("REGIME", regime_label, regime_color),
        ]
        boxes = []
        for label, value, color in items:
            boxes.append(html.Div([
                html.Div(value, style={"fontSize": "11px", "fontWeight": "700",
                                       "color": color, "fontFamily": _MONO}),
                html.Div(label, style={"fontSize": "8px", "color": "#808080",
                                       "letterSpacing": "1px", "fontFamily": _MONO}),
            ], style={**STAT_BOX_STYLE, "marginBottom": "4px"}))
        return boxes
    except Exception:
        return [html.Div("Error loading stats", style={"color": "#808080", "fontSize": "10px"})]


# ═══════════════════════════════════════════════════════════════════════════
# SKEW VIEW — RR surface, smile, wings, BF, tails (from skew_lab.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_skew_surface():
    """30×6 RR heatmap."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        z_vals = []
        text_vals = []
        for pair in ALL_PAIRS:
            row_z = []
            row_t = []
            surf = get_fx_vol_surface(pair) or {}
            for tenor in SURFACE_TENORS:
                rr = _extract_rr(surf, tenor)
                row_z.append(np.clip(rr, -3, 3))
                row_t.append(f"{rr:+.2f}")
            z_vals.append(row_z)
            text_vals.append(row_t)

        fig = go.Figure(go.Heatmap(
            z=z_vals, x=SURFACE_TENORS, y=ALL_PAIRS, text=text_vals,
            texttemplate="%{text}", textfont=dict(size=8),
            colorscale=SKEW_COLORSCALE, zmin=-3, zmax=3,
            hovertemplate="<b>%{y}</b> %{x}<br>25D RR: %{z:+.2f}<extra></extra>",
            colorbar=dict(title="RR", tickfont=dict(size=8), len=0.5),
            xgap=1, ygap=1,
        ))
        fig.update_layout(**_chart_layout(height=CHART_LG,
                          margin=dict(l=65, r=60, t=30, b=20),
                          title=dict(text="25D RISK REVERSAL SURFACE",
                                     font=dict(size=10, color="#808080")),
                          yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
                          xaxis=dict(tickfont=dict(size=9))))
        return fig
    except Exception:
        return _empty_fig("SKEW SURFACE")


def _build_rr_spot_chart(pair, tenor):
    """RR vs spot dual-axis chart."""
    try:
        from core.bloomberg_fx import get_fx_historical_vol, get_fx_historical_spot
        rr_hist = get_fx_historical_vol(pair, tenor, "25D_RR", 252)
        spot_hist = get_fx_historical_spot(pair, 252)

        if rr_hist is None:
            from core.fx_analytics import _synth_vol_history
            rr_hist = _synth_vol_history(pair, tenor, "25D_RR", 252)

        rr_arr = np.array(rr_hist, dtype=float) if not isinstance(rr_hist, np.ndarray) else rr_hist

        # Extract spot series
        if spot_hist is not None:
            if hasattr(spot_hist, 'values'):
                if hasattr(spot_hist, 'columns') and 'close' in spot_hist.columns:
                    spot_arr = spot_hist['close'].values.astype(float)
                else:
                    spot_arr = spot_hist.values.flatten().astype(float)
            else:
                spot_arr = np.array(spot_hist, dtype=float)
        else:
            spot_arr = np.ones(len(rr_arr))

        min_len = min(len(rr_arr), len(spot_arr))
        rr_arr = rr_arr[-min_len:]
        spot_arr = spot_arr[-min_len:]
        x = list(range(min_len))

        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=x, y=rr_arr, mode="lines",
                                 line=dict(color="#ff8800", width=2.5), name="25D RR"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=spot_arr, mode="lines",
                                 line=dict(color="#ffffff", width=1, dash="dot"), name="Spot"),
                      secondary_y=True)
        fig.update_layout(**_chart_layout(height=CHART_SM,
                          margin=dict(l=50, r=50, t=25, b=15), showlegend=True,
                          legend=dict(x=0.02, y=0.98, font=dict(size=8)),
                          title=dict(text=f"{pair} RR vs SPOT ({tenor})",
                                     font=dict(size=9, color="#808080"))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} RR vs SPOT")


def _build_smile_comparison(pair):
    """Smile shape overlay for 1M/3M/1Y using proper FX smile construction."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        surf = get_fx_vol_surface(pair) or {}

        tenors = ["1M", "3M", "1Y"]
        colors = ["#ff8800", "#ffffff", "#00cc66"]
        delta_labels = ["10P", "25P", "ATM", "25C", "10C"]

        fig = go.Figure()
        for tenor, color in zip(tenors, colors):
            atm = _extract_atm(surf, tenor)
            rr25 = _extract_rr(surf, tenor)
            bf25 = _extract_bf(surf, tenor)
            t = surf.get(tenor, {})
            rr10 = _sf(t.get("rr10", t.get("10D_RR", rr25 * 1.8)))
            bf10 = _sf(t.get("bf10", t.get("10D_BF", bf25 * 1.6)))
            if atm == 0:
                continue
            # Proper FX smile: BF adjusts the wings symmetrically, RR tilts them
            c25 = atm + bf25 + rr25 / 2
            p25 = atm + bf25 - rr25 / 2
            c10 = atm + bf10 + rr10 / 2
            p10 = atm + bf10 - rr10 / 2
            vols = [p10, p25, atm, c25, c10]
            fig.add_trace(go.Scatter(x=delta_labels, y=vols, mode="lines+markers",
                                     line=dict(color=color, width=1.5),
                                     marker=dict(size=4), name=tenor))

        fig.update_layout(**_chart_layout(height=CHART_SM,
                          margin=dict(l=40, r=10, t=25, b=15), showlegend=True,
                          legend=dict(x=0.02, y=0.98, font=dict(size=8)),
                          title=dict(text=f"{pair} SMILE COMPARISON",
                                     font=dict(size=9, color="#808080"))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} SMILE")


def _build_bf_map():
    """25D BF across all pairs, horizontal bars."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        data = []
        for pair in ALL_PAIRS:
            try:
                surf = get_fx_vol_surface(pair) or {}
                bf = _extract_bf(surf, "3M")
                data.append({"pair": pair, "bf": bf})
            except Exception:
                continue

        if not data:
            return _empty_fig("BUTTERFLY MAP")

        data.sort(key=lambda d: d["bf"])
        pairs_l = [d["pair"] for d in data]
        bfs = [d["bf"] for d in data]
        colors = ["#ff3333" if b > 0.8 else "#ff8800" if b > 0.4 else "#808080" for b in bfs]

        fig = go.Figure(go.Bar(y=pairs_l, x=bfs, orientation="h",
                                marker_color=colors,
                                text=[f"{b:.2f}" for b in bfs],
                                textposition="outside", textfont=dict(size=7)))
        fig.update_layout(**_chart_layout(height=CHART_MD,
                          margin=dict(l=55, r=20, t=25, b=10), showlegend=False,
                          title=dict(text="25D BUTTERFLY (3M)",
                                     font=dict(size=10, color="#808080")),
                          xaxis=dict(tickfont=dict(size=8)),
                          yaxis=dict(tickfont=dict(size=7))))
        return fig
    except Exception:
        return _empty_fig("BUTTERFLY MAP")


def _build_tail_table(pair, tenor):
    """Tail probability table."""
    try:
        from core.fx_analytics import tail_probabilities
        header = html.Tr([html.Th(h, style={**TABLE_HEADER_STYLE, "fontSize": "8px"})
                          for h in ["MOVE", "UP %", "DOWN %"]])
        body = []
        for move in TAIL_MOVES:
            try:
                tp = tail_probabilities(pair, tenor, move)
                if isinstance(tp, dict):
                    up = _sf(tp.get("prob_up", 0))
                    dn = _sf(tp.get("prob_down", 0))
                elif hasattr(tp, 'iloc'):
                    up = _sf(tp["prob_up"].iloc[0] if "prob_up" in tp.columns else 0)
                    dn = _sf(tp["prob_down"].iloc[0] if "prob_down" in tp.columns else 0)
                else:
                    up, dn = 0, 0
            except Exception:
                up, dn = 0, 0

            up_color = "#ff3333" if up > 15 else "#ff8800" if up > 5 else "#808080"
            dn_color = "#ff3333" if dn > 15 else "#ff8800" if dn > 5 else "#808080"
            body.append(html.Tr([
                html.Td(f"±{move}%", style={**TABLE_CELL_STYLE, "fontWeight": "600"}),
                html.Td(f"{up:.1f}%", style={**TABLE_CELL_STYLE, "color": up_color}),
                html.Td(f"{dn:.1f}%", style={**TABLE_CELL_STYLE, "color": dn_color}),
            ]))
        return html.Table([html.Thead(header), html.Tbody(body)],
                          style={"width": "100%", "borderCollapse": "collapse",
                                 "fontFamily": _MONO, "fontSize": "10px"})
    except Exception:
        return html.Div("No tail data", style={"color": "#808080", "fontSize": "10px"})


# ═══════════════════════════════════════════════════════════════════════════
# TABLE VIEW — Sparkline drill-down charts
# ═══════════════════════════════════════════════════════════════════════════

def _spark_term_structure(pair):
    """Term structure snapshot for drill-down."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        surf = get_fx_vol_surface(pair) or {}
        tenors = ["1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]
        vols = [_extract_atm(surf, t) for t in tenors]
        vols = [v for v in vols if v > 0]
        labels = tenors[:len(vols)]

        if not vols:
            return _empty_fig(f"{pair} TERM")

        fig = go.Figure(go.Scatter(x=labels, y=vols, mode="lines+markers",
                                    line=dict(color="#ff8800", width=2.5),
                                    marker=dict(size=5)))
        fig.update_layout(**_chart_layout(height=CHART_SM,
                          margin=dict(l=40, r=10, t=20, b=15), showlegend=False,
                          title=dict(text=f"{pair} TERM STRUCTURE",
                                     font=dict(size=9, color="#808080"))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} TERM")


# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

def layout():
    return html.Div([
        # Stores
        dcc.Store(id=f"{_P}-drill-pair", data="EURUSD"),
        dcc.Store(id=f"{_P}-drill-tenor", data="3M"),
        dcc.Store(id=f"{_P}-scanner-store", data=[]),
        dcc.Interval(id=f"{_P}-interval", interval=60_000, n_intervals=0),

        # ── Title + Controls ──
        html.Div([
            html.Span("VOL SCANNER", style={
                "color": "#ffffff", "fontSize": "13px", "fontWeight": "700",
                "letterSpacing": "2px", "fontFamily": _MONO,
            }),
            html.Div([
                dcc.Dropdown(id=f"{_P}-group", options=GROUP_OPTIONS, value="ALL",
                             clearable=False, style={**DROPDOWN_STYLE, "width": "110px"}),
                dcc.Dropdown(id=f"{_P}-lookback", options=LOOKBACK_OPTIONS, value=252,
                             clearable=False, style={**DROPDOWN_STYLE, "width": "80px"}),
                dcc.Dropdown(id=f"{_P}-signal-filter", options=SIGNAL_OPTIONS, value="ALL",
                             clearable=False, style={**DROPDOWN_STYLE, "width": "120px"}),
                dcc.Dropdown(id=f"{_P}-heatmap-metric", options=METRIC_OPTIONS, value="ATM",
                             clearable=False, style={**DROPDOWN_STYLE, "width": "90px"}),
            ], style={"display": "flex", "gap": GAP}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "padding": "8px 0",
                  "borderBottom": "1px solid #222240"}),

        # ── View Tabs ──
        dcc.Tabs(id=f"{_P}-view-tabs", value="table", children=[
            dcc.Tab(label="TABLE", value="table", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="HEATMAP", value="heatmap", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="SKEW", value="skew", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ], style={"marginTop": GAP, "marginBottom": "0"}),

        # ── Table View Container ──
        html.Div(id=f"{_P}-table-container", children=[
            # Top movers
            html.Div(id=f"{_P}-top-movers", style={
                "display": "flex", "gap": GAP, "padding": "8px 0", "flexWrap": "wrap",
            }),
            # DataTable
            html.Div(id=f"{_P}-table-wrapper", style={"marginTop": GAP}),
            # Drill-down
            html.Div(id=f"{_P}-table-drilldown", style={
                "display": "none", "marginTop": "8px", "border": "1px solid #222240",
                "padding": "8px",
            }, children=[
                html.Div(id=f"{_P}-drilldown-title", style={
                    "color": "#ff8800", "fontSize": "11px", "fontWeight": "700",
                    "marginBottom": "8px", "fontFamily": _MONO,
                }),
                html.Div([
                    dcc.Graph(id=f"{_P}-spark-atm", config={"displayModeBar": False, "responsive": True},
                              style={"flex": "1", "minWidth": "200px"}),
                    dcc.Graph(id=f"{_P}-spark-ivrv", config={"displayModeBar": False, "responsive": True},
                              style={"flex": "1", "minWidth": "200px"}),
                    dcc.Graph(id=f"{_P}-spark-term", config={"displayModeBar": False, "responsive": True},
                              style={"flex": "1", "minWidth": "200px"}),
                ], style={"display": "flex", "gap": GAP}),
            ]),
        ]),

        # ── Heatmap View Container ──
        html.Div(id=f"{_P}-heatmap-container", style={"display": "none"}, children=[
            html.Div([
                # Main heatmap
                html.Div([
                    dcc.Graph(id=f"{_P}-heatmap", config={"displayModeBar": False, "responsive": True},
                              style={"height": f"{CHART_LG}px"}),
                ], style={"flex": "3"}),
                # Cross-pair bar
                html.Div([
                    dcc.Graph(id=f"{_P}-cross-bar", config={"displayModeBar": False, "responsive": True},
                              style={"height": f"{CHART_MD}px"}),
                ], style={"flex": "2"}),
            ], style={"display": "flex", "gap": GAP}),

            # Drill-down section
            html.Div(id=f"{_P}-hm-drill", style={
                "marginTop": SECTION_GAP, "border": "1px solid #222240", "padding": GAP,
            }, children=[
                html.Div(id=f"{_P}-drill-label", style={
                    "color": "#ff8800", "fontSize": "11px", "fontWeight": "700",
                    "marginBottom": GAP, "fontFamily": _MONO,
                }),
                html.Div([
                    dcc.Graph(id=f"{_P}-drill-atm", config={"displayModeBar": False, "responsive": True},
                              style={"flex": "1", "minWidth": "200px"}),
                    dcc.Graph(id=f"{_P}-drill-ivrv", config={"displayModeBar": False, "responsive": True},
                              style={"flex": "1", "minWidth": "200px"}),
                    dcc.Graph(id=f"{_P}-drill-cone", config={"displayModeBar": False, "responsive": True},
                              style={"flex": "1", "minWidth": "200px"}),
                    html.Div(id=f"{_P}-drill-stats", style={"minWidth": "120px"}),
                ], style={"display": "flex", "gap": GAP}),
            ]),
        ]),

        # ── Skew View Container ──
        html.Div(id=f"{_P}-skew-container", style={"display": "none"}, children=[
            # Skew surface
            dcc.Graph(id=f"{_P}-skew-surface", config={"displayModeBar": False, "responsive": True},
                      style={"height": f"{CHART_LG}px"}),

            # RR vs Spot + Smile comparison
            html.Div([
                html.Div([
                    html.Span("PAIR", style=LABEL_STYLE),
                    dcc.Dropdown(id=f"{_P}-skew-pair", options=[{"label": p, "value": p} for p in ALL_PAIRS],
                                 value="EURUSD", clearable=False,
                                 style={**DROPDOWN_STYLE, "width": "120px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": GAP,
                          "padding": "4px 0"}),
                html.Div([
                    dcc.Graph(id=f"{_P}-rr-spot", config={"displayModeBar": False, "responsive": True},
                              style={"flex": "1", "minWidth": "300px"}),
                    dcc.Graph(id=f"{_P}-smile", config={"displayModeBar": False, "responsive": True},
                              style={"flex": "1", "minWidth": "300px"}),
                ], style={"display": "flex", "gap": GAP}),
            ]),

            # BF map + Tail table
            html.Div([
                dcc.Graph(id=f"{_P}-bf-map", config={"displayModeBar": False, "responsive": True},
                          style={"flex": "3", "height": f"{CHART_MD}px"}),
                html.Div([
                    html.Div("TAIL PROBABILITIES", style={
                        "color": "#808080", "fontSize": "9px", "fontWeight": "700",
                        "letterSpacing": "1.5px", "marginBottom": GAP, "fontFamily": _MONO,
                    }),
                    html.Div(id=f"{_P}-tail-table"),
                ], style={"flex": "2", "padding": GAP, "border": "1px solid #222240"}),
            ], style={"display": "flex", "gap": GAP, "marginTop": GAP}),
        ]),

    ], style={"fontFamily": _MONO})


# ═══════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app):

    # ── View tab visibility ──
    @app.callback(
        [
            Output(f"{_P}-table-container", "style"),
            Output(f"{_P}-heatmap-container", "style"),
            Output(f"{_P}-skew-container", "style"),
        ],
        Input(f"{_P}-view-tabs", "value"),
    )
    def toggle_views(view):
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if view == "table" else hide,
            show if view == "heatmap" else hide,
            show if view == "skew" else hide,
        )

    # ── TABLE VIEW: update table ──
    @app.callback(
        [
            Output(f"{_P}-top-movers", "children"),
            Output(f"{_P}-table-wrapper", "children"),
            Output(f"{_P}-scanner-store", "data"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-group", "value"),
            Input(f"{_P}-lookback", "value"),
            Input(f"{_P}-signal-filter", "value"),
        ],
    )
    def update_table(n, group, lookback, signal_filter):
        pairs = _pairs_for(group or "ALL")
        rows = _build_scanner_rows(pairs, lookback or 252)

        # Filter by signal
        if signal_filter and signal_filter != "ALL":
            filtered = [r for r in rows if signal_filter.replace("_", " ") in r.get("signal", "")]
            if not filtered:
                filtered = rows
        else:
            filtered = rows

        movers = _build_top_movers(filtered)

        # Build DataTable columns
        columns = [
            {"name": "PAIR", "id": "pair"},
            {"name": "GRP", "id": "group"},
            {"name": "SPOT", "id": "spot", "type": "numeric"},
            {"name": "CHG%", "id": "chg_pct", "type": "numeric"},
            {"name": "ATM 1M", "id": "atm_1m", "type": "numeric"},
            {"name": "ATM 3M", "id": "atm_3m", "type": "numeric"},
            {"name": "ATM 1Y", "id": "atm_1y", "type": "numeric"},
            {"name": "RR25", "id": "rr25_3m", "type": "numeric"},
            {"name": "BF25", "id": "bf25_3m", "type": "numeric"},
            {"name": "IV-RV", "id": "ivrv_3m", "type": "numeric"},
            {"name": "%ILE", "id": "atm3m_pct", "type": "numeric"},
            {"name": "RR %ILE", "id": "rr3m_pct", "type": "numeric"},
            {"name": "Z", "id": "z_atm3m", "type": "numeric"},
            {"name": "TERM", "id": "term_spread", "type": "numeric"},
            {"name": "SIGNAL", "id": "signal"},
        ]

        table = dash_table.DataTable(
            id=f"{_P}-datatable",
            columns=columns,
            data=filtered,
            sort_action="native",
            filter_action="native",
            page_size=30,
            row_selectable="single",
            style_table={"overflowX": "auto", "maxHeight": "450px", "overflowY": "auto"},
            style_header={
                "backgroundColor": "#000000", "color": "#808080",
                "fontWeight": "700", "fontSize": "9px", "textTransform": "uppercase",
                "letterSpacing": "1px", "border": "1px solid #222240",
                "fontFamily": _MONO,
            },
            style_cell={
                "backgroundColor": "#000000", "color": "#d4d4d4",
                "fontSize": "11px", "fontFamily": _MONO,
                "border": "1px solid #222240", "padding": "4px 6px",
                "textAlign": "right", "minWidth": "60px",
            },
            style_cell_conditional=[
                {"if": {"column_id": "pair"}, "textAlign": "left", "fontWeight": "700"},
                {"if": {"column_id": "group"}, "textAlign": "center", "color": "#808080"},
                {"if": {"column_id": "spot"}, "color": "#888888"},
                {"if": {"column_id": "signal"}, "textAlign": "left", "fontSize": "9px"},
            ],
            style_data_conditional=[
                # Spot change coloring
                {"if": {"filter_query": "{chg_pct} > 0", "column_id": "chg_pct"},
                 "color": "#34d399"},
                {"if": {"filter_query": "{chg_pct} < 0", "column_id": "chg_pct"},
                 "color": "#f87171"},
                # ATM percentile coloring
                {"if": {"filter_query": "{atm3m_pct} < 20", "column_id": "atm3m_pct"},
                 "color": "#60a5fa", "fontWeight": "bold"},
                {"if": {"filter_query": "{atm3m_pct} > 80", "column_id": "atm3m_pct"},
                 "color": "#f87171", "fontWeight": "bold"},
                # RR percentile coloring
                {"if": {"filter_query": "{rr3m_pct} < 20", "column_id": "rr3m_pct"},
                 "color": "#60a5fa", "fontWeight": "bold"},
                {"if": {"filter_query": "{rr3m_pct} > 80", "column_id": "rr3m_pct"},
                 "color": "#f87171", "fontWeight": "bold"},
                # Z-score coloring
                {"if": {"filter_query": "{z_atm3m} > 2", "column_id": "z_atm3m"},
                 "color": "#f87171", "fontWeight": "bold"},
                {"if": {"filter_query": "{z_atm3m} < -2", "column_id": "z_atm3m"},
                 "color": "#60a5fa", "fontWeight": "bold"},
                # Signal coloring
                {"if": {"filter_query": '{signal} contains "CHEAP"', "column_id": "signal"},
                 "color": "#34d399", "borderLeft": "2px solid #34d399"},
                {"if": {"filter_query": '{signal} contains "RICH"', "column_id": "signal"},
                 "color": "#f87171", "borderLeft": "2px solid #f87171"},
                {"if": {"filter_query": '{signal} contains "SKEW"', "column_id": "signal"},
                 "color": "#fbbf24", "borderLeft": "2px solid #fbbf24"},
            ],
        )

        return movers, table, rows

    # ── TABLE VIEW: drill-down on row select ──
    @app.callback(
        [
            Output(f"{_P}-table-drilldown", "style"),
            Output(f"{_P}-drilldown-title", "children"),
            Output(f"{_P}-spark-atm", "figure"),
            Output(f"{_P}-spark-ivrv", "figure"),
            Output(f"{_P}-spark-term", "figure"),
        ],
        Input(f"{_P}-datatable", "selected_rows"),
        State(f"{_P}-scanner-store", "data"),
        prevent_initial_call=True,
    )
    def table_drilldown(selected, store):
        if not selected or not store:
            raise PreventUpdate
        idx = selected[0]
        if idx >= len(store):
            raise PreventUpdate
        row = store[idx]
        pair = row.get("pair", "EURUSD")

        return (
            {"display": "block", "marginTop": "8px", "border": "1px solid #222240", "padding": "8px"},
            f"{pair} — DRILL-DOWN",
            _build_atm_history(pair, "3M", 252),
            _build_ivrv_chart(pair, "3M", 252),
            _spark_term_structure(pair),
        )

    # ── HEATMAP VIEW: build heatmap + cross bar ──
    @app.callback(
        [
            Output(f"{_P}-heatmap", "figure"),
            Output(f"{_P}-cross-bar", "figure"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-heatmap-metric", "value"),
            Input(f"{_P}-lookback", "value"),
            Input(f"{_P}-view-tabs", "value"),
        ],
    )
    def update_heatmap(n, metric, lookback, view):
        if view != "heatmap":
            raise PreventUpdate
        return _build_heatmap(metric or "ATM", lookback or 252), _build_cross_bar(metric or "ATM", lookback or 252)

    # ── HEATMAP VIEW: click → drill pair/tenor ──
    @app.callback(
        [Output(f"{_P}-drill-pair", "data"), Output(f"{_P}-drill-tenor", "data")],
        Input(f"{_P}-heatmap", "clickData"),
        prevent_initial_call=True,
    )
    def heatmap_click(click_data):
        if not click_data or "points" not in click_data:
            raise PreventUpdate
        pt = click_data["points"][0]
        pair = pt.get("y", "EURUSD")
        tenor = pt.get("x", "3M")
        if pair in ALL_PAIRS and tenor in HEATMAP_TENORS:
            return pair, tenor
        raise PreventUpdate

    # ── HEATMAP VIEW: drill-down charts + stats ──
    @app.callback(
        [
            Output(f"{_P}-drill-label", "children"),
            Output(f"{_P}-drill-atm", "figure"),
            Output(f"{_P}-drill-ivrv", "figure"),
            Output(f"{_P}-drill-cone", "figure"),
            Output(f"{_P}-drill-stats", "children"),
        ],
        [Input(f"{_P}-drill-pair", "data"), Input(f"{_P}-drill-tenor", "data")],
        State(f"{_P}-lookback", "value"),
    )
    def heatmap_drilldown(pair, tenor, lookback):
        pair = pair or "EURUSD"
        tenor = tenor or "3M"
        lookback = lookback or 252
        return (
            f"{pair} — {tenor} — {lookback}D DRILL-DOWN",
            _build_atm_history(pair, tenor, lookback),
            _build_ivrv_chart(pair, tenor, lookback),
            _build_volcone_chart(pair, lookback),
            _build_drill_stats(pair, tenor, lookback),
        )

    # ── SKEW VIEW: surface + BF map (on interval or view switch) ──
    @app.callback(
        [Output(f"{_P}-skew-surface", "figure"), Output(f"{_P}-bf-map", "figure")],
        [Input(f"{_P}-interval", "n_intervals"), Input(f"{_P}-view-tabs", "value")],
    )
    def update_skew_surface(n, view):
        if view != "skew":
            raise PreventUpdate
        return _build_skew_surface(), _build_bf_map()

    # ── SKEW VIEW: drill-down (RR vs spot, smile, tails) ──
    @app.callback(
        [
            Output(f"{_P}-rr-spot", "figure"),
            Output(f"{_P}-smile", "figure"),
            Output(f"{_P}-tail-table", "children"),
        ],
        [Input(f"{_P}-skew-pair", "value")],
    )
    def skew_drilldown(pair):
        pair = pair or "EURUSD"
        return (
            _build_rr_spot_chart(pair, "3M"),
            _build_smile_comparison(pair),
            _build_tail_table(pair, "3M"),
        )

    # ── SKEW VIEW: surface click → pair selector ──
    @app.callback(
        Output(f"{_P}-skew-pair", "value", allow_duplicate=True),
        Input(f"{_P}-skew-surface", "clickData"),
        prevent_initial_call=True,
    )
    def skew_surface_click(click_data):
        if not click_data or "points" not in click_data:
            raise PreventUpdate
        pair = click_data["points"][0].get("y", "")
        if pair in ALL_PAIRS:
            return pair
        raise PreventUpdate
        raise PreventUpdate
