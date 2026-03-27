"""
Relative Value Plus — RISK & ANALYTICS workspace
=================================================
Replaces: Relative Value + Correlation Monitor + Macro Regime + Vol Carry (4 → 1).

Four sub-tabs:
  [CROSS-PAIR]   Vol spread TS, term structure, skew scatter, z-score matrix, IV-RV, correlation
  [CORRELATION]  15×15 matrix, regime badges, breakdown table, rolling chart
  [MACRO]        Rate differentials, DXY proxy, risk sentiment, CB divergence, carry scatter
  [CARRY]        Carry table, term structure overlay, forward vol, calendar spread
"""

import json
import logging
import numpy as np
import pandas as pd
from dash import html, dcc, Input, Output, State, callback_context, ALL, MATCH, dash_table, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, TAB_STYLE, TAB_SELECTED_STYLE,
    TABLE_HEADER_STYLE, TABLE_CELL_STYLE, clickable_stat, make_stat_style,
    GAP, SECTION_GAP, CHART_SM, CHART_MD, CHART_LG, CSV_BTN_STYLE,
    no_data_fig,
)
from core.csv_export import export_csv
from core.fx_conventions import FX_PAIR_REGISTRY, tenor_to_days, tenor_to_years

# ── Constants ────────────────────────────────────────────────────────────────

_P = "rvp"  # prefix
_MONO = "'JetBrains Mono', monospace"

ALL_PAIRS = sorted(FX_PAIR_REGISTRY.keys())
G10_PAIRS = [p for p in ALL_PAIRS if FX_PAIR_REGISTRY[p].group == "G10"]
EM_PAIRS  = [p for p in ALL_PAIRS if p not in G10_PAIRS]

MONITOR_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD",
    "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY",
    "USDMXN", "USDZAR", "USDTRY", "USDCNH", "USDSGD",
]

HEATMAP_TENORS = ["1M", "3M", "6M", "1Y"]
TERM_TENORS = ["1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]

LOOKBACK_OPTIONS = [
    {"label": "60D",  "value": 60},
    {"label": "120D", "value": 120},
    {"label": "252D", "value": 252},
    {"label": "504D", "value": 504},
]

_CB_BANKS = {
    "FED":  {"rate": 4.25, "direction": "HOLD"},
    "ECB":  {"rate": 2.75, "direction": "HOLD"},
    "BOE":  {"rate": 4.25, "direction": "CUTTING"},
    "BOJ":  {"rate": 0.50, "direction": "HIKING"},
    "SNB":  {"rate": 0.75, "direction": "HOLD"},
    "RBA":  {"rate": 3.85, "direction": "HOLD"},
    "RBNZ": {"rate": 3.75, "direction": "CUTTING"},
    "BOC":  {"rate": 2.75, "direction": "HOLD"},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sf(v, d=0.0):
    try:
        f = float(v)
        return d if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return d


def _chart_layout(**overrides):
    """Merge CHART_TEMPLATE with overrides including deep-merged axes."""
    from core.theme import chart_layout
    return chart_layout(**overrides)


def _empty_fig(title=""):
    fig = go.Figure()
    for y in [0.2, 0.4, 0.6, 0.8]:
        fig.add_shape(type="line", x0=0, x1=1, y0=y, y1=y,
                      xref="paper", yref="paper", line=dict(color="#0d0d1a", width=1))
    fig.update_layout(**_chart_layout(
        height=220, margin=dict(l=20, r=10, t=30, b=10),
        title=dict(text=title, font=dict(size=10, color="#808080")),
        annotations=[dict(text="LOADING", x=0.5, y=0.5, showarrow=False,
                          font=dict(color="#333355", size=10, family="'JetBrains Mono', monospace"),
                          xref="paper", yref="paper")]))
    return fig


def _extract_atm(surface, tenor):
    t = surface.get(tenor, {})
    return _sf(t.get("atm", 0))


def _extract_rr(surface, tenor):
    t = surface.get(tenor, {})
    return _sf(t.get("rr25", t.get("25D_RR", 0)))


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-PAIR TAB — Charts (from relative_value.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_vol_spread_ts(pair_a, pair_b, tenor, lookback):
    """Vol spread time series with z-score overlay."""
    try:
        from core.bloomberg_fx import get_fx_historical_vol
        from core.fx_analytics import vol_zscore, vol_percentile

        hist_a = get_fx_historical_vol(pair_a, tenor, "ATM", lookback)
        hist_b = get_fx_historical_vol(pair_b, tenor, "ATM", lookback)
        if hist_a is None or hist_b is None:
            return no_data_fig(msg="NO VOL HISTORY")

        a = np.array(hist_a, dtype=float)
        b = np.array(hist_b, dtype=float)
        min_len = min(len(a), len(b))
        a, b = a[-min_len:], b[-min_len:]
        spread = a - b
        x = list(range(min_len))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=x, y=a, mode="lines",
                                 line=dict(color="#ff8800", width=2.5), name=pair_a,
                                 hovertemplate="Day %{x}<br>" + pair_a + ": %{y:.2f}v<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=b, mode="lines",
                                 line=dict(color="#ff3333", width=1.5), name=pair_b,
                                 hovertemplate="Day %{x}<br>" + pair_b + ": %{y:.2f}v<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=a, mode="lines", line=dict(width=0), showlegend=False),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=b, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(255,136,0,0.06)", showlegend=False),
                      secondary_y=False)

        # Z-score line
        if min_len > 20:
            z_line = (spread - np.mean(spread)) / max(np.std(spread), 1e-6)
            fig.add_trace(go.Scatter(x=x, y=z_line, mode="lines",
                                     line=dict(color="#ffffff", width=1, dash="dash"), name="Z-Score",
                                     hovertemplate="Day %{x}<br>Z-Score: %{y:+.2f}\u03c3<extra></extra>"),
                          secondary_y=True)
            fig.add_hline(y=2, line=dict(color="#ff3333", width=0.5, dash="dot"), secondary_y=True)
            fig.add_hline(y=-2, line=dict(color="#1565c0", width=0.5, dash="dot"), secondary_y=True)

        fig.update_layout(**_chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=50, t=30, b=20),
                          title=dict(text=f"VOL SPREAD: {pair_a} vs {pair_b} ({tenor})",
                                     font=dict(size=10, color="#808080")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return fig
    except Exception:
        return _empty_fig("VOL SPREAD")


def _build_zscore_matrix(lookback):
    """Z-score heatmap across all pairs × key tenors."""
    try:
        from core.fx_analytics import vol_zscore
        z_matrix = []
        for pair in ALL_PAIRS:
            row = []
            for tenor in HEATMAP_TENORS:
                try:
                    info = vol_zscore(pair, tenor, "ATM", lookback)
                    z = _sf(info.get("zscore", 0) if isinstance(info, dict) else info)
                except Exception:
                    z = 0
                row.append(np.clip(z, -3, 3))
            z_matrix.append(row)

        z = np.array(z_matrix)
        text = [[f"{z[i][j]:+.1f}" for j in range(len(HEATMAP_TENORS))]
                for i in range(len(ALL_PAIRS))]

        fig = go.Figure(go.Heatmap(
            z=z, x=HEATMAP_TENORS, y=ALL_PAIRS, text=text,
            texttemplate="%{text}", textfont=dict(size=9, color="#c0c0c0"),
            colorscale=[[0, "#1565c0"], [0.25, "#0a1628"],
                        [0.50, "#0e0e0e"],
                        [0.75, "#2a1200"], [1.0, "#ff3333"]],
            zmin=-3, zmax=3,
            hovertemplate="<b>%{y}</b> %{x}<br>Z: %{z:+.2f}<extra></extra>",
            colorbar=dict(
                title=dict(text="Z", font=dict(size=9, color="#808080")),
                tickfont=dict(size=8, color="#808080"),
                len=0.6, thickness=10, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
            ),
            xgap=2, ygap=2,
        ))
        fig.update_layout(**_chart_layout(height=CHART_LG,
                          margin=dict(l=65, r=60, t=30, b=20),
                          title=dict(text=f"ATM Z-SCORE MATRIX ({lookback}D)",
                                     font=dict(size=10, color="#808080")),
                          yaxis=dict(autorange="reversed", tickfont=dict(size=8, color="#808080"),
                                     showgrid=False),
                          xaxis=dict(tickfont=dict(size=9, color="#808080"),
                                     showgrid=False)))
        return fig
    except Exception:
        return _empty_fig("Z-SCORE MATRIX")


def _build_ivrv_panel(pair, tenor, lookback):
    """IV-RV spread with filled area."""
    try:
        from core.fx_analytics import iv_rv_spread
        df = iv_rv_spread(pair, tenor, lookback=lookback)
        if df is None or (hasattr(df, '__len__') and len(df) < 10):
            return _empty_fig(f"{pair} IV-RV")

        spread = df["spread"].values if "spread" in df.columns else (df["iv"].values - df["rv"].values)
        x = list(range(len(spread)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=spread, mode="lines",
                                 line=dict(color="#808080", width=1), name="IV-RV",
                                 hovertemplate="Day %{x}<br>IV-RV: %{y:+.2f}v<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=[max(0, s) for s in spread], mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[0]*len(x), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(255,51,51,0.15)", showlegend=False))
        fig.add_hline(y=0, line=dict(color="#808080", width=0.5, dash="dash"))

        fig.update_layout(**_chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"{pair} IV-RV SPREAD ({tenor})",
                                     font=dict(size=10, color="#808080")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} IV-RV")


def _build_skew_scatter(pair_a, pair_b, tenor, lookback):
    """25D RR scatter with regression."""
    try:
        from core.bloomberg_fx import get_fx_historical_vol
        from scipy.stats import linregress

        rr_a = get_fx_historical_vol(pair_a, tenor, "25D_RR", lookback)
        rr_b = get_fx_historical_vol(pair_b, tenor, "25D_RR", lookback)
        if rr_a is None or rr_b is None:
            return _empty_fig("NO SKEW DATA")

        a = np.array(rr_a, dtype=float)
        b = np.array(rr_b, dtype=float)
        min_len = min(len(a), len(b))
        a, b = a[-min_len:], b[-min_len:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=a[:-1], y=b[:-1], mode="markers",
                                 marker=dict(color="#808080", size=3, opacity=0.5),
                                 name="History",
                                 hovertemplate=pair_a + " RR: %{x:.2f}<br>" + pair_b + " RR: %{y:.2f}<extra>History</extra>"))
        fig.add_trace(go.Scatter(x=[a[-1]], y=[b[-1]], mode="markers",
                                 marker=dict(color="#ff3333", size=10, symbol="diamond"),
                                 name="Current",
                                 hovertemplate=pair_a + " RR: %{x:.2f}<br>" + pair_b + " RR: %{y:.2f}<extra>Current</extra>"))
        if min_len > 5:
            slope, intercept, r, _, _ = linregress(a, b)
            x_line = np.linspace(a.min(), a.max(), 50)
            fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines",
                                     line=dict(color="#ffffff", width=1, dash="dash"),
                                     name=f"R²={r**2:.2f}" if np.isfinite(r) else "R²=—"))

        fig.update_layout(**_chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=30),
                          title=dict(text=f"SKEW SCATTER: {pair_a} vs {pair_b} ({tenor})",
                                     font=dict(size=10, color="#808080")),
                          xaxis=dict(title=dict(text=f"{pair_a} 25D RR", font=dict(size=9))),
                          yaxis=dict(title=dict(text=f"{pair_b} 25D RR", font=dict(size=9))),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return fig
    except Exception:
        return _empty_fig("SKEW SCATTER")


def _build_signal_table(lookback):
    """RV composite signal table."""
    try:
        from core.fx_analytics import vol_zscore, iv_rv_spread, rv_signal_composite
        rows = []
        for pair in ALL_PAIRS:
            try:
                try:
                    atm_info = vol_zscore(pair, "3M", "ATM", lookback)
                    z_atm = _sf(atm_info.get("zscore", 0) if isinstance(atm_info, dict) else 0)
                except Exception:
                    z_atm = 0
                try:
                    rr_info = vol_zscore(pair, "3M", "25D_RR", lookback)
                    z_rr = _sf(rr_info.get("zscore", 0) if isinstance(rr_info, dict) else 0)
                except Exception:
                    z_rr = 0
                try:
                    comp_info = rv_signal_composite(pair, lookback)
                    composite = _sf(comp_info.get("composite_score", 0)
                                    if isinstance(comp_info, dict) else 0)
                except Exception:
                    composite = z_atm * 10 + z_rr * 5

                direction = ("BUY VOL" if composite < -20 else "SELL VOL" if composite > 20
                             else "BUY SKEW" if z_rr < -1.5 else "SELL SKEW" if z_rr > 1.5
                             else "NEUTRAL")
                confidence = "HIGH" if abs(composite) > 30 else "MEDIUM" if abs(composite) > 15 else "LOW"

                rows.append({
                    "pair": pair, "atm_z": round(z_atm, 2), "rr_z": round(z_rr, 2),
                    "composite": round(composite, 1), "direction": direction,
                    "confidence": confidence,
                })
            except Exception:
                rows.append({"pair": pair, "atm_z": 0, "rr_z": 0, "composite": 0,
                             "direction": "NEUTRAL", "confidence": "LOW"})
        return rows
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# CORRELATION TAB — Matrix + regime + rolling (from correlation_monitor.py)
# ═══════════════════════════════════════════════════════════════════════════

def _safe_corr_matrix(window=60):
    """Correlation matrix with fallback. Always returns numpy ndarray."""
    try:
        from core.fx_analytics import spot_correlation_matrix
        matrix = spot_correlation_matrix(MONITOR_PAIRS, window)
        if matrix is not None:
            if hasattr(matrix, 'values'):
                return matrix.values  # DataFrame → ndarray
            return np.array(matrix)
    except Exception:
        pass

    return None


def _build_corr_heatmap(window):
    """15×15 correlation heatmap."""
    corr = _safe_corr_matrix(window)
    if corr is None:
        return no_data_fig(msg="NO CORRELATION DATA")
    n = len(MONITOR_PAIRS)
    # Ensure corr matrix matches expected dimensions
    corr = np.array(corr, dtype=float)
    rows, cols = corr.shape if corr.ndim == 2 else (0, 0)
    if rows < n or cols < n:
        # Pad with NaN if matrix is smaller than expected
        padded = np.full((n, n), np.nan)
        padded[:min(rows, n), :min(cols, n)] = corr[:min(rows, n), :min(cols, n)]
        corr = padded
    else:
        corr = corr[:n, :n]
    labels = [f"{p[:3]}/{p[3:]}" for p in MONITOR_PAIRS]
    text = [[f"{corr[i][j]:.2f}" if np.isfinite(corr[i][j]) else ""
             for j in range(n)]
            for i in range(n)]

    # Convert NaN to None for Plotly compatibility
    z_data = [[None if not np.isfinite(corr[i][j]) else corr[i][j]
                for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=z_data, x=labels, y=labels, text=text,
        texttemplate="%{text}", textfont=dict(size=8, color="#c0c0c0"),
        colorscale=[[0, "#ff3333"], [0.35, "#1a0e0e"], [0.50, "#0e0e0e"],
                    [0.65, "#0a1628"], [1.0, "#00cc66"]],
        zmin=-1, zmax=1,
        xgap=2, ygap=2,
        hovertemplate="<b>%{x} vs %{y}</b><br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="ρ", font=dict(size=9, color="#808080")),
            tickfont=dict(size=8, color="#808080"),
            len=0.6, thickness=10, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
    ))
    fig.update_layout(**_chart_layout(height=CHART_LG,
                      margin=dict(l=60, r=50, t=30, b=50),
                      title=dict(text=f"SPOT CORRELATION ({window}D)",
                                 font=dict(size=10, color="#808080")),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
                      xaxis=dict(tickfont=dict(size=8), tickangle=45)))
    return fig


def _build_regime_badges():
    """Vol regime badges for 15 pairs."""
    try:
        from core.fx_analytics import vol_regime_detect
    except Exception:
        vol_regime_detect = None

    badges = []
    for pair in MONITOR_PAIRS:
        try:
            if vol_regime_detect:
                info = vol_regime_detect(pair)
                regime = info.get("regime", "NORMAL") if isinstance(info, dict) else "NORMAL"
                color = info.get("color", "#d4d4d4") if isinstance(info, dict) else "#d4d4d4"
            else:
                regime = "NORMAL"
                color = "#d4d4d4"
        except Exception:
            regime = "NORMAL"
            color = "#d4d4d4"

        regime_colors = {"LOW": "#00cc66", "NORMAL": "#d4d4d4", "ELEVATED": "#ff8800",
                         "HIGH": "#ff3333", "CRISIS": "#ff3333"}
        color = regime_colors.get(regime, color)

        badges.append(html.Div([
            html.Div(f"{pair[:3]}/{pair[3:]}", style={
                "fontSize": "9px", "fontWeight": "700", "color": "#d4d4d4", "fontFamily": _MONO,
            }),
            html.Div(regime, style={
                "fontSize": "8px", "fontWeight": "600", "color": color, "fontFamily": _MONO,
            }),
        ], style={
            "border": f"1px solid {color}", "borderRadius": "0px",
            "padding": "4px 6px", "textAlign": "center", "minWidth": "70px",
            "backgroundColor": "#000000",
        }))
    return badges


def _build_breakdown_table(window_short=20, window_long=120):
    """Pairs with divergent 20d vs 120d correlation."""
    corr_s = _safe_corr_matrix(window_short)
    corr_l = _safe_corr_matrix(window_long)
    if corr_s is None or corr_l is None:
        return html.Div("NO CORRELATION DATA",
                        style={"color": "#808080", "fontSize": "10px", "padding": "8px"})
    n = len(MONITOR_PAIRS)
    # Ensure matrices match expected dimensions
    corr_s = np.array(corr_s, dtype=float)
    corr_l = np.array(corr_l, dtype=float)
    for arr_name, arr in [("corr_s", corr_s), ("corr_l", corr_l)]:
        if arr.ndim != 2 or arr.shape[0] < n or arr.shape[1] < n:
            return html.Div("CORRELATION DATA DIMENSION MISMATCH",
                            style={"color": "#808080", "fontSize": "10px", "padding": "8px"})
    corr_s, corr_l = corr_s[:n, :n], corr_l[:n, :n]
    divergences = []
    for i in range(n):
        for j in range(i + 1, n):
            s_val, l_val = corr_s[i][j], corr_l[i][j]
            if not (np.isfinite(s_val) and np.isfinite(l_val)):
                continue
            gap = s_val - l_val
            if abs(gap) > 0.3:
                signal = "DIVERGING" if abs(s_val) < abs(l_val) else "CONVERGING"
                divergences.append({
                    "pair_a": MONITOR_PAIRS[i], "pair_b": MONITOR_PAIRS[j],
                    "corr_20d": round(float(s_val), 2), "corr_120d": round(float(l_val), 2),
                    "gap": round(float(gap), 2), "signal": signal,
                })
    divergences.sort(key=lambda d: abs(d["gap"]), reverse=True)

    if not divergences:
        return html.Div("No significant divergences (|gap| > 0.3)",
                        style={"color": "#808080", "fontSize": "10px", "padding": "8px"})

    header = html.Tr([html.Th(h, style={**TABLE_HEADER_STYLE, "fontSize": "8px"})
                       for h in ["PAIR A", "PAIR B", "20D", "120D", "GAP", "SIGNAL"]])
    body = []
    for d in divergences[:min(len(divergences), 10)]:
        gap_color = "#ff3333" if abs(d["gap"]) > 0.4 else "#ff8800"
        sig_color = "#ff3333" if d["signal"] == "DIVERGING" else "#00cc66"
        body.append(html.Tr([
            html.Td(d["pair_a"], style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
            html.Td(d["pair_b"], style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
            html.Td(f"{d['corr_20d']:+.2f}", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
            html.Td(f"{d['corr_120d']:+.2f}", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
            html.Td(f"{d['gap']:+.2f}", style={**TABLE_CELL_STYLE, "color": gap_color, "fontWeight": "600"}),
            html.Td(d["signal"], style={**TABLE_CELL_STYLE, "color": sig_color, "fontWeight": "600"}),
        ]))
    return html.Table([html.Thead(header), html.Tbody(body)],
                      style={"width": "100%", "borderCollapse": "collapse", "fontFamily": _MONO, "fontSize": "10px"})


def _build_rolling_chart(pair_a, pair_b):
    """Rolling correlation chart for 3 windows."""
    try:
        from core.bloomberg_fx import get_fx_historical_spot
        spot_a = get_fx_historical_spot(pair_a, 252)
        spot_b = get_fx_historical_spot(pair_b, 252)

        def _to_array(s):
            if s is None:
                return None
            if hasattr(s, 'columns') and 'close' in s.columns:
                return s['close'].values.astype(float)
            if hasattr(s, 'values'):
                return s.values.flatten().astype(float)
            return np.array(s, dtype=float)

        a = _to_array(spot_a)
        b = _to_array(spot_b)
        if a is None or b is None:
            return no_data_fig(msg="NO SPOT HISTORY")

        min_len = min(len(a), len(b))
        a, b = a[-min_len:], b[-min_len:]
        ret_a = np.diff(np.log(np.maximum(a, 1e-10)))
        ret_b = np.diff(np.log(np.maximum(b, 1e-10)))

        fig = go.Figure()
        windows = [20, 60, 120]
        colors = ["#ff8800", "#ffffff", "#00cc66"]
        for w, c in zip(windows, colors):
            if len(ret_a) < w:
                continue
            rolling = []
            for i in range(w, len(ret_a)):
                ra = ret_a[i - w:i]
                rb = ret_b[i - w:i]
                if np.std(ra) > 1e-10 and np.std(rb) > 1e-10:
                    rolling.append(np.corrcoef(ra, rb)[0, 1])
                else:
                    rolling.append(0)
            fig.add_trace(go.Scatter(x=list(range(len(rolling))), y=rolling,
                                     mode="lines", line=dict(color=c, width=1.5),
                                     name=f"{w}d",
                                     hovertemplate="Day %{x}<br>\u03c1: %{y:.3f}<extra>" + f"{w}d" + "</extra>"))

        fig.add_hline(y=0, line=dict(color="#808080", width=0.5, dash="dot"))
        fig.add_hline(y=0.7, line=dict(color="#00cc66", width=0.5, dash="dash"))
        fig.add_hline(y=-0.7, line=dict(color="#00cc66", width=0.5, dash="dash"))

        fig.update_layout(**_chart_layout(height=CHART_SM,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"ROLLING CORRELATION: {pair_a} vs {pair_b}",
                                     font=dict(size=10, color="#808080")),
                          yaxis=dict(range=[-1.05, 1.05]),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8), orientation="h")))
        return fig
    except Exception:
        return _empty_fig(f"ROLLING CORR: {pair_a} vs {pair_b}")


# ═══════════════════════════════════════════════════════════════════════════
# MACRO TAB (from macro_regime.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_rate_table():
    """Rate differentials table."""
    from core.bloomberg_fx import get_fx_rates

    header = html.Tr([html.Th(h, style={**TABLE_HEADER_STYLE, "fontSize": "8px"})
                       for h in ["PAIR", "DOM", "FOR", "DIFF", "CARRY"]])
    body = []
    for pair in ALL_PAIRS[:20]:
        try:
            r = get_fx_rates(pair) or {}
        except Exception:
            r = {}
        dom = _sf(r.get("r_dom", 0)) * 100  # convert decimal to percent
        fgn = _sf(r.get("r_for", 0)) * 100
        diff = dom - fgn
        carry = "RECEIVE" if diff > 0.5 else "PAY" if diff < -0.5 else "FLAT"
        carry_color = "#00cc66" if carry == "RECEIVE" else "#ff3333" if carry == "PAY" else "#808080"
        body.append(html.Tr([
            html.Td(pair, style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
            html.Td(f"{dom:.2f}%", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
            html.Td(f"{fgn:.2f}%", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
            html.Td(f"{diff:+.2f}", style={**TABLE_CELL_STYLE, "textAlign": "right",
                     "color": "#00cc66" if diff > 0 else "#ff3333"}),
            html.Td(carry, style={**TABLE_CELL_STYLE, "color": carry_color, "fontWeight": "600"}),
        ]))
    return html.Table([html.Thead(header), html.Tbody(body)],
                      style={"width": "100%", "borderCollapse": "collapse",
                             "fontFamily": _MONO, "fontSize": "10px"})


def _build_dxy_chart(lookback):
    """DXY proxy chart — trade-weighted USD index from spot data."""
    try:
        from core.bloomberg_fx import get_fx_historical_spot
        # Build a trade-weighted USD index from spot history
        weights = {"EURUSD": -0.576, "USDJPY": 0.136, "GBPUSD": -0.119,
                   "USDCHF": 0.036, "USDCAD": 0.091, "NZDUSD": -0.021, "AUDUSD": -0.021}

        all_series = {}
        min_len = lookback
        for pair, w in weights.items():
            hist = get_fx_historical_spot(pair, lookback + 10)
            if hist is not None and len(hist) > 20:
                if hasattr(hist, 'columns') and 'close' in hist.columns:
                    arr = hist['close'].values.astype(float)
                elif hasattr(hist, 'values'):
                    arr = hist.values.flatten().astype(float)
                else:
                    arr = np.array(hist, dtype=float)
                all_series[pair] = arr
                min_len = min(min_len, len(arr))

        if len(all_series) < 3:
            return no_data_fig(msg="INSUFFICIENT SPOT DATA")
        else:
            # Compute weighted log returns
            n = min_len
            index = np.ones(n) * 100
            for pair, w in weights.items():
                if pair not in all_series:
                    continue
                arr = all_series[pair][-n:]
                returns = np.diff(np.log(arr)) * w
                # Compound into index
                for i in range(len(returns)):
                    index[i + 1] = index[i] * np.exp(returns[i])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(index))), y=index, mode="lines",
                                 line=dict(color="#ff8800", width=2.5), name="DXY Proxy",
                                 hovertemplate="Day %{x}<br>DXY: %{y:.2f}<extra></extra>"))
        fig.update_layout(**_chart_layout(height=CHART_SM,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"DXY PROXY ({lookback}D)",
                                     font=dict(size=10, color="#808080")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return fig
    except Exception:
        return _empty_fig("DXY PROXY")


def _build_risk_sentiment():
    """Risk sentiment composite."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface, get_fx_spots
        spots = get_fx_spots() or {}
        factors = []

        # G10 avg vol
        g10_vols = []
        for pair in G10_PAIRS[:8]:
            surf = get_fx_vol_surface(pair) or {}
            atm = _extract_atm(surf, "1M")
            if atm > 0:
                g10_vols.append(atm)
        g10_avg = np.mean(g10_vols) if g10_vols else 8.0
        vol_score = min(100, max(0, (g10_avg - 5) * 8))
        factors.append(("G10 AVG VOL", f"{g10_avg:.1f}v", vol_score))

        # Haven bid
        jpy_chg = _sf(spots.get("USDJPY", {}).get("change_pct", 0))
        chf_chg = _sf(spots.get("USDCHF", {}).get("change_pct", 0))
        haven_score = min(100, max(0, 50 + (jpy_chg + chf_chg) * 10))
        factors.append(("HAVEN BID", f"JPY{jpy_chg:+.1f}% CHF{chf_chg:+.1f}%", haven_score))

        # Composite
        composite = np.mean([f[2] for f in factors])
        label = "RISK-OFF" if composite > 65 else "RISK-ON" if composite < 35 else "NEUTRAL"
        color = "#ff3333" if composite > 65 else "#00cc66" if composite < 35 else "#ff8800"

        items = [html.Div([
            html.Div(label, style={"fontSize": "16px", "fontWeight": "700",
                                   "color": color, "fontFamily": _MONO}),
            html.Div(f"COMPOSITE: {composite:.0f}" if np.isfinite(composite) else "COMPOSITE: —", style={
                "fontSize": "9px", "color": "#808080", "fontFamily": _MONO}),
        ], style={**STAT_BOX_STYLE, "borderTop": f"2px solid {color}"})]

        for name, val, score in factors:
            items.append(html.Div([
                html.Div(val, style={"fontSize": "11px", "color": "#d4d4d4", "fontFamily": _MONO}),
                html.Div(name, style={"fontSize": "8px", "color": "#808080", "fontFamily": _MONO}),
            ], style={**STAT_BOX_STYLE}))

        return items
    except Exception:
        return [html.Div("Error", style={"color": "#808080"})]


def _build_cb_chart():
    """Central bank rates bar chart — tries to update from Bloomberg rates."""
    # Try to refresh rates from Bloomberg data
    try:
        from core.bloomberg_fx import get_fx_rates
        # Map each central bank to a currency pair and which leg to use
        bank_map = {
            "FED":  {"pair": "EURUSD", "key": "r_dom"},   # USD = domestic in EURUSD
            "ECB":  {"pair": "EURUSD", "key": "r_for"},   # EUR = foreign in EURUSD
            "BOE":  {"pair": "GBPUSD", "key": "r_for"},   # GBP = foreign in GBPUSD
            "BOJ":  {"pair": "USDJPY", "key": "r_dom"},   # JPY = domestic in USDJPY
            "SNB":  {"pair": "USDCHF", "key": "r_dom"},   # CHF = domestic in USDCHF
            "RBA":  {"pair": "AUDUSD", "key": "r_for"},   # AUD = foreign in AUDUSD
            "RBNZ": {"pair": "NZDUSD", "key": "r_for"},   # NZD = foreign in NZDUSD
            "BOC":  {"pair": "USDCAD", "key": "r_dom"},   # CAD = domestic in USDCAD
        }
        for bank, info in bank_map.items():
            try:
                rates_data = get_fx_rates(info["pair"])
                if rates_data and isinstance(rates_data, dict):
                    actual = _sf(rates_data.get(info["key"], 0))
                    if actual > 0:
                        # Convert from decimal to percentage if needed
                        rate_pct = actual * 100 if actual < 1 else actual
                        _CB_BANKS[bank]["rate"] = round(rate_pct, 2)
            except Exception:
                pass  # Keep hardcoded default for this bank
    except Exception:
        pass  # Use hardcoded defaults entirely

    banks = list(_CB_BANKS.keys())
    rates = [_CB_BANKS[b]["rate"] for b in banks]
    dirs = [_CB_BANKS[b]["direction"] for b in banks]
    colors = ["#ff3333" if d == "HIKING" else "#00cc66" if d == "CUTTING" else "#ff8800" for d in dirs]

    fig = go.Figure(go.Bar(y=banks, x=rates, orientation="h",
                            marker_color=colors,
                            text=[f"{r:.2f}% ({d})" for r, d in zip(rates, dirs)],
                            textposition="outside", textfont=dict(size=8)))
    fig.update_layout(**_chart_layout(height=CHART_SM,
                      margin=dict(l=40, r=80, t=30, b=10), showlegend=False,
                      title=dict(text="CENTRAL BANK POLICY RATES",
                                 font=dict(size=10, color="#808080")),
                      xaxis=dict(tickfont=dict(size=8)),
                      yaxis=dict(tickfont=dict(size=9))))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CARRY TAB (from vol_carry.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_carry_data():
    """Carry table data."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
    except Exception:
        pass

    rows = []
    for pair in ALL_PAIRS:
        try:
            surf = get_fx_vol_surface(pair) or {}
            atm_1m = _extract_atm(surf, "1M")
            atm_3m = _extract_atm(surf, "3M")
            atm_1y = _extract_atm(surf, "1Y")
            term_spread = atm_1m - atm_1y if atm_1m > 0 and atm_1y > 0 else 0

            # Proper variance-based forward vol: σ_fwd = sqrt((σ₂²T₂ - σ₁²T₁) / (T₂ - T₁))
            t1 = 1.0 / 12.0  # 1M in years
            t3 = 3.0 / 12.0  # 3M in years
            v1 = atm_1m / 100.0 if atm_1m > 0 else 0
            v3 = atm_3m / 100.0 if atm_3m > 0 else 0
            fwd_var = (v3 ** 2 * t3 - v1 ** 2 * t1) / max(t3 - t1, 1e-6)
            fwd_3x3 = np.sqrt(max(fwd_var, 0)) * 100 if fwd_var > 0 else atm_3m

            # Carry (computed from term structure)
            carry_day = (atm_3m - fwd_3x3) / 90 if atm_3m > 0 else 0
            carry_vol = carry_day / max(atm_3m, 0.01) * 100 if atm_3m > 0 else 0

            signal = ("SELL VOL" if carry_vol > 2 else "BUY VOL" if carry_vol < -1 else "—")

            rows.append({
                "pair": pair, "atm_1m": round(atm_1m, 2), "atm_3m": round(atm_3m, 2),
                "atm_1y": round(atm_1y, 2), "term_spread": round(term_spread, 2),
                "fwd_3x3": round(fwd_3x3, 2), "carry_day": round(carry_day, 3),
                "carry_vol": round(carry_vol, 2), "signal": signal,
            })
        except Exception:
            rows.append({"pair": pair, "atm_1m": 0, "atm_3m": 0, "atm_1y": 0,
                         "term_spread": 0, "fwd_3x3": 0, "carry_day": 0, "carry_vol": 0, "signal": "—"})
    rows.sort(key=lambda r: r.get("carry_vol", 0), reverse=True)
    return rows


def _build_term_chart(pair, comp_pair=None):
    """Term structure overlay."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        surf = get_fx_vol_surface(pair) or {}
        vols = [_extract_atm(surf, t) for t in TERM_TENORS]
        labels = TERM_TENORS

        if all(v == 0 or v is None for v in vols):
            return no_data_fig(msg="NO VOL DATA")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=labels, y=vols, mode="lines+markers",
                                 line=dict(color="#ff8800", width=2), marker=dict(size=5),
                                 name=pair,
                                 hovertemplate="%{x}: %{y:.2f}v<extra>" + pair + "</extra>"))
        if comp_pair and comp_pair != pair:
            surf2 = get_fx_vol_surface(comp_pair) or {}
            vols2 = [_extract_atm(surf2, t) for t in TERM_TENORS]
            fig.add_trace(go.Scatter(x=labels, y=vols2, mode="lines+markers",
                                     line=dict(color="#ffffff", width=1.5, dash="dash"),
                                     marker=dict(size=4), name=comp_pair,
                                     hovertemplate="%{x}: %{y:.2f}v<extra>" + comp_pair + "</extra>"))

        fig.update_layout(**_chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"TERM STRUCTURE: {pair}",
                                     font=dict(size=10, color="#808080")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} TERM")


def _build_fwd_vol(pair):
    """Forward vol bar chart."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        surf = get_fx_vol_surface(pair) or {}
        tenors = ["1M", "2M", "3M", "6M", "1Y"]
        spot_vols = [_extract_atm(surf, t) for t in tenors]

        if all(v == 0 or v is None for v in spot_vols):
            return no_data_fig(msg="NO VOL DATA")

        # Calculate forward vols (variance-based)
        fwd_vols = [spot_vols[0]]
        for i in range(1, len(spot_vols)):
            if spot_vols[i] > 0 and spot_vols[i - 1] > 0:
                t_prev = tenor_to_years(tenors[i - 1])
                t_curr = tenor_to_years(tenors[i])
                var_prev = (spot_vols[i - 1] / 100.0) ** 2 * t_prev
                var_curr = (spot_vols[i] / 100.0) ** 2 * t_curr
                fwd_var = (var_curr - var_prev) / max(t_curr - t_prev, 1e-6)
                fwd = np.sqrt(max(fwd_var, 0)) * 100
                fwd_vols.append(fwd)
            else:
                fwd_vols.append(0)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=tenors, y=fwd_vols, marker_color="#ff8800",
                              text=[f"{v:.1f}" for v in fwd_vols],
                              textposition="outside", textfont=dict(size=8),
                              name="Forward Vol",
                              hovertemplate="%{x}: %{y:.2f}v<extra>Forward</extra>"))
        fig.add_trace(go.Scatter(x=tenors, y=spot_vols, mode="lines+markers",
                                  line=dict(color="#ffffff", width=1.5, dash="dot"),
                                  marker=dict(size=4), name="Spot Vol",
                                  hovertemplate="%{x}: %{y:.2f}v<extra>Spot</extra>"))
        fig.update_layout(**_chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"FORWARD VOL: {pair}",
                                     font=dict(size=10, color="#808080")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} FWD VOL")


def _build_calendar_spread(pair):
    """Calendar spread: difference between near (1M) and far (3M) tenor ATM vols over time."""
    try:
        from core.bloomberg_fx import get_fx_historical_vol
        near = get_fx_historical_vol(pair, "1M", "atm", 252)
        far = get_fx_historical_vol(pair, "3M", "atm", 252)

        if near is None or far is None or len(near) < 10 or len(far) < 10:
            raise ValueError("Insufficient vol history")

        # Align lengths
        n = min(len(near), len(far))
        near_vals = near.values[-n:].astype(float) if hasattr(near, 'values') else np.array(near[-n:], dtype=float)
        far_vals = far.values[-n:].astype(float) if hasattr(far, 'values') else np.array(far[-n:], dtype=float)
        spread = near_vals - far_vals

        fig = go.Figure()
        x = list(range(n))
        # Shade positive (backwardation) vs negative (contango)
        pos_y = [s if s >= 0 else 0 for s in spread]
        neg_y = [s if s < 0 else 0 for s in spread]
        fig.add_trace(go.Bar(x=x, y=pos_y, marker_color="rgba(255,51,51,0.5)",
                              name="Backwardation", showlegend=True,
                              hovertemplate="Day %{x}<br>Backwardation: %{y:.2f}v<extra></extra>"))
        fig.add_trace(go.Bar(x=x, y=neg_y, marker_color="rgba(21,101,192,0.5)",
                              name="Contango", showlegend=True,
                              hovertemplate="Day %{x}<br>Contango: %{y:.2f}v<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=spread, mode="lines",
                                  line=dict(color="#ff8800", width=2.5), name="1M-3M Spread",
                                  hovertemplate="Day %{x}<br>Spread: %{y:+.2f}v<extra></extra>"))
        fig.add_hline(y=0, line_dash="dot", line_color="#444444", line_width=0.8)
        fig.update_layout(**_chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=20),
                          barmode="overlay",
                          title=dict(text=f"CALENDAR SPREAD (1M-3M): {pair}",
                                     font=dict(size=10, color="#808080")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return fig
    except Exception:
        return _empty_fig(f"{pair} CALENDAR SPREAD")


# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

def layout():
    return html.Div([
        # CSV download
        dcc.Download(id=f"{_P}-csv-download"),
        # Stores
        dcc.Store(id=f"{_P}-corr-pair-a", data="EURUSD"),
        dcc.Store(id=f"{_P}-corr-pair-b", data="USDJPY"),
        dcc.Store(id=f"{_P}-carry-pair", data="EURUSD"),
        dcc.Interval(id=f"{_P}-interval", interval=300_000, n_intervals=0),

        # ── Title + Controls ──
        html.Div([
            html.Span("RELATIVE VALUE", style={
                "color": "#ffffff", "fontSize": "13px", "fontWeight": "700",
                "letterSpacing": "2px", "fontFamily": _MONO,
            }),
            html.Div([
                dcc.Dropdown(id=f"{_P}-pair-a", options=[{"label": p, "value": p} for p in ALL_PAIRS],
                             value="EURUSD", clearable=False, style={**DROPDOWN_STYLE, "width": "110px"}),
                dcc.Dropdown(id=f"{_P}-pair-b", options=[{"label": p, "value": p} for p in ALL_PAIRS],
                             value="USDJPY", clearable=False, style={**DROPDOWN_STYLE, "width": "110px"}),
                dcc.Dropdown(id=f"{_P}-tenor", options=[{"label": t, "value": t} for t in TERM_TENORS],
                             value="3M", clearable=False, style={**DROPDOWN_STYLE, "width": "70px"}),
                dcc.Dropdown(id=f"{_P}-lookback", options=LOOKBACK_OPTIONS, value=252,
                             clearable=False, style={**DROPDOWN_STYLE, "width": "80px"}),
            ], style={"display": "flex", "gap": GAP}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "padding": f"{GAP} 0",
                  "borderBottom": "1px solid #222240"}),

        # ── Sub-Tabs ──
        dcc.Tabs(id=f"{_P}-tabs", value="cross-pair", children=[
            dcc.Tab(label="CROSS-PAIR", value="cross-pair", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="CORRELATION", value="correlation", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="MACRO", value="macro", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="CARRY", value="carry", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ], style={"marginTop": GAP, "marginBottom": "0"}),

        # ══════════ CROSS-PAIR TAB ══════════
        html.Div(id=f"{_P}-cross-container", children=[
            html.Div([
                html.Div([
                    html.Button("CSV", id=f"{_P}-csv-vol-spread", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-vol-spread", config={"displayModeBar": False, "responsive": True}),
                ], style={"flex": "1", "minWidth": "350px"}),
                html.Div([
                    html.Button("CSV", id=f"{_P}-csv-skew-scatter", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-skew-scatter", config={"displayModeBar": False, "responsive": True}),
                ], style={"flex": "1", "minWidth": "350px"}),
            ], style={"display": "flex", "gap": GAP}),
            html.Div([
                html.Div([
                    html.Button("CSV", id=f"{_P}-csv-zscore", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-zscore-matrix", config={"displayModeBar": False, "responsive": True}),
                ], style={"flex": "1", "minWidth": "350px"}),
                html.Div([
                    html.Button("CSV", id=f"{_P}-csv-ivrv", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-ivrv-panel", config={"displayModeBar": False, "responsive": True}),
                ], style={"flex": "1", "minWidth": "350px"}),
            ], style={"display": "flex", "gap": GAP, "marginTop": GAP}),
            # Signal table
            html.Div(id=f"{_P}-signal-table", style={"marginTop": SECTION_GAP}),
        ]),

        # ══════════ CORRELATION TAB ══════════
        html.Div(id=f"{_P}-corr-container", style={"display": "none"}, children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("WINDOW", style=LABEL_STYLE),
                        dcc.Dropdown(id=f"{_P}-corr-window",
                                     options=[{"label": f"{w}D", "value": w} for w in [20, 60, 120]],
                                     value=60, clearable=False,
                                     style={**DROPDOWN_STYLE, "width": "80px"}),
                    ], style={"display": "flex", "alignItems": "center", "gap": GAP, "marginBottom": GAP}),
                    html.Button("CSV", id=f"{_P}-csv-corr", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-corr-heatmap", config={"displayModeBar": False, "responsive": True},
                              style={"height": f"{CHART_LG}px"}),
                ], style={"flex": "3"}),
                html.Div([
                    html.Div("VOL REGIME", style={
                        "color": "#808080", "fontSize": "9px", "fontWeight": "700",
                        "letterSpacing": "1.5px", "marginBottom": GAP, "fontFamily": _MONO,
                    }),
                    html.Div(id=f"{_P}-regime-badges", style={
                        "display": "flex", "flexWrap": "wrap", "gap": GAP,
                    }),
                ], style={"flex": "2", "padding": GAP}),
            ], style={"display": "flex", "gap": GAP}),
            html.Div(id=f"{_P}-corr-breakdown", style={"marginTop": SECTION_GAP}),
            html.Button("CSV", id=f"{_P}-csv-rolling", n_clicks=0, style=CSV_BTN_STYLE),
            dcc.Graph(id=f"{_P}-rolling-chart", config={"displayModeBar": False, "responsive": True},
                      style={"height": f"{CHART_SM}px", "marginTop": GAP}),
        ]),

        # ══════════ MACRO TAB ══════════
        html.Div(id=f"{_P}-macro-container", style={"display": "none"}, children=[
            html.Div([
                html.Div([
                    html.Div("RATE DIFFERENTIALS", style={
                        "color": "#808080", "fontSize": "9px", "fontWeight": "700",
                        "letterSpacing": "1.5px", "marginBottom": GAP, "fontFamily": _MONO,
                    }),
                    html.Div(id=f"{_P}-rate-table", style={"overflowY": "auto", "maxHeight": "350px"}),
                ], style={"flex": "1", "border": "1px solid #222240", "padding": GAP}),
                html.Div([
                    html.Div(id=f"{_P}-risk-sentiment", style={"marginBottom": GAP}),
                    html.Button("CSV", id=f"{_P}-csv-cb", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-cb-chart", config={"displayModeBar": False, "responsive": True},
                              style={"height": f"{CHART_SM}px"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": GAP}),
            html.Div([
                html.Div([
                    html.Span("LOOKBACK", style=LABEL_STYLE),
                    dcc.Dropdown(id=f"{_P}-dxy-lookback", options=LOOKBACK_OPTIONS, value=252,
                                 clearable=False, style={**DROPDOWN_STYLE, "width": "80px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": GAP, "padding": f"{GAP} 0"}),
                html.Button("CSV", id=f"{_P}-csv-dxy", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id=f"{_P}-dxy-chart", config={"displayModeBar": False, "responsive": True},
                          style={"height": f"{CHART_SM}px"}),
            ], style={"marginTop": SECTION_GAP}),
        ]),

        # ══════════ CARRY TAB ══════════
        html.Div(id=f"{_P}-carry-container", style={"display": "none"}, children=[
            html.Div(id=f"{_P}-carry-table-wrapper", style={
                "overflowY": "auto", "maxHeight": "350px", "border": "1px solid #222240",
            }),
            html.Div([
                html.Div([
                    html.Span("PAIR", style=LABEL_STYLE),
                    dcc.Dropdown(id=f"{_P}-carry-pair-dd",
                                 options=[{"label": p, "value": p} for p in ALL_PAIRS],
                                 value="EURUSD", clearable=False,
                                 style={**DROPDOWN_STYLE, "width": "110px"}),
                    html.Span("COMPARE", style={**LABEL_STYLE, "marginLeft": "12px"}),
                    dcc.Dropdown(id=f"{_P}-carry-comp-dd",
                                 options=[{"label": p, "value": p} for p in ALL_PAIRS],
                                 value="USDJPY", clearable=False,
                                 style={**DROPDOWN_STYLE, "width": "110px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": GAP, "padding": f"{GAP} 0"}),
                html.Div([
                    html.Div([
                        html.Button("CSV", id=f"{_P}-csv-carry-term", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id=f"{_P}-carry-term", config={"displayModeBar": False, "responsive": True}),
                    ], style={"flex": "1", "minWidth": "350px"}),
                    html.Div([
                        html.Button("CSV", id=f"{_P}-csv-carry-fwd", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id=f"{_P}-carry-fwd", config={"displayModeBar": False, "responsive": True}),
                    ], style={"flex": "1", "minWidth": "350px"}),
                ], style={"display": "flex", "gap": GAP}),
                html.Button("CSV", id=f"{_P}-csv-carry-cal", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id=f"{_P}-carry-calendar", config={"displayModeBar": False, "responsive": True},
                          style={"height": f"{CHART_MD}px", "marginTop": GAP}),
            ], style={"marginTop": SECTION_GAP}),
        ]),

    ], style={"fontFamily": _MONO})


# ═══════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app):

    # ── Tab visibility ──
    @app.callback(
        [
            Output(f"{_P}-cross-container", "style"),
            Output(f"{_P}-corr-container", "style"),
            Output(f"{_P}-macro-container", "style"),
            Output(f"{_P}-carry-container", "style"),
        ],
        Input(f"{_P}-tabs", "value"),
    )
    def toggle_tabs(tab):
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if tab == "cross-pair" else hide,
            show if tab == "correlation" else hide,
            show if tab == "macro" else hide,
            show if tab == "carry" else hide,
        )

    # ══════════ CROSS-PAIR CALLBACKS ══════════

    @app.callback(
        [
            Output(f"{_P}-vol-spread", "figure"),
            Output(f"{_P}-skew-scatter", "figure"),
            Output(f"{_P}-zscore-matrix", "figure"),
            Output(f"{_P}-ivrv-panel", "figure"),
        ],
        [
            Input(f"{_P}-pair-a", "value"),
            Input(f"{_P}-pair-b", "value"),
            Input(f"{_P}-tenor", "value"),
            Input(f"{_P}-lookback", "value"),
        ],
    )
    def update_cross_pair(pair_a, pair_b, tenor, lookback):
        try:
            pa, pb = pair_a or "EURUSD", pair_b or "USDJPY"
            t = tenor or "3M"
            lb = lookback or 252
            return (
                _build_vol_spread_ts(pa, pb, t, lb),
                _build_skew_scatter(pa, pb, t, lb),
                _build_zscore_matrix(lb),
                _build_ivrv_panel(pa, t, lb),
            )
        except Exception as exc:
            logging.exception("update_cross_pair failed")
            empty = no_data_fig("Error")
            return empty, empty, empty, empty

    @app.callback(
        Output(f"{_P}-signal-table", "children"),
        Input(f"{_P}-lookback", "value"),
    )
    def update_signal_table(lookback):
        rows = _build_signal_table(lookback or 252)
        if not rows:
            return html.Div("No signal data", style={"color": "#808080", "fontSize": "10px"})

        columns = [
            {"name": "PAIR", "id": "pair"},
            {"name": "ATM Z", "id": "atm_z", "type": "numeric"},
            {"name": "RR Z", "id": "rr_z", "type": "numeric"},
            {"name": "COMPOSITE", "id": "composite", "type": "numeric"},
            {"name": "SIGNAL", "id": "direction"},
            {"name": "CONF", "id": "confidence"},
        ]
        return dash_table.DataTable(
            columns=columns, data=rows, sort_action="native", page_size=30,
            style_table={"overflowX": "auto", "maxHeight": "400px"},
            style_header={
                "backgroundColor": "#000000", "color": "#808080",
                "fontWeight": "700", "fontSize": "9px", "textTransform": "uppercase",
                "border": "1px solid #222240", "fontFamily": _MONO,
            },
            style_cell={
                "backgroundColor": "#000000", "color": "#d4d4d4",
                "fontSize": "11px", "fontFamily": _MONO,
                "border": "1px solid #222240", "padding": "4px 6px",
                "textAlign": "right",
            },
            style_cell_conditional=[
                {"if": {"column_id": "pair"}, "textAlign": "left", "fontWeight": "700"},
                {"if": {"column_id": "direction"}, "textAlign": "center"},
                {"if": {"column_id": "confidence"}, "textAlign": "center"},
            ],
            style_data_conditional=[
                {"if": {"filter_query": '{direction} = "BUY VOL"', "column_id": "direction"},
                 "color": "#1565c0", "fontWeight": "bold"},
                {"if": {"filter_query": '{direction} = "SELL VOL"', "column_id": "direction"},
                 "color": "#ff3333", "fontWeight": "bold"},
                {"if": {"filter_query": "{composite} > 30", "column_id": "composite"},
                 "color": "#ff3333", "fontWeight": "bold"},
                {"if": {"filter_query": "{composite} < -30", "column_id": "composite"},
                 "color": "#1565c0", "fontWeight": "bold"},
                {"if": {"filter_query": '{confidence} = "HIGH"', "column_id": "confidence"},
                 "color": "#00cc66", "fontWeight": "bold"},
            ],
        )

    # ══════════ CORRELATION CALLBACKS ══════════

    @app.callback(
        [
            Output(f"{_P}-corr-heatmap", "figure"),
            Output(f"{_P}-regime-badges", "children"),
            Output(f"{_P}-corr-breakdown", "children"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-corr-window", "value"),
            Input(f"{_P}-tabs", "value"),
        ],
    )
    def update_correlation(n, window, tab):
        if tab != "correlation":
            raise PreventUpdate
        w = window or 60
        return (
            _build_corr_heatmap(w),
            _build_regime_badges(),
            _build_breakdown_table(20, 120),
        )

    @app.callback(
        [Output(f"{_P}-corr-pair-a", "data"), Output(f"{_P}-corr-pair-b", "data")],
        Input(f"{_P}-corr-heatmap", "clickData"),
        prevent_initial_call=True,
    )
    def corr_heatmap_click(click_data):
        if not click_data or not click_data.get("points"):
            raise PreventUpdate
        pt = click_data["points"][0]
        x_label = pt.get("x", "")
        y_label = pt.get("y", "")
        # Parse "EUR/USD" → "EURUSD"
        pair_a = y_label.replace("/", "")
        pair_b = x_label.replace("/", "")
        if pair_a in MONITOR_PAIRS and pair_b in MONITOR_PAIRS:
            return pair_a, pair_b
        raise PreventUpdate

    @app.callback(
        Output(f"{_P}-rolling-chart", "figure"),
        [Input(f"{_P}-corr-pair-a", "data"), Input(f"{_P}-corr-pair-b", "data")],
    )
    def update_rolling(pair_a, pair_b):
        pa = pair_a or "EURUSD"
        pb = pair_b or "USDJPY"
        if pa == pb:
            return _empty_fig("Select two different pairs")
        return _build_rolling_chart(pa, pb)

    # ══════════ MACRO CALLBACKS ══════════

    @app.callback(
        [
            Output(f"{_P}-rate-table", "children"),
            Output(f"{_P}-risk-sentiment", "children"),
            Output(f"{_P}-cb-chart", "figure"),
            Output(f"{_P}-dxy-chart", "figure"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-dxy-lookback", "value"),
            Input(f"{_P}-tabs", "value"),
        ],
    )
    def update_macro(n, lookback, tab):
        if tab != "macro":
            raise PreventUpdate
        try:
            return (
                _build_rate_table(),
                _build_risk_sentiment(),
                _build_cb_chart(),
                _build_dxy_chart(lookback or 252),
            )
        except Exception as exc:
            logging.exception("update_macro failed")
            empty = no_data_fig("Error")
            return html.Div("Macro data error", style={"color": "#ff3333"}), empty, empty, empty

    # ══════════ CARRY CALLBACKS ══════════

    @app.callback(
        Output(f"{_P}-carry-table-wrapper", "children"),
        [Input(f"{_P}-interval", "n_intervals"), Input(f"{_P}-tabs", "value")],
    )
    def update_carry_table(n, tab):
        if tab != "carry":
            raise PreventUpdate
        try:
            rows = _build_carry_data()
            if not rows:
                return html.Div("No carry data", style={"color": "#808080"})

            header = html.Tr([html.Th(h, style={**TABLE_HEADER_STYLE, "fontSize": "8px"})
                               for h in ["PAIR", "ATM 1M", "ATM 3M", "ATM 1Y", "TERM", "FWD 3×3",
                                         "CARRY/D", "CARRY/VOL", "SIGNAL"]])
            body = []
            for r in rows:
                sig_color = "#00cc66" if "SELL" in r.get("signal", "") else "#ff3333" if "BUY" in r.get("signal", "") else "#808080"
                def _rv_fmt(v, fmt=".1f", suffix=""):
                    return f"{v:{fmt}}{suffix}" if np.isfinite(v) else "—"
                ts = r['term_spread']
                cv = r['carry_vol']
                ts_color = "#ff3333" if np.isfinite(ts) and ts > 0.5 else "#00cc66" if np.isfinite(ts) and ts < -0.5 else "#d4d4d4"
                cv_color = "#00cc66" if np.isfinite(cv) and cv > 2 else "#ff3333" if np.isfinite(cv) and cv < -1 else "#d4d4d4"
                body.append(html.Tr([
                    html.Td(r["pair"], style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
                    html.Td(_rv_fmt(r['atm_1m'], ".1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(r['atm_3m'], ".1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(r['atm_1y'], ".1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(ts, "+.1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right", "color": ts_color}),
                    html.Td(_rv_fmt(r['fwd_3x3'], ".1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(r['carry_day'], ".3f"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(cv, ".1f", "%"), style={**TABLE_CELL_STYLE, "textAlign": "right",
                             "fontWeight": "700", "color": cv_color}),
                    html.Td(r["signal"], style={**TABLE_CELL_STYLE, "color": sig_color, "fontWeight": "600"}),
                ]))

            return html.Table([html.Thead(header), html.Tbody(body)],
                              style={"width": "100%", "borderCollapse": "collapse",
                                     "fontFamily": _MONO, "fontSize": "10px"})
        except Exception as exc:
            logging.exception("update_carry_table failed")
            return html.Div("Carry data error", style={"color": "#ff3333", "padding": "8px"})

    @app.callback(
        [Output(f"{_P}-carry-term", "figure"), Output(f"{_P}-carry-fwd", "figure"),
         Output(f"{_P}-carry-calendar", "figure")],
        [Input(f"{_P}-carry-pair-dd", "value"), Input(f"{_P}-carry-comp-dd", "value")],
    )
    def update_carry_charts(pair, comp):
        try:
            p = pair or "EURUSD"
            return _build_term_chart(p, comp), _build_fwd_vol(p), _build_calendar_spread(p)
        except Exception as exc:
            logging.exception("update_carry_charts failed")
            empty = no_data_fig("Error")
            return empty, empty, empty

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output(f"{_P}-csv-download", "data"),
        [Input(f"{_P}-csv-vol-spread", "n_clicks"),
         Input(f"{_P}-csv-skew-scatter", "n_clicks"),
         Input(f"{_P}-csv-zscore", "n_clicks"),
         Input(f"{_P}-csv-ivrv", "n_clicks"),
         Input(f"{_P}-csv-corr", "n_clicks"),
         Input(f"{_P}-csv-rolling", "n_clicks"),
         Input(f"{_P}-csv-cb", "n_clicks"),
         Input(f"{_P}-csv-dxy", "n_clicks"),
         Input(f"{_P}-csv-carry-term", "n_clicks"),
         Input(f"{_P}-csv-carry-fwd", "n_clicks"),
         Input(f"{_P}-csv-carry-cal", "n_clicks")],
        [State(f"{_P}-vol-spread", "figure"),
         State(f"{_P}-skew-scatter", "figure"),
         State(f"{_P}-zscore-matrix", "figure"),
         State(f"{_P}-ivrv-panel", "figure"),
         State(f"{_P}-corr-heatmap", "figure"),
         State(f"{_P}-rolling-chart", "figure"),
         State(f"{_P}-cb-chart", "figure"),
         State(f"{_P}-dxy-chart", "figure"),
         State(f"{_P}-carry-term", "figure"),
         State(f"{_P}-carry-fwd", "figure"),
         State(f"{_P}-carry-calendar", "figure")],
        prevent_initial_call=True,
    )
    def rvp_csv_export(*args):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        names = ["vol-spread", "skew-scatter", "zscore", "ivrv",
                 "corr", "rolling", "cb", "dxy",
                 "carry-term", "carry-fwd", "carry-cal"]
        labels = ["VolSpread", "SkewScatter", "ZScoreMatrix", "IVRV",
                  "CorrHeatmap", "RollingCorr", "CBDivergence", "DXYProxy",
                  "CarryTerm", "CarryFwd", "CarryCalendar"]
        n = len(names)
        for i, name in enumerate(names):
            if btn == f"{_P}-csv-{name}":
                fig = args[n + i]
                if fig:
                    return export_csv(fig, "RelValue", labels[i])
                return no_update
        return no_update
