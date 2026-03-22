"""
Cross-Pair Relative Value Analytics Panel
==========================================
Institutional FX options relative-value workstation.

Provides cross-pair vol spread analysis, z-score screens, IV-RV spread
monitoring, skew scatter regression, term-structure comparison, rolling
correlation tracking, and a composite signal table for rapid opportunity
identification across the full FX pair universe.
"""

import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, linregress

from core.theme import COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE, LABEL_STYLE, DROPDOWN_STYLE
from core.bloomberg_fx import (
    get_fx_vol_surface, get_fx_spots, get_fx_rates, get_all_pairs,
    get_fx_historical_vol, get_fx_historical_spot,
)
from core.fx_analytics import (
    cross_pair_vol_spread, cross_pair_rr_spread, vol_beta, rv_scanner,
    rv_signal_composite, implied_correlation, correlation_richness,
    spot_correlation_matrix, vol_zscore, vol_percentile, iv_rv_spread,
    carry_per_vol, vol_zscore_surface,
)
from core.fx_conventions import FX_PAIR_REGISTRY, tenor_to_years


# ============================================================================
# Constants
# ============================================================================

ALL_PAIRS = sorted(FX_PAIR_REGISTRY.keys())

METRIC_OPTIONS = [
    {"label": "ATM Vol", "value": "ATM"},
    {"label": "25D RR", "value": "25D_RR"},
    {"label": "25D BF", "value": "25D_BF"},
    {"label": "IV-RV Spread", "value": "IV_RV"},
    {"label": "Term Spread", "value": "TERM"},
]

TENOR_OPTIONS = [
    {"label": t, "value": t}
    for t in ["1M", "2M", "3M", "6M", "1Y", "2Y"]
]

LOOKBACK_OPTIONS = [
    {"label": "60d", "value": 60},
    {"label": "120d", "value": 120},
    {"label": "252d (1Y)", "value": 252},
    {"label": "504d (2Y)", "value": 504},
]

MODE_OPTIONS = [
    {"label": "Pair vs Pair", "value": "pair_vs_pair"},
    {"label": "Multi-Pair Scan", "value": "multi_scan"},
    {"label": "Correlation", "value": "correlation"},
]

HEATMAP_TENORS = ["1M", "3M", "6M", "1Y"]

CARD_HEADER_STYLE = {
    "color": COLORS["text_primary"],
    "fontSize": "13px",
    "fontWeight": "700",
    "fontFamily": "'JetBrains Mono', monospace",
    "marginBottom": "18px",
    "paddingBottom": "12px",
    "borderBottom": f"1px solid {COLORS['border_subtle']}",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": COLORS["bg_secondary"],
    "color": COLORS["text_secondary"],
    "fontWeight": "700",
    "fontSize": "10px",
    "textTransform": "uppercase",
    "letterSpacing": "1.2px",
    "border": f"1px solid {COLORS['border_subtle']}",
    "padding": "10px 12px",
}

TABLE_CELL_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["text_primary"],
    "fontSize": "12px",
    "fontFamily": "'JetBrains Mono', monospace",
    "border": f"1px solid {COLORS['border_subtle']}",
    "padding": "8px 12px",
}


# ============================================================================
# Helpers
# ============================================================================

def _empty_fig(title: str = "") -> go.Figure:
    """Return a transparent placeholder figure with optional title."""
    fig = go.Figure()
    fig.update_layout(
        **CHART_TEMPLATE["layout"],
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=COLORS["text_muted"], size=14),
        )],
    )
    return fig


def _safe_array(val):
    """Convert series / list to numpy array."""
    if isinstance(val, pd.Series):
        return val.values.astype(float)
    if isinstance(val, list):
        return np.array(val, dtype=float)
    return np.asarray(val, dtype=float)


def _extract_atm_from_surface(surface: dict, tenor: str) -> float:
    """Pull ATM vol from the surface dict returned by get_fx_vol_surface."""
    if isinstance(surface, dict) and tenor in surface:
        v = surface[tenor]
        if isinstance(v, dict):
            return v.get("atm", v.get("ATM", 8.0))
    for t in ["3M", "1M", "6M", "1Y"]:
        if t in surface:
            v = surface[t]
            if isinstance(v, dict):
                return v.get("atm", v.get("ATM", 8.0))
    return 8.0


def _extract_rr25_from_surface(surface: dict, tenor: str) -> float:
    """Pull 25D RR from the surface dict."""
    if isinstance(surface, dict) and tenor in surface:
        v = surface[tenor]
        if isinstance(v, dict):
            return v.get("rr25", v.get("25D_RR", 0.0))
    return 0.0


def _zscore_color(z: float) -> str:
    """Map a z-score to a colour string."""
    if z > 2.0:
        return COLORS["accent_red"]
    elif z > 1.0:
        return COLORS["accent_orange"]
    elif z < -2.0:
        return COLORS["accent_cyan"]
    elif z < -1.0:
        return COLORS["accent_blue"]
    return COLORS["text_muted"]


def _signal_direction(composite: float, rr_z: float) -> str:
    """Derive a human-readable signal direction from composite + skew z."""
    if abs(rr_z) > abs(composite) and abs(rr_z) > 1.0:
        return "BUY SKEW" if rr_z < -1.0 else "SELL SKEW"
    if composite < -30:
        return "BUY VOL"
    if composite > 30:
        return "SELL VOL"
    if composite < -15:
        return "BUY VOL"
    if composite > 15:
        return "SELL VOL"
    return "NEUTRAL"


def _confidence_label(composite: float) -> str:
    """Confidence bucket from |composite|."""
    a = abs(composite)
    if a > 50:
        return "HIGH"
    if a > 25:
        return "MEDIUM"
    return "LOW"


def _confidence_color(label: str) -> str:
    """Colour for confidence label."""
    return {
        "HIGH": COLORS["accent_green"],
        "MEDIUM": COLORS["accent_orange"],
        "LOW": COLORS["text_muted"],
    }.get(label, COLORS["text_muted"])


# ============================================================================
# Layout
# ============================================================================

def layout():
    pair_opts = [{"label": p, "value": p} for p in ALL_PAIRS]

    return html.Div([
        # ── Controls Row ────────────────────────────────────────────
        html.Div([
            html.Div("CROSS-PAIR RV ANALYTICS", style=CARD_HEADER_STYLE),
            html.Div([
                # Pair A
                html.Div([
                    html.Label("PAIR A", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="rv-pair-a",
                        options=pair_opts,
                        value="EURUSD",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "130px", "marginRight": "12px"}),

                # Pair B
                html.Div([
                    html.Label("PAIR B", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="rv-pair-b",
                        options=pair_opts,
                        value="GBPUSD",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "130px", "marginRight": "12px"}),

                # Metric
                html.Div([
                    html.Label("METRIC", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="rv-metric",
                        options=METRIC_OPTIONS,
                        value="ATM",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "140px", "marginRight": "12px"}),

                # Tenor
                html.Div([
                    html.Label("TENOR", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="rv-tenor",
                        options=TENOR_OPTIONS,
                        value="3M",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "100px", "marginRight": "12px"}),

                # Lookback
                html.Div([
                    html.Label("LOOKBACK", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="rv-lookback",
                        options=LOOKBACK_OPTIONS,
                        value=252,
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "110px", "marginRight": "12px"}),

                # Analysis mode
                html.Div([
                    html.Label("MODE", style=LABEL_STYLE),
                    dcc.RadioItems(
                        id="rv-mode",
                        options=MODE_OPTIONS,
                        value="pair_vs_pair",
                        inline=True,
                        style={"fontSize": "11px", "color": COLORS["text_secondary"]},
                        inputStyle={"marginRight": "4px"},
                        labelStyle={
                            "marginRight": "14px",
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "10px",
                            "letterSpacing": "0.5px",
                        },
                    ),
                ], style={"flex": "2", "minWidth": "280px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Chart Grid (3x2) ───────────────────────────────────────
        # Row 1
        html.Div([
            html.Div([
                dcc.Graph(id="rv-vol-spread-ts", style={"height": "380px"}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
               className="dashboard-card"),

            html.Div([
                dcc.Graph(id="rv-term-structure", style={"height": "380px"}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
               className="dashboard-card"),

            html.Div([
                dcc.Graph(id="rv-skew-scatter", style={"height": "380px"}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
               className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # Row 2
        html.Div([
            html.Div([
                dcc.Graph(id="rv-zscore-matrix", style={"height": "400px"}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
               className="dashboard-card"),

            html.Div([
                dcc.Graph(id="rv-ivrv-panel", style={"height": "400px"}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
               className="dashboard-card"),

            html.Div([
                dcc.Graph(id="rv-correlation-dash", style={"height": "400px"}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
               className="dashboard-card"),
        ], style={
            "display": "flex", "gap": "16px", "flexWrap": "wrap",
            "marginTop": "16px",
        }),

        # ── RV Signal Summary Table ────────────────────────────────
        html.Div([
            html.Div("RV SIGNAL SUMMARY — ALL PAIRS RANKED BY COMPOSITE STRENGTH",
                      style=CARD_HEADER_STYLE),
            html.Div(id="rv-signal-table-container"),
        ], style={**CARD_STYLE, "marginTop": "16px"}, className="dashboard-card"),
    ])


# ============================================================================
# Callbacks
# ============================================================================

def register_callbacks(app):

    # ------------------------------------------------------------------
    # Main callback: update all 6 charts
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("rv-vol-spread-ts", "figure"),
            Output("rv-term-structure", "figure"),
            Output("rv-skew-scatter", "figure"),
            Output("rv-zscore-matrix", "figure"),
            Output("rv-ivrv-panel", "figure"),
            Output("rv-correlation-dash", "figure"),
        ],
        [
            Input("rv-pair-a", "value"),
            Input("rv-pair-b", "value"),
            Input("rv-metric", "value"),
            Input("rv-tenor", "value"),
            Input("rv-lookback", "value"),
            Input("rv-mode", "value"),
        ],
    )
    def update_all_charts(pair_a, pair_b, metric, tenor, lookback, mode):
        pair_a = pair_a or "EURUSD"
        pair_b = pair_b or "GBPUSD"
        metric = metric or "ATM"
        tenor = tenor or "3M"
        lookback = lookback or 252

        # Build each chart independently; if one fails, return a clean
        # empty figure so the remaining charts still render.
        def _safe_build(builder, *args, title=""):
            try:
                return builder(*args)
            except Exception:
                return _empty_fig(f"{title} -- build error")

        fig_spread = _safe_build(_build_vol_spread_ts, pair_a, pair_b, metric, tenor, lookback,
                                 title="VOL SPREAD")
        fig_term = _safe_build(_build_term_structure, pair_a, pair_b, lookback,
                               title="TERM STRUCTURE")
        fig_scatter = _safe_build(_build_skew_scatter, pair_a, pair_b, tenor, lookback,
                                  title="SKEW SCATTER")
        fig_zscore = _safe_build(_build_zscore_matrix, lookback,
                                 title="Z-SCORE MATRIX")
        fig_ivrv = _safe_build(_build_ivrv_panel, pair_a, tenor, lookback,
                               title="IV-RV SPREAD")
        fig_corr = _safe_build(_build_correlation_dash, pair_a, pair_b, lookback,
                               title="CORRELATION")

        return fig_spread, fig_term, fig_scatter, fig_zscore, fig_ivrv, fig_corr

    # ------------------------------------------------------------------
    # Signal table callback — always shows full scan
    # ------------------------------------------------------------------
    @app.callback(
        Output("rv-signal-table-container", "children"),
        [
            Input("rv-lookback", "value"),
        ],
    )
    def update_signal_table(lookback):
        lookback = lookback or 252
        return _build_signal_table(lookback)


# ============================================================================
# Chart Builders
# ============================================================================

def _build_vol_spread_ts(pair_a, pair_b, metric, tenor, lookback):
    """
    Chart 1 — Vol Spread Time Series.
    Pair A metric (bold cyan), Pair B metric (bold rose), spread as filled
    area between, z-score on secondary Y-axis (dashed white), current level
    annotated with value + z-score + percentile.
    """
    try:
        # Resolve metric label for data retrieval
        metric_key = "ATM" if metric in ("ATM", "IV_RV", "TERM") else metric

        # Fetch historical data for both pairs
        hist_a = get_fx_historical_vol(pair_a, tenor, metric_key, lookback)
        hist_b = get_fx_historical_vol(pair_b, tenor, metric_key, lookback)

        arr_a = _safe_array(hist_a) if hist_a is not None and len(hist_a) > 10 else None
        arr_b = _safe_array(hist_b) if hist_b is not None and len(hist_b) > 10 else None

        if arr_a is None or arr_b is None:
            return _empty_fig("VOL SPREAD — insufficient data")

        n = min(len(arr_a), len(arr_b))
        arr_a = arr_a[-n:]
        arr_b = arr_b[-n:]
        days = np.arange(n)

        spread = arr_a - arr_b
        mu = np.mean(spread)
        sigma = np.std(spread)
        zscore_ts = (spread - mu) / max(sigma, 1e-6)

        current_a = arr_a[-1]
        current_b = arr_b[-1]
        current_spread = spread[-1]
        current_z = zscore_ts[-1]
        current_pct = percentileofscore(spread, current_spread)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Pair A line
        fig.add_trace(go.Scatter(
            x=days, y=arr_a, mode="lines",
            name=f"{pair_a} {metric_key}",
            line=dict(color=COLORS["accent_cyan"], width=2.5),
            hovertemplate="%{y:.2f}",
        ), secondary_y=False)

        # Pair B line
        fig.add_trace(go.Scatter(
            x=days, y=arr_b, mode="lines",
            name=f"{pair_b} {metric_key}",
            line=dict(color=COLORS["accent_rose"], width=2.5),
            hovertemplate="%{y:.2f}",
        ), secondary_y=False)

        # Filled area between the two
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([arr_a, arr_b[::-1]]),
            fill="toself",
            fillcolor=f"{COLORS['accent_purple']}18",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ), secondary_y=False)

        # Z-score on secondary axis
        fig.add_trace(go.Scatter(
            x=days, y=zscore_ts, mode="lines",
            name="Z-Score",
            line=dict(color="rgba(255,255,255,0.55)", width=1.5, dash="dash"),
            hovertemplate="z=%{y:.2f}",
        ), secondary_y=True)

        # Zero line for z-score
        fig.add_hline(y=0, line=dict(color=COLORS["border"], width=0.5, dash="dot"),
                       secondary_y=True)
        fig.add_hline(y=2.0, line=dict(color=COLORS["accent_red"], width=0.5, dash="dot"),
                       secondary_y=True)
        fig.add_hline(y=-2.0, line=dict(color=COLORS["accent_cyan"], width=0.5, dash="dot"),
                       secondary_y=True)

        # Current level annotation
        annotation_text = (
            f"<b>Spread: {current_spread:+.2f}</b> | "
            f"Z: {current_z:+.2f} | "
            f"Pctl: {current_pct:.0f}%"
        )
        fig.add_annotation(
            x=days[-1], y=current_a, xref="x", yref="y",
            text=annotation_text,
            showarrow=True, arrowhead=2, arrowcolor=COLORS["accent_cyan"],
            ax=-120, ay=-35,
            font=dict(
                color=COLORS["text_primary"], size=10,
                family="'JetBrains Mono', monospace",
            ),
            bgcolor=f"{COLORS['bg_secondary']}E0",
            bordercolor=COLORS["border"],
            borderwidth=1,
            borderpad=6,
        )

        fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title=f"VOL SPREAD — {pair_a} vs {pair_b} ({metric_key} {tenor})",
            xaxis_title="Trading Days",
            yaxis_title="Vol Level (%)",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9)),
            margin=dict(l=55, r=55, t=45, b=40),
        )
        fig.update_yaxes(title_text="Z-Score", secondary_y=True,
                         gridcolor="rgba(30,42,69,0.25)")

        return fig

    except Exception as e:
        return _empty_fig(f"VOL SPREAD — error: {str(e)[:60]}")


def _build_term_structure(pair_a, pair_b, lookback):
    """
    Chart 2 — Cross-Pair Term Structure.
    Pair A ATM across tenors (cyan), Pair B ATM (rose). Difference as
    coloured bars in a subplot below; colour bars by z-score magnitude.
    """
    try:
        tenors_list = ["1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]
        surface_a = get_fx_vol_surface(pair_a)
        surface_b = get_fx_vol_surface(pair_b)

        atm_a = []
        atm_b = []
        diffs = []
        bar_colors = []

        for t in tenors_list:
            va = _extract_atm_from_surface(surface_a, t)
            vb = _extract_atm_from_surface(surface_b, t)
            atm_a.append(va)
            atm_b.append(vb)
            d = va - vb
            diffs.append(d)

            # Z-score of this difference vs lookback
            try:
                z_info = vol_zscore(pair_a, t, "ATM", lookback)
                z_val = z_info.get("zscore", 0.0)
            except Exception:
                z_val = 0.0

            if abs(z_val) > 2.0:
                bar_colors.append(COLORS["accent_red"] if z_val > 0 else COLORS["accent_cyan"])
            elif abs(z_val) > 1.0:
                bar_colors.append(COLORS["accent_orange"] if z_val > 0 else COLORS["accent_blue"])
            else:
                bar_colors.append(COLORS["accent_purple"])

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.08,
        )

        # Pair A term structure
        fig.add_trace(go.Scatter(
            x=tenors_list, y=atm_a, mode="lines+markers",
            name=pair_a,
            line=dict(color=COLORS["accent_cyan"], width=2.5),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="%{x}: %{y:.2f}%",
        ), row=1, col=1)

        # Pair B term structure
        fig.add_trace(go.Scatter(
            x=tenors_list, y=atm_b, mode="lines+markers",
            name=pair_b,
            line=dict(color=COLORS["accent_rose"], width=2.5),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="%{x}: %{y:.2f}%",
        ), row=1, col=1)

        # Difference bars
        fig.add_trace(go.Bar(
            x=tenors_list, y=diffs, name="Diff (A-B)",
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate="%{x}: %{y:+.2f}",
            opacity=0.85,
        ), row=2, col=1)

        # Zero line on bar chart
        fig.add_hline(y=0, row=2, col=1,
                       line=dict(color=COLORS["border"], width=0.5, dash="dot"))

        fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title=f"TERM STRUCTURE — {pair_a} vs {pair_b} (ATM)",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9)),
            margin=dict(l=55, r=20, t=45, b=40),
        )
        fig.update_yaxes(title_text="ATM Vol (%)", row=1, col=1)
        fig.update_yaxes(title_text="Diff (vol pts)", row=2, col=1)

        return fig

    except Exception as e:
        return _empty_fig(f"TERM STRUCTURE — error: {str(e)[:60]}")


def _build_skew_scatter(pair_a, pair_b, tenor, lookback):
    """
    Chart 3 — Skew Scatter.
    X = Pair A 25D RR, Y = Pair B 25D RR. Historical points as small gray
    dots, current point as large red diamond, regression line (dashed), and
    R-squared annotation.
    """
    try:
        hist_a = get_fx_historical_vol(pair_a, tenor, "25D_RR", lookback)
        hist_b = get_fx_historical_vol(pair_b, tenor, "25D_RR", lookback)

        arr_a = _safe_array(hist_a) if hist_a is not None and len(hist_a) > 10 else None
        arr_b = _safe_array(hist_b) if hist_b is not None and len(hist_b) > 10 else None

        if arr_a is None or arr_b is None:
            return _empty_fig("SKEW SCATTER — insufficient data")

        n = min(len(arr_a), len(arr_b))
        xa = arr_a[-n:]
        yb = arr_b[-n:]

        # Regression
        slope, intercept, r_value, _, _ = linregress(xa, yb)
        r_sq = r_value ** 2
        fit_x = np.linspace(xa.min(), xa.max(), 50)
        fit_y = slope * fit_x + intercept

        fig = go.Figure()

        # Historical points (small gray)
        fig.add_trace(go.Scatter(
            x=xa[:-1], y=yb[:-1], mode="markers",
            name="Historical",
            marker=dict(size=4, color=COLORS["text_muted"], opacity=0.4),
            hovertemplate=f"{pair_a} RR: %{{x:.2f}}<br>{pair_b} RR: %{{y:.2f}}",
        ))

        # Current point (large red diamond)
        fig.add_trace(go.Scatter(
            x=[xa[-1]], y=[yb[-1]], mode="markers",
            name="Current",
            marker=dict(
                size=14, color=COLORS["accent_red"],
                symbol="diamond", line=dict(width=2, color="white"),
            ),
            hovertemplate=(
                f"<b>CURRENT</b><br>"
                f"{pair_a} RR: %{{x:.2f}}<br>"
                f"{pair_b} RR: %{{y:.2f}}"
            ),
        ))

        # Regression line
        fig.add_trace(go.Scatter(
            x=fit_x, y=fit_y, mode="lines",
            name=f"Regression (R²={r_sq:.3f})",
            line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="dash"),
            hoverinfo="skip",
        ))

        # R² annotation
        fig.add_annotation(
            x=0.95, y=0.05, xref="paper", yref="paper",
            text=(
                f"<b>R² = {r_sq:.3f}</b><br>"
                f"beta = {slope:.3f}<br>"
                f"n = {n}"
            ),
            showarrow=False,
            font=dict(
                color=COLORS["accent_cyan"], size=11,
                family="'JetBrains Mono', monospace",
            ),
            bgcolor=f"{COLORS['bg_secondary']}D0",
            bordercolor=COLORS["border"],
            borderpad=8,
            align="right",
        )

        fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title=f"SKEW SCATTER — {pair_a} vs {pair_b} (25D RR, {tenor})",
            xaxis_title=f"{pair_a} 25D RR",
            yaxis_title=f"{pair_b} 25D RR",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9)),
            margin=dict(l=55, r=20, t=45, b=45),
        )

        return fig

    except Exception as e:
        return _empty_fig(f"SKEW SCATTER — error: {str(e)[:60]}")


def _build_zscore_matrix(lookback):
    """
    Chart 4 — Z-Score Matrix (opportunity screen).
    ALL pairs x key tenors (1M, 3M, 6M, 1Y) heatmap. Colour = z-score of
    ATM vol: blue (cheap) to red (rich). Numbers = z-score value.
    """
    try:
        pairs = ALL_PAIRS
        tenors = HEATMAP_TENORS

        z_matrix = np.zeros((len(pairs), len(tenors)))

        for i, p in enumerate(pairs):
            for j, t in enumerate(tenors):
                try:
                    info = vol_zscore(p, t, "ATM", lookback)
                    z_matrix[i, j] = info.get("zscore", 0.0)
                except Exception:
                    z_matrix[i, j] = 0.0

        # Annotation text = z-score values
        text_matrix = [[f"{z_matrix[i, j]:+.1f}" for j in range(len(tenors))]
                        for i in range(len(pairs))]

        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=tenors,
            y=pairs,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=9, family="'JetBrains Mono', monospace"),
            colorscale=[
                [0.0, COLORS["accent_cyan"]],
                [0.25, COLORS["accent_blue"]],
                [0.5, COLORS["bg_secondary"]],
                [0.75, COLORS["accent_orange"]],
                [1.0, COLORS["accent_red"]],
            ],
            zmid=0,
            zmin=-3.0,
            zmax=3.0,
            colorbar=dict(
                title=dict(text="Z-Score", font=dict(size=10)),
                thickness=12,
                len=0.9,
                tickfont=dict(size=9),
            ),
            hovertemplate=(
                "Pair: %{y}<br>"
                "Tenor: %{x}<br>"
                "Z-Score: %{z:.2f}<extra></extra>"
            ),
        ))

        fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title="Z-SCORE MATRIX — ATM VOL OPPORTUNITY SCREEN",
            xaxis=dict(
                tickfont=dict(size=10, color=COLORS["text_secondary"]),
                side="top",
            ),
            yaxis=dict(
                tickfont=dict(size=9, color=COLORS["text_secondary"]),
                autorange="reversed",
            ),
            margin=dict(l=80, r=30, t=55, b=20),
        )

        return fig

    except Exception as e:
        return _empty_fig(f"Z-SCORE MATRIX — error: {str(e)[:60]}")


def _build_ivrv_panel(pair, tenor, lookback):
    """
    Chart 5 — IV-RV Spread Panel.
    IV minus RV over lookback period. Filled area (green when IV > RV, red
    when IV < RV). Distribution histogram on right margin. Current percentile
    annotated.
    """
    try:
        df = iv_rv_spread(pair, tenor, lookback=lookback)

        if df is None or df.empty:
            return _empty_fig(f"IV-RV SPREAD — {pair} no data")

        days = df["day"].values
        spread_vals = df["spread"].values
        iv_vals = df["iv"].values
        rv_vals = df["rv"].values

        # Compute percentile of current spread
        current_spread = spread_vals[-1]
        spread_pct = percentileofscore(spread_vals, current_spread)

        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.78, 0.22],
            shared_yaxes=True, horizontal_spacing=0.03,
        )

        # Positive spread fill (IV > RV) — green
        pos_spread = np.where(spread_vals >= 0, spread_vals, 0)
        neg_spread = np.where(spread_vals < 0, spread_vals, 0)

        # Full spread line
        fig.add_trace(go.Scatter(
            x=days, y=spread_vals, mode="lines",
            name="IV - RV",
            line=dict(color="rgba(255,255,255,0.5)", width=1),
            hovertemplate="Day %{x}: %{y:.2f}",
        ), row=1, col=1)

        # Green fill (IV > RV)
        fig.add_trace(go.Scatter(
            x=days, y=pos_spread, mode="lines",
            fill="tozeroy",
            fillcolor=f"{COLORS['accent_green']}30",
            line=dict(width=0),
            name="IV Rich",
            showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)

        # Red fill (IV < RV)
        fig.add_trace(go.Scatter(
            x=days, y=neg_spread, mode="lines",
            fill="tozeroy",
            fillcolor=f"{COLORS['accent_red']}30",
            line=dict(width=0),
            name="IV Cheap",
            showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)

        # Zero line
        fig.add_hline(y=0, row=1, col=1,
                       line=dict(color=COLORS["border"], width=0.8, dash="dot"))

        # Current spread marker
        fig.add_trace(go.Scatter(
            x=[days[-1]], y=[current_spread], mode="markers",
            name="Current",
            marker=dict(
                size=10, color=COLORS["accent_cyan"],
                symbol="circle", line=dict(width=2, color="white"),
            ),
            showlegend=False,
        ), row=1, col=1)

        # Distribution histogram on right margin
        fig.add_trace(go.Histogram(
            y=spread_vals, nbinsy=35,
            marker=dict(
                color=COLORS["accent_purple"],
                line=dict(width=0.5, color=COLORS["border"]),
            ),
            opacity=0.7,
            name="Distribution",
            showlegend=False,
            hovertemplate="Spread: %{y:.2f}<br>Count: %{x}",
        ), row=1, col=2)

        # Current spread horizontal line
        fig.add_hline(
            y=current_spread, row=1, col=2,
            line=dict(color=COLORS["accent_cyan"], width=1.5, dash="dash"),
        )

        # Annotate current percentile
        spread_color = COLORS["accent_green"] if current_spread >= 0 else COLORS["accent_red"]
        fig.add_annotation(
            x=0.72, y=0.95, xref="paper", yref="paper",
            text=(
                f"<b>Current: {current_spread:+.2f}</b><br>"
                f"Percentile: {spread_pct:.0f}%<br>"
                f"IV: {iv_vals[-1]:.2f}  RV: {rv_vals[-1]:.2f}"
            ),
            showarrow=False,
            font=dict(
                color=spread_color, size=10,
                family="'JetBrains Mono', monospace",
            ),
            bgcolor=f"{COLORS['bg_secondary']}E0",
            bordercolor=COLORS["border"],
            borderpad=8,
            align="left",
        )

        fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title=f"IV-RV SPREAD — {pair} ({tenor})",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9)),
            margin=dict(l=55, r=20, t=45, b=40),
        )
        fig.update_yaxes(title_text="Spread (vol pts)", row=1, col=1)
        fig.update_xaxes(title_text="Trading Days", row=1, col=1)
        fig.update_xaxes(title_text="Count", row=1, col=2)

        return fig

    except Exception as e:
        return _empty_fig(f"IV-RV SPREAD — error: {str(e)[:60]}")


def _build_correlation_dash(pair_a, pair_b, lookback):
    """
    Chart 6 — Correlation Dashboard.
    Rolling spot correlation (120d window) between Pair A and Pair B.
    Shows current correlation, 1Y range, and regime tag
    (HIGH CORR / LOW CORR / DECORRELATED).
    """
    try:
        corr_window = 120
        hist_days = max(lookback + corr_window + 50, 504)

        # Fetch spot histories
        spot_a = get_fx_historical_spot(pair_a, hist_days)
        spot_b = get_fx_historical_spot(pair_b, hist_days)

        if spot_a is None or spot_b is None:
            return _empty_fig(f"CORRELATION — no spot data")

        # Handle DataFrame vs array
        if isinstance(spot_a, pd.DataFrame):
            close_a = spot_a["close"].values if "close" in spot_a.columns else spot_a.iloc[:, -1].values
        else:
            close_a = _safe_array(spot_a)

        if isinstance(spot_b, pd.DataFrame):
            close_b = spot_b["close"].values if "close" in spot_b.columns else spot_b.iloc[:, -1].values
        else:
            close_b = _safe_array(spot_b)

        n = min(len(close_a), len(close_b))
        close_a = close_a[-n:]
        close_b = close_b[-n:]

        if n < corr_window + 10:
            return _empty_fig(f"CORRELATION — insufficient history")

        # Log returns
        ret_a = np.diff(np.log(close_a))
        ret_b = np.diff(np.log(close_b))

        # Rolling correlation
        rolling_corr = np.full(len(ret_a), np.nan)
        for i in range(corr_window - 1, len(ret_a)):
            window_a = ret_a[i - corr_window + 1:i + 1]
            window_b = ret_b[i - corr_window + 1:i + 1]
            if len(window_a) == corr_window:
                c = np.corrcoef(window_a, window_b)[0, 1]
                rolling_corr[i] = c

        valid_mask = ~np.isnan(rolling_corr)
        valid_corr = rolling_corr[valid_mask]
        valid_days = np.where(valid_mask)[0]

        if len(valid_corr) < 5:
            return _empty_fig(f"CORRELATION — not enough overlap")

        current_corr = valid_corr[-1]
        corr_max = np.max(valid_corr)
        corr_min = np.min(valid_corr)
        corr_mean = np.mean(valid_corr)

        # Regime tag
        if abs(current_corr) > 0.7:
            regime_tag = "HIGH CORR"
            regime_color = COLORS["accent_green"]
        elif abs(current_corr) > 0.3:
            regime_tag = "LOW CORR"
            regime_color = COLORS["accent_orange"]
        else:
            regime_tag = "DECORRELATED"
            regime_color = COLORS["accent_red"]

        fig = go.Figure()

        # Rolling correlation line
        fig.add_trace(go.Scatter(
            x=valid_days, y=valid_corr, mode="lines",
            name=f"Rolling {corr_window}d Correlation",
            line=dict(color=COLORS["accent_cyan"], width=2),
            hovertemplate="Day %{x}: %{y:.3f}",
        ))

        # Fill area
        fig.add_trace(go.Scatter(
            x=valid_days, y=valid_corr, mode="lines",
            fill="tozeroy",
            fillcolor=f"{COLORS['accent_cyan']}15",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Current level marker
        fig.add_trace(go.Scatter(
            x=[valid_days[-1]], y=[current_corr], mode="markers",
            name=f"Current: {current_corr:.3f}",
            marker=dict(
                size=12, color=regime_color,
                symbol="diamond", line=dict(width=2, color="white"),
            ),
        ))

        # Mean line
        fig.add_hline(
            y=corr_mean,
            line=dict(color=COLORS["text_muted"], width=1, dash="dash"),
            annotation_text=f"Mean: {corr_mean:.3f}",
            annotation_position="bottom right",
            annotation_font=dict(size=9, color=COLORS["text_muted"]),
        )

        # Reference lines
        fig.add_hline(y=0, line=dict(color=COLORS["border"], width=0.5, dash="dot"))
        fig.add_hline(y=0.7, line=dict(color=f"{COLORS['accent_green']}50", width=0.5, dash="dot"))
        fig.add_hline(y=-0.7, line=dict(color=f"{COLORS['accent_green']}50", width=0.5, dash="dot"))

        # Regime and stats annotation
        fig.add_annotation(
            x=0.02, y=0.97, xref="paper", yref="paper",
            text=(
                f"<b>{regime_tag}</b><br>"
                f"Current: {current_corr:+.3f}<br>"
                f"1Y Range: [{corr_min:+.3f}, {corr_max:+.3f}]"
            ),
            showarrow=False,
            font=dict(
                color=regime_color, size=11,
                family="'JetBrains Mono', monospace",
            ),
            bgcolor=f"{COLORS['bg_secondary']}E0",
            bordercolor=regime_color,
            borderwidth=1,
            borderpad=8,
            align="left",
        )

        fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title=f"CORRELATION — {pair_a} vs {pair_b} ({corr_window}d Rolling)",
            yaxis_title="Correlation",
            xaxis_title="Trading Days",
            yaxis=dict(range=[-1.05, 1.05]),
            legend=dict(x=0.55, y=0.05, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9)),
            margin=dict(l=55, r=20, t=45, b=40),
        )

        return fig

    except Exception as e:
        return _empty_fig(f"CORRELATION — error: {str(e)[:60]}")


# ============================================================================
# Signal Table Builder
# ============================================================================

def _build_signal_table(lookback):
    """
    RV Signal Summary Table — all pairs ranked by composite signal strength.
    Columns: Pair | ATM 3M Z | RR 3M Z | IV-RV Z | Term Z | Composite |
             Signal Direction | Confidence
    """
    try:
        pairs = ALL_PAIRS
        records = []

        for pair in pairs:
            try:
                # ATM z-score
                atm_info = vol_zscore(pair, "3M", "ATM", lookback)
                atm_z = atm_info.get("zscore", 0.0)

                # RR z-score
                rr_info = vol_zscore(pair, "3M", "25D_RR", lookback)
                rr_z = rr_info.get("zscore", 0.0)

                # IV-RV spread z-score
                try:
                    ivrv_df = iv_rv_spread(pair, "3M", lookback=lookback)
                    if ivrv_df is not None and not ivrv_df.empty:
                        spread_vals = ivrv_df["spread"].values
                        current_sp = spread_vals[-1]
                        sp_mu = np.mean(spread_vals)
                        sp_std = np.std(spread_vals)
                        ivrv_z = (current_sp - sp_mu) / max(sp_std, 1e-6)
                    else:
                        ivrv_z = 0.0
                except Exception:
                    ivrv_z = 0.0

                # Term spread z-score (1Y - 1M)
                try:
                    z_1m = vol_zscore(pair, "1M", "ATM", lookback).get("zscore", 0.0)
                    z_1y = vol_zscore(pair, "1Y", "ATM", lookback).get("zscore", 0.0)
                    term_z = z_1m - z_1y
                except Exception:
                    term_z = 0.0

                # Composite signal
                try:
                    composite_info = rv_signal_composite(pair, lookback)
                    composite = composite_info.get("composite_score", 0.0)
                except Exception:
                    composite = (0.35 * atm_z + 0.25 * ivrv_z +
                                 0.20 * term_z + 0.20 * rr_z) * 25

                # Direction and confidence
                direction = _signal_direction(composite, rr_z)
                confidence = _confidence_label(composite)

                records.append({
                    "Pair": pair,
                    "ATM 3M Z": round(atm_z, 2),
                    "RR 3M Z": round(rr_z, 2),
                    "IV-RV Z": round(ivrv_z, 2),
                    "Term Z": round(term_z, 2),
                    "Composite": round(composite, 1),
                    "Signal Direction": direction,
                    "Confidence": confidence,
                })

            except Exception:
                records.append({
                    "Pair": pair,
                    "ATM 3M Z": 0.0,
                    "RR 3M Z": 0.0,
                    "IV-RV Z": 0.0,
                    "Term Z": 0.0,
                    "Composite": 0.0,
                    "Signal Direction": "N/A",
                    "Confidence": "LOW",
                })

        # Sort by absolute composite score descending
        records.sort(key=lambda r: abs(r["Composite"]), reverse=True)

        df = pd.DataFrame(records)

        # Conditional formatting style rules
        style_data_conditional = []

        # Z-score cells colouring
        z_columns = ["ATM 3M Z", "RR 3M Z", "IV-RV Z", "Term Z"]
        for col in z_columns:
            # Very high (rich)
            style_data_conditional.append({
                "if": {
                    "filter_query": f"{{{col}}} > 2",
                    "column_id": col,
                },
                "color": COLORS["accent_red"],
                "fontWeight": "bold",
            })
            # High
            style_data_conditional.append({
                "if": {
                    "filter_query": f"{{{col}}} > 1 && {{{col}}} <= 2",
                    "column_id": col,
                },
                "color": COLORS["accent_orange"],
            })
            # Very low (cheap)
            style_data_conditional.append({
                "if": {
                    "filter_query": f"{{{col}}} < -2",
                    "column_id": col,
                },
                "color": COLORS["accent_cyan"],
                "fontWeight": "bold",
            })
            # Low
            style_data_conditional.append({
                "if": {
                    "filter_query": f"{{{col}}} < -1 && {{{col}}} >= -2",
                    "column_id": col,
                },
                "color": COLORS["accent_blue"],
            })

        # Composite colouring
        style_data_conditional.extend([
            {
                "if": {
                    "filter_query": "{Composite} > 30",
                    "column_id": "Composite",
                },
                "color": COLORS["accent_red"],
                "fontWeight": "bold",
            },
            {
                "if": {
                    "filter_query": "{Composite} < -30",
                    "column_id": "Composite",
                },
                "color": COLORS["accent_cyan"],
                "fontWeight": "bold",
            },
        ])

        # Signal direction colouring
        style_data_conditional.extend([
            {
                "if": {
                    "filter_query": '{Signal Direction} = "BUY VOL"',
                    "column_id": "Signal Direction",
                },
                "color": COLORS["accent_cyan"],
                "fontWeight": "bold",
            },
            {
                "if": {
                    "filter_query": '{Signal Direction} = "SELL VOL"',
                    "column_id": "Signal Direction",
                },
                "color": COLORS["accent_red"],
                "fontWeight": "bold",
            },
            {
                "if": {
                    "filter_query": '{Signal Direction} = "BUY SKEW"',
                    "column_id": "Signal Direction",
                },
                "color": COLORS["accent_blue"],
                "fontWeight": "bold",
            },
            {
                "if": {
                    "filter_query": '{Signal Direction} = "SELL SKEW"',
                    "column_id": "Signal Direction",
                },
                "color": COLORS["accent_orange"],
                "fontWeight": "bold",
            },
        ])

        # Confidence colouring
        style_data_conditional.extend([
            {
                "if": {
                    "filter_query": '{Confidence} = "HIGH"',
                    "column_id": "Confidence",
                },
                "color": COLORS["accent_green"],
                "fontWeight": "bold",
            },
            {
                "if": {
                    "filter_query": '{Confidence} = "MEDIUM"',
                    "column_id": "Confidence",
                },
                "color": COLORS["accent_orange"],
            },
            {
                "if": {
                    "filter_query": '{Confidence} = "LOW"',
                    "column_id": "Confidence",
                },
                "color": COLORS["text_muted"],
            },
        ])

        # Striped rows
        style_data_conditional.append({
            "if": {"row_index": "odd"},
            "backgroundColor": COLORS["bg_secondary"],
        })

        table = dash_table.DataTable(
            id="rv-signal-datatable",
            columns=[{"name": c, "id": c} for c in df.columns],
            data=df.to_dict("records"),
            sort_action="native",
            sort_mode="single",
            page_size=30,
            style_table={
                "overflowX": "auto",
                "maxHeight": "450px",
                "overflowY": "auto",
                "border": f"1px solid {COLORS['border_subtle']}",
                "borderRadius": "8px",
            },
            style_header={
                **TABLE_HEADER_STYLE,
                "textAlign": "center",
                "whiteSpace": "normal",
            },
            style_cell={
                **TABLE_CELL_STYLE,
                "textAlign": "center",
                "minWidth": "90px",
                "maxWidth": "140px",
            },
            style_data_conditional=style_data_conditional,
            style_as_list_view=True,
        )

        return table

    except Exception as e:
        return html.Div(
            f"Error building signal table: {str(e)}",
            style={"color": COLORS["accent_red"], "padding": "20px"},
        )
