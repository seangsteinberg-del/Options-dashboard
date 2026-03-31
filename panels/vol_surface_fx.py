"""
FX Vol Analytics Workstation
=============================
Flagship panel for FX options vol surface analysis.  Provides 14 surface
chart types **plus** Chart-Lab-style configurable time-series / study
charts in a 2×2 grid with 10 KPI stat boxes, model selection
(Market / SABR / Vanna-Volga), view presets, comparison modes, a deep
analytical-studies section, and auto-refresh.

Exports: layout(), register_callbacks(app)
"""

import logging
import traceback

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE, AXIS_DEFAULTS,
    make_stat_style, TAB_STYLE, LABEL_STYLE, DROPDOWN_STYLE, INPUT_STYLE,
    clickable_stat, section_header,
    GAP, SECTION_GAP, CHART_SM, CHART_MD, CHART_LG,
    CSV_BTN_STYLE, no_data_fig,
)
from core.csv_export import export_csv
from core.bloomberg_fx import (
    get_fx_vol_surface, get_fx_spots, get_fx_rates,
    get_fx_historical_vol, get_fx_realized_vol, get_all_pairs,
    get_fx_historical_spot, get_fx_term_structure,
)
from core.fx_analytics import (
    vol_percentile, vol_zscore, vol_regime_detect, vol_cone,
    iv_rv_spread, forward_vol_curve, smile_skewness, smile_kurtosis,
    vol_percentile_surface, vol_zscore_surface, vol_change,
    vol_surface_diff, forward_vol_surface, smile_implied_pdf,
    iv_rv_percentile,
    vol_regime_history, tail_probabilities, breakeven_vol,
    carry_per_vol, carry_momentum, rate_differential_history,
    spot_correlation_matrix, vol_correlation_matrix,
    rv_scanner,
)
from core.fx_conventions import (
    tenor_to_years, bf_rr_to_smile, build_smile_spline,
    FX_PAIR_REGISTRY,
)
from core.vanna_volga import vv_smile, sabr_vol, vv_vs_sabr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TENORS_LIST = ["ON", "1W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "5Y"]
DELTA_LABELS = ["10P", "25P", "ATM", "25C", "10C"]
# Monotonic smile-position values for the 3D surface X-axis.
# Ordered by strike: 10P (deep OTM put) → ATM → 10C (deep OTM call).
# NOT actual Black-Scholes delta (which is non-monotonic in smile order).
DELTA_NUMERIC = [-0.50, -0.25, 0.0, 0.25, 0.50]

CHART_OPTIONS = [
    {"label": "── SURFACE ──────", "value": "_sfc_header", "disabled": True},
    {"label": "3D Vol Surface", "value": "surface_3d"},
    {"label": "Vol Heatmap", "value": "heatmap"},
    {"label": "ATM Term Structure", "value": "atm_term"},
    {"label": "Skew Profile (25D RR)", "value": "skew_rr"},
    {"label": "Smile Curvature (25D BF)", "value": "smile_bf"},
    {"label": "Rich/Cheap Heatmap", "value": "rich_cheap"},
    {"label": "Smile Curve", "value": "smile_curve"},
    {"label": "Vol Time Series", "value": "vol_ts"},
    {"label": "IV vs RV", "value": "iv_rv"},
    {"label": "Vol Cone", "value": "vol_cone_chart"},
    {"label": "Forward Vol", "value": "fwd_vol"},
    {"label": "Surface Change", "value": "surface_change"},
    {"label": "SABR Parameters", "value": "sabr_params"},
    {"label": "Implied Distribution", "value": "implied_dist"},
    {"label": "── LAB: TIME SERIES ─", "value": "_ts_header", "disabled": True},
    {"label": "TS: ATM Vol",       "value": "lab_atm"},
    {"label": "TS: 25D RR",        "value": "lab_25d_rr"},
    {"label": "TS: 25D BF",        "value": "lab_25d_bf"},
    {"label": "TS: Spot",          "value": "lab_spot"},
    {"label": "TS: IV-RV Spread",  "value": "lab_iv_rv"},
    {"label": "TS: Realized Vol",  "value": "lab_rv"},
    {"label": "TS: Forward Vol",   "value": "lab_fwd_vol"},
    {"label": "TS: Term Spread",   "value": "lab_term_spread"},
    {"label": "TS: Carry (bps)",   "value": "lab_carry"},
    {"label": "── LAB: STUDIES ────", "value": "_st_header", "disabled": True},
    {"label": "Vol Cone Study",         "value": "lab_study_vol_cone"},
    {"label": "Vol Smile Study",        "value": "lab_study_smile"},
    {"label": "Implied PDF Study",      "value": "lab_study_implied_pdf"},
    {"label": "Vol Regime Study",       "value": "lab_study_vol_regime"},
    {"label": "Forward Vol Curve Study", "value": "lab_study_fwd_vol_curve"},
    {"label": "Percentile Surface",     "value": "lab_study_pctile_surface"},
    {"label": "Z-Score Surface",        "value": "lab_study_zscore_surface"},
    {"label": "Tail Probabilities",     "value": "lab_study_tail_probs"},
    {"label": "Breakeven Vol",          "value": "lab_study_breakeven"},
    {"label": "Carry Landscape",        "value": "lab_study_carry_landscape"},
]

# Map lab metric keys → bloomberg_fx fetch params
_LAB_METRIC_MAP = {
    "lab_atm":          ("ATM",    "vol"),
    "lab_25d_rr":       ("25D_RR", "vol"),
    "lab_25d_bf":       ("25D_BF", "vol"),
    "lab_spot":         ("SPOT",   "spot"),
    "lab_iv_rv":        ("IV_RV",  "derived"),
    "lab_rv":           ("RV",     "derived"),
    "lab_fwd_vol":      ("FWD_VOL","derived"),
    "lab_term_spread":  ("TERM_SPREAD", "derived"),
    "lab_carry":        ("CARRY",  "derived"),
}

# Overlay options for Lab time-series charts
LAB_OVERLAY_OPTIONS = [
    {"label": "— None —", "value": ""},
    {"label": "ATM Vol",      "value": "lab_atm"},
    {"label": "25D RR",       "value": "lab_25d_rr"},
    {"label": "25D BF",       "value": "lab_25d_bf"},
    {"label": "Spot",         "value": "lab_spot"},
    {"label": "IV-RV Spread", "value": "lab_iv_rv"},
    {"label": "Realized Vol", "value": "lab_rv"},
    {"label": "Forward Vol",  "value": "lab_fwd_vol"},
    {"label": "Term Spread",  "value": "lab_term_spread"},
    {"label": "Carry",        "value": "lab_carry"},
]

LAB_WINDOW_OPTIONS = [
    {"label": "30d",  "value": 30},
    {"label": "60d",  "value": 60},
    {"label": "120d", "value": 120},
    {"label": "252d", "value": 252},
    {"label": "504d", "value": 504},
]

LAB_NORMALIZE_OPTIONS = [
    {"label": "Raw",     "value": "raw"},
    {"label": "Indexed", "value": "indexed"},
    {"label": "Z-Score", "value": "zscore"},
    {"label": "% Chg",   "value": "pct_change"},
]

VIEW_PRESETS = {
    "Trader":     ["surface_3d", "atm_term", "skew_rr", "iv_rv"],
    "Skew":       ["smile_curve", "skew_rr", "smile_bf", "rich_cheap"],
    "Term":       ["atm_term", "fwd_vol", "heatmap", "vol_cone_chart"],
    "Rich-Cheap": ["rich_cheap", "surface_change", "vol_ts", "implied_dist"],
    "Lab: Vol Monitor": ["lab_atm", "lab_25d_rr", "lab_iv_rv", "lab_study_vol_regime"],
    "Lab: Smile":       ["lab_study_smile", "lab_study_implied_pdf", "lab_study_tail_probs", "lab_25d_rr"],
    "Lab: RV":          ["lab_study_vol_cone", "lab_iv_rv", "lab_study_breakeven", "lab_rv"],
    "Lab: Carry":       ["lab_carry", "lab_study_carry_landscape", "lab_term_spread", "lab_fwd_vol"],
}

_PAIR_GROUPS = {
    "G10 Major": ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"],
    "G10 Cross": ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "EURAUD",
                  "EURNZD", "NZDJPY", "AUDNZD", "CADCHF", "CADJPY"],
    "Scandie":   ["EURNOK", "EURSEK", "USDSEK", "USDNOK"],
    "EM":        ["USDMXN", "USDBRL", "USDTRY", "USDZAR", "USDCNH", "USDINR",
                  "USDSGD", "USDKRW"],
}

HISTORY_OFFSETS = [
    {"label": "1D ago", "value": 1},
    {"label": "1W ago", "value": 5},
    {"label": "1M ago", "value": 22},
    {"label": "3M ago", "value": 66},
]

SIDEBAR_STYLE = {
    "width": "250px",
    "minWidth": "250px",
    "backgroundColor": COLORS["bg_secondary"],
    "borderRight": f"1px solid {COLORS['border']}",
    "padding": "16px 14px",
    "overflowY": "auto",
    "height": "100%",
}

SIDEBAR_SECTION = {
    "marginBottom": "14px",
}

SIDEBAR_LABEL = {
    **LABEL_STYLE,
    "fontSize": "9px",
    "marginBottom": "4px",
}

PRESET_BTN = {
    "backgroundColor": COLORS["bg_input"],
    "color": COLORS["text_secondary"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "0px",
    "padding": "5px 10px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "10px",
    "fontWeight": "600",
    "cursor": "pointer",
    "letterSpacing": "0.8px",
    "marginRight": "4px",
    "marginBottom": "4px",
}


# ═══════════════════════════════════════════════════════════════════════════
# Helper: extract vol surface data into matrices
# ═══════════════════════════════════════════════════════════════════════════

def _get_surface_data(pair):
    """Fetch vol surface and decompose into tenor x delta grid."""
    surface = get_fx_vol_surface(pair) or {}
    tenors_avail = []
    atm_vals, rr25_vals, bf25_vals, rr10_vals, bf10_vals = [], [], [], [], []

    for t in TENORS_LIST:
        if t in surface:
            row = surface[t]
            tenors_avail.append(t)
            atm_vals.append(row.get("atm", row.get("ATM", 0)))
            rr25_vals.append(row.get("rr25", row.get("25D_RR", 0)))
            bf25_vals.append(row.get("bf25", row.get("25D_BF", 0)))
            rr10_vals.append(row.get("rr10", row.get("10D_RR", 0)))
            bf10_vals.append(row.get("bf10", row.get("10D_BF", 0)))

    if not tenors_avail:
        logging.getLogger(__name__).warning("Vol surface empty for %s", pair)
        return None

    n = len(tenors_avail)
    vol_grid = np.zeros((n, 5))
    for i in range(n):
        smile = bf_rr_to_smile(atm_vals[i], rr25_vals[i], bf25_vals[i],
                               rr10_vals[i], bf10_vals[i])
        vol_grid[i] = [smile.get("p10", smile["p25"]),
                       smile["p25"], smile["atm"],
                       smile["c25"], smile.get("c10", smile["c25"])]

    T_years = np.array([tenor_to_years(t) for t in tenors_avail])
    return {
        "tenors": tenors_avail,
        "T_years": T_years,
        "atm": np.array(atm_vals),
        "rr25": np.array(rr25_vals),
        "bf25": np.array(bf25_vals),
        "rr10": np.array(rr10_vals),
        "bf10": np.array(bf10_vals),
        "vol_grid": vol_grid,
        "delta_labels": DELTA_LABELS,
        "delta_numeric": np.array(DELTA_NUMERIC),
        "surface_raw": surface,
    }


def _filter_delta_range(sd, delta_range):
    """Filter a surface data dict to show only selected delta columns.

    When delta_range == "25-50", keep only 25P, ATM, 25C (indices 1,2,3).
    When delta_range == "10-50" (default), keep all five columns.
    """
    if delta_range != "25-50":
        return sd  # "10-50" -> show all deltas
    # Column indices: 0=10P, 1=25P, 2=ATM, 3=25C, 4=10C
    keep_cols = [1, 2, 3]
    full_labels = sd["delta_labels"]  # list of strings
    full_numeric = sd["delta_numeric"]  # numpy array
    return {
        **sd,
        "vol_grid": sd["vol_grid"][:, keep_cols],
        "delta_labels": [full_labels[i] for i in keep_cols],
        "delta_numeric": full_numeric[keep_cols],
    }


def _filter_surface_data(sd, selected_tenors):
    """Filter a surface data dict to keep only the selected tenors."""
    if not selected_tenors:
        return sd  # nothing selected -> keep all
    keep_idx = [i for i, t in enumerate(sd["tenors"]) if t in selected_tenors]
    if not keep_idx:
        return sd  # none match -> keep all to avoid empty charts
    return {
        "tenors": [sd["tenors"][i] for i in keep_idx],
        "T_years": sd["T_years"][keep_idx],
        "atm": sd["atm"][keep_idx],
        "rr25": sd["rr25"][keep_idx],
        "bf25": sd["bf25"][keep_idx],
        "rr10": sd["rr10"][keep_idx],
        "bf10": sd["bf10"][keep_idx],
        "vol_grid": sd["vol_grid"][keep_idx],
        "delta_labels": sd["delta_labels"],
        "delta_numeric": sd["delta_numeric"],
        "surface_raw": sd["surface_raw"],
    }


def _get_spot_and_rates(pair):
    """Fetch spot mid, forward 1M, and interest rates."""
    spots = get_fx_spots([pair]) or {}
    spot_info = spots.get(pair, {})
    spot = spot_info.get("mid", spot_info.get("price", 1.0))
    rates = get_fx_rates(pair) or {}
    r_dom = rates.get("r_dom", 0.03)
    r_for = rates.get("r_for", 0.02)
    fwd_1m = spot * np.exp((r_dom - r_for) * tenor_to_years("1M"))
    return spot, fwd_1m, r_dom, r_for


def _apply_chart_template(fig, title=""):
    """Apply the standard theme template to a figure."""
    tpl = CHART_TEMPLATE["layout"]
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text_primary"], size=13,
                   family="'JetBrains Mono', monospace")),
        paper_bgcolor=tpl["paper_bgcolor"],
        plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"],
        margin=dict(l=50, r=20, t=40, b=40),
        hoverlabel=tpl["hoverlabel"],
        legend=dict(font=dict(color=COLORS["text_secondary"], size=10),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=AXIS_DEFAULTS,
        yaxis=AXIS_DEFAULTS,
    )
    return fig


def _empty_fig(msg="No data"):
    """Return a blank figure with a message."""
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color=COLORS["text_muted"], size=14))
    _apply_chart_template(fig)
    return fig


logger = logging.getLogger(__name__)


def _safe_chart(fn):
    """Decorator: wrap every chart function in try/except returning a clean
    empty figure with the error message instead of crashing the panel."""
    def wrapper(pair, sd, spot, r_dom, r_for, **kw):
        try:
            return fn(pair, sd, spot, r_dom, r_for, **kw)
        except Exception as exc:
            logger.exception("Chart %s failed for %s", fn.__name__, pair)
            return _empty_fig(f"{fn.__name__}: {exc}")
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _ordinal(n):
    """Return an integer as an ordinal string: 1 -> '1st', 23 -> '23rd'."""
    n = int(n)
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _is_jpy_pair(pair):
    """Return True if the pair is quoted against JPY (2-decimal spot)."""
    return pair.endswith("JPY") or pair.startswith("JPY")


def _fmt_spot(spot, pair):
    """Format spot with correct decimal places (2 for JPY, 4 otherwise)."""
    if _is_jpy_pair(pair):
        return f"{spot:.2f}"
    return f"{spot:.4f}"


def _fmt_vol(v):
    """Format vol with 2 decimal places + 'v' suffix (e.g., '8.52v')."""
    return f"{v:.2f}v"


def _fmt_pctile(p):
    """Format percentile with ordinal suffix (e.g., '23rd', '45th')."""
    return _ordinal(int(round(p)))


# ═══════════════════════════════════════════════════════════════════════════
# 14 Chart Functions  (each wrapped with @_safe_chart for error resilience)
# ═══════════════════════════════════════════════════════════════════════════

@_safe_chart
def chart_surface_3d(pair, sd, spot, r_dom, r_for, **kw):
    """1. 3D Vol Surface -- go.Surface in delta-space."""
    fig = go.Figure()
    delta_pos = sd["delta_numeric"]  # monotonic smile-position values
    delta_labels = sd["delta_labels"]
    # Build custom hover text with proper delta labels instead of numeric values
    n_tenors = sd["vol_grid"].shape[0]
    n_deltas = len(delta_labels)
    hover_text = [[None] * n_deltas for _ in range(n_tenors)]
    for ti in range(n_tenors):
        for di in range(n_deltas):
            hover_text[ti][di] = (
                f"Delta: {delta_labels[di]}<br>"
                f"Tenor: {sd['tenors'][ti]}<br>"
                f"Vol: {sd['vol_grid'][ti, di]:.2f}%"
            )
    fig.add_trace(go.Surface(
        x=delta_pos * 100,
        y=sd["T_years"],
        z=sd["vol_grid"],
        colorscale=[[0, "#0e0e0e"], [0.25, "#1a1a2e"], [0.5, "#bf5b00"],
                    [0.75, "#ff8800"], [1.0, "#ffbb55"]],
        opacity=0.92,
        colorbar=dict(
            title=dict(text="Vol %", font=dict(color=COLORS["text_muted"], size=10)),
            tickfont=dict(color=COLORS["text_muted"], size=9),
            len=0.6, thickness=12, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.3, roughness=0.5),
    ))
    # Map numeric tick positions to delta labels on the X-axis
    tick_vals = (delta_pos * 100).tolist()
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Delta", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor=COLORS["border_subtle"], color=COLORS["text_muted"],
                       tickvals=tick_vals, ticktext=delta_labels),
            yaxis=dict(title="Tenor (yrs)", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor=COLORS["border_subtle"], color=COLORS["text_muted"]),
            zaxis=dict(title="Vol (%)", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor=COLORS["border_subtle"], color=COLORS["text_muted"]),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.85)),
        ),
    )
    _apply_chart_template(fig, f"3D Vol Surface -- {pair}")
    return fig


@_safe_chart
def chart_heatmap(pair, sd, spot, r_dom, r_for, **kw):
    """2. Vol Heatmap -- tenor x delta grid with annotations."""
    fig = go.Figure()
    text_vals = [[f"{v:.2f}" if np.isfinite(v) else "" for v in row] for row in sd["vol_grid"]]
    fig.add_trace(go.Heatmap(
        x=sd["delta_labels"],
        y=sd["tenors"],
        z=sd["vol_grid"],
        colorscale=[[0, "#0e0e0e"], [0.25, "#1a1a2e"], [0.5, "#bf5b00"],
                    [0.75, "#ff8800"], [1.0, "#ffbb55"]],
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=10, color="#c0c0c0"),
        hovertemplate="Delta: %{x}<br>Tenor: %{y}<br>Vol: %{z:.2f}%<extra></extra>",
        colorbar=dict(
            title=dict(text="Vol %", font=dict(color=COLORS["text_muted"], size=10)),
            tickfont=dict(color=COLORS["text_muted"], size=9),
            len=0.8, thickness=12, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
        xgap=2, ygap=2,
    ))
    _apply_chart_template(fig, f"Vol Heatmap -- {pair}")
    fig.update_layout(
        xaxis=dict(title="Delta", type="category"),
        yaxis=dict(title="Tenor", type="category"),
    )
    return fig


@_safe_chart
def chart_atm_term(pair, sd, spot, r_dom, r_for, **kw):
    """3. ATM Term Structure with history overlays and forward vol."""
    fig = go.Figure()
    tenors = sd["tenors"]
    T_years = sd["T_years"]

    # Current ATM line (bold cyan)
    fig.add_trace(go.Scatter(
        x=tenors, y=sd["atm"], mode="lines+markers",
        name="ATM (current)", line=dict(color=COLORS["accent_cyan"], width=3),
        marker=dict(size=6, color=COLORS["accent_cyan"]),
        hovertemplate="%{x}: %{y:.2f}%<extra>ATM</extra>",
    ))

    # 1W ago overlay (dashed)
    try:
        hist_1w = []
        for t in tenors:
            ch = vol_change(pair, t, "ATM", days_ago=5)
            hist_1w.append(ch.get("previous") if ch else None)
        fig.add_trace(go.Scatter(
            x=tenors, y=hist_1w, mode="lines",
            name="1W ago", line=dict(color=COLORS["accent_blue"], width=1.5, dash="dash"),
            hovertemplate="%{x}: %{y:.2f}%<extra>1W ago</extra>",
        ))
    except Exception:
        pass

    # 1M ago overlay (dotted)
    try:
        hist_1m = []
        for t in tenors:
            ch = vol_change(pair, t, "ATM", days_ago=22)
            hist_1m.append(ch.get("previous") if ch else None)
        fig.add_trace(go.Scatter(
            x=tenors, y=hist_1m, mode="lines",
            name="1M ago", line=dict(color=COLORS["accent_purple"], width=1.5, dash="dot"),
            hovertemplate="%{x}: %{y:.2f}%<extra>1M ago</extra>",
        ))
    except Exception:
        pass

    # Forward vol curve on secondary y-axis
    try:
        fwd_df = forward_vol_curve(pair, start_tenor="1M")
        if not fwd_df.empty:
            fig.add_trace(go.Scatter(
                x=fwd_df["end_tenor"], y=fwd_df["forward_vol"],
                mode="lines+markers", name="Forward Vol",
                line=dict(color=COLORS["accent_orange"], width=2, dash="dashdot"),
                marker=dict(size=5, symbol="diamond"),
                yaxis="y2",
            ))
            fig.update_layout(
                yaxis2=dict(
                    title="Forward Vol (%)", overlaying="y", side="right",
                    gridcolor="#1a1a30",
                    tickfont=dict(size=10, color=COLORS["accent_orange"]),
                    title_font=dict(color=COLORS["accent_orange"], size=11),
                ),
            )
    except Exception:
        pass

    _apply_chart_template(fig, f"ATM Term Structure -- {pair}")
    fig.update_layout(
        xaxis=dict(title="Tenor", type="category"),
        yaxis=dict(title="ATM Vol (%)"),
    )
    return fig


@_safe_chart
def chart_skew_rr(pair, sd, spot, r_dom, r_for, **kw):
    """4. Skew Profile (25D RR) -- bar chart with percentile coloring."""
    tenors = sd["tenors"]
    rr_vals = sd["rr25"]

    # Get percentiles for color intensity
    pctiles = []
    for t in tenors:
        p = vol_percentile(pair, t, "25D_RR")
        pctiles.append(p.get("percentile", 50.0) if isinstance(p, dict) else 50.0)

    # Color by percentile: deep blue (low) to red (high)
    colors = []
    for pct in pctiles:
        r = int(min(255, pct * 2.55))
        b = int(min(255, (100 - pct) * 2.55))
        colors.append(f"rgb({r},60,{b})")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tenors, y=rr_vals, name="25D RR",
        marker=dict(color=colors, line=dict(width=1, color=COLORS["border"])),
        text=[f"{v:+.2f}" if np.isfinite(v) else "" for v in rr_vals],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_secondary"]),
        hovertemplate="Tenor: %{x}<br>25D RR: %{y:.2f}<br>%ile: %{customdata:.0f}<extra></extra>",
        customdata=pctiles,
    ))

    # 1Y range whiskers
    for i, t in enumerate(tenors):
        p = vol_percentile(pair, t, "25D_RR")
        if isinstance(p, dict):
            p_min = p.get("min", rr_vals[i] if i < len(rr_vals) else 0)
            p_max = p.get("max", rr_vals[i] if i < len(rr_vals) else 0)
            fig.add_shape(type="line",
                x0=i, x1=i, y0=p_min, y1=p_max,
                line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
                xref="x", yref="y")

    fig.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    _apply_chart_template(fig, f"25D Risk Reversal -- {pair}")
    fig.update_layout(xaxis=dict(title="Tenor", type="category"),
                      yaxis=dict(title="25D RR (vol pts)"))
    return fig


@_safe_chart
def chart_smile_bf(pair, sd, spot, r_dom, r_for, **kw):
    """5. Smile Curvature (25D BF) -- bar chart with percentile coloring."""
    tenors = sd["tenors"]
    bf_vals = sd["bf25"]

    pctiles = []
    for t in tenors:
        p = vol_percentile(pair, t, "25D_BF")
        pctiles.append(p.get("percentile", 50.0) if isinstance(p, dict) else 50.0)

    colors = []
    for pct in pctiles:
        r = int(min(255, pct * 2.55))
        b = int(min(255, (100 - pct) * 2.55))
        colors.append(f"rgb({r},60,{b})")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tenors, y=bf_vals, name="25D BF",
        marker=dict(color=colors, line=dict(width=1, color=COLORS["border"])),
        text=[f"{v:.2f}" if np.isfinite(v) else "" for v in bf_vals],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_secondary"]),
        hovertemplate="Tenor: %{x}<br>25D BF: %{y:.2f}<br>%ile: %{customdata:.0f}<extra></extra>",
        customdata=pctiles,
    ))

    for i, t in enumerate(tenors):
        p = vol_percentile(pair, t, "25D_BF")
        if isinstance(p, dict):
            p_min = p.get("min", bf_vals[i] if i < len(bf_vals) else 0)
            p_max = p.get("max", bf_vals[i] if i < len(bf_vals) else 0)
            fig.add_shape(type="line",
                x0=i, x1=i, y0=p_min, y1=p_max,
                line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
                xref="x", yref="y")

    _apply_chart_template(fig, f"25D Butterfly -- {pair}")
    fig.update_layout(xaxis=dict(title="Tenor", type="category"),
                      yaxis=dict(title="25D BF (vol pts)"))
    return fig


@_safe_chart
def chart_rich_cheap(pair, sd, spot, r_dom, r_for, **kw):
    """6. Rich/Cheap Heatmap -- tenor x metric, color by percentile."""
    tenors = sd["tenors"]
    metrics = ["ATM", "25D_RR", "25D_BF"]
    metric_labels = ["ATM", "25D RR", "25D BF"]

    z_data = []
    text_data = []
    for t in tenors:
        row_z = []
        row_txt = []
        for m in metrics:
            p = vol_percentile(pair, t, m)
            if p is not None:
                pct_val = p.get("percentile", 50.0)
                row_z.append(pct_val)
                row_txt.append(f"{pct_val:.0f}%ile")
            else:
                row_z.append(50.0)
                row_txt.append("--")
        z_data.append(row_z)
        text_data.append(row_txt)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=metric_labels,
        y=tenors,
        z=z_data,
        colorscale=[[0, "#1565c0"], [0.15, "#0d47a1"], [0.30, "#0a1628"],
                    [0.50, "#0e0e0e"], [0.70, "#2a1200"], [0.85, "#bf5b00"],
                    [1.0, "#ff8800"]],
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#c0c0c0"),
        hovertemplate="Metric: %{x}<br>Tenor: %{y}<br>Percentile: %{z:.0f}<extra></extra>",
        colorbar=dict(
            title=dict(text="%ile", font=dict(color=COLORS["text_muted"], size=10)),
            tickfont=dict(color=COLORS["text_muted"], size=9),
            len=0.8, thickness=12, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
            tickvals=[0, 25, 50, 75, 100],
        ),
        zmin=0, zmax=100,
        xgap=2, ygap=2,
    ))
    _apply_chart_template(fig, f"Rich/Cheap -- {pair}")
    fig.update_layout(
        xaxis=dict(title="Metric", type="category"),
        yaxis=dict(title="Tenor", type="category"),
    )
    return fig


@_safe_chart
def chart_smile_curve(pair, sd, spot, r_dom, r_for, **kw):
    """7. Smile Curve for selected tenor: vol vs delta with spline fit."""
    sel_tenor = kw.get("smile_tenor", "3M")
    surface = sd["surface_raw"]

    # Find the closest available tenor
    avail = sd["tenors"]
    tenor_use = sel_tenor if sel_tenor in avail else (avail[len(avail) // 2] if avail else "1M")
    row = surface.get(tenor_use, {})
    atm = row.get("atm", row.get("ATM", 0))
    rr25 = row.get("rr25", row.get("25D_RR", 0.0))
    bf25 = row.get("bf25", row.get("25D_BF", 0.0))
    rr10 = row.get("rr10", row.get("10D_RR", 0.0))
    bf10 = row.get("bf10", row.get("10D_BF", 0.0))

    smile = bf_rr_to_smile(atm, rr25, bf25, rr10, bf10)
    pillar_deltas = [-10, -25, 0, 25, 10]
    pillar_labels = ["10P", "25P", "ATM", "25C", "10C"]
    pillar_vols = [smile.get("p10", smile["p25"]), smile["p25"],
                   smile["atm"], smile["c25"], smile.get("c10", smile["c25"])]

    # Build spline for smooth curve
    try:
        T = tenor_to_years(tenor_use)
        spline = build_smile_spline(atm / 100, smile["c25"] / 100, smile["p25"] / 100,
                                    smile.get("c10", smile["c25"]) / 100,
                                    smile.get("p10", smile["p25"]) / 100,
                                    spot, T, r_dom, r_for)
        delta_fine = np.linspace(-0.25, 0.25, 100)
        vol_fine = spline(delta_fine) * 100
    except Exception:
        delta_fine = np.array([-0.25, -0.10, 0.0, 0.10, 0.25])
        vol_fine = np.array([pillar_vols[1], pillar_vols[0], pillar_vols[2],
                             pillar_vols[4], pillar_vols[3]])

    fig = go.Figure()

    # Spline fit line
    fig.add_trace(go.Scatter(
        x=delta_fine * 100 if delta_fine.max() <= 1.0 else delta_fine,
        y=vol_fine,
        mode="lines", name="Spline Fit",
        line=dict(color=COLORS["accent_cyan"], width=2.5),
        hovertemplate="Delta: %{x:.1f}<br>Vol: %{y:.2f}%<extra>Spline</extra>",
    ))

    # Market points
    fig.add_trace(go.Scatter(
        x=pillar_deltas, y=pillar_vols,
        mode="markers+text", name="Market",
        marker=dict(size=10, color=COLORS["accent_rose"],
                    line=dict(width=2, color=COLORS["text_primary"])),
        text=pillar_labels,
        textposition="top center",
        textfont=dict(size=9, color=COLORS["text_secondary"]),
        hovertemplate="%{text}: %{y:.2f}%<extra>Market</extra>",
    ))

    # ── Model overlays (SABR / Vanna-Volga) ──
    model_sel = kw.get("model", "market")

    if model_sel == "sabr":
        try:
            T = tenor_to_years(tenor_use)
            F = spot * np.exp((r_dom - r_for) * T)
            beta = 0.5
            atm_dec = atm / 100.0
            rr25_dec = rr25 / 100.0
            bf25_dec = bf25 / 100.0
            alpha_est = atm_dec * F ** (1 - beta)
            rho_est = float(np.clip(rr25_dec / max(atm_dec, 0.001) * (-0.8), -0.95, 0.95))
            nu_est = float(np.clip(bf25_dec / max(atm_dec, 0.001) * 3.0 + 0.3, 0.05, 3.0))

            delta_sabr = np.linspace(-0.25, 0.25, 80)
            sabr_strikes = np.array([
                F * np.exp(-d * atm_dec * np.sqrt(T)) for d in delta_sabr
            ])
            sabr_vols = np.array([
                sabr_vol(F, K, T, alpha_est, beta, rho_est, nu_est) * 100.0
                for K in sabr_strikes
            ])
            fig.add_trace(go.Scatter(
                x=delta_sabr * 100, y=sabr_vols,
                mode="lines", name="SABR Fit",
                line=dict(color=COLORS["accent_orange"], width=2, dash="dash"),
                hovertemplate="Delta: %{x:.0f}<br>SABR Vol: %{y:.2f}%<extra>SABR</extra>",
            ))
        except Exception:
            logging.getLogger(__name__).debug("SABR fit overlay failed for %s", pair)

    elif model_sel == "vv":
        try:
            T = tenor_to_years(tenor_use)
            vv_result = vv_smile(
                spot, T, r_dom, r_for,
                atm / 100.0, smile["p25"] / 100.0, smile["c25"] / 100.0,
                n_strikes=80,
            )
            vv_deltas = (vv_result["deltas"] - 0.5) * 100  # centre around 0
            vv_vols = vv_result["vols"] * 100.0
            fig.add_trace(go.Scatter(
                x=vv_deltas, y=vv_vols,
                mode="lines", name="Vanna-Volga",
                line=dict(color=COLORS["accent_purple"], width=2, dash="dashdot"),
                hovertemplate="Delta: %{x:.0f}<br>VV Vol: %{y:.2f}%<extra>Vanna-Volga</extra>",
            ))
        except Exception:
            logging.getLogger(__name__).debug("VV fit overlay failed for %s", pair)

    _apply_chart_template(fig, f"Smile -- {pair} {tenor_use}")
    fig.update_layout(
        xaxis=dict(title="Delta"),
        yaxis=dict(title="Vol (%)"),
    )
    return fig


@_safe_chart
def chart_vol_ts(pair, sd, spot, r_dom, r_for, **kw):
    """8. Vol Time Series with Bollinger bands."""
    sel_tenor = kw.get("ts_tenor", "3M")
    try:
        hist = get_fx_historical_vol(pair, sel_tenor, "ATM", 252)
        if isinstance(hist, dict):
            hist = list(hist.values()) if hist else None
        if hist is None or len(hist) < 20:
            raise ValueError("insufficient data")
    except Exception:
        return no_data_fig(height=CHART_MD, msg="NO VOL HISTORY")

    # Use actual dates from Bloomberg index if available, else sequential days
    if hasattr(hist, 'index') and hasattr(hist.index, 'date'):
        days = hist.index
    else:
        days = np.arange(len(hist))
    series = pd.Series(hist)
    ma20 = series.rolling(20).mean()
    std20 = series.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    fig = go.Figure()

    # Bollinger band fill
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([upper.values, lower.values[::-1]]),
        fill="toself", fillcolor="rgba(255,136,0,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Upper/lower bands
    fig.add_trace(go.Scatter(x=days, y=upper, mode="lines", name="BB Upper",
        line=dict(color=COLORS["accent_cyan"], width=1, dash="dot"), showlegend=False,
        hovertemplate="Day %{x}: %{y:.2f}%<extra>BB Upper</extra>"))
    fig.add_trace(go.Scatter(x=days, y=lower, mode="lines", name="BB Lower",
        line=dict(color=COLORS["accent_cyan"], width=1, dash="dot"), showlegend=False,
        hovertemplate="Day %{x}: %{y:.2f}%<extra>BB Lower</extra>"))

    # MA line
    fig.add_trace(go.Scatter(x=days, y=ma20, mode="lines", name="20d MA",
        line=dict(color=COLORS["accent_blue"], width=1.5, dash="dash"),
        hovertemplate="Day %{x}: %{y:.2f}%<extra>20d MA</extra>"))

    # ATM vol line
    fig.add_trace(go.Scatter(x=days, y=hist, mode="lines", name=f"ATM {sel_tenor}",
        line=dict(color=COLORS["accent_cyan"], width=2),
        hovertemplate="Day %{x}: %{y:.2f}%<extra>ATM</extra>"))

    # Current level marker
    if len(days) == 0:
        return no_data_fig(height=CHART_MD, msg="NO VOL HISTORY")
    current_vol = float(hist.iloc[-1]) if hasattr(hist, 'iloc') else float(hist[-1])
    fig.add_trace(go.Scatter(
        x=[days[-1]], y=[current_vol], mode="markers",
        marker=dict(color=COLORS["accent_orange"], size=8, symbol="diamond"),
        name=f"Current: {current_vol:.2f}%", showlegend=True,
        hovertemplate=f"Current: {current_vol:.2f}%<extra></extra>"))

    _apply_chart_template(fig, f"ATM Vol Time Series -- {pair} {sel_tenor}")
    fig.update_layout(xaxis=dict(title="Date"), yaxis=dict(title="Vol (%)"))
    return fig


@_safe_chart
def chart_iv_rv(pair, sd, spot, r_dom, r_for, **kw):
    """9. IV vs RV with filled spread area.

    iv_rv_spread() returns a DataFrame with columns: day, iv, rv, spread, spread_pct.
    Guard against dict return (edge cases) and missing/renamed columns.
    """
    result = iv_rv_spread(pair, tenor="3M", rv_window=20, lookback=252)

    # Handle dict return format: convert to DataFrame
    if isinstance(result, dict):
        if "iv" in result and "rv" in result:
            iv_arr = np.asarray(result["iv"])
            rv_arr = np.asarray(result["rv"])
            n = min(len(iv_arr), len(rv_arr))
            result = pd.DataFrame({
                "day": np.arange(n),
                "iv": iv_arr[-n:],
                "rv": rv_arr[-n:],
                "spread": iv_arr[-n:] - rv_arr[-n:],
            })
        else:
            return _empty_fig("IV vs RV: unexpected dict format")

    if not isinstance(result, pd.DataFrame) or result.empty:
        return _empty_fig("IV vs RV: insufficient data")

    df = result

    # Resolve column names flexibly (guard against naming drift)
    day_col = "day" if "day" in df.columns else None
    iv_col = next((c for c in df.columns if c.lower() in ("iv", "atm_iv", "impl_vol")), None)
    rv_col = next((c for c in df.columns if c.lower() in ("rv", "realized_vol", "real_vol")), None)

    if iv_col is None or rv_col is None:
        return _empty_fig("IV vs RV: missing iv/rv columns")

    days = df[day_col].values if day_col else np.arange(len(df))
    iv_arr = df[iv_col].values
    rv_arr = df[rv_col].values

    fig = go.Figure()

    # Spread filled area
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([iv_arr, rv_arr[::-1]]),
        fill="toself",
        fillcolor="rgba(255,51,51,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(x=days, y=iv_arr, mode="lines", name="ATM IV 3M",
        line=dict(color=COLORS["accent_cyan"], width=2.5),
        hovertemplate="Day %{x}: %{y:.2f}%<extra>IV</extra>"))
    fig.add_trace(go.Scatter(x=days, y=rv_arr, mode="lines", name="RV 20d",
        line=dict(color=COLORS["accent_rose"], width=2),
        hovertemplate="Day %{x}: %{y:.2f}%<extra>RV</extra>"))

    # Spread bar on secondary y-axis
    spread_arr = iv_arr - rv_arr
    spread_colors = [COLORS["accent_green"] if s > 0 else COLORS["accent_red"]
                     for s in spread_arr]
    fig.add_trace(go.Bar(
        x=days, y=spread_arr, name="IV-RV Spread",
        marker=dict(color=spread_colors, opacity=0.25),
        yaxis="y2",
    ))

    fig.update_layout(
        yaxis2=dict(
            title="Spread (vol pts)", overlaying="y", side="right",
            gridcolor="#1a1a30",
            tickfont=dict(size=9, color=COLORS["text_muted"]),
            title_font=dict(color=COLORS["text_muted"], size=10),
        ),
    )

    # Zero line for spread (on secondary y2 axis where bars are plotted)
    fig.add_shape(
        type="line", x0=0, x1=1, y0=0, y1=0,
        xref="paper", yref="y2",
        line=dict(color=COLORS["border"], width=0.5),
    )

    # Current spread annotation
    if len(spread_arr) == 0 or len(days) == 0:
        return _empty_fig("IV vs RV: insufficient data points")
    current_spread = float(spread_arr[-1])
    spread_label = f"IV-RV: {current_spread:+.1f}v"
    spread_color = COLORS["accent_green"] if current_spread > 0 else COLORS["accent_red"]
    fig.add_annotation(
        x=days[-1], y=float(iv_arr[-1]),
        text=spread_label, showarrow=True, arrowhead=2,
        font=dict(color=spread_color, size=10),
        arrowcolor=spread_color, ax=40, ay=-25)

    _apply_chart_template(fig, f"IV vs Realized Vol -- {pair}")
    fig.update_layout(xaxis=dict(title="Date"), yaxis=dict(title="Vol (%)"))
    return fig


@_safe_chart
def chart_vol_cone_chart(pair, sd, spot, r_dom, r_for, **kw):
    """10. Vol Cone -- percentile fans at multiple windows."""
    try:
        cone_df = vol_cone(pair, windows=[5, 10, 20, 60, 90, 252])
        if cone_df.empty:
            raise ValueError("empty cone")
    except Exception:
        return _empty_fig("Vol cone: insufficient data")

    windows = cone_df["window"].values
    fig = go.Figure()

    # Percentile bands (symmetric fill)
    bands = [
        ("p10", "p90", "rgba(255,136,0,0.04)", "10-90%ile"),
        ("p25", "p75", "rgba(255,136,0,0.08)", "25-75%ile"),
    ]
    for lo, hi, color, name in bands:
        fig.add_trace(go.Scatter(
            x=np.concatenate([windows, windows[::-1]]),
            y=np.concatenate([cone_df[hi].values, cone_df[lo].values[::-1]]),
            fill="toself", fillcolor=color,
            line=dict(width=0), name=name, hoverinfo="skip",
        ))

    # Median line
    fig.add_trace(go.Scatter(x=windows, y=cone_df["median"], mode="lines",
        name="Median", line=dict(color=COLORS["accent_blue"], width=1.5, dash="dash"),
        hovertemplate="%{x}d: %{y:.2f}%<extra>Median</extra>"))

    # Min/Max
    fig.add_trace(go.Scatter(x=windows, y=cone_df["min"], mode="lines",
        name="Min", line=dict(color=COLORS["accent_green"], width=1, dash="dot"),
        hovertemplate="%{x}d: %{y:.2f}%<extra>Min</extra>"))
    fig.add_trace(go.Scatter(x=windows, y=cone_df["max"], mode="lines",
        name="Max", line=dict(color=COLORS["accent_red"], width=1, dash="dot"),
        hovertemplate="%{x}d: %{y:.2f}%<extra>Max</extra>"))

    # Current RV
    fig.add_trace(go.Scatter(x=windows, y=cone_df["current_c2c"], mode="lines+markers",
        name="Current RV", line=dict(color=COLORS["accent_cyan"], width=2.5),
        marker=dict(size=7, color=COLORS["accent_cyan"]),
        hovertemplate="%{x}d: %{y:.2f}%<extra>Current RV</extra>"))

    # ATM IV reference line (where the market is pricing vol)
    atm_3m = sd["atm"][min(4, len(sd["atm"]) - 1)] if len(sd["atm"]) > 0 else None
    if atm_3m and atm_3m > 0:
        fig.add_hline(y=atm_3m,
                      line=dict(color=COLORS["accent_orange"], width=1.5, dash="dashdot"),
                      annotation_text=f"ATM IV: {atm_3m:.1f}%",
                      annotation_font=dict(color=COLORS["accent_orange"], size=9))

    _apply_chart_template(fig, f"Realized Vol Cone -- {pair}")
    fig.update_layout(xaxis=dict(title="Window (days)"), yaxis=dict(title="Vol (%)"))
    return fig


@_safe_chart
def chart_fwd_vol(pair, sd, spot, r_dom, r_for, **kw):
    """11. Forward Vol curve as line chart."""
    try:
        df = forward_vol_curve(pair, start_tenor="1M")
        if df.empty:
            raise ValueError("empty")
    except Exception:
        return _empty_fig("Forward vol: insufficient data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["end_tenor"], y=df["forward_vol"], mode="lines+markers",
        name="Forward Vol",
        line=dict(color=COLORS["accent_orange"], width=2.5),
        marker=dict(size=7, color=COLORS["accent_orange"]),
        hovertemplate="%{x}: %{y:.2f}%<extra>Forward</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["end_tenor"], y=df["spot_vol"], mode="lines+markers",
        name="Spot Vol",
        line=dict(color=COLORS["accent_cyan"], width=2, dash="dash"),
        marker=dict(size=5, color=COLORS["accent_cyan"]),
        hovertemplate="%{x}: %{y:.2f}%<extra>Spot</extra>",
    ))

    _apply_chart_template(fig, f"Forward Vol Curve -- {pair}")
    fig.update_layout(xaxis=dict(title="End Tenor", type="category"),
                      yaxis=dict(title="Vol (%)"))
    return fig


@_safe_chart
def chart_surface_change(pair, sd, spot, r_dom, r_for, **kw):
    """12. Surface Change heatmap -- today vs N days ago."""
    days_ago = kw.get("days_ago", 1)
    try:
        diff_df = vol_surface_diff(pair, days_ago=days_ago)
        if diff_df.empty:
            raise ValueError("empty diff")
    except Exception:
        return _empty_fig("Surface change: insufficient data")

    tenors = diff_df.index.tolist()
    deltas = diff_df.columns.tolist()
    z = diff_df.values

    text_vals = [[f"{v:+.2f}" if np.isfinite(v) else "" for v in row] for row in z]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=deltas,
        y=tenors,
        z=z,
        colorscale=[[0, "#00cc66"], [0.35, "#0a2618"], [0.50, "#0e0e0e"],
                    [0.65, "#2a1200"], [1.0, "#ff3333"]],
        zmid=0,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=10, color="#c0c0c0"),
        hovertemplate="Delta: %{x}<br>Tenor: %{y}<br>Change: %{z:+.2f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Vol Chg", font=dict(color=COLORS["text_muted"], size=10)),
            tickfont=dict(color=COLORS["text_muted"], size=9),
            len=0.8, thickness=12, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
        xgap=2, ygap=2,
    ))
    label = {1: "1D", 5: "1W", 22: "1M", 66: "3M"}.get(days_ago, f"{days_ago}D")
    _apply_chart_template(fig, f"Surface Change ({label}) -- {pair}")
    fig.update_layout(
        xaxis=dict(title="Delta", type="category"),
        yaxis=dict(title="Tenor", type="category"),
    )
    return fig


@_safe_chart
def chart_sabr_params(pair, sd, spot, r_dom, r_for, **kw):
    """13. SABR Parameters across tenors (fitted alpha, rho, nu)."""
    tenors = sd["tenors"]
    T_arr = sd["T_years"]
    surface = sd["surface_raw"]

    alphas, rhos, nus = [], [], []
    fwd = spot * np.exp((r_dom - r_for) * T_arr)

    for i, t in enumerate(tenors):
        row = surface.get(t, {})
        atm_vol = row.get("atm", row.get("ATM", 0)) / 100.0
        rr25 = row.get("rr25", row.get("25D_RR", 0.0)) / 100.0
        bf25 = row.get("bf25", row.get("25D_BF", 0.0)) / 100.0

        # Approximate SABR params from market quotes
        F = fwd[i] if i < len(fwd) else spot
        T = T_arr[i] if i < len(T_arr) else 0.25
        beta = 0.5
        alpha_est = atm_vol * F ** (1 - beta)
        rho_est = np.clip(rr25 / max(atm_vol, 0.01) * (-0.8), -0.95, 0.95)
        nu_est = np.clip(bf25 / max(atm_vol, 0.01) * 3.0 + 0.3, 0.05, 3.0)

        alphas.append(alpha_est)
        rhos.append(rho_est)
        nus.append(nu_est)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Alpha", "Rho", "Nu"])

    fig.add_trace(go.Scatter(x=tenors, y=alphas, mode="lines+markers", name="Alpha",
        line=dict(color=COLORS["accent_cyan"], width=2),
        marker=dict(size=6),
        hovertemplate="%{x}: \u03b1=%{y:.4f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=tenors, y=rhos, mode="lines+markers", name="Rho",
        line=dict(color=COLORS["accent_rose"], width=2),
        marker=dict(size=6),
        hovertemplate="%{x}: \u03c1=%{y:.3f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=tenors, y=nus, mode="lines+markers", name="Nu",
        line=dict(color=COLORS["accent_orange"], width=2),
        marker=dict(size=6),
        hovertemplate="%{x}: \u03bd=%{y:.3f}<extra></extra>"), row=3, col=1)

    _apply_chart_template(fig, f"SABR Parameters -- {pair}")
    fig.update_layout(height=400, showlegend=False)
    for i in range(1, 4):
        fig.update_yaxes(gridcolor="#1a1a30", row=i, col=1)
        fig.update_xaxes(gridcolor="#1a1a30", row=i, col=1)
    return fig


@_safe_chart
def chart_implied_dist(pair, sd, spot, r_dom, r_for, **kw):
    """14. Implied Distribution -- risk-neutral PDF for selected tenor."""
    sel_tenor = kw.get("pdf_tenor", "3M")
    try:
        pdf_df = smile_implied_pdf(pair, sel_tenor, n_points=150)
        if pdf_df.empty:
            raise ValueError("empty pdf")
        strikes = pdf_df["strike"].values
        pdf_vals = pdf_df["pdf"].values
    except Exception:
        return no_data_fig(height=CHART_MD, msg="NO PDF DATA")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strikes, y=pdf_vals, mode="lines", name="Risk-Neutral PDF",
        fill="tozeroy", fillcolor="rgba(255,136,0,0.10)",
        line=dict(color=COLORS["accent_cyan"], width=2),
    ))

    # Forward price marker
    T = tenor_to_years(sel_tenor)
    F = spot * np.exp((r_dom - r_for) * T)
    fig.add_vline(x=F, line=dict(color=COLORS["accent_orange"], width=1.5, dash="dash"),
                  annotation_text=f"Fwd {F:.4f}",
                  annotation_font=dict(color=COLORS["accent_orange"], size=10))
    fig.add_vline(x=spot, line=dict(color=COLORS["accent_rose"], width=1, dash="dot"),
                  annotation_text=f"Spot {spot:.4f}",
                  annotation_font=dict(color=COLORS["accent_rose"], size=10))

    _apply_chart_template(fig, f"Implied PDF -- {pair} {sel_tenor}")
    fig.update_layout(xaxis=dict(title="Strike"), yaxis=dict(title="Density"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Lab: Time-series fetch + rendering (ported from chart_lab.py)
# ═══════════════════════════════════════════════════════════════════════════

_CW = CHART_TEMPLATE["layout"].get("colorway") or [
    "#ff8800", "#00cc66", "#1565c0", "#d4d4d4", "#ff3333", "#808080", "#ffaa33", "#ffffff"]
_FONT = "'JetBrains Mono', monospace"


def _lab_fetch_series(pair, metric_key, tenor, window):
    """Fetch a time series for a lab metric.  Returns (pd.Series|None, label)."""
    info = _LAB_METRIC_MAP.get(metric_key)
    if info is None:
        return None, f"{pair} {metric_key}"
    bbg_metric, kind = info
    label = f"{pair} {bbg_metric} {tenor}"
    try:
        if kind == "vol":
            data = get_fx_historical_vol(pair, tenor, bbg_metric, int(window))
            if data is not None and len(data) > 0:
                return (data if isinstance(data, pd.Series)
                        else pd.Series(np.asarray(data))), label
        elif bbg_metric == "SPOT":
            label = f"{pair} Spot"
            df = get_fx_historical_spot(pair, int(window))
            if df is not None and not df.empty:
                col = ("close" if "close" in df.columns
                       else ("Close" if "Close" in df.columns else df.columns[-1]))
                s = df[col]; s.name = label
                return s, label
        elif bbg_metric == "IV_RV":
            label = f"{pair} IV-RV {tenor}"
            df = iv_rv_spread(pair, tenor, lookback=int(window))
            if df is not None and not df.empty and "spread" in df.columns:
                s = df["spread"]; s.name = label; return s, label
        elif bbg_metric == "RV":
            label = f"{pair} RV"
            df = get_fx_historical_spot(pair, int(window) + 30)
            if df is not None and not df.empty:
                closes = df["close"] if "close" in df.columns else df.iloc[:, -1]
                rv = (np.log(closes / closes.shift(1)).dropna()
                      .rolling(20).std() * np.sqrt(252) * 100)
                rv = rv.dropna(); rv.name = label; return rv, label
        elif bbg_metric == "FWD_VOL":
            label = f"{pair} Fwd Vol"
            df = forward_vol_curve(pair)
            if df is not None and not df.empty and "forward_vol" in df.columns:
                idx = (df["end_tenor"].values if "end_tenor" in df.columns
                       else np.arange(len(df)))
                s = pd.Series(df["forward_vol"].values, index=idx, name=label)
                return s, label
        elif bbg_metric == "TERM_SPREAD":
            label = f"{pair} 1M-1Y Spread"
            v1m = get_fx_historical_vol(pair, "1M", "ATM", int(window))
            v1y = get_fx_historical_vol(pair, "1Y", "ATM", int(window))
            if v1m is not None and v1y is not None:
                a, b = np.asarray(v1m), np.asarray(v1y)
                n = min(len(a), len(b))
                if n > 0:
                    return pd.Series(a[-n:] - b[-n:], name=label), label
        elif bbg_metric == "CARRY":
            label = f"{pair} Carry (bps)"
            df = rate_differential_history(pair, lookback=int(window))
            if df is not None and not df.empty and "rate_diff" in df.columns:
                return pd.Series(df["rate_diff"].values * 10000, name=label), label
    except Exception:
        logger.debug("_lab_fetch_series error for %s/%s", pair, metric_key)
    return None, label


def _lab_apply_norm(s, mode):
    """Apply normalisation to a series."""
    if s is None or len(s) == 0 or mode == "raw":
        return s
    arr = s.values if isinstance(s, pd.Series) else np.asarray(s)
    if mode == "indexed":
        f = arr[0]
        if abs(f) < 1e-6 or np.isnan(f):
            return s
        result = (arr / f) * 100
    elif mode == "zscore":
        mu, sig = np.nanmean(arr), np.nanstd(arr)
        if sig == 0 or np.isnan(sig):
            return s
        result = (arr - mu) / sig
    elif mode == "pct_change":
        f = arr[0]
        if f == 0 or np.isnan(f):
            return s
        result = ((arr - f) / abs(f)) * 100
    else:
        return s
    return pd.Series(result,
                     index=s.index if isinstance(s, pd.Series) else None,
                     name=getattr(s, "name", None))


def _lab_build_ts_chart(chart_type, pair, sd, spot, r_dom, r_for, **kw):
    """Build a Lab time-series chart with multi-pair + overlay support."""
    lab_pairs = kw.get("lab_pairs") or [pair]
    if isinstance(lab_pairs, str):
        lab_pairs = [lab_pairs]
    lab_pairs = lab_pairs[:5]
    overlay = kw.get("lab_overlay") or ""
    window = kw.get("lab_window") or 252
    normalize = kw.get("lab_normalize") or "raw"
    tenor = kw.get("smile_tenor") or "3M"

    has_overlay = bool(overlay) and overlay != chart_type
    if has_overlay:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    any_data = False

    def _xy(series):
        x = (series.index if isinstance(series, pd.Series)
             and series.index.dtype != object else list(range(len(series))))
        return x, (series.values if isinstance(series, pd.Series) else series)

    # Primary metric
    for idx, p in enumerate(lab_pairs):
        series, label = _lab_fetch_series(p, chart_type, tenor, window)
        if series is None or len(series) == 0:
            continue
        any_data = True
        series = _lab_apply_norm(series, normalize)
        color = _CW[idx % len(_CW)]
        x, y = _xy(series)
        trace = go.Scatter(x=x, y=y, mode="lines", name=label,
                           line=dict(color=color, width=1.5))
        if has_overlay:
            fig.add_trace(trace, secondary_y=False)
        else:
            fig.add_trace(trace)

    # Overlay (secondary Y)
    if has_overlay:
        ov_colors = ["#ffffff", "#00cc66", "#ff3333", "#d4d4d4", "#ffaa33"]
        for idx, p in enumerate(lab_pairs):
            series, label = _lab_fetch_series(p, overlay, tenor, window)
            if series is None or len(series) == 0:
                continue
            any_data = True
            series = _lab_apply_norm(series, normalize)
            x, y = _xy(series)
            ov_name = next((m["label"] for m in LAB_OVERLAY_OPTIONS
                            if m["value"] == overlay), overlay)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=f"{p} {ov_name}",
                line=dict(color=ov_colors[idx % len(ov_colors)],
                          width=1.5, dash="dot")),
                secondary_y=True)

    if not any_data:
        return no_data_fig(height=CHART_MD, msg="NO DATA")

    m_label = next((m["label"] for m in CHART_OPTIONS
                    if m["value"] == chart_type), chart_type)
    title = f"{m_label} | {tenor} | {window}d"
    if normalize != "raw":
        title += f" [{normalize}]"

    _apply_chart_template(fig, title)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=45, r=50 if has_overlay else 15, t=40, b=30),
        hovermode="x unified")

    if has_overlay:
        ov_label = next((m["label"] for m in LAB_OVERLAY_OPTIONS
                         if m["value"] == overlay), overlay)
        fig.update_yaxes(title_text=m_label, secondary_y=False,
                         title_font=dict(size=9, color="#ff8800"))
        fig.update_yaxes(title_text=ov_label, secondary_y=True,
                         title_font=dict(size=9, color="#ffffff"),
                         gridcolor="rgba(34,34,64,0.2)")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Lab: Study chart builders (ported from chart_lab.py)
# ═══════════════════════════════════════════════════════════════════════════

def _lab_study_vol_cone(pair, sd, spot, r_dom, r_for, **kw):
    window = kw.get("lab_window") or 252
    df = vol_cone(pair, lookback=int(window))
    if df is None or df.empty:
        return _empty_fig("No vol cone data")
    fig = go.Figure()
    w = df["window"].tolist()
    for col, name in [("p90", "90th"), ("p75", "75th"),
                      ("p25", "25th"), ("p10", "10th")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=w, y=df[col].tolist(), mode="lines",
                          name=name, line=dict(color="#808080", width=1, dash="dot"),
                          hovertemplate="%{x}d: %{y:.2f}%<extra>" + name + "</extra>"))
    if "median" in df.columns:
        fig.add_trace(go.Scatter(x=w, y=df["median"].tolist(), mode="lines",
                      name="Median", line=dict(color="#d4d4d4", width=1.5, dash="dash"),
                      hovertemplate="%{x}d: %{y:.2f}%<extra>Median</extra>"))
    if "current_c2c" in df.columns:
        fig.add_trace(go.Scatter(x=w, y=df["current_c2c"].tolist(),
                      mode="lines+markers", name="Current",
                      line=dict(color="#ff8800", width=2.5),
                      marker=dict(size=6, color="#ff8800"),
                      hovertemplate="%{x}d: %{y:.2f}%<extra>Current RV</extra>"))
    _apply_chart_template(fig, f"{pair} Realized Vol Cone")
    fig.update_layout(xaxis_title="Window (days)", yaxis_title="RV (%)",
                      hovermode="x unified")
    return fig


def _lab_study_smile(pair, sd, spot, r_dom, r_for, **kw):
    surface = get_fx_vol_surface(pair)
    if not surface:
        return _empty_fig("No surface data")
    fig = go.Figure()
    for idx, t in enumerate(["1M", "3M", "6M", "1Y"]):
        td = surface.get(t, {})
        atm = td.get("atm", 0)
        if atm == 0:
            continue
        rr25, bf25 = td.get("rr25", 0), td.get("bf25", 0)
        rr10, bf10 = td.get("rr10", 0), td.get("bf10", 0)
        vols = [atm - rr10/2 + bf10, atm - rr25/2 + bf25, atm,
                atm + rr25/2 + bf25, atm + rr10/2 + bf10]
        fig.add_trace(go.Scatter(x=["10P", "25P", "ATM", "25C", "10C"],
                      y=vols, mode="lines+markers", name=t,
                      line=dict(color=_CW[idx % len(_CW)], width=2),
                      hovertemplate="%{x}: %{y:.2f}%<extra>" + t + "</extra>"))
    _apply_chart_template(fig, f"{pair} Vol Smile")
    fig.update_layout(xaxis_title="Delta", yaxis_title="IV (%)",
                      hovermode="x unified")
    return fig


def _lab_study_implied_pdf(pair, sd, spot, r_dom, r_for, **kw):
    tenor = kw.get("smile_tenor") or "3M"
    df = smile_implied_pdf(pair, tenor)
    if df is None or df.empty:
        return _empty_fig("No PDF data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["strike"].tolist(), y=df["pdf"].tolist(),
                  mode="lines", name="Implied PDF", fill="tozeroy",
                  line=dict(color="#ff8800", width=2),
                  fillcolor="rgba(255,136,0,0.15)",
                  hovertemplate="Strike: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>"))
    spots = get_fx_spots([pair]) or {}
    s = spots.get(pair, {}).get("mid")
    if s:
        fig.add_vline(x=s, line_dash="dash", line_color="#d4d4d4",
                      annotation_text=f"Spot {s:.4f}")
    _apply_chart_template(fig, f"{pair} {tenor} Implied PDF")
    fig.update_layout(xaxis_title="Strike", yaxis_title="Density",
                      hovermode="x unified")
    return fig


def _lab_study_vol_regime(pair, sd, spot, r_dom, r_for, **kw):
    window = kw.get("lab_window") or 252
    df = vol_regime_history(pair, lookback=int(window))
    if df is None or df.empty:
        return _empty_fig("No regime data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"].tolist(), y=df["vol"].tolist(),
                  mode="lines", name=f"{pair} ATM",
                  line=dict(color="#ff8800", width=2),
                  hovertemplate="Day %{x}<br>Vol: %{y:.2f}%<extra></extra>"))
    for level, color, lbl in [(20, "#ff3333", "CRISIS"), (14, "#ff8800", "HIGH"),
                               (10, "#ffaa33", "ELEVATED"), (6, "#808080", "NORMAL")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                      annotation_text=lbl, annotation_font_size=8,
                      annotation_font_color=color)
    regime = vol_regime_detect(pair) or {}
    _apply_chart_template(
        fig, f"{pair} Vol Regime: {regime.get('regime','?')} | "
             f"Trend: {regime.get('trend','?')}")
    fig.update_layout(xaxis_title="Date", yaxis_title="ATM Vol (%)",
                      hovermode="x unified")
    return fig


def _lab_study_fwd_vol_curve(pair, sd, spot, r_dom, r_for, **kw):
    tenor = kw.get("smile_tenor") or "3M"
    df = forward_vol_curve(pair, start_tenor=tenor)
    if df is None or df.empty:
        return _empty_fig("No forward vol data")
    fig = go.Figure()
    if "spot_vol" in df.columns:
        fig.add_trace(go.Scatter(x=df["end_tenor"].tolist(),
                      y=df["spot_vol"].tolist(), mode="lines+markers",
                      name="Spot Vol", line=dict(color="#d4d4d4", width=1.5),
                      hovertemplate="%{x}: %{y:.2f}%<extra>Spot</extra>"))
    fig.add_trace(go.Scatter(x=df["end_tenor"].tolist(),
                  y=df["forward_vol"].tolist(), mode="lines+markers",
                  name="Forward Vol", line=dict(color="#ff8800", width=2.5),
                  marker=dict(size=6),
                  hovertemplate="%{x}: %{y:.2f}%<extra>Forward</extra>"))
    _apply_chart_template(fig, f"{pair} Forward Vol (from {tenor})")
    fig.update_layout(xaxis_title="End Tenor", yaxis_title="Vol (%)",
                      hovermode="x unified")
    return fig


def _lab_study_pctile_surface(pair, sd, spot, r_dom, r_for, **kw):
    df = vol_percentile_surface(pair)
    if df is None or df.empty:
        return _empty_fig("No percentile surface data")
    fig = go.Figure(data=go.Heatmap(
        z=df.values, x=df.columns.tolist(), y=df.index.tolist(),
        colorscale=[[0, "#00cc66"], [0.25, "#222240"], [0.5, "#808080"],
                    [0.75, "#222240"], [1, "#ff3333"]],
        text=np.round(df.values, 1).astype(str), texttemplate="%{text}",
        textfont=dict(size=10, color="#d4d4d4"),
        hovertemplate="Tenor: %{y}<br>Delta: %{x}<br>Percentile: %{z:.1f}%<extra></extra>",
        xgap=2, ygap=2))
    _apply_chart_template(fig, f"{pair} Percentile Surface")
    fig.update_layout(xaxis_title="Delta", yaxis_title="Tenor")
    return fig


def _lab_study_zscore_surface(pair, sd, spot, r_dom, r_for, **kw):
    df = vol_zscore_surface(pair)
    if df is None or df.empty:
        return _empty_fig("No z-score surface data")
    fig = go.Figure(data=go.Heatmap(
        z=df.values, x=df.columns.tolist(), y=df.index.tolist(),
        colorscale=[[0, "#00cc66"], [0.5, "#000000"], [1, "#ff3333"]],
        text=np.round(df.values, 1).astype(str), texttemplate="%{text}",
        textfont=dict(size=10, color="#d4d4d4"),
        hovertemplate="Tenor: %{y}<br>Delta: %{x}<br>Z-Score: %{z:.2f}<extra></extra>",
        xgap=2, ygap=2))
    _apply_chart_template(fig, f"{pair} Z-Score Surface")
    fig.update_layout(xaxis_title="Delta", yaxis_title="Tenor")
    return fig


def _lab_study_tail_probs(pair, sd, spot, r_dom, r_for, **kw):
    tenor = kw.get("smile_tenor") or "3M"
    df = tail_probabilities(pair, tenor)
    if df is None or df.empty:
        return _empty_fig("No tail prob data")
    fig = go.Figure()
    move_labels = [f"+{m:.0f}%" if np.isfinite(m) else "?" for m in df["move_pct"]]
    fig.add_trace(go.Bar(x=move_labels,
                  y=df["prob_up"].tolist(), name="Up",
                  marker_color="#00cc66", opacity=0.85))
    fig.add_trace(go.Bar(x=move_labels,
                  y=df["prob_down"].tolist(), name="Down",
                  marker_color="#ff3333", opacity=0.85))
    _apply_chart_template(fig, f"{pair} {tenor} Tail Probabilities")
    fig.update_layout(xaxis_title="Move Size", yaxis_title="Probability (%)",
                      barmode="group")
    return fig


def _lab_study_breakeven(pair, sd, spot, r_dom, r_for, **kw):
    from core.fx_conventions import tenor_to_days
    tenor = kw.get("smile_tenor") or "3M"
    days = tenor_to_days(tenor)
    info = breakeven_vol(pair, tenor, days)
    if not info:
        return _empty_fig("No breakeven data")
    fig = go.Figure()
    labels = ["ATM IV", "Breakeven RV", "Cushion"]
    vals = [info.get("atm_iv", 0), info.get("breakeven_rv", 0),
            info.get("iv_rv_cushion", 0)]
    colors = ["#ff8800", "#d4d4d4",
              "#00cc66" if info.get("iv_rv_cushion", 0) > 0 else "#ff3333"]
    fig.add_trace(go.Bar(x=labels, y=vals, marker_color=colors,
                  text=[f"{v:.2f}" if np.isfinite(v) else "—" for v in vals], textposition="outside",
                  textfont=dict(color="#d4d4d4", size=11)))
    _apply_chart_template(fig, f"{pair} {tenor} Breakeven Analysis")
    fig.update_layout(yaxis_title="Vol (%)")
    return fig


def _lab_study_carry_landscape(pair, sd, spot, r_dom, r_for, **kw):
    lab_pairs = kw.get("lab_pairs") or [pair]
    if isinstance(lab_pairs, str):
        lab_pairs = [lab_pairs]
    df = carry_per_vol(lab_pairs)
    if df is None or df.empty:
        return _empty_fig("No carry data")
    colors = ["#00cc66" if s == "ATTRACTIVE"
              else ("#ff8800" if s == "MODERATE" else "#ff3333")
              for s in df["rank_signal"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["pair"].tolist(), y=df["sharpe_proxy"].tolist(),
                  marker_color=colors,
                  text=[f"{v:.2f}" if np.isfinite(v) else "—" for v in df["sharpe_proxy"]],
                  textposition="outside",
                  textfont=dict(color="#d4d4d4", size=10)))
    _apply_chart_template(fig, "Carry / Vol Ranking (Sharpe Proxy)")
    fig.update_layout(yaxis_title="Sharpe Proxy")
    return fig


# Chart dispatch table
CHART_DISPATCH = {
    # Surface charts
    "surface_3d": chart_surface_3d,
    "heatmap": chart_heatmap,
    "atm_term": chart_atm_term,
    "skew_rr": chart_skew_rr,
    "smile_bf": chart_smile_bf,
    "rich_cheap": chart_rich_cheap,
    "smile_curve": chart_smile_curve,
    "vol_ts": chart_vol_ts,
    "iv_rv": chart_iv_rv,
    "vol_cone_chart": chart_vol_cone_chart,
    "fwd_vol": chart_fwd_vol,
    "surface_change": chart_surface_change,
    "sabr_params": chart_sabr_params,
    "implied_dist": chart_implied_dist,
    # Lab study charts
    "lab_study_vol_cone": _lab_study_vol_cone,
    "lab_study_smile": _lab_study_smile,
    "lab_study_implied_pdf": _lab_study_implied_pdf,
    "lab_study_vol_regime": _lab_study_vol_regime,
    "lab_study_fwd_vol_curve": _lab_study_fwd_vol_curve,
    "lab_study_pctile_surface": _lab_study_pctile_surface,
    "lab_study_zscore_surface": _lab_study_zscore_surface,
    "lab_study_tail_probs": _lab_study_tail_probs,
    "lab_study_breakeven": _lab_study_breakeven,
    "lab_study_carry_landscape": _lab_study_carry_landscape,
}


def _render_chart(chart_type, pair, sd, spot, r_dom, r_for, **kw):
    """Dispatch to the correct chart function.

    Lab time-series metrics (lab_atm, lab_25d_rr, etc.) are handled
    by _lab_build_ts_chart.  Lab study metrics and surface charts go
    through the dispatch table.
    """
    # Lab time-series metrics → multi-pair TS builder
    if chart_type in _LAB_METRIC_MAP:
        try:
            return _lab_build_ts_chart(chart_type, pair, sd, spot, r_dom, r_for, **kw)
        except Exception as exc:
            return _empty_fig(f"Lab TS error: {exc}")

    # Surface + study dispatch
    fn = CHART_DISPATCH.get(chart_type)
    if fn is None:
        return _empty_fig(f"Unknown chart: {chart_type}")
    try:
        return fn(pair, sd, spot, r_dom, r_for, **kw)
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Stat Boxes builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_stat_boxes(pair, sd, spot, fwd_1m, r_dom, r_for):
    """Build the 10 KPI stat boxes with institutional-grade formatting.

    Formatting rules:
      - Spot: 4dp for most pairs, 2dp for JPY crosses
      - Vols: 2dp + 'v' suffix (e.g. '8.52v')
      - Percentiles: integer + ordinal suffix ('23rd', '45th')
      - Regime: coloured badge
    """
    # -- Extract metrics safely --
    def _safe_tenor_idx(tenors, label):
        try:
            return tenors.index(label)
        except ValueError:
            return None

    _idx_1m = _safe_tenor_idx(sd["tenors"], "1M")
    atm_1m = sd["atm"][_idx_1m] if _idx_1m is not None else (sd["atm"][0] if len(sd["atm"]) > 0 else 0)
    _idx_1y = _safe_tenor_idx(sd["tenors"], "1Y")
    atm_1y = sd["atm"][_idx_1y] if _idx_1y is not None else (sd["atm"][-1] if len(sd["atm"]) > 0 else 0)

    # ATM 1M change
    atm_1m_delta = 0.0
    try:
        ch = vol_change(pair, "1M", "ATM", days_ago=1)
        atm_1m_delta = ch.get("abs_change", 0) if ch else 0
    except Exception:
        pass

    # Vol percentile for ATM 1M
    atm_pctile = 50.0
    try:
        p = vol_percentile(pair, "1M", "ATM")
        atm_pctile = p.get("percentile", 50.0) if p else 50.0
    except Exception:
        pass

    # 25D RR 3M + percentile
    rr_3m = 0.0
    rr_pctile = 50.0
    try:
        if "3M" in sd["tenors"]:
            idx = sd["tenors"].index("3M")
        elif len(sd["tenors"]) > 0:
            idx = min(2, len(sd["tenors"]) - 1)
        else:
            idx = -1
        rr_3m = sd["rr25"][idx] if 0 <= idx < len(sd.get("rr25", [])) else 0.0
        p = vol_percentile(pair, "3M", "25D_RR")
        rr_pctile = p.get("percentile", 50.0) if p else 50.0
    except Exception:
        pass

    # 25D BF 3M
    bf_3m = 0.0
    try:
        idx = sd["tenors"].index("3M") if "3M" in sd["tenors"] else min(2, len(sd["tenors"]) - 1)
        bf_3m = sd["bf25"][idx] if 0 <= idx < len(sd.get("bf25", [])) else 0.0
    except Exception:
        pass

    # IV-RV spread
    iv_rv_spr = 0.0
    try:
        irp = iv_rv_percentile(pair, "3M", rv_window=20)
        iv_rv_spr = irp.get("current_spread", 0) if irp else 0
    except Exception:
        pass

    # Term spread 1Y-1M
    term_spr = atm_1y - atm_1m

    # Skew percentile
    skew_pctile = 50.0
    try:
        p = vol_percentile(pair, "3M", "25D_RR")
        skew_pctile = p.get("percentile", 50.0) if p else 50.0
    except Exception:
        pass

    # Regime
    regime_text = "NORMAL"
    regime_color = COLORS["accent_blue"]
    try:
        reg = vol_regime_detect(pair)
        regime_text = reg.get("regime", "NORMAL") if reg else "NORMAL"
        regime_color = reg.get("color", COLORS["accent_blue"]) if reg else COLORS["accent_blue"]
    except Exception:
        pass

    # -- Build boxes with proper formatting --
    def _box(label, value_str, color):
        """Plain stat box for values without a vol-history context."""
        return html.Div([
            html.Div(value_str, style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "28px", "fontWeight": "700",
                "color": color, "lineHeight": "1.1",
            }),
            html.Div(label, style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "9px", "fontWeight": "600",
                "color": COLORS["text_muted"], "textTransform": "uppercase",
                "letterSpacing": "1px", "marginTop": "4px",
            }),
        ], style={**make_stat_style(color), "flex": "1", "minWidth": "110px"})

    def _cbox(label, value_str, pair, metric, tenor, color):
        """Clickable stat box -- wraps clickable_stat() so each KPI opens
        its 252-day history via the universal metric popup."""
        inner = clickable_stat(value_str, label, pair, metric, tenor, color)
        return html.Div(inner, style={"flex": "1", "minWidth": "110px"})

    # Vol change direction coloring: green = vol cheaper (down), red = vol richer (up)
    delta_color = COLORS["accent_green"] if atm_1m_delta < 0 else COLORS["accent_red"]
    delta_arrow = "\u25bc" if atm_1m_delta < 0 else ("\u25b2" if atm_1m_delta > 0 else "\u25ac")
    delta_str = f"{delta_arrow}{abs(atm_1m_delta):.2f}"

    boxes = [
        _box("SPOT", _fmt_spot(spot, pair), COLORS["text_primary"]),
        _box("FWD 1M", _fmt_spot(fwd_1m, pair), COLORS["accent_blue"]),
        _cbox("ATM 1M", f"{_fmt_vol(atm_1m)} ({delta_str})", pair, "ATM", "1M", COLORS["accent_cyan"]),
        _cbox("ATM 1Y", _fmt_vol(atm_1y), pair, "ATM", "1Y", COLORS["accent_purple"]),
        _cbox("25D RR 3M", f"{rr_3m:+.2f}v ({_fmt_pctile(rr_pctile)})", pair, "25D_RR", "3M", COLORS["accent_orange"]),
        _cbox("25D BF 3M", _fmt_vol(bf_3m), pair, "25D_BF", "3M", COLORS["accent_pink"]),
        _cbox("IV-RV SPREAD", f"{iv_rv_spr:+.2f}v", pair, "IV_RV", "3M", delta_color),
        _box("TERM 1Y-1M", f"{term_spr:+.2f}v", COLORS["accent_teal"]),
        _cbox("SKEW %ILE", _fmt_pctile(skew_pctile), pair, "25D_RR", "3M", COLORS["accent_indigo"]),
        # Regime badge with colored background indicator
        html.Div([
            html.Div(regime_text, style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "20px", "fontWeight": "700",
                "color": regime_color, "lineHeight": "1.1",
            }),
            html.Div("REGIME", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "9px", "fontWeight": "600",
                "color": COLORS["text_muted"], "textTransform": "uppercase",
                "letterSpacing": "1px", "marginTop": "4px",
            }),
        ], style={
            **make_stat_style(regime_color), "flex": "1", "minWidth": "110px",
            "borderLeft": f"3px solid {regime_color}",
        }),
    ]
    return boxes


def _build_overnight_summary(pair, spot):
    """Build the overnight change summary strip.

    Shows ATM 1M vol change (bp + direction arrow) and spot change (pips + direction).
    Color coding: green = vol cheaper (down), red = vol richer (up).
    """
    # ATM 1M vol change
    vol_chg_bp = 0.0
    vol_current = 0.0
    try:
        ch = vol_change(pair, "1M", "ATM", days_ago=1)
        vol_chg_bp = ch.get("abs_change", 0) if ch else 0
        vol_current = ch.get("current", 0) if ch else 0
    except Exception:
        pass

    # Spot change in pips (1 pip = 0.0001 for most pairs, 0.01 for JPY)
    spot_chg_pips = 0.0
    spot_prev = spot
    try:
        hist_raw = get_fx_historical_spot(pair, 5)
        # Extract close array from whatever format get_fx_historical_spot returns
        hist = None
        if hist_raw is not None:
            if isinstance(hist_raw, pd.DataFrame):
                for col in ("close", "Close"):
                    if col in hist_raw.columns:
                        hist = hist_raw[col].values
                        break
                if hist is None:
                    hist = hist_raw.iloc[:, -1].values
            elif isinstance(hist_raw, pd.Series):
                hist = hist_raw.values
            else:
                hist = np.asarray(hist_raw)
        if hist is not None and len(hist) >= 2:
            spot_prev = float(hist[-2])
            pip_unit = 0.01 if _is_jpy_pair(pair) else 0.0001
            spot_chg_pips = (spot - spot_prev) / pip_unit
    except Exception:
        pass

    mono = "'JetBrains Mono', monospace"

    # Vol change chip
    vol_arrow = "\u25bc" if vol_chg_bp < 0 else ("\u25b2" if vol_chg_bp > 0 else "\u25ac")
    # Green = vol down (cheaper), Red = vol up (richer)
    vol_color = COLORS["accent_green"] if vol_chg_bp < 0 else (COLORS["accent_red"] if vol_chg_bp > 0 else COLORS["text_muted"])
    vol_chip = html.Span([
        html.Span("ATM 1M: ", style={"color": COLORS["text_muted"], "fontSize": "10px"}),
        html.Span(f"{vol_arrow} {abs(vol_chg_bp):.1f}bp", style={
            "color": vol_color, "fontWeight": "700", "fontSize": "12px",
        }),
        html.Span(f" ({_fmt_vol(vol_current)})", style={
            "color": COLORS["text_secondary"], "fontSize": "10px",
        }),
    ], style={"marginRight": "24px", "fontFamily": mono})

    # Spot change chip
    spot_arrow = "\u25b2" if spot_chg_pips > 0 else ("\u25bc" if spot_chg_pips < 0 else "\u25ac")
    spot_color = COLORS["accent_green"] if spot_chg_pips > 0 else (COLORS["accent_red"] if spot_chg_pips < 0 else COLORS["text_muted"])
    spot_chip = html.Span([
        html.Span("SPOT: ", style={"color": COLORS["text_muted"], "fontSize": "10px"}),
        html.Span(f"{spot_arrow} {abs(spot_chg_pips):.1f} pips", style={
            "color": spot_color, "fontWeight": "700", "fontSize": "12px",
        }),
        html.Span(f" ({_fmt_spot(spot, pair)})", style={
            "color": COLORS["text_secondary"], "fontSize": "10px",
        }),
    ], style={"marginRight": "24px", "fontFamily": mono})

    # 25D RR 3M change chip
    rr_chg = 0.0
    try:
        rr_ch = vol_change(pair, "3M", "25D_RR", days_ago=1)
        rr_chg = rr_ch.get("abs_change", 0) if rr_ch else 0
    except Exception:
        pass
    rr_arrow = "\u25bc" if rr_chg < 0 else ("\u25b2" if rr_chg > 0 else "\u25ac")
    rr_color = COLORS["text_muted"] if abs(rr_chg) < 0.05 else (COLORS["accent_orange"] if rr_chg > 0 else COLORS["accent_blue"])
    rr_chip = html.Span([
        html.Span("RR 3M: ", style={"color": COLORS["text_muted"], "fontSize": "10px"}),
        html.Span(f"{rr_arrow} {abs(rr_chg):.2f}v", style={
            "color": rr_color, "fontWeight": "700", "fontSize": "12px",
        }),
    ], style={"fontFamily": mono})

    return html.Div([
        html.Span("OVERNIGHT ", style={
            "fontFamily": mono, "fontSize": "9px", "fontWeight": "700",
            "color": COLORS["accent_cyan"], "letterSpacing": "1.5px",
            "marginRight": "12px",
        }),
        vol_chip, spot_chip, rr_chip,
    ], style={
        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
        "padding": "8px 14px",
        "backgroundColor": COLORS["bg_secondary"],
        "borderRadius": "0px",
        "border": f"1px solid {COLORS['border']}",
        "marginBottom": "10px",
    })


# ═══════════════════════════════════════════════════════════════════════════
# Pair dropdown options (grouped)
# ═══════════════════════════════════════════════════════════════════════════

def _pair_options():
    """Build grouped dropdown options for 30 pairs."""
    opts = []
    for group, pairs in _PAIR_GROUPS.items():
        for p in pairs:
            opts.append({"label": f"{group} | {p[:3]}/{p[3:]}", "value": p})
    return opts


# ═══════════════════════════════════════════════════════════════════════════
# Comparison Overlay Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _overlay_history_on_atm(fig, pair, sd, days_ago):
    """Add a historical ATM term structure overlay to an ATM term chart.

    Draws a dashed line for the ATM levels 'days_ago' trading days back.
    """
    tenors = sd["tenors"]
    hist_vals = []
    for t in tenors:
        try:
            ch = vol_change(pair, t, "ATM", days_ago=days_ago)
            hist_vals.append(ch["previous"])
        except Exception:
            hist_vals.append(None)

    # Only draw if we have some data
    valid = [v for v in hist_vals if v is not None]
    if not valid:
        return

    label_map = {1: "1D ago", 5: "1W ago", 22: "1M ago", 66: "3M ago"}
    label = label_map.get(days_ago, f"{days_ago}D ago")

    fig.add_trace(go.Scatter(
        x=tenors,
        y=[v if v is not None else float("nan") for v in hist_vals],
        mode="lines+markers",
        name=label,
        line=dict(color=COLORS["accent_orange"], width=2, dash="dash"),
        marker=dict(size=5, symbol="diamond", color=COLORS["accent_orange"]),
    ))


def _overlay_history_annotation(fig, pair, sd, days_ago):
    """Add a text annotation to heatmap charts noting the comparison period."""
    label_map = {1: "1D", 5: "1W", 22: "1M", 66: "3M"}
    label = label_map.get(days_ago, f"{days_ago}D")
    fig.add_annotation(
        text=f"vs {label} ago",
        xref="paper", yref="paper", x=0.98, y=0.02,
        showarrow=False,
        font=dict(color=COLORS["accent_orange"], size=10,
                  family="'JetBrains Mono', monospace"),
        bgcolor=COLORS["bg_secondary"], bordercolor=COLORS["border"],
        borderwidth=1, borderpad=4,
    )


def _overlay_cross_pair_on_atm(fig, cross_pair, cross_sd):
    """Overlay another pair's ATM term structure on the ATM term chart."""
    tenors = cross_sd["tenors"]
    atm = cross_sd["atm"]

    fig.add_trace(go.Scatter(
        x=tenors, y=atm, mode="lines+markers",
        name=f"{cross_pair} ATM",
        line=dict(color=COLORS["accent_rose"], width=2, dash="dashdot"),
        marker=dict(size=5, symbol="square", color=COLORS["accent_rose"]),
    ))


def _overlay_cross_surface_wireframe(fig, cross_pair, cross_sd):
    """Add a wireframe overlay of the cross-pair surface on a 3D surface chart."""
    delta_pos = cross_sd["delta_numeric"]
    delta_labels = cross_sd["delta_labels"]
    n_tenors = cross_sd["vol_grid"].shape[0]
    n_deltas = len(delta_labels)
    hover_text = [[None] * n_deltas for _ in range(n_tenors)]
    for ti in range(n_tenors):
        for di in range(n_deltas):
            hover_text[ti][di] = (
                f"{cross_pair}<br>"
                f"Delta: {delta_labels[di]}<br>"
                f"Tenor: {cross_sd['tenors'][ti]}<br>"
                f"Vol: {cross_sd['vol_grid'][ti, di]:.2f}%"
            )
    fig.add_trace(go.Surface(
        x=delta_pos * 100,
        y=cross_sd["T_years"],
        z=cross_sd["vol_grid"],
        colorscale=[[0, "#0e0e0e"], [0.5, "#1565c0"], [1.0, "#42a5f5"]],
        opacity=0.35,
        showscale=False,
        name=f"{cross_pair}",
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
    ))


# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

def layout():
    pair_opts = _pair_options()
    return html.Div([
        # Hidden stores and interval
        dcc.Interval(id="vsfx-interval", interval=30000, n_intervals=0, disabled=True),
        dcc.Store(id="vsfx-preset-store", data="Trader"),
        dcc.Download(id="vsfx-csv-download"),

        # ── Outer flex container ──────────────────────────────────
        html.Div([
            # ──── Sidebar ─────────────────────────────────────────
            html.Div([
                html.Div("FX VOL WORKSTATION", style={
                    "color": COLORS["accent_cyan"], "fontSize": "12px",
                    "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace",
                    "letterSpacing": "1.5px", "textTransform": "uppercase",
                    "marginBottom": "16px", "textAlign": "center",
                    "paddingBottom": "10px",
                    "borderBottom": f"1px solid {COLORS['border']}",
                }),

                # Pair dropdown
                html.Div([
                    html.Label("PAIR", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-pair", options=pair_opts,
                                 value="EURUSD", clearable=False,
                                 style={"fontSize": "11px"}),
                ], style=SIDEBAR_SECTION),

                # Model radio
                html.Div([
                    html.Label("MODEL", style=SIDEBAR_LABEL),
                    dcc.RadioItems(id="vsfx-model", options=[
                        {"label": " Market", "value": "market"},
                        {"label": " SABR", "value": "sabr"},
                        {"label": " Vanna-Volga", "value": "vv"},
                    ], value="market",
                    style={"color": COLORS["text_secondary"], "fontSize": "10px",
                           "fontFamily": "'JetBrains Mono', monospace"},
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"display": "block", "marginBottom": "3px", "cursor": "pointer"}),
                ], style=SIDEBAR_SECTION),

                # View presets
                html.Div([
                    html.Label("VIEW PRESET", style=SIDEBAR_LABEL),
                    html.Div([
                        html.Button("Trader", id="vsfx-preset-trader", n_clicks=0, style=PRESET_BTN),
                        html.Button("Skew", id="vsfx-preset-skew", n_clicks=0, style=PRESET_BTN),
                        html.Button("Term", id="vsfx-preset-term", n_clicks=0, style=PRESET_BTN),
                        html.Button("Rich-Cheap", id="vsfx-preset-rc", n_clicks=0, style=PRESET_BTN),
                    ], style={"display": "flex", "flexWrap": "wrap"}),
                    html.Div(style={"height": "4px"}),
                    html.Div([
                        html.Button("Lab: Vol", id="vsfx-preset-lab-vol", n_clicks=0,
                                    style={**PRESET_BTN, "color": COLORS["accent_cyan"]}),
                        html.Button("Lab: Smile", id="vsfx-preset-lab-smile", n_clicks=0,
                                    style={**PRESET_BTN, "color": COLORS["accent_cyan"]}),
                        html.Button("Lab: RV", id="vsfx-preset-lab-rv", n_clicks=0,
                                    style={**PRESET_BTN, "color": COLORS["accent_cyan"]}),
                        html.Button("Lab: Carry", id="vsfx-preset-lab-carry", n_clicks=0,
                                    style={**PRESET_BTN, "color": COLORS["accent_cyan"]}),
                    ], style={"display": "flex", "flexWrap": "wrap"}),
                ], style=SIDEBAR_SECTION),

                # ── Group divider: core selection → comparison settings ──
                html.Div(style={"borderBottom": "1px solid #333355", "margin": "8px 0 12px 0"}),

                # Comparison toggle
                html.Div([
                    html.Label("COMPARISON", style=SIDEBAR_LABEL),
                    dcc.RadioItems(id="vsfx-compare", options=[
                        {"label": " None", "value": "none"},
                        {"label": " vs History", "value": "history"},
                        {"label": " vs Pair", "value": "cross"},
                    ], value="none",
                    style={"color": COLORS["text_secondary"], "fontSize": "10px",
                           "fontFamily": "'JetBrains Mono', monospace"},
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"display": "block", "marginBottom": "3px", "cursor": "pointer"}),
                ], style=SIDEBAR_SECTION),

                # History offset
                html.Div([
                    html.Label("HISTORY OFFSET", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-hist-offset", options=HISTORY_OFFSETS,
                                 value=1, clearable=False,
                                 style={"fontSize": "10px"}),
                ], style=SIDEBAR_SECTION),

                # Cross-pair picker
                html.Div([
                    html.Label("CROSS PAIR", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-cross-pair", options=pair_opts,
                                 value="USDJPY", clearable=False,
                                 style={"fontSize": "10px"}),
                ], style=SIDEBAR_SECTION),

                # ── Group divider: comparison settings → view settings ──
                html.Div(style={"borderBottom": "1px solid #333355", "margin": "8px 0 12px 0"}),

                # Tenor checklist
                html.Div([
                    html.Label("TENORS", style=SIDEBAR_LABEL),
                    dcc.Checklist(id="vsfx-tenors",
                        options=[{"label": f" {t}", "value": t} for t in TENORS_LIST],
                        value=["1W", "1M", "3M", "6M", "1Y"],
                        style={"color": COLORS["text_secondary"], "fontSize": "10px",
                               "fontFamily": "'JetBrains Mono', monospace"},
                        inputStyle={"marginRight": "3px"},
                        labelStyle={"display": "inline-block", "marginRight": "6px",
                                    "marginBottom": "3px", "cursor": "pointer"}),
                ], style=SIDEBAR_SECTION),

                # Delta range
                html.Div([
                    html.Label("DELTA RANGE", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-delta-range", options=[
                        {"label": "10D - 50D", "value": "10-50"},
                        {"label": "25D - 50D", "value": "25-50"},
                    ], value="10-50", clearable=False,
                    style={"fontSize": "10px"}),
                ], style=SIDEBAR_SECTION),

                # Chart type dropdowns for quadrants 1-4
                html.Div([
                    html.Label("CHART Q1", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-q1", options=CHART_OPTIONS,
                                 value="surface_3d", clearable=False,
                                 style={"fontSize": "10px", "marginBottom": "6px"}),
                    html.Label("CHART Q2", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-q2", options=CHART_OPTIONS,
                                 value="atm_term", clearable=False,
                                 style={"fontSize": "10px", "marginBottom": "6px"}),
                    html.Label("CHART Q3", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-q3", options=CHART_OPTIONS,
                                 value="skew_rr", clearable=False,
                                 style={"fontSize": "10px", "marginBottom": "6px"}),
                    html.Label("CHART Q4", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-q4", options=CHART_OPTIONS,
                                 value="iv_rv", clearable=False,
                                 style={"fontSize": "10px", "marginBottom": "6px"}),
                ], style=SIDEBAR_SECTION),

                # Smile / timeseries / PDF tenor
                html.Div([
                    html.Label("SMILE TENOR", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-smile-tenor", options=[
                        {"label": t, "value": t} for t in TENORS_LIST
                    ], value="3M", clearable=False,
                    style={"fontSize": "10px"}),
                ], style=SIDEBAR_SECTION),

                # Auto-refresh
                html.Div([
                    html.Label("AUTO REFRESH", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-refresh", options=[
                        {"label": "15s", "value": 15000},
                        {"label": "30s", "value": 30000},
                        {"label": "60s", "value": 60000},
                        {"label": "Off", "value": 0},
                    ], value=0, clearable=False,
                    style={"fontSize": "10px"}),
                ], style=SIDEBAR_SECTION),

                # Build Trade link → jumps to Trade Workshop with current pair
                html.Div([
                    html.Button("\u2192 BUILD TRADE", id="vsfx-build-trade-btn",
                        n_clicks=0, style={
                            "backgroundColor": COLORS["accent_orange"],
                            "color": "#000", "border": "none", "borderRadius": "0px",
                            "padding": "6px 12px", "width": "100%",
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "10px", "fontWeight": "700",
                            "cursor": "pointer", "letterSpacing": "1px",
                            "textTransform": "uppercase",
                        }),
                ], style=SIDEBAR_SECTION),

                # ── Group divider: view → lab controls ──
                html.Div(style={"borderBottom": "1px solid #333355", "margin": "8px 0 12px 0"}),

                html.Div("LAB CONTROLS", style={
                    "color": COLORS["accent_cyan"], "fontSize": "9px",
                    "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace",
                    "letterSpacing": "1.2px", "textTransform": "uppercase",
                    "marginBottom": "8px",
                }),

                # Multi-pair selector (used by lab TS metrics)
                html.Div([
                    html.Label("MULTI-PAIR", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-lab-pairs",
                        options=[{"label": p, "value": p}
                                 for p in sorted(FX_PAIR_REGISTRY.keys())],
                        value=["EURUSD", "USDJPY", "GBPUSD"],
                        multi=True, clearable=False,
                        style={"fontSize": "10px"}),
                ], style=SIDEBAR_SECTION),

                # Overlay metric (secondary axis)
                html.Div([
                    html.Label("OVERLAY", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-lab-overlay",
                        options=LAB_OVERLAY_OPTIONS,
                        value="", clearable=True, placeholder="2nd axis",
                        style={"fontSize": "10px"}),
                ], style=SIDEBAR_SECTION),

                # Lookback window
                html.Div([
                    html.Label("WINDOW", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-lab-window",
                        options=LAB_WINDOW_OPTIONS,
                        value=252, clearable=False,
                        style={"fontSize": "10px"}),
                ], style=SIDEBAR_SECTION),

                # Normalization
                html.Div([
                    html.Label("NORMALIZE", style=SIDEBAR_LABEL),
                    dcc.Dropdown(id="vsfx-lab-normalize",
                        options=LAB_NORMALIZE_OPTIONS,
                        value="raw", clearable=False,
                        style={"fontSize": "10px"}),
                ], style=SIDEBAR_SECTION),

            ], style=SIDEBAR_STYLE),

            # ──── Main Content Area ────────────────────────────────
            html.Div([

                # 2x2 Chart Grid
                html.Div([
                    html.Div([
                        html.Button("CSV", id="vsfx-csv-q1", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Loading(
                            dcc.Graph(id="vsfx-chart-q1",
                                      config={"displayModeBar": True, "scrollZoom": True},
                                      style={"height": f"{CHART_MD}px"}),
                            type="dot", color=COLORS["accent_cyan"],
                        ),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px",
                              "padding": "12px", "marginRight": GAP, "marginBottom": GAP}),
                    html.Div([
                        html.Button("CSV", id="vsfx-csv-q2", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Loading(
                            dcc.Graph(id="vsfx-chart-q2",
                                      config={"displayModeBar": True, "scrollZoom": True},
                                      style={"height": f"{CHART_MD}px"}),
                            type="dot", color=COLORS["accent_cyan"],
                        ),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px",
                              "padding": "12px", "marginBottom": GAP}),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": GAP}),
                html.Div([
                    html.Div([
                        html.Button("CSV", id="vsfx-csv-q3", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Loading(
                            dcc.Graph(id="vsfx-chart-q3",
                                      config={"displayModeBar": True, "scrollZoom": True},
                                      style={"height": f"{CHART_MD}px"}),
                            type="dot", color=COLORS["accent_cyan"],
                        ),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px",
                              "padding": "12px", "marginRight": GAP, "marginBottom": GAP}),
                    html.Div([
                        html.Button("CSV", id="vsfx-csv-q4", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Loading(
                            dcc.Graph(id="vsfx-chart-q4",
                                      config={"displayModeBar": True, "scrollZoom": True},
                                      style={"height": f"{CHART_MD}px"}),
                            type="dot", color=COLORS["accent_cyan"],
                        ),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px",
                              "padding": "12px", "marginBottom": GAP}),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": GAP}),

                # Stat Boxes Row (10 KPIs)
                html.Div(id="vsfx-stat-row", className="stat-row", style={
                    "display": "flex", "gap": GAP, "flexWrap": "wrap",
                    "marginBottom": "10px",
                }),

                # Overnight Summary Strip
                html.Div(id="vsfx-overnight-summary"),

                # ── Deep Study Section (from Chart Lab) ───────────
                html.Div([
                    section_header("ANALYTICAL STUDIES"),
                    html.Div([
                        html.Div([
                            html.Label("STUDY", style=SIDEBAR_LABEL),
                            dcc.Dropdown(id="vsfx-study-type", options=[
                                {"label": "Vol Deep Dive",      "value": "vol_deep_dive"},
                                {"label": "Smile Deep Dive",    "value": "smile_deep_dive"},
                                {"label": "RV Scanner",         "value": "rv_scanner"},
                                {"label": "Correlation Lab",    "value": "correlation_lab"},
                                {"label": "Carry Dashboard",    "value": "carry_dashboard"},
                                {"label": "Forward Vol Lab",    "value": "fwd_vol_lab"},
                            ], value="vol_deep_dive", clearable=False,
                            style={"fontSize": "10px"}),
                        ], style={"flex": "1", "minWidth": "140px"}),
                        html.Div([
                            html.Label("PAIR(S)", style=SIDEBAR_LABEL),
                            dcc.Dropdown(id="vsfx-study-pairs",
                                options=[{"label": p, "value": p}
                                         for p in sorted(FX_PAIR_REGISTRY.keys())],
                                value=["EURUSD", "USDJPY", "GBPUSD"],
                                multi=True, style={"fontSize": "10px"}),
                        ], style={"flex": "2", "minWidth": "200px"}),
                        html.Div([
                            html.Label("TENOR", style=SIDEBAR_LABEL),
                            dcc.Dropdown(id="vsfx-study-tenor",
                                options=[{"label": t, "value": t}
                                         for t in ["1M", "2M", "3M", "6M", "1Y", "2Y"]],
                                value="3M", clearable=False,
                                style={"fontSize": "10px"}),
                        ], style={"flex": "0.6", "minWidth": "65px"}),
                        html.Div([
                            html.Button("RUN STUDY", id="vsfx-study-run",
                                n_clicks=0, style={
                                    "backgroundColor": "#ff8800", "color": "#000",
                                    "border": "none", "borderRadius": "0px",
                                    "padding": "4px 12px",
                                    "fontFamily": "'JetBrains Mono', monospace",
                                    "fontSize": "9px", "fontWeight": "700",
                                    "cursor": "pointer", "letterSpacing": "0.8px",
                                    "textTransform": "uppercase",
                                }),
                        ], style={"display": "flex", "alignItems": "flex-end"}),
                    ], style={"display": "flex", "gap": GAP, "flexWrap": "wrap",
                              "alignItems": "flex-end", "marginBottom": GAP}),
                    dcc.Graph(id="vsfx-study-chart",
                              style={"height": f"{CHART_LG}px"},
                              config={"displayModeBar": True, "displaylogo": False}),
                ], style={**CARD_STYLE, "marginTop": SECTION_GAP,
                          "padding": "12px"}),

                # ── Comparison Overlay Section ────────────────────
                html.Div([
                    section_header("COMPARISON OVERLAY"),
                    html.Div([
                        html.Div([
                            html.Label("PAIRS (2-5)", style=SIDEBAR_LABEL),
                            dcc.Dropdown(id="vsfx-comp-pairs",
                                options=[{"label": p, "value": p}
                                         for p in sorted(FX_PAIR_REGISTRY.keys())],
                                value=["EURUSD", "USDJPY", "GBPUSD"],
                                multi=True, style={"fontSize": "10px"}),
                        ], style={"flex": "2", "minWidth": "200px"}),
                        html.Div([
                            html.Label("TYPE", style=SIDEBAR_LABEL),
                            dcc.Dropdown(id="vsfx-comp-type", options=[
                                {"label": "Term Structure",     "value": "term_structure"},
                                {"label": "Skew Profile",       "value": "skew_profile"},
                                {"label": "Smile",              "value": "smile"},
                                {"label": "Vol Cone",           "value": "vol_cone"},
                                {"label": "IV-RV Overlay",      "value": "iv_rv_overlay"},
                                {"label": "Correlation Matrix", "value": "correlation_matrix"},
                                {"label": "Vol Corr Matrix",    "value": "vol_correlation_matrix"},
                                {"label": "RV Heatmap",         "value": "rv_heatmap"},
                                {"label": "Carry Ranking",      "value": "carry_ranking"},
                            ], value="term_structure", clearable=False,
                            style={"fontSize": "10px"}),
                        ], style={"flex": "1", "minWidth": "120px"}),
                        html.Div([
                            html.Label("TENOR", style=SIDEBAR_LABEL),
                            dcc.Dropdown(id="vsfx-comp-tenor",
                                options=[{"label": t, "value": t}
                                         for t in ["1M", "2M", "3M", "6M", "1Y", "2Y"]],
                                value="3M", clearable=False,
                                style={"fontSize": "10px"}),
                        ], style={"flex": "0.6", "minWidth": "65px"}),
                        html.Div([
                            html.Button("REFRESH", id="vsfx-comp-refresh",
                                n_clicks=0, style={
                                    "backgroundColor": "#ff8800", "color": "#000",
                                    "border": "none", "borderRadius": "0px",
                                    "padding": "4px 12px",
                                    "fontFamily": "'JetBrains Mono', monospace",
                                    "fontSize": "9px", "fontWeight": "700",
                                    "cursor": "pointer", "letterSpacing": "0.8px",
                                    "textTransform": "uppercase",
                                }),
                        ], style={"display": "flex", "alignItems": "flex-end"}),
                    ], style={"display": "flex", "gap": GAP, "flexWrap": "wrap",
                              "alignItems": "flex-end", "marginBottom": GAP}),
                    dcc.Graph(id="vsfx-comp-chart",
                              style={"height": f"{CHART_LG}px"},
                              config={"displayModeBar": True, "displaylogo": False}),
                ], style={**CARD_STYLE, "marginTop": SECTION_GAP,
                          "padding": "12px"}),

            ], style={"flex": "1", "padding": SECTION_GAP, "overflowY": "auto"}),

        ], style={"display": "flex", "height": "100vh",
                  "backgroundColor": COLORS["bg_primary"]}),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app):

    # ------------------------------------------------------------------
    # Callback 1: View preset buttons set the 4 chart type dropdowns
    # ------------------------------------------------------------------
    @app.callback(
        [Output("vsfx-q1", "value"),
         Output("vsfx-q2", "value"),
         Output("vsfx-q3", "value"),
         Output("vsfx-q4", "value")],
        [Input("vsfx-preset-trader", "n_clicks"),
         Input("vsfx-preset-skew", "n_clicks"),
         Input("vsfx-preset-term", "n_clicks"),
         Input("vsfx-preset-rc", "n_clicks"),
         Input("vsfx-preset-lab-vol", "n_clicks"),
         Input("vsfx-preset-lab-smile", "n_clicks"),
         Input("vsfx-preset-lab-rv", "n_clicks"),
         Input("vsfx-preset-lab-carry", "n_clicks")],
        prevent_initial_call=True,
    )
    def update_preset(*args):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        preset_map = {
            "vsfx-preset-trader": "Trader",
            "vsfx-preset-skew": "Skew",
            "vsfx-preset-term": "Term",
            "vsfx-preset-rc": "Rich-Cheap",
            "vsfx-preset-lab-vol": "Lab: Vol Monitor",
            "vsfx-preset-lab-smile": "Lab: Smile",
            "vsfx-preset-lab-rv": "Lab: RV",
            "vsfx-preset-lab-carry": "Lab: Carry",
        }
        preset_name = preset_map.get(trigger_id, "Trader")
        charts = VIEW_PRESETS.get(preset_name, VIEW_PRESETS["Trader"])
        return charts[0], charts[1], charts[2], charts[3]

    # ------------------------------------------------------------------
    # Callback 2: Auto-refresh interval control
    # ------------------------------------------------------------------
    @app.callback(
        [Output("vsfx-interval", "interval"),
         Output("vsfx-interval", "disabled")],
        [Input("vsfx-refresh", "value")],
    )
    def set_refresh(interval_ms):
        if not interval_ms or interval_ms == 0:
            return 86400000, True
        return interval_ms, False

    # ------------------------------------------------------------------
    # Callback 3: Main update -- all 4 charts + stat boxes + overnight
    # ------------------------------------------------------------------
    @app.callback(
        [Output("vsfx-chart-q1", "figure"),
         Output("vsfx-chart-q2", "figure"),
         Output("vsfx-chart-q3", "figure"),
         Output("vsfx-chart-q4", "figure"),
         Output("vsfx-stat-row", "children"),
         Output("vsfx-overnight-summary", "children")],
        [Input("vsfx-pair", "value"),
         Input("vsfx-model", "value"),
         Input("vsfx-q1", "value"),
         Input("vsfx-q2", "value"),
         Input("vsfx-q3", "value"),
         Input("vsfx-q4", "value"),
         Input("vsfx-compare", "value"),
         Input("vsfx-hist-offset", "value"),
         Input("vsfx-cross-pair", "value"),
         Input("vsfx-tenors", "value"),
         Input("vsfx-delta-range", "value"),
         Input("vsfx-smile-tenor", "value"),
         Input("vsfx-interval", "n_intervals"),
         Input("vsfx-lab-pairs", "value"),
         Input("vsfx-lab-overlay", "value"),
         Input("vsfx-lab-window", "value"),
         Input("vsfx-lab-normalize", "value")],
    )
    def update_workstation(pair, model, q1, q2, q3, q4,
                           compare, hist_offset, cross_pair,
                           selected_tenors, delta_range, smile_tenor,
                           n_intervals,
                           lab_pairs, lab_overlay, lab_window, lab_normalize):
        pair = pair or "EURUSD"
        model = model or "market"

        # Fetch data and apply tenor / delta filters
        sd_full = _get_surface_data(pair)
        if sd_full is None:
            ndf = no_data_fig(height=CHART_MD, msg="NO VOL SURFACE DATA")
            # Still try to render lab charts that don't need surface data
            any_lab = any(q.startswith("lab_") for q in [q1, q2, q3, q4] if q)
            if not any_lab:
                return ndf, ndf, ndf, ndf, html.Div(), html.Div()
            # Create minimal placeholder sd for surface charts
            sd_full = {"tenors": [], "T_years": np.array([]),
                       "atm": np.array([]), "rr25": np.array([]),
                       "bf25": np.array([]), "rr10": np.array([]),
                       "bf10": np.array([]), "vol_grid": np.zeros((0, 5)),
                       "delta_labels": DELTA_LABELS,
                       "delta_numeric": np.array(DELTA_NUMERIC),
                       "surface_raw": {}}
        sd = _filter_surface_data(sd_full, selected_tenors)
        sd = _filter_delta_range(sd, delta_range or "10-50")
        spot, fwd_1m, r_dom, r_for = _get_spot_and_rates(pair)

        # Extra kwargs for chart functions
        extra = {
            "days_ago": hist_offset or 1,
            "smile_tenor": smile_tenor or "3M",
            "ts_tenor": smile_tenor or "3M",
            "pdf_tenor": smile_tenor or "3M",
            "compare": compare,
            "cross_pair": cross_pair,
            "model": model,
            "selected_tenors": selected_tenors,
            "delta_range": delta_range or "10-50",
            # Lab controls
            "lab_pairs": lab_pairs or [pair],
            "lab_overlay": lab_overlay or "",
            "lab_window": lab_window or 252,
            "lab_normalize": lab_normalize or "raw",
        }

        # If model is SABR or VV, overlay on smile_curve and surface_3d
        if model == "sabr":
            extra["model"] = "sabr"
        elif model == "vv":
            extra["model"] = "vv"

        # Render 4 charts
        fig1 = _render_chart(q1, pair, sd, spot, r_dom, r_for, **extra)
        fig2 = _render_chart(q2, pair, sd, spot, r_dom, r_for, **extra)
        fig3 = _render_chart(q3, pair, sd, spot, r_dom, r_for, **extra)
        fig4 = _render_chart(q4, pair, sd, spot, r_dom, r_for, **extra)

        # ── Comparison overlay support (item 5) ──
        # Apply comparison overlays to figures that are ATM term structure
        # or 3D surface views.
        try:
            if compare == "history":
                # Overlay historical ATM term structure on current figures
                offset = hist_offset or 5
                for chart_type, fig in [
                    (q1, fig1), (q2, fig2), (q3, fig3), (q4, fig4),
                ]:
                    if chart_type == "atm_term":
                        _overlay_history_on_atm(fig, pair, sd, offset)
                    elif chart_type == "heatmap":
                        _overlay_history_annotation(fig, pair, sd, offset)

            elif compare == "cross" and cross_pair and cross_pair != pair:
                # Overlay cross-pair ATM term structure
                cross_sd = _get_surface_data(cross_pair)
                if cross_sd is not None:
                    for chart_type, fig in [
                        (q1, fig1), (q2, fig2), (q3, fig3), (q4, fig4),
                    ]:
                        if chart_type == "atm_term":
                            _overlay_cross_pair_on_atm(fig, cross_pair, cross_sd)
                        elif chart_type == "surface_3d":
                            _overlay_cross_surface_wireframe(fig, cross_pair, cross_sd)
        except Exception:
            logger.exception("Comparison overlay failed for %s (compare=%s, cross_pair=%s)", pair, compare, cross_pair)

        # Build stat boxes (always use full surface for KPIs)
        stats = _build_stat_boxes(pair, sd_full, spot, fwd_1m, r_dom, r_for)

        # Build overnight summary strip
        try:
            overnight = _build_overnight_summary(pair, spot)
        except Exception:
            overnight = html.Div()

        return fig1, fig2, fig3, fig4, stats, overnight

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output("vsfx-csv-download", "data"),
        [Input("vsfx-csv-q1", "n_clicks"),
         Input("vsfx-csv-q2", "n_clicks"),
         Input("vsfx-csv-q3", "n_clicks"),
         Input("vsfx-csv-q4", "n_clicks")],
        [State("vsfx-chart-q1", "figure"),
         State("vsfx-chart-q2", "figure"),
         State("vsfx-chart-q3", "figure"),
         State("vsfx-chart-q4", "figure")],
        prevent_initial_call=True,
    )
    def vsfx_csv_export(n1, n2, n3, n4, fig1, fig2, fig3, fig4):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        mapping = {
            "vsfx-csv-q1": (fig1, "VolSurface", "Quad1"),
            "vsfx-csv-q2": (fig2, "VolSurface", "Quad2"),
            "vsfx-csv-q3": (fig3, "VolSurface", "Quad3"),
            "vsfx-csv-q4": (fig4, "VolSurface", "Quad4"),
        }
        if btn not in mapping:
            return no_update
        fig, panel, chart_type = mapping[btn]
        if not fig:
            return no_update
        try:
            return export_csv(fig, panel, chart_type)
        except Exception:
            logger.exception("CSV export failed for %s/%s", panel, chart_type)
            return no_update

    # ------------------------------------------------------------------
    # Callback 5: Deep Analytical Study
    # ------------------------------------------------------------------
    @app.callback(
        Output("vsfx-study-chart", "figure"),
        [Input("vsfx-study-run", "n_clicks")],
        [State("vsfx-study-type", "value"),
         State("vsfx-study-pairs", "value"),
         State("vsfx-study-tenor", "value")],
        prevent_initial_call=True,
    )
    def run_deep_study(n_clicks, study_type, pairs, tenor):
        if not pairs:
            return _empty_fig("Select pairs")
        if isinstance(pairs, str):
            pairs = [pairs]
        pair = pairs[0]
        tenor = tenor or "3M"

        try:
            if study_type == "vol_deep_dive":
                fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    f"{pair} ATM Vol + Regime", f"{pair} Vol Cone",
                    f"{pair} IV-RV Spread", "Stats"],
                    vertical_spacing=0.12, horizontal_spacing=0.08)
                rh = vol_regime_history(pair, 252)
                if rh is not None and not rh.empty:
                    fig.add_trace(go.Scatter(x=rh["day"].tolist(), y=rh["vol"].tolist(),
                                  mode="lines", name="ATM Vol",
                                  line=dict(color="#ff8800", width=2)), row=1, col=1)
                vc = vol_cone(pair)
                if vc is not None and not vc.empty:
                    w = vc["window"].tolist()
                    if "median" in vc.columns:
                        fig.add_trace(go.Scatter(x=w, y=vc["median"].tolist(),
                                      mode="lines", name="Median",
                                      line=dict(color="#808080", dash="dash")), row=1, col=2)
                    if "current_c2c" in vc.columns:
                        fig.add_trace(go.Scatter(x=w, y=vc["current_c2c"].tolist(),
                                      mode="lines+markers", name="Current RV",
                                      line=dict(color="#ff8800", width=2)), row=1, col=2)
                ivr = iv_rv_spread(pair, tenor, lookback=252)
                if ivr is not None and not ivr.empty:
                    if "iv" in ivr.columns:
                        fig.add_trace(go.Scatter(x=list(range(len(ivr))),
                                      y=ivr["iv"].tolist(), mode="lines", name="IV",
                                      line=dict(color="#ff8800")), row=2, col=1)
                    if "rv" in ivr.columns:
                        fig.add_trace(go.Scatter(x=list(range(len(ivr))),
                                      y=ivr["rv"].tolist(), mode="lines", name="RV",
                                      line=dict(color="#d4d4d4", dash="dash")), row=2, col=1)
                r = vol_regime_detect(pair) or {}
                z = vol_zscore(pair, tenor, "ATM") or {}
                stats_text = (f"Regime: {r.get('regime', 'N/A')}<br>"
                             f"Trend: {r.get('trend', 'N/A')}<br>"
                             f"ATM IV: {r.get('atm_iv', 0):.2f}<br>"
                             f"Z-Score: {z.get('zscore', 0):+.2f}<br>"
                             f"Percentile: {_ordinal(z.get('percentile', 50))}")
                fig.add_annotation(text=stats_text, xref="x4", yref="y4",
                                   x=0.5, y=0.5, showarrow=False,
                                   font=dict(color="#d4d4d4", size=11),
                                   align="left", row=2, col=2)

            elif study_type == "smile_deep_dive":
                fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    f"{pair} Smile ({tenor})", f"{pair} Implied PDF ({tenor})",
                    "Skew Across Tenors", "Tail Probabilities"])
                surface = get_fx_vol_surface(pair)
                if surface:
                    td = surface.get(tenor, {})
                    atm = td.get("atm", 0)
                    if atm > 0:
                        rr25, bf25 = td.get("rr25", 0), td.get("bf25", 0)
                        rr10, bf10 = td.get("rr10", 0), td.get("bf10", 0)
                        vols = [atm-rr10/2+bf10, atm-rr25/2+bf25, atm,
                                atm+rr25/2+bf25, atm+rr10/2+bf10]
                        fig.add_trace(go.Scatter(x=["10P","25P","ATM","25C","10C"],
                                      y=vols, mode="lines+markers", name="Smile",
                                      line=dict(color="#ff8800", width=2)), row=1, col=1)
                pdf = smile_implied_pdf(pair, tenor)
                if pdf is not None and not pdf.empty:
                    fig.add_trace(go.Scatter(x=pdf["strike"].tolist(),
                                  y=pdf["pdf"].tolist(), mode="lines", name="PDF",
                                  fill="tozeroy", line=dict(color="#ff8800"),
                                  fillcolor="rgba(255,136,0,0.15)"), row=1, col=2)
                if surface:
                    for t in ["1M", "3M", "6M", "1Y"]:
                        rr = surface.get(t, {}).get("rr25", 0)
                        fig.add_trace(go.Bar(x=[t], y=[rr], name=f"RR {t}",
                                      marker_color="#ff8800", showlegend=False),
                                      row=2, col=1)
                tp = tail_probabilities(pair, tenor)
                if tp is not None and not tp.empty:
                    fig.add_trace(go.Bar(
                        x=[f"{m:.0f}%" if np.isfinite(m) else "?" for m in tp["move_pct"]],
                        y=tp["prob_either"].tolist(), name="Either",
                        marker_color="#ff8800"), row=2, col=2)

            elif study_type == "rv_scanner":
                fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Vol Z-Score Heatmap", "IV-RV Spread"])
                sc = rv_scanner(pairs, ["1M", "3M", "1Y"])
                if sc is not None and not sc.empty:
                    pvt = sc.pivot_table(values="zscore", index="pair",
                                         columns="tenor", aggfunc="first")
                    if not pvt.empty:
                        fig.add_trace(go.Heatmap(
                            z=pvt.values, x=pvt.columns.tolist(),
                            y=pvt.index.tolist(),
                            colorscale=[[0,"#00cc66"],[0.5,"#000000"],[1,"#ff3333"]],
                            zmid=0, text=np.round(pvt.values, 2).astype(str),
                            texttemplate="%{text}",
                            textfont=dict(size=10)), row=1, col=1)
                    ivrv = sc.pivot_table(values="iv_rv_spread", index="pair",
                                          columns="tenor", aggfunc="first")
                    if not ivrv.empty:
                        for ci, t in enumerate(ivrv.columns):
                            fig.add_trace(go.Bar(x=ivrv.index.tolist(),
                                y=ivrv[t].tolist(), name=t,
                                marker_color=_CW[ci % len(_CW)]), row=1, col=2)

            elif study_type == "correlation_lab":
                fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Spot Correlation", "Vol Correlation"])
                sc = spot_correlation_matrix(pairs, 60)
                if sc is not None and not sc.empty:
                    fig.add_trace(go.Heatmap(
                        z=sc.values, x=sc.columns.tolist(), y=sc.index.tolist(),
                        colorscale=[[0,"#ff3333"],[0.5,"#000000"],[1,"#00cc66"]],
                        zmin=-1, zmax=1,
                        text=np.round(sc.values, 2).astype(str),
                        texttemplate="%{text}",
                        textfont=dict(size=10)), row=1, col=1)
                vc = vol_correlation_matrix(pairs, tenor, 60)
                if vc is not None and not vc.empty:
                    fig.add_trace(go.Heatmap(
                        z=vc.values, x=vc.columns.tolist(), y=vc.index.tolist(),
                        colorscale=[[0,"#ff3333"],[0.5,"#000000"],[1,"#00cc66"]],
                        zmin=-1, zmax=1,
                        text=np.round(vc.values, 2).astype(str),
                        texttemplate="%{text}",
                        textfont=dict(size=10)), row=1, col=2)

            elif study_type == "carry_dashboard":
                fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Carry / Vol Ranking", "Carry Momentum"])
                cpv = carry_per_vol(pairs)
                if cpv is not None and not cpv.empty:
                    colors = ["#00cc66" if s == "ATTRACTIVE"
                              else ("#ff8800" if s == "MODERATE" else "#ff3333")
                              for s in cpv["rank_signal"]]
                    fig.add_trace(go.Bar(x=cpv["pair"].tolist(),
                                  y=cpv["sharpe_proxy"].tolist(),
                                  marker_color=colors, name="Sharpe Proxy"),
                                  row=1, col=1)
                for idx, p in enumerate(pairs[:6]):
                    try:
                        cm = carry_momentum(p)
                        fig.add_trace(go.Bar(x=[p], y=[cm["change_20d_bps"]],
                                      name=f"{p} 20D",
                                      marker_color=_CW[idx % len(_CW)],
                                      showlegend=False), row=1, col=2)
                    except Exception:
                        pass

            elif study_type == "fwd_vol_lab":
                fig = make_subplots(rows=1, cols=2,
                    subplot_titles=[f"{pair} Forward Vol Curve",
                                    f"{pair} Forward Vol Surface"])
                fc = forward_vol_curve(pair)
                if fc is not None and not fc.empty:
                    if "spot_vol" in fc.columns:
                        fig.add_trace(go.Scatter(x=fc["end_tenor"].tolist(),
                                      y=fc["spot_vol"].tolist(),
                                      mode="lines+markers", name="Spot Vol",
                                      line=dict(color="#d4d4d4")), row=1, col=1)
                    fig.add_trace(go.Scatter(x=fc["end_tenor"].tolist(),
                                  y=fc["forward_vol"].tolist(),
                                  mode="lines+markers", name="Fwd Vol",
                                  line=dict(color="#ff8800", width=2)), row=1, col=1)
                fs = forward_vol_surface(pair)
                if fs is not None and not fs.empty:
                    fig.add_trace(go.Heatmap(
                        z=fs.values, x=fs.columns.tolist(), y=fs.index.tolist(),
                        colorscale=[[0,"#000000"],[1,"#ff8800"]],
                        text=np.where(np.isnan(fs.values), "",
                                      np.round(fs.values, 1).astype(str)),
                        texttemplate="%{text}",
                        textfont=dict(size=9)), row=1, col=2)

            else:
                return _empty_fig(f"Unknown study: {study_type}")

            if not fig.data:
                return no_data_fig(height=CHART_LG, msg="NO DATA")

            _apply_chart_template(fig, "")
            fig.update_layout(height=CHART_LG, showlegend=True,
                              hovermode="x unified",
                              margin=dict(l=50, r=20, t=45, b=35))
            for ann in fig.layout.annotations:
                ann.font = dict(size=10, color="#ff8800")
            return fig

        except Exception:
            logger.exception("Deep study error for %s", study_type)
            return _empty_fig("Error building study")

    # ------------------------------------------------------------------
    # Callback 6: Comparison Overlay
    # ------------------------------------------------------------------
    @app.callback(
        Output("vsfx-comp-chart", "figure"),
        [Input("vsfx-comp-refresh", "n_clicks"),
         Input("vsfx-comp-pairs", "value"),
         Input("vsfx-comp-type", "value"),
         Input("vsfx-comp-tenor", "value")],
    )
    def run_comparison(n_clicks, pairs, comp_type, tenor):
        if isinstance(pairs, str):
            pairs = [pairs]
        if not pairs or len(pairs) < 2:
            return _empty_fig("Select 2-5 pairs")
        pairs = pairs[:5]
        tenor = tenor or "3M"
        comp_type = comp_type or "term_structure"
        fig = go.Figure()

        try:
            if comp_type == "term_structure":
                for idx, pair in enumerate(pairs):
                    ts = get_fx_term_structure(pair)
                    if ts is not None and not ts.empty and "atm" in ts.columns:
                        fig.add_trace(go.Scatter(
                            x=ts["tenor"].tolist(), y=ts["atm"].tolist(),
                            mode="lines+markers", name=pair,
                            line=dict(color=_CW[idx % len(_CW)], width=2)))
                _apply_chart_template(fig, "ATM Vol Term Structure")
                fig.update_layout(xaxis_title="Tenor", yaxis_title="ATM Vol (%)")

            elif comp_type == "skew_profile":
                for idx, pair in enumerate(pairs):
                    surface = get_fx_vol_surface(pair)
                    if surface:
                        rr_vals, t_labels = [], []
                        for t in ["1M", "3M", "6M", "1Y"]:
                            rr_vals.append(surface.get(t, {}).get("rr25", 0))
                            t_labels.append(t)
                        fig.add_trace(go.Scatter(
                            x=t_labels, y=rr_vals, mode="lines+markers",
                            name=pair, line=dict(color=_CW[idx % len(_CW)], width=2),
                            hovertemplate="%{x}: %{y:.2f}v<extra>" + pair + "</extra>"))
                _apply_chart_template(fig, "25D RR Skew Profile")
                fig.update_layout(xaxis_title="Tenor", yaxis_title="25D RR (vol pts)")

            elif comp_type == "smile":
                for idx, pair in enumerate(pairs):
                    surface = get_fx_vol_surface(pair)
                    if surface:
                        td = surface.get(tenor, surface.get("3M", {}))
                        atm = td.get("atm", 0)
                        if atm == 0:
                            continue
                        rr25, bf25 = td.get("rr25", 0), td.get("bf25", 0)
                        rr10, bf10 = td.get("rr10", 0), td.get("bf10", 0)
                        vols = [atm-rr10/2+bf10, atm-rr25/2+bf25, atm,
                                atm+rr25/2+bf25, atm+rr10/2+bf10]
                        fig.add_trace(go.Scatter(
                            x=["10P","25P","ATM","25C","10C"], y=vols,
                            mode="lines+markers", name=f"{pair} {tenor}",
                            line=dict(color=_CW[idx % len(_CW)], width=2),
                            hovertemplate="%{x}: %{y:.2f}%<extra>" + pair + "</extra>"))
                _apply_chart_template(fig, f"{tenor} Smile Comparison")
                fig.update_layout(xaxis_title="Delta", yaxis_title="Vol (%)")

            elif comp_type == "vol_cone":
                for idx, pair in enumerate(pairs):
                    cone = vol_cone(pair)
                    if cone is not None and not cone.empty and "current_c2c" in cone.columns:
                        fig.add_trace(go.Scatter(
                            x=cone["window"].tolist(),
                            y=cone["current_c2c"].tolist(),
                            mode="lines+markers", name=pair,
                            line=dict(color=_CW[idx % len(_CW)], width=2),
                            hovertemplate="%{x}d: %{y:.2f}%<extra>" + pair + "</extra>"))
                _apply_chart_template(fig, "RV Cone: Current Level")
                fig.update_layout(xaxis_title="Window (days)", yaxis_title="RV (%)")

            elif comp_type == "iv_rv_overlay":
                for idx, pair in enumerate(pairs):
                    df = iv_rv_spread(pair, tenor, lookback=120)
                    if df is not None and not df.empty:
                        c = _CW[idx % len(_CW)]
                        x_vals = df["day"].tolist() if "day" in df.columns else list(range(len(df)))
                        if "iv" in df.columns:
                            fig.add_trace(go.Scatter(
                                x=x_vals, y=df["iv"].tolist(),
                                mode="lines", name=f"{pair} IV",
                                line=dict(color=c, width=2),
                                hovertemplate="%{x}<br>IV: %{y:.2f}%<extra>" + pair + "</extra>"))
                        if "rv" in df.columns:
                            fig.add_trace(go.Scatter(
                                x=x_vals, y=df["rv"].tolist(),
                                mode="lines", name=f"{pair} RV",
                                line=dict(color=c, width=1.5, dash="dash"),
                                hovertemplate="%{x}<br>RV: %{y:.2f}%<extra>" + pair + "</extra>"))
                _apply_chart_template(fig, f"IV vs RV ({tenor})")
                fig.update_layout(xaxis_title="Date", yaxis_title="Vol (%)")

            elif comp_type == "correlation_matrix":
                corr = spot_correlation_matrix(pairs, window=60)
                if corr is not None and not corr.empty:
                    fig = go.Figure(data=go.Heatmap(
                        z=corr.values, x=corr.columns.tolist(),
                        y=corr.index.tolist(),
                        colorscale=[[0,"#ff3333"],[0.5,"#000000"],[1,"#00cc66"]],
                        zmin=-1, zmax=1,
                        text=np.round(corr.values, 2).astype(str),
                        texttemplate="%{text}",
                        textfont=dict(size=11, color="#d4d4d4"),
                        hovertemplate="<b>%{x} vs %{y}</b><br>\u03c1 = %{z:.3f}<extra></extra>",
                        xgap=2, ygap=2))
                _apply_chart_template(fig, "60D Spot Correlation")

            elif comp_type == "vol_correlation_matrix":
                corr = vol_correlation_matrix(pairs, tenor=tenor, window=60)
                if corr is not None and not corr.empty:
                    fig = go.Figure(data=go.Heatmap(
                        z=corr.values, x=corr.columns.tolist(),
                        y=corr.index.tolist(),
                        colorscale=[[0,"#ff3333"],[0.5,"#000000"],[1,"#00cc66"]],
                        zmin=-1, zmax=1,
                        text=np.round(corr.values, 2).astype(str),
                        texttemplate="%{text}",
                        textfont=dict(size=11, color="#d4d4d4"),
                        hovertemplate="<b>%{x} vs %{y}</b><br>\u03c1 = %{z:.3f}<extra></extra>",
                        xgap=2, ygap=2))
                _apply_chart_template(fig, f"60D Vol Change Correlation ({tenor})")

            elif comp_type == "rv_heatmap":
                df = rv_scanner(pairs, ["1M", "3M", "1Y"])
                if df is not None and not df.empty:
                    pvt = df.pivot_table(values="zscore", index="pair",
                                         columns="tenor", aggfunc="first")
                    if not pvt.empty:
                        fig = go.Figure(data=go.Heatmap(
                            z=pvt.values, x=pvt.columns.tolist(),
                            y=pvt.index.tolist(),
                            colorscale=[[0,"#00cc66"],[0.5,"#000000"],[1,"#ff3333"]],
                            zmid=0, text=np.round(pvt.values, 2).astype(str),
                            texttemplate="%{text}",
                            textfont=dict(size=11, color="#d4d4d4")))
                _apply_chart_template(fig, "Vol Z-Score Heatmap (cheap=green, rich=red)")

            elif comp_type == "carry_ranking":
                df = carry_per_vol(pairs)
                if df is not None and not df.empty:
                    colors = ["#00cc66" if s == "ATTRACTIVE"
                              else ("#ff8800" if s == "MODERATE" else "#ff3333")
                              for s in df["rank_signal"]]
                    fig.add_trace(go.Bar(
                        x=df["pair"].tolist(), y=df["sharpe_proxy"].tolist(),
                        marker_color=colors,
                        text=[f"{v:.2f}" if np.isfinite(v) else "—" for v in df["sharpe_proxy"]],
                        textposition="outside",
                        textfont=dict(color="#d4d4d4", size=10)))
                _apply_chart_template(fig, "Carry / Vol Ranking")
                fig.update_layout(yaxis_title="Sharpe Proxy")

            if not fig.data:
                return no_data_fig(height=CHART_LG, msg="NO DATA")

            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0, font=dict(size=9),
                            bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=50, r=20, t=45, b=35),
                hovermode="x unified")
            return fig

        except Exception:
            logger.exception("Comparison error for %s", comp_type)
            return _empty_fig("Error building comparison")
