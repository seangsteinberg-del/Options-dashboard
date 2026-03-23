"""
FX Vol Analytics Workstation
=============================
Flagship panel for FX options vol surface analysis. Provides 14 chart types
in a configurable 2x2 grid with 10 KPI stat boxes, model selection
(Market / SABR / Vanna-Volga), view presets, comparison modes, and
auto-refresh.

Exports: layout(), register_callbacks(app)
"""

import logging

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE, AXIS_DEFAULTS,
    make_stat_style, TAB_STYLE, LABEL_STYLE, DROPDOWN_STYLE, INPUT_STYLE,
    clickable_stat,
    GAP, SECTION_GAP, CHART_SM, CHART_MD, CHART_LG,
    CSV_BTN_STYLE,
)
from core.csv_export import export_csv
from core.bloomberg_fx import (
    get_fx_vol_surface, get_fx_spots, get_fx_rates,
    get_fx_historical_vol, get_fx_realized_vol, get_all_pairs,
    get_fx_historical_spot,
)
from core.fx_analytics import (
    vol_percentile, vol_zscore, vol_regime_detect, vol_cone,
    iv_rv_spread, forward_vol_curve, smile_skewness, smile_kurtosis,
    vol_percentile_surface, vol_zscore_surface, vol_change,
    vol_surface_diff, forward_vol_surface, smile_implied_pdf,
    iv_rv_percentile,
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
DELTA_NUMERIC = [-0.10, -0.25, 0.0, 0.25, 0.10]

CHART_OPTIONS = [
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
]

VIEW_PRESETS = {
    "Trader":     ["surface_3d", "atm_term", "skew_rr", "iv_rv"],
    "Skew":       ["smile_curve", "skew_rr", "smile_bf", "rich_cheap"],
    "Term":       ["atm_term", "fwd_vol", "heatmap", "vol_cone_chart"],
    "Rich-Cheap": ["rich_cheap", "surface_change", "vol_ts", "implied_dist"],
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
    surface = get_fx_vol_surface(pair)
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
        logging.getLogger(__name__).warning("Vol surface empty for %s — using hardcoded fallback", pair)
        tenors_avail = ["1M"]
        atm_vals, rr25_vals, bf25_vals = [8.0], [0.0], [0.2]
        rr10_vals, bf10_vals = [0.0], [0.5]

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
        "delta_numeric": np.array(sorted(DELTA_NUMERIC)),
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
    spots = get_fx_spots([pair])
    spot_info = spots.get(pair, {})
    spot = spot_info.get("mid", spot_info.get("price", 1.0))
    rates = get_fx_rates(pair)
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
    delta_grid_sorted = sd["delta_numeric"]
    fig.add_trace(go.Surface(
        x=delta_grid_sorted * 100,
        y=sd["T_years"],
        z=sd["vol_grid"],
        colorscale="Plasma",
        opacity=0.92,
        colorbar=dict(
            title=dict(text="Vol %", font=dict(color=COLORS["text_muted"], size=10)),
            tickfont=dict(color=COLORS["text_muted"], size=9),
            len=0.6, thickness=12, outlinewidth=0,
        ),
        hovertemplate="Delta: %{x:.0f}<br>Tenor: %{y:.3f}y<br>Vol: %{z:.2f}%<extra></extra>",
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.3, roughness=0.5),
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Delta", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor=COLORS["border_subtle"], color=COLORS["text_muted"]),
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
    text_vals = [[f"{v:.2f}" for v in row] for row in sd["vol_grid"]]
    fig.add_trace(go.Heatmap(
        x=sd["delta_labels"],
        y=sd["tenors"],
        z=sd["vol_grid"],
        colorscale="Plasma",
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=10, color=COLORS["text_primary"]),
        hovertemplate="Delta: %{x}<br>Tenor: %{y}<br>Vol: %{z:.2f}%<extra></extra>",
        colorbar=dict(
            title=dict(text="Vol %", font=dict(color=COLORS["text_muted"], size=10)),
            tickfont=dict(color=COLORS["text_muted"], size=9),
            len=0.8, thickness=12, outlinewidth=0,
        ),
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
    ))

    # 1W ago overlay (dashed)
    try:
        hist_1w = []
        for t in tenors:
            ch = vol_change(pair, t, "ATM", days_ago=5)
            hist_1w.append(ch["previous"])
        fig.add_trace(go.Scatter(
            x=tenors, y=hist_1w, mode="lines",
            name="1W ago", line=dict(color=COLORS["accent_blue"], width=1.5, dash="dash"),
        ))
    except Exception:
        pass

    # 1M ago overlay (dotted)
    try:
        hist_1m = []
        for t in tenors:
            ch = vol_change(pair, t, "ATM", days_ago=22)
            hist_1m.append(ch["previous"])
        fig.add_trace(go.Scatter(
            x=tenors, y=hist_1m, mode="lines",
            name="1M ago", line=dict(color=COLORS["accent_purple"], width=1.5, dash="dot"),
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
                    gridcolor="rgba(30,42,69,0.3)",
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
        try:
            p = vol_percentile(pair, t, "25D_RR")
            pctiles.append(p["percentile"])
        except Exception:
            pctiles.append(50.0)

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
        text=[f"{v:+.2f}" for v in rr_vals],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_secondary"]),
        hovertemplate="Tenor: %{x}<br>25D RR: %{y:.2f}<br>%ile: %{customdata:.0f}<extra></extra>",
        customdata=pctiles,
    ))

    # 1Y range whiskers
    for i, t in enumerate(tenors):
        try:
            p = vol_percentile(pair, t, "25D_RR")
            fig.add_shape(type="line",
                x0=i, x1=i, y0=p["min"], y1=p["max"],
                line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
                xref="x", yref="y")
        except Exception:
            pass

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
        try:
            p = vol_percentile(pair, t, "25D_BF")
            pctiles.append(p["percentile"])
        except Exception:
            pctiles.append(50.0)

    colors = []
    for pct in pctiles:
        r = int(min(255, pct * 2.55))
        b = int(min(255, (100 - pct) * 2.55))
        colors.append(f"rgb({r},60,{b})")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tenors, y=bf_vals, name="25D BF",
        marker=dict(color=colors, line=dict(width=1, color=COLORS["border"])),
        text=[f"{v:.2f}" for v in bf_vals],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_secondary"]),
        hovertemplate="Tenor: %{x}<br>25D BF: %{y:.2f}<br>%ile: %{customdata:.0f}<extra></extra>",
        customdata=pctiles,
    ))

    for i, t in enumerate(tenors):
        try:
            p = vol_percentile(pair, t, "25D_BF")
            fig.add_shape(type="line",
                x0=i, x1=i, y0=p["min"], y1=p["max"],
                line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
                xref="x", yref="y")
        except Exception:
            pass

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
            try:
                p = vol_percentile(pair, t, m)
                row_z.append(p["percentile"])
                row_txt.append(f"{p['percentile']:.0f}%ile")
            except Exception:
                row_z.append(50.0)
                row_txt.append("--")
        z_data.append(row_z)
        text_data.append(row_txt)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=metric_labels,
        y=tenors,
        z=z_data,
        colorscale=[[0, "#3b82f6"], [0.5, "#1e293b"], [1.0, "#ef4444"]],
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=11, color=COLORS["text_primary"]),
        hovertemplate="Metric: %{x}<br>Tenor: %{y}<br>Percentile: %{z:.0f}<extra></extra>",
        colorbar=dict(
            title=dict(text="%ile", font=dict(color=COLORS["text_muted"], size=10)),
            tickfont=dict(color=COLORS["text_muted"], size=9),
            len=0.8, thickness=12, outlinewidth=0,
        ),
        zmin=0, zmax=100,
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
    atm = row.get("atm", row.get("ATM", 8.0))
    rr25 = row.get("rr25", row.get("25D_RR", 0.0))
    bf25 = row.get("bf25", row.get("25D_BF", 0.2))
    rr10 = row.get("rr10", row.get("10D_RR", 0.0))
    bf10 = row.get("bf10", row.get("10D_BF", 0.5))

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
        delta_fine = np.array(sorted([-0.10, -0.25, 0.0, 0.25, 0.10]))
        vol_fine = np.array(sorted(pillar_vols))

    fig = go.Figure()

    # Spline fit line
    fig.add_trace(go.Scatter(
        x=delta_fine * 100 if delta_fine.max() <= 1.0 else delta_fine,
        y=vol_fine,
        mode="lines", name="Spline Fit",
        line=dict(color=COLORS["accent_cyan"], width=2.5),
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
            ))
        except Exception:
            pass

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
            ))
        except Exception:
            pass

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
        hist = np.random.RandomState(abs(hash(pair)) % 2**31).normal(8, 1.5, 252)
        hist = np.clip(np.cumsum(np.random.RandomState(abs(hash(pair)) % 2**31).normal(0, 0.1, 252)) + 8, 3, 25)

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
        fill="toself", fillcolor="rgba(6,182,212,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Upper/lower bands
    fig.add_trace(go.Scatter(x=days, y=upper, mode="lines", name="BB Upper",
        line=dict(color=COLORS["accent_cyan"], width=1, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=lower, mode="lines", name="BB Lower",
        line=dict(color=COLORS["accent_cyan"], width=1, dash="dot"), showlegend=False))

    # MA line
    fig.add_trace(go.Scatter(x=days, y=ma20, mode="lines", name="20d MA",
        line=dict(color=COLORS["accent_blue"], width=1.5, dash="dash")))

    # ATM vol line
    fig.add_trace(go.Scatter(x=days, y=hist, mode="lines", name=f"ATM {sel_tenor}",
        line=dict(color=COLORS["accent_cyan"], width=2)))

    _apply_chart_template(fig, f"ATM Vol Time Series -- {pair} {sel_tenor}")
    fig.update_layout(xaxis=dict(title="Days"), yaxis=dict(title="Vol (%)"))
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
        fillcolor="rgba(244,63,94,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(x=days, y=iv_arr, mode="lines", name="ATM IV 3M",
        line=dict(color=COLORS["accent_cyan"], width=2.5)))
    fig.add_trace(go.Scatter(x=days, y=rv_arr, mode="lines", name="RV 20d",
        line=dict(color=COLORS["accent_rose"], width=2)))

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
            gridcolor="rgba(30,42,69,0.15)",
            tickfont=dict(size=9, color=COLORS["text_muted"]),
            title_font=dict(color=COLORS["text_muted"], size=10),
        ),
    )

    # Zero line for spread
    fig.add_hline(y=0, line=dict(color=COLORS["border"], width=0.5))

    _apply_chart_template(fig, f"IV vs Realized Vol -- {pair}")
    fig.update_layout(xaxis=dict(title="Days"), yaxis=dict(title="Vol (%)"))
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
        ("p10", "p90", "rgba(6,182,212,0.06)", "10-90%ile"),
        ("p25", "p75", "rgba(6,182,212,0.12)", "25-75%ile"),
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
        name="Median", line=dict(color=COLORS["accent_blue"], width=1.5, dash="dash")))

    # Min/Max
    fig.add_trace(go.Scatter(x=windows, y=cone_df["min"], mode="lines",
        name="Min", line=dict(color=COLORS["accent_green"], width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=windows, y=cone_df["max"], mode="lines",
        name="Max", line=dict(color=COLORS["accent_red"], width=1, dash="dot")))

    # Current RV
    fig.add_trace(go.Scatter(x=windows, y=cone_df["current_c2c"], mode="lines+markers",
        name="Current RV", line=dict(color=COLORS["accent_cyan"], width=2.5),
        marker=dict(size=7, color=COLORS["accent_cyan"])))

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
    ))
    fig.add_trace(go.Scatter(
        x=df["end_tenor"], y=df["spot_vol"], mode="lines+markers",
        name="Spot Vol",
        line=dict(color=COLORS["accent_cyan"], width=2, dash="dash"),
        marker=dict(size=5, color=COLORS["accent_cyan"]),
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

    text_vals = [[f"{v:+.2f}" for v in row] for row in z]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=deltas,
        y=tenors,
        z=z,
        colorscale=[[0, "#10b981"], [0.5, "#1e293b"], [1.0, "#ef4444"]],
        zmid=0,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=10, color=COLORS["text_primary"]),
        hovertemplate="Delta: %{x}<br>Tenor: %{y}<br>Change: %{z:+.2f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Vol Chg", font=dict(color=COLORS["text_muted"], size=10)),
            tickfont=dict(color=COLORS["text_muted"], size=9),
            len=0.8, thickness=12, outlinewidth=0,
        ),
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
        atm_vol = row.get("atm", row.get("ATM", 8.0)) / 100.0
        rr25 = row.get("rr25", row.get("25D_RR", 0.0)) / 100.0
        bf25 = row.get("bf25", row.get("25D_BF", 0.2)) / 100.0

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
        marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tenors, y=rhos, mode="lines+markers", name="Rho",
        line=dict(color=COLORS["accent_rose"], width=2),
        marker=dict(size=6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=tenors, y=nus, mode="lines+markers", name="Nu",
        line=dict(color=COLORS["accent_orange"], width=2),
        marker=dict(size=6)), row=3, col=1)

    _apply_chart_template(fig, f"SABR Parameters -- {pair}")
    fig.update_layout(height=400, showlegend=False)
    for i in range(1, 4):
        fig.update_yaxes(gridcolor="rgba(30,42,69,0.5)", row=i, col=1)
        fig.update_xaxes(gridcolor="rgba(30,42,69,0.5)", row=i, col=1)
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
        # Synthetic fallback: log-normal approximation
        T = tenor_to_years(sel_tenor)
        atm_vol = sd["atm"][len(sd["atm"]) // 2] / 100.0 if len(sd["atm"]) > 0 else 0.08
        F = spot * np.exp((r_dom - r_for) * T)
        sigma_total = atm_vol * np.sqrt(T)
        strikes = np.linspace(F * 0.7, F * 1.3, 150)
        log_FK = np.log(strikes / F)
        pdf_vals = (1.0 / (strikes * sigma_total * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((log_FK + 0.5 * sigma_total ** 2) / sigma_total) ** 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strikes, y=pdf_vals, mode="lines", name="Risk-Neutral PDF",
        fill="tozeroy", fillcolor="rgba(6,182,212,0.15)",
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


# Chart dispatch table
CHART_DISPATCH = {
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
}


def _render_chart(chart_type, pair, sd, spot, r_dom, r_for, **kw):
    """Dispatch to the correct chart function."""
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
    atm_1m = sd["atm"][sd["tenors"].index("1M")] if "1M" in sd["tenors"] else (sd["atm"][0] if len(sd["atm"]) > 0 else 0)
    atm_1y = sd["atm"][sd["tenors"].index("1Y")] if "1Y" in sd["tenors"] else (sd["atm"][-1] if len(sd["atm"]) > 0 else 0)

    # ATM 1M change
    atm_1m_delta = 0.0
    try:
        ch = vol_change(pair, "1M", "ATM", days_ago=1)
        atm_1m_delta = ch["abs_change"]
    except Exception:
        pass

    # Vol percentile for ATM 1M
    atm_pctile = 50.0
    try:
        p = vol_percentile(pair, "1M", "ATM")
        atm_pctile = p["percentile"]
    except Exception:
        pass

    # 25D RR 3M + percentile
    rr_3m = 0.0
    rr_pctile = 50.0
    try:
        idx = sd["tenors"].index("3M") if "3M" in sd["tenors"] else min(2, len(sd["tenors"]) - 1)
        rr_3m = sd["rr25"][idx] if 0 <= idx < len(sd.get("rr25", [])) else 0.0
        p = vol_percentile(pair, "3M", "25D_RR")
        rr_pctile = p["percentile"]
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
        iv_rv_spr = irp["current_spread"]
    except Exception:
        pass

    # Term spread 1Y-1M
    term_spr = atm_1y - atm_1m

    # Skew percentile
    skew_pctile = 50.0
    try:
        p = vol_percentile(pair, "3M", "25D_RR")
        skew_pctile = p["percentile"]
    except Exception:
        pass

    # Regime
    regime_text = "NORMAL"
    regime_color = COLORS["accent_blue"]
    try:
        reg = vol_regime_detect(pair)
        regime_text = reg["regime"]
        regime_color = reg["color"]
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
        vol_chg_bp = ch["abs_change"]
        vol_current = ch["current"]
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
        rr_chg = rr_ch["abs_change"]
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
        "borderRadius": "8px",
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
    delta_grid_sorted = cross_sd["delta_numeric"]
    fig.add_trace(go.Surface(
        x=delta_grid_sorted * 100,
        y=cross_sd["T_years"],
        z=cross_sd["vol_grid"],
        colorscale="Viridis",
        opacity=0.35,
        showscale=False,
        name=f"{cross_pair}",
        hovertemplate=(f"{cross_pair}<br>"
                       "Delta: %{x:.0f}<br>Tenor: %{y:.3f}y<br>Vol: %{z:.2f}%<extra></extra>"),
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
         Input("vsfx-preset-rc", "n_clicks")],
        prevent_initial_call=True,
    )
    def update_preset(trader_n, skew_n, term_n, rc_n):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        preset_map = {
            "vsfx-preset-trader": "Trader",
            "vsfx-preset-skew": "Skew",
            "vsfx-preset-term": "Term",
            "vsfx-preset-rc": "Rich-Cheap",
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
            return 60000, True
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
         Input("vsfx-interval", "n_intervals")],
    )
    def update_workstation(pair, model, q1, q2, q3, q4,
                           compare, hist_offset, cross_pair,
                           selected_tenors, delta_range, n_intervals):
        pair = pair or "EURUSD"
        model = model or "market"

        # Fetch data and apply tenor / delta filters
        sd_full = _get_surface_data(pair)
        sd = _filter_surface_data(sd_full, selected_tenors)
        sd = _filter_delta_range(sd, delta_range or "10-50")
        spot, fwd_1m, r_dom, r_for = _get_spot_and_rates(pair)

        # Extra kwargs for chart functions
        extra = {
            "days_ago": hist_offset or 1,
            "smile_tenor": "3M",
            "ts_tenor": "3M",
            "pdf_tenor": "3M",
            "compare": compare,
            "cross_pair": cross_pair,
            "model": model,
            "selected_tenors": selected_tenors,
            "delta_range": delta_range or "10-50",
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
                for chart_type, fig in [
                    (q1, fig1), (q2, fig2), (q3, fig3), (q4, fig4),
                ]:
                    if chart_type == "atm_term":
                        _overlay_cross_pair_on_atm(fig, cross_pair, cross_sd)
                    elif chart_type == "surface_3d":
                        _overlay_cross_surface_wireframe(fig, cross_pair, cross_sd)
        except Exception:
            logger.exception("Comparison overlay failed for %s", pair)

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
        return export_csv(fig, panel, chart_type)
