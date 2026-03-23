"""
Chart Lab — Analytical Workshop
=================================
A standalone analytical workstation for FX vol traders. Four configurable
chart slots with 21 metrics (11 time-series + 10 analytical studies),
a deep-dive study panel, 9 comparison overlay types, 12 one-click presets,
cross-slot sync, auto-refresh, persistent pinned charts, and scratchpad.

Every analytics function in the engine is surfaced here.

Exports layout() and register_callbacks(app).
"""

import logging, traceback, json
from datetime import datetime

import dash
from dash import html, dcc, Input, Output, State, callback_context, ALL, MATCH, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    make_stat_style, grid_cell, section_header, chart_layout,
    GAP, SECTION_GAP, CHART_SM, CHART_MD, CHART_LG, CSV_BTN_STYLE,
)
from core.csv_export import export_csv
from core.bloomberg_fx import (
    get_fx_spots, get_fx_vol_surface, get_fx_rates, get_all_pairs,
    get_fx_historical_spot, get_fx_historical_vol, get_fx_term_structure,
)
from core.fx_analytics import (
    vol_percentile, vol_zscore, vol_percentile_surface, vol_zscore_surface,
    vol_change, vol_regime_detect, vol_regime_history, vol_cone,
    iv_rv_spread, iv_rv_percentile, breakeven_vol,
    forward_vol_curve, forward_vol_surface,
    smile_skewness, smile_kurtosis, wing_richness,
    smile_implied_pdf, tail_probabilities,
    carry_table, carry_per_vol, carry_momentum, rate_differential_history,
    spot_correlation_matrix, vol_correlation_matrix, spot_vol_correlation,
    correlation_regime,
    rv_scanner, rv_signal_composite,
)
from core.fx_conventions import FX_PAIR_REGISTRY

logger = logging.getLogger(__name__)

# ============================================================================
#  Constants
# ============================================================================

ALL_PAIRS = sorted(FX_PAIR_REGISTRY.keys())
DEFAULT_PAIRS = ["EURUSD", "USDJPY", "GBPUSD"]
_FONT = "'JetBrains Mono', monospace"
NUM_SLOTS = 4

METRIC_OPTIONS = [
    {"label": "─── TIME SERIES ───", "value": "_ts", "disabled": True},
    {"label": "ATM Vol",        "value": "ATM"},
    {"label": "25D RR",         "value": "25D_RR"},
    {"label": "25D BF",         "value": "25D_BF"},
    {"label": "10D RR",         "value": "10D_RR"},
    {"label": "10D BF",         "value": "10D_BF"},
    {"label": "Spot",           "value": "SPOT"},
    {"label": "IV-RV Spread",   "value": "IV_RV"},
    {"label": "Forward Vol",    "value": "FWD_VOL"},
    {"label": "Realized Vol",   "value": "RV"},
    {"label": "Term Spread",    "value": "TERM_SPREAD"},
    {"label": "Carry",          "value": "CARRY"},
    {"label": "─── STUDIES ───", "value": "_st", "disabled": True},
    {"label": "\u2022 Vol Cone",           "value": "STUDY_VOL_CONE"},
    {"label": "\u2022 Vol Smile",          "value": "STUDY_SMILE"},
    {"label": "\u2022 Implied PDF",        "value": "STUDY_IMPLIED_PDF"},
    {"label": "\u2022 Vol Regime",         "value": "STUDY_VOL_REGIME"},
    {"label": "\u2022 Fwd Vol Curve",      "value": "STUDY_FWD_VOL_CURVE"},
    {"label": "\u2022 Percentile Sfc",     "value": "STUDY_PCTILE_SURFACE"},
    {"label": "\u2022 Z-Score Surface",    "value": "STUDY_ZSCORE_SURFACE"},
    {"label": "\u2022 Tail Probabilities", "value": "STUDY_TAIL_PROBS"},
    {"label": "\u2022 Breakeven Vol",      "value": "STUDY_BREAKEVEN"},
    {"label": "\u2022 Carry Landscape",    "value": "STUDY_CARRY_LANDSCAPE"},
]

TENOR_OPTIONS = [{"label": t, "value": t} for t in ["1M", "2M", "3M", "6M", "1Y", "2Y"]]
TIMEFRAME_OPTIONS = [
    {"label": "30d", "value": 30}, {"label": "60d", "value": 60},
    {"label": "120d", "value": 120}, {"label": "252d", "value": 252},
    {"label": "504d", "value": 504},
]
CHART_TYPE_OPTIONS = [
    {"label": "Line", "value": "line"}, {"label": "Area", "value": "area"},
    {"label": "Bar", "value": "bar"},
]
NORMALIZE_OPTIONS = [
    {"label": "Raw", "value": "raw"}, {"label": "Indexed (=100)", "value": "indexed"},
    {"label": "Z-Score", "value": "zscore"}, {"label": "% Change", "value": "pct_change"},
]
OVERLAY_OPTIONS = [{"label": "— None —", "value": ""}] + [
    o for o in [
        {"label": "ATM Vol",      "value": "ATM"},   {"label": "25D RR",     "value": "25D_RR"},
        {"label": "25D BF",       "value": "25D_BF"}, {"label": "10D RR",     "value": "10D_RR"},
        {"label": "10D BF",       "value": "10D_BF"}, {"label": "Spot",       "value": "SPOT"},
        {"label": "IV-RV Spread", "value": "IV_RV"},  {"label": "Forward Vol","value": "FWD_VOL"},
        {"label": "Realized Vol", "value": "RV"},     {"label": "Term Spread","value": "TERM_SPREAD"},
        {"label": "Carry",        "value": "CARRY"},
    ]
]
COMPARISON_TYPE_OPTIONS = [
    {"label": "Term Structure",     "value": "term_structure"},
    {"label": "Skew Profile",       "value": "skew_profile"},
    {"label": "Smile",              "value": "smile"},
    {"label": "Vol Cone",           "value": "vol_cone"},
    {"label": "IV-RV Overlay",      "value": "iv_rv_overlay"},
    {"label": "Correlation Matrix", "value": "correlation_matrix"},
    {"label": "Vol Corr Matrix",    "value": "vol_correlation_matrix"},
    {"label": "RV Heatmap",         "value": "rv_heatmap"},
    {"label": "Carry Ranking",      "value": "carry_ranking"},
]
DEEP_STUDY_OPTIONS = [
    {"label": "Vol Deep Dive",      "value": "vol_deep_dive"},
    {"label": "Smile Deep Dive",    "value": "smile_deep_dive"},
    {"label": "RV Scanner",         "value": "rv_scanner"},
    {"label": "Correlation Lab",    "value": "correlation_lab"},
    {"label": "Carry Dashboard",    "value": "carry_dashboard"},
    {"label": "Forward Vol Lab",    "value": "fwd_vol_lab"},
]
PAIR_OPTIONS = [{"label": p, "value": p} for p in ALL_PAIRS]

# ============================================================================
#  Presets
# ============================================================================

PRESETS = {
    "g10_vol_monitor": {"label": "G10 Vol Monitor", "layout": 4, "slots": [
        {"metric": "ATM", "pairs": ["EURUSD", "USDJPY"], "tenor": "1M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "ATM", "pairs": ["GBPUSD", "AUDUSD"], "tenor": "1M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "ATM", "pairs": ["USDCAD", "USDCHF"], "tenor": "1M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "ATM", "pairs": ["NZDUSD"],            "tenor": "1M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
    ]},
    "em_vs_g10": {"label": "EM vs G10", "layout": 2, "slots": [
        {"metric": "ATM", "pairs": ["USDMXN", "USDZAR", "USDTRY", "USDBRL"], "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "indexed"},
        {"metric": "ATM", "pairs": ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"], "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "indexed"},
    ]},
    "skew_dashboard": {"label": "Skew Dashboard", "layout": 4, "slots": [
        {"metric": "25D_RR", "pairs": ["EURUSD", "USDJPY"], "tenor": "3M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
        {"metric": "25D_RR", "pairs": ["GBPUSD", "AUDUSD"], "tenor": "3M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
        {"metric": "25D_RR", "pairs": ["USDCAD", "USDCHF"], "tenor": "3M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
        {"metric": "25D_RR", "pairs": ["EURUSD", "USDJPY", "GBPUSD"], "tenor": "3M", "timeframe": 120, "chart_type": "line", "normalize": "indexed"},
    ]},
    "iv_rv_screen": {"label": "IV-RV Screen", "layout": 2, "slots": [
        {"metric": "ATM", "overlay": "RV", "pairs": ["EURUSD", "USDJPY"], "tenor": "3M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
        {"metric": "ATM", "overlay": "RV", "pairs": ["GBPUSD", "AUDUSD"], "tenor": "3M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
    ]},
    "term_spread": {"label": "Term Spread", "layout": 2, "slots": [
        {"metric": "TERM_SPREAD", "pairs": ["EURUSD", "USDJPY", "GBPUSD"], "tenor": "1M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "TERM_SPREAD", "pairs": ["AUDUSD", "USDCAD"],           "tenor": "1M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
    ]},
    "carry_landscape": {"label": "Carry Landscape", "layout": 1, "slots": [
        {"metric": "STUDY_CARRY_LANDSCAPE", "pairs": ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "USDMXN"], "tenor": "3M", "timeframe": 252, "chart_type": "bar", "normalize": "raw"},
    ]},
    "jpy_complex": {"label": "JPY Complex", "layout": 4, "slots": [
        {"metric": "ATM", "pairs": ["USDJPY"],  "tenor": "1M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
        {"metric": "ATM", "pairs": ["EURJPY"],  "tenor": "1M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
        {"metric": "ATM", "pairs": ["GBPJPY"],  "tenor": "1M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
        {"metric": "ATM", "pairs": ["AUDJPY"],  "tenor": "1M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
    ]},
    "risk_barometer": {"label": "Risk Barometer", "layout": 2, "slots": [
        {"metric": "ATM", "overlay": "SPOT", "pairs": ["EURUSD", "USDJPY"], "tenor": "1M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
        {"metric": "ATM", "overlay": "RV",   "pairs": ["EURUSD", "USDJPY"], "tenor": "1M", "timeframe": 120, "chart_type": "line", "normalize": "raw"},
    ]},
    "vol_regime_check": {"label": "Vol Regime Check", "layout": 4, "slots": [
        {"metric": "STUDY_VOL_REGIME", "pairs": ["EURUSD"],  "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "STUDY_VOL_CONE",   "pairs": ["EURUSD"],  "tenor": "3M", "timeframe": 504, "chart_type": "line", "normalize": "raw"},
        {"metric": "IV_RV",            "pairs": ["EURUSD"],  "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "STUDY_SMILE",      "pairs": ["EURUSD"],  "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
    ]},
    "smile_lab": {"label": "Smile Lab", "layout": 4, "slots": [
        {"metric": "STUDY_SMILE",        "pairs": ["EURUSD"], "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "STUDY_IMPLIED_PDF",  "pairs": ["EURUSD"], "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "STUDY_TAIL_PROBS",   "pairs": ["EURUSD"], "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "25D_RR",             "pairs": ["EURUSD", "USDJPY", "GBPUSD"], "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
    ]},
    "surface_scanner": {"label": "Surface Scanner", "layout": 4, "slots": [
        {"metric": "STUDY_PCTILE_SURFACE",  "pairs": ["EURUSD"], "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "STUDY_ZSCORE_SURFACE",  "pairs": ["EURUSD"], "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "STUDY_PCTILE_SURFACE",  "pairs": ["USDJPY"], "tenor": "3M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
        {"metric": "STUDY_FWD_VOL_CURVE",   "pairs": ["EURUSD"], "tenor": "1M", "timeframe": 252, "chart_type": "line", "normalize": "raw"},
    ]},
    "morning_briefing": {"label": "Morning Briefing", "layout": 4, "slots": [
        {"metric": "ATM",              "pairs": ["EURUSD", "USDJPY", "GBPUSD"], "tenor": "1M", "timeframe": 30, "chart_type": "line", "normalize": "raw"},
        {"metric": "25D_RR",           "pairs": ["EURUSD", "USDJPY", "GBPUSD"], "tenor": "3M", "timeframe": 30, "chart_type": "line", "normalize": "raw"},
        {"metric": "STUDY_VOL_REGIME", "pairs": ["EURUSD"],                      "tenor": "3M", "timeframe": 60, "chart_type": "line", "normalize": "raw"},
        {"metric": "IV_RV",            "pairs": ["EURUSD", "USDJPY"],            "tenor": "3M", "timeframe": 60, "chart_type": "line", "normalize": "raw"},
    ]},
}

PRESET_OPTIONS = [{"label": "-- Select Preset --", "value": ""}] + [
    {"label": v["label"], "value": k} for k, v in PRESETS.items()
]

# ============================================================================
#  Styles
# ============================================================================

LABEL_STYLE = {"color": "#808080", "fontSize": "9px", "fontWeight": "600", "fontFamily": _FONT,
               "textTransform": "uppercase", "letterSpacing": "1.2px", "marginBottom": "2px", "display": "block"}
SLOT_STYLE = {"backgroundColor": "#000000", "border": "1px solid #222240", "borderRadius": "0px", "padding": "6px"}
CTRL_WRAP = {"display": "flex", "gap": GAP, "flexWrap": "wrap", "alignItems": "flex-end", "marginBottom": GAP}
BTN_STYLE = {"backgroundColor": "#ff8800", "color": "#000000", "border": "none", "borderRadius": "0px",
             "padding": "4px 12px", "fontFamily": _FONT, "fontSize": "9px", "fontWeight": "700",
             "cursor": "pointer", "letterSpacing": "0.8px", "textTransform": "uppercase"}
BTN_TOGGLE_STYLE = {"backgroundColor": "#000000", "color": "#808080", "border": "1px solid #222240",
                    "borderRadius": "0px", "padding": "4px 10px", "fontFamily": _FONT, "fontSize": "9px",
                    "fontWeight": "700", "cursor": "pointer", "letterSpacing": "0.8px",
                    "textTransform": "uppercase", "marginRight": "4px"}
BTN_TOGGLE_ACTIVE = {**BTN_TOGGLE_STYLE, "backgroundColor": "#ff8800", "color": "#000000", "border": "1px solid #ff8800"}
BTN_PIN = {"backgroundColor": "transparent", "color": "#808080", "border": "1px solid #222240",
           "borderRadius": "0px", "padding": "2px 6px", "fontFamily": _FONT, "fontSize": "10px",
           "cursor": "pointer", "marginLeft": "auto"}
TEXTAREA_STYLE = {"backgroundColor": "#000000", "color": "#d4d4d4", "border": "1px solid #222240",
                  "borderRadius": "0px", "fontFamily": _FONT, "fontSize": "11px", "width": "100%",
                  "minHeight": "100px", "padding": "8px", "resize": "vertical"}
GRAPH_CONFIG = {"displayModeBar": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d",
                                           "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                "toImageButtonOptions": {"format": "png", "height": 600, "width": 1000}}
STAT_STRIP = {"display": "flex", "gap": "12px", "padding": f"{GAP} {GAP}", "borderTop": "1px solid #222240",
              "marginTop": GAP, "flexWrap": "wrap"}
STAT_ITEM = {"fontFamily": _FONT, "fontSize": "9px", "color": "#808080", "letterSpacing": "0.5px"}
STAT_VAL = {"fontFamily": _FONT, "fontSize": "10px", "fontWeight": "700", "marginLeft": "4px"}


# ============================================================================
#  Chart Slot Builder
# ============================================================================

def _chart_slot(i):
    """Return layout for chart slot i (0-3)."""
    defaults_m = ["ATM", "25D_RR", "SPOT", "IV_RV"]
    defaults_t = ["1M", "3M", "1M", "3M"]
    default_pair = DEFAULT_PAIRS[i % len(DEFAULT_PAIRS)]

    return html.Div(id={"type": "lab-slot-wrapper", "index": i}, children=[
        # Header: SLOT N | summary | pin
        html.Div([
            html.Span(f"SLOT {i+1}", style={"color": "#ff8800", "fontSize": "9px", "fontWeight": "700",
                                              "fontFamily": _FONT, "letterSpacing": "1.5px"}),
            html.Span(id={"type": "lab-slot-summary", "index": i}, children="",
                      style={"color": "#808080", "fontSize": "9px", "fontFamily": _FONT, "marginLeft": "8px"}),
            html.Button("\U0001F4CC", id={"type": "lab-pin-btn", "index": i}, n_clicks=0,
                        style=BTN_PIN, title="Pin this chart config"),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": GAP}),
        # Controls row
        html.Div([
            html.Div([html.Label("METRIC", style=LABEL_STYLE),
                       dcc.Dropdown(id={"type": "lab-metric", "index": i}, options=METRIC_OPTIONS,
                                    value=defaults_m[i], clearable=False, style={"fontSize": "10px"})],
                     style={"flex": "1.2", "minWidth": "120px"}),
            html.Div([html.Label("OVERLAY (R)", style=LABEL_STYLE),
                       dcc.Dropdown(id={"type": "lab-overlay", "index": i}, options=OVERLAY_OPTIONS,
                                    value="", clearable=True, placeholder="2nd axis",
                                    style={"fontSize": "10px"})],
                     style={"flex": "1", "minWidth": "100px"}),
            html.Div([html.Label("TENOR", style=LABEL_STYLE),
                       dcc.Dropdown(id={"type": "lab-tenor", "index": i}, options=TENOR_OPTIONS,
                                    value=defaults_t[i], clearable=False, style={"fontSize": "10px"})],
                     style={"flex": "0.6", "minWidth": "65px"}),
            html.Div([html.Label("PAIRS", style=LABEL_STYLE),
                       dcc.Dropdown(id={"type": "lab-pairs", "index": i}, options=PAIR_OPTIONS,
                                    value=[default_pair], multi=True, style={"fontSize": "10px"})],
                     style={"flex": "2", "minWidth": "160px"}),
            html.Div([html.Label("WINDOW", style=LABEL_STYLE),
                       dcc.Dropdown(id={"type": "lab-timeframe", "index": i}, options=TIMEFRAME_OPTIONS,
                                    value=252, clearable=False, style={"fontSize": "10px"})],
                     style={"flex": "0.6", "minWidth": "65px"}),
            html.Div([html.Label("TYPE", style=LABEL_STYLE),
                       dcc.Dropdown(id={"type": "lab-charttype", "index": i}, options=CHART_TYPE_OPTIONS,
                                    value="line", clearable=False, style={"fontSize": "10px"})],
                     style={"flex": "0.5", "minWidth": "55px"}),
            html.Div([html.Label("NORM", style=LABEL_STYLE),
                       dcc.Dropdown(id={"type": "lab-normalize", "index": i}, options=NORMALIZE_OPTIONS,
                                    value="raw", clearable=False, style={"fontSize": "10px"})],
                     style={"flex": "0.6", "minWidth": "70px"}),
        ], style=CTRL_WRAP),
        # Chart
        html.Button("CSV", id={"type": "lab-csv-btn", "index": i}, n_clicks=0, style=CSV_BTN_STYLE),
        dcc.Graph(id={"type": "lab-chart", "index": i}, style={"height": f"{CHART_MD}px"}, config=GRAPH_CONFIG),
        # Stats strip
        html.Div(id={"type": "lab-slot-stats", "index": i}, children=[], style=STAT_STRIP),
    ], style=SLOT_STYLE)


# ============================================================================
#  Layout
# ============================================================================

def layout():
    return html.Div([
        dcc.Download(id="lab-csv-download"),
        # ── Title bar + controls ─────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("CHART LAB", style={"color": "#ff8800", "fontSize": "13px", "fontWeight": "700",
                                               "fontFamily": _FONT, "letterSpacing": "2px"}),
                html.Span("  |  Analytical Workshop", style={"color": "#808080", "fontSize": "10px",
                                                              "fontFamily": _FONT}),
            ], style={"flex": "1"}),
            # Preset
            html.Div([
                html.Label("PRESET", style={**LABEL_STYLE, "marginRight": "4px"}),
                dcc.Dropdown(id="lab-preset-dropdown", options=PRESET_OPTIONS, value="",
                             clearable=False, style={"fontSize": "10px", "width": "160px"}),
            ], style={"display": "flex", "alignItems": "center", "marginRight": "12px"}),
            # LINK
            html.Button("\U0001F517 LINK", id="lab-link-toggle", n_clicks=0,
                        style={**BTN_TOGGLE_STYLE, "fontSize": "9px", "marginRight": "8px"},
                        title="Sync pairs across all slots"),
            # LIVE
            html.Div([
                html.Button("LIVE", id="lab-live-toggle", n_clicks=0,
                            style={**BTN_TOGGLE_STYLE, "fontSize": "9px"}, title="Auto-refresh every 60s"),
                html.Span(id="lab-live-indicator", className="pulse-dot",
                          style={"width": "6px", "height": "6px", "borderRadius": "50%",
                                 "backgroundColor": "transparent", "display": "inline-block", "marginLeft": "4px"}),
            ], style={"display": "flex", "alignItems": "center", "marginRight": "8px"}),
            # Layout
            html.Div([
                html.Label("LAYOUT", style={**LABEL_STYLE, "marginRight": "6px"}),
                html.Button("1", id="lab-layout-1", n_clicks=0, style=BTN_TOGGLE_STYLE),
                html.Button("2", id="lab-layout-2", n_clicks=0, style=BTN_TOGGLE_ACTIVE),
                html.Button("4", id="lab-layout-4", n_clicks=0, style=BTN_TOGGLE_STYLE),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "padding": f"{GAP} {GAP}", "borderBottom": "1px solid #222240", "marginBottom": GAP,
                  "flexWrap": "wrap", "gap": GAP}),

        # Stores
        dcc.Store(id="lab-layout-store", data=2),
        dcc.Store(id="lab-link-store", data=False),
        dcc.Store(id="lab-live-store", data=False),
        dcc.Store(id="lab-refresh-counter", data=0),
        dcc.Interval(id="lab-auto-refresh-interval", interval=60_000, n_intervals=0, disabled=True),

        # ── Chart grid ───────────────────────────────────────────────────
        html.Div(id="lab-row-1", children=[
            html.Div(_chart_slot(0), style={"flex": "1", "minWidth": "400px"}),
            html.Div(_chart_slot(1), style={"flex": "1", "minWidth": "400px"}),
        ], style={"display": "flex", "gap": GAP, "marginBottom": GAP}),
        html.Div(id="lab-row-2", children=[
            html.Div(_chart_slot(2), style={"flex": "1", "minWidth": "400px"}),
            html.Div(_chart_slot(3), style={"flex": "1", "minWidth": "400px"}),
        ], style={"display": "flex", "gap": GAP, "marginBottom": SECTION_GAP}),

        # ── Deep Study Panel ─────────────────────────────────────────────
        html.Div([
            section_header("ANALYTICAL STUDIES"),
            html.Div([
                html.Div([html.Label("STUDY", style=LABEL_STYLE),
                           dcc.Dropdown(id="lab-study-type", options=DEEP_STUDY_OPTIONS,
                                        value="vol_deep_dive", clearable=False, style={"fontSize": "10px"})],
                         style={"flex": "1", "minWidth": "140px"}),
                html.Div([html.Label("PAIR(S)", style=LABEL_STYLE),
                           dcc.Dropdown(id="lab-study-pairs", options=PAIR_OPTIONS,
                                        value=["EURUSD", "USDJPY", "GBPUSD"], multi=True,
                                        style={"fontSize": "10px"})],
                         style={"flex": "2", "minWidth": "200px"}),
                html.Div([html.Label("TENOR", style=LABEL_STYLE),
                           dcc.Dropdown(id="lab-study-tenor", options=TENOR_OPTIONS,
                                        value="3M", clearable=False, style={"fontSize": "10px"})],
                         style={"flex": "0.6", "minWidth": "65px"}),
                html.Div([html.Button("RUN STUDY", id="lab-study-run", n_clicks=0, style=BTN_STYLE)],
                         style={"display": "flex", "alignItems": "flex-end"}),
            ], style=CTRL_WRAP),
            html.Button("CSV", id="lab-csv-study", n_clicks=0, style=CSV_BTN_STYLE),
            dcc.Graph(id="lab-study-chart", style={"height": f"{CHART_LG}px"}, config=GRAPH_CONFIG),
            html.Div(id="lab-study-stats", children=[], style=STAT_STRIP),
        ], style={**CARD_STYLE, "marginTop": SECTION_GAP, "marginBottom": SECTION_GAP}),

        # ── Comparison ───────────────────────────────────────────────────
        html.Div([
            section_header("COMPARISON OVERLAY"),
            html.Div([
                html.Div([html.Label("PAIRS (2-5)", style=LABEL_STYLE),
                           dcc.Dropdown(id="lab-comp-pairs", options=PAIR_OPTIONS,
                                        value=["EURUSD", "USDJPY", "GBPUSD"], multi=True,
                                        style={"fontSize": "10px"})],
                         style={"flex": "2", "minWidth": "200px"}),
                html.Div([html.Label("TYPE", style=LABEL_STYLE),
                           dcc.Dropdown(id="lab-comp-type", options=COMPARISON_TYPE_OPTIONS,
                                        value="term_structure", clearable=False, style={"fontSize": "10px"})],
                         style={"flex": "1", "minWidth": "120px"}),
                html.Div([html.Label("TENOR", style=LABEL_STYLE),
                           dcc.Dropdown(id="lab-comp-tenor", options=TENOR_OPTIONS,
                                        value="3M", clearable=False, style={"fontSize": "10px"})],
                         style={"flex": "0.6", "minWidth": "65px"}),
                html.Div([html.Button("REFRESH", id="lab-comp-refresh", n_clicks=0, style=BTN_STYLE)],
                         style={"display": "flex", "alignItems": "flex-end"}),
            ], style=CTRL_WRAP),
            html.Button("CSV", id="lab-csv-comp", n_clicks=0, style=CSV_BTN_STYLE),
            dcc.Graph(id="lab-comp-chart", style={"height": f"{CHART_LG}px"}, config=GRAPH_CONFIG),
        ], style={**CARD_STYLE, "marginTop": SECTION_GAP, "marginBottom": SECTION_GAP}),

        # ── Scratchpad + Pinned ──────────────────────────────────────────
        html.Div([
            section_header("SCRATCH PAD"),
            html.Div([
                html.Div([
                    html.Div([html.Label("NOTES", style=LABEL_STYLE),
                               html.Span(id="lab-scratchpad-meta", children="",
                                         style={"color": "#808080", "fontSize": "9px", "fontFamily": _FONT,
                                                "marginLeft": "8px"})],
                             style={"display": "flex", "alignItems": "center", "marginBottom": "2px"}),
                    dcc.Textarea(id="lab-scratchpad", value="", placeholder="Type session notes here...",
                                 style=TEXTAREA_STYLE),
                ], style={"flex": "2", "minWidth": "300px"}),
                html.Div([
                    html.Label("PINNED CHARTS", style=LABEL_STYLE),
                    html.Div(id="lab-pinned-list", children=[
                        html.Div("No charts pinned yet.", style={"color": "#808080", "fontSize": "10px",
                                                                  "fontFamily": _FONT, "padding": "8px"})],
                             style={"border": "1px solid #222240", "minHeight": "100px", "maxHeight": "200px",
                                    "overflowY": "auto", "padding": GAP}),
                ], style={"flex": "1", "minWidth": "200px"}),
            ], style={"display": "flex", "gap": GAP, "flexWrap": "wrap"}),
        ], style={**CARD_STYLE, "marginTop": SECTION_GAP}),

        # Persistent stores (local)
        dcc.Store(id="lab-scratchpad-store", storage_type="local", data=""),
        dcc.Store(id="lab-pinned-store", storage_type="local", data=[]),
    ], style={"backgroundColor": "#000000", "fontFamily": _FONT, "padding": GAP, "minHeight": "100vh"})


# ============================================================================
#  Data Helpers
# ============================================================================

def _fetch_series(pair, metric, tenor, timeframe):
    """Fetch a time series. Returns (pd.Series|None, label_str)."""
    label = f"{pair} {metric} {tenor}"
    try:
        if metric in ("ATM", "25D_RR", "25D_BF", "10D_RR", "10D_BF"):
            data = get_fx_historical_vol(pair, tenor, metric, int(timeframe))
            if data is not None and len(data) > 0:
                return (data if isinstance(data, pd.Series) else pd.Series(np.asarray(data))), label
            return None, label
        if metric == "SPOT":
            label = f"{pair} Spot"
            df = get_fx_historical_spot(pair, int(timeframe))
            if df is not None and not df.empty:
                col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else df.columns[-1])
                s = df[col]; s.name = label
                return s, label
            return None, label
        if metric == "IV_RV":
            label = f"{pair} IV-RV {tenor}"
            df = iv_rv_spread(pair, tenor, lookback=int(timeframe))
            if df is not None and not df.empty and "spread" in df.columns:
                s = df["spread"]; s.name = label; return s, label
            return None, label
        if metric == "FWD_VOL":
            label = f"{pair} Fwd Vol"
            df = forward_vol_curve(pair)
            if df is not None and not df.empty and "forward_vol" in df.columns:
                idx = df["end_tenor"].values if "end_tenor" in df.columns else np.arange(len(df))
                s = pd.Series(df["forward_vol"].values, index=idx, name=label)
                return s, label
            return None, label
        if metric == "RV":
            label = f"{pair} RV"
            df = get_fx_historical_spot(pair, int(timeframe) + 30)
            if df is not None and not df.empty:
                closes = df["close"] if "close" in df.columns else df.iloc[:, -1]
                rv = np.log(closes / closes.shift(1)).dropna().rolling(20).std() * np.sqrt(252) * 100
                rv = rv.dropna(); rv.name = label; return rv, label
            return None, label
        if metric == "TERM_SPREAD":
            label = f"{pair} 1M-1Y Spread"
            v1m = get_fx_historical_vol(pair, "1M", "ATM", int(timeframe))
            v1y = get_fx_historical_vol(pair, "1Y", "ATM", int(timeframe))
            if v1m is not None and v1y is not None:
                a, b = np.asarray(v1m), np.asarray(v1y)
                n = min(len(a), len(b))
                if n > 0:
                    s = pd.Series(a[-n:] - b[-n:], name=label); return s, label
            return None, label
        if metric == "CARRY":
            label = f"{pair} Carry (bps)"
            df = rate_differential_history(pair, lookback=int(timeframe))
            if df is not None and not df.empty and "rate_diff" in df.columns:
                s = pd.Series((df["rate_diff"].values * 10000), name=label)
                return s, label
            return None, label
    except Exception:
        logger.debug("_fetch_series error: %s", traceback.format_exc())
    return None, label


def _apply_norm(s, mode):
    """Apply normalization."""
    if s is None or len(s) == 0 or mode == "raw":
        return s
    arr = s.values if isinstance(s, pd.Series) else np.asarray(s)
    if mode == "indexed":
        f = arr[0]
        if f == 0 or np.isnan(f): return s
        result = (arr / f) * 100
    elif mode == "zscore":
        mu, sig = np.nanmean(arr), np.nanstd(arr)
        if sig == 0 or np.isnan(sig): return s
        result = (arr - mu) / sig
    elif mode == "pct_change":
        f = arr[0]
        if f == 0 or np.isnan(f): return s
        result = ((arr - f) / abs(f)) * 100
    else:
        return s
    return pd.Series(result, index=s.index if isinstance(s, pd.Series) else None, name=getattr(s, 'name', None))


def _empty(msg="No data"):
    fig = go.Figure()
    fig.update_layout(**chart_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(color="#808080", size=12, family=_FONT))]))
    return fig


_CW = CHART_TEMPLATE["layout"]["colorway"]


# ============================================================================
#  Study Figure Builders (STUDY_* metrics)
# ============================================================================

def _study_vol_cone(pair, tenor, timeframe):
    """Vol cone with percentile bands and current level."""
    try:
        df = vol_cone(pair, lookback=int(timeframe))
        if df is None or df.empty: return _empty("No vol cone data")
        fig = go.Figure()
        w = df["window"].tolist()
        # Bands
        for col, color, name in [("p90", "rgba(255,136,0,0.15)", "90th"),
                                  ("p75", "rgba(255,136,0,0.1)", "75th"),
                                  ("p25", "rgba(255,136,0,0.1)", "25th"),
                                  ("p10", "rgba(255,136,0,0.15)", "10th")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=w, y=df[col].tolist(), mode="lines", name=name,
                                         line=dict(color="#808080", width=1, dash="dot")))
        if "median" in df.columns:
            fig.add_trace(go.Scatter(x=w, y=df["median"].tolist(), mode="lines", name="Median",
                                     line=dict(color="#d4d4d4", width=1.5, dash="dash")))
        if "current_c2c" in df.columns:
            fig.add_trace(go.Scatter(x=w, y=df["current_c2c"].tolist(), mode="lines+markers",
                                     name="Current", line=dict(color="#ff8800", width=2.5),
                                     marker=dict(size=6, color="#ff8800")))
        fig.update_layout(**chart_layout(
            title=dict(text=f"{pair} Realized Vol Cone", font=dict(size=11, color="#ff8800", family=_FONT)),
            xaxis_title="Window (days)", yaxis_title="Realized Vol (%)",
            margin=dict(l=45, r=15, t=40, b=30), hovermode="x unified"))
        return fig
    except Exception:
        logger.debug("vol_cone study error: %s", traceback.format_exc())
        return _empty("Error building vol cone")


def _study_smile(pair, tenor, timeframe):
    """5-point vol smile across multiple tenors."""
    try:
        surface = get_fx_vol_surface(pair)
        if not surface: return _empty("No surface data")
        fig = go.Figure()
        for idx, t in enumerate(["1M", "3M", "6M", "1Y"]):
            td = surface.get(t, {})
            atm = td.get("atm", 10); rr25 = td.get("rr25", 0); bf25 = td.get("bf25", 0)
            rr10 = td.get("rr10", 0); bf10 = td.get("bf10", 0)
            vols = [atm - rr10/2 + bf10, atm - rr25/2 + bf25, atm, atm + rr25/2 + bf25, atm + rr10/2 + bf10]
            fig.add_trace(go.Scatter(x=["10P", "25P", "ATM", "25C", "10C"], y=vols,
                                     mode="lines+markers", name=t, line=dict(color=_CW[idx % len(_CW)], width=2),
                                     marker=dict(size=5)))
        fig.update_layout(**chart_layout(
            title=dict(text=f"{pair} Vol Smile", font=dict(size=11, color="#ff8800", family=_FONT)),
            xaxis_title="Delta", yaxis_title="Implied Vol (%)",
            margin=dict(l=45, r=15, t=40, b=30), hovermode="x unified"))
        return fig
    except Exception:
        return _empty("Error building smile")


def _study_implied_pdf(pair, tenor, timeframe):
    """Risk-neutral probability density (Breeden-Litzenberger)."""
    try:
        df = smile_implied_pdf(pair, tenor)
        if df is None or df.empty: return _empty("No PDF data")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["strike"].tolist(), y=df["pdf"].tolist(), mode="lines",
                                  name="Implied PDF", fill="tozeroy",
                                  line=dict(color="#ff8800", width=2),
                                  fillcolor="rgba(255,136,0,0.15)"))
        spots = get_fx_spots([pair])
        spot = spots.get(pair, {}).get("mid")
        if spot:
            fig.add_vline(x=spot, line_dash="dash", line_color="#d4d4d4", annotation_text=f"Spot {spot:.4f}")
        fig.update_layout(**chart_layout(
            title=dict(text=f"{pair} {tenor} Implied PDF", font=dict(size=11, color="#ff8800", family=_FONT)),
            xaxis_title="Strike", yaxis_title="Probability Density",
            margin=dict(l=45, r=15, t=40, b=30), hovermode="x unified"))
        return fig
    except Exception:
        return _empty("Error building PDF")


def _study_vol_regime(pair, tenor, timeframe):
    """ATM vol time series with regime-colored background."""
    try:
        df = vol_regime_history(pair, lookback=int(timeframe))
        if df is None or df.empty: return _empty("No regime data")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["day"].tolist(), y=df["vol"].tolist(), mode="lines",
                                  name=f"{pair} ATM", line=dict(color="#ff8800", width=2)))
        # Regime bands
        for level, color, label in [(20, "#ff3333", "CRISIS"), (14, "#ff8800", "HIGH"),
                                     (10, "#ffaa33", "ELEVATED"), (6, "#808080", "NORMAL")]:
            fig.add_hline(y=level, line_dash="dot", line_color=color,
                          annotation_text=label, annotation_font_size=8, annotation_font_color=color)
        regime = vol_regime_detect(pair)
        fig.update_layout(**chart_layout(
            title=dict(text=f"{pair} Vol Regime: {regime.get('regime','?')} | Trend: {regime.get('trend','?')}",
                       font=dict(size=11, color=regime.get('color', '#ff8800'), family=_FONT)),
            xaxis_title="Days", yaxis_title="ATM Vol (%)",
            margin=dict(l=45, r=15, t=40, b=30), hovermode="x unified"))
        return fig
    except Exception:
        return _empty("Error building regime")


def _study_fwd_vol_curve(pair, tenor, timeframe):
    """Forward vol curve vs spot vol."""
    try:
        df = forward_vol_curve(pair, start_tenor=tenor)
        if df is None or df.empty: return _empty("No forward vol data")
        fig = go.Figure()
        if "spot_vol" in df.columns:
            fig.add_trace(go.Scatter(x=df["end_tenor"].tolist(), y=df["spot_vol"].tolist(),
                                      mode="lines+markers", name="Spot Vol", line=dict(color="#d4d4d4", width=1.5)))
        fig.add_trace(go.Scatter(x=df["end_tenor"].tolist(), y=df["forward_vol"].tolist(),
                                  mode="lines+markers", name="Forward Vol",
                                  line=dict(color="#ff8800", width=2.5), marker=dict(size=6)))
        fig.update_layout(**chart_layout(
            title=dict(text=f"{pair} Forward Vol (from {tenor})", font=dict(size=11, color="#ff8800", family=_FONT)),
            xaxis_title="End Tenor", yaxis_title="Vol (%)",
            margin=dict(l=45, r=15, t=40, b=30), hovermode="x unified"))
        return fig
    except Exception:
        return _empty("Error building fwd vol curve")


def _study_surface_heatmap(pair, func, title_prefix):
    """Generic percentile/zscore surface heatmap."""
    try:
        df = func(pair)
        if df is None or df.empty: return _empty("No surface data")
        is_pctile = "percentile" in title_prefix.lower() or df.values.max() > 10
        cscale = [[0, "#00cc66"], [0.5, "#000000"], [1, "#ff3333"]] if not is_pctile else \
                 [[0, "#00cc66"], [0.25, "#222240"], [0.5, "#808080"], [0.75, "#222240"], [1, "#ff3333"]]
        fig = go.Figure(data=go.Heatmap(
            z=df.values, x=df.columns.tolist(), y=df.index.tolist(),
            colorscale=cscale, text=np.round(df.values, 1).astype(str), texttemplate="%{text}",
            textfont=dict(size=10, family=_FONT, color="#d4d4d4")))
        fig.update_layout(**chart_layout(
            title=dict(text=f"{pair} {title_prefix}", font=dict(size=11, color="#ff8800", family=_FONT)),
            margin=dict(l=60, r=20, t=40, b=40)))
        return fig
    except Exception:
        return _empty(f"Error building {title_prefix}")


def _study_tail_probs(pair, tenor, timeframe):
    """Implied tail probabilities from the smile."""
    try:
        df = tail_probabilities(pair, tenor)
        if df is None or df.empty: return _empty("No tail prob data")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"+{m:.0f}%" for m in df["move_pct"]], y=df["prob_up"].tolist(),
                              name="Up", marker_color="#00cc66", opacity=0.85))
        fig.add_trace(go.Bar(x=[f"+{m:.0f}%" for m in df["move_pct"]], y=df["prob_down"].tolist(),
                              name="Down", marker_color="#ff3333", opacity=0.85))
        fig.update_layout(**chart_layout(
            title=dict(text=f"{pair} {tenor} Tail Probabilities", font=dict(size=11, color="#ff8800", family=_FONT)),
            xaxis_title="Move Size", yaxis_title="Probability (%)", barmode="group",
            margin=dict(l=45, r=15, t=40, b=30)))
        return fig
    except Exception:
        return _empty("Error building tail probs")


def _study_breakeven(pair, tenor, timeframe):
    """Breakeven vol for ATM straddle."""
    try:
        from core.fx_conventions import tenor_to_days
        days = tenor_to_days(tenor)
        info = breakeven_vol(pair, tenor, days)
        fig = go.Figure()
        labels = ["ATM IV", "Breakeven RV", "Cushion"]
        vals = [info["atm_iv"], info["breakeven_rv"], info["iv_rv_cushion"]]
        colors = ["#ff8800", "#d4d4d4", "#00cc66" if info["iv_rv_cushion"] > 0 else "#ff3333"]
        fig.add_trace(go.Bar(x=labels, y=vals, marker_color=colors,
                              text=[f"{v:.2f}" for v in vals], textposition="outside",
                              textfont=dict(color="#d4d4d4", family=_FONT, size=11)))
        fig.update_layout(**chart_layout(
            title=dict(text=f"{pair} {tenor} Breakeven Analysis", font=dict(size=11, color="#ff8800", family=_FONT)),
            yaxis_title="Vol (%)", margin=dict(l=45, r=15, t=40, b=30)))
        return fig
    except Exception:
        return _empty("Error building breakeven")


def _study_carry_landscape(pairs, tenor, timeframe):
    """Carry per vol ranked bar chart."""
    try:
        df = carry_per_vol(pairs)
        if df is None or df.empty: return _empty("No carry data")
        colors = ["#00cc66" if s == "ATTRACTIVE" else ("#ff8800" if s == "MODERATE" else "#ff3333")
                  for s in df["rank_signal"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["pair"].tolist(), y=df["sharpe_proxy"].tolist(), marker_color=colors,
                              text=[f"{v:.2f}" for v in df["sharpe_proxy"]], textposition="outside",
                              textfont=dict(color="#d4d4d4", family=_FONT, size=10)))
        fig.update_layout(**chart_layout(
            title=dict(text="Carry / Vol Ranking (Sharpe Proxy)", font=dict(size=11, color="#ff8800", family=_FONT)),
            yaxis_title="Sharpe Proxy", margin=dict(l=45, r=15, t=40, b=30)))
        return fig
    except Exception:
        return _empty("Error building carry landscape")


def _build_study_figure(metric, pairs, tenor, timeframe):
    """Dispatch to the right study builder."""
    pair = pairs[0] if pairs else "EURUSD"
    if metric == "STUDY_VOL_CONE":         return _study_vol_cone(pair, tenor, timeframe)
    if metric == "STUDY_SMILE":            return _study_smile(pair, tenor, timeframe)
    if metric == "STUDY_IMPLIED_PDF":      return _study_implied_pdf(pair, tenor, timeframe)
    if metric == "STUDY_VOL_REGIME":       return _study_vol_regime(pair, tenor, timeframe)
    if metric == "STUDY_FWD_VOL_CURVE":    return _study_fwd_vol_curve(pair, tenor, timeframe)
    if metric == "STUDY_PCTILE_SURFACE":   return _study_surface_heatmap(pair, vol_percentile_surface, "Percentile Surface")
    if metric == "STUDY_ZSCORE_SURFACE":   return _study_surface_heatmap(pair, vol_zscore_surface, "Z-Score Surface")
    if metric == "STUDY_TAIL_PROBS":       return _study_tail_probs(pair, tenor, timeframe)
    if metric == "STUDY_BREAKEVEN":        return _study_breakeven(pair, tenor, timeframe)
    if metric == "STUDY_CARRY_LANDSCAPE":  return _study_carry_landscape(pairs, tenor, timeframe)
    return _empty(f"Unknown study: {metric}")


# ============================================================================
#  Slot Figure Builder
# ============================================================================

def _build_slot_figure(metric, pairs, tenor, timeframe, chart_type, normalize, overlay=""):
    """Build the plotly figure for a single chart slot, with optional dual-axis overlay."""
    if not pairs:
        return _empty("Select at least one pair")
    if isinstance(pairs, str):
        pairs = [pairs]
    pairs = pairs[:5]

    if metric.startswith("STUDY_"):
        return _build_study_figure(metric, pairs, tenor, timeframe)

    has_overlay = bool(overlay) and overlay != metric and not overlay.startswith("STUDY_")
    norm_mode = normalize if isinstance(normalize, str) else "raw"

    if has_overlay:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    any_data = False

    def _hex_fill(color):
        if color.startswith("rgb"):
            return color.replace(")", ",0.15)").replace("rgb", "rgba")
        if color.startswith("#") and len(color) >= 7:
            return f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"
        return "rgba(255,136,0,0.15)"

    def _xy(series):
        x = series.index if isinstance(series, pd.Series) and series.index.dtype != object else list(range(len(series)))
        y = series.values if isinstance(series, pd.Series) else series
        return x, y

    # ── Primary metric (left axis) ──
    for idx, pair in enumerate(pairs):
        series, label = _fetch_series(pair, metric, tenor, timeframe)
        if series is None or len(series) == 0:
            continue
        any_data = True
        series = _apply_norm(series, norm_mode)
        color = _CW[idx % len(_CW)]
        x, y = _xy(series)

        if chart_type == "area":
            trace = go.Scatter(x=x, y=y, mode="lines", name=label,
                               line=dict(color=color, width=1.5), fill="tozeroy", fillcolor=_hex_fill(color))
        elif chart_type == "bar":
            trace = go.Bar(x=x, y=y, name=label, marker_color=color, opacity=0.85)
        else:
            trace = go.Scatter(x=x, y=y, mode="lines", name=label, line=dict(color=color, width=1.5))

        if has_overlay:
            fig.add_trace(trace, secondary_y=False)
        else:
            fig.add_trace(trace)

    # ── Overlay metric (right axis, dashed/dotted) ──
    if has_overlay:
        ov_colors = ["#ffffff", "#88ff88", "#ff8888", "#88bbff", "#ffcc44", "#cc88ff", "#88ffff", "#ffaacc"]
        for idx, pair in enumerate(pairs):
            series, label = _fetch_series(pair, overlay, tenor, timeframe)
            if series is None or len(series) == 0:
                continue
            any_data = True
            series = _apply_norm(series, norm_mode)
            color = ov_colors[idx % len(ov_colors)]
            x, y = _xy(series)
            ov_name = next((m["label"] for m in OVERLAY_OPTIONS if m["value"] == overlay), overlay)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{pair} {ov_name}",
                                     line=dict(color=color, width=1.5, dash="dot")),
                          secondary_y=True)

    if not any_data:
        return _empty("No data available")

    m_label = next((m["label"] for m in METRIC_OPTIONS if m["value"] == metric), metric)
    title = m_label
    if has_overlay:
        ov_label = next((m["label"] for m in OVERLAY_OPTIONS if m["value"] == overlay), overlay)
        title += f"  vs  {ov_label}"
    title += f" | {tenor} | {timeframe}d"
    if norm_mode != "raw":
        title += f" [{next((n['label'] for n in NORMALIZE_OPTIONS if n['value'] == norm_mode), norm_mode)}]"

    fig.update_layout(**chart_layout(
        title=dict(text=title, font=dict(size=11, color="#ff8800", family=_FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=9, color="#d4d4d4", family=_FONT), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=45, r=50 if has_overlay else 15, t=40, b=30), hovermode="x unified"))

    if has_overlay:
        fig.update_yaxes(title_text=m_label, secondary_y=False,
                         title_font=dict(size=9, color="#ff8800", family=_FONT))
        fig.update_yaxes(title_text=ov_label, secondary_y=True,
                         title_font=dict(size=9, color="#ffffff", family=_FONT),
                         gridcolor="rgba(26,26,46,0.3)")

    return fig


# ============================================================================
#  Slot Stats Builder
# ============================================================================

def _build_slot_stats(metric, pairs, tenor, timeframe):
    """Build contextual stats for the strip under each chart."""
    if not pairs:
        return []
    pair = pairs[0] if isinstance(pairs, list) else pairs

    def _sv(label, value, color="#d4d4d4"):
        return html.Span([html.Span(f"{label}: ", style=STAT_ITEM),
                           html.Span(str(value), style={**STAT_VAL, "color": color})])

    try:
        if metric.startswith("STUDY_"):
            if metric == "STUDY_VOL_CONE":
                df = vol_cone(pair, lookback=int(timeframe))
                if df is not None and not df.empty and "percentile_rank" in df.columns:
                    pct = df["percentile_rank"].iloc[-1]
                    c = "#ff3333" if pct > 75 else ("#00cc66" if pct < 25 else "#d4d4d4")
                    return [_sv("RV %ile", f"{pct:.0f}th", c)]
            if metric == "STUDY_VOL_REGIME":
                r = vol_regime_detect(pair)
                return [_sv("Regime", r["regime"], r.get("color", "#d4d4d4")),
                        _sv("Trend", r["trend"]), _sv("ATM", f"{r['atm_iv']:.2f}")]
            if metric == "STUDY_SMILE":
                sk = smile_skewness(pair, tenor)
                ku = smile_kurtosis(pair, tenor)
                return [_sv("Skew", f"{sk['rr_25d']:.2f} ({sk['direction']})"),
                        _sv("Kurtosis", f"{ku['bf_25d']:.2f} ({ku['tail_assessment']})")]
            return []

        # Time series stats
        if metric in ("ATM", "25D_RR", "25D_BF", "10D_RR", "10D_BF"):
            info = vol_percentile(pair, tenor, metric, min(int(timeframe), 252))
            chg = vol_change(pair, tenor, metric, 1)
            z = vol_zscore(pair, tenor, metric, min(int(timeframe), 252))
            chg_c = "#00cc66" if chg["abs_change"] < 0 else ("#ff3333" if chg["abs_change"] > 0 else "#d4d4d4")
            z_c = "#ff3333" if abs(z["zscore"]) > 2 else ("#ff8800" if abs(z["zscore"]) > 1 else "#d4d4d4")
            return [_sv("Last", f"{info['current']:.2f}"), _sv("1D", f"{chg['abs_change']:+.2f}", chg_c),
                    _sv("Z", f"{z['zscore']:+.1f}", z_c), _sv("%ile", f"{info['percentile']:.0f}th"),
                    _sv("Range", f"{info['min']:.1f}-{info['max']:.1f}")]
        if metric == "IV_RV":
            ivr = iv_rv_percentile(pair, tenor, lookback=min(int(timeframe), 252))
            c = "#ff3333" if ivr["signal"] == "IV_RICH" else ("#00cc66" if ivr["signal"] == "IV_CHEAP" else "#d4d4d4")
            return [_sv("Spread", f"{ivr['current_spread']:.2f}"), _sv("Signal", ivr["signal"], c),
                    _sv("%ile", f"{ivr['percentile']:.0f}th")]
    except Exception:
        pass
    return []


# ============================================================================
#  Comparison Figure Builder
# ============================================================================

def _build_comparison_figure(pairs, comp_type, tenor="3M"):
    """Build comparison overlay chart."""
    if not pairs or len(pairs) < 2:
        return _empty("Select 2-5 pairs")
    pairs = pairs[:5]
    fig = go.Figure()

    try:
        if comp_type == "term_structure":
            for idx, pair in enumerate(pairs):
                try:
                    ts = get_fx_term_structure(pair)
                    if ts is not None and not ts.empty and "atm" in ts.columns:
                        fig.add_trace(go.Scatter(x=ts["tenor"].tolist(), y=ts["atm"].tolist(),
                                                  mode="lines+markers", name=pair,
                                                  line=dict(color=_CW[idx % len(_CW)], width=2),
                                                  marker=dict(size=5)))
                except Exception: pass
            fig.update_layout(**chart_layout(title=dict(text="ATM Vol Term Structure",
                              font=dict(size=11, color="#ff8800", family=_FONT)),
                              xaxis_title="Tenor", yaxis_title="ATM Vol (%)"))

        elif comp_type == "skew_profile":
            for idx, pair in enumerate(pairs):
                try:
                    surface = get_fx_vol_surface(pair)
                    if surface:
                        rr_vals, t_labels = [], []
                        for t in ["1M", "3M", "6M", "1Y"]:
                            rr_vals.append(surface.get(t, {}).get("rr25", 0)); t_labels.append(t)
                        fig.add_trace(go.Scatter(x=t_labels, y=rr_vals, mode="lines+markers", name=pair,
                                                  line=dict(color=_CW[idx % len(_CW)], width=2)))
                except Exception: pass
            fig.update_layout(**chart_layout(title=dict(text="25D RR Skew Profile",
                              font=dict(size=11, color="#ff8800", family=_FONT)),
                              xaxis_title="Tenor", yaxis_title="25D RR (vol pts)"))

        elif comp_type == "smile":
            for idx, pair in enumerate(pairs):
                try:
                    surface = get_fx_vol_surface(pair)
                    if surface:
                        td = surface.get(tenor, surface.get("3M", {}))
                        atm, rr25, bf25 = td.get("atm", 10), td.get("rr25", 0), td.get("bf25", 0)
                        rr10, bf10 = td.get("rr10", 0), td.get("bf10", 0)
                        vols = [atm-rr10/2+bf10, atm-rr25/2+bf25, atm, atm+rr25/2+bf25, atm+rr10/2+bf10]
                        fig.add_trace(go.Scatter(x=["10P","25P","ATM","25C","10C"], y=vols,
                                                  mode="lines+markers", name=f"{pair} {tenor}",
                                                  line=dict(color=_CW[idx % len(_CW)], width=2)))
                except Exception: pass
            fig.update_layout(**chart_layout(title=dict(text=f"{tenor} Smile Comparison",
                              font=dict(size=11, color="#ff8800", family=_FONT)),
                              xaxis_title="Delta", yaxis_title="Vol (%)"))

        elif comp_type == "vol_cone":
            for idx, pair in enumerate(pairs):
                try:
                    cone = vol_cone(pair)
                    if cone is not None and not cone.empty and "current_c2c" in cone.columns:
                        fig.add_trace(go.Scatter(x=cone["window"].tolist(), y=cone["current_c2c"].tolist(),
                                                  mode="lines+markers", name=pair,
                                                  line=dict(color=_CW[idx % len(_CW)], width=2)))
                except Exception: pass
            fig.update_layout(**chart_layout(title=dict(text="RV Cone: Current Level",
                              font=dict(size=11, color="#ff8800", family=_FONT)),
                              xaxis_title="Window (days)", yaxis_title="RV (%)"))

        elif comp_type == "iv_rv_overlay":
            for idx, pair in enumerate(pairs):
                try:
                    df = iv_rv_spread(pair, tenor, lookback=120)
                    if df is not None and not df.empty:
                        c = _CW[idx % len(_CW)]
                        if "iv" in df.columns:
                            fig.add_trace(go.Scatter(x=list(range(len(df))), y=df["iv"].tolist(),
                                                      mode="lines", name=f"{pair} IV", line=dict(color=c, width=2)))
                        if "rv" in df.columns:
                            fig.add_trace(go.Scatter(x=list(range(len(df))), y=df["rv"].tolist(),
                                                      mode="lines", name=f"{pair} RV", line=dict(color=c, width=1.5, dash="dash")))
                except Exception: pass
            fig.update_layout(**chart_layout(title=dict(text=f"IV vs RV ({tenor})",
                              font=dict(size=11, color="#ff8800", family=_FONT)),
                              xaxis_title="Days", yaxis_title="Vol (%)"))

        elif comp_type == "correlation_matrix":
            corr = spot_correlation_matrix(pairs, window=60)
            if corr is not None and not corr.empty:
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale=[[0,"#ff3333"],[0.5,"#000000"],[1,"#00cc66"]], zmin=-1, zmax=1,
                    text=np.round(corr.values, 2).astype(str), texttemplate="%{text}",
                    textfont=dict(size=11, family=_FONT, color="#d4d4d4")))
                fig.update_layout(**chart_layout(title=dict(text="60D Spot Correlation",
                                  font=dict(size=11, color="#ff8800", family=_FONT))))

        elif comp_type == "vol_correlation_matrix":
            corr = vol_correlation_matrix(pairs, tenor=tenor, window=60)
            if corr is not None and not corr.empty:
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale=[[0,"#ff3333"],[0.5,"#000000"],[1,"#00cc66"]], zmin=-1, zmax=1,
                    text=np.round(corr.values, 2).astype(str), texttemplate="%{text}",
                    textfont=dict(size=11, family=_FONT, color="#d4d4d4")))
                fig.update_layout(**chart_layout(title=dict(text=f"60D Vol Change Correlation ({tenor})",
                                  font=dict(size=11, color="#ff8800", family=_FONT))))

        elif comp_type == "rv_heatmap":
            df = rv_scanner(pairs, ["1M", "3M", "1Y"])
            if df is not None and not df.empty:
                pvt = df.pivot_table(values="zscore", index="pair", columns="tenor", aggfunc="first")
                if not pvt.empty:
                    fig = go.Figure(data=go.Heatmap(z=pvt.values, x=pvt.columns.tolist(), y=pvt.index.tolist(),
                        colorscale=[[0,"#00cc66"],[0.5,"#000000"],[1,"#ff3333"]], zmid=0,
                        text=np.round(pvt.values, 2).astype(str), texttemplate="%{text}",
                        textfont=dict(size=11, family=_FONT, color="#d4d4d4")))
                    fig.update_layout(**chart_layout(title=dict(text="Vol Z-Score Heatmap (cheap=green, rich=red)",
                                      font=dict(size=11, color="#ff8800", family=_FONT))))

        elif comp_type == "carry_ranking":
            df = carry_per_vol(pairs)
            if df is not None and not df.empty:
                colors = ["#00cc66" if s == "ATTRACTIVE" else ("#ff8800" if s == "MODERATE" else "#ff3333")
                          for s in df["rank_signal"]]
                fig.add_trace(go.Bar(x=df["pair"].tolist(), y=df["sharpe_proxy"].tolist(), marker_color=colors,
                                      text=[f"{v:.2f}" for v in df["sharpe_proxy"]], textposition="outside",
                                      textfont=dict(color="#d4d4d4", family=_FONT, size=10)))
                fig.update_layout(**chart_layout(title=dict(text="Carry / Vol Ranking",
                                  font=dict(size=11, color="#ff8800", family=_FONT)),
                                  yaxis_title="Sharpe Proxy"))

    except Exception:
        logger.error("Comparison error: %s", traceback.format_exc())
        return _empty("Error building comparison")

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                   font=dict(size=9, color="#d4d4d4", family=_FONT), bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=50, r=20, t=45, b=35), hovermode="x unified")
    return fig


# ============================================================================
#  Deep Study Builder (Study Panel)
# ============================================================================

def _build_deep_study(study_type, pairs, tenor):
    """Build multi-panel deep study figure."""
    pair = pairs[0] if pairs else "EURUSD"

    try:
        if study_type == "vol_deep_dive":
            fig = make_subplots(rows=2, cols=2, subplot_titles=[
                f"{pair} ATM Vol + Regime", f"{pair} Vol Cone",
                f"{pair} IV-RV Spread", "Stats"],
                vertical_spacing=0.12, horizontal_spacing=0.08)
            # Panel 1: Vol regime history
            rh = vol_regime_history(pair, 252)
            if rh is not None and not rh.empty:
                fig.add_trace(go.Scatter(x=rh["day"].tolist(), y=rh["vol"].tolist(), mode="lines",
                              name="ATM Vol", line=dict(color="#ff8800", width=2)), row=1, col=1)
            # Panel 2: Vol cone
            vc = vol_cone(pair)
            if vc is not None and not vc.empty:
                w = vc["window"].tolist()
                if "median" in vc.columns:
                    fig.add_trace(go.Scatter(x=w, y=vc["median"].tolist(), mode="lines", name="Median",
                                  line=dict(color="#808080", dash="dash")), row=1, col=2)
                if "current_c2c" in vc.columns:
                    fig.add_trace(go.Scatter(x=w, y=vc["current_c2c"].tolist(), mode="lines+markers",
                                  name="Current RV", line=dict(color="#ff8800", width=2)), row=1, col=2)
            # Panel 3: IV-RV spread
            ivr = iv_rv_spread(pair, tenor, lookback=252)
            if ivr is not None and not ivr.empty:
                if "iv" in ivr.columns:
                    fig.add_trace(go.Scatter(x=list(range(len(ivr))), y=ivr["iv"].tolist(), mode="lines",
                                  name="IV", line=dict(color="#ff8800")), row=2, col=1)
                if "rv" in ivr.columns:
                    fig.add_trace(go.Scatter(x=list(range(len(ivr))), y=ivr["rv"].tolist(), mode="lines",
                                  name="RV", line=dict(color="#d4d4d4", dash="dash")), row=2, col=1)
            # Panel 4: Stats as annotations
            r = vol_regime_detect(pair)
            z = vol_zscore(pair, tenor, "ATM")
            stats_text = (f"Regime: {r['regime']}<br>Trend: {r['trend']}<br>"
                         f"ATM IV: {r['atm_iv']:.2f}<br>Z-Score: {z.get('zscore', 0):+.2f}<br>"
                         f"Percentile: {z.get('percentile', 50):.0f}th")
            fig.add_annotation(text=stats_text, xref="x4", yref="y4", x=0.5, y=0.5,
                               showarrow=False, font=dict(color="#d4d4d4", size=11, family=_FONT),
                               align="left", row=2, col=2)

        elif study_type == "smile_deep_dive":
            fig = make_subplots(rows=2, cols=2, subplot_titles=[
                f"{pair} Smile ({tenor})", f"{pair} Implied PDF ({tenor})",
                "Skew Across Tenors", "Tail Probabilities"])
            # Smile
            surface = get_fx_vol_surface(pair)
            if surface:
                td = surface.get(tenor, {})
                atm, rr25, bf25 = td.get("atm", 10), td.get("rr25", 0), td.get("bf25", 0)
                rr10, bf10 = td.get("rr10", 0), td.get("bf10", 0)
                vols = [atm-rr10/2+bf10, atm-rr25/2+bf25, atm, atm+rr25/2+bf25, atm+rr10/2+bf10]
                fig.add_trace(go.Scatter(x=["10P","25P","ATM","25C","10C"], y=vols,
                              mode="lines+markers", name="Smile", line=dict(color="#ff8800", width=2)), row=1, col=1)
            # PDF
            pdf = smile_implied_pdf(pair, tenor)
            if pdf is not None and not pdf.empty:
                fig.add_trace(go.Scatter(x=pdf["strike"].tolist(), y=pdf["pdf"].tolist(), mode="lines",
                              name="PDF", fill="tozeroy", line=dict(color="#ff8800"),
                              fillcolor="rgba(255,136,0,0.15)"), row=1, col=2)
            # Skew across tenors
            if surface:
                for t in ["1M", "3M", "6M", "1Y"]:
                    rr = surface.get(t, {}).get("rr25", 0)
                    fig.add_trace(go.Bar(x=[t], y=[rr], name=f"RR {t}",
                                  marker_color="#ff8800", showlegend=False), row=2, col=1)
            # Tail probs
            tp = tail_probabilities(pair, tenor)
            if tp is not None and not tp.empty:
                fig.add_trace(go.Bar(x=[f"{m:.0f}%" for m in tp["move_pct"]], y=tp["prob_either"].tolist(),
                              name="Either", marker_color="#ff8800"), row=2, col=2)

        elif study_type == "rv_scanner":
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Vol Z-Score Heatmap", "IV-RV Spread"])
            sc = rv_scanner(pairs, ["1M", "3M", "1Y"])
            if sc is not None and not sc.empty:
                pvt = sc.pivot_table(values="zscore", index="pair", columns="tenor", aggfunc="first")
                if not pvt.empty:
                    fig.add_trace(go.Heatmap(z=pvt.values, x=pvt.columns.tolist(), y=pvt.index.tolist(),
                        colorscale=[[0,"#00cc66"],[0.5,"#000000"],[1,"#ff3333"]], zmid=0,
                        text=np.round(pvt.values, 2).astype(str), texttemplate="%{text}",
                        textfont=dict(size=10, family=_FONT)), row=1, col=1)
                ivrv_pvt = sc.pivot_table(values="iv_rv_spread", index="pair", columns="tenor", aggfunc="first")
                if not ivrv_pvt.empty:
                    for cidx, t in enumerate(ivrv_pvt.columns):
                        fig.add_trace(go.Bar(x=ivrv_pvt.index.tolist(), y=ivrv_pvt[t].tolist(),
                                      name=t, marker_color=_CW[cidx % len(_CW)]), row=1, col=2)

        elif study_type == "correlation_lab":
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Spot Correlation", "Vol Correlation"])
            sc = spot_correlation_matrix(pairs, 60)
            if sc is not None and not sc.empty:
                fig.add_trace(go.Heatmap(z=sc.values, x=sc.columns.tolist(), y=sc.index.tolist(),
                    colorscale=[[0,"#ff3333"],[0.5,"#000000"],[1,"#00cc66"]], zmin=-1, zmax=1,
                    text=np.round(sc.values, 2).astype(str), texttemplate="%{text}",
                    textfont=dict(size=10, family=_FONT)), row=1, col=1)
            vc = vol_correlation_matrix(pairs, tenor, 60)
            if vc is not None and not vc.empty:
                fig.add_trace(go.Heatmap(z=vc.values, x=vc.columns.tolist(), y=vc.index.tolist(),
                    colorscale=[[0,"#ff3333"],[0.5,"#000000"],[1,"#00cc66"]], zmin=-1, zmax=1,
                    text=np.round(vc.values, 2).astype(str), texttemplate="%{text}",
                    textfont=dict(size=10, family=_FONT)), row=1, col=2)

        elif study_type == "carry_dashboard":
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Carry / Vol Ranking", "Carry Momentum"])
            cpv = carry_per_vol(pairs)
            if cpv is not None and not cpv.empty:
                colors = ["#00cc66" if s == "ATTRACTIVE" else ("#ff8800" if s == "MODERATE" else "#ff3333")
                          for s in cpv["rank_signal"]]
                fig.add_trace(go.Bar(x=cpv["pair"].tolist(), y=cpv["sharpe_proxy"].tolist(),
                              marker_color=colors, name="Sharpe Proxy"), row=1, col=1)
            for idx, p in enumerate(pairs[:6]):
                try:
                    cm = carry_momentum(p)
                    fig.add_trace(go.Bar(x=[p], y=[cm["change_20d_bps"]], name=f"{p} 20D",
                                  marker_color=_CW[idx % len(_CW)], showlegend=False), row=1, col=2)
                except Exception: pass

        elif study_type == "fwd_vol_lab":
            fig = make_subplots(rows=1, cols=2, subplot_titles=[
                f"{pair} Forward Vol Curve", f"{pair} Forward Vol Surface"])
            fc = forward_vol_curve(pair)
            if fc is not None and not fc.empty:
                if "spot_vol" in fc.columns:
                    fig.add_trace(go.Scatter(x=fc["end_tenor"].tolist(), y=fc["spot_vol"].tolist(),
                                  mode="lines+markers", name="Spot Vol", line=dict(color="#d4d4d4")), row=1, col=1)
                fig.add_trace(go.Scatter(x=fc["end_tenor"].tolist(), y=fc["forward_vol"].tolist(),
                              mode="lines+markers", name="Fwd Vol", line=dict(color="#ff8800", width=2)), row=1, col=1)
            fs = forward_vol_surface(pair)
            if fs is not None and not fs.empty:
                fig.add_trace(go.Heatmap(z=fs.values, x=fs.columns.tolist(), y=fs.index.tolist(),
                    colorscale=[[0,"#000000"],[1,"#ff8800"]],
                    text=np.where(np.isnan(fs.values), "", np.round(fs.values, 1).astype(str)),
                    texttemplate="%{text}", textfont=dict(size=9, family=_FONT)), row=1, col=2)

        else:
            return _empty(f"Unknown study: {study_type}")

        fig.update_layout(**chart_layout(margin=dict(l=50, r=20, t=45, b=35), hovermode="x unified",
                          height=CHART_LG, showlegend=True))
        for ann in fig.layout.annotations:
            ann.font = dict(size=10, color="#ff8800", family=_FONT)
        return fig

    except Exception:
        logger.error("Deep study error: %s", traceback.format_exc())
        return _empty("Error building study")


# ============================================================================
#  Callbacks — to be filled by register_callbacks
# ============================================================================

def register_callbacks(app):
    """Register all Chart Lab callbacks."""

    # ── 1. Layout toggle ─────────────────────────────────────────────
    @app.callback(
        [Output("lab-layout-store", "data"), Output("lab-layout-1", "style"),
         Output("lab-layout-2", "style"), Output("lab-layout-4", "style")],
        [Input("lab-layout-1", "n_clicks"), Input("lab-layout-2", "n_clicks"),
         Input("lab-layout-4", "n_clicks")],
        prevent_initial_call=True)
    def _toggle_layout(n1, n2, n4):
        try:
            ctx = callback_context
            if not ctx.triggered: raise PreventUpdate
            btn = ctx.triggered[0]["prop_id"].split(".")[0]
            if btn == "lab-layout-1": return 1, BTN_TOGGLE_ACTIVE, BTN_TOGGLE_STYLE, BTN_TOGGLE_STYLE
            if btn == "lab-layout-4": return 4, BTN_TOGGLE_STYLE, BTN_TOGGLE_STYLE, BTN_TOGGLE_ACTIVE
            return 2, BTN_TOGGLE_STYLE, BTN_TOGGLE_ACTIVE, BTN_TOGGLE_STYLE
        except PreventUpdate: raise
        except Exception: return no_update, no_update, no_update, no_update

    # ── 2. Slot visibility ───────────────────────────────────────────
    @app.callback(
        [Output("lab-row-1", "style"), Output("lab-row-2", "style"),
         Output({"type": "lab-slot-wrapper", "index": 1}, "style"),
         Output({"type": "lab-slot-wrapper", "index": 2}, "style"),
         Output({"type": "lab-slot-wrapper", "index": 3}, "style")],
        Input("lab-layout-store", "data"))
    def _visibility(lc):
        lc = lc or 2
        r1 = {"display": "flex", "gap": GAP, "marginBottom": GAP}
        r2 = {**r1}
        s1, s2, s3 = {**SLOT_STYLE}, {**SLOT_STYLE}, {**SLOT_STYLE}
        if lc == 1:
            s1["display"] = "none"; r2["display"] = "none"
            s2["display"] = "none"; s3["display"] = "none"
        elif lc == 2:
            r2["display"] = "none"; s2["display"] = "none"; s3["display"] = "none"
        return r1, r2, s1, s2, s3

    # ── 3. Per-slot chart (MATCH) ────────────────────────────────────
    @app.callback(
        Output({"type": "lab-chart", "index": MATCH}, "figure"),
        [Input({"type": "lab-metric", "index": MATCH}, "value"),
         Input({"type": "lab-pairs", "index": MATCH}, "value"),
         Input({"type": "lab-tenor", "index": MATCH}, "value"),
         Input({"type": "lab-timeframe", "index": MATCH}, "value"),
         Input({"type": "lab-charttype", "index": MATCH}, "value"),
         Input({"type": "lab-normalize", "index": MATCH}, "value"),
         Input({"type": "lab-overlay", "index": MATCH}, "value"),
         Input("lab-refresh-counter", "data")])
    def _slot_chart(metric, pairs, tenor, timeframe, chart_type, normalize, overlay, _rc):
        try:
            if not metric: return _empty("Select a metric")
            if not pairs: return _empty("Select a pair")
            if isinstance(pairs, str): pairs = [pairs]
            return _build_slot_figure(metric, pairs, tenor or "1M", timeframe or 252,
                                      chart_type or "line", normalize or "raw", overlay or "")
        except Exception:
            logger.error("slot chart error: %s", traceback.format_exc())
            return _empty("Error")

    # ── 4. Per-slot stats (MATCH) ────────────────────────────────────
    @app.callback(
        Output({"type": "lab-slot-stats", "index": MATCH}, "children"),
        [Input({"type": "lab-metric", "index": MATCH}, "value"),
         Input({"type": "lab-pairs", "index": MATCH}, "value"),
         Input({"type": "lab-tenor", "index": MATCH}, "value"),
         Input({"type": "lab-timeframe", "index": MATCH}, "value")])
    def _slot_stats(metric, pairs, tenor, timeframe):
        try:
            if not metric or not pairs: return []
            if isinstance(pairs, str): pairs = [pairs]
            return _build_slot_stats(metric, pairs, tenor or "1M", timeframe or 252)
        except Exception:
            return []

    # ── 5. Slot summary (MATCH) ──────────────────────────────────────
    @app.callback(
        Output({"type": "lab-slot-summary", "index": MATCH}, "children"),
        [Input({"type": "lab-metric", "index": MATCH}, "value"),
         Input({"type": "lab-pairs", "index": MATCH}, "value"),
         Input({"type": "lab-tenor", "index": MATCH}, "value"),
         Input({"type": "lab-timeframe", "index": MATCH}, "value"),
         Input({"type": "lab-charttype", "index": MATCH}, "value"),
         Input({"type": "lab-overlay", "index": MATCH}, "value")])
    def _slot_summary(metric, pairs, tenor, timeframe, chart_type, overlay):
        try:
            if not pairs: return ""
            if isinstance(pairs, str): pairs = [pairs]
            p = ", ".join(pairs[:5])
            s = f"{p} | {metric or 'ATM'}"
            if overlay:
                s += f" vs {overlay}"
            s += f" | {tenor or '1M'} | {timeframe or 252}D | {(chart_type or 'line').upper()}"
            return s
        except Exception:
            return ""

    # ── 6. Comparison chart ──────────────────────────────────────────
    @app.callback(
        Output("lab-comp-chart", "figure"),
        [Input("lab-comp-refresh", "n_clicks"), Input("lab-comp-pairs", "value"),
         Input("lab-comp-type", "value"), Input("lab-comp-tenor", "value")])
    def _comparison(n, pairs, comp_type, tenor):
        try:
            if not pairs or len(pairs) < 2: return _empty("Select 2-5 pairs")
            return _build_comparison_figure(pairs, comp_type or "term_structure", tenor or "3M")
        except Exception:
            return _empty("Error")

    # ── 7. Deep study ────────────────────────────────────────────────
    @app.callback(
        Output("lab-study-chart", "figure"),
        [Input("lab-study-run", "n_clicks")],
        [State("lab-study-type", "value"), State("lab-study-pairs", "value"),
         State("lab-study-tenor", "value")],
        prevent_initial_call=True)
    def _deep_study(n, study_type, pairs, tenor):
        try:
            if not pairs: return _empty("Select pairs")
            if isinstance(pairs, str): pairs = [pairs]
            return _build_deep_study(study_type or "vol_deep_dive", pairs, tenor or "3M")
        except Exception:
            return _empty("Error building study")

    # ── 8. Deep study stats ──────────────────────────────────────────
    @app.callback(
        Output("lab-study-stats", "children"),
        Input("lab-study-chart", "figure"),
        [State("lab-study-type", "value"), State("lab-study-pairs", "value"),
         State("lab-study-tenor", "value")])
    def _deep_study_stats(fig, study_type, pairs, tenor):
        try:
            if not pairs: return []
            if isinstance(pairs, str): pairs = [pairs]
            pair = pairs[0]
            def _sv(l, v, c="#d4d4d4"):
                return html.Span([html.Span(f"{l}: ", style=STAT_ITEM),
                                   html.Span(str(v), style={**STAT_VAL, "color": c})])
            if study_type == "vol_deep_dive":
                r = vol_regime_detect(pair)
                z = vol_zscore(pair, tenor or "3M", "ATM")
                p = vol_percentile(pair, tenor or "3M", "ATM")
                return [_sv("Regime", r["regime"], r.get("color")),
                        _sv("Z-Score", f"{z['zscore']:+.2f}"),
                        _sv("Percentile", f"{p['percentile']:.0f}th")]
            if study_type == "smile_deep_dive":
                sk = smile_skewness(pair, tenor or "3M")
                return [_sv("Skew", f"{sk['rr_25d']:.2f}"), _sv("Direction", sk["direction"])]
            if study_type == "rv_scanner":
                return [_sv("Pairs scanned", str(len(pairs)))]
        except Exception: pass
        return []

    # ── 9. Scratchpad save ───────────────────────────────────────────
    @app.callback(
        [Output("lab-scratchpad-store", "data"), Output("lab-scratchpad-meta", "children")],
        Input("lab-scratchpad", "value"), prevent_initial_call=True)
    def _save_pad(text):
        text = text or ""
        return text, f"Saved {datetime.now().strftime('%H:%M')} | {len(text)} chars"

    # ── 10. Scratchpad load (only on initial page load) ──────────────
    @app.callback(
        Output("lab-scratchpad", "value"),
        Input("lab-scratchpad-store", "modified_timestamp"),
        State("lab-scratchpad-store", "data"),
        State("lab-scratchpad", "value"))
    def _load_pad(ts, stored, current):
        # Only load from store when textarea is still empty (initial load)
        if current:
            raise PreventUpdate
        return stored or ""

    # ── 11. PIN chart ────────────────────────────────────────────────
    @app.callback(
        Output("lab-pinned-store", "data", allow_duplicate=True),
        Input({"type": "lab-pin-btn", "index": ALL}, "n_clicks"),
        [State({"type": "lab-metric", "index": ALL}, "value"),
         State({"type": "lab-overlay", "index": ALL}, "value"),
         State({"type": "lab-pairs", "index": ALL}, "value"),
         State({"type": "lab-tenor", "index": ALL}, "value"),
         State({"type": "lab-timeframe", "index": ALL}, "value"),
         State({"type": "lab-charttype", "index": ALL}, "value"),
         State({"type": "lab-normalize", "index": ALL}, "value"),
         State("lab-pinned-store", "data")],
        prevent_initial_call=True)
    def _pin(clicks, metrics, overlays, pairs_l, tenors, tfs, cts, norms, pinned):
        ctx = callback_context
        if not ctx.triggered: raise PreventUpdate
        t = ctx.triggered[0]
        if not t["value"]: raise PreventUpdate
        i = json.loads(t["prop_id"].split(".")[0])["index"]
        prs = pairs_l[i] if i < len(pairs_l) else []
        if isinstance(prs, str): prs = [prs]
        m = metrics[i] if i < len(metrics) else "ATM"
        ov = overlays[i] if i < len(overlays) else ""
        label = f"{', '.join(prs[:5])} | {m}"
        if ov:
            label += f" vs {ov}"
        label += f" | {tenors[i] if i < len(tenors) else '1M'}"
        entry = {"label": label, "metric": m, "overlay": ov or "",
                 "pairs": prs, "tenor": tenors[i] if i < len(tenors) else "1M",
                 "timeframe": tfs[i] if i < len(tfs) else 252,
                 "chart_type": cts[i] if i < len(cts) else "line",
                 "normalize": norms[i] if i < len(norms) else "raw"}
        return list(pinned or []) + [entry]

    # ── 12. Render pinned list ───────────────────────────────────────
    @app.callback(Output("lab-pinned-list", "children"), Input("lab-pinned-store", "data"))
    def _render_pinned(pinned):
        if not pinned:
            return html.Div("No charts pinned yet.", style={"color": "#808080", "fontSize": "10px",
                                                             "fontFamily": _FONT, "padding": "8px"})
        items = []
        for idx, e in enumerate(pinned):
            items.append(html.Div([
                html.Span(e.get("label", "Chart"), id={"type": "lab-pinned-restore", "index": idx},
                           n_clicks=0, style={"color": "#d4d4d4", "fontSize": "10px", "fontFamily": _FONT,
                                              "cursor": "pointer", "flex": "1"}),
                html.Button("\u00D7", id={"type": "lab-pinned-unpin", "index": idx}, n_clicks=0,
                            style={"backgroundColor": "transparent", "color": "#ff3333", "border": "none",
                                   "cursor": "pointer", "fontFamily": _FONT, "fontSize": "12px", "padding": "0 4px"}),
            ], style={"display": "flex", "alignItems": "center", "padding": "4px 8px",
                      "borderBottom": "1px solid #222240"}))
        return items

    # ── 13. Unpin ────────────────────────────────────────────────────
    @app.callback(
        Output("lab-pinned-store", "data", allow_duplicate=True),
        Input({"type": "lab-pinned-unpin", "index": ALL}, "n_clicks"),
        State("lab-pinned-store", "data"), prevent_initial_call=True)
    def _unpin(clicks, pinned):
        ctx = callback_context
        if not ctx.triggered: raise PreventUpdate
        t = ctx.triggered[0]
        if not t["value"]: raise PreventUpdate
        i = json.loads(t["prop_id"].split(".")[0])["index"]
        p = list(pinned or [])
        if 0 <= i < len(p): p.pop(i)
        return p

    # ── 14. Restore pinned → slot 0 ─────────────────────────────────
    @app.callback(
        [Output({"type": "lab-metric", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-overlay", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-pairs", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-tenor", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-timeframe", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-charttype", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-normalize", "index": 0}, "value", allow_duplicate=True)],
        Input({"type": "lab-pinned-restore", "index": ALL}, "n_clicks"),
        State("lab-pinned-store", "data"), prevent_initial_call=True)
    def _restore(clicks, pinned):
        ctx = callback_context
        if not ctx.triggered: raise PreventUpdate
        t = ctx.triggered[0]
        if not t["value"]: raise PreventUpdate
        i = json.loads(t["prop_id"].split(".")[0])["index"]
        p = pinned or []
        if i < 0 or i >= len(p): raise PreventUpdate
        e = p[i]
        return (e.get("metric", "ATM"), e.get("overlay", ""), e.get("pairs", ["EURUSD"]),
                e.get("tenor", "1M"), e.get("timeframe", 252), e.get("chart_type", "line"),
                e.get("normalize", "raw"))

    # ── 15. Preset selector ──────────────────────────────────────────
    @app.callback(
        [Output({"type": "lab-metric", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-overlay", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-pairs", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-tenor", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-timeframe", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-charttype", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-normalize", "index": 0}, "value", allow_duplicate=True),
         Output({"type": "lab-metric", "index": 1}, "value", allow_duplicate=True),
         Output({"type": "lab-overlay", "index": 1}, "value", allow_duplicate=True),
         Output({"type": "lab-pairs", "index": 1}, "value", allow_duplicate=True),
         Output({"type": "lab-tenor", "index": 1}, "value", allow_duplicate=True),
         Output({"type": "lab-timeframe", "index": 1}, "value", allow_duplicate=True),
         Output({"type": "lab-charttype", "index": 1}, "value", allow_duplicate=True),
         Output({"type": "lab-normalize", "index": 1}, "value", allow_duplicate=True),
         Output({"type": "lab-metric", "index": 2}, "value", allow_duplicate=True),
         Output({"type": "lab-overlay", "index": 2}, "value", allow_duplicate=True),
         Output({"type": "lab-pairs", "index": 2}, "value", allow_duplicate=True),
         Output({"type": "lab-tenor", "index": 2}, "value", allow_duplicate=True),
         Output({"type": "lab-timeframe", "index": 2}, "value", allow_duplicate=True),
         Output({"type": "lab-charttype", "index": 2}, "value", allow_duplicate=True),
         Output({"type": "lab-normalize", "index": 2}, "value", allow_duplicate=True),
         Output({"type": "lab-metric", "index": 3}, "value", allow_duplicate=True),
         Output({"type": "lab-overlay", "index": 3}, "value", allow_duplicate=True),
         Output({"type": "lab-pairs", "index": 3}, "value", allow_duplicate=True),
         Output({"type": "lab-tenor", "index": 3}, "value", allow_duplicate=True),
         Output({"type": "lab-timeframe", "index": 3}, "value", allow_duplicate=True),
         Output({"type": "lab-charttype", "index": 3}, "value", allow_duplicate=True),
         Output({"type": "lab-normalize", "index": 3}, "value", allow_duplicate=True),
         Output("lab-layout-store", "data", allow_duplicate=True),
         Output("lab-layout-1", "style", allow_duplicate=True),
         Output("lab-layout-2", "style", allow_duplicate=True),
         Output("lab-layout-4", "style", allow_duplicate=True)],
        Input("lab-preset-dropdown", "value"), prevent_initial_call=True)
    def _preset(key):
        if not key or key not in PRESETS: raise PreventUpdate
        p = PRESETS[key]
        slots = p["slots"]
        lc = p.get("layout", 2)
        out = []
        for i in range(4):
            s = slots[i] if i < len(slots) else {}
            out.extend([s.get("metric", "ATM"), s.get("overlay", ""), s.get("pairs", ["EURUSD"]),
                        s.get("tenor", "1M"), s.get("timeframe", 252), s.get("chart_type", "line"),
                        s.get("normalize", "raw")])
        if lc == 1: out.extend([1, BTN_TOGGLE_ACTIVE, BTN_TOGGLE_STYLE, BTN_TOGGLE_STYLE])
        elif lc == 4: out.extend([4, BTN_TOGGLE_STYLE, BTN_TOGGLE_STYLE, BTN_TOGGLE_ACTIVE])
        else: out.extend([2, BTN_TOGGLE_STYLE, BTN_TOGGLE_ACTIVE, BTN_TOGGLE_STYLE])
        return tuple(out)

    # ── 16. LINK toggle ──────────────────────────────────────────────
    @app.callback(
        [Output("lab-link-store", "data"), Output("lab-link-toggle", "style")],
        Input("lab-link-toggle", "n_clicks"), State("lab-link-store", "data"),
        prevent_initial_call=True)
    def _link_toggle(n, linked):
        new = not (linked or False)
        style = {**BTN_TOGGLE_ACTIVE, "fontSize": "9px", "color": "#000000"} if new else {**BTN_TOGGLE_STYLE, "fontSize": "9px"}
        return new, style

    # ── 17. Cross-slot pair sync ─────────────────────────────────────
    @app.callback(
        [Output({"type": "lab-pairs", "index": 1}, "value", allow_duplicate=True),
         Output({"type": "lab-pairs", "index": 2}, "value", allow_duplicate=True),
         Output({"type": "lab-pairs", "index": 3}, "value", allow_duplicate=True)],
        Input({"type": "lab-pairs", "index": 0}, "value"),
        State("lab-link-store", "data"), prevent_initial_call=True)
    def _sync_pairs(p0, linked):
        if not linked: raise PreventUpdate
        return p0, p0, p0

    # ── 18. LIVE toggle ──────────────────────────────────────────────
    @app.callback(
        [Output("lab-live-store", "data"), Output("lab-live-toggle", "style"),
         Output("lab-live-indicator", "style"), Output("lab-auto-refresh-interval", "disabled")],
        Input("lab-live-toggle", "n_clicks"), State("lab-live-store", "data"),
        prevent_initial_call=True)
    def _live_toggle(n, live):
        new = not (live or False)
        if new:
            return (True, {**BTN_TOGGLE_ACTIVE, "fontSize": "9px", "color": "#000000"},
                    {"width": "6px", "height": "6px", "borderRadius": "50%", "backgroundColor": "#00cc66",
                     "display": "inline-block", "marginLeft": "4px"}, False)
        return (False, {**BTN_TOGGLE_STYLE, "fontSize": "9px"},
                {"width": "6px", "height": "6px", "borderRadius": "50%", "backgroundColor": "transparent",
                 "display": "inline-block", "marginLeft": "4px"}, True)

    # ── 19. Auto-refresh counter ─────────────────────────────────────
    @app.callback(
        Output("lab-refresh-counter", "data"),
        Input("lab-auto-refresh-interval", "n_intervals"),
        State("lab-live-store", "data"), prevent_initial_call=True)
    def _auto_refresh(n, live):
        if not live: raise PreventUpdate
        return n

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output("lab-csv-download", "data"),
        [Input({"type": "lab-csv-btn", "index": ALL}, "n_clicks"),
         Input("lab-csv-study", "n_clicks"),
         Input("lab-csv-comp", "n_clicks")],
        [State({"type": "lab-chart", "index": ALL}, "figure"),
         State("lab-study-chart", "figure"),
         State("lab-comp-chart", "figure")],
        prevent_initial_call=True,
    )
    def lab_csv_export(slot_clicks, study_n, comp_n, slot_figs, study_fig, comp_fig):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        prop_id = ctx.triggered[0]["prop_id"]
        # Pattern-matching buttons
        if "lab-csv-btn" in prop_id:
            import json as _json
            info = _json.loads(prop_id.split(".")[0])
            idx = info["index"]
            if idx < len(slot_figs) and slot_figs[idx]:
                return export_csv(slot_figs[idx], "ChartLab", f"Slot{idx}")
            return no_update
        btn = prop_id.split(".")[0]
        if btn == "lab-csv-study" and study_fig:
            return export_csv(study_fig, "ChartLab", "Study")
        if btn == "lab-csv-comp" and comp_fig:
            return export_csv(comp_fig, "ChartLab", "Comparison")
        return no_update
