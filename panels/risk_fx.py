"""
Multi-Pair FX Risk Dashboard
==============================
Institutional FX options risk panel with 5 tabbed views:
  1. Greeks (vega heatmap, delta/gamma bars)
  2. VaR (distribution, component VaR)
  3. Stress (scenario bars, custom builder, per-position table)
  4. P&L Attribution (waterfall, by-pair stacked bar)
  5. What-If (add trade, impact preview, hedge suggestions)

Plus 8 KPI stat boxes and a persistent position table at the bottom.
"""

import dash
from dash import html, dcc, Input, Output, State, no_update, dash_table, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE, LABEL_STYLE,
    DROPDOWN_STYLE, INPUT_STYLE, BUTTON_STYLE,
    GAP, SECTION_GAP, CHART_SM, CHART_MD, CHART_LG,
    clickable_stat, chart_layout, CSV_BTN_STYLE,
    no_data_fig,
)
from core.csv_export import export_csv
from core.bloomberg_fx import get_fx_vol_surface, get_fx_spots, get_fx_rates, get_all_pairs
from core.fx_portfolio import (
    get_all_positions, compute_portfolio_risk, pnl_attribution,
    vega_by_bucket, gamma_by_bucket, delta_by_pair, check_risk_limits,
    exposure_summary, hedge_suggestion, what_if_add, create_sample_portfolio,
    BOOKS, TENOR_BUCKETS,
)
from core.fx_stress import (
    get_scenarios, stress_portfolio, compare_scenarios, custom_stress,
    FX_STRESS_SCENARIOS,
)
from core.fx_analytics import historical_var, expected_shortfall, parametric_var, vol_percentile, vol_regime_detect
from core.fx_conventions import FX_PAIR_REGISTRY


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

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

TAB_STYLE = {
    "backgroundColor": "transparent",
    "border": f"1px solid {COLORS['border']}",
    "borderBottom": "none",
    "borderRadius": "0px",
    "color": COLORS["text_muted"],
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "11px",
    "fontWeight": "600",
    "padding": "14px 24px",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["accent_cyan"],
    "borderBottom": "none",
    "borderTop": f"2px solid {COLORS['accent_cyan']}",
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

BUTTON_DANGER_STYLE = {
    **BUTTON_STYLE,
    "backgroundColor": COLORS["accent_red"],
}

BUTTON_SUCCESS_STYLE = {
    **BUTTON_STYLE,
    "backgroundColor": COLORS["accent_green"],
}


def _make_stat_style(color=None):
    style = {**STAT_BOX_STYLE}
    if color:
        style["borderLeft"] = f"3px solid {color}"
    return style


def _fmt_usd(val):
    """Format a USD value with sign and K/M suffix."""
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:+,.2f}M"
    if abs(val) >= 1_000:
        return f"${val / 1_000:+,.1f}K"
    return f"${val:+,.0f}"


def _pnl_color(val):
    if val > 0:
        return COLORS["pnl_profit"]
    if val < 0:
        return COLORS["pnl_loss"]
    return COLORS["pnl_neutral"]


def _severity_color(severity):
    mapping = {
        "EXTREME": COLORS["accent_red"],
        "SEVERE": COLORS["accent_orange"],
        "MODERATE": COLORS["accent_amber"],
        "CUSTOM": COLORS["accent_purple"],
    }
    return mapping.get(severity, COLORS["accent_cyan"])


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

def _safe_atm_vol(surface):
    """Safely extract ATM vol from a vol surface dict or scalar.

    Handles the case where ``vol_surfaces`` values are nested dicts
    (keyed by tenor -> strike type) instead of plain floats.  Always
    returns a decimal vol (e.g. 0.08 for 8%).
    """
    if surface is None:
        return 0.10
    if isinstance(surface, dict):
        for tenor in ("3M", "1M", "6M", "1Y"):
            if tenor in surface and isinstance(surface[tenor], dict):
                v = surface[tenor].get("atm", 8.0)
                return v / 100.0 if v > 1.0 else v
        # Fallback: if the dict has no recognized tenor keys, try to
        # find any dict-valued entry with an "atm" key
        for k, v in surface.items():
            if isinstance(v, dict) and "atm" in v:
                raw = v["atm"]
                return raw / 100.0 if raw > 1.0 else raw
    if isinstance(surface, (int, float)):
        return surface if surface < 1.0 else surface / 100.0
    return 0.10


def _load_market_data():
    """Load spots, rates, and vol surfaces for the whole portfolio."""
    pairs = list(FX_PAIR_REGISTRY.keys())
    spots_raw = get_fx_spots(pairs) or {}
    spots = {}
    for p, data in spots_raw.items():
        if isinstance(data, dict):
            spots[p] = data.get("mid", data.get("bid", 1.0))
        else:
            spots[p] = float(data)

    rates = {}
    vol_surfaces = {}
    for p in pairs:
        r = get_fx_rates(p)
        if isinstance(r, dict):
            rates[p] = {"r_d": r.get("r_dom", 0.04), "r_f": r.get("r_for", 0.02)}
        else:
            rates[p] = {"r_d": 0.04, "r_f": 0.02}
        surf = get_fx_vol_surface(p)
        # Use _safe_atm_vol so nested dicts never blow up downstream
        vol_surfaces[p] = _safe_atm_vol(surf)

    return spots, rates, vol_surfaces


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout():
    all_pairs = sorted(FX_PAIR_REGISTRY.keys())
    scenario_opts = [
        {"label": sc["display_name"], "value": sc["name"]}
        for sc in get_scenarios()
    ]

    return html.Div([
        # CSV download
        dcc.Download(id="fxrisk-csv-download"),
        # Hidden stores
        dcc.Store(id="fxrisk-init-flag", data=False),
        dcc.Store(id="fxrisk-selected-pair", data=None),
        dcc.Interval(id="fxrisk-interval", interval=210_000, n_intervals=0),

        # ---- 8 KPI Stat Boxes ----
        html.Div(id="fxrisk-stat-boxes", className="stat-row", style={
            "marginBottom": SECTION_GAP,
        }),

        # ---- Risk Limits Configuration ----
        html.Div([
            html.Div("RISK LIMITS", style={
                **CARD_HEADER_STYLE, "marginBottom": "12px",
                "cursor": "pointer", "display": "inline-block",
            }),
            html.Div([
                html.Div([
                    html.Label("DELTA LIMIT ($M)", style=LABEL_STYLE),
                    dcc.Input(
                        id="fxrisk-limit-delta", type="number",
                        value=10, step=1, min=1, max=500,
                        style={**INPUT_STYLE, "width": "100px"},
                        debounce=True,
                    ),
                ], style={"flex": "1", "minWidth": "120px"}),
                html.Div([
                    html.Label("VEGA LIMIT ($K)", style=LABEL_STYLE),
                    dcc.Input(
                        id="fxrisk-limit-vega", type="number",
                        value=500, step=50, min=10, max=10000,
                        style={**INPUT_STYLE, "width": "100px"},
                        debounce=True,
                    ),
                ], style={"flex": "1", "minWidth": "120px"}),
                html.Div([
                    html.Label("GAMMA LIMIT ($K)", style=LABEL_STYLE),
                    dcc.Input(
                        id="fxrisk-limit-gamma", type="number",
                        value=100, step=10, min=5, max=5000,
                        style={**INPUT_STYLE, "width": "100px"},
                        debounce=True,
                    ),
                ], style={"flex": "1", "minWidth": "120px"}),
            ], style={"display": "flex", "gap": GAP, "flexWrap": "wrap"}),
        ], style={**CARD_STYLE, "marginBottom": SECTION_GAP, "padding": "12px 18px"}),

        # ---- Tabbed Risk Views ----
        html.Div([
            dcc.Tabs(id="fxrisk-tabs", value="greeks", children=[
                dcc.Tab(label="GREEKS", value="greeks",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="VaR", value="var",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="STRESS", value="stress",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="ATTRIBUTION", value="attribution",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="WHAT-IF", value="whatif",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="HEDGE", value="hedge",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            ], style={"marginBottom": "0"}),
        ], style={"marginBottom": SECTION_GAP}),

        # ---- Tab content containers (visibility toggled by callback) ----
        # Greeks
        html.Div(id="fxrisk-greeks-container", style={"display": "none"}, children=[
            html.Div([
                html.Div("VEGA HEATMAP (PAIR x TENOR BUCKET)", style=CARD_HEADER_STYLE),
                html.Button("CSV", id="fxrisk-csv-vega", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="fxrisk-vega-heatmap", config={"displayModeBar": False}),
            ], style={**CARD_STYLE, "marginBottom": SECTION_GAP}),
            html.Div(id="fxrisk-click-detail", style={
                "border": f"1px solid {COLORS['border_subtle']}",
                "padding": GAP,
                "marginTop": GAP,
                "marginBottom": GAP,
                "display": "none",
            }),
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("DELTA BY PAIR (USD)", style=CARD_HEADER_STYLE),
                        html.Button("CSV", id="fxrisk-csv-delta", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="fxrisk-delta-bar", config={"displayModeBar": False}),
                    ], style={"flex": "1", "minWidth": "400px"}),
                    html.Div([
                        html.Div("GAMMA BY PAIR (USD/%)", style=CARD_HEADER_STYLE),
                        html.Button("CSV", id="fxrisk-csv-gamma", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="fxrisk-gamma-bar", config={"displayModeBar": False}),
                    ], style={"flex": "1", "minWidth": "400px"}),
                ], style={"display": "flex", "gap": SECTION_GAP, "flexWrap": "wrap"}),
            ]),
        ]),

        # VaR
        html.Div(id="fxrisk-var-container", style={"display": "none"}, children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("VaR DISTRIBUTION (10K SIMULATIONS)", style=CARD_HEADER_STYLE),
                        html.Button("CSV", id="fxrisk-csv-var", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="fxrisk-var-dist", config={"displayModeBar": False}),
                    ], style={"flex": "3", "minWidth": "500px"}),
                    html.Div([
                        html.Div("COMPONENT VaR BY PAIR", style=CARD_HEADER_STYLE),
                        html.Button("CSV", id="fxrisk-csv-cvar", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="fxrisk-component-var", config={"displayModeBar": False}),
                    ], style={"flex": "2", "minWidth": "350px"}),
                ], style={"display": "flex", "gap": SECTION_GAP, "flexWrap": "wrap"}),
            ]),
        ]),

        # Stress
        html.Div(id="fxrisk-stress-container", style={"display": "none"}, children=[
            html.Div([
                html.Div("SCENARIO COMPARISON (ALL SCENARIOS)", style=CARD_HEADER_STYLE),
                html.Button("CSV", id="fxrisk-csv-scenario", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="fxrisk-scenario-bars", config={"displayModeBar": False}),
            ], style={**CARD_STYLE, "marginBottom": SECTION_GAP}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("CUSTOM SCENARIO BUILDER", style=CARD_HEADER_STYLE),
                        html.Div([
                            html.Label("SPOT SHOCK (%)", style=LABEL_STYLE),
                            dcc.Slider(
                                id="fxrisk-stress-spot", min=-20, max=20, step=0.5,
                                value=0, marks={i: f"{i}%" for i in range(-20, 25, 5)},
                                tooltip={"placement": "bottom"},
                            ),
                        ], style={"marginBottom": "18px"}),
                        html.Div([
                            html.Label("VOL MULTIPLIER", style=LABEL_STYLE),
                            dcc.Slider(
                                id="fxrisk-stress-vol", min=0.5, max=4.0, step=0.1,
                                value=1.0, marks={v: f"{v}x" for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]},
                                tooltip={"placement": "bottom"},
                            ),
                        ], style={"marginBottom": "18px"}),
                        html.Div([
                            html.Label("RATE SHOCK (BPS)", style=LABEL_STYLE),
                            dcc.Slider(
                                id="fxrisk-stress-rate", min=-200, max=200, step=5,
                                value=0, marks={i: f"{i}" for i in range(-200, 250, 50)},
                                tooltip={"placement": "bottom"},
                            ),
                        ], style={"marginBottom": "18px"}),
                        html.Button("RUN STRESS", id="fxrisk-run-stress",
                                    style=BUTTON_STYLE, n_clicks=0),
                        html.Div(id="fxrisk-custom-stress-result", style={
                            "marginTop": SECTION_GAP,
                            "color": COLORS["text_primary"],
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "13px",
                        }),
                    ], style={"flex": "1", "minWidth": "400px"}),
                    html.Div([
                        html.Div("PER-POSITION IMPACT", style=CARD_HEADER_STYLE),
                        html.Div(id="fxrisk-stress-position-table"),
                    ], style={"flex": "1", "minWidth": "400px"}),
                ], style={"display": "flex", "gap": SECTION_GAP, "flexWrap": "wrap"}),
            ], style=CARD_STYLE),
        ]),

        # Attribution
        html.Div(id="fxrisk-attr-container", style={"display": "none"}, children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("P&L ATTRIBUTION WATERFALL", style=CARD_HEADER_STYLE),
                        html.Button("CSV", id="fxrisk-csv-waterfall", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="fxrisk-waterfall", config={"displayModeBar": False}),
                    ], style={"flex": "1", "minWidth": "500px"}),
                    html.Div([
                        html.Div("P&L BY PAIR", style=CARD_HEADER_STYLE),
                        html.Button("CSV", id="fxrisk-csv-pnl", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="fxrisk-pnl-pair", config={"displayModeBar": False}),
                    ], style={"flex": "1", "minWidth": "400px"}),
                ], style={"display": "flex", "gap": SECTION_GAP, "flexWrap": "wrap"}),
            ]),
        ]),

        # What-If
        html.Div(id="fxrisk-whatif-container", style={"display": "none"}, children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("ADD TRADE", style=CARD_HEADER_STYLE),
                        html.Div([
                            html.Div([
                                html.Label("PAIR", style=LABEL_STYLE),
                                dcc.Dropdown(
                                    id="fxrisk-wi-pair",
                                    options=[{"label": p, "value": p} for p in all_pairs],
                                    value="EURUSD",
                                    clearable=False,
                                    style={"fontSize": "12px"},
                                ),
                            ], style={"flex": "1", "minWidth": "120px", "marginRight": GAP}),
                            html.Div([
                                html.Label("TYPE", style=LABEL_STYLE),
                                dcc.Dropdown(
                                    id="fxrisk-wi-type",
                                    options=[
                                        {"label": "CALL", "value": "call"},
                                        {"label": "PUT", "value": "put"},
                                    ],
                                    value="call",
                                    clearable=False,
                                    style={"fontSize": "12px"},
                                ),
                            ], style={"flex": "1", "minWidth": "90px", "marginRight": GAP}),
                            html.Div([
                                html.Label("DIRECTION", style=LABEL_STYLE),
                                dcc.Dropdown(
                                    id="fxrisk-wi-dir",
                                    options=[
                                        {"label": "BUY", "value": "buy"},
                                        {"label": "SELL", "value": "sell"},
                                    ],
                                    value="buy",
                                    clearable=False,
                                    style={"fontSize": "12px"},
                                ),
                            ], style={"flex": "1", "minWidth": "90px", "marginRight": GAP}),
                        ], style={"display": "flex", "flexWrap": "wrap", "gap": GAP,
                                  "marginBottom": "14px"}),
                        html.Div([
                            html.Div([
                                html.Label("DELTA", style=LABEL_STYLE),
                                dcc.Input(id="fxrisk-wi-delta", type="number",
                                          value=0.25, step=0.05, min=0.01, max=0.99,
                                          style=INPUT_STYLE, debounce=True),
                            ], style={"flex": "1", "minWidth": "80px", "marginRight": GAP}),
                            html.Div([
                                html.Label("TENOR", style=LABEL_STYLE),
                                dcc.Dropdown(
                                    id="fxrisk-wi-tenor",
                                    options=[{"label": t, "value": t} for t in
                                             ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]],
                                    value="3M",
                                    clearable=False,
                                    style={"fontSize": "12px"},
                                ),
                            ], style={"flex": "1", "minWidth": "80px", "marginRight": GAP}),
                            html.Div([
                                html.Label("NOTIONAL (M)", style=LABEL_STYLE),
                                dcc.Input(id="fxrisk-wi-notional", type="number",
                                          value=10, step=1, min=1, max=500,
                                          style=INPUT_STYLE, debounce=True),
                            ], style={"flex": "1", "minWidth": "100px"}),
                        ], style={"display": "flex", "flexWrap": "wrap", "gap": GAP,
                                  "marginBottom": "18px"}),
                        html.Button("PREVIEW IMPACT", id="fxrisk-wi-preview",
                                    style=BUTTON_STYLE, n_clicks=0),
                    ], style={"flex": "1", "minWidth": "380px"}),
                    html.Div([
                        html.Div("IMPACT PREVIEW (BEFORE / AFTER)", style=CARD_HEADER_STYLE),
                        html.Div(id="fxrisk-wi-impact-table"),
                    ], style={"flex": "1", "minWidth": "380px"}),
                ], style={"display": "flex", "gap": SECTION_GAP, "flexWrap": "wrap"}),
            ], style={**CARD_STYLE, "marginBottom": SECTION_GAP}),
            html.Div([
                html.Div("HEDGE SUGGESTIONS", style=CARD_HEADER_STYLE),
                html.Div([
                    html.Button("NEUTRALIZE DELTA", id="fxrisk-hedge-delta",
                                style=BUTTON_SUCCESS_STYLE, n_clicks=0),
                    html.Button("NEUTRALIZE VEGA 3M", id="fxrisk-hedge-vega",
                                style={**BUTTON_STYLE, "marginLeft": GAP,
                                       "backgroundColor": COLORS["accent_purple"],
                                       },
                                n_clicks=0),
                ], style={"marginBottom": SECTION_GAP}),
                html.Div(id="fxrisk-hedge-result"),
            ], style=CARD_STYLE),
        ]),

        # ---- HEDGE Tab ----
        html.Div(id="fxrisk-hedge-container", style={"display": "none"}, children=[
            html.Div([
                html.Div("DELTA HEDGE SIMULATOR", style=CARD_HEADER_STYLE),
                html.Div([
                    html.Div([
                        html.Label("PAIR", style=LABEL_STYLE),
                        dcc.Dropdown(id="fxrisk-hsim-pair",
                                     options=[{"label": p, "value": p} for p in
                                              ["EURUSD","USDJPY","GBPUSD","USDCHF","AUDUSD",
                                               "NZDUSD","USDCAD","EURGBP","EURJPY","GBPJPY"]],
                                     value="EURUSD", clearable=False, style={"fontSize": "11px"}),
                    ], style={"flex": "1", "minWidth": "100px"}),
                    html.Div([
                        html.Label("TYPE", style=LABEL_STYLE),
                        dcc.Dropdown(id="fxrisk-hsim-cp",
                                     options=[{"label": "Call", "value": "Call"},
                                              {"label": "Put", "value": "Put"}],
                                     value="Call", clearable=False, style={"fontSize": "11px"}),
                    ], style={"flex": "1", "minWidth": "80px"}),
                    html.Div([
                        html.Label("DELTA", style=LABEL_STYLE),
                        dcc.Dropdown(id="fxrisk-hsim-delta",
                                     options=[{"label": "25Δ", "value": 0.25},
                                              {"label": "50Δ", "value": 0.50}],
                                     value=0.25, clearable=False, style={"fontSize": "11px"}),
                    ], style={"flex": "1", "minWidth": "80px"}),
                    html.Div([
                        html.Label("NOTIONAL", style=LABEL_STYLE),
                        dcc.Input(id="fxrisk-hsim-notional", type="number",
                                  value=10_000_000, step=1_000_000, style=INPUT_STYLE),
                    ], style={"flex": "1", "minWidth": "120px"}),
                    html.Div([
                        html.Label("FREQUENCY", style=LABEL_STYLE),
                        dcc.Dropdown(id="fxrisk-hsim-freq",
                                     options=[{"label": "Daily", "value": "Daily"},
                                              {"label": "Weekly", "value": "Weekly"},
                                              {"label": "No Hedge", "value": "No Hedge"}],
                                     value="Daily", clearable=False, style={"fontSize": "11px"}),
                    ], style={"flex": "1", "minWidth": "100px"}),
                ], style={"display": "flex", "gap": GAP, "flexWrap": "wrap", "marginBottom": GAP}),
                html.Button("RUN SIMULATION", id="fxrisk-hsim-run", style=BUTTON_STYLE, n_clicks=0),
                html.Div(id="fxrisk-hsim-stats", style={
                    "display": "flex", "gap": GAP, "marginTop": GAP, "flexWrap": "wrap",
                }),
                html.Div([
                    html.Div([
                        html.Button("CSV", id="fxrisk-csv-hsim-pnl", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="fxrisk-hsim-pnl", config={"displayModeBar": False}),
                    ], style={"flex": "1", "minWidth": "350px"}),
                    html.Div([
                        html.Button("CSV", id="fxrisk-csv-hsim-gamma", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="fxrisk-hsim-gamma", config={"displayModeBar": False}),
                    ], style={"flex": "1", "minWidth": "350px"}),
                ], style={"display": "flex", "gap": GAP, "marginTop": GAP}),
            ], style={**CARD_STYLE, "marginBottom": SECTION_GAP}),

            # Cross-hedge section
            html.Div([
                html.Div("CROSS-HEDGE OPTIMIZER", style=CARD_HEADER_STYLE),
                html.Div([
                    html.Div([
                        html.Label("TARGET", style=LABEL_STYLE),
                        dcc.Dropdown(id="fxrisk-xh-target",
                                     options=[{"label": p, "value": p} for p in
                                              ["EURJPY","GBPJPY","AUDJPY","EURCHF","GBPCHF"]],
                                     value="EURJPY", clearable=False, style={"fontSize": "11px"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("HEDGE 1", style=LABEL_STYLE),
                        dcc.Dropdown(id="fxrisk-xh-h1",
                                     options=[{"label": p, "value": p} for p in
                                              ["EURUSD","GBPUSD","AUDUSD","USDCHF","USDJPY"]],
                                     value="EURUSD", clearable=False, style={"fontSize": "11px"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("HEDGE 2", style=LABEL_STYLE),
                        dcc.Dropdown(id="fxrisk-xh-h2",
                                     options=[{"label": p, "value": p} for p in
                                              ["EURUSD","GBPUSD","AUDUSD","USDCHF","USDJPY"]],
                                     value="USDJPY", clearable=False, style={"fontSize": "11px"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": GAP, "marginBottom": GAP}),
                html.Button("COMPUTE", id="fxrisk-xh-run", style=BUTTON_STYLE, n_clicks=0),
                html.Div(id="fxrisk-xh-stats", style={
                    "display": "flex", "gap": GAP, "marginTop": GAP, "flexWrap": "wrap",
                }),
                html.Button("CSV", id="fxrisk-csv-xh", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="fxrisk-xh-chart", config={"displayModeBar": False},
                          style={"height": f"{CHART_SM}px", "marginTop": GAP}),
            ], style={**CARD_STYLE, "marginBottom": SECTION_GAP}),

            # Effectiveness monitor
            html.Div([
                html.Div("HEDGE EFFECTIVENESS MONITOR", style=CARD_HEADER_STYLE),
                html.Div(id="fxrisk-heff-table"),
            ], style=CARD_STYLE),
        ]),

        # ---- Position Table (always visible) ----
        html.Div([
            html.Div("POSITION BOOK", style=CARD_HEADER_STYLE),
            html.Div(id="fxrisk-position-table"),
        ], style=CARD_STYLE),

    ], style={"padding": "0"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):

    # -----------------------------------------------------------------------
    # 1. Main callback: init portfolio, compute risk, update stat boxes
    # -----------------------------------------------------------------------
    @app.callback(
        [
            Output("fxrisk-stat-boxes", "children"),
            Output("fxrisk-position-table", "children"),
            Output("fxrisk-init-flag", "data"),
        ],
        [
            Input("fxrisk-interval", "n_intervals"),
            Input("fxrisk-tabs", "value"),
            Input("fxrisk-limit-delta", "value"),
            Input("fxrisk-limit-vega", "value"),
            Input("fxrisk-limit-gamma", "value"),
        ],
        [State("fxrisk-init-flag", "data")],
        prevent_initial_call=False,
    )
    def update_main_risk(n_intervals, tab, limit_delta_m, limit_vega_k,
                         limit_gamma_k, init_flag):
        # Initialize portfolio on first load
        if not init_flag:
            positions = get_all_positions()
            if len(positions) == 0:
                create_sample_portfolio()

        positions = get_all_positions()
        if len(positions) == 0:
            empty_msg = html.Div("No positions loaded.", style={
                "color": COLORS["text_muted"], "padding": "20px"})
            return [empty_msg], empty_msg, True

        spots, rates, vol_surfaces = _load_market_data()

        # Portfolio risk
        risk = compute_portfolio_risk(spots, rates, vol_surfaces)
        totals = risk.get("totals", {})

        # Limit breaches -- use configurable limits from UI inputs
        custom_limits = {
            "max_total_delta": (limit_delta_m or 10) * 1_000_000,
            "max_delta_per_pair": (limit_delta_m or 10) * 1_000_000 / 2,
            "max_total_vega": (limit_vega_k or 500) * 1_000,
            "max_vega_per_pair": (limit_vega_k or 500) * 1_000 / 3,
            "max_vega_per_tenor_bucket": (limit_vega_k or 500) * 1_000 / 5,
            "max_gamma_per_pair": (limit_gamma_k or 100) * 1_000,
            "max_daily_theta": -50_000,
            "max_var_95_1d": 500_000,
            "max_notional_per_pair": 100_000_000,
            "max_em_notional_pct": 0.30,
        }
        breaches = check_risk_limits(risk, limits=custom_limits)
        n_breaches = len([b for b in breaches if b["severity"] in ("BREACH", "CRITICAL")])

        # Unrealised P&L estimate (price field from risk is mark-to-market)
        unreal_pnl = totals.get("price", 0.0)

        # VaR (parametric quick estimate)
        total_notional = sum(p["notional"] for p in positions)
        avg_vol = np.mean([
            _safe_atm_vol(vol_surfaces.get(p["pair"], 0.10))
            for p in positions
        ])
        var_result = parametric_var(avg_vol, total_notional, 0.95, 1)
        var_95 = var_result.get("var", 0.0)

        # Generate scenarios for CVaR
        rng = np.random.RandomState(seed=42)
        daily_vol = avg_vol / np.sqrt(252)
        sim_returns = rng.normal(0, daily_vol, 2_000)
        sim_pnl = sim_returns * total_notional
        sorted_pnl = np.sort(sim_pnl)
        cvar_idx = max(int(0.05 * len(sorted_pnl)), 1)
        cvar_95 = -np.mean(sorted_pnl[:cvar_idx])

        # Build stat boxes
        stats = [
            ("PORTFOLIO DELTA", totals.get("delta", 0), COLORS["accent_cyan"]),
            ("TOTAL GAMMA", totals.get("gamma", 0), COLORS["accent_blue"]),
            ("TOTAL VEGA", totals.get("vega", 0), COLORS["accent_purple"]),
            ("DAILY THETA", totals.get("theta", 0), COLORS["accent_orange"]),
            ("UNREALIZED P&L", unreal_pnl, _pnl_color(unreal_pnl)),
            ("VaR 95% 1d", -var_95, COLORS["accent_red"]),
            ("CVaR 95%", -cvar_95, COLORS["accent_rose"]),
            ("LIMIT BREACHES", n_breaches, COLORS["accent_red"] if n_breaches > 0 else COLORS["accent_green"]),
        ]

        stat_boxes = []
        for label, value, color in stats:
            if label == "LIMIT BREACHES":
                display_val = str(int(value))
            else:
                display_val = _fmt_usd(value)
            stat_boxes.append(
                html.Div([
                    html.Div(label, style={
                        "color": COLORS["text_secondary"],
                        "fontSize": "9px",
                        "fontWeight": "700",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "letterSpacing": "1.2px",
                        "textTransform": "uppercase",
                        "marginBottom": GAP,
                    }),
                    html.Div(display_val, style={
                        "color": color,
                        "fontSize": "18px",
                        "fontWeight": "800",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                ], style=_make_stat_style(color))
            )

        # Breach warning banner
        if n_breaches > 0:
            breach_details = [b for b in breaches if b["severity"] in ("BREACH", "CRITICAL")]
            breach_text = " | ".join(
                f"{b.get('metric', '?').upper()}: {b.get('current', '?')} vs limit {b.get('limit', '?')}"
                for b in breach_details[:4]
            )
            stat_boxes.insert(0, html.Div(
                f"\u26A0 LIMIT BREACH: {breach_text}",
                style={
                    "backgroundColor": "rgba(255,51,51,0.12)",
                    "border": "1px solid #ff3333",
                    "color": "#ff3333",
                    "padding": "6px 12px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontSize": "10px",
                    "fontWeight": "700",
                    "letterSpacing": "0.5px",
                    "width": "100%",
                    "marginBottom": GAP,
                },
            ))

        # Build position table
        pos_table = _build_position_table(positions, spots, rates, vol_surfaces)

        return stat_boxes, pos_table, True

    # -----------------------------------------------------------------------
    # 2. Greeks tab callback
    # -----------------------------------------------------------------------
    @app.callback(
        [
            Output("fxrisk-vega-heatmap", "figure"),
            Output("fxrisk-delta-bar", "figure"),
            Output("fxrisk-gamma-bar", "figure"),
        ],
        Input("fxrisk-tabs", "value"),
        prevent_initial_call=True,
    )
    def update_greeks_tab(tab):
        if tab != "greeks":
            return no_update, no_update, no_update

        try:
            positions = get_all_positions()
            if not positions:
                empty = go.Figure()
                empty.update_layout(**chart_layout(height=CHART_MD,
                                    annotations=[{
                                        "text": "No positions",
                                        "xref": "paper", "yref": "paper",
                                        "x": 0.5, "y": 0.5, "showarrow": False,
                                        "font": {"color": COLORS["text_muted"], "size": 14},
                                    }]))
                return empty, empty, empty

            spots, rates, vol_surfaces = _load_market_data()

            # --- Vega Heatmap ---
            vega_data = vega_by_bucket(positions, spots, rates, vol_surfaces)
            pairs_sorted = sorted(vega_data.keys())
            buckets = TENOR_BUCKETS

            z_vals = []
            annotations = []
            row_totals = []
            for i, pair in enumerate(pairs_sorted):
                row = []
                row_total = 0.0
                for j, bucket in enumerate(buckets):
                    v = vega_data[pair].get(bucket, 0.0)
                    row.append(v)
                    row_total += v
                    annotations.append(dict(
                        x=j, y=i, text=f"{v:+,.0f}",
                        showarrow=False,
                        font=dict(
                            color=COLORS["text_bright"] if abs(v) > 5000 else COLORS["text_secondary"],
                            size=10,
                            family="'JetBrains Mono', monospace",
                        ),
                    ))
                z_vals.append(row)
                row_totals.append(row_total)

            # Column totals
            col_totals = [sum(z_vals[i][j] for i in range(len(pairs_sorted)))
                          for j in range(len(buckets))]

            # Add row totals column
            extended_buckets = buckets + ["TOTAL"]
            for i in range(len(z_vals)):
                z_vals[i].append(row_totals[i])
                annotations.append(dict(
                    x=len(buckets), y=i, text=f"{row_totals[i]:+,.0f}",
                    showarrow=False,
                    font=dict(color=COLORS["accent_cyan"], size=10,
                              family="'JetBrains Mono', monospace"),
                ))

            # Add column totals row
            grand_total = sum(col_totals)
            col_totals_extended = col_totals + [grand_total]
            z_vals.append(col_totals_extended)
            extended_pairs = pairs_sorted + ["TOTAL"]
            for j in range(len(extended_buckets)):
                annotations.append(dict(
                    x=j, y=len(pairs_sorted), text=f"{col_totals_extended[j]:+,.0f}",
                    showarrow=False,
                    font=dict(color=COLORS["accent_cyan"], size=10,
                              family="'JetBrains Mono', monospace"),
                ))

            # Use absolute values for color scale
            z_abs = [[abs(v) for v in row] for row in z_vals]

            heatmap_fig = go.Figure(data=go.Heatmap(
                z=z_abs,
                x=extended_buckets,
                y=extended_pairs,
                colorscale=[
                    [0.0, COLORS["bg_secondary"]],
                    [0.3, COLORS["accent_blue"]],
                    [0.6, COLORS["accent_purple"]],
                    [1.0, COLORS["accent_red"]],
                ],
                showscale=True,
                colorbar=dict(
                    title=dict(text="|Vega|", font=dict(color=COLORS["text_secondary"], size=10)),
                    tickfont=dict(color=COLORS["text_muted"], size=9),
                    bgcolor="rgba(0,0,0,0)",
                ),
                hovertemplate="<b>%{y}</b> / %{x}<br>Vega: %{text}<extra></extra>",
                text=[[f"{v:+,.0f}" for v in row] for row in z_vals],
            ))
            heatmap_fig.update_layout(
                **chart_layout(
                height=max(CHART_LG, 30 * len(extended_pairs) + 80),
                annotations=annotations,
                xaxis=dict(side="top", tickfont=dict(size=10, color=COLORS["text_secondary"])),
                yaxis=dict(autorange="reversed", tickfont=dict(size=10, color=COLORS["text_secondary"])),
                margin=dict(l=80, r=80, t=50, b=20),
            ))

            # --- Delta Bar ---
            delta_data = delta_by_pair(positions, spots, rates, vol_surfaces)
            dpairs = sorted(delta_data.keys(), key=lambda p: delta_data[p])
            dvals = [delta_data[p] for p in dpairs]
            dcolors = [COLORS["accent_green"] if v >= 0 else COLORS["accent_red"] for v in dvals]

            delta_fig = go.Figure(go.Bar(
                y=dpairs, x=dvals, orientation="h",
                marker_color=dcolors,
                text=[f"{v:+,.0f}" for v in dvals],
                textposition="outside",
                textfont=dict(size=10, color=COLORS["text_secondary"],
                              family="'JetBrains Mono', monospace"),
                hovertemplate="<b>%{y}</b><br>Delta: %{x:+,.0f} USD<extra></extra>",
            ))
            delta_fig.update_layout(
                **chart_layout(
                height=max(CHART_MD, 28 * len(dpairs) + 60),
                xaxis_title="Delta (USD)",
                margin=dict(l=80, r=80, t=30, b=40),
                showlegend=False,
            ))

            # --- Gamma Bar ---
            gamma_data = gamma_by_bucket(positions, spots, rates, vol_surfaces)
            gamma_by_p = {}
            for pair, buckets_dict in gamma_data.items():
                gamma_by_p[pair] = sum(buckets_dict.values())
            gpairs = sorted(gamma_by_p.keys(), key=lambda p: gamma_by_p[p])
            gvals = [gamma_by_p[p] for p in gpairs]
            gcolors = [COLORS["accent_green"] if v >= 0 else COLORS["accent_red"] for v in gvals]

            gamma_fig = go.Figure(go.Bar(
                y=gpairs, x=gvals, orientation="h",
                marker_color=gcolors,
                text=[f"{v:+,.0f}" for v in gvals],
                textposition="outside",
                textfont=dict(size=10, color=COLORS["text_secondary"],
                              family="'JetBrains Mono', monospace"),
                hovertemplate="<b>%{y}</b><br>Gamma: %{x:+,.0f}<extra></extra>",
            ))
            gamma_fig.update_layout(
                **chart_layout(
                height=max(CHART_MD, 28 * len(gpairs) + 60),
                xaxis_title="Gamma (USD/%)",
                margin=dict(l=80, r=80, t=30, b=40),
                showlegend=False,
            ))

            return heatmap_fig, delta_fig, gamma_fig

        except Exception:
            _err = go.Figure()
            _err.update_layout(**chart_layout(height=CHART_MD,
                               annotations=[{
                                   "text": "Greeks tab error",
                                   "xref": "paper", "yref": "paper",
                                   "x": 0.5, "y": 0.5, "showarrow": False,
                                   "font": {"color": COLORS["accent_red"], "size": 14},
                               }]))
            return _err, _err, _err

    # -----------------------------------------------------------------------
    # 3b. Vega heatmap click-through detail
    # -----------------------------------------------------------------------
    # Map tenor bucket labels to standard tenors for vol_percentile lookup
    _BUCKET_TO_TENOR = {
        "0-1M": "1M", "1-3M": "3M", "3-6M": "6M",
        "6-12M": "1Y", "1-2Y": "1Y", "2-5Y": "1Y", "TOTAL": "3M",
    }

    @app.callback(
        [
            Output("fxrisk-click-detail", "children"),
            Output("fxrisk-click-detail", "style"),
            Output("fxrisk-selected-pair", "data"),
        ],
        Input("fxrisk-vega-heatmap", "clickData"),
        prevent_initial_call=True,
    )
    def vega_heatmap_click(click_data):
        if not click_data or "points" not in click_data:
            return no_update, no_update, no_update

        pt = click_data["points"][0]
        pair = pt.get("y", "")
        bucket = pt.get("x", "")
        # The heatmap 'text' stores the signed vega string (e.g. "+1,234")
        vega_text = pt.get("text", "0")

        if not pair or not bucket or pair == "TOTAL":
            return no_update, no_update, no_update

        # Resolve the standard tenor for analytics lookups
        tenor = _BUCKET_TO_TENOR.get(bucket, "3M")

        # --- Fetch analytics for the clicked cell ---
        try:
            pct_info = vol_percentile(pair, tenor, "ATM")
            pct_val = f"{pct_info['percentile']:.0f}th"
            pct_color = (
                COLORS["accent_red"] if pct_info["percentile"] > 80
                else COLORS["accent_green"] if pct_info["percentile"] < 20
                else COLORS["accent_cyan"]
            )
        except Exception:
            pct_val = "N/A"
            pct_color = COLORS["text_muted"]

        try:
            regime_info = vol_regime_detect(pair)
            regime_label = regime_info["regime"]
            regime_color = regime_info["color"]
        except Exception:
            regime_label = "N/A"
            regime_color = COLORS["text_muted"]

        # --- Build the detail section ---
        header = html.Div(
            f"VEGA DETAIL  --  {pair} / {bucket}",
            style={
                "color": COLORS["accent_cyan"],
                "fontSize": "11px",
                "fontWeight": "700",
                "fontFamily": "'JetBrains Mono', monospace",
                "letterSpacing": "1.5px",
                "textTransform": "uppercase",
                "marginBottom": "8px",
            },
        )

        stat_row = html.Div([
            clickable_stat(
                value=vega_text,
                label="Vega (USD)",
                pair=pair,
                metric="vega",
                tenor=tenor,
                color=COLORS["accent_purple"],
            ),
            clickable_stat(
                value=pct_val,
                label="ATM Vol Pctile",
                pair=pair,
                metric="ATM",
                tenor=tenor,
                color=pct_color,
            ),
            clickable_stat(
                value=regime_label,
                label="Vol Regime",
                pair=pair,
                metric="regime",
                tenor=tenor,
                color=regime_color,
            ),
        ], style={
            "display": "flex",
            "gap": GAP,
            "flexWrap": "wrap",
        })

        detail_style = {
            "border": f"1px solid {COLORS['border_subtle']}",
            "borderLeft": f"3px solid {COLORS['accent_cyan']}",
            "backgroundColor": COLORS["bg_card"],
            "padding": GAP,
            "marginTop": GAP,
            "marginBottom": GAP,
            "borderRadius": "6px",
            "display": "block",
        }

        return [header, stat_row], detail_style, pair

    # -----------------------------------------------------------------------
    # 4. VaR tab callback
    # -----------------------------------------------------------------------
    @app.callback(
        [
            Output("fxrisk-var-dist", "figure"),
            Output("fxrisk-component-var", "figure"),
        ],
        Input("fxrisk-tabs", "value"),
        prevent_initial_call=True,
    )
    def update_var_tab(tab):
        if tab != "var":
            return no_update, no_update

        try:
            positions = get_all_positions()
            if not positions:
                return no_data_fig(height=CHART_LG, msg="NO POSITIONS"), no_data_fig(height=CHART_LG, msg="NO POSITIONS")

            spots, rates, vol_surfaces = _load_market_data()

            # Aggregate portfolio notional and vol
            total_notional = sum(p["notional"] for p in positions)
            avg_vol = np.mean([
                _safe_atm_vol(vol_surfaces.get(p["pair"], 0.10))
                for p in positions
            ])

            # Generate normal P&L scenarios
            rng = np.random.RandomState(seed=42)
            daily_vol = avg_vol / np.sqrt(252)
            sim_returns = rng.normal(0, daily_vol, 2_000)
            sim_pnl = sim_returns * total_notional

            # VaR lines
            sorted_pnl = np.sort(sim_pnl)
            var_95_idx = int(0.05 * len(sorted_pnl))
            var_99_idx = int(0.01 * len(sorted_pnl))
            var_95 = sorted_pnl[var_95_idx]
            var_99 = sorted_pnl[var_99_idx]
            cvar_95 = np.mean(sorted_pnl[:var_95_idx]) if var_95_idx > 0 else var_95

            # Distribution figure
            dist_fig = go.Figure()

            # Histogram
            dist_fig.add_trace(go.Histogram(
                x=sim_pnl,
                nbinsx=80,
                marker_color=COLORS["accent_blue"],
                opacity=0.7,
                name="Simulated P&L",
                hovertemplate="P&L: %{x:,.0f}<br>Count: %{y}<extra></extra>",
            ))

            # CVaR shaded region
            cvar_x = sorted_pnl[:var_95_idx]
            if len(cvar_x) > 0:
                dist_fig.add_trace(go.Histogram(
                    x=cvar_x,
                    nbinsx=20,
                    marker_color=COLORS["accent_red"],
                    opacity=0.6,
                    name=f"CVaR Region ({cvar_95:,.0f})",
                ))

            # VaR 95% line
            dist_fig.add_vline(
                x=var_95, line_dash="dash", line_color=COLORS["accent_orange"],
                line_width=2,
                annotation_text=f"VaR 95%: {var_95:,.0f}",
                annotation_position="top left",
                annotation_font=dict(color=COLORS["accent_orange"], size=11,
                                     family="'JetBrains Mono', monospace"),
            )

            # VaR 99% line
            dist_fig.add_vline(
                x=var_99, line_dash="dot", line_color=COLORS["accent_red"],
                line_width=2,
                annotation_text=f"VaR 99%: {var_99:,.0f}",
                annotation_position="top left",
                annotation_font=dict(color=COLORS["accent_red"], size=11,
                                     family="'JetBrains Mono', monospace"),
            )

            dist_fig.update_layout(
                **chart_layout(
                height=CHART_LG,
                xaxis_title="P&L (USD)",
                yaxis_title="Frequency",
                barmode="overlay",
                showlegend=True,
                legend=dict(
                    x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)",
                    font=dict(color=COLORS["text_secondary"], size=10),
                ),
                margin=dict(l=60, r=30, t=40, b=50),
            ))

            # --- Component VaR by pair ---
            pair_var = {}
            for pos in positions:
                pair = pos["pair"]
                vol = _safe_atm_vol(vol_surfaces.get(pair, 0.10))
                d_vol = vol / np.sqrt(252)
                notional = pos["notional"]
                sign = 1.0 if pos.get("direction", "buy") == "buy" else -1.0
                # Parametric VaR per position
                from scipy.stats import norm as scipy_norm
                z95 = scipy_norm.ppf(0.95)
                pos_var = z95 * d_vol * notional * abs(sign)
                pair_var[pair] = pair_var.get(pair, 0.0) + pos_var

            cv_pairs = sorted(pair_var.keys(), key=lambda p: pair_var[p], reverse=True)
            cv_vals = [pair_var[p] for p in cv_pairs]

            comp_fig = go.Figure(go.Bar(
                x=cv_vals, y=cv_pairs, orientation="h",
                marker_color=[COLORS["accent_cyan"] if i < 3 else COLORS["accent_blue"]
                              for i in range(len(cv_pairs))],
                text=[f"${v:,.0f}" for v in cv_vals],
                textposition="outside",
                textfont=dict(size=10, color=COLORS["text_secondary"],
                              family="'JetBrains Mono', monospace"),
                hovertemplate="<b>%{y}</b><br>Component VaR: $%{x:,.0f}<extra></extra>",
            ))
            comp_fig.update_layout(
                **chart_layout(
                height=max(CHART_MD, 28 * len(cv_pairs) + 60),
                xaxis_title="Component VaR 95% (USD)",
                margin=dict(l=80, r=80, t=30, b=40),
                showlegend=False,
            ))

            return dist_fig, comp_fig

        except Exception:
            _err = go.Figure()
            _err.update_layout(**chart_layout(height=CHART_LG,
                               annotations=[{
                                   "text": "VaR tab error",
                                   "xref": "paper", "yref": "paper",
                                   "x": 0.5, "y": 0.5, "showarrow": False,
                                   "font": {"color": COLORS["accent_red"], "size": 14},
                               }]))
            return _err, _err

    # -----------------------------------------------------------------------
    # 5. Stress tab: scenario comparison (runs on tab switch)
    # -----------------------------------------------------------------------
    @app.callback(
        Output("fxrisk-scenario-bars", "figure"),
        Input("fxrisk-tabs", "value"),
        prevent_initial_call=True,
    )
    def update_scenario_comparison(tab):
        if tab != "stress":
            return no_update

        try:
            positions = get_all_positions()
            if not positions:
                return no_data_fig(msg="NO POSITIONS")

            spots, rates, vol_surfaces = _load_market_data()

            # Prepare positions for stress engine format
            stress_positions = _prepare_stress_positions(positions)

            # Compare all scenarios
            comp_df = compare_scenarios(stress_positions, spots, rates, vol_surfaces)

            if comp_df.empty:
                return no_data_fig(msg="NO SCENARIO DATA")

            # Build horizontal bar chart sorted by P&L impact
            scenario_names = comp_df["Scenario"].tolist()
            pnl_values = comp_df["Total PnL"].tolist()
            severities = comp_df["Severity"].tolist()
            bar_colors = [_severity_color(s) for s in severities]

            fig = go.Figure(go.Bar(
                y=scenario_names,
                x=pnl_values,
                orientation="h",
                marker_color=bar_colors,
                text=[f"${v:+,.0f}" for v in pnl_values],
                textposition="outside",
                textfont=dict(size=10, color=COLORS["text_secondary"],
                              family="'JetBrains Mono', monospace"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "P&L Impact: $%{x:+,.0f}<br>"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                **chart_layout(
                height=max(CHART_LG, 30 * len(scenario_names) + 80),
                xaxis_title="Portfolio P&L Impact (USD)",
                margin=dict(l=250, r=100, t=30, b=40),
                showlegend=False,
            ))

            return fig

        except Exception:
            _err = go.Figure()
            _err.update_layout(**chart_layout(height=CHART_LG,
                               annotations=[{
                                   "text": "Stress tab error",
                                   "xref": "paper", "yref": "paper",
                                   "x": 0.5, "y": 0.5, "showarrow": False,
                                   "font": {"color": COLORS["accent_red"], "size": 14},
                               }]))
            return _err

    # -----------------------------------------------------------------------
    # 6. Custom stress test callback
    # -----------------------------------------------------------------------
    @app.callback(
        [
            Output("fxrisk-custom-stress-result", "children"),
            Output("fxrisk-stress-position-table", "children"),
        ],
        Input("fxrisk-run-stress", "n_clicks"),
        [
            State("fxrisk-stress-spot", "value"),
            State("fxrisk-stress-vol", "value"),
            State("fxrisk-stress-rate", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_custom_stress(n_clicks, spot_shock, vol_mult, rate_shock):
        if not n_clicks:
            return no_update, no_update

        try:
            positions = get_all_positions()
            if not positions:
                return html.Div("No positions.", style={"color": COLORS["text_muted"]}), ""

            spots, rates, vol_surfaces = _load_market_data()
            stress_positions = _prepare_stress_positions(positions)

            # Build custom shocks for all pairs in portfolio
            pairs_in_portfolio = list({p["pair"] for p in positions})
            custom_shocks = {
                pair: {
                    "spot": (spot_shock or 0) / 100.0,
                    "vol_mult": vol_mult or 1.0,
                    "rate": rate_shock or 0,
                }
                for pair in pairs_in_portfolio
            }

            result = custom_stress(stress_positions, spots, rates, vol_surfaces, custom_shocks)

            total_pnl = result.get("total_pnl", 0.0)
            n_losers = result.get("n_losers", 0)
            n_winners = result.get("n_winners", 0)

            summary = html.Div([
                html.Div([
                    html.Span("TOTAL P&L: ", style={"color": COLORS["text_secondary"]}),
                    html.Span(f"${total_pnl:+,.0f}", style={
                        "color": _pnl_color(total_pnl),
                        "fontWeight": "800",
                        "fontSize": "16px",
                    }),
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Span(f"Losers: {n_losers}", style={"color": COLORS["accent_red"],
                                                              "marginRight": "20px"}),
                    html.Span(f"Winners: {n_winners}", style={"color": COLORS["accent_green"]}),
                ]),
                html.Div([
                    html.Span(f"Spot Shock: {spot_shock or 0:+.1f}%  |  "
                               f"Vol Mult: {vol_mult or 1.0:.1f}x  |  "
                               f"Rate: {rate_shock or 0:+d}bps",
                               style={"color": COLORS["text_muted"], "fontSize": "11px"}),
                ], style={"marginTop": GAP}),
            ])

            # Per-position impact table
            by_position = result.get("by_position", [])
            if by_position:
                table_data = []
                for p in by_position[:20]:
                    table_data.append({
                        "Pair": p.get("pair", ""),
                        "Type": p.get("option_type", ""),
                        "Strike": f"{p.get('strike', 0):.4f}",
                        "Notional": f"{p.get('notional', 0):,.0f}",
                        "Base Value": f"${p.get('base_value', 0):,.0f}",
                        "Stressed Value": f"${p.get('stressed_value', 0):,.0f}",
                        "P&L Impact": f"${p.get('pnl_impact', 0):+,.0f}",
                    })

                pos_table = dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in table_data[0].keys()],
                    data=table_data,
                    style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE,
                    style_data_conditional=[
                        {
                            "if": {"filter_query": '{P&L Impact} contains "-"'},
                            "color": COLORS["accent_red"],
                        },
                    ],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                )
            else:
                pos_table = html.Div("No position data.", style={"color": COLORS["text_muted"]})

            return summary, pos_table

        except Exception as exc:
            err_msg = html.Div(
                f"Custom stress error: {str(exc)[:80]}",
                style={"color": COLORS["accent_red"], "fontFamily": "monospace",
                       "fontSize": "12px"},
            )
            return err_msg, ""

    # -----------------------------------------------------------------------
    # 7. Attribution tab callback
    # -----------------------------------------------------------------------
    @app.callback(
        [
            Output("fxrisk-waterfall", "figure"),
            Output("fxrisk-pnl-pair", "figure"),
        ],
        Input("fxrisk-tabs", "value"),
        prevent_initial_call=True,
    )
    def update_attribution_tab(tab):
        if tab != "attribution":
            return no_update, no_update

        try:
            positions = get_all_positions()
            if not positions:
                return no_data_fig(msg="NO POSITIONS"), no_data_fig(msg="NO POSITIONS")

            spots, rates, vol_surfaces = _load_market_data()

            # Simulate a 1-day move: shift spots slightly for attribution
            rng = np.random.RandomState(seed=99)
            spots_new = {}
            surfaces_new = {}
            for pair, spot_val in spots.items():
                vol = _safe_atm_vol(vol_surfaces.get(pair, 0.10))
                daily_ret = rng.normal(0, vol / np.sqrt(252))
                spots_new[pair] = spot_val * (1 + daily_ret)
                # Slightly change vol (mean-reverting bump)
                surfaces_new[pair] = vol * (1 + rng.normal(0, 0.02))

            attr = pnl_attribution(positions, spots, spots_new, vol_surfaces,
                                    surfaces_new, rates, dt=1 / 252)

            # Waterfall chart
            components = [
                ("Delta P&L", attr.get("delta_pnl", 0)),
                ("Gamma P&L", attr.get("gamma_pnl", 0)),
                ("Vega P&L", attr.get("vega_pnl", 0)),
                ("Theta", attr.get("theta_pnl", 0)),
                ("Rho", attr.get("rho_pnl", 0)),
                ("Unexplained", attr.get("unexplained", 0)),
            ]
            total_pnl = attr.get("total_pnl", 0)

            wf_labels = [c[0] for c in components] + ["Total"]
            wf_values = [c[1] for c in components] + [total_pnl]
            wf_measures = ["relative"] * len(components) + ["total"]

            wf_colors = []
            for v in wf_values[:-1]:
                wf_colors.append(COLORS["accent_green"] if v >= 0 else COLORS["accent_red"])
            wf_colors.append(COLORS["accent_cyan"])

            waterfall_fig = go.Figure(go.Waterfall(
                name="P&L Attribution",
                orientation="v",
                measure=wf_measures,
                x=wf_labels,
                y=wf_values,
                textposition="outside",
                text=[f"${v:+,.0f}" for v in wf_values],
                textfont=dict(size=10, color=COLORS["text_secondary"],
                              family="'JetBrains Mono', monospace"),
                connector=dict(line=dict(color=COLORS["border"], width=1)),
                increasing=dict(marker=dict(color=COLORS["accent_green"])),
                decreasing=dict(marker=dict(color=COLORS["accent_red"])),
                totals=dict(marker=dict(color=COLORS["accent_cyan"])),
                hovertemplate="<b>%{x}</b><br>$%{y:+,.0f}<extra></extra>",
            ))
            waterfall_fig.update_layout(
                **chart_layout(
                height=CHART_LG,
                yaxis_title="P&L (USD)",
                margin=dict(l=60, r=30, t=40, b=50),
                showlegend=False,
            ))

            # --- By-pair P&L stacked bar ---
            by_position = attr.get("by_position", [])
            pair_pnl = {}
            pair_components = {}
            for entry in by_position:
                pair = entry.get("pair", "UNKNOWN")
                if pair not in pair_pnl:
                    pair_pnl[pair] = 0.0
                    pair_components[pair] = {
                        "Delta": 0.0, "Gamma": 0.0, "Vega": 0.0,
                        "Theta": 0.0, "Other": 0.0,
                    }
                pair_pnl[pair] += entry.get("total_pnl", 0)
                pair_components[pair]["Delta"] += entry.get("delta_pnl", 0)
                pair_components[pair]["Gamma"] += entry.get("gamma_pnl", 0)
                pair_components[pair]["Vega"] += entry.get("vega_pnl", 0)
                pair_components[pair]["Theta"] += entry.get("theta_pnl", 0)
                pair_components[pair]["Other"] += (
                    entry.get("rho_pnl", 0) + entry.get("unexplained", 0)
                )

            sorted_pairs = sorted(pair_pnl.keys(), key=lambda p: pair_pnl[p])
            component_names = ["Delta", "Gamma", "Vega", "Theta", "Other"]
            component_colors = [
                COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_purple"],
                COLORS["accent_orange"], COLORS["accent_teal"],
            ]

            pnl_pair_fig = go.Figure()
            for comp_name, comp_color in zip(component_names, component_colors):
                vals = [pair_components.get(p, {}).get(comp_name, 0) for p in sorted_pairs]
                pnl_pair_fig.add_trace(go.Bar(
                    name=comp_name,
                    y=sorted_pairs,
                    x=vals,
                    orientation="h",
                    marker_color=comp_color,
                    hovertemplate=(f"<b>%{{y}}</b><br>{comp_name}: $%{{x:+,.0f}}<extra></extra>"),
                ))

            pnl_pair_fig.update_layout(
                **chart_layout(
                barmode="stack",
                height=max(CHART_MD, 28 * len(sorted_pairs) + 80),
                xaxis_title="P&L (USD)",
                margin=dict(l=80, r=40, t=30, b=40),
                legend=dict(
                    orientation="h", y=1.12, x=0.5, xanchor="center",
                    font=dict(color=COLORS["text_secondary"], size=10),
                    bgcolor="rgba(0,0,0,0)",
                ),
            ))

            return waterfall_fig, pnl_pair_fig

        except Exception:
            _err = go.Figure()
            _err.update_layout(**chart_layout(height=CHART_LG,
                               annotations=[{
                                   "text": "Attribution tab error",
                                   "xref": "paper", "yref": "paper",
                                   "x": 0.5, "y": 0.5, "showarrow": False,
                                   "font": {"color": COLORS["accent_red"], "size": 14},
                               }]))
            return _err, _err

    # -----------------------------------------------------------------------
    # 8. What-If: preview trade impact
    # -----------------------------------------------------------------------
    @app.callback(
        Output("fxrisk-wi-impact-table", "children"),
        Input("fxrisk-wi-preview", "n_clicks"),
        [
            State("fxrisk-wi-pair", "value"),
            State("fxrisk-wi-type", "value"),
            State("fxrisk-wi-dir", "value"),
            State("fxrisk-wi-delta", "value"),
            State("fxrisk-wi-tenor", "value"),
            State("fxrisk-wi-notional", "value"),
        ],
        prevent_initial_call=True,
    )
    def preview_what_if(n_clicks, pair, opt_type, direction, delta_val,
                        tenor, notional_m):
        if not n_clicks:
            return no_update

        try:
            positions = get_all_positions()
            spots, rates, vol_surfaces = _load_market_data()

            # Convert tenor to expiry
            from core.fx_conventions import tenor_to_days
            from datetime import date, timedelta
            today = date.today()
            days = tenor_to_days(tenor)
            expiry_date = today + timedelta(days=days)

            # Estimate strike from delta (approximate: use ATM spot +/- adjustment)
            spot = spots.get(pair, 1.0)
            vol = _safe_atm_vol(vol_surfaces.get(pair, 0.10))
            # Simple approximation: strike = spot * exp(-/+ delta_adjustment)
            from scipy.stats import norm as scipy_norm
            T = days / 365.0
            cp_sign = 1 if opt_type == "call" else -1
            # Invert delta to strike via normal approximation
            d_val = delta_val if opt_type == "call" else delta_val
            z = scipy_norm.ppf(d_val)
            K = spot * np.exp(-cp_sign * z * vol * np.sqrt(T) + 0.5 * vol ** 2 * T)

            new_trade = {
                "pair": pair,
                "option_type": opt_type,
                "direction": direction,
                "strike": round(float(K), 5),
                "expiry": expiry_date.isoformat(),
                "notional": (notional_m or 10) * 1_000_000,
                "book": "G10_PROP",
                "status": "open",
            }

            result = what_if_add(positions, new_trade, spots, rates, vol_surfaces)

            before = result.get("before", {})
            after = result.get("after", {})
            change = result.get("change", {})

            greeks = ["delta", "gamma", "vega", "theta", "price"]
            greek_labels = ["Delta (USD)", "Gamma", "Vega", "Theta", "MtM Value"]

            table_data = []
            for gk, label in zip(greeks, greek_labels):
                b_val = before.get(gk, 0)
                a_val = after.get(gk, 0)
                c_val = change.get(gk, 0)
                table_data.append({
                    "Greek": label,
                    "Before": _fmt_usd(b_val),
                    "After": _fmt_usd(a_val),
                    "Change": _fmt_usd(c_val),
                })

            impact_table = dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in ["Greek", "Before", "After", "Change"]],
                data=table_data,
                style_header=TABLE_HEADER_STYLE,
                style_cell=TABLE_CELL_STYLE,
                style_data_conditional=[
                    {
                        "if": {"column_id": "Change", "filter_query": '{Change} contains "-"'},
                        "color": COLORS["accent_red"],
                    },
                    {
                        "if": {"column_id": "Change", "filter_query": '{Change} contains "+"'},
                        "color": COLORS["accent_green"],
                    },
                ],
                style_table={"overflowX": "auto"},
            )

            new_greeks = result.get("new_trade_greeks", {})
            trade_summary = html.Div([
                html.Div(f"NEW TRADE: {direction.upper()} {pair} {opt_type.upper()} "
                          f"K={K:.5f} {tenor} {(notional_m or 10):.0f}M", style={
                    "color": COLORS["accent_cyan"],
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginBottom": GAP,
                    "letterSpacing": "1px",
                }),
                impact_table,
            ])

            return trade_summary

        except Exception as exc:
            return html.Div(
                f"What-if error: {str(exc)[:80]}",
                style={"color": COLORS["accent_red"], "fontFamily": "monospace",
                       "fontSize": "12px", "padding": "20px"},
            )

    # -----------------------------------------------------------------------
    # 9. Hedge suggestion callbacks
    # -----------------------------------------------------------------------
    @app.callback(
        Output("fxrisk-hedge-result", "children"),
        [
            Input("fxrisk-hedge-delta", "n_clicks"),
            Input("fxrisk-hedge-vega", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def show_hedge_suggestions(delta_clicks, vega_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        try:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            positions = get_all_positions()
            if not positions:
                return html.Div("No positions.", style={"color": COLORS["text_muted"]})

            spots, rates, vol_surfaces = _load_market_data()
            risk = compute_portfolio_risk(spots, rates, vol_surfaces)

            if trigger_id == "fxrisk-hedge-delta":
                target = "delta_neutral"
            else:
                target = "vega_neutral_3M"

            suggestions = hedge_suggestion(risk, target=target)

            if not suggestions:
                return html.Div("Portfolio already within tolerance. No hedges needed.",
                                style={"color": COLORS["accent_green"],
                                       "fontFamily": "'JetBrains Mono', monospace",
                                       "fontSize": "12px"})

            # Build enhanced table data with actionable descriptions and cost
            table_data = []
            action_lines = []
            total_cost = 0.0
            for s in suggestions:
                pair = s.get("pair", "")
                instrument = s.get("instrument", "")
                direction = s.get("direction", "").upper()
                notional = s.get("notional", 0)
                spot = spots.get(pair, 1.0)
                vol = _safe_atm_vol(vol_surfaces.get(pair, 0.10))

                # Estimate approximate cost
                if instrument == "SPOT":
                    # Spot hedge: cost is spread (~2-5 pips)
                    spread_cost = notional * 0.0003  # ~3 pips spread cost
                    action_desc = f"{direction} {notional:,.0f} units of {pair} spot"
                    cost_str = f"~${spread_cost:,.0f} spread"
                    total_cost += spread_cost
                elif "STRADDLE" in instrument:
                    # Straddle cost: approximate as 2 * BS premium for ATM
                    tenor_str = instrument.split()[0] if instrument else "3M"
                    tenor_map = {"1W": 7/365, "2W": 14/365, "1M": 30/365,
                                 "2M": 60/365, "3M": 90/365, "6M": 180/365,
                                 "9M": 270/365, "1Y": 1.0, "2Y": 2.0}
                    T = tenor_map.get(tenor_str, 0.25)
                    # ATM straddle premium ~ 2 * S * vol * sqrt(T) * 0.4 (approx)
                    straddle_prem_pct = 2 * vol * np.sqrt(T) * 0.4
                    straddle_cost = notional * straddle_prem_pct
                    action_desc = (f"{direction} {notional:,.0f} notional "
                                   f"{pair} {tenor_str} ATM straddle")
                    cost_str = f"~${straddle_cost:,.0f} premium"
                    total_cost += straddle_cost
                else:
                    action_desc = f"{direction} {notional:,.0f} {pair} {instrument}"
                    cost_str = "N/A"

                table_data.append({
                    "Pair": pair,
                    "Action": action_desc,
                    "Direction": direction,
                    "Notional": f"{notional:,.0f}",
                    "Est. Cost": cost_str,
                    "Rationale": s.get("rationale", ""),
                })
                action_lines.append(action_desc)

            hedge_table = dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in table_data[0].keys()],
                data=table_data,
                style_header=TABLE_HEADER_STYLE,
                style_cell={**TABLE_CELL_STYLE, "whiteSpace": "normal", "maxWidth": "300px"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{Direction} = "BUY"'},
                        "color": COLORS["accent_green"],
                    },
                    {
                        "if": {"filter_query": '{Direction} = "SELL"'},
                        "color": COLORS["accent_red"],
                    },
                ],
                style_table={"overflowX": "auto"},
            )

            return html.Div([
                html.Div(f"SUGGESTED HEDGES ({target.upper().replace('_', ' ')})", style={
                    "color": COLORS["accent_cyan"],
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginBottom": GAP,
                    "letterSpacing": "1px",
                }),
                html.Div(f"Total estimated hedge cost: ${total_cost:,.0f}", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "12px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginBottom": GAP,
                }),
                hedge_table,
            ])

        except Exception as exc:
            return html.Div(
                f"Hedge suggestion error: {str(exc)[:80]}",
                style={"color": COLORS["accent_red"], "fontFamily": "monospace",
                       "fontSize": "12px", "padding": "20px"},
            )

    # -----------------------------------------------------------------------
    # 10. Tab content visibility (show/hide the correct container)
    # -----------------------------------------------------------------------
    @app.callback(
        [
            Output("fxrisk-greeks-container", "style"),
            Output("fxrisk-var-container", "style"),
            Output("fxrisk-stress-container", "style"),
            Output("fxrisk-attr-container", "style"),
            Output("fxrisk-whatif-container", "style"),
            Output("fxrisk-hedge-container", "style"),
        ],
        Input("fxrisk-tabs", "value"),
        prevent_initial_call=False,
    )
    def toggle_tab_containers(tab):
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if tab == "greeks" else hide,
            show if tab == "var" else hide,
            show if tab == "stress" else hide,
            show if tab == "attribution" else hide,
            show if tab == "whatif" else hide,
            show if tab == "hedge" else hide,
        )

    # -----------------------------------------------------------------------
    # 11. HEDGE Tab: Delta Hedge Simulation
    # -----------------------------------------------------------------------
    @app.callback(
        [
            Output("fxrisk-hsim-stats", "children"),
            Output("fxrisk-hsim-pnl", "figure"),
            Output("fxrisk-hsim-gamma", "figure"),
        ],
        Input("fxrisk-hsim-run", "n_clicks"),
        [
            State("fxrisk-hsim-pair", "value"),
            State("fxrisk-hsim-cp", "value"),
            State("fxrisk-hsim-delta", "value"),
            State("fxrisk-hsim-notional", "value"),
            State("fxrisk-hsim-freq", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_hedge_sim(n_clicks, pair, cp, delta, notional, freq):
        if not n_clicks:
            raise PreventUpdate
        import numpy as np
        rng = np.random.RandomState(42)
        notional = notional or 10_000_000
        n_days = 66  # ~3M

        # Simulate spot path
        try:
            from core.bloomberg_fx import get_fx_spots
            spots = get_fx_spots() or {}
            s0 = float(spots.get(pair, {}).get("mid", 1.0))
        except Exception:
            s0 = 1.08 if pair == "EURUSD" else 150 if pair == "USDJPY" else 1.0

        vol = 0.08
        dt = 1 / 252
        spots_path = [s0]
        for _ in range(n_days):
            ds = spots_path[-1] * vol * np.sqrt(dt) * rng.normal()
            spots_path.append(spots_path[-1] + ds)

        # Hedge PnL simulation
        hedge_freq = 1 if freq == "Daily" else 5 if freq == "Weekly" else n_days + 1
        cum_pnl = [0.0]
        gamma_pnl = [0.0]
        for i in range(1, n_days + 1):
            ds = spots_path[i] - spots_path[i - 1]
            gamma_contrib = 0.5 * delta * notional * (ds / spots_path[i - 1]) ** 2
            if i % hedge_freq == 0:
                hedge_cost = abs(ds) * notional * 0.00005
            else:
                hedge_cost = 0
            pnl = gamma_contrib - hedge_cost
            cum_pnl.append(cum_pnl[-1] + pnl)
            gamma_pnl.append(gamma_pnl[-1] + gamma_contrib)

        total_pnl = cum_pnl[-1]
        n_hedges = n_days // max(hedge_freq, 1)

        # Stats
        stats = [
            _make_stat_box("TOTAL P&L", f"${total_pnl:,.0f}",
                           COLORS["accent_green"] if total_pnl > 0 else COLORS["accent_red"]),
            _make_stat_box("HEDGES", str(n_hedges), COLORS["accent_orange"]),
            _make_stat_box("FREQUENCY", freq, "#d4d4d4"),
            _make_stat_box("PAIR", pair, "#ff8800"),
        ]

        # PnL chart
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(y=cum_pnl, mode="lines",
                                      line=dict(color="#ff8800", width=1.5), name="Cumulative P&L"))
        fig_pnl.add_hline(y=0, line=dict(color="#808080", width=0.5, dash="dash"))
        fig_pnl.update_layout(**chart_layout(height=CHART_SM,
                               margin=dict(l=60, r=20, t=30, b=20),
                               title=dict(text="CUMULATIVE HEDGE P&L", font=dict(size=10, color="#808080"))))

        # Gamma PnL chart
        fig_gamma = go.Figure()
        fig_gamma.add_trace(go.Scatter(y=gamma_pnl, mode="lines",
                                        line=dict(color="#00cc66", width=1.5), name="Gamma P&L"))
        fig_gamma.add_hline(y=0, line=dict(color="#808080", width=0.5, dash="dash"))
        fig_gamma.update_layout(**chart_layout(height=CHART_SM,
                                 margin=dict(l=60, r=20, t=30, b=20),
                                 title=dict(text="GAMMA P&L", font=dict(size=10, color="#808080"))))

        return stats, fig_pnl, fig_gamma

    # -----------------------------------------------------------------------
    # 12. HEDGE Tab: Cross-Hedge
    # -----------------------------------------------------------------------
    @app.callback(
        [
            Output("fxrisk-xh-stats", "children"),
            Output("fxrisk-xh-chart", "figure"),
        ],
        Input("fxrisk-xh-run", "n_clicks"),
        [
            State("fxrisk-xh-target", "value"),
            State("fxrisk-xh-h1", "value"),
            State("fxrisk-xh-h2", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_cross_hedge(n_clicks, target, h1, h2):
        if not n_clicks:
            raise PreventUpdate
        import numpy as np
        from numpy.linalg import lstsq
        from core.bloomberg_fx import get_fx_historical_spot

        # Fetch real historical spot data
        hist_t = get_fx_historical_spot(target, days=252)
        hist_1 = get_fx_historical_spot(h1, days=252)
        hist_2 = get_fx_historical_spot(h2, days=252)

        def _to_closes(df):
            if df is None:
                return None
            if hasattr(df, 'columns') and 'close' in df.columns:
                return df['close'].values.astype(float)
            if hasattr(df, 'values'):
                return df.values.flatten().astype(float)
            return None

        closes_t = _to_closes(hist_t)
        closes_1 = _to_closes(hist_1)
        closes_2 = _to_closes(hist_2)

        if closes_t is None or closes_1 is None or closes_2 is None or \
           len(closes_t) < 30 or len(closes_1) < 30 or len(closes_2) < 30:
            stats = [
                _make_stat_box("TARGET", target, "#ff8800"),
                _make_stat_box("STATUS", "NO DATA", "#808080"),
            ]
            return stats, no_data_fig(height=CHART_SM, msg="NO HISTORICAL DATA FOR CROSS-HEDGE")

        # Compute log returns from real data
        ret_t = np.diff(np.log(np.maximum(closes_t[-120:], 1e-10)))
        ret_1 = np.diff(np.log(np.maximum(closes_1[-120:], 1e-10)))
        ret_2 = np.diff(np.log(np.maximum(closes_2[-120:], 1e-10)))
        n = min(len(ret_t), len(ret_1), len(ret_2))
        ret_t, ret_1, ret_2 = ret_t[-n:], ret_1[-n:], ret_2[-n:]

        corr_1 = round(float(np.corrcoef(ret_t, ret_1)[0, 1]), 2)
        corr_2 = round(float(np.corrcoef(ret_t, ret_2)[0, 1]), 2)

        # OLS for betas
        X = np.column_stack([ret_1, ret_2])
        betas, _, _, _ = lstsq(X, ret_t, rcond=None)
        beta_1, beta_2 = round(float(betas[0]), 2), round(float(betas[1]), 2)
        residual = ret_t - X @ betas
        r_sq = round(max(0, 1 - np.var(residual) / max(np.var(ret_t), 1e-20)), 2)

        stats = [
            _make_stat_box("TARGET", target, "#ff8800"),
            _make_stat_box(f"\u03b2 ({h1})", f"{beta_1:.2f}", "#d4d4d4"),
            _make_stat_box(f"\u03b2 ({h2})", f"{beta_2:.2f}", "#d4d4d4"),
            _make_stat_box("R\u00b2", f"{r_sq:.2f}", "#00cc66" if r_sq > 0.7 else "#ff3333"),
        ]

        # Real PnL comparison
        target_pnl = np.cumsum(ret_t)
        hedged_pnl = np.cumsum(residual)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=target_pnl, mode="lines",
                                  line=dict(color="#ff3333", width=1.5), name=f"{target} unhedged"))
        fig.add_trace(go.Scatter(y=hedged_pnl, mode="lines",
                                  line=dict(color="#00cc66", width=1.5), name="Cross-hedged"))
        fig.add_hline(y=0, line=dict(color="#808080", width=0.5, dash="dash"))
        fig.update_layout(**chart_layout(height=CHART_SM,
                          margin=dict(l=60, r=20, t=30, b=20),
                          title=dict(text=f"CROSS-HEDGE: {target} via {h1}+{h2}",
                                     font=dict(size=10, color="#808080")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return stats, fig

    # -----------------------------------------------------------------------
    # 13. HEDGE Tab: Effectiveness
    # -----------------------------------------------------------------------
    @app.callback(
        Output("fxrisk-heff-table", "children"),
        Input("fxrisk-tabs", "value"),
    )
    def update_hedge_effectiveness(tab):
        if tab != "hedge":
            raise PreventUpdate
        import numpy as np
        from core.bloomberg_fx import get_fx_historical_spot

        pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD"]
        header = html.Tr([html.Th(h, style=TABLE_HEADER_STYLE)
                          for h in ["PAIR", "HEDGE RATIO", "EFFECTIVENESS", "VAR REDUCTION", "STATUS"]])
        body = []
        for pair in pairs:
            hist = get_fx_historical_spot(pair, days=252)
            closes = None
            if hist is not None:
                if hasattr(hist, 'columns') and 'close' in hist.columns:
                    closes = hist['close'].values.astype(float)
                elif hasattr(hist, 'values'):
                    closes = hist.values.flatten().astype(float)

            if closes is None or len(closes) < 60:
                # No data available for this pair
                body.append(html.Tr([
                    html.Td(pair, style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
                    html.Td("N/A", style={**TABLE_CELL_STYLE, "textAlign": "right", "color": "#808080"}),
                    html.Td("N/A", style={**TABLE_CELL_STYLE, "textAlign": "right", "color": "#808080"}),
                    html.Td("N/A", style={**TABLE_CELL_STYLE, "textAlign": "right", "color": "#808080"}),
                    html.Td("NO DATA", style={**TABLE_CELL_STYLE, "color": "#808080", "fontWeight": "600"}),
                ]))
                continue

            # Compute hedge metrics from real data
            returns = np.diff(np.log(np.maximum(closes, 1e-10)))
            # Use 60-day rolling window for hedge ratio (variance ratio)
            window = min(60, len(returns))
            recent = returns[-window:]
            # Hedge ratio = 1.0 for perfect delta hedge; compute min-variance ratio
            var_full = np.var(recent)
            if var_full < 1e-20:
                ratio, eff, var_red = 1.0, 0.0, 0.0
            else:
                # Simple hedge ratio from autocov structure
                ratio = 1.0  # delta-one hedge
                hedged = recent - ratio * np.mean(recent)
                var_hedged = np.var(hedged)
                eff = round(max(0, 1 - var_hedged / var_full), 2)
                var_red = round(eff * 100, 0)
                ratio = round(ratio, 2)

            status = "EFFECTIVE" if eff > 0.8 else "REVIEW"
            s_color = COLORS["accent_green"] if eff > 0.8 else COLORS["accent_red"]
            body.append(html.Tr([
                html.Td(pair, style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
                html.Td(f"{ratio:.2f}", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                html.Td(f"{eff:.0%}", style={**TABLE_CELL_STYLE, "textAlign": "right",
                         "color": COLORS["accent_green"] if eff > 0.8 else COLORS["accent_red"]}),
                html.Td(f"{var_red:.0f}%", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                html.Td(status, style={**TABLE_CELL_STYLE, "color": s_color, "fontWeight": "600"}),
            ]))
        return html.Table([html.Thead(header), html.Tbody(body)],
                          style={"width": "100%", "borderCollapse": "collapse",
                                 "fontFamily": "'JetBrains Mono', monospace", "fontSize": "11px"})

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output("fxrisk-csv-download", "data"),
        [Input("fxrisk-csv-vega", "n_clicks"),
         Input("fxrisk-csv-delta", "n_clicks"),
         Input("fxrisk-csv-gamma", "n_clicks"),
         Input("fxrisk-csv-var", "n_clicks"),
         Input("fxrisk-csv-cvar", "n_clicks"),
         Input("fxrisk-csv-scenario", "n_clicks"),
         Input("fxrisk-csv-waterfall", "n_clicks"),
         Input("fxrisk-csv-pnl", "n_clicks"),
         Input("fxrisk-csv-hsim-pnl", "n_clicks"),
         Input("fxrisk-csv-hsim-gamma", "n_clicks"),
         Input("fxrisk-csv-xh", "n_clicks")],
        [State("fxrisk-vega-heatmap", "figure"),
         State("fxrisk-delta-bar", "figure"),
         State("fxrisk-gamma-bar", "figure"),
         State("fxrisk-var-dist", "figure"),
         State("fxrisk-component-var", "figure"),
         State("fxrisk-scenario-bars", "figure"),
         State("fxrisk-waterfall", "figure"),
         State("fxrisk-pnl-pair", "figure"),
         State("fxrisk-hsim-pnl", "figure"),
         State("fxrisk-hsim-gamma", "figure"),
         State("fxrisk-xh-chart", "figure")],
        prevent_initial_call=True,
    )
    def fxrisk_csv_export(*args):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        btn_names = ["fxrisk-csv-vega", "fxrisk-csv-delta", "fxrisk-csv-gamma",
                     "fxrisk-csv-var", "fxrisk-csv-cvar", "fxrisk-csv-scenario",
                     "fxrisk-csv-waterfall", "fxrisk-csv-pnl",
                     "fxrisk-csv-hsim-pnl", "fxrisk-csv-hsim-gamma", "fxrisk-csv-xh"]
        labels = ["VegaHeatmap", "DeltaBar", "GammaBar", "VaRDist",
                  "ComponentVaR", "ScenarioBars", "Waterfall", "PnLPair",
                  "HSimPnL", "HSimGamma", "CrossHedge"]
        n = len(btn_names)
        for i, name in enumerate(btn_names):
            if btn == name:
                fig = args[n + i]
                if fig:
                    return export_csv(fig, "Risk", labels[i])
                return no_update
        return no_update


def _make_stat_box(label, value, color):
    return html.Div([
        html.Div(str(value), style={"fontSize": "16px", "fontWeight": "700",
                                     "color": color, "fontFamily": "'JetBrains Mono', monospace"}),
        html.Div(label, style={"fontSize": "10px", "color": "#808080",
                                "letterSpacing": "1px", "fontFamily": "'JetBrains Mono', monospace",
                                "marginTop": "4px"}),
    ], style={**STAT_BOX_STYLE, "borderLeft": f"3px solid {color}"})


# ---------------------------------------------------------------------------
# Helper: build position table
# ---------------------------------------------------------------------------

def _build_position_table(positions, spots, rates, vol_surfaces):
    """Build a DataTable of current positions with live Greeks."""
    from core.fx_portfolio import compute_position_greeks, _get_rate

    table_rows = []
    for pos in positions:
        if pos.get("status") != "open":
            continue
        pair = pos["pair"]
        S = spots.get(pair, 1.0)
        r_d = _get_rate(pair, rates, "domestic")
        r_f = _get_rate(pair, rates, "foreign")
        pg = compute_position_greeks(pos, S, r_d, r_f, vol_surfaces)

        direction = pos.get("direction", "buy")
        opt_type = pos.get("option_type", "call")
        display_type = f"{direction.upper()} {opt_type.upper()}"

        entry_prem = pos.get("entry_premium", 0)
        mtm = pg.get("price", 0)
        pnl = mtm - entry_prem if direction == "buy" else entry_prem - mtm

        table_rows.append({
            "Pair": pair,
            "Type": display_type,
            "Strike": f"{pos.get('strike', 0):.4f}",
            "Delta": f"{pg.get('delta', 0):+,.0f}",
            "Expiry": str(pos.get("expiry", ""))[:10],
            "Notional": f"{pos['notional']:,.0f}",
            "Book": pos.get("book", ""),
            "Strategy": pos.get("strategy", ""),
            "P&L": f"${pnl:+,.0f}",
            "Vega": f"{pg.get('vega', 0):+,.0f}",
            "Gamma": f"{pg.get('gamma', 0):+,.0f}",
            "Theta": f"{pg.get('theta', 0):+,.0f}",
        })

    if not table_rows:
        return html.Div("No open positions.", style={
            "color": COLORS["text_muted"], "padding": "20px",
            "fontFamily": "'JetBrains Mono', monospace",
        })

    columns = [
        {"name": "Pair", "id": "Pair"},
        {"name": "Type", "id": "Type"},
        {"name": "Strike", "id": "Strike"},
        {"name": "Delta", "id": "Delta"},
        {"name": "Expiry", "id": "Expiry"},
        {"name": "Notional", "id": "Notional"},
        {"name": "Book", "id": "Book"},
        {"name": "Strategy", "id": "Strategy"},
        {"name": "P&L", "id": "P&L"},
        {"name": "Vega", "id": "Vega"},
        {"name": "Gamma", "id": "Gamma"},
        {"name": "Theta", "id": "Theta"},
    ]

    return dash_table.DataTable(
        columns=columns,
        data=table_rows,
        style_header=TABLE_HEADER_STYLE,
        style_cell=TABLE_CELL_STYLE,
        style_data_conditional=[
            {
                "if": {"column_id": "P&L", "filter_query": '{P&L} contains "-"'},
                "color": COLORS["accent_red"],
            },
            {
                "if": {"column_id": "P&L", "filter_query": '{P&L} contains "+"'},
                "color": COLORS["accent_green"],
            },
            {
                "if": {"column_id": "Delta", "filter_query": '{Delta} contains "-"'},
                "color": COLORS["accent_red"],
            },
            {
                "if": {"column_id": "Delta", "filter_query": '{Delta} contains "+"'},
                "color": COLORS["accent_green"],
            },
            {
                "if": {"column_id": "Type", "filter_query": '{Type} contains "BUY"'},
                "color": COLORS["accent_green"],
            },
            {
                "if": {"column_id": "Type", "filter_query": '{Type} contains "SELL"'},
                "color": COLORS["accent_red"],
            },
        ],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_filter={
            "backgroundColor": COLORS["bg_input"],
            "color": COLORS["text_primary"],
            "fontFamily": "'JetBrains Mono', monospace",
            "fontSize": "11px",
        },
    )


# ---------------------------------------------------------------------------
# Helper: convert portfolio positions to stress-engine format
# ---------------------------------------------------------------------------

def _prepare_stress_positions(positions):
    """
    Convert the portfolio manager position format to the stress engine format.
    The stress engine expects: pair, strike, expiry (years), option_type, notional.
    """
    from core.fx_portfolio import _years_to_expiry
    stress_positions = []
    for pos in positions:
        if pos.get("status") != "open":
            continue
        T = _years_to_expiry(pos.get("expiry", 0.25))
        stress_positions.append({
            "pair": pos["pair"],
            "strike": pos["strike"],
            "expiry": T,
            "option_type": pos.get("option_type", "call"),
            "notional": pos["notional"],
            "direction": pos.get("direction", "buy"),
            "book": pos.get("book", ""),
            "strategy": pos.get("strategy", ""),
        })
    return stress_positions
