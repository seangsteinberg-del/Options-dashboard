"""
FX Trade Blotter Panel
======================
Full FX options trade entry, execution log, flow analytics, and day summary.

Provides:
  - Delta-based or strike-based trade entry with inline GK pricing
  - Real-time premium and delta auto-computation
  - Execution log with 20+ pre-populated sample trades
  - Flow analytics: notional by pair, cumulative premium, activity timeline
  - Day summary statistics: trades, notional, net delta/vega, premium
"""

import dash
from dash import html, dcc, Input, Output, State, no_update, dash_table, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE, LABEL_STYLE,
    DROPDOWN_STYLE, INPUT_STYLE, BUTTON_STYLE, BUTTON_SUCCESS_STYLE,
    make_stat_style, CSV_BTN_STYLE,
)
from core.csv_export import export_csv
from core.bloomberg_fx import get_fx_vol_surface, get_fx_spots, get_fx_rates, get_all_pairs
from core.fx_portfolio import add_position, get_all_positions, BOOKS
from core.fx_conventions import (
    FX_PAIR_REGISTRY, tenor_to_years, tenor_to_days, CUT_TIMES,
    delta_to_strike, atm_dns_strike,
)


# ============================================================================
# Constants
# ============================================================================

TENORS = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]
FX_PAIRS = sorted(FX_PAIR_REGISTRY.keys())

STRATEGIES = ["RR", "Straddle", "Strangle", "Call Spread", "Put Spread",
              "Butterfly", "Condor", "Seagull", "Collar", "Hedge", "Prop",
              "Client", "Custom"]
CUT_OPTIONS = list(CUT_TIMES.keys())

BOOK_COLORS = {
    "G10_FLOW": COLORS["accent_cyan"],
    "G10_PROP": COLORS["accent_blue"],
    "EM_FLOW": COLORS["accent_orange"],
    "EM_PROP": COLORS["accent_purple"],
    "HEDGE": COLORS["accent_green"],
    "CLIENT_FACILITATION": COLORS["accent_pink"],
}

PAIR_COLORS = [
    COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_purple"],
    COLORS["accent_green"], COLORS["accent_orange"], COLORS["accent_pink"],
    COLORS["accent_red"], COLORS["accent_teal"], COLORS["accent_indigo"],
    COLORS["accent_lime"], COLORS["accent_amber"], COLORS["accent_rose"],
]


# ============================================================================
# Garman-Kohlhagen Inline Pricing
# ============================================================================

def _gk_price(S, K, T, r_d, r_f, sigma, cp):
    """Garman-Kohlhagen option price. cp: +1 call, -1 put."""
    if T <= 1e-10 or sigma <= 1e-10 or S <= 0 or K <= 0:
        return max(cp * (S - K), 0.0)
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return cp * (S * np.exp(-r_f * T) * norm.cdf(cp * d1)
                 - K * np.exp(-r_d * T) * norm.cdf(cp * d2))


def _gk_delta(S, K, T, r_d, r_f, sigma, cp):
    """Garman-Kohlhagen spot delta."""
    if T <= 1e-10 or sigma <= 1e-10 or S <= 0 or K <= 0:
        return cp * (1.0 if cp * (S - K) > 0 else 0.0)
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return cp * np.exp(-r_f * T) * norm.cdf(cp * d1)


def _gk_vega(S, K, T, r_d, r_f, sigma):
    """Garman-Kohlhagen vega (per 1 vol point = 0.01)."""
    if T <= 1e-10 or sigma <= 1e-10 or S <= 0 or K <= 0:
        return 0.0
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-r_f * T) * norm.pdf(d1) * np.sqrt(T) / 100.0


def _build_exec_log_table(trades):
    """Build the execution log DataTable from a list of trade dicts."""
    table_data = []
    for t in trades[:100]:
        table_data.append({
            "Time": t.get("timestamp", ""),
            "Pair": t.get("pair", ""),
            "Type": t.get("type", ""),
            "Side": (t.get("side", "BUY") or "BUY").upper(),
            "Strike": f"{t.get('strike', 0):.5g}",
            "Delta": f"{t.get('delta', 0):.3f}",
            "Expiry": t.get("expiry", ""),
            "Notional": f"{t.get('notional', 0):,.0f}",
            "Premium": f"{t.get('premium', 0):,.0f}",
            "Book": t.get("book", ""),
            "Strategy": t.get("strategy", ""),
            "Cpty": t.get("counterparty", ""),
            "Status": t.get("status", ""),
        })
    table_cols = [{"name": c, "id": c} for c in
                  ["Time", "Pair", "Type", "Side", "Strike", "Delta",
                   "Expiry", "Notional", "Premium", "Book", "Strategy",
                   "Cpty", "Status"]]
    return dash_table.DataTable(
        data=table_data,
        columns=table_cols,
        style_header={
            "backgroundColor": COLORS["bg_secondary"],
            "color": COLORS["text_secondary"],
            "fontWeight": "700", "fontSize": "10px",
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "border": f"1px solid {COLORS['border_subtle']}",
            "padding": "8px 6px",
        },
        style_cell={
            "backgroundColor": COLORS["bg_card"],
            "color": COLORS["text_primary"],
            "fontSize": "11px",
            "fontFamily": "'JetBrains Mono', monospace",
            "border": f"1px solid {COLORS['border_subtle']}",
            "padding": "5px 7px", "textAlign": "center",
            "minWidth": "65px", "maxWidth": "110px",
            "overflow": "hidden", "textOverflow": "ellipsis",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"},
             "backgroundColor": COLORS["bg_secondary"]},
            {"if": {"filter_query": "{Side} = BUY", "column_id": "Side"},
             "color": COLORS["accent_green"], "fontWeight": "700"},
            {"if": {"filter_query": "{Side} = SELL", "column_id": "Side"},
             "color": COLORS["accent_red"], "fontWeight": "700"},
            {"if": {"filter_query": "{Type} = CALL", "column_id": "Type"},
             "color": COLORS["accent_cyan"]},
            {"if": {"filter_query": "{Type} = PUT", "column_id": "Type"},
             "color": COLORS["accent_purple"]},
            {"if": {"filter_query": "{Status} = FILLED", "column_id": "Status"},
             "color": COLORS["accent_green"]},
            {"if": {"filter_query": "{Status} = REJECTED", "column_id": "Status"},
             "color": COLORS["accent_red"]},
        ],
        sort_action="native", filter_action="native",
        page_size=12, page_action="native",
        style_table={"overflowX": "auto"},
    )


def _get_market_params(pair, tenor):
    """Fetch spot, rates, and ATM vol for a pair/tenor combination."""
    try:
        spots = get_fx_spots([pair]) or {}
        S = spots.get(pair, {}).get("mid", 1.0)
        rates = get_fx_rates(pair) or {}
        r_d = rates.get("r_dom", 0.04)
        r_f = rates.get("r_for", 0.03)
        vol_surf = get_fx_vol_surface(pair) or {}
        atm_vol_raw = vol_surf.get(tenor, {}).get("atm", 8.0)
        atm_vol = atm_vol_raw / 100.0 if atm_vol_raw > 1.0 else atm_vol_raw
    except Exception:
        S, r_d, r_f, atm_vol = 1.0, 0.04, 0.03, 0.08
    T = tenor_to_years(tenor)
    return S, T, r_d, r_f, atm_vol


# ============================================================================
# Layout
# ============================================================================

_FORM_LABEL = {**LABEL_STYLE, "marginTop": "10px"}
_FORM_INPUT = {**INPUT_STYLE, "marginBottom": "4px"}

_EXECUTE_BTN_STYLE = {
    **BUTTON_SUCCESS_STYLE,
    "width": "100%",
    "padding": "14px 20px",
    "fontSize": "14px",
    "fontWeight": "800",
    "letterSpacing": "2px",
    "marginTop": "18px",
    "boxShadow": f"0 6px 20px rgba(255,136,0,0.25)",
    "border": f"1px solid {COLORS['accent_green']}",
}

_LEFT_PANEL_STYLE = {
    **CARD_STYLE,
    "width": "400px",
    "minWidth": "360px",
    "flexShrink": "0",
    "maxHeight": "calc(100vh - 120px)",
    "overflowY": "auto",
}

_MAIN_AREA_STYLE = {
    **CARD_STYLE,
    "flex": "1",
    "minWidth": "500px",
    "overflowX": "auto",
}

_HEADER = {
    "color": COLORS["text_primary"],
    "fontSize": "13px",
    "fontWeight": "700",
    "fontFamily": "'JetBrains Mono', monospace",
    "marginBottom": "14px",
    "paddingBottom": "10px",
    "borderBottom": f"1px solid {COLORS['border_subtle']}",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
}


def layout():
    pair_opts = [{"label": p, "value": p} for p in FX_PAIRS]
    tenor_opts = [{"label": t, "value": t} for t in TENORS]
    cut_opts = [{"label": f"{k} - {v['label']}", "value": k} for k, v in CUT_TIMES.items()]
    strat_opts = [{"label": s, "value": s} for s in STRATEGIES]
    book_opts = [{"label": b.replace("_", " "), "value": b} for b in BOOKS]

    return html.Div([
        dcc.Download(id="fxb-csv-download"),
        dcc.Store(id="fxb-pending-legs-store", data=None),

        # ── Structure Review Panel (hidden until SEND TO BLOTTER) ──
        html.Div(id="fxb-review-container", children=[
            html.Div([
                html.Div([
                    html.Span("STRUCTURE REVIEW", style={
                        "color": COLORS["accent_green"], "fontWeight": "800",
                        "fontSize": "12px", "letterSpacing": "2px",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                    html.Span(id="fxb-review-struct-name", style={
                        "color": COLORS["accent_orange"], "fontWeight": "700",
                        "fontSize": "12px", "marginLeft": "12px",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                    html.Button("CANCEL", id="fxb-review-cancel-btn", n_clicks=0,
                                style={**BUTTON_STYLE, "fontSize": "9px",
                                       "padding": "4px 12px", "marginLeft": "auto"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),

                html.Div(id="fxb-review-info", style={
                    "color": COLORS["text_secondary"], "fontSize": "10px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginTop": "6px",
                }),

                html.Div(id="fxb-review-legs-table", style={"marginTop": "10px"}),

                html.Div([
                    html.Button("RE-PRICE ALL", id="fxb-review-reprice-btn", n_clicks=0,
                                style={**BUTTON_STYLE, "fontSize": "10px",
                                       "padding": "8px 16px", "letterSpacing": "1px"}),
                    html.Button("EXECUTE ALL LEGS", id="fxb-review-confirm-btn", n_clicks=0,
                                style={**BUTTON_SUCCESS_STYLE, "fontSize": "11px",
                                       "padding": "10px 24px", "fontWeight": "800",
                                       "letterSpacing": "2px"}),
                ], style={"display": "flex", "gap": "12px", "marginTop": "12px",
                          "justifyContent": "flex-end"}),

                html.Div(id="fxb-review-status", style={
                    "color": COLORS["text_muted"], "fontSize": "9px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginTop": "6px", "textAlign": "right", "minHeight": "14px",
                }),
            ], style={**CARD_STYLE, "borderLeft": f"3px solid {COLORS['accent_green']}",
                      "padding": "12px 16px"}),
        ], style={"display": "none", "marginBottom": "12px"}),

        # ── Row 1: Trade Entry + Execution Log ──────────────────
        html.Div([
            # LEFT: Trade Entry Form
            html.Div([
                html.Div("FX TRADE ENTRY", style=_HEADER),

                html.Label("PAIR", style=_FORM_LABEL),
                dcc.Dropdown(id="fxb-pair", options=pair_opts, value="EURUSD",
                             clearable=False, style={**DROPDOWN_STYLE, "marginBottom": "4px"}),

                html.Div([
                    html.Div([
                        html.Label("TYPE", style=_FORM_LABEL),
                        dcc.RadioItems(
                            id="fxb-cp", options=[
                                {"label": " CALL", "value": "call"},
                                {"label": " PUT", "value": "put"},
                            ], value="call", inline=True,
                            style={"color": COLORS["text_primary"], "fontSize": "12px",
                                   "fontFamily": "monospace"},
                            labelStyle={"marginRight": "14px"},
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("SIDE", style=_FORM_LABEL),
                        dcc.RadioItems(
                            id="fxb-side", options=[
                                {"label": " BUY", "value": "buy"},
                                {"label": " SELL", "value": "sell"},
                            ], value="buy", inline=True,
                            style={"color": COLORS["text_primary"], "fontSize": "12px",
                                   "fontFamily": "monospace"},
                            labelStyle={"marginRight": "14px"},
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px"}),

                html.Label("ENTRY MODE", style=_FORM_LABEL),
                dcc.RadioItems(
                    id="fxb-entry-mode", options=[
                        {"label": " Delta-based", "value": "delta"},
                        {"label": " Strike-based", "value": "strike"},
                    ], value="delta", inline=True,
                    style={"color": COLORS["text_primary"], "fontSize": "12px",
                           "fontFamily": "monospace"},
                    labelStyle={"marginRight": "14px"},
                ),

                # Delta input (shown when delta-based)
                html.Div(id="fxb-delta-row", children=[
                    html.Label("DELTA", style=_FORM_LABEL),
                    dcc.Input(id="fxb-delta", type="number", value=0.25,
                              min=0.05, max=0.95, step=0.01,
                              style=_FORM_INPUT, debounce=True),
                    html.Div(id="fxb-computed-strike",
                             style={"color": COLORS["accent_cyan"], "fontSize": "11px",
                                    "fontFamily": "monospace", "marginTop": "2px"}),
                ]),

                # Strike input (shown when strike-based)
                html.Div(id="fxb-strike-row", children=[
                    html.Label("STRIKE", style=_FORM_LABEL),
                    dcc.Input(id="fxb-strike", type="number", value=1.0800,
                              step=0.0001, style=_FORM_INPUT, debounce=True),
                    html.Div(id="fxb-computed-delta",
                             style={"color": COLORS["accent_cyan"], "fontSize": "11px",
                                    "fontFamily": "monospace", "marginTop": "2px"}),
                ], style={"display": "none"}),

                html.Label("TENOR", style=_FORM_LABEL),
                dcc.Dropdown(id="fxb-tenor", options=tenor_opts, value="3M",
                             clearable=False, style={**DROPDOWN_STYLE, "marginBottom": "4px"}),

                html.Label("NOTIONAL", style=_FORM_LABEL),
                dcc.Input(id="fxb-notional", type="number", value=10_000_000,
                          step=1_000_000, min=100_000, style=_FORM_INPUT, debounce=True),

                html.Div([
                    html.Label("PREMIUM (auto)", style=_FORM_LABEL),
                    html.Div(id="fxb-premium-display",
                             style={"color": COLORS["accent_orange"], "fontSize": "14px",
                                    "fontWeight": "700", "fontFamily": "monospace",
                                    "padding": "8px 12px",
                                    "backgroundColor": COLORS["bg_input"],
                                    "borderRadius": "0px",
                                    "border": f"1px solid {COLORS['border']}"}),
                ]),

                html.Label("CUT TIME", style=_FORM_LABEL),
                dcc.Dropdown(id="fxb-cut", options=cut_opts, value="NY",
                             clearable=False, style={**DROPDOWN_STYLE, "marginBottom": "4px"}),

                html.Div([
                    html.Div([
                        html.Label("STRATEGY", style=_FORM_LABEL),
                        dcc.Dropdown(id="fxb-strategy", options=strat_opts, value="Prop",
                                     clearable=False,
                                     style={**DROPDOWN_STYLE, "marginBottom": "4px"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("BOOK", style=_FORM_LABEL),
                        dcc.Dropdown(id="fxb-book", options=book_opts, value="G10_FLOW",
                                     clearable=False,
                                     style={**DROPDOWN_STYLE, "marginBottom": "4px"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "10px"}),

                html.Label("COUNTERPARTY", style=_FORM_LABEL),
                dcc.Input(id="fxb-cpty", type="text", value="",
                          placeholder="e.g. JPM, GS, CITI...",
                          style=_FORM_INPUT, debounce=True),

                html.Label("NOTES", style=_FORM_LABEL),
                dcc.Textarea(id="fxb-notes", value="",
                             placeholder="Optional trade notes...",
                             style={**_FORM_INPUT, "height": "48px", "resize": "vertical"}),

                html.Button("EXECUTE TRADE", id="fxb-execute-btn",
                            style=_EXECUTE_BTN_STYLE),

                html.Div(id="fxb-exec-status",
                         style={"marginTop": "8px", "fontSize": "11px",
                                "fontFamily": "monospace", "textAlign": "center"}),
            ], style=_LEFT_PANEL_STYLE),

            # RIGHT: Execution Log
            html.Div([
                html.Div("EXECUTION LOG", style=_HEADER),
                html.Div(id="fxb-exec-table"),
            ], style=_MAIN_AREA_STYLE),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start",
                  "flexWrap": "wrap"}),

        # ── Row 2: Flow Analytics (3 charts) ────────────────────
        html.Div([
            html.Div([
                html.Button("CSV", id="fxb-csv-notional", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="fxb-notional-chart", style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "320px"},
               className="dashboard-card"),
            html.Div([
                html.Button("CSV", id="fxb-csv-premium", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="fxb-premium-flow-chart", style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "320px"},
               className="dashboard-card"),
            html.Div([
                html.Button("CSV", id="fxb-csv-activity", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="fxb-activity-chart", style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "320px"},
               className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Row 3: Day Summary Stats ────────────────────────────
        html.Div(id="fxb-day-stats",
                 style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                        "marginBottom": "16px"}),

        # Hidden stores
        dcc.Store(id="fxb-trade-store", data=json.dumps([])),
        dcc.Store(id="fxb-market-cache", data="{}"),
    ])


# ============================================================================
# Callbacks
# ============================================================================

def register_callbacks(app):

    # ── 1. Entry mode toggle: show/hide delta vs strike row ──────────────
    @app.callback(
        [Output("fxb-delta-row", "style"),
         Output("fxb-strike-row", "style")],
        [Input("fxb-entry-mode", "value")],
    )
    def toggle_entry_mode(mode):
        if mode == "delta":
            return {"display": "block"}, {"display": "none"}
        return {"display": "none"}, {"display": "block"}

    # ── 2. Auto-compute strike from delta (and vice versa) + premium ─────
    @app.callback(
        [Output("fxb-computed-strike", "children"),
         Output("fxb-computed-delta", "children"),
         Output("fxb-premium-display", "children")],
        [Input("fxb-pair", "value"),
         Input("fxb-cp", "value"),
         Input("fxb-delta", "value"),
         Input("fxb-strike", "value"),
         Input("fxb-tenor", "value"),
         Input("fxb-notional", "value"),
         Input("fxb-entry-mode", "value"),
         Input("fxb-side", "value")],
    )
    def compute_pricing(pair, cp_str, delta_in, strike_in, tenor, notional,
                        entry_mode, side):
        pair = pair or "EURUSD"
        tenor = tenor or "3M"
        notional = notional or 10_000_000
        cp = 1 if cp_str == "call" else -1

        S, T, r_d, r_f, atm_vol = _get_market_params(pair, tenor)
        sigma = atm_vol if atm_vol > 0 else 0.10

        strike_text = ""
        delta_text = ""
        premium_text = "--"

        try:
            if entry_mode == "delta":
                d = float(delta_in or 0.25)
                d = max(0.05, min(0.95, d))
                signed_delta = d if cp == 1 else -d
                K = delta_to_strike(signed_delta, S, T, r_d, r_f, sigma, cp)
                if np.isnan(K) or K <= 0:
                    K = atm_dns_strike(S, T, r_d, r_f, sigma)
                strike_text = f"Computed Strike: {K:.5g}"
                actual_delta = _gk_delta(S, K, T, r_d, r_f, sigma, cp)
                prem_unit = abs(_gk_price(S, K, T, r_d, r_f, sigma, cp))
                prem_total = prem_unit * notional
                sign_label = "PAY" if side == "buy" else "RCV"
                premium_text = f"{sign_label} {prem_total:,.0f} ({prem_unit:.6f}/unit)" if np.isfinite(prem_total) else "Error: NaN premium"
            else:
                K = float(strike_in or S)
                if K <= 0:
                    K = S
                actual_delta = _gk_delta(S, K, T, r_d, r_f, sigma, cp)
                delta_text = f"Computed Delta: {actual_delta:.4f}" if np.isfinite(actual_delta) else "Computed Delta: —"
                prem_unit = abs(_gk_price(S, K, T, r_d, r_f, sigma, cp))
                prem_total = prem_unit * notional
                sign_label = "PAY" if side == "buy" else "RCV"
                premium_text = f"{sign_label} {prem_total:,.0f} ({prem_unit:.6f}/unit)" if np.isfinite(prem_total) else "Error: NaN premium"
        except Exception:
            premium_text = "Error computing premium"

        return strike_text, delta_text, premium_text

    # ── 3. Execute trade ─────────────────────────────────────────────────
    # ── 4. Update execution log + flow analytics + summary stats ─────────
    @app.callback(
        [Output("fxb-exec-table", "children", allow_duplicate=True),
         Output("fxb-notional-chart", "figure"),
         Output("fxb-premium-flow-chart", "figure"),
         Output("fxb-activity-chart", "figure"),
         Output("fxb-day-stats", "children"),
         Output("fxb-trade-store", "data", allow_duplicate=True),
         Output("fxb-exec-status", "children", allow_duplicate=True),
         Output("fxb-exec-status", "style", allow_duplicate=True),
         Output("global-portfolio-version", "data", allow_duplicate=True)],
        [Input("fxb-execute-btn", "n_clicks")],
        [State("fxb-trade-store", "data"),
         State("global-portfolio-version", "data"),
         State("fxb-pair", "value"),
         State("fxb-cp", "value"),
         State("fxb-side", "value"),
         State("fxb-entry-mode", "value"),
         State("fxb-delta", "value"),
         State("fxb-strike", "value"),
         State("fxb-tenor", "value"),
         State("fxb-notional", "value"),
         State("fxb-cut", "value"),
         State("fxb-strategy", "value"),
         State("fxb-book", "value"),
         State("fxb-cpty", "value"),
         State("fxb-notes", "value")],
        prevent_initial_call="initial_duplicate",
    )
    def update_all(n_clicks, store_data, portfolio_version,
                   pair, cp_str, side, entry_mode,
                   delta_in, strike_in, tenor, notional, cut, strategy,
                   book, cpty, notes):
        tpl = CHART_TEMPLATE["layout"]
        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        try:
            trades = json.loads(store_data) if store_data else []
        except (json.JSONDecodeError, TypeError):
            trades = []
        new_store = no_update
        trade_executed = False
        exec_msg = ""
        exec_style = {"marginTop": "8px", "fontSize": "11px",
                      "fontFamily": "monospace", "textAlign": "center"}

        # ── Handle new trade execution ──────────────────────────
        if "fxb-execute-btn" in triggered and n_clicks and n_clicks > 0:
            pair = pair or "EURUSD"
            tenor = tenor or "3M"
            notional = notional or 10_000_000
            cp = 1 if cp_str == "call" else -1

            S, T, r_d, r_f, atm_vol = _get_market_params(pair, tenor)
            sigma = atm_vol if atm_vol > 0 else 0.10

            try:
                if entry_mode == "delta":
                    d = float(delta_in or 0.25)
                    d = max(0.05, min(0.95, d))
                    signed_delta = d if cp == 1 else -d
                    K = delta_to_strike(signed_delta, S, T, r_d, r_f, sigma, cp)
                    if np.isnan(K) or K <= 0:
                        K = atm_dns_strike(S, T, r_d, r_f, sigma)
                else:
                    K = float(strike_in or S)
                    if K <= 0:
                        K = S

                actual_delta = _gk_delta(S, K, T, r_d, r_f, sigma, cp)
                prem_unit = abs(_gk_price(S, K, T, r_d, r_f, sigma, cp))
                prem_total = prem_unit * notional
                vega_val = _gk_vega(S, K, T, r_d, r_f, sigma) * notional
                expiry_date = datetime.now() + timedelta(days=tenor_to_days(tenor))

                # Compute gamma and theta at entry
                from scipy.stats import norm as _norm
                if T > 1e-6 and sigma > 1e-6:
                    _d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    _npd1 = _norm.pdf(_d1)
                    _df_f = np.exp(-r_f * T)
                    gamma_val = _df_f * _npd1 / (S * sigma * np.sqrt(T)) * notional
                    theta_val = (-0.5 * S * _df_f * _npd1 * sigma / np.sqrt(T)) / 365.0 * notional
                else:
                    gamma_val = 0.0
                    theta_val = 0.0

                new_trade = {
                    "id": f"FX-{30000 + n_clicks:05d}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pair": pair,
                    "type": "CALL" if cp == 1 else "PUT",
                    "side": side.upper(),
                    "strike": round(K, 5),
                    "delta": round(actual_delta, 4),
                    "tenor": tenor,
                    "expiry": expiry_date.strftime("%Y-%m-%d"),
                    "notional": notional,
                    "premium": round(prem_total, 2),
                    "premium_per_unit": round(prem_unit, 6),
                    "vol": round(sigma * 100, 2),
                    "vega": round(vega_val, 2),
                    "gamma": round(gamma_val, 4),
                    "theta": round(theta_val, 2),
                    "entry_spot": round(S, 5),
                    "book": book or "G10_FLOW",
                    "strategy": strategy or "Prop",
                    "counterparty": cpty or "INTERBANK",
                    "cut": cut or "NY",
                    "notes": notes or "",
                    "status": "FILLED",
                }
                # Add to portfolio engine — only prepend trade if validation succeeds
                add_position(book, {
                    "pair": pair,
                    "option_type": cp_str,
                    "direction": side,
                    "strike": round(K, 5),
                    "expiry": expiry_date.strftime("%Y-%m-%d"),
                    "notional": notional,
                    "entry_vol": sigma,
                    "entry_premium": round(prem_total, 2),
                    "delta_at_entry": round(actual_delta, 4),
                    "cut": cut or "NY",
                    "counterparty": cpty or "INTERBANK",
                    "strategy": strategy or "Prop",
                    "notes": notes or "",
                })

                trades = [new_trade] + trades
                new_store = json.dumps(trades)
                trade_executed = True
                exec_msg = f"FILLED  {pair} {tenor} {K:.5g} {side.upper()}"
                exec_style = {**exec_style, "color": COLORS["accent_green"]}
            except Exception as exc:
                exec_msg = f"REJECTED  {str(exc)[:120]}"
                exec_style = {**exec_style, "color": COLORS["accent_red"]}

        # ── Build Execution Log Table ───────────────────────────
        exec_table = _build_exec_log_table(trades)

        # ── Flow Analytics & Day Summary ────────────────────────
        # Wrapped in try/except so a malformed trade does not crash the panel.
        try:
            # ── Flow Analytics: Notional by Pair ────────────────────
            pair_book_not = {}
            for t in trades:
                key = (t["pair"], t.get("book", "G10_FLOW"))
                pair_book_not[key] = pair_book_not.get(key, 0) + t["notional"]

            pairs_seen = sorted(set(t["pair"] for t in trades))
            books_seen = sorted(set(t.get("book", "G10_FLOW") for t in trades))

            notional_fig = go.Figure()
            for bk in books_seen:
                vals = [pair_book_not.get((p, bk), 0) for p in pairs_seen]
                notional_fig.add_trace(go.Bar(
                    name=bk.replace("_", " "),
                    x=pairs_seen,
                    y=vals,
                    marker_color=BOOK_COLORS.get(bk, COLORS["accent_blue"]),
                    hovertemplate="%{x}: %{y:,.0f}<extra>" + bk + "</extra>",
                ))
            notional_fig.update_layout(
                title=dict(text="Notional by Pair & Book",
                           font=dict(color=COLORS["text_primary"], size=13)),
                barmode="stack",
                paper_bgcolor=tpl["paper_bgcolor"],
                plot_bgcolor=tpl["plot_bgcolor"],
                font=tpl["font"],
                margin=dict(l=60, r=20, t=45, b=50),
                xaxis=dict(gridcolor="rgba(34,34,64,0.5)",
                           tickfont=dict(size=9)),
                yaxis=dict(title="Notional (USD)", gridcolor="rgba(34,34,64,0.5)"),
                legend=dict(font=dict(size=9), orientation="h",
                            yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                hoverlabel=tpl["hoverlabel"],
            )

            # ── Flow Analytics: Cumulative Premium Flow ─────────────
            sorted_trades = sorted(trades, key=lambda x: x.get("timestamp", ""))
            cum_prems = []
            running = 0.0
            times = []
            daily_bars = {}
            for t in sorted_trades:
                sign = 1 if t.get("side", "BUY").upper() == "BUY" else -1
                prem_signed = -sign * t.get("premium", 0)  # pay on buy, receive on sell
                running += prem_signed
                cum_prems.append(running)
                times.append(t.get("timestamp", ""))
                day_key = t.get("timestamp", "")[:10]
                daily_bars[day_key] = daily_bars.get(day_key, 0) + prem_signed

            prem_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     row_heights=[0.65, 0.35],
                                     vertical_spacing=0.08)
            prem_fig.add_trace(
                go.Scatter(x=times, y=cum_prems, mode="lines",
                           line=dict(color=COLORS["accent_cyan"], width=2),
                           fill="tozeroy",
                           fillcolor="rgba(255,136,0,0.06)",
                           name="Cumulative",
                           hovertemplate="%{x}<br>Cumulative: $%{y:,.0f}<extra></extra>"),
                row=1, col=1,
            )
            prem_fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1,
                                               dash="dot"), row=1, col=1)

            day_keys = sorted(daily_bars.keys())
            day_vals = [daily_bars[d] for d in day_keys]
            day_colors = [COLORS["accent_green"] if v >= 0 else COLORS["accent_red"]
                          for v in day_vals]
            prem_fig.add_trace(
                go.Bar(x=day_keys, y=day_vals, marker_color=day_colors,
                       name="Daily",
                       hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"),
                row=2, col=1,
            )
            prem_fig.update_layout(
                title=dict(text="Cumulative Premium Flow",
                           font=dict(color=COLORS["text_primary"], size=13)),
                paper_bgcolor=tpl["paper_bgcolor"],
                plot_bgcolor=tpl["plot_bgcolor"],
                font=tpl["font"],
                margin=dict(l=60, r=20, t=45, b=40),
                showlegend=False,
                hoverlabel=tpl["hoverlabel"],
            )
            prem_fig.update_xaxes(gridcolor="rgba(34,34,64,0.5)")
            prem_fig.update_yaxes(gridcolor="rgba(34,34,64,0.5)")
            prem_fig.update_yaxes(title_text="Cumulative ($)", row=1, col=1)
            prem_fig.update_yaxes(title_text="Daily ($)", row=2, col=1)
            prem_fig.update_xaxes(title_text="Date", row=2, col=1)

            # ── Flow Analytics: Activity Timeline ───────────────────
            activity_fig = go.Figure()
            pair_list = sorted(set(t.get("pair", "") for t in trades))
            for idx, p in enumerate(pair_list):
                p_trades = [t for t in trades if t.get("pair") == p]
                activity_fig.add_trace(go.Scatter(
                    x=[t.get("timestamp", "") for t in p_trades],
                    y=[t.get("notional", 0) for t in p_trades],
                    mode="markers",
                    marker=dict(
                        size=[max(6, min(30, t.get("notional", 0) / 3_000_000))
                              for t in p_trades],
                        color=PAIR_COLORS[idx % len(PAIR_COLORS)],
                        opacity=0.75,
                        line=dict(width=1, color=COLORS["border"]),
                    ),
                    name=p,
                    hovertemplate=(
                        "%{x}<br>" + p +
                        "<br>Notional: %{y:,.0f}<extra></extra>"
                    ),
                ))
            activity_fig.update_layout(
                title=dict(text="Activity Timeline",
                           font=dict(color=COLORS["text_primary"], size=13)),
                paper_bgcolor=tpl["paper_bgcolor"],
                plot_bgcolor=tpl["plot_bgcolor"],
                font=tpl["font"],
                margin=dict(l=60, r=20, t=45, b=40),
                xaxis=dict(gridcolor="rgba(34,34,64,0.5)", title="Time"),
                yaxis=dict(gridcolor="rgba(34,34,64,0.5)", title="Notional (USD)"),
                legend=dict(font=dict(size=9), orientation="h",
                            yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                hoverlabel=tpl["hoverlabel"],
            )

            # ── Day Summary Stats ───────────────────────────────────
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_trades = [t for t in trades
                            if t.get("timestamp", "")[:10] == today_str]
            if not today_trades:
                today_trades = trades[:10]

            n_trades = len(today_trades)
            total_notional = sum(t.get("notional", 0) for t in today_trades)
            net_delta = sum(
                t.get("delta", 0) * t.get("notional", 0)
                * (1 if t.get("side", "BUY").upper() == "BUY" else -1)
                for t in today_trades
            )
            net_vega = sum(
                t.get("vega", 0)
                * (1 if t.get("side", "BUY").upper() == "BUY" else -1)
                for t in today_trades
            )
            net_premium = sum(
                t.get("premium", 0)
                * (-1 if t.get("side", "BUY").upper() == "BUY" else 1)
                for t in today_trades
            )
            net_gamma = sum(
                t.get("gamma", 0)
                * (1 if t.get("side", "BUY").upper() == "BUY" else -1)
                for t in today_trades
            )

            def _stat_box(label, value, color):
                return html.Div([
                    html.Div(str(value), style={
                        "color": color, "fontSize": "18px", "fontWeight": "700",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                    html.Div(label, style={
                        "color": COLORS["text_secondary"], "fontSize": "10px",
                        "fontWeight": "600", "textTransform": "uppercase",
                        "letterSpacing": "1px", "marginTop": "4px",
                    }),
                ], style={**make_stat_style(color), "flex": "1", "minWidth": "150px"},
                   className="stat-box")

            def _nf(v, fmt=",.0f"):
                return f"{v:{fmt}}" if np.isfinite(v) else "—"

            prem_label = "Premium Paid" if np.isfinite(net_premium) and net_premium < 0 else "Premium Received"
            prem_color = COLORS["accent_red"] if np.isfinite(net_premium) and net_premium < 0 else COLORS["accent_green"]

            day_stats = [
                _stat_box("Trades Today", f"{n_trades}", COLORS["accent_cyan"]),
                _stat_box("Total Notional", _nf(total_notional), COLORS["accent_blue"]),
                _stat_box("Net Delta Added", _nf(net_delta), COLORS["accent_purple"]),
                _stat_box("Net Vega Added", _nf(net_vega), COLORS["accent_orange"]),
                _stat_box("Net Gamma", _nf(net_gamma, "+,.4f"), COLORS["accent_teal"]),
                _stat_box(prem_label, _nf(abs(net_premium)) if np.isfinite(net_premium) else "—", prem_color),
            ]

        except Exception:
            # On any chart-building error, return safe empty figures and stats
            _empty = go.Figure()
            _empty.update_layout(
                paper_bgcolor=tpl["paper_bgcolor"],
                plot_bgcolor=tpl["plot_bgcolor"],
                font=tpl["font"],
                annotations=[dict(
                    text="Chart error", x=0.5, y=0.5,
                    xref="paper", yref="paper", showarrow=False,
                    font=dict(color=COLORS["text_muted"], size=14),
                )],
            )
            notional_fig = _empty
            prem_fig = _empty
            activity_fig = _empty
            day_stats = [html.Div("Error computing stats",
                                  style={"color": COLORS["accent_red"],
                                         "fontFamily": "monospace"})]

        new_version = (portfolio_version or 0) + 1 if trade_executed else no_update
        return (exec_table, notional_fig, prem_fig, activity_fig,
                day_stats, new_store, exec_msg, exec_style, new_version)

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output("fxb-csv-download", "data"),
        [Input("fxb-csv-notional", "n_clicks"),
         Input("fxb-csv-premium", "n_clicks"),
         Input("fxb-csv-activity", "n_clicks")],
        [State("fxb-notional-chart", "figure"),
         State("fxb-premium-flow-chart", "figure"),
         State("fxb-activity-chart", "figure")],
        prevent_initial_call=True,
    )
    def fxb_csv_export(n1, n2, n3, fig1, fig2, fig3):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        mapping = {
            "fxb-csv-notional": (fig1, "Blotter", "Notional"),
            "fxb-csv-premium": (fig2, "Blotter", "PremiumFlow"),
            "fxb-csv-activity": (fig3, "Blotter", "Activity"),
        }
        if btn not in mapping:
            return no_update
        fig, panel, chart_type = mapping[btn]
        if not fig:
            return no_update
        try:
            return export_csv(fig, panel, chart_type)
        except Exception:
            return no_update

    # ==================================================================
    # STRUCTURE BUILDER → BLOTTER: Receive, Review, Execute
    # ==================================================================

    def _build_review_table(legs):
        """Build an HTML table showing pending legs for review."""
        header_style = {
            "backgroundColor": COLORS["bg_secondary"], "color": COLORS["text_secondary"],
            "fontWeight": "700", "fontSize": "9px", "textTransform": "uppercase",
            "letterSpacing": "1px", "padding": "6px 8px",
            "border": f"1px solid {COLORS['border_subtle']}",
            "fontFamily": "'JetBrains Mono', monospace",
        }
        cell_style = {
            "padding": "5px 8px", "fontSize": "11px", "color": COLORS["text_primary"],
            "border": f"1px solid {COLORS['border_subtle']}",
            "fontFamily": "'JetBrains Mono', monospace",
            "backgroundColor": COLORS["bg_primary"],
        }
        cols = ["#", "C/P", "Side", "Strike", "Delta", "Vol%", "Notional", "Premium", "Vega", "Theta"]
        header = html.Tr([html.Th(c, style=header_style) for c in cols])
        rows = []
        for i, lg in enumerate(legs):
            side_color = COLORS["accent_green"] if lg["side"].lower() == "buy" else COLORS["accent_red"]
            rows.append(html.Tr([
                html.Td(str(i + 1), style=cell_style),
                html.Td(lg["cp"].upper(), style=cell_style),
                html.Td(lg["side"].upper(), style={**cell_style, "color": side_color, "fontWeight": "700"}),
                html.Td(f"{lg['strike']:.5f}", style=cell_style),
                html.Td(f"{lg['greeks']['delta']:.4f}", style=cell_style),
                html.Td(f"{lg['vol'] * 100:.2f}" if lg["vol"] < 1 else f"{lg['vol']:.2f}", style=cell_style),
                html.Td(f"{lg['notional']:,.0f}", style=cell_style),
                html.Td(f"{lg['premium_total']:,.2f}", style=cell_style),
                html.Td(f"{lg['greeks']['vega']:,.2f}", style=cell_style),
                html.Td(f"{lg['greeks']['theta']:,.2f}", style=cell_style),
            ]))
        return html.Table([html.Thead(header), html.Tbody(rows)],
                          style={"width": "100%", "borderCollapse": "collapse"})

    # ── Callback 1: Receive structure from Structure Builder ──────────
    @app.callback(
        [Output("fxb-review-container", "style"),
         Output("fxb-review-struct-name", "children"),
         Output("fxb-review-info", "children"),
         Output("fxb-review-legs-table", "children"),
         Output("fxb-pending-legs-store", "data")],
        [Input("stb-to-blotter-store", "data")],
        prevent_initial_call=True,
    )
    def receive_structure(transfer_data):
        if not transfer_data or not transfer_data.get("legs"):
            raise PreventUpdate
        legs = transfer_data["legs"]
        name = transfer_data.get("structure_name", "Custom")
        pair = transfer_data.get("pair", "?")
        tenor = transfer_data.get("tenor", "?")
        notional = transfer_data.get("notional", 0)
        ts = transfer_data.get("timestamp", "")

        info = f"{pair}  |  {tenor}  |  {notional:,.0f} notional  |  priced {ts}"
        table = _build_review_table(legs)

        return (
            {"display": "block", "marginBottom": "12px"},
            name.upper(),
            info,
            table,
            transfer_data,
        )

    # ── Callback 2: Re-price legs with fresh market data ──────────────
    @app.callback(
        [Output("fxb-review-legs-table", "children", allow_duplicate=True),
         Output("fxb-pending-legs-store", "data", allow_duplicate=True),
         Output("fxb-review-status", "children", allow_duplicate=True)],
        [Input("fxb-review-reprice-btn", "n_clicks")],
        [State("fxb-pending-legs-store", "data")],
        prevent_initial_call=True,
    )
    def reprice_legs(n_clicks, pending_data):
        if not n_clicks or not pending_data:
            raise PreventUpdate
        legs = pending_data.get("legs", [])
        pair = pending_data.get("pair", "EURUSD")
        if not legs:
            raise PreventUpdate

        # Fetch fresh market data
        S, T_base, r_d, r_f, atm_vol = _get_market_params(pair, pending_data.get("tenor", "3M"))

        updated_legs = []
        for lg in legs:
            leg_tenor = lg.get("tenor", pending_data.get("tenor", "3M"))
            S_fresh, T, r_d_l, r_f_l, _ = _get_market_params(pair, leg_tenor)

            # Look up vol for this leg's strike from vol surface
            vol_surf = get_fx_vol_surface(pair) or {}
            tenor_data = vol_surf.get(leg_tenor, {})
            sigma = tenor_data.get("atm", 8.0)
            sigma = sigma / 100.0 if sigma > 1.0 else sigma
            if sigma <= 0:
                sigma = lg.get("vol", 0.08)

            K = lg["strike"]
            cp = 1 if lg["cp"].lower() == "call" else -1
            side_sign = 1 if lg["side"].lower() == "buy" else -1
            notional = lg["notional"]

            price_unit = _gk_price(S_fresh, K, T, r_d_l, r_f_l, sigma, cp)
            delta_val = _gk_delta(S_fresh, K, T, r_d_l, r_f_l, sigma, cp)
            vega_val = _gk_vega(S_fresh, K, T, r_d_l, r_f_l, sigma) * notional
            gamma_val = 0.0
            theta_val = 0.0
            if T > 1e-10 and sigma > 1e-10:
                from scipy.stats import norm
                d1 = (np.log(S_fresh / K) + (r_d_l - r_f_l + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                _df_f = np.exp(-r_f_l * T)
                _npd1 = norm.pdf(d1)
                gamma_val = _df_f * _npd1 / (S_fresh * sigma * np.sqrt(T)) * notional
                theta_val = (-0.5 * S_fresh * _df_f * _npd1 * sigma / np.sqrt(T)) / 365.0 * notional

            updated = {**lg,
                       "S": S_fresh, "T": T, "r_d": r_d_l, "r_f": r_f_l,
                       "vol": sigma,
                       "price_unit": price_unit,
                       "premium_total": round(price_unit * notional * side_sign, 2),
                       "greeks": {
                           "delta": round(delta_val * side_sign, 4),
                           "gamma": round(gamma_val, 6),
                           "vega": round(vega_val * side_sign, 2),
                           "theta": round(theta_val * side_sign, 2),
                       }}
            updated_legs.append(updated)

        updated_data = {**pending_data, "legs": updated_legs,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        table = _build_review_table(updated_legs)
        status = f"Re-priced at {updated_data['timestamp']}  |  Spot: {S:.5f}"
        return table, updated_data, status

    # ── Callback 3: Execute all confirmed legs ────────────────────────
    @app.callback(
        [Output("fxb-review-container", "style", allow_duplicate=True),
         Output("fxb-review-status", "children", allow_duplicate=True),
         Output("fxb-trade-store", "data", allow_duplicate=True),
         Output("fxb-exec-status", "children", allow_duplicate=True),
         Output("fxb-exec-status", "style", allow_duplicate=True),
         Output("global-portfolio-version", "data", allow_duplicate=True),
         Output("fxb-exec-table", "children", allow_duplicate=True)],
        [Input("fxb-review-confirm-btn", "n_clicks")],
        [State("fxb-pending-legs-store", "data"),
         State("fxb-trade-store", "data"),
         State("global-portfolio-version", "data")],
        prevent_initial_call=True,
    )
    def execute_structure(n_clicks, pending_data, trade_store_json, portfolio_version):
        if not n_clicks or not pending_data:
            raise PreventUpdate
        legs = pending_data.get("legs", [])
        if not legs:
            raise PreventUpdate

        pair = pending_data.get("pair", "EURUSD")
        structure_name = pending_data.get("structure_name", "Custom")
        base_tenor = pending_data.get("tenor", "3M")

        try:
            trades = json.loads(trade_store_json) if trade_store_json else []
        except (json.JSONDecodeError, TypeError):
            trades = []
        struct_id = len(trades) + 30000
        new_trades = []
        filled = 0

        for i, lg in enumerate(legs):
            try:
                leg_tenor = lg.get("tenor", base_tenor)
                S, T, r_d, r_f, atm_vol = _get_market_params(pair, leg_tenor)
                K = lg["strike"]
                cp = 1 if lg["cp"].lower() == "call" else -1
                cp_str = "call" if cp == 1 else "put"
                side = lg["side"].lower()
                side_sign = 1 if side == "buy" else -1
                notional = lg["notional"]
                sigma = lg.get("vol", atm_vol)
                if sigma > 1.0:
                    sigma = sigma / 100.0

                # Re-price at execution time
                prem_unit = _gk_price(S, K, T, r_d, r_f, sigma, cp)
                prem_total = prem_unit * notional * side_sign
                actual_delta = _gk_delta(S, K, T, r_d, r_f, sigma, cp)
                vega_val = _gk_vega(S, K, T, r_d, r_f, sigma) * notional * side_sign
                gamma_val = 0.0
                theta_val = 0.0
                if T > 1e-10 and sigma > 1e-10:
                    from scipy.stats import norm
                    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                    _df_f = np.exp(-r_f * T)
                    _npd1 = norm.pdf(d1)
                    gamma_val = _df_f * _npd1 / (S * sigma * np.sqrt(T)) * notional
                    theta_val = (-0.5 * S * _df_f * _npd1 * sigma / np.sqrt(T)) / 365.0 * notional

                expiry_date = datetime.now() + timedelta(days=tenor_to_days(leg_tenor))
                trade_id = f"FX-S{struct_id:05d}-L{i + 1}"

                new_trade = {
                    "id": trade_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pair": pair,
                    "type": "CALL" if cp == 1 else "PUT",
                    "side": side.upper(),
                    "strike": round(K, 5),
                    "delta": round(actual_delta, 4),
                    "tenor": leg_tenor,
                    "expiry": expiry_date.strftime("%Y-%m-%d"),
                    "notional": notional,
                    "premium": round(prem_total, 2),
                    "premium_per_unit": round(prem_unit, 6),
                    "vol": round(sigma * 100, 2),
                    "vega": round(vega_val, 2),
                    "gamma": round(gamma_val, 4),
                    "theta": round(theta_val, 2),
                    "entry_spot": round(S, 5),
                    "book": "G10_FLOW",
                    "strategy": structure_name,
                    "counterparty": "INTERBANK",
                    "cut": "NY",
                    "notes": f"Leg {i + 1}/{len(legs)} of {structure_name}",
                    "status": "FILLED",
                }
                # Sync with portfolio engine
                add_position("G10_FLOW", {
                    "pair": pair,
                    "option_type": cp_str,
                    "direction": side,
                    "strike": round(K, 5),
                    "expiry": expiry_date.strftime("%Y-%m-%d"),
                    "notional": notional,
                    "entry_vol": sigma,
                    "entry_premium": round(prem_total, 2),
                    "delta_at_entry": round(actual_delta, 4),
                    "cut": "NY",
                    "counterparty": "INTERBANK",
                    "strategy": structure_name,
                    "notes": f"Leg {i + 1}/{len(legs)} of {structure_name}",
                })
                new_trades.append(new_trade)
                filled += 1
            except Exception as exc:
                new_trades.append({
                    "id": f"FX-S{struct_id:05d}-L{i + 1}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pair": pair, "type": lg["cp"].upper(),
                    "side": lg["side"].upper(), "strike": lg["strike"],
                    "delta": 0, "tenor": lg.get("tenor", base_tenor),
                    "expiry": "", "notional": lg["notional"],
                    "premium": 0, "premium_per_unit": 0, "vol": 0,
                    "vega": 0, "gamma": 0, "theta": 0, "entry_spot": 0,
                    "book": "G10_FLOW", "strategy": structure_name,
                    "counterparty": "INTERBANK", "cut": "NY",
                    "notes": f"REJECTED: {str(exc)[:80]}",
                    "status": "REJECTED",
                })

        all_trades = new_trades + trades
        new_version = (portfolio_version or 0) + 1
        exec_msg = f"FILLED  {structure_name} ({filled}/{len(legs)} legs)  {pair}"
        exec_style_out = {
            "color": COLORS["accent_green"] if filled == len(legs) else COLORS["accent_orange"],
            "fontSize": "11px", "fontFamily": "'JetBrains Mono', monospace",
            "padding": "4px 0",
        }

        return (
            {"display": "none"},
            f"Executed {filled}/{len(legs)} legs",
            json.dumps(all_trades),
            exec_msg,
            exec_style_out,
            new_version,
            _build_exec_log_table(all_trades),
        )

    # ── Callback 4: Cancel review ─────────────────────────────────────
    @app.callback(
        [Output("fxb-review-container", "style", allow_duplicate=True),
         Output("fxb-pending-legs-store", "data", allow_duplicate=True)],
        [Input("fxb-review-cancel-btn", "n_clicks")],
        prevent_initial_call=True,
    )
    def cancel_review(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return {"display": "none"}, None
