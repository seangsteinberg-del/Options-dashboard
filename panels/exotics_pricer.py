"""
Exotic Options Pricing Lab
===========================
Institutional-grade FX exotic options pricer supporting 11 product types:
barriers, double barriers, digitals, one-touch, no-touch, double no-touch,
range accrual, Asians, lookbacks, forward starts, best-of/worst-of, and TARFs.

Fully integrated with the Bloomberg FX data provider and core exotic pricing
engine.  Every product generates payoff diagrams, MC path visualisations,
and spot/vol sensitivity charts.
"""

import dash
from dash import html, dcc, Input, Output, State, no_update, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, INPUT_STYLE, BUTTON_STYLE,
    make_stat_style, CARD_HEADER_STYLE, CSV_BTN_STYLE,
)
from core.csv_export import export_csv
from core.bloomberg_fx import get_fx_vol_surface, get_fx_spots, get_fx_rates, get_all_pairs, get_fx_correlation
from core.fx_exotics import (
    barrier_price, double_barrier_price, digital_price, digital_greeks,
    one_touch_price, no_touch_price, double_no_touch_price, range_accrual_price,
    asian_price, lookback_price, forward_start_price, best_of_price, tarf_price,
    exotic_greeks, _mc_paths, _gk_price,
)
from core.fx_conventions import (
    FX_PAIR_REGISTRY, all_pairs, tenor_to_years, delta_to_strike,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TENORS = ["1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"]

PRODUCTS = [
    {"label": "Barrier (KI/KO)",       "value": "barrier"},
    {"label": "Double Barrier",         "value": "double_barrier"},
    {"label": "Digital / Binary",       "value": "digital"},
    {"label": "One-Touch / No-Touch",   "value": "one_touch"},
    {"label": "Double No-Touch",        "value": "dnt"},
    {"label": "Range Accrual",          "value": "range_accrual"},
    {"label": "Asian (Fix/Float)",      "value": "asian"},
    {"label": "Lookback",              "value": "lookback"},
    {"label": "Forward Start",          "value": "forward_start"},
    {"label": "Best-of / Worst-of",     "value": "best_of"},
    {"label": "TARF",                   "value": "tarf"},
]

BARRIER_TYPES = [
    {"label": "Down-and-In",  "value": "down-and-in"},
    {"label": "Down-and-Out", "value": "down-and-out"},
    {"label": "Up-and-In",    "value": "up-and-in"},
    {"label": "Up-and-Out",   "value": "up-and-out"},
]

FIXING_FREQS = [
    {"label": "Daily",   "value": "daily"},
    {"label": "Weekly",  "value": "weekly"},
    {"label": "Monthly", "value": "monthly"},
]

AVG_TYPES = [
    {"label": "Arithmetic", "value": "arithmetic"},
    {"label": "Geometric",  "value": "geometric"},
]

LOOKBACK_TYPES = [
    {"label": "Floating Strike", "value": "floating"},
    {"label": "Fixed Strike",    "value": "fixed"},
]

START_OFFSETS = [
    {"label": "3M", "value": "3M"},
    {"label": "6M", "value": "6M"},
]

MONEYNESS_OPTS = [
    {"label": "95%", "value": 0.95},
    {"label": "100% (ATM)", "value": 1.00},
    {"label": "105%", "value": 1.05},
]

TPL = CHART_TEMPLATE["layout"]

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

_SECTION_LABEL = {
    **LABEL_STYLE,
    "fontSize": "11px",
    "color": COLORS["accent_cyan"],
    "marginBottom": "10px",
    "marginTop": "12px",
}

_FIELD_MB = {"marginBottom": "10px"}

_PRICE_BTN = {
    **BUTTON_STYLE,
    "width": "100%",
    "marginTop": "16px",
    "padding": "14px 28px",
    "fontSize": "14px",
    "letterSpacing": "2px",
    "background": f"linear-gradient(135deg, {COLORS['accent_blue']}, {COLORS['accent_purple']})",
    "boxShadow": "0 6px 20px rgba(255,136,0,0.25)",
}


def _field(label, component):
    """Wrap a label + input together."""
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        component,
    ], style=_FIELD_MB)


def _inp(id_, value=None, step=None, placeholder=""):
    return dcc.Input(
        id=id_, type="number", value=value, step=step,
        placeholder=placeholder, debounce=True,
        style={**INPUT_STYLE, "width": "100%"},
    )


def _drop(id_, options, value=None, clearable=False):
    return dcc.Dropdown(
        id=id_, options=options, value=value,
        clearable=clearable,
        style={"fontSize": "12px"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout():
    pair_opts = [{"label": p, "value": p} for p in sorted(get_all_pairs())]
    default_pair = "EURUSD"

    return html.Div([
        dcc.Download(id="exo-csv-download"),
        # ── Top row: inputs | output ──────────────────────────────────────
        html.Div([
            # LEFT PANEL — product selector + inputs
            html.Div([
                html.Div("EXOTIC OPTIONS PRICING LAB", style=CARD_HEADER_STYLE),

                # Product selector
                html.Div("PRODUCT TYPE", style=_SECTION_LABEL),
                _drop("exo-product", PRODUCTS, "barrier"),

                # Common fields
                html.Div("MARKET DATA", style=_SECTION_LABEL),
                _field("PAIR", _drop("exo-pair", pair_opts, default_pair)),
                _field("NOTIONAL", _inp("exo-notional", 1_000_000, 100_000)),
                _field("TENOR", _drop("exo-tenor", [{"label": t, "value": t} for t in TENORS], "3M")),

                # Auto-filled market data (read-only display)
                html.Div(id="exo-mkt-display", style={
                    "padding": "8px 12px", "borderRadius": "0px",
                    "backgroundColor": COLORS["bg_secondary"],
                    "border": f"1px solid {COLORS['border_subtle']}",
                    "fontSize": "11px", "fontFamily": "monospace",
                    "color": COLORS["text_secondary"], "marginBottom": "10px",
                }),
                # Hidden stores for market data
                dcc.Store(id="exo-spot-store", data=None),
                dcc.Store(id="exo-rd-store", data=None),
                dcc.Store(id="exo-rf-store", data=None),
                dcc.Store(id="exo-vol-store", data=None),

                # ── Product-specific fields ───────────────────────────────
                html.Div("PRODUCT PARAMETERS", style=_SECTION_LABEL),

                # -- Call/Put (shared by many products) --
                html.Div(id="exo-cp-wrap", children=[
                    _field("CALL / PUT", _drop("exo-cp", [
                        {"label": "Call", "value": 1},
                        {"label": "Put", "value": -1},
                    ], 1)),
                ]),

                # -- Barrier-specific --
                html.Div(id="exo-barrier-wrap", children=[
                    _field("BARRIER LEVEL", _inp("exo-barrier-level", placeholder="e.g. 1.0500")),
                    _field("BARRIER TYPE", _drop("exo-barrier-type", BARRIER_TYPES, "down-and-out")),
                    _field("REBATE", _inp("exo-rebate", 0, 0.0001)),
                ]),

                # -- Double Barrier --
                html.Div(id="exo-dbl-barrier-wrap", children=[
                    _field("UPPER BARRIER", _inp("exo-upper-barrier", placeholder="e.g. 1.1200")),
                    _field("LOWER BARRIER", _inp("exo-lower-barrier", placeholder="e.g. 1.0200")),
                ]),

                # -- Strike (Digital, Asian, Lookback fixed, TARF) --
                html.Div(id="exo-strike-wrap", children=[
                    _field("STRIKE", _inp("exo-strike", placeholder="e.g. 1.0800")),
                ]),

                # -- Payout --
                html.Div(id="exo-payout-wrap", children=[
                    _field("PAYOUT AMOUNT", _inp("exo-payout", 1.0, 0.01)),
                ]),

                # -- Range Accrual --
                html.Div(id="exo-range-wrap", children=[
                    _field("LOWER RANGE", _inp("exo-range-low", placeholder="e.g. 1.0400")),
                    _field("UPPER RANGE", _inp("exo-range-high", placeholder="e.g. 1.1000")),
                ]),

                # -- Fixing frequency --
                html.Div(id="exo-fixfreq-wrap", children=[
                    _field("FIXING FREQUENCY", _drop("exo-fixfreq", FIXING_FREQS, "monthly")),
                ]),

                # -- Average type (Asian) --
                html.Div(id="exo-avgtype-wrap", children=[
                    _field("AVERAGE TYPE", _drop("exo-avgtype", AVG_TYPES, "arithmetic")),
                ]),

                # -- Lookback type --
                html.Div(id="exo-lbtype-wrap", children=[
                    _field("LOOKBACK TYPE", _drop("exo-lbtype", LOOKBACK_TYPES, "floating")),
                ]),

                # -- Forward Start --
                html.Div(id="exo-fwdstart-wrap", children=[
                    _field("START OFFSET", _drop("exo-fwd-offset", START_OFFSETS, "3M")),
                    _field("MONEYNESS", _drop("exo-fwd-money", MONEYNESS_OPTS, 1.00)),
                ]),

                # -- Best-of / Worst-of --
                html.Div(id="exo-bestof-wrap", children=[
                    _field("PAIR 2", _drop("exo-pair2", pair_opts, "USDJPY")),
                    _field("TYPE", _drop("exo-bestof-type", [
                        {"label": "Best-of", "value": "best-of"},
                        {"label": "Worst-of", "value": "worst-of"},
                    ], "best-of")),
                ]),

                # -- TARF --
                html.Div(id="exo-tarf-wrap", children=[
                    _field("TARF BARRIER", _inp("exo-tarf-barrier", placeholder="e.g. 1.1200")),
                    _field("TARGET PROFIT", _inp("exo-tarf-target", 0.05, 0.01)),
                    _field("LEVERAGE", _inp("exo-tarf-leverage", 2, 1)),
                    _field("NUM FIXINGS", _inp("exo-tarf-fixings", 12, 1)),
                ]),

                # PRICE button
                html.Button("PRICE", id="exo-price-btn", n_clicks=0, style=_PRICE_BTN),

            ], style={**CARD_STYLE, "width": "400px", "flexShrink": "0",
                      "overflowY": "auto", "maxHeight": "calc(100vh - 80px)"},
               className="dashboard-card"),

            # RIGHT PANEL — output
            html.Div([
                # Price output
                html.Div([
                    html.Div("PRICING RESULTS", style=CARD_HEADER_STYLE),
                    html.Div(id="exo-price-output", style={"minHeight": "60px"}),
                ], style=CARD_STYLE, className="dashboard-card"),

                # Greeks table
                html.Div([
                    html.Div("GREEKS", style=CARD_HEADER_STYLE),
                    html.Div(id="exo-greeks-output", style={"minHeight": "40px"}),
                ], style=CARD_STYLE, className="dashboard-card"),

                # Probabilities
                html.Div([
                    html.Div("KEY PROBABILITIES", style=CARD_HEADER_STYLE),
                    html.Div(id="exo-prob-output", style={"minHeight": "40px"}),
                ], style=CARD_STYLE, className="dashboard-card"),

                # Vanilla comparison
                html.Div([
                    html.Div("VANILLA COMPARISON", style=CARD_HEADER_STYLE),
                    html.Div(id="exo-vanilla-output", style={"minHeight": "30px"}),
                ], style=CARD_STYLE, className="dashboard-card"),

            ], style={"flex": "1", "minWidth": "0"}),

        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),

        # ── Charts 2x2 ───────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Button("CSV", id="exo-csv-payoff", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="exo-payoff-chart", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
            html.Div([
                html.Button("CSV", id="exo-csv-mc", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="exo-mc-chart", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "16px"}),

        html.Div([
            html.Div([
                html.Button("CSV", id="exo-csv-spot", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="exo-spot-sens-chart", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
            html.Div([
                html.Button("CSV", id="exo-csv-vol", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id="exo-vol-sens-chart", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "16px"}),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Visibility maps — which fields each product needs
# ═══════════════════════════════════════════════════════════════════════════

_VIS = {
    "barrier":       {"cp": True,  "barrier": True,  "dbl_barrier": False, "strike": True,
                      "payout": False, "range": False, "fixfreq": False, "avgtype": False,
                      "lbtype": False, "fwdstart": False, "bestof": False, "tarf": False},
    "double_barrier":{"cp": True,  "barrier": False, "dbl_barrier": True,  "strike": True,
                      "payout": False, "range": False, "fixfreq": False, "avgtype": False,
                      "lbtype": False, "fwdstart": False, "bestof": False, "tarf": False},
    "digital":       {"cp": True,  "barrier": False, "dbl_barrier": False, "strike": True,
                      "payout": True,  "range": False, "fixfreq": False, "avgtype": False,
                      "lbtype": False, "fwdstart": False, "bestof": False, "tarf": False},
    "one_touch":     {"cp": False, "barrier": True,  "dbl_barrier": False, "strike": False,
                      "payout": True,  "range": False, "fixfreq": False, "avgtype": False,
                      "lbtype": False, "fwdstart": False, "bestof": False, "tarf": False},
    "dnt":           {"cp": False, "barrier": False, "dbl_barrier": True,  "strike": False,
                      "payout": True,  "range": False, "fixfreq": False, "avgtype": False,
                      "lbtype": False, "fwdstart": False, "bestof": False, "tarf": False},
    "range_accrual": {"cp": False, "barrier": False, "dbl_barrier": False, "strike": False,
                      "payout": True,  "range": True,  "fixfreq": True,  "avgtype": False,
                      "lbtype": False, "fwdstart": False, "bestof": False, "tarf": False},
    "asian":         {"cp": True,  "barrier": False, "dbl_barrier": False, "strike": True,
                      "payout": False, "range": False, "fixfreq": True,  "avgtype": True,
                      "lbtype": False, "fwdstart": False, "bestof": False, "tarf": False},
    "lookback":      {"cp": True,  "barrier": False, "dbl_barrier": False, "strike": True,
                      "payout": False, "range": False, "fixfreq": False, "avgtype": False,
                      "lbtype": True,  "fwdstart": False, "bestof": False, "tarf": False},
    "forward_start": {"cp": True,  "barrier": False, "dbl_barrier": False, "strike": False,
                      "payout": False, "range": False, "fixfreq": False, "avgtype": False,
                      "lbtype": False, "fwdstart": True,  "bestof": False, "tarf": False},
    "best_of":       {"cp": True,  "barrier": False, "dbl_barrier": False, "strike": True,
                      "payout": False, "range": False, "fixfreq": False, "avgtype": False,
                      "lbtype": False, "fwdstart": False, "bestof": True,  "tarf": False},
    "tarf":          {"cp": False, "barrier": False, "dbl_barrier": False, "strike": True,
                      "payout": False, "range": False, "fixfreq": False, "avgtype": False,
                      "lbtype": False, "fwdstart": False, "bestof": False, "tarf": True},
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _empty_fig(title=""):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text_primary"], size=14)),
        paper_bgcolor=TPL["paper_bgcolor"], plot_bgcolor=TPL["plot_bgcolor"],
        font=TPL["font"], margin=dict(l=50, r=20, t=45, b=40),
        xaxis=dict(gridcolor="#1a1a30", visible=False),
        yaxis=dict(gridcolor="#1a1a30", visible=False),
        hoverlabel=TPL["hoverlabel"],
    )
    fig.add_annotation(text="Press PRICE to compute", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=COLORS["text_muted"], size=16))
    return fig


def _stat_box(label, value_str, color=COLORS["accent_cyan"]):
    return html.Div([
        html.Div(value_str, style={"color": color, "fontSize": "16px",
                                    "fontWeight": "700", "fontFamily": "monospace"}),
        html.Div(label, style={"color": COLORS["text_muted"], "fontSize": "10px",
                                "textTransform": "uppercase", "letterSpacing": "1px",
                                "marginTop": "4px"}),
    ], style={**make_stat_style(color), "flex": "1", "minWidth": "110px"})


def _get_mkt(pair, tenor):
    """Fetch spot, rates, ATM vol for a pair/tenor."""
    spots = get_fx_spots([pair]) or {}
    spot_data = spots.get(pair, {})
    if isinstance(spot_data, dict):
        spot = spot_data.get("mid", 1.0)
    else:
        spot = float(spot_data) if spot_data else 1.0
    rates = get_fx_rates(pair) or {}
    r_d = rates.get("r_dom", 0.05) if isinstance(rates, dict) else 0.05
    r_f = rates.get("r_for", 0.03) if isinstance(rates, dict) else 0.03
    vol_surf = get_fx_vol_surface(pair) or {}
    T = tenor_to_years(tenor)
    atm_vol_raw = 8.0
    if isinstance(vol_surf, dict):
        if tenor in vol_surf and isinstance(vol_surf[tenor], dict):
            atm_vol_raw = vol_surf[tenor].get("atm", 8.0)
        else:
            # Pick nearest available tenor from dict-valued entries
            available = [k for k in vol_surf.keys() if isinstance(vol_surf[k], dict)]
            if available:
                nearest = min(available, key=lambda t: abs(tenor_to_years(t) - T))
                atm_vol_raw = vol_surf[nearest].get("atm", 8.0)
    elif isinstance(vol_surf, (int, float)):
        atm_vol_raw = float(vol_surf)
    # Convert vol-points (e.g. 8.5) to decimal (0.085) for GK pricing
    atm_vol = atm_vol_raw / 100.0 if atm_vol_raw > 1.0 else atm_vol_raw
    atm_vol = max(atm_vol, 0.001)  # guard against zero/negative vol
    return spot, r_d, r_f, atm_vol, T


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app):

    # ── 1. Show/hide product-specific inputs ──────────────────────────────
    @app.callback(
        [Output("exo-cp-wrap", "style"),
         Output("exo-barrier-wrap", "style"),
         Output("exo-dbl-barrier-wrap", "style"),
         Output("exo-strike-wrap", "style"),
         Output("exo-payout-wrap", "style"),
         Output("exo-range-wrap", "style"),
         Output("exo-fixfreq-wrap", "style"),
         Output("exo-avgtype-wrap", "style"),
         Output("exo-lbtype-wrap", "style"),
         Output("exo-fwdstart-wrap", "style"),
         Output("exo-bestof-wrap", "style"),
         Output("exo-tarf-wrap", "style")],
        [Input("exo-product", "value")],
    )
    def toggle_inputs(product):
        vis = _VIS.get(product, _VIS["barrier"])
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if vis["cp"] else hide,
            show if vis["barrier"] else hide,
            show if vis["dbl_barrier"] else hide,
            show if vis["strike"] else hide,
            show if vis["payout"] else hide,
            show if vis["range"] else hide,
            show if vis["fixfreq"] else hide,
            show if vis["avgtype"] else hide,
            show if vis["lbtype"] else hide,
            show if vis["fwdstart"] else hide,
            show if vis["bestof"] else hide,
            show if vis["tarf"] else hide,
        )

    # ── 2. Auto-fill market data when pair/tenor changes ──────────────────
    @app.callback(
        [Output("exo-mkt-display", "children"),
         Output("exo-spot-store", "data"),
         Output("exo-rd-store", "data"),
         Output("exo-rf-store", "data"),
         Output("exo-vol-store", "data")],
        [Input("exo-pair", "value"), Input("exo-tenor", "value")],
    )
    def update_market_data(pair, tenor):
        pair = pair or "EURUSD"
        tenor = tenor or "3M"
        spot, r_d, r_f, vol, T = _get_mkt(pair, tenor)
        display = (f"Spot: {spot:.5f}  |  "
                   f"r_d: {r_d*100:.2f}%  |  r_f: {r_f*100:.2f}%  |  "
                   f"ATM Vol: {vol*100:.2f}%  |  T: {T:.4f}y")
        return display, spot, r_d, r_f, vol

    # ── 3. Main pricing callback ──────────────────────────────────────────
    @app.callback(
        [Output("exo-price-output", "children"),
         Output("exo-greeks-output", "children"),
         Output("exo-prob-output", "children"),
         Output("exo-vanilla-output", "children"),
         Output("exo-payoff-chart", "figure"),
         Output("exo-mc-chart", "figure"),
         Output("exo-spot-sens-chart", "figure"),
         Output("exo-vol-sens-chart", "figure")],
        [Input("exo-price-btn", "n_clicks")],
        [State("exo-product", "value"),
         State("exo-pair", "value"),
         State("exo-notional", "value"),
         State("exo-tenor", "value"),
         State("exo-spot-store", "data"),
         State("exo-rd-store", "data"),
         State("exo-rf-store", "data"),
         State("exo-vol-store", "data"),
         State("exo-cp", "value"),
         State("exo-barrier-level", "value"),
         State("exo-barrier-type", "value"),
         State("exo-rebate", "value"),
         State("exo-upper-barrier", "value"),
         State("exo-lower-barrier", "value"),
         State("exo-strike", "value"),
         State("exo-payout", "value"),
         State("exo-range-low", "value"),
         State("exo-range-high", "value"),
         State("exo-fixfreq", "value"),
         State("exo-avgtype", "value"),
         State("exo-lbtype", "value"),
         State("exo-fwd-offset", "value"),
         State("exo-fwd-money", "value"),
         State("exo-pair2", "value"),
         State("exo-bestof-type", "value"),
         State("exo-tarf-barrier", "value"),
         State("exo-tarf-target", "value"),
         State("exo-tarf-leverage", "value"),
         State("exo-tarf-fixings", "value")],
    )
    def run_pricer(n_clicks,
                   product, pair, notional, tenor,
                   spot, r_d, r_f, vol,
                   cp, barrier_level, barrier_type, rebate,
                   upper_barrier, lower_barrier,
                   strike, payout,
                   range_low, range_high,
                   fixfreq, avgtype,
                   lbtype, fwd_offset, fwd_money,
                   pair2, bestof_type,
                   tarf_barrier, tarf_target, tarf_leverage, tarf_fixings):

        if not n_clicks:
            empty = _empty_fig
            return (
                html.Div("Press PRICE to compute", style={"color": COLORS["text_muted"]}),
                html.Div("--", style={"color": COLORS["text_muted"]}),
                html.Div("--", style={"color": COLORS["text_muted"]}),
                html.Div("--", style={"color": COLORS["text_muted"]}),
                empty("Payoff Diagram"), empty("MC Paths"),
                empty("Price vs Spot"), empty("Price vs Vol"),
            )

        # Defaults
        pair = pair or "EURUSD"
        tenor = tenor or "3M"
        notional = _safe_float(notional, 1_000_000)
        cp = int(cp) if cp is not None else 1
        S = _safe_float(spot, 1.0)
        rd = _safe_float(r_d, 0.05)
        rf = _safe_float(r_f, 0.03)
        sigma = _safe_float(vol, 0.10)
        T = tenor_to_years(tenor)
        K = _safe_float(strike, S)
        B = _safe_float(barrier_level, S * (0.95 if "down" in str(barrier_type) else 1.05))
        B_up = _safe_float(upper_barrier, S * 1.05)
        B_dn = _safe_float(lower_barrier, S * 0.95)
        pay = _safe_float(payout, 1.0)
        reb = _safe_float(rebate, 0.0)
        r_low = _safe_float(range_low, S * 0.97)
        r_high = _safe_float(range_high, S * 1.03)
        fixfreq = fixfreq or "monthly"
        avgtype = avgtype or "arithmetic"
        lbtype = lbtype or "floating"
        fwd_offset = fwd_offset or "3M"
        fwd_money_val = _safe_float(fwd_money, 1.0)
        bestof_type = bestof_type or "best-of"
        pair2 = pair2 or "USDJPY"
        t_barrier = _safe_float(tarf_barrier, S * 1.05)
        t_target = _safe_float(tarf_target, 0.05)
        t_lev = int(_safe_float(tarf_leverage, 2))
        t_fix = int(_safe_float(tarf_fixings, 12))

        # Determine pip size for the pair
        pip_size = 0.0001
        if pair in FX_PAIR_REGISTRY:
            pip_size = FX_PAIR_REGISTRY[pair].pip

        price_val = 0.0
        price_result = None
        greeks_dict = {}
        probs = {}
        vanilla_px = 0.0
        extra_levels = {}  # barrier/range levels for charts
        n_mc_paths = 10000
        n_mc_steps = 252

        cp_label = "Call" if cp == 1 else "Put"

        try:
            # ── Compute price per product ──────────────────────────────
            if product == "barrier":
                bt = f"{barrier_type}-{cp_label.lower()}"
                price_val = barrier_price(S, K, B, T, rd, rf, sigma, cp, bt, reb)
                greeks_dict = exotic_greeks(
                    barrier_price,
                    dict(S=S, K=K, B=B, T=T, r_d=rd, r_f=rf, sigma=sigma,
                         cp=cp, barrier_type=bt, rebate=reb))
                vanilla_px = _gk_price(S, K, T, rd, rf, sigma, cp)
                # Probability of knock via MC
                paths, _ = _mc_paths(S, T, rd, rf, sigma, n_mc_paths, n_mc_steps, seed=42)
                if "down" in barrier_type:
                    p_hit = float(np.mean(paths.min(axis=1) <= B))
                else:
                    p_hit = float(np.mean(paths.max(axis=1) >= B))
                if "in" in barrier_type:
                    probs["P(Knock-In)"] = p_hit
                else:
                    probs["P(Knock-Out)"] = p_hit
                probs["P(ITM at expiry)"] = float(np.mean(
                    (cp * (paths[:, -1] - K)) > 0))
                extra_levels = {"barrier": B, "strike": K}

            elif product == "double_barrier":
                result = double_barrier_price(S, K, B_up, B_dn, T, rd, rf, sigma, cp,
                                              barrier_type="knock-out",
                                              n_paths=20000, n_steps=n_mc_steps, seed=42)
                price_val = result["price"]
                price_result = result
                greeks_dict = exotic_greeks(
                    lambda **kw: double_barrier_price(**kw),
                    dict(S=S, K=K, B_up=B_up, B_down=B_dn, T=T, r_d=rd, r_f=rf,
                         sigma=sigma, cp=cp, barrier_type="knock-out",
                         n_paths=5000, n_steps=100, seed=42))
                vanilla_px = _gk_price(S, K, T, rd, rf, sigma, cp)
                paths, _ = _mc_paths(S, T, rd, rf, sigma, n_mc_paths, n_mc_steps, seed=42)
                p_survive = float(np.mean(
                    (paths.min(axis=1) > B_dn) & (paths.max(axis=1) < B_up)))
                probs["P(Survive)"] = p_survive
                probs["P(Hit Upper)"] = float(np.mean(paths.max(axis=1) >= B_up))
                probs["P(Hit Lower)"] = float(np.mean(paths.min(axis=1) <= B_dn))
                extra_levels = {"upper": B_up, "lower": B_dn, "strike": K}

            elif product == "digital":
                price_val = digital_price(S, K, T, rd, rf, sigma, cp, pay)
                greeks_dict = digital_greeks(S, K, T, rd, rf, sigma, cp, pay)
                greeks_dict["price"] = price_val
                vanilla_px = _gk_price(S, K, T, rd, rf, sigma, cp)
                from scipy.stats import norm as ndist
                d1, d2 = ((np.log(S / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)),
                          (np.log(S / K) + (rd - rf - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)))
                probs["P(ITM)"] = float(ndist.cdf(cp * d2))
                extra_levels = {"strike": K}

            elif product == "one_touch":
                price_val = one_touch_price(S, B, T, rd, rf, sigma, pay)
                greeks_dict = exotic_greeks(
                    one_touch_price,
                    dict(S=S, B=B, T=T, r_d=rd, r_f=rf, sigma=sigma, payout=pay))
                vanilla_px = pay * np.exp(-rd * T)
                paths, _ = _mc_paths(S, T, rd, rf, sigma, n_mc_paths, n_mc_steps, seed=42)
                if B > S:
                    p_touch = float(np.mean(paths.max(axis=1) >= B))
                else:
                    p_touch = float(np.mean(paths.min(axis=1) <= B))
                probs["P(Touch)"] = p_touch
                probs["Implied Prob"] = price_val / (pay * np.exp(-rd * T)) if pay > 0 else 0.0
                extra_levels = {"barrier": B}

            elif product == "dnt":
                result = double_no_touch_price(S, B_up, B_dn, T, rd, rf, sigma, pay,
                                               n_paths=20000, n_steps=n_mc_steps, seed=42)
                price_val = result["price"]
                price_result = result
                greeks_dict = exotic_greeks(
                    lambda **kw: double_no_touch_price(**kw),
                    dict(S=S, B_up=B_up, B_down=B_dn, T=T, r_d=rd, r_f=rf,
                         sigma=sigma, payout=pay,
                         n_paths=5000, n_steps=100, seed=42))
                vanilla_px = pay * np.exp(-rd * T)
                probs["P(No Touch)"] = result.get("prob_no_touch", 0.0)
                probs["P(Touch Either)"] = 1.0 - probs["P(No Touch)"]
                extra_levels = {"upper": B_up, "lower": B_dn}

            elif product == "range_accrual":
                result = range_accrual_price(S, r_low, r_high, T, rd, rf, sigma, pay,
                                             fixfreq, n_paths=20000, seed=42)
                price_val = result["price"]
                price_result = result
                greeks_dict = exotic_greeks(
                    lambda **kw: range_accrual_price(**kw),
                    dict(S=S, B_low=r_low, B_high=r_high, T=T, r_d=rd, r_f=rf,
                         sigma=sigma, payout=pay, fixing_freq=fixfreq,
                         n_paths=5000, seed=42))
                vanilla_px = pay * np.exp(-rd * T)
                probs["E[Accrual Frac]"] = result.get("expected_accrual", 0.0)
                probs["N Fixings"] = result.get("n_fixings", 0)
                extra_levels = {"lower": r_low, "upper": r_high}

            elif product == "asian":
                result = asian_price(S, K, T, rd, rf, sigma, cp, fixfreq, avgtype,
                                     n_paths=20000, seed=42)
                price_val = result["price"]
                price_result = result
                greeks_dict = exotic_greeks(
                    lambda **kw: asian_price(**kw),
                    dict(S=S, K=K, T=T, r_d=rd, r_f=rf, sigma=sigma, cp=cp,
                         fixing_freq=fixfreq, average_type=avgtype,
                         n_paths=5000, seed=42))
                vanilla_px = _gk_price(S, K, T, rd, rf, sigma, cp)
                probs["N Fixings"] = result.get("n_fixings", 0)
                probs["Method"] = result.get("method", "mc")
                extra_levels = {"strike": K}

            elif product == "lookback":
                K_lb = K if lbtype == "fixed" else None
                result = lookback_price(S, T, rd, rf, sigma, cp, lbtype, K_lb,
                                        n_paths=20000, n_steps=n_mc_steps, seed=42)
                price_val = result["price"]
                price_result = result
                greeks_dict = exotic_greeks(
                    lambda **kw: lookback_price(**kw),
                    dict(S=S, T=T, r_d=rd, r_f=rf, sigma=sigma, cp=cp,
                         lookback_type=lbtype, K=K_lb,
                         n_paths=5000, n_steps=100, seed=42))
                vanilla_px = _gk_price(S, K if K_lb else S, T, rd, rf, sigma, cp)
                probs["Method"] = result.get("method", "mc")
                if K_lb:
                    extra_levels = {"strike": K_lb}

            elif product == "forward_start":
                T_start = tenor_to_years(fwd_offset)
                T_end = T
                if T_end <= T_start:
                    T_end = T_start + T
                price_val = forward_start_price(S, T_start, T_end, rd, rf, sigma, cp,
                                                fwd_money_val)
                greeks_dict = exotic_greeks(
                    lambda **kw: forward_start_price(**kw),
                    dict(S=S, T_start=T_start, T_end=T_end, r_d=rd, r_f=rf,
                         sigma=sigma, cp=cp, moneyness=fwd_money_val))
                K_equiv = fwd_money_val * S
                vanilla_px = _gk_price(S, K_equiv, T_end - T_start, rd, rf, sigma, cp)
                probs["T Start"] = f"{T_start:.3f}y"
                probs["T End"] = f"{T_end:.3f}y"
                probs["Equiv Strike"] = f"{K_equiv:.5f}"
                extra_levels = {"strike": K_equiv}

            elif product == "best_of":
                spot2_data = get_fx_spots([pair2]) or {}
                S2 = spot2_data.get(pair2, {}).get("mid", 1.0)
                rates2 = get_fx_rates(pair2) or {}
                rd2 = rates2.get("r_dom", 0.05)
                vol_surf2 = get_fx_vol_surface(pair2) or {}
                sigma2_raw = 8.0
                if isinstance(vol_surf2, dict):
                    if tenor in vol_surf2 and isinstance(vol_surf2[tenor], dict):
                        sigma2_raw = vol_surf2[tenor].get("atm", 8.0)
                    else:
                        avail2 = [k for k in vol_surf2.keys() if isinstance(vol_surf2[k], dict)]
                        if avail2:
                            near2 = min(avail2, key=lambda t: abs(tenor_to_years(t) - T))
                            sigma2_raw = vol_surf2[near2].get("atm", 8.0)
                elif isinstance(vol_surf2, (int, float)):
                    sigma2_raw = float(vol_surf2)
                # Convert vol-points (e.g. 8.5) to decimal (0.085) for GK pricing
                sigma2 = sigma2_raw / 100.0 if sigma2_raw > 1.0 else sigma2_raw
                # Look up realized correlation from bloomberg_fx data
                try:
                    corr_series = get_fx_correlation(pair, pair2, window=120, days=252)
                    rho = float(corr_series.iloc[-1]) if corr_series is not None and hasattr(corr_series, '__len__') and len(corr_series) > 0 else 0.5
                except Exception:
                    rho = 0.5
                K_perf = _safe_float(strike, 0.0)
                result = best_of_price(S, S2, K_perf, T, rd, rd2, rf, sigma, sigma2, rho, cp,
                                       bestof_type, n_paths=20000, seed=42)
                price_val = result["price"]
                price_result = result
                greeks_dict = {"price": price_val}  # complex multi-asset greeks
                vanilla_px = _gk_price(S, S * (1 + K_perf), T, rd, rf, sigma, cp)
                probs["Avg Perf 1"] = f"{result.get('avg_perf1', 0)*100:.2f}%"
                probs["Avg Perf 2"] = f"{result.get('avg_perf2', 0)*100:.2f}%"
                probs["Correlation"] = f"{rho:.2f}"

            elif product == "tarf":
                result = tarf_price(S, K, t_barrier, T, rd, rf, sigma,
                                    n_fixings=t_fix, target_profit=t_target,
                                    leverage=t_lev, n_paths=20000, seed=42)
                price_val = result["price"]
                price_result = result
                greeks_dict = exotic_greeks(
                    lambda **kw: tarf_price(**kw),
                    dict(S=S, K=K, B=t_barrier, T=T, r_d=rd, r_f=rf, sigma=sigma,
                         n_fixings=t_fix, target_profit=t_target, leverage=t_lev,
                         n_paths=5000, seed=42))
                vanilla_px = 0.0  # TARF is a structure, no direct vanilla equivalent
                probs["P(Early Term.)"] = result.get("prob_early_termination", 0.0)
                probs["E[Fixings]"] = result.get("expected_fixings", 0)
                probs["E[P&L]"] = result.get("expected_pnl", 0.0)
                extra_levels = {"strike": K, "barrier": t_barrier}

            else:
                price_val = 0.0
                probs["WARNING"] = f"Unknown product '{product}' -- no pricing model available"

        except Exception as exc:
            err_msg = f"Pricing error: {exc}"
            empty = _empty_fig
            return (
                html.Div(err_msg, style={"color": COLORS["accent_red"]}),
                html.Div("--", style={"color": COLORS["text_muted"]}),
                html.Div("--", style={"color": COLORS["text_muted"]}),
                html.Div("--", style={"color": COLORS["text_muted"]}),
                empty("Payoff Diagram"), empty("MC Paths"),
                empty("Price vs Spot"), empty("Price vs Vol"),
            )

        # ── Format outputs ────────────────────────────────────────────

        # Price output in multiple conventions
        base_ccy = pair[:3]
        quote_ccy = pair[3:6] if len(pair) >= 6 else "USD"
        price_ccy2 = price_val * notional
        price_pips = price_val / pip_size if pip_size > 0 else 0.0
        price_pct = (price_val / S * 100) if S > 0 else 0.0

        se_str = ""
        if price_result and isinstance(price_result, dict) and "std_error" in price_result:
            se_str = f"  (SE: {price_result['std_error']:.6f})"

        price_children = html.Div([
            html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}, children=[
                _stat_box(f"PRICE ({quote_ccy})", f"{price_val:.6f}{se_str}",
                          COLORS["accent_cyan"]),
                _stat_box(f"TOTAL ({quote_ccy})", f"{price_ccy2:,.2f}",
                          COLORS["accent_blue"]),
                _stat_box("PIPS", f"{price_pips:.2f}", COLORS["accent_purple"]),
                _stat_box("% NOTIONAL", f"{price_pct:.4f}%", COLORS["accent_teal"]),
            ]),
        ])

        # Greeks output
        g_delta = greeks_dict.get("delta", 0.0)
        g_gamma = greeks_dict.get("gamma", 0.0)
        g_vega = greeks_dict.get("vega", 0.0)
        g_theta = greeks_dict.get("theta", 0.0)

        greeks_children = html.Div([
            html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}, children=[
                _stat_box("DELTA", f"{g_delta:.6f}", COLORS["accent_cyan"]),
                _stat_box("GAMMA", f"{g_gamma:.6f}", COLORS["accent_blue"]),
                _stat_box("VEGA (1%)", f"{g_vega:.6f}", COLORS["accent_purple"]),
                _stat_box("THETA (/day)", f"{g_theta:.6f}", COLORS["accent_orange"]),
            ]),
        ])

        # Probabilities output
        prob_items = []
        for k, v in probs.items():
            if isinstance(v, float):
                if v < 1.5:  # likely a probability
                    display_val = f"{v*100:.2f}%"
                else:
                    display_val = f"{v:.4f}"
            else:
                display_val = str(v)
            box_color = COLORS["accent_red"] if k == "WARNING" else COLORS["accent_green"]
            prob_items.append(
                _stat_box(k, display_val, box_color))
        prob_children = html.Div(
            style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
            children=prob_items if prob_items else [
                html.Div("N/A", style={"color": COLORS["text_muted"]})],
        )

        # Vanilla comparison
        if vanilla_px > 0:
            exotic_prem = price_val / vanilla_px * 100 if vanilla_px != 0 else 0
            discount = (1 - price_val / vanilla_px) * 100 if vanilla_px != 0 else 0
            vanilla_children = html.Div([
                html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}, children=[
                    _stat_box("VANILLA PRICE", f"{vanilla_px:.6f}", COLORS["text_secondary"]),
                    _stat_box("EXOTIC / VANILLA",
                              f"{exotic_prem:.1f}%",
                              COLORS["accent_green"] if exotic_prem < 100 else COLORS["accent_red"]),
                    _stat_box("DISCOUNT",
                              f"{discount:.1f}%",
                              COLORS["accent_green"] if discount > 0 else COLORS["accent_red"]),
                ]),
            ])
        else:
            vanilla_children = html.Div(
                "No direct vanilla equivalent",
                style={"color": COLORS["text_muted"], "fontSize": "12px"})

        # ── Charts ────────────────────────────────────────────────────

        # 1. Payoff Diagram
        payoff_fig = _build_payoff_chart(product, S, T, rd, rf, sigma, cp, K,
                                          extra_levels, pay, notional, pip_size,
                                          lbtype, fwd_money_val, fwd_offset,
                                          r_low, r_high, t_barrier, t_target, t_lev)

        # 2. MC Paths
        mc_fig = _build_mc_chart(product, S, T, rd, rf, sigma, extra_levels)

        # 3. Price vs Spot sensitivity
        spot_fig = _build_spot_sensitivity(product, S, T, rd, rf, sigma, cp, K, B,
                                            B_up, B_dn, pay, reb, barrier_type,
                                            fixfreq, avgtype, lbtype, fwd_offset,
                                            fwd_money_val, r_low, r_high,
                                            t_barrier, t_target, t_lev, t_fix,
                                            pair2, bestof_type, pair, tenor)

        # 4. Price vs Vol sensitivity
        vol_fig = _build_vol_sensitivity(product, S, T, rd, rf, sigma, cp, K, B,
                                          B_up, B_dn, pay, reb, barrier_type,
                                          fixfreq, avgtype, lbtype, fwd_offset,
                                          fwd_money_val, r_low, r_high,
                                          t_barrier, t_target, t_lev, t_fix,
                                          pair2, bestof_type, pair, tenor)

        return (price_children, greeks_children, prob_children, vanilla_children,
                payoff_fig, mc_fig, spot_fig, vol_fig)

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output("exo-csv-download", "data"),
        [Input("exo-csv-payoff", "n_clicks"),
         Input("exo-csv-mc", "n_clicks"),
         Input("exo-csv-spot", "n_clicks"),
         Input("exo-csv-vol", "n_clicks")],
        [State("exo-payoff-chart", "figure"),
         State("exo-mc-chart", "figure"),
         State("exo-spot-sens-chart", "figure"),
         State("exo-vol-sens-chart", "figure")],
        prevent_initial_call=True,
    )
    def exo_csv_export(n1, n2, n3, n4, fig1, fig2, fig3, fig4):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        mapping = {
            "exo-csv-payoff": (fig1, "Exotics", "Payoff"),
            "exo-csv-mc": (fig2, "Exotics", "MonteCarlo"),
            "exo-csv-spot": (fig3, "Exotics", "SpotSens"),
            "exo-csv-vol": (fig4, "Exotics", "VolSens"),
        }
        if btn not in mapping:
            return no_update
        fig, panel, chart_type = mapping[btn]
        if not fig:
            return no_update
        return export_csv(fig, panel, chart_type)


# ═══════════════════════════════════════════════════════════════════════════
# Chart builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_payoff_chart(product, S, T, rd, rf, sigma, cp, K,
                        levels, payout, notional, pip_size,
                        lbtype, fwd_money, fwd_offset,
                        r_low, r_high, t_barrier, t_target, t_lev):
    """Build a payoff diagram adapted for each product type."""
    fig = go.Figure()
    spot_range = np.linspace(S * 0.85, S * 1.15, 300)

    if product in ("barrier", "double_barrier", "asian", "lookback", "forward_start"):
        # Standard option payoff shape
        payoff = np.maximum(cp * (spot_range - K), 0.0)
        fig.add_trace(go.Scatter(
            x=spot_range, y=payoff, mode="lines",
            name="Expiry Payoff",
            line=dict(color=COLORS["accent_cyan"], width=2.5),
            fill="tozeroy", fillcolor="rgba(255,136,0,0.06)",
            hovertemplate="Spot: %{x:.5f}<br>Payoff: %{y:.4f}<extra>Expiry</extra>"))

    elif product == "digital":
        payoff = np.where(cp * (spot_range - K) > 0, payout, 0.0)
        fig.add_trace(go.Scatter(
            x=spot_range, y=payoff, mode="lines",
            name="Digital Payoff", line=dict(color=COLORS["accent_cyan"], width=3, shape="hv"),
            hovertemplate="Spot: %{x:.5f}<br>Payoff: %{y:.4f}<extra>Digital</extra>"))

    elif product in ("one_touch", "dnt"):
        payoff = np.ones_like(spot_range) * payout
        fig.add_trace(go.Scatter(
            x=spot_range, y=payoff, mode="lines",
            name="Max Payout", line=dict(color=COLORS["accent_cyan"], width=2, dash="dash"),
            hovertemplate="Spot: %{x:.5f}<br>Payout: %{y:.4f}<extra>Max</extra>"))

    elif product == "range_accrual":
        in_range = (spot_range >= r_low) & (spot_range <= r_high)
        payoff = np.where(in_range, payout, 0.0)
        fig.add_trace(go.Scatter(
            x=spot_range, y=payoff, mode="lines",
            name="Accrual Zone", line=dict(color=COLORS["accent_cyan"], width=2.5),
            hovertemplate="Spot: %{x:.5f}<br>Payoff: %{y:.4f}<extra>Accrual</extra>"))
        fig.add_vrect(x0=r_low, x1=r_high, fillcolor="rgba(255,136,0,0.08)", line_width=0)

    elif product == "tarf":
        gain = np.maximum(spot_range / K - 1, 0.0)
        loss = -t_lev * np.maximum(1 - spot_range / K, 0.0)
        payoff = np.where(spot_range > K, gain, loss)
        fig.add_trace(go.Scatter(
            x=spot_range, y=payoff, mode="lines",
            name="Per-Fixing P&L",
            line=dict(color=COLORS["accent_cyan"], width=2.5),
            fill="tozeroy", fillcolor="rgba(255,136,0,0.06)",
            hovertemplate="Spot: %{x:.5f}<br>P&L: %{y:.4f}<extra>TARF</extra>"))

    else:
        payoff = np.maximum(cp * (spot_range - K), 0.0)
        fig.add_trace(go.Scatter(
            x=spot_range, y=payoff, mode="lines",
            name="Payoff", line=dict(color=COLORS["accent_cyan"], width=2.5),
            hovertemplate="Spot: %{x:.5f}<br>Payoff: %{y:.4f}<extra>Expiry</extra>"))

    # Overlay levels
    for lbl, val in levels.items():
        color = COLORS["accent_red"] if "barrier" in lbl else COLORS["accent_orange"]
        if "upper" in lbl:
            color = COLORS["accent_red"]
        elif "lower" in lbl:
            color = COLORS["accent_pink"]
        elif "strike" in lbl:
            color = COLORS["accent_orange"]
        fig.add_vline(x=val, line=dict(color=color, width=1.5, dash="dash"),
                      annotation_text=f"{lbl.title()}: {val:.5f}",
                      annotation_font=dict(color=color, size=10))

    # Spot marker
    fig.add_vline(x=S, line=dict(color=COLORS["accent_blue"], width=1, dash="dot"),
                  annotation_text=f"Spot: {S:.5f}",
                  annotation_font=dict(color=COLORS["accent_blue"], size=10))
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"))

    fig.update_layout(
        title=dict(text=f"Payoff Diagram -- {product.replace('_',' ').title()}",
                   font=dict(color=COLORS["text_primary"], size=14)),
        xaxis_title="Spot at Expiry", yaxis_title="Payoff",
        paper_bgcolor=TPL["paper_bgcolor"], plot_bgcolor=TPL["plot_bgcolor"],
        font=TPL["font"], margin=dict(l=50, r=20, t=45, b=40),
        xaxis=dict(gridcolor="#1a1a30"),
        yaxis=dict(gridcolor="#1a1a30"),
        legend=dict(font=dict(color=COLORS["text_secondary"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=TPL["hoverlabel"],
    )
    return fig


def _build_mc_chart(product, S, T, rd, rf, sigma, levels):
    """Show 20 sample MC paths with barrier/level annotations."""
    fig = go.Figure()
    n_show = 20
    n_steps = 252
    paths, dt = _mc_paths(S, T, rd, rf, sigma, n_show * 2, n_steps, seed=123)
    # Only take first n_show after antithetic
    paths = paths[:n_show]
    t_axis = np.linspace(0, T, n_steps + 1)

    # Determine barrier levels for coloring
    upper_lev = levels.get("upper", levels.get("barrier", None))
    lower_lev = levels.get("lower", None)
    if upper_lev is None and "barrier" in levels:
        barrier_val = levels["barrier"]
        if barrier_val > S:
            upper_lev = barrier_val
        else:
            lower_lev = barrier_val

    for i in range(n_show):
        path = paths[i]
        hit = False
        if upper_lev and np.max(path) >= upper_lev:
            hit = True
        if lower_lev and np.min(path) <= lower_lev:
            hit = True

        color = "rgba(255,51,51,0.35)" if hit else "rgba(255,136,0,0.25)"
        fig.add_trace(go.Scatter(
            x=t_axis, y=path, mode="lines",
            line=dict(width=1.2, color=color),
            showlegend=False, hoverinfo="skip"))

    # Percentile bands
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    fig.add_trace(go.Scatter(
        x=t_axis, y=p90, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=t_axis, y=p10, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(255,136,0,0.08)",
        name="10-90 %ile", hoverinfo="skip"))

    # Mean path
    mean_path = paths.mean(axis=0)
    fig.add_trace(go.Scatter(
        x=t_axis, y=mean_path, mode="lines",
        name="Mean Path", line=dict(color=COLORS["accent_cyan"], width=2.5),
        hovertemplate="T: %{x:.3f}y<br>Mean: %{y:.5f}<extra></extra>"))

    # Draw level lines
    for lbl, val in levels.items():
        if lbl == "strike":
            fig.add_hline(y=val, line=dict(color=COLORS["accent_orange"], width=1.5, dash="dash"),
                          annotation_text=f"K={val:.5f}",
                          annotation_font=dict(color=COLORS["accent_orange"], size=10))
        else:
            c = COLORS["accent_red"]
            fig.add_hline(y=val, line=dict(color=c, width=1.5, dash="dash"),
                          annotation_text=f"{lbl.title()}={val:.5f}",
                          annotation_font=dict(color=c, size=10))

    fig.update_layout(
        title=dict(text="Monte Carlo Paths (20 sample)",
                   font=dict(color=COLORS["text_primary"], size=14)),
        xaxis_title="Time (years)", yaxis_title="Spot",
        paper_bgcolor=TPL["paper_bgcolor"], plot_bgcolor=TPL["plot_bgcolor"],
        font=TPL["font"], margin=dict(l=50, r=20, t=45, b=40),
        xaxis=dict(gridcolor="#1a1a30"),
        yaxis=dict(gridcolor="#1a1a30"),
        legend=dict(font=dict(color=COLORS["text_secondary"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=TPL["hoverlabel"],
    )
    return fig


def _price_at_spot(product, s, T, rd, rf, sigma, cp, K, B,
                   B_up, B_dn, pay, reb, barrier_type,
                   fixfreq, avgtype, lbtype, fwd_offset,
                   fwd_money, r_low, r_high,
                   t_barrier, t_target, t_lev, t_fix,
                   pair2, bestof_type, pair, tenor):
    """Compute exotic price for a given spot level — used for sensitivity charts."""
    try:
        if product == "barrier":
            cp_label = "call" if cp == 1 else "put"
            bt = f"{barrier_type}-{cp_label}"
            return barrier_price(s, K, B, T, rd, rf, sigma, cp, bt, reb)
        elif product == "double_barrier":
            r = double_barrier_price(s, K, B_up, B_dn, T, rd, rf, sigma, cp,
                                     n_paths=5000, n_steps=100, seed=42)
            return r["price"]
        elif product == "digital":
            return digital_price(s, K, T, rd, rf, sigma, cp, pay)
        elif product == "one_touch":
            return one_touch_price(s, B, T, rd, rf, sigma, pay)
        elif product == "dnt":
            r = double_no_touch_price(s, B_up, B_dn, T, rd, rf, sigma, pay,
                                      n_paths=5000, n_steps=100, seed=42)
            return r["price"]
        elif product == "range_accrual":
            r = range_accrual_price(s, r_low, r_high, T, rd, rf, sigma, pay,
                                    fixfreq, n_paths=5000, seed=42)
            return r["price"]
        elif product == "asian":
            r = asian_price(s, K, T, rd, rf, sigma, cp, fixfreq, avgtype,
                            n_paths=5000, seed=42)
            return r["price"]
        elif product == "lookback":
            K_lb = K if lbtype == "fixed" else None
            r = lookback_price(s, T, rd, rf, sigma, cp, lbtype, K_lb,
                               n_paths=5000, n_steps=100, seed=42)
            return r["price"]
        elif product == "forward_start":
            T_start = tenor_to_years(fwd_offset)
            T_end = T if T > T_start else T_start + T
            return forward_start_price(s, T_start, T_end, rd, rf, sigma, cp, fwd_money)
        elif product == "best_of":
            # For spot sensitivity, vary S1 (primary pair) while keeping S2 fixed
            spot2_data = get_fx_spots([pair2]) or {}
            S2 = spot2_data.get(pair2, {}).get("mid", 1.0)
            rates2 = get_fx_rates(pair2) or {}
            rd2 = rates2.get("r_dom", 0.05)
            vol_surf2 = get_fx_vol_surface(pair2) or {}
            sigma2_raw = 8.0
            if vol_surf2 and tenor in vol_surf2:
                sigma2_raw = vol_surf2[tenor].get("atm", 8.0)
            elif vol_surf2:
                avail2 = list(vol_surf2.keys())
                if avail2:
                    near2 = min(avail2, key=lambda t: abs(tenor_to_years(t) - T))
                    sigma2_raw = vol_surf2[near2].get("atm", 8.0)
            sigma2 = sigma2_raw / 100.0 if sigma2_raw > 1.0 else sigma2_raw
            try:
                corr_series = get_fx_correlation(pair, pair2, window=120, days=252)
                rho = float(corr_series.iloc[-1]) if corr_series is not None and hasattr(corr_series, '__len__') and len(corr_series) > 0 else 0.5
            except Exception:
                rho = 0.5
            bo_type = bestof_type or "best-of"
            r = best_of_price(s, S2, K, T, rd, rd2, rf, sigma, sigma2, rho, cp,
                              bo_type, n_paths=5000, seed=42)
            return r["price"]
        elif product == "tarf":
            r = tarf_price(s, K, t_barrier, T, rd, rf, sigma,
                           n_fixings=t_fix, target_profit=t_target,
                           leverage=t_lev, n_paths=5000, seed=42)
            return r["price"]
        else:
            return 0.0
    except Exception:
        return 0.0


def _price_at_vol(product, S, T, rd, rf, v, cp, K, B,
                  B_up, B_dn, pay, reb, barrier_type,
                  fixfreq, avgtype, lbtype, fwd_offset,
                  fwd_money, r_low, r_high,
                  t_barrier, t_target, t_lev, t_fix,
                  pair2, bestof_type, pair, tenor):
    """Compute exotic price for a given vol level — used for vol sensitivity."""
    try:
        if product == "barrier":
            cp_label = "call" if cp == 1 else "put"
            bt = f"{barrier_type}-{cp_label}"
            return barrier_price(S, K, B, T, rd, rf, v, cp, bt, reb)
        elif product == "double_barrier":
            r = double_barrier_price(S, K, B_up, B_dn, T, rd, rf, v, cp,
                                     n_paths=5000, n_steps=100, seed=42)
            return r["price"]
        elif product == "digital":
            return digital_price(S, K, T, rd, rf, v, cp, pay)
        elif product == "one_touch":
            return one_touch_price(S, B, T, rd, rf, v, pay)
        elif product == "dnt":
            r = double_no_touch_price(S, B_up, B_dn, T, rd, rf, v, pay,
                                      n_paths=5000, n_steps=100, seed=42)
            return r["price"]
        elif product == "range_accrual":
            r = range_accrual_price(S, r_low, r_high, T, rd, rf, v, pay,
                                    fixfreq, n_paths=5000, seed=42)
            return r["price"]
        elif product == "asian":
            r = asian_price(S, K, T, rd, rf, v, cp, fixfreq, avgtype,
                            n_paths=5000, seed=42)
            return r["price"]
        elif product == "lookback":
            K_lb = K if lbtype == "fixed" else None
            r = lookback_price(S, T, rd, rf, v, cp, lbtype, K_lb,
                               n_paths=5000, n_steps=100, seed=42)
            return r["price"]
        elif product == "forward_start":
            T_start = tenor_to_years(fwd_offset)
            T_end = T if T > T_start else T_start + T
            return forward_start_price(S, T_start, T_end, rd, rf, v, cp, fwd_money)
        elif product == "best_of":
            # For vol sensitivity, vary sigma1 while keeping sigma2 proportionally scaled
            spot2_data = get_fx_spots([pair2]) or {}
            S2 = spot2_data.get(pair2, {}).get("mid", 1.0)
            rates2 = get_fx_rates(pair2) or {}
            rd2 = rates2.get("r_dom", 0.05)
            vol_surf2 = get_fx_vol_surface(pair2) or {}
            sigma2_raw = 8.0
            if vol_surf2 and tenor in vol_surf2:
                sigma2_raw = vol_surf2[tenor].get("atm", 8.0)
            elif vol_surf2:
                avail2 = list(vol_surf2.keys())
                if avail2:
                    near2 = min(avail2, key=lambda t: abs(tenor_to_years(t) - T))
                    sigma2_raw = vol_surf2[near2].get("atm", 8.0)
            sigma2 = sigma2_raw / 100.0 if sigma2_raw > 1.0 else sigma2_raw
            try:
                corr_series = get_fx_correlation(pair, pair2, window=120, days=252)
                rho = float(corr_series.iloc[-1]) if corr_series is not None and hasattr(corr_series, '__len__') and len(corr_series) > 0 else 0.5
            except Exception:
                rho = 0.5
            bo_type = bestof_type or "best-of"
            r = best_of_price(S, S2, K, T, rd, rd2, rf, v, sigma2, rho, cp,
                              bo_type, n_paths=5000, seed=42)
            return r["price"]
        elif product == "tarf":
            r = tarf_price(S, K, t_barrier, T, rd, rf, v,
                           n_fixings=t_fix, target_profit=t_target,
                           leverage=t_lev, n_paths=5000, seed=42)
            return r["price"]
        else:
            return 0.0
    except Exception:
        return 0.0


def _build_spot_sensitivity(product, S, T, rd, rf, sigma, cp, K, B,
                            B_up, B_dn, pay, reb, barrier_type,
                            fixfreq, avgtype, lbtype, fwd_offset,
                            fwd_money, r_low, r_high,
                            t_barrier, t_target, t_lev, t_fix,
                            pair2, bestof_type, pair, tenor):
    """Price vs Spot chart at +/-10% of current spot."""
    fig = go.Figure()
    spot_grid = np.linspace(S * 0.90, S * 1.10, 40)
    prices = []
    for s in spot_grid:
        px = _price_at_spot(product, s, T, rd, rf, sigma, cp, K, B,
                            B_up, B_dn, pay, reb, barrier_type,
                            fixfreq, avgtype, lbtype, fwd_offset,
                            fwd_money, r_low, r_high,
                            t_barrier, t_target, t_lev, t_fix,
                            pair2, bestof_type, pair, tenor)
        prices.append(px)

    fig.add_trace(go.Scatter(
        x=spot_grid, y=prices, mode="lines+markers",
        name="Exotic Price",
        line=dict(color=COLORS["accent_cyan"], width=2.5),
        marker=dict(size=3),
        hovertemplate="Spot: %{x:.5f}<br>Price: %{y:.6f}<extra></extra>"))

    fig.add_vline(x=S, line=dict(color=COLORS["accent_blue"], width=1, dash="dash"),
                  annotation_text=f"Spot={S:.5f}",
                  annotation_font=dict(color=COLORS["accent_blue"], size=10))

    fig.update_layout(
        title=dict(text="Sensitivity: Price vs Spot",
                   font=dict(color=COLORS["text_primary"], size=14)),
        xaxis_title="Spot Level", yaxis_title="Option Price",
        paper_bgcolor=TPL["paper_bgcolor"], plot_bgcolor=TPL["plot_bgcolor"],
        font=TPL["font"], margin=dict(l=50, r=20, t=45, b=40),
        xaxis=dict(gridcolor="#1a1a30"),
        yaxis=dict(gridcolor="#1a1a30"),
        legend=dict(font=dict(color=COLORS["text_secondary"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=TPL["hoverlabel"],
    )
    return fig


def _build_vol_sensitivity(product, S, T, rd, rf, sigma, cp, K, B,
                           B_up, B_dn, pay, reb, barrier_type,
                           fixfreq, avgtype, lbtype, fwd_offset,
                           fwd_money, r_low, r_high,
                           t_barrier, t_target, t_lev, t_fix,
                           pair2, bestof_type, pair, tenor):
    """Price vs Vol chart at +/-50% of current vol."""
    fig = go.Figure()
    vol_lo = max(sigma * 0.5, 0.01)
    vol_hi = sigma * 1.5
    vol_grid = np.linspace(vol_lo, vol_hi, 40)
    prices = []
    for v in vol_grid:
        px = _price_at_vol(product, S, T, rd, rf, v, cp, K, B,
                           B_up, B_dn, pay, reb, barrier_type,
                           fixfreq, avgtype, lbtype, fwd_offset,
                           fwd_money, r_low, r_high,
                           t_barrier, t_target, t_lev, t_fix,
                           pair2, bestof_type, pair, tenor)
        prices.append(px)

    fig.add_trace(go.Scatter(
        x=[v * 100 for v in vol_grid], y=prices, mode="lines+markers",
        name="Exotic Price",
        line=dict(color=COLORS["accent_purple"], width=2.5),
        marker=dict(size=3),
        hovertemplate="Vol: %{x:.1f}%<br>Price: %{y:.6f}<extra></extra>"))

    fig.add_vline(x=sigma * 100,
                  line=dict(color=COLORS["accent_blue"], width=1, dash="dash"),
                  annotation_text=f"ATM Vol={sigma*100:.1f}%",
                  annotation_font=dict(color=COLORS["accent_blue"], size=10))

    fig.update_layout(
        title=dict(text="Sensitivity: Price vs Volatility",
                   font=dict(color=COLORS["text_primary"], size=14)),
        xaxis_title="Implied Vol (%)", yaxis_title="Option Price",
        paper_bgcolor=TPL["paper_bgcolor"], plot_bgcolor=TPL["plot_bgcolor"],
        font=TPL["font"], margin=dict(l=50, r=20, t=45, b=40),
        xaxis=dict(gridcolor="#1a1a30"),
        yaxis=dict(gridcolor="#1a1a30"),
        legend=dict(font=dict(color=COLORS["text_secondary"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=TPL["hoverlabel"],
    )
    return fig
