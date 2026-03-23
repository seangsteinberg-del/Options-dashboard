#!/usr/bin/env python3
"""
FX Options Workstation v5 — Consolidated
=========================================
Institutional-grade FX options analytics platform.
10 panels across 4 workspaces. Zero duplication.

Run:  python app.py
Open: http://localhost:8050
"""

# ── Auto-install dependencies ─────────────────────────────────────────────
import subprocess, sys

def _ensure_packages():
    required = {
        "dash": "dash>=2.14.0",
        "dash_bootstrap_components": "dash-bootstrap-components>=1.5.0",
        "plotly": "plotly>=5.18.0",
        "numpy": "numpy>=1.24.0",
        "scipy": "scipy>=1.11.0",
        "pandas": "pandas>=2.0.0",
        "webview": "pywebview>=4.0",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  Installing missing packages: {', '.join(missing)}\n")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("  Done.\n")

    # Bloomberg API — try to install if missing, but don't fail if it can't
    try:
        __import__("blpapi")
    except ImportError:
        print("  blpapi not found — attempting install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "blpapi"],
                                  stderr=subprocess.DEVNULL)
            print("  [OK] blpapi installed.")
        except Exception:
            print("  [!] blpapi install failed — running with synthetic data.")
            print("      If Bloomberg Terminal is on this machine, you may need to:")
            print("      1. Install Bloomberg C++ SDK (WAPI<GO> on terminal)")
            print("      2. Set BLPAPI_ROOT environment variable")
            print("      3. Then: pip install blpapi\n")

_ensure_packages()

# ── Imports ───────────────────────────────────────────────────────────────
import json
import time
import logging
import numpy as np

import dash
from dash import html, dcc, Input, Output, State, callback_context, ALL, MATCH
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from core.theme import COLORS, TAB_STYLE, TAB_SELECTED_STYLE, status_color, status_text, CHART_TEMPLATE
from core.bloomberg import is_connected

logger = logging.getLogger(__name__)

# ── 10 Consolidated Panels ───────────────────────────────────────────────
from panels import market_dashboard          # DESK
from panels import vol_surface_fx            # VOL
from panels import vol_scanner_unified       # VOL
from panels import structure_builder         # TRADE
from panels import exotics_pricer            # TRADE
from panels import blotter_fx               # TRADE
from panels import risk_fx                   # RISK & ANALYTICS
from panels import relative_value_plus      # RISK & ANALYTICS
from panels import chart_lab                 # RISK & ANALYTICS
from panels import backtest                  # RISK & ANALYTICS


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

VERSION = "5.0"

# Major FX pairs for the ticker tape
FX_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "NZDUSD",
    "USDCAD", "EURGBP", "EURJPY", "GBPJPY", "USDMXN", "USDZAR",
]

# All 30 pairs for watchlist
ALL_FX_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "EURAUD", "EURNZD",
    "NZDJPY", "AUDNZD", "CADCHF", "CADJPY",
    "EURNOK", "EURSEK", "USDSEK", "USDNOK",
    "USDMXN", "USDBRL", "USDTRY", "USDZAR", "USDCNH", "USDINR", "USDSGD", "USDKRW",
]

# Default watchlist
DEFAULT_WATCHLIST = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "EURGBP", "EURJPY", "USDMXN", "USDZAR", "USDCNH",
]

# Synthetic mid-market rates (used when Bloomberg is not connected)
_FX_SEED_RATES = {
    "EURUSD": 1.0842, "USDJPY": 149.72, "GBPUSD": 1.2715,
    "USDCHF": 0.8794, "AUDUSD": 0.6538, "NZDUSD": 0.6127,
    "USDCAD": 1.3562, "EURGBP": 0.8527, "EURJPY": 162.34,
    "GBPJPY": 190.31, "USDMXN": 17.142, "USDZAR": 18.695,
}

# ── Workspace Definition (4 workspaces, 10 panels) ──────────────────────
WORKSPACES = [
    {
        "id": "desk",
        "label": "DESK",
        "accent": COLORS["accent_orange"],
        "tabs": [
            {"id": "market-dashboard", "label": "DASHBOARD", "module": market_dashboard},
        ],
    },
    {
        "id": "vol",
        "label": "VOL",
        "accent": COLORS["accent_orange"],
        "tabs": [
            {"id": "vol-surface-fx",    "label": "VOL SURFACE",  "module": vol_surface_fx},
            {"id": "vol-scanner-unified","label": "VOL SCANNER",  "module": vol_scanner_unified},
        ],
    },
    {
        "id": "trade",
        "label": "TRADE",
        "accent": COLORS["accent_orange"],
        "tabs": [
            {"id": "structure-builder", "label": "STRUCTURE BUILDER", "module": structure_builder},
            {"id": "exotics-pricer",    "label": "EXOTICS PRICER",    "module": exotics_pricer},
            {"id": "blotter-fx",        "label": "BLOTTER",           "module": blotter_fx},
        ],
    },
    {
        "id": "risk-analytics",
        "label": "RISK & ANALYTICS",
        "accent": COLORS["accent_red"],
        "tabs": [
            {"id": "risk-fx",             "label": "RISK DASHBOARD",   "module": risk_fx},
            {"id": "relative-value-plus", "label": "RELATIVE VALUE",   "module": relative_value_plus},
            {"id": "chart-lab",           "label": "CHART LAB",         "module": chart_lab},
            {"id": "backtest",            "label": "BACKTEST",           "module": backtest},
        ],
    },
]

# Flat lookup: tab_id -> module (for routing)
_TAB_MODULE_MAP = {}
for ws in WORKSPACES:
    for tab in ws["tabs"]:
        _TAB_MODULE_MAP[tab["id"]] = tab["module"]

# Workspace presets
WORKSPACE_PRESETS = {
    "Desk":  ("desk",           "market-dashboard"),
    "Vol":   ("vol",            "vol-surface-fx"),
    "Trade": ("trade",          "structure-builder"),
    "Risk":  ("risk-analytics", "risk-fx"),
}

# Command palette search items
COMMAND_ITEMS = []
for ws in WORKSPACES:
    for tab in ws["tabs"]:
        COMMAND_ITEMS.append({
            "label": f"{tab['label']}  [{ws['label']}]",
            "workspace": ws["id"],
            "tab": tab["id"],
        })
for pair in ALL_FX_PAIRS:
    COMMAND_ITEMS.append({
        "label": f"{pair}  [SET PAIR]",
        "workspace": "__pair__",
        "tab": pair,
    })


# ═══════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="FX Options Workstation",
    update_title="Calculating...",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
server = app.server


# ═══════════════════════════════════════════════════════════════════════════
# FX Ticker Tape
# ═══════════════════════════════════════════════════════════════════════════

def _fx_spot_data():
    """Generate synthetic FX spot data with small random perturbation."""
    rng = np.random.RandomState(int(time.time()))
    data = {}
    for pair in FX_PAIRS:
        base = _FX_SEED_RATES.get(pair, 1.0)
        jitter = base * rng.uniform(-0.004, 0.004)
        price = base + jitter
        change_pct = (jitter / base) * 100.0
        is_jpy = "JPY" in pair
        is_em = pair in ("USDMXN", "USDZAR")
        if is_jpy:
            fmt = f"{price:.2f}"
        elif is_em:
            fmt = f"{price:.3f}"
        else:
            fmt = f"{price:.4f}"
        data[pair] = {"price_str": fmt, "price": price, "change_pct": change_pct}
    return data


def _build_ticker_items():
    """Build ticker tape span elements from live spot data."""
    spot = _fx_spot_data()
    items = []
    for pair in FX_PAIRS:
        info = spot[pair]
        chg = info["change_pct"]
        color = COLORS["accent_green"] if chg >= 0 else COLORS["accent_red"]
        arrow = "\u25B2" if chg >= 0 else "\u25BC"

        items.append(html.Span([
            html.Span(pair, style={
                "color": COLORS["text_primary"], "fontWeight": "700",
                "marginRight": "6px",
            }),
            html.Span(info["price_str"], style={
                "color": COLORS["text_secondary"], "marginRight": "5px",
            }),
            html.Span(f"{arrow} {chg:+.2f}%", style={
                "color": color, "fontWeight": "600",
            }),
        ], style={
            "marginRight": "36px", "fontSize": "11px",
            "display": "inline-block",
            "fontFamily": "'JetBrains Mono', monospace",
        }))
    return items


def make_ticker_tape():
    """Scrolling FX ticker tape across the top of the workstation (auto-refreshes)."""
    items = _build_ticker_items()
    return html.Div([
        html.Div(items + items, id="ticker-tape-content",
                 className="ticker-tape-inner",
                 style={"display": "inline-flex", "whiteSpace": "nowrap"}),
        dcc.Interval(id="ticker-refresh-interval", interval=30_000, n_intervals=0),
    ], className="ticker-tape")


# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════

def _preset_button(label, idx):
    """Render a single workspace-preset button."""
    return html.Button(
        label,
        id={"type": "preset-btn", "index": idx},
        n_clicks=0,
        style={
            "backgroundColor": "#000000",
            "color": "#808080",
            "border": "1px solid #222240",
            "borderRadius": "0px",
            "padding": "4px 10px",
            "fontSize": "9px",
            "fontFamily": "'JetBrains Mono', monospace",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "textTransform": "uppercase",
            "cursor": "pointer",
            "marginLeft": "4px",
        },
    )


def make_header():
    """Full header: ticker tape, logo, status badges, preset buttons, watchlist editor."""
    bbg = is_connected()
    dot_color = COLORS["accent_green"] if bbg else COLORS["accent_orange"]
    badge_class = "bbg-badge connected" if bbg else "bbg-badge disconnected"
    badge_text = "BLOOMBERG LIVE" if bbg else "SYNTHETIC MODE"

    preset_buttons = [
        _preset_button(name, i) for i, name in enumerate(WORKSPACE_PRESETS.keys())
    ]

    return html.Div([
        make_ticker_tape(),

        html.Div([
            # ── Logo ──
            html.Div([
                html.Div([
                    html.Span("FX OPTIONS", style={
                        "fontWeight": "800", "fontSize": "18px",
                        "letterSpacing": "3px",
                        "color": "#ff8800",
                    }),
                    html.Span(" WORKSTATION", style={
                        "fontWeight": "300", "color": "#808080",
                        "fontSize": "18px", "letterSpacing": "3px",
                    }),
                ]),
                html.Div("Institutional Derivatives Analytics", style={
                    "color": "#808080", "fontSize": "9px",
                    "textTransform": "uppercase", "letterSpacing": "4px",
                    "marginTop": "2px",
                }),
            ], style={"flex": "1"}),

            # ── Workspace Presets ──
            html.Div([
                html.Span("WORKSPACE", style={
                    "color": "#808080", "fontSize": "9px",
                    "letterSpacing": "1px", "marginRight": "6px",
                    "fontWeight": "600",
                }),
                *preset_buttons,
            ], style={
                "display": "flex", "alignItems": "center",
                "marginRight": "16px",
            }),

            # ── Watchlist Editor Button ──
            html.Div([
                html.Button("WATCHLIST", id="watchlist-edit-btn", n_clicks=0, style={
                    "backgroundColor": "#000000",
                    "color": "#ff8800",
                    "border": "1px solid #222240",
                    "borderRadius": "0px",
                    "padding": "4px 10px",
                    "fontSize": "9px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontWeight": "700",
                    "letterSpacing": "1px",
                    "cursor": "pointer",
                }),
            ], style={"marginRight": "16px"}),

            # ── Status Block ──
            html.Div([
                html.Div([
                    html.Div(className="pulse-dot", style={
                        "width": "6px", "height": "6px", "borderRadius": "50%",
                        "backgroundColor": dot_color, "display": "inline-block",
                        "marginRight": "6px",
                    }),
                    html.Span(badge_text, className=badge_class),
                ], style={
                    "display": "flex", "alignItems": "center",
                    "marginRight": "16px",
                }),
                # ── Data integrity status (LIVE / DEGRADED / SYNTHETIC) ──
                html.Div(id="data-source-status", children=[], style={
                    "display": "flex", "alignItems": "center",
                    "marginRight": "16px",
                }),
                html.Div([
                    html.Span("MODEL ", style={
                        "color": "#808080", "fontSize": "9px",
                        "letterSpacing": "1px",
                    }),
                    html.Span("GK/SABR/VV/MC", style={
                        "color": "#ff8800", "fontSize": "9px",
                        "fontWeight": "700",
                    }),
                ]),
                # ── Last Updated Timestamp ──
                html.Div([
                    html.Span("UPDATED ", style={
                        "color": "#808080", "fontSize": "9px",
                        "letterSpacing": "1px",
                    }),
                    html.Span(id="header-last-updated", children="—", style={
                        "color": "#d4d4d4", "fontSize": "9px",
                        "fontWeight": "600",
                    }),
                ], style={"marginLeft": "16px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "padding": "10px 24px",
            "backgroundColor": "#000000",
            "borderBottom": "1px solid #222240",
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Watchlist Editor Modal
# ═══════════════════════════════════════════════════════════════════════════

def make_watchlist_modal():
    """Watchlist editor modal — multi-select pairs."""
    return html.Div(
        id="watchlist-modal-overlay",
        children=[
            html.Div([
                html.Div("WATCHLIST EDITOR", style={
                    "color": "#ffffff", "fontSize": "11px", "fontWeight": "700",
                    "letterSpacing": "1.5px", "marginBottom": "8px",
                    "paddingBottom": "4px", "borderBottom": "1px solid #222240",
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
                dcc.Dropdown(
                    id="watchlist-editor-dropdown",
                    options=[{"label": p, "value": p} for p in ALL_FX_PAIRS],
                    value=DEFAULT_WATCHLIST,
                    multi=True,
                    placeholder="Select pairs...",
                    style={"fontSize": "11px", "fontFamily": "'JetBrains Mono', monospace"},
                ),
                html.Div([
                    html.Button("SAVE", id="watchlist-save-btn", n_clicks=0, style={
                        "backgroundColor": "#ff8800", "color": "#000000",
                        "border": "none", "borderRadius": "0px",
                        "padding": "6px 16px", "fontSize": "10px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontWeight": "700", "cursor": "pointer", "marginRight": "8px",
                    }),
                    html.Button("CLOSE", id="watchlist-close-btn", n_clicks=0, style={
                        "backgroundColor": "#000000", "color": "#808080",
                        "border": "1px solid #222240", "borderRadius": "0px",
                        "padding": "6px 16px", "fontSize": "10px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontWeight": "700", "cursor": "pointer",
                    }),
                ], style={"marginTop": "12px", "display": "flex"}),
            ], style={
                "backgroundColor": "#000000",
                "border": "1px solid #222240",
                "padding": "16px",
                "width": "500px",
                "maxWidth": "90vw",
            }),
        ],
        style={
            "display": "none",
            "position": "fixed",
            "top": "0", "left": "0", "right": "0", "bottom": "0",
            "backgroundColor": "rgba(0,0,0,0.85)",
            "zIndex": "9000",
            "justifyContent": "center",
            "alignItems": "center",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Universal Metric Popup Modal
# ═══════════════════════════════════════════════════════════════════════════

def make_metric_popup():
    """Universal metric popup — click any number across the entire app."""
    return html.Div(
        id="metric-popup-overlay",
        children=[
            html.Div([
                # Header
                html.Div([
                    html.Span(id="metric-popup-title", style={
                        "color": "#ffffff", "fontSize": "13px", "fontWeight": "700",
                        "letterSpacing": "1px", "fontFamily": "'JetBrains Mono', monospace",
                    }),
                    html.Button("\u2715", id="metric-popup-close-btn", n_clicks=0,
                                className="metric-popup-close"),
                ], className="metric-popup-header"),

                # Stats row
                html.Div(id="metric-popup-stats", className="metric-popup-stats"),

                # Chart
                dcc.Graph(id="metric-popup-chart", config={"displayModeBar": False},
                          style={"height": "280px"}),

                # Action buttons
                html.Div([
                    html.Button("SEND TO LAB", id="metric-popup-lab-btn", n_clicks=0,
                                className="metric-popup-btn primary"),
                    html.Button("COMPARE", id="metric-popup-compare-btn", n_clicks=0,
                                className="metric-popup-btn"),
                ], className="metric-popup-actions"),
            ], className="metric-popup"),
        ],
        className="metric-popup-overlay",
        style={"display": "none"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Command Palette (Ctrl+K overlay)
# ═══════════════════════════════════════════════════════════════════════════

def make_command_palette():
    """Hidden modal overlay revealed by Ctrl+K."""
    search_options = [
        {"label": item["label"], "value": json.dumps(item)}
        for item in COMMAND_ITEMS
    ]

    return html.Div(
        id="command-palette-overlay",
        children=[
            html.Div([
                html.Div([
                    html.Span("COMMAND PALETTE", style={
                        "color": "#808080", "fontSize": "9px",
                        "letterSpacing": "2px", "fontWeight": "700",
                    }),
                    html.Span("Ctrl+K", style={
                        "color": "#ff8800", "fontSize": "9px",
                        "fontWeight": "600", "marginLeft": "12px",
                        "padding": "2px 8px",
                        "border": "1px solid #222240",
                        "borderRadius": "0px",
                    }),
                ], style={
                    "display": "flex", "justifyContent": "space-between",
                    "alignItems": "center", "marginBottom": "12px",
                }),

                dcc.Dropdown(
                    id="command-palette-input",
                    options=search_options,
                    placeholder="Search panels, pairs, metrics ...",
                    searchable=True,
                    clearable=True,
                    style={
                        "backgroundColor": "#000000",
                        "color": "#d4d4d4",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "13px",
                        "border": "none",
                    },
                ),
            ], style={
                "backgroundColor": "#000000",
                "border": "1px solid #222240",
                "padding": "16px 20px",
                "width": "560px",
                "maxWidth": "90vw",
            }),
        ],
        style={
            "display": "none",
            "position": "fixed",
            "top": "0", "left": "0", "right": "0", "bottom": "0",
            "backgroundColor": "rgba(0,0,0,0.85)",
            "zIndex": "9999",
            "justifyContent": "center",
            "alignItems": "flex-start",
            "paddingTop": "18vh",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Two-Tier Tab System
# ═══════════════════════════════════════════════════════════════════════════

def _workspace_tab_style(accent):
    return {
        "backgroundColor": "transparent",
        "border": "1px solid #222240",
        "borderBottom": "none",
        "borderRadius": "0px",
        "color": "#808080",
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "10px",
        "fontWeight": "700",
        "padding": "8px 14px",
        "letterSpacing": "1.5px",
        "textTransform": "uppercase",
        "cursor": "pointer",
    }


def _workspace_tab_selected_style(accent):
    base = _workspace_tab_style(accent)
    return {
        **base,
        "backgroundColor": "#000000",
        "color": accent,
        "borderTop": f"2px solid {accent}",
    }


def _subtab_style():
    return {
        "backgroundColor": "transparent",
        "border": "1px solid #222240",
        "borderBottom": "none",
        "borderRadius": "0px",
        "color": "#808080",
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "9px",
        "fontWeight": "600",
        "padding": "6px 12px",
        "letterSpacing": "1.2px",
        "textTransform": "uppercase",
        "cursor": "pointer",
    }


def _subtab_selected_style(accent):
    base = _subtab_style()
    return {
        **base,
        "backgroundColor": "#000000",
        "color": accent,
        "borderTop": f"2px solid {accent}",
    }


def make_workspace_tabs():
    children = []
    for ws in WORKSPACES:
        children.append(
            dcc.Tab(
                label=ws["label"],
                value=ws["id"],
                style=_workspace_tab_style(ws["accent"]),
                selected_style=_workspace_tab_selected_style(ws["accent"]),
            )
        )
    return dcc.Tabs(
        id="workspace-tabs",
        value=WORKSPACES[0]["id"],
        children=children,
        style={"marginBottom": "0"},
    )


def make_sub_tabs_container():
    """Pre-render initial sub-tabs so 'sub-tabs' ID exists from page load."""
    ws = WORKSPACES[0]
    accent = ws["accent"]
    initial_tabs = dcc.Tabs(
        id="sub-tabs",
        value=ws["tabs"][0]["id"],
        children=[
            dcc.Tab(
                label=tab["label"],
                value=tab["id"],
                style=_subtab_style(),
                selected_style=_subtab_selected_style(accent),
            )
            for tab in ws["tabs"]
        ],
        style={"marginTop": "2px", "borderBottom": "1px solid #222240"},
    )
    return html.Div(initial_tabs, id="subtab-container", style={"marginTop": "0"})


# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════

def make_footer():
    data_label = "BLOOMBERG API" if is_connected() else "SYNTHETIC DATA"
    total_panels = sum(len(ws["tabs"]) for ws in WORKSPACES)
    return html.Div([
        html.Div([
            html.Span(f"FX OPTIONS WORKSTATION v{VERSION}", style={
                "color": "#808080", "fontSize": "9px",
                "letterSpacing": "2px",
            }),
            html.Span(" \u2502 ", style={
                "color": "#222240", "margin": "0 8px",
            }),
            html.Span(f"{total_panels} PANELS", style={
                "color": "#808080", "fontSize": "9px",
                "letterSpacing": "2px",
            }),
            html.Span(" \u2502 ", style={
                "color": "#222240", "margin": "0 8px",
            }),
            html.Span(data_label, style={
                "color": status_color(), "fontSize": "9px",
                "letterSpacing": "2px", "fontWeight": "700",
            }),
            html.Span(" \u2502 ", style={
                "color": "#222240", "margin": "0 8px",
            }),
            html.Span("GK \u00b7 SABR \u00b7 VV \u00b7 MC", style={
                "color": "#808080", "fontSize": "9px",
                "letterSpacing": "2px",
            }),
        ], style={
            "textAlign": "center", "padding": "12px",
            "borderTop": "1px solid #222240",
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# App Layout
# ═══════════════════════════════════════════════════════════════════════════

def serve_layout():
    return html.Div([

        # ── Global State Stores ──
        dcc.Store(id="global-pair",  data="EURUSD"),
        dcc.Store(id="preset-result", data=None),
        dcc.Store(id="cmd-nav-result", data=None),
        dcc.Store(id="global-tenor", data="3M"),
        dcc.Store(id="watchlist-store", data=DEFAULT_WATCHLIST),
        dcc.Store(id="metric-popup-data", data=None),

        # ── Data source status refresh (every 10s) ──
        dcc.Interval(id="data-source-interval", interval=10_000, n_intervals=0),

        # ── Hidden keyboard listener for Ctrl+K ──
        html.Div(id="kb-listener", style={"display": "none"}),

        # ── Modals ──
        make_command_palette(),
        make_watchlist_modal(),
        make_metric_popup(),

        # ── Header (ticker tape + logo + status) ──
        make_header(),

        # ── Main Content Area ──
        html.Div([
            make_workspace_tabs(),
            make_sub_tabs_container(),

            html.Div(id="panel-content", style={"marginTop": "4px"}),
        ], style={
            "padding": "8px 16px",
            "maxWidth": "1920px",
            "margin": "0 auto",
        }),

        # ── Footer ──
        make_footer(),

    ], style={
        "backgroundColor": "#000000",
        "minHeight": "100vh",
        "fontFamily": "'JetBrains Mono', monospace",
    })


app.layout = serve_layout


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 1. Workspace tabs -> render sub-tabs
# ---------------------------------------------------------------------------
@app.callback(
    Output("subtab-container", "children"),
    [Input("workspace-tabs", "value")],
)
def render_subtabs(workspace_id):
    ws = next((w for w in WORKSPACES if w["id"] == workspace_id), None)
    if ws is None:
        raise PreventUpdate

    accent = ws["accent"]
    sub_children = []
    for tab in ws["tabs"]:
        sub_children.append(
            dcc.Tab(
                label=tab["label"],
                value=tab["id"],
                style=_subtab_style(),
                selected_style=_subtab_selected_style(accent),
            )
        )

    return dcc.Tabs(
        id="sub-tabs",
        value=ws["tabs"][0]["id"],
        children=sub_children,
        style={
            "marginTop": "2px",
            "borderBottom": "1px solid #222240",
        },
    )


# ---------------------------------------------------------------------------
# 2. Sub-tabs -> render panel content
# ---------------------------------------------------------------------------
@app.callback(
    Output("panel-content", "children"),
    Input("sub-tabs", "value"),
)
def render_panel(tab_id):
    if tab_id is None:
        raise PreventUpdate

    module = _TAB_MODULE_MAP.get(tab_id)
    if module is None:
        return html.Div(
            "Panel not found.",
            style={"color": "#808080", "padding": "40px",
                   "textAlign": "center", "fontSize": "11px"},
        )
    try:
        return module.layout()
    except Exception as exc:
        import traceback
        return html.Div([
            html.Div("PANEL LOAD ERROR", style={
                "color": "#ff3333", "fontWeight": "700",
                "fontSize": "13px", "marginBottom": "8px",
            }),
            html.Pre(traceback.format_exc(), style={
                "color": "#808080", "fontSize": "10px",
                "whiteSpace": "pre-wrap",
            }),
        ], style={"padding": "40px"})


# ---------------------------------------------------------------------------
# 3. Workspace Preset Buttons
# ---------------------------------------------------------------------------
@app.callback(
    Output("preset-result", "data"),
    [Input({"type": "preset-btn", "index": ALL}, "n_clicks")],
    prevent_initial_call=True,
)
def apply_preset(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered or all(n == 0 for n in (n_clicks_list or [])):
        raise PreventUpdate

    prop_id = ctx.triggered[0]["prop_id"]
    try:
        btn_info = json.loads(prop_id.rsplit(".", 1)[0])
        idx = btn_info["index"]
    except Exception:
        raise PreventUpdate

    preset_names = list(WORKSPACE_PRESETS.keys())
    if idx < 0 or idx >= len(preset_names):
        raise PreventUpdate

    ws_id, tab_id = WORKSPACE_PRESETS[preset_names[idx]]
    return {"workspace": ws_id, "tab": tab_id}


# Preset → workspace + sub-tabs navigation
app.clientside_callback(
    """function(data) {
        if (!data) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        return [data.workspace, data.tab];
    }""",
    [Output("workspace-tabs", "value", allow_duplicate=True),
     Output("sub-tabs", "value", allow_duplicate=True)],
    Input("preset-result", "data"),
    prevent_initial_call=True,
)

# Command palette → navigate workspace/tab or set pair
app.clientside_callback(
    """function(data) {
        if (!data) return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
        if (data.pair) return [window.dash_clientside.no_update, window.dash_clientside.no_update, data.pair];
        return [data.workspace || window.dash_clientside.no_update, data.tab || window.dash_clientside.no_update, window.dash_clientside.no_update];
    }""",
    [Output("workspace-tabs", "value", allow_duplicate=True),
     Output("sub-tabs", "value", allow_duplicate=True),
     Output("global-pair", "data", allow_duplicate=True)],
    Input("cmd-nav-result", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# 4. Command Palette: toggle visibility (Ctrl+K)
# ---------------------------------------------------------------------------
app.clientside_callback(
    """
    function(id) {
        if (!window._cmdPaletteListenerAttached) {
            window._cmdPaletteListenerAttached = true;
            document.addEventListener('keydown', function(e) {
                if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                    e.preventDefault();
                    var overlay = document.getElementById('command-palette-overlay');
                    if (overlay) {
                        var visible = overlay.style.display === 'flex';
                        overlay.style.display = visible ? 'none' : 'flex';
                    }
                }
                if (e.key === 'Escape') {
                    var overlay = document.getElementById('command-palette-overlay');
                    if (overlay) overlay.style.display = 'none';
                    var wl = document.getElementById('watchlist-modal-overlay');
                    if (wl) wl.style.display = 'none';
                    var mp = document.getElementById('metric-popup-overlay');
                    if (mp) mp.style.display = 'none';
                }
            });
            var overlay = document.getElementById('command-palette-overlay');
            if (overlay) {
                overlay.addEventListener('click', function(e) {
                    if (e.target === overlay) overlay.style.display = 'none';
                });
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("kb-listener", "children"),
    Input("kb-listener", "id"),
)


# ---------------------------------------------------------------------------
# 5. Command Palette: selection -> navigate
# ---------------------------------------------------------------------------
@app.callback(
    [Output("cmd-nav-result", "data"),
     Output("command-palette-input", "value")],
    [Input("command-palette-input", "value")],
    prevent_initial_call=True,
)
def command_palette_navigate(selected_value):
    if not selected_value:
        raise PreventUpdate
    try:
        item = json.loads(selected_value)
    except Exception:
        raise PreventUpdate

    workspace = item.get("workspace", "")
    tab = item.get("tab", "")

    if workspace == "__pair__":
        return {"pair": tab}, None
    return {"workspace": workspace, "tab": tab}, None


# ---------------------------------------------------------------------------
# 6. Global Pair Linking (uses set_props to avoid errors on unmounted panels)
# ---------------------------------------------------------------------------
app.clientside_callback(
    """function(pair) {
        if (!pair) return window.dash_clientside.no_update;
        var targets = ['vsfx-pair', 'stb-pair', 'fxb-pair', 'rvp-pair-a',
                        'bt-pair', 'fxrisk-wi-pair'];
        targets.forEach(function(id) {
            dash_clientside.set_props(id, {value: pair});
        });
        return window.dash_clientside.no_update;
    }""",
    Output("global-pair", "id"),
    Input("global-pair", "data"),
)
# Market Dashboard → global pair
app.clientside_callback(
    """function(pair) { return pair || window.dash_clientside.no_update; }""",
    Output("global-pair", "data", allow_duplicate=True),
    Input("mdash-selected-pair", "data"),
    prevent_initial_call=True,
)
# Risk FX → global pair
app.clientside_callback(
    """function(pair) { return pair || window.dash_clientside.no_update; }""",
    Output("global-pair", "data", allow_duplicate=True),
    Input("fxrisk-selected-pair", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# 7. Watchlist Modal: open/close/save
# ---------------------------------------------------------------------------
app.clientside_callback(
    """
    function(n_open, n_close, n_save) {
        var overlay = document.getElementById('watchlist-modal-overlay');
        if (!overlay) return window.dash_clientside.no_update;
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || ctx.triggered.length === 0)
            return window.dash_clientside.no_update;
        var trigger = ctx.triggered[0].prop_id;
        if (trigger === 'watchlist-edit-btn.n_clicks') {
            overlay.style.display = 'flex';
        } else {
            overlay.style.display = 'none';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("watchlist-modal-overlay", "id"),
    [Input("watchlist-edit-btn", "n_clicks"),
     Input("watchlist-close-btn", "n_clicks"),
     Input("watchlist-save-btn", "n_clicks")],
)

@app.callback(
    Output("watchlist-store", "data"),
    Input("watchlist-save-btn", "n_clicks"),
    State("watchlist-editor-dropdown", "value"),
    prevent_initial_call=True,
)
def save_watchlist(n_clicks, pairs):
    if not pairs:
        raise PreventUpdate
    return pairs


# ---------------------------------------------------------------------------
# 8. Universal Metric Popup
# ---------------------------------------------------------------------------
@app.callback(
    [Output("metric-popup-overlay", "style"),
     Output("metric-popup-title", "children"),
     Output("metric-popup-stats", "children"),
     Output("metric-popup-chart", "figure")],
    [Input({"type": "clickable-metric", "pair": ALL, "metric": ALL, "tenor": ALL}, "n_clicks"),
     Input("metric-popup-close-btn", "n_clicks")],
    prevent_initial_call=True,
)
def handle_metric_popup(metric_clicks, close_clicks):
    """Universal metric popup — disabled (always hidden)."""
    return {"display": "none"}, "", [], go.Figure()

    # -- Original popup logic below (kept for reference) --
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"]

    # Close button
    if "metric-popup-close-btn" in trigger:
        return {"display": "none"}, "", [], go.Figure()

    # Parse the clicked metric ID
    try:
        id_str = trigger.rsplit(".", 1)[0]
        id_dict = json.loads(id_str)
        pair = id_dict["pair"]
        metric = id_dict["metric"]
        tenor = id_dict["tenor"]
    except Exception:
        raise PreventUpdate

    # Fetch history and compute stats
    try:
        from core.fx_analytics import vol_percentile
        info = vol_percentile(pair, tenor, metric, 252)

        current = info.get("current", 0)
        mean_val = info.get("mean", 0)
        std_val = info.get("std", 0)
        pct = info.get("percentile", 50)
        min_val = info.get("min", 0)
        max_val = info.get("max", 0)
        z_score = (current - mean_val) / max(std_val, 1e-6)

        # Get history for chart
        from core.bloomberg_fx import get_fx_historical_vol
        hist = get_fx_historical_vol(pair, tenor, metric, 252)
        if hist is None or len(hist) < 10:
            from core.fx_analytics import _synth_vol_history
            hist = _synth_vol_history(pair, tenor, metric, 252)

        # Build chart
        fig = go.Figure()
        x_days = list(range(len(hist)))

        # ±1σ band
        fig.add_trace(go.Scatter(
            x=x_days, y=[mean_val + std_val] * len(x_days),
            mode="lines", line=dict(color="#222240", width=1, dash="dot"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x_days, y=[mean_val - std_val] * len(x_days),
            mode="lines", line=dict(color="#222240", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(26,26,46,0.3)",
            showlegend=False,
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=x_days, y=[mean_val] * len(x_days),
            mode="lines", line=dict(color="#808080", width=1, dash="dash"),
            showlegend=False,
        ))

        # History line
        fig.add_trace(go.Scatter(
            x=x_days, y=hist.tolist() if hasattr(hist, 'tolist') else list(hist),
            mode="lines", line=dict(color="#ff8800", width=1.5),
            name=f"{metric} {tenor}",
        ))

        # Current dot
        fig.add_trace(go.Scatter(
            x=[len(hist) - 1], y=[current],
            mode="markers", marker=dict(color="#ff8800", size=8),
            showlegend=False,
        ))

        fig.update_layout(
            **CHART_TEMPLATE["layout"],
            height=280,
            margin=dict(l=50, r=20, t=10, b=30),
            showlegend=False,
        )

        # Stats
        stats_children = [
            html.Div([
                html.Span("Current: ", style={"color": "#808080", "fontSize": "9px"}),
                html.Span(f"{current:.2f}v", style={"color": "#d4d4d4", "fontSize": "11px", "fontWeight": "700"}),
            ]),
            html.Div([
                html.Span("Pctl: ", style={"color": "#808080", "fontSize": "9px"}),
                html.Span(f"{pct:.0f}th", style={
                    "color": "#00cc66" if pct < 30 else "#ff3333" if pct > 70 else "#d4d4d4",
                    "fontSize": "11px", "fontWeight": "700",
                }),
            ]),
            html.Div([
                html.Span("Mean: ", style={"color": "#808080", "fontSize": "9px"}),
                html.Span(f"{mean_val:.2f}v", style={"color": "#d4d4d4", "fontSize": "11px"}),
            ]),
            html.Div([
                html.Span("Std: ", style={"color": "#808080", "fontSize": "9px"}),
                html.Span(f"{std_val:.2f}v", style={"color": "#d4d4d4", "fontSize": "11px"}),
            ]),
            html.Div([
                html.Span("Z: ", style={"color": "#808080", "fontSize": "9px"}),
                html.Span(f"{z_score:+.2f}", style={
                    "color": "#00cc66" if abs(z_score) < 1 else "#ff3333",
                    "fontSize": "11px", "fontWeight": "700",
                }),
            ]),
            html.Div([
                html.Span("Min/Max: ", style={"color": "#808080", "fontSize": "9px"}),
                html.Span(f"{min_val:.2f} / {max_val:.2f}", style={"color": "#d4d4d4", "fontSize": "11px"}),
            ]),
        ]

        title = f"{pair} \u2014 {metric} {tenor} \u2014 252D HISTORY"

        return (
            {"display": "flex", "position": "fixed", "top": "0", "left": "0",
             "right": "0", "bottom": "0", "backgroundColor": "rgba(0,0,0,0.85)",
             "zIndex": "9000", "justifyContent": "center", "alignItems": "center"},
            title,
            stats_children,
            fig,
        )

    except Exception as exc:
        logger.exception("Metric popup error")
        return (
            {"display": "flex", "position": "fixed", "top": "0", "left": "0",
             "right": "0", "bottom": "0", "backgroundColor": "rgba(0,0,0,0.85)",
             "zIndex": "9000", "justifyContent": "center", "alignItems": "center"},
            f"{pair} \u2014 {metric} {tenor}",
            [html.Div(f"Error: {exc}", style={"color": "#ff3333", "fontSize": "11px"})],
            go.Figure(),
        )


# ---------------------------------------------------------------------------
# 9. Send to Lab button (metric popup → chart lab)
# ---------------------------------------------------------------------------
@app.callback(
    [Output("workspace-tabs", "value", allow_duplicate=True),
     Output("sub-tabs", "value", allow_duplicate=True)],
    Input("metric-popup-lab-btn", "n_clicks"),
    prevent_initial_call=True,
)
def send_to_lab(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return "risk-analytics", "chart-lab"


# ---------------------------------------------------------------------------
# 10. Compare button (metric popup → adds to comparison, for now just closes)
# ---------------------------------------------------------------------------
app.clientside_callback(
    """function(n) {
        if (!n) return window.dash_clientside.no_update;
        var el = document.getElementById('metric-popup-overlay');
        if (el) el.style.display = 'none';
        return window.dash_clientside.no_update;
    }""",
    Output("metric-popup-compare-btn", "id"),
    Input("metric-popup-compare-btn", "n_clicks"),
    prevent_initial_call=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# Ticker Tape Live Refresh (every 30s)
# ═══════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("ticker-tape-content", "children"),
    Input("ticker-refresh-interval", "n_intervals"),
    prevent_initial_call=True,
)
def _refresh_ticker(_n):
    items = _build_ticker_items()
    return items + items


# ═══════════════════════════════════════════════════════════════════════════
# Data Source Status (live provenance tracking)
# ═══════════════════════════════════════════════════════════════════════════

@app.callback(
    [Output("data-source-status", "children"),
     Output("header-last-updated", "children")],
    Input("data-source-interval", "n_intervals"),
)
def _update_data_source_status(_n):
    from core.bloomberg_fx import get_data_mode, get_recent_errors
    from datetime import datetime

    mode = get_data_mode()
    errors = get_recent_errors()
    timestamp = datetime.now().strftime("%H:%M:%S")

    if mode == "LIVE":
        return html.Span("ALL DATA LIVE", style={
            "color": "#00cc66", "fontSize": "9px", "fontWeight": "700",
            "fontFamily": "'JetBrains Mono', monospace", "letterSpacing": "0.5px",
        }), timestamp
    elif mode == "DEGRADED":
        # Group by function for a cleaner summary
        funcs = {}
        for e in errors:
            fn = e["function"].replace("get_fx_", "")
            funcs[fn] = funcs.get(fn, 0) + 1
        summary_parts = [f"{fn}({n})" for fn, n in sorted(funcs.items(), key=lambda x: -x[1])]
        error_summary = f"{len(errors)} failures: {', '.join(summary_parts[:4])}"
        # Full detail on hover
        detail = "\n".join(f"{e['function']}({e['pair']}): {e['error']}" for e in errors[-10:])
        return html.Span([
            html.Span("\u26A0 DEGRADED: ", style={
                "color": "#ff3333", "fontSize": "9px", "fontWeight": "700",
                "fontFamily": "'JetBrains Mono', monospace",
            }),
            html.Span(f"{error_summary}", title=detail, style={
                "color": "#ff3333", "fontSize": "9px",
                "fontFamily": "'JetBrains Mono', monospace",
                "cursor": "help",
            }),
        ]), timestamp
    else:
        # SYNTHETIC — dev mode, no Bloomberg. This is expected.
        return html.Span(""), timestamp  # Bloomberg badge already says "SYNTHETIC MODE"


# ═══════════════════════════════════════════════════════════════════════════
# Register All Panel Callbacks (10 panels)
# ═══════════════════════════════════════════════════════════════════════════

market_dashboard.register_callbacks(app)
vol_surface_fx.register_callbacks(app)
vol_scanner_unified.register_callbacks(app)
structure_builder.register_callbacks(app)
exotics_pricer.register_callbacks(app)
blotter_fx.register_callbacks(app)
risk_fx.register_callbacks(app)
relative_value_plus.register_callbacks(app)
chart_lab.register_callbacks(app)
backtest.register_callbacks(app)


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bbg_status = "BLOOMBERG LIVE" if is_connected() else "SYNTHETIC MODE"
    panels_total = sum(len(ws["tabs"]) for ws in WORKSPACES)

    print()
    print("=" * 64)
    print(f"  FX OPTIONS WORKSTATION v{VERSION} — CONSOLIDATED")
    print(f"  Data : {bbg_status}")
    print(f"  Model: Garman-Kohlhagen / SABR / Vanna-Volga / Monte Carlo")
    print(f"  Panels: {panels_total} across {len(WORKSPACES)} workspaces")
    print(f"  Workspaces: {' | '.join(w['label'] for w in WORKSPACES)}")
    print("=" * 64)
    # Try to launch as desktop app via pywebview, fall back to browser
    try:
        import webview
        import threading

        def _start_server():
            app.run(debug=False, host="127.0.0.1", port=8765,
                    use_reloader=False, dev_tools_ui=False, dev_tools_props_check=False)

        print("\n  Launching as desktop application...")
        print("  Close the window to stop.\n")

        server_thread = threading.Thread(target=_start_server, daemon=True)
        server_thread.start()

        webview.create_window(
            "FX Options Workstation",
            "http://127.0.0.1:8765",
            width=1920, height=1080,
            min_size=(1200, 700),
        )
        webview.start()

    except ImportError:
        print("\n  pywebview not available — opening in browser instead.")
        print(f"  ->  Open: http://localhost:8765")
        print(f"  ->  Ctrl+K for command palette\n")

        app.run(debug=True, host="0.0.0.0", port=8765,
                dev_tools_ui=False, dev_tools_props_check=False)
