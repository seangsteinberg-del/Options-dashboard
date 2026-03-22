#!/usr/bin/env python3
"""
FX Options Workstation v3
=========================
Institutional-grade FX options analytics platform.
Two-tier workspace with 19 panels, live ticker tape, command palette,
and global pair linking.

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

    # Optional: blpapi (don't fail if unavailable)
    try:
        __import__("blpapi")
    except ImportError:
        print("  Note: blpapi not installed -- running with synthetic data.")
        print("  To connect to Bloomberg: pip install blpapi\n")

_ensure_packages()

# ── Imports ───────────────────────────────────────────────────────────────
import json
import time
import numpy as np

import dash
from dash import html, dcc, Input, Output, State, callback_context, ALL, MATCH
from dash.exceptions import PreventUpdate

from core.theme import COLORS, TAB_STYLE, TAB_SELECTED_STYLE, status_color, status_text
from core.bloomberg import is_connected

# ── FX Panels (new) ──────────────────────────────────────────────────────
from panels import vol_surface_fx, vol_scanner, structure_builder, relative_value
from panels import risk_fx, blotter_fx, events_calendar, exotics_pricer
from panels import hedging_tools, backtest

# ── Equity Panels (legacy, kept for backwards compatibility) ─────────────
from panels import vol_surface, pricer, risk, blotter, chain
from panels import portfolio_panel, pnl_panel, analytics_panel, stress_panel


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

VERSION = "3.0"

# Major FX pairs for the ticker tape
FX_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "NZDUSD",
    "USDCAD", "EURGBP", "EURJPY", "GBPJPY", "USDMXN", "USDZAR",
]

# Synthetic mid-market rates (used when Bloomberg is not connected)
_FX_SEED_RATES = {
    "EURUSD": 1.0842, "USDJPY": 149.72, "GBPUSD": 1.2715,
    "USDCHF": 0.8794, "AUDUSD": 0.6538, "NZDUSD": 0.6127,
    "USDCAD": 1.3562, "EURGBP": 0.8527, "EURJPY": 162.34,
    "GBPJPY": 190.31, "USDMXN": 17.142, "USDZAR": 18.695,
}

# Two-tier workspace definition
# Each workspace has a label, an id, an accent color, and its sub-tabs.
WORKSPACES = [
    {
        "id": "fx-vol",
        "label": "FX VOL",
        "accent": COLORS["accent_cyan"],
        "tabs": [
            {"id": "vol-surface-fx", "label": "VOL SURFACE FX", "module": vol_surface_fx},
            {"id": "vol-scanner",    "label": "VOL SCANNER",    "module": vol_scanner},
        ],
    },
    {
        "id": "structuring",
        "label": "STRUCTURING",
        "accent": COLORS["accent_purple"],
        "tabs": [
            {"id": "structure-builder", "label": "STRUCTURE BUILDER", "module": structure_builder},
            {"id": "exotics-pricer",    "label": "EXOTICS PRICER",    "module": exotics_pricer},
        ],
    },
    {
        "id": "risk",
        "label": "RISK",
        "accent": COLORS["accent_red"],
        "tabs": [
            {"id": "risk-fx",       "label": "RISK FX",       "module": risk_fx},
            {"id": "hedging-tools", "label": "HEDGING TOOLS", "module": hedging_tools},
        ],
    },
    {
        "id": "rv-analytics",
        "label": "RV & ANALYTICS",
        "accent": COLORS["accent_green"],
        "tabs": [
            {"id": "relative-value",  "label": "RELATIVE VALUE",  "module": relative_value},
            {"id": "events-calendar", "label": "EVENTS CALENDAR", "module": events_calendar},
            {"id": "backtest",        "label": "BACKTEST",        "module": backtest},
        ],
    },
    {
        "id": "trading",
        "label": "TRADING",
        "accent": COLORS["accent_orange"],
        "tabs": [
            {"id": "blotter-fx", "label": "BLOTTER FX", "module": blotter_fx},
        ],
    },
    {
        "id": "equity-legacy",
        "label": "EQUITY (LEGACY)",
        "accent": COLORS["accent_indigo"],
        "tabs": [
            {"id": "eq-vol-surface", "label": "VOL SURFACE",   "module": vol_surface},
            {"id": "eq-pricer",      "label": "PRICER",         "module": pricer},
            {"id": "eq-risk",        "label": "RISK / GREEKS",  "module": risk},
            {"id": "eq-chain",       "label": "OPTIONS CHAIN",  "module": chain},
            {"id": "eq-blotter",     "label": "TRADE BLOTTER",  "module": blotter},
            {"id": "eq-portfolio",   "label": "PORTFOLIO",       "module": portfolio_panel},
            {"id": "eq-pnl",         "label": "P&L ATTRIB",     "module": pnl_panel},
            {"id": "eq-analytics",   "label": "ANALYTICS",       "module": analytics_panel},
            {"id": "eq-stress",      "label": "STRESS TEST",     "module": stress_panel},
        ],
    },
]

# Flat lookup: tab_id -> module (for routing)
_TAB_MODULE_MAP = {}
for ws in WORKSPACES:
    for tab in ws["tabs"]:
        _TAB_MODULE_MAP[tab["id"]] = tab["module"]

# Workspace presets: map preset name -> (workspace_id, first_tab_id)
WORKSPACE_PRESETS = {
    "Vol Trader":   ("fx-vol",       "vol-surface-fx"),
    "Structuring":  ("structuring",  "structure-builder"),
    "Risk Manager": ("risk",         "risk-fx"),
    "RV Analyst":   ("rv-analytics", "relative-value"),
}

# Command palette search items: (display_label, target_workspace, target_tab)
COMMAND_ITEMS = []
for ws in WORKSPACES:
    for tab in ws["tabs"]:
        COMMAND_ITEMS.append({
            "label": f"{tab['label']}  [{ws['label']}]",
            "workspace": ws["id"],
            "tab": tab["id"],
        })
for pair in FX_PAIRS:
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
        # Pip display: 4-decimal for most, 2-decimal for JPY and MXN/ZAR crosses
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


def make_ticker_tape():
    """Scrolling FX ticker tape across the top of the workstation."""
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

    # Duplicate items for seamless CSS scroll loop
    return html.Div([
        html.Div(items + items,
                 className="ticker-tape-inner",
                 style={"display": "inline-flex", "whiteSpace": "nowrap"}),
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
            "backgroundColor": COLORS["bg_secondary"],
            "color": COLORS["text_secondary"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "6px",
            "padding": "5px 14px",
            "fontSize": "9px",
            "fontFamily": "'JetBrains Mono', monospace",
            "fontWeight": "600",
            "letterSpacing": "1px",
            "textTransform": "uppercase",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
            "marginLeft": "6px",
        },
    )


def make_header():
    """Full header: ticker tape, logo, status badges, preset buttons."""
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
                        "fontWeight": "800", "fontSize": "22px",
                        "letterSpacing": "3px",
                        "background": (
                            f"linear-gradient(135deg, {COLORS['accent_cyan']}, "
                            f"{COLORS['accent_blue']}, {COLORS['accent_purple']})"
                        ),
                        "backgroundSize": "200% 200%",
                        "animation": "gradient-shift 4s ease infinite",
                        "-webkit-background-clip": "text",
                        "-webkit-text-fill-color": "transparent",
                    }),
                    html.Span(" WORKSTATION", style={
                        "fontWeight": "300", "color": COLORS["text_muted"],
                        "fontSize": "22px", "letterSpacing": "3px",
                    }),
                ]),
                html.Div("Institutional Derivatives Analytics", style={
                    "color": COLORS["text_muted"], "fontSize": "9px",
                    "textTransform": "uppercase", "letterSpacing": "4px",
                    "marginTop": "2px",
                }),
            ], style={"flex": "1"}),

            # ── Workspace Presets ──
            html.Div([
                html.Span("WORKSPACE", style={
                    "color": COLORS["text_muted"], "fontSize": "9px",
                    "letterSpacing": "1px", "marginRight": "8px",
                    "fontWeight": "600",
                }),
                *preset_buttons,
            ], style={
                "display": "flex", "alignItems": "center",
                "marginRight": "28px",
            }),

            # ── Status Block ──
            html.Div([
                # Connection badge
                html.Div([
                    html.Div(style={
                        "width": "8px", "height": "8px", "borderRadius": "50%",
                        "backgroundColor": dot_color, "display": "inline-block",
                        "marginRight": "8px",
                        "boxShadow": f"0 0 8px {dot_color}",
                        "animation": "pulse-dot 2s infinite",
                    }),
                    html.Span(badge_text, className=badge_class),
                ], style={
                    "display": "flex", "alignItems": "center",
                    "marginRight": "24px",
                }),

                # Model badge
                html.Div([
                    html.Span("MODEL ", style={
                        "color": COLORS["text_muted"], "fontSize": "10px",
                        "letterSpacing": "1px",
                    }),
                    html.Span("GK / SABR / VV / MC", style={
                        "color": COLORS["accent_purple"], "fontSize": "10px",
                        "fontWeight": "700",
                    }),
                ]),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "padding": "16px 36px",
            "backgroundColor": COLORS["bg_header"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ])


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
                        "color": COLORS["text_muted"], "fontSize": "9px",
                        "letterSpacing": "2px", "fontWeight": "700",
                    }),
                    html.Span("Ctrl+K", style={
                        "color": COLORS["accent_cyan"], "fontSize": "9px",
                        "fontWeight": "600", "marginLeft": "12px",
                        "padding": "2px 8px",
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "4px",
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
                        "backgroundColor": COLORS["bg_input"],
                        "color": COLORS["text_primary"],
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "14px",
                        "border": "none",
                    },
                ),
            ], style={
                "backgroundColor": COLORS["bg_card"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "16px",
                "padding": "20px 24px",
                "width": "560px",
                "maxWidth": "90vw",
                "boxShadow": "0 24px 80px rgba(0,0,0,0.7), 0 0 40px rgba(59,130,246,0.08)",
                "animation": "slide-up 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
            }),
        ],
        style={
            "display": "none",  # hidden by default
            "position": "fixed",
            "top": "0", "left": "0", "right": "0", "bottom": "0",
            "backgroundColor": "rgba(0,0,0,0.60)",
            "backdropFilter": "blur(4px)",
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
    """Style for a top-level workspace tab (unselected)."""
    return {
        "backgroundColor": "transparent",
        "border": f"1px solid {COLORS['border']}",
        "borderBottom": "none",
        "borderRadius": "8px 8px 0 0",
        "color": COLORS["text_muted"],
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "11px",
        "fontWeight": "700",
        "padding": "12px 22px",
        "letterSpacing": "1.5px",
        "textTransform": "uppercase",
        "cursor": "pointer",
        "transition": "all 0.2s ease",
    }


def _workspace_tab_selected_style(accent):
    """Style for a top-level workspace tab (selected)."""
    base = _workspace_tab_style(accent)
    return {
        **base,
        "backgroundColor": COLORS["bg_card"],
        "color": accent,
        "borderTop": f"2px solid {accent}",
        "boxShadow": f"0 -2px 12px {accent}26",
    }


def _subtab_style():
    """Style for second-tier (panel) tabs."""
    return {
        "backgroundColor": "transparent",
        "border": f"1px solid {COLORS['border_subtle']}",
        "borderBottom": "none",
        "borderRadius": "6px 6px 0 0",
        "color": COLORS["text_muted"],
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "10px",
        "fontWeight": "600",
        "padding": "10px 18px",
        "letterSpacing": "1.2px",
        "textTransform": "uppercase",
        "cursor": "pointer",
        "transition": "all 0.2s ease",
    }


def _subtab_selected_style(accent):
    """Style for second-tier tab (selected)."""
    base = _subtab_style()
    return {
        **base,
        "backgroundColor": COLORS["bg_secondary"],
        "color": accent,
        "borderTop": f"2px solid {accent}",
    }


def make_workspace_tabs():
    """Build the top-level workspace category tabs."""
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
    """Placeholder div whose children are swapped by the workspace callback."""
    return html.Div(id="subtab-container", style={"marginTop": "0"})


# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════

def make_footer():
    data_label = "BLOOMBERG API" if is_connected() else "SYNTHETIC DATA"
    return html.Div([
        html.Div([
            html.Span(f"FX OPTIONS WORKSTATION v{VERSION}", style={
                "color": COLORS["text_muted"], "fontSize": "9px",
                "letterSpacing": "2px",
            }),
            html.Span(" \u2502 ", style={
                "color": COLORS["border"], "margin": "0 10px",
            }),
            html.Span("GK \u00b7 SABR \u00b7 VANNA-VOLGA \u00b7 MONTE CARLO", style={
                "color": COLORS["text_muted"], "fontSize": "9px",
                "letterSpacing": "2px",
            }),
            html.Span(" \u2502 ", style={
                "color": COLORS["border"], "margin": "0 10px",
            }),
            html.Span(data_label, style={
                "color": status_color(), "fontSize": "9px",
                "letterSpacing": "2px", "fontWeight": "700",
            }),
            html.Span(" \u2502 ", style={
                "color": COLORS["border"], "margin": "0 10px",
            }),
            html.Span("PYTHON + DASH + PLOTLY + NUMPY + SCIPY", style={
                "color": COLORS["text_muted"], "fontSize": "9px",
                "letterSpacing": "2px",
            }),
        ], style={
            "textAlign": "center", "padding": "18px",
            "borderTop": f"1px solid {COLORS['border_subtle']}",
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# App Layout
# ═══════════════════════════════════════════════════════════════════════════

def serve_layout():
    """Return a fresh layout on every page load so stores reset and
    the ticker tape uses a new random seed each time."""
    return html.Div([

        # ── Global State Stores ──
        dcc.Store(id="global-pair",  data="EURUSD"),
        dcc.Store(id="global-tenor", data="3M"),

        # ── Hidden keyboard listener for Ctrl+K ──
        # We attach a clientside callback to the document via a hidden div.
        html.Div(id="kb-listener", style={"display": "none"}),

        # ── Command Palette Overlay ──
        make_command_palette(),

        # ── Header (ticker tape + logo + status) ──
        make_header(),

        # ── Main Content Area ──
        html.Div([
            # Top-tier workspace tabs
            make_workspace_tabs(),

            # Second-tier sub-tabs (rendered dynamically)
            make_sub_tabs_container(),

            # Panel content
            html.Div(id="panel-content", style={
                "marginTop": "16px",
                "animation": "slide-up 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            }),
        ], style={
            "padding": "24px 36px",
            "maxWidth": "1920px",
            "margin": "0 auto",
        }),

        # ── Footer ──
        make_footer(),

    ], style={
        "backgroundColor": COLORS["bg_primary"],
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
    """When the top-level workspace changes, render its sub-tab row."""
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
            "marginTop": "4px",
            "borderBottom": f"1px solid {COLORS['border_subtle']}",
        },
    )


# ---------------------------------------------------------------------------
# 2. Sub-tabs -> render panel content
# ---------------------------------------------------------------------------
@app.callback(
    Output("panel-content", "children"),
    [Input("sub-tabs", "value")],
)
def render_panel(tab_id):
    """Route the selected sub-tab to the correct panel layout."""
    if tab_id is None:
        raise PreventUpdate

    module = _TAB_MODULE_MAP.get(tab_id)
    if module is None:
        return html.Div(
            "Panel not found.",
            style={"color": COLORS["text_muted"], "padding": "40px",
                   "textAlign": "center", "fontSize": "13px"},
        )
    try:
        return module.layout()
    except Exception as exc:
        return html.Div([
            html.Div("PANEL LOAD ERROR", style={
                "color": COLORS["accent_red"], "fontWeight": "700",
                "fontSize": "14px", "marginBottom": "8px",
            }),
            html.Pre(str(exc), style={
                "color": COLORS["text_muted"], "fontSize": "11px",
                "whiteSpace": "pre-wrap",
            }),
        ], style={"padding": "40px"})


# ---------------------------------------------------------------------------
# 3. Workspace Preset Buttons -> switch workspace + sub-tab
# ---------------------------------------------------------------------------
@app.callback(
    [Output("workspace-tabs", "value"),
     Output("sub-tabs", "value", allow_duplicate=True),
     Output("global-pair", "data", allow_duplicate=True)],
    [Input({"type": "preset-btn", "index": ALL}, "n_clicks")],
    prevent_initial_call=True,
)
def apply_preset(n_clicks_list):
    """When a preset button is clicked, jump to the associated workspace and sub-tab."""
    ctx = callback_context
    if not ctx.triggered or all(n == 0 for n in (n_clicks_list or [])):
        raise PreventUpdate

    # Determine which button was clicked
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
    # Navigate to workspace and sub-tab; keep current pair unchanged
    return ws_id, tab_id, dash.no_update


# ---------------------------------------------------------------------------
# 4. Command Palette: toggle visibility (Ctrl+K)
# ---------------------------------------------------------------------------
# Clientside JS to listen for Ctrl+K and toggle the overlay display.
app.clientside_callback(
    """
    function(id) {
        // Attach keyboard listener once
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
                }
            });
            // Click outside to close
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
# 5. Command Palette: selection -> navigate to panel or set pair
# ---------------------------------------------------------------------------
@app.callback(
    [Output("workspace-tabs", "value", allow_duplicate=True),
     Output("sub-tabs", "value", allow_duplicate=True),
     Output("global-pair", "data", allow_duplicate=True),
     Output("command-palette-input", "value")],
    [Input("command-palette-input", "value")],
    prevent_initial_call=True,
)
def command_palette_navigate(selected_value):
    """When the user picks an item from the palette, navigate or set pair."""
    if not selected_value:
        raise PreventUpdate
    try:
        item = json.loads(selected_value)
    except Exception:
        raise PreventUpdate

    workspace = item.get("workspace", "")
    tab = item.get("tab", "")

    if workspace == "__pair__":
        # Set the global pair; don't change workspace or sub-tab
        return dash.no_update, dash.no_update, tab, None

    # Navigate to the workspace AND the specific sub-tab
    return workspace, tab, dash.no_update, None


# ---------------------------------------------------------------------------
# 6. Global Pair Linking: global-pair store -> vsfx-pair dropdown
# ---------------------------------------------------------------------------
app.clientside_callback(
    """
    function(pair) {
        return pair;
    }
    """,
    Output("vsfx-pair", "value"),
    Input("global-pair", "data"),
)


# ═══════════════════════════════════════════════════════════════════════════
# Register All Panel Callbacks (19 panels)
# ═══════════════════════════════════════════════════════════════════════════

# FX panels
vol_surface_fx.register_callbacks(app)
vol_scanner.register_callbacks(app)
structure_builder.register_callbacks(app)
relative_value.register_callbacks(app)
risk_fx.register_callbacks(app)
blotter_fx.register_callbacks(app)
events_calendar.register_callbacks(app)
exotics_pricer.register_callbacks(app)
hedging_tools.register_callbacks(app)
backtest.register_callbacks(app)

# Legacy equity panels
vol_surface.register_callbacks(app)
pricer.register_callbacks(app)
risk.register_callbacks(app)
chain.register_callbacks(app)
blotter.register_callbacks(app)
portfolio_panel.register_callbacks(app)
pnl_panel.register_callbacks(app)
analytics_panel.register_callbacks(app)
stress_panel.register_callbacks(app)


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bbg_status = "BLOOMBERG LIVE" if is_connected() else "SYNTHETIC MODE"
    panels_fx = sum(len(ws["tabs"]) for ws in WORKSPACES if ws["id"] != "equity-legacy")
    panels_eq = sum(len(ws["tabs"]) for ws in WORKSPACES if ws["id"] == "equity-legacy")

    print()
    print("\u2550" * 64)
    print(f"  FX OPTIONS WORKSTATION v{VERSION}")
    print(f"  Institutional Derivatives Analytics")
    print(f"  Data : {bbg_status}")
    print(f"  Model: Garman-Kohlhagen / SABR / Vanna-Volga / Monte Carlo")
    print(f"  Panels: {panels_fx} FX + {panels_eq} Equity (legacy) = {panels_fx + panels_eq} total")
    print("\u2550" * 64)
    print(f"\n  \u27a4  Open: http://localhost:8050")
    print(f"  \u27a4  Ctrl+K for command palette\n")

    app.run(debug=True, host="0.0.0.0", port=8050)
