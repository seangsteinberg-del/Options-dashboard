#!/usr/bin/env python3
"""
Options Dashboard v2
====================
Bloomberg-connected quantitative options analytics platform.

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
        print("  Note: blpapi not installed — running with synthetic data.")
        print("  To connect to Bloomberg: pip install blpapi\n")

_ensure_packages()

# ── Imports ───────────────────────────────────────────────────────────────
import dash
from dash import html, dcc

from core.theme import COLORS, TAB_STYLE, TAB_SELECTED_STYLE, status_color, status_text
from core.bloomberg import is_connected, get_spot_prices
from panels import vol_surface, pricer, risk, blotter
from panels import chain


# ═══════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Options Dashboard",
    update_title="Calculating...",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
)
server = app.server


# ═══════════════════════════════════════════════════════════════════════════
# Ticker Tape
# ═══════════════════════════════════════════════════════════════════════════

def make_ticker_tape():
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "GS", "XOM", "GLD"]
    spot_data = get_spot_prices(tickers)

    items = []
    for tk in tickers:
        info = spot_data.get(tk, {})
        price = info.get("price", 0)
        change = info.get("change_pct", 0)
        color = COLORS["accent_green"] if change >= 0 else COLORS["accent_red"]
        arrow = "\u25B2" if change >= 0 else "\u25BC"

        items.append(html.Span([
            html.Span(tk, style={"color": COLORS["text_primary"], "fontWeight": "700", "marginRight": "6px"}),
            html.Span(f"${price:,.2f}", style={"color": COLORS["text_secondary"], "marginRight": "5px"}),
            html.Span(f"{arrow} {change:+.2f}%", style={"color": color, "fontWeight": "600"}),
        ], style={"marginRight": "36px", "fontSize": "11px", "display": "inline-block",
                  "fontFamily": "'JetBrains Mono', monospace"}))

    # Duplicate for seamless scroll
    return html.Div([
        html.Div(items + items, className="ticker-tape-inner",
                 style={"display": "inline-flex", "whiteSpace": "nowrap"}),
    ], className="ticker-tape")


# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════

def make_header():
    bbg_connected = is_connected()
    badge_class = "bbg-badge connected" if bbg_connected else "bbg-badge disconnected"
    badge_text = "BLOOMBERG LIVE" if bbg_connected else "SYNTHETIC MODE"
    dot_color = COLORS["accent_green"] if bbg_connected else COLORS["accent_orange"]

    return html.Div([
        make_ticker_tape(),
        html.Div([
            # Logo
            html.Div([
                html.Div([
                    html.Span("OPTIONS", style={
                        "fontWeight": "800", "fontSize": "22px", "letterSpacing": "3px",
                        "background": f"linear-gradient(135deg, {COLORS['accent_cyan']}, {COLORS['accent_blue']}, {COLORS['accent_purple']})",
                        "backgroundSize": "200% 200%",
                        "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent",
                    }),
                    html.Span(" DASHBOARD", style={
                        "fontWeight": "300", "color": COLORS["text_muted"],
                        "fontSize": "22px", "letterSpacing": "3px",
                    }),
                ]),
                html.Div("Quantitative Analytics Platform", style={
                    "color": COLORS["text_muted"], "fontSize": "9px",
                    "textTransform": "uppercase", "letterSpacing": "4px", "marginTop": "2px",
                }),
            ], style={"flex": "1"}),

            # Status
            html.Div([
                html.Div([
                    html.Div(style={
                        "width": "8px", "height": "8px", "borderRadius": "50%",
                        "backgroundColor": dot_color, "display": "inline-block",
                        "marginRight": "8px", "boxShadow": f"0 0 8px {dot_color}",
                        "animation": "pulse-dot 2s infinite",
                    }),
                    html.Span(badge_text, className=badge_class),
                ], style={"display": "flex", "alignItems": "center", "marginRight": "28px"}),

                html.Div([
                    html.Span("MODEL ", style={"color": COLORS["text_muted"], "fontSize": "10px", "letterSpacing": "1px"}),
                    html.Span("BLACK-SCHOLES", style={"color": COLORS["accent_blue"], "fontSize": "10px", "fontWeight": "700"}),
                ], style={"marginRight": "20px"}),

                html.Div([
                    html.Span("ENGINE ", style={"color": COLORS["text_muted"], "fontSize": "10px", "letterSpacing": "1px"}),
                    html.Span("BS / MC / BINOM / SABR", style={"color": COLORS["accent_purple"], "fontSize": "10px", "fontWeight": "700"}),
                ]),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "padding": "18px 36px",
            "backgroundColor": COLORS["bg_header"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    # Auto-refresh interval (for live Bloomberg data)
    dcc.Interval(id="live-interval", interval=30 * 1000, n_intervals=0),

    make_header(),

    html.Div([
        dcc.Tabs(id="main-tabs", value="vol-surface", children=[
            dcc.Tab(label="VOL SURFACE", value="vol-surface",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="PRICER", value="pricer",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="RISK / GREEKS", value="risk",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="OPTIONS CHAIN", value="chain",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="TRADE BLOTTER", value="blotter",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ], style={"marginBottom": "0"}),

        html.Div(id="tab-content", style={"marginTop": "16px"}),
    ], style={"padding": "24px 36px", "maxWidth": "1920px", "margin": "0 auto"}),

    # Footer
    html.Div([
        html.Div([
            html.Span("OPTIONS DASHBOARD v2.0", style={"color": COLORS["text_muted"], "fontSize": "9px", "letterSpacing": "2px"}),
            html.Span(" \u2502 ", style={"color": COLORS["border"], "margin": "0 10px"}),
            html.Span("BS \u00b7 MONTE CARLO \u00b7 BINOMIAL \u00b7 SABR", style={"color": COLORS["text_muted"], "fontSize": "9px", "letterSpacing": "2px"}),
            html.Span(" \u2502 ", style={"color": COLORS["border"], "margin": "0 10px"}),
            html.Span("BLOOMBERG API" if is_connected() else "SYNTHETIC DATA", style={
                "color": status_color(), "fontSize": "9px", "letterSpacing": "2px", "fontWeight": "700"}),
            html.Span(" \u2502 ", style={"color": COLORS["border"], "margin": "0 10px"}),
            html.Span("PYTHON + DASH + PLOTLY", style={"color": COLORS["text_muted"], "fontSize": "9px", "letterSpacing": "2px"}),
        ], style={
            "textAlign": "center", "padding": "18px",
            "borderTop": f"1px solid {COLORS['border_subtle']}",
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ]),
], style={"backgroundColor": COLORS["bg_primary"], "minHeight": "100vh",
          "fontFamily": "'JetBrains Mono', monospace"})


# ═══════════════════════════════════════════════════════════════════════════
# Tab Routing
# ═══════════════════════════════════════════════════════════════════════════

@app.callback(
    dash.Output("tab-content", "children"),
    [dash.Input("main-tabs", "value")],
)
def render_tab(tab):
    if tab == "vol-surface":
        return vol_surface.layout()
    elif tab == "pricer":
        return pricer.layout()
    elif tab == "risk":
        return risk.layout()
    elif tab == "chain":
        return chain.layout()
    elif tab == "blotter":
        return blotter.layout()
    return html.Div("Select a tab", style={"color": COLORS["text_muted"]})


# ═══════════════════════════════════════════════════════════════════════════
# Register All Callbacks
# ═══════════════════════════════════════════════════════════════════════════

vol_surface.register_callbacks(app)
pricer.register_callbacks(app)
risk.register_callbacks(app)
chain.register_callbacks(app)
blotter.register_callbacks(app)


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bbg_status = "BLOOMBERG LIVE" if is_connected() else "SYNTHETIC MODE"
    print("\n" + "\u2550" * 60)
    print(f"  OPTIONS DASHBOARD v2.0")
    print(f"  Quantitative Analytics Platform")
    print(f"  Data: {bbg_status}")
    print("\u2550" * 60)
    print(f"\n  \u27a4  Open: http://localhost:8050\n")

    app.run(debug=True, host="0.0.0.0", port=8050)
