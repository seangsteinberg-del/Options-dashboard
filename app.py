#!/usr/bin/env python3
"""
Options Dashboard
=================
Professional-grade options analytics platform.

Run:
    python app.py

Then open http://localhost:8050 in your browser.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from core.theme import COLORS, TAB_STYLE, TAB_SELECTED_STYLE, CARD_HEADER_STYLE
from panels import vol_surface, pricer, risk, blotter


# ═══════════════════════════════════════════════════════════════════════════
# App Initialization
# ═══════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Options Dashboard",
    update_title="Calculating...",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)

server = app.server


# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════

def make_header():
    return html.Div([
        html.Div([
            # Logo / Title
            html.Div([
                html.Div([
                    html.Span("OPTIONS", style={
                        "fontWeight": "700",
                        "background": f"linear-gradient(135deg, {COLORS['accent_cyan']}, {COLORS['accent_blue']})",
                        "-webkit-background-clip": "text",
                        "-webkit-text-fill-color": "transparent",
                        "fontSize": "20px",
                        "letterSpacing": "2px",
                    }),
                    html.Span(" DASHBOARD", style={
                        "fontWeight": "300",
                        "color": COLORS["text_secondary"],
                        "fontSize": "20px",
                        "letterSpacing": "2px",
                    }),
                ]),
                html.Div("Quantitative Analytics Platform", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "10px",
                    "textTransform": "uppercase",
                    "letterSpacing": "3px",
                    "marginTop": "2px",
                }),
            ], style={"flex": "1"}),

            # Status indicators
            html.Div([
                html.Div([
                    html.Div(style={
                        "width": "8px", "height": "8px", "borderRadius": "50%",
                        "backgroundColor": COLORS["accent_green"],
                        "display": "inline-block", "marginRight": "6px",
                        "boxShadow": f"0 0 8px {COLORS['accent_green']}",
                        "animation": "pulse-glow 2s infinite",
                    }),
                    html.Span("LIVE", style={
                        "color": COLORS["accent_green"],
                        "fontSize": "11px", "fontWeight": "600",
                        "letterSpacing": "1px",
                    }),
                ], style={"display": "flex", "alignItems": "center", "marginRight": "24px"}),

                html.Div([
                    html.Span("MODEL ", style={"color": COLORS["text_muted"], "fontSize": "10px"}),
                    html.Span("BLACK-SCHOLES", style={
                        "color": COLORS["accent_blue"], "fontSize": "10px",
                        "fontWeight": "600",
                    }),
                ], style={"marginRight": "24px"}),

                html.Div([
                    html.Span("ENGINE ", style={"color": COLORS["text_muted"], "fontSize": "10px"}),
                    html.Span("ANALYTICAL", style={
                        "color": COLORS["accent_purple"], "fontSize": "10px",
                        "fontWeight": "600",
                    }),
                ]),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "padding": "16px 32px",
            "backgroundColor": COLORS["bg_header"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    make_header(),

    html.Div([
        dcc.Tabs(
            id="main-tabs",
            value="vol-surface",
            children=[
                dcc.Tab(
                    label="VOL SURFACE",
                    value="vol-surface",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
                dcc.Tab(
                    label="PRICER",
                    value="pricer",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
                dcc.Tab(
                    label="RISK / GREEKS",
                    value="risk",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
                dcc.Tab(
                    label="TRADE BLOTTER",
                    value="blotter",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
            ],
            style={"marginBottom": "0px"},
        ),

        html.Div(id="tab-content", style={"marginTop": "16px"}),
    ], style={
        "padding": "20px 32px",
        "maxWidth": "1800px",
        "margin": "0 auto",
    }),

    # Footer
    html.Div([
        html.Div([
            html.Span("OPTIONS DASHBOARD v1.0", style={
                "color": COLORS["text_muted"], "fontSize": "10px",
                "letterSpacing": "2px",
            }),
            html.Span("  |  ", style={"color": COLORS["border"], "margin": "0 12px"}),
            html.Span("BLACK-SCHOLES ANALYTICAL ENGINE", style={
                "color": COLORS["text_muted"], "fontSize": "10px",
                "letterSpacing": "2px",
            }),
            html.Span("  |  ", style={"color": COLORS["border"], "margin": "0 12px"}),
            html.Span("PYTHON + DASH + PLOTLY", style={
                "color": COLORS["text_muted"], "fontSize": "10px",
                "letterSpacing": "2px",
            }),
        ], style={
            "textAlign": "center",
            "padding": "16px",
            "borderTop": f"1px solid {COLORS['border_subtle']}",
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ]),

], style={
    "backgroundColor": COLORS["bg_primary"],
    "minHeight": "100vh",
    "fontFamily": "'JetBrains Mono', monospace",
})


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
    elif tab == "blotter":
        return blotter.layout()
    return html.Div("Select a tab", style={"color": COLORS["text_muted"]})


# ═══════════════════════════════════════════════════════════════════════════
# Register All Panel Callbacks
# ═══════════════════════════════════════════════════════════════════════════

vol_surface.register_callbacks(app)
pricer.register_callbacks(app)
risk.register_callbacks(app)
blotter.register_callbacks(app)


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  OPTIONS DASHBOARD")
    print("  Quantitative Analytics Platform")
    print("=" * 60)
    print(f"\n  Open: http://localhost:8050\n")

    app.run(debug=True, host="0.0.0.0", port=8050)
