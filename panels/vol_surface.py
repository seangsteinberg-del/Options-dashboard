"""
Volatility Surface Panel
========================
Interactive 3-D implied vol surface with customizable skew, term structure,
smile parameters, and multiple visualization modes.
"""

from dash import html, dcc, dash_table, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np

from core.pricing import generate_vol_surface, implied_vol, bs_price
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE,
)


def make_param_input(label, id_, value, min_val, max_val, step, tooltip=""):
    return html.Div([
        html.Label(label, style=LABEL_STYLE, title=tooltip),
        dcc.Input(
            id=id_, type="number", value=value,
            min=min_val, max=max_val, step=step,
            style={**INPUT_STYLE, "marginBottom": "12px"},
            debounce=True,
        ),
    ], style={"flex": "1", "minWidth": "130px", "marginRight": "12px"})


def layout():
    return html.Div([
        # ── Controls Row ──────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div("VOL SURFACE PARAMETERS", style=CARD_HEADER_STYLE),
                html.Div([
                    make_param_input("SPOT PRICE", "vol-spot", 100, 1, 10000, 0.5,
                                     "Underlying spot price"),
                    make_param_input("BASE VOL", "vol-base", 0.20, 0.01, 2.0, 0.01,
                                     "ATM implied volatility"),
                    make_param_input("SKEW SLOPE", "vol-skew", -0.15, -1.0, 1.0, 0.01,
                                     "Put skew steepness (negative = normal skew)"),
                    make_param_input("SMILE CONVEXITY", "vol-convex", 0.10, 0.0, 1.0, 0.01,
                                     "Smile curvature / butterfly"),
                    make_param_input("TERM SLOPE", "vol-term", 0.02, -0.2, 0.2, 0.005,
                                     "Vol increase with maturity"),
                    make_param_input("STRIKE RANGE ±%", "vol-strike-range", 30, 5, 80, 5,
                                     "Strike range as % around ATM"),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),

                html.Div([
                    html.Label("VISUALIZATION MODE", style=LABEL_STYLE),
                    dcc.RadioItems(
                        id="vol-viz-mode",
                        options=[
                            {"label": "  3D Surface", "value": "surface"},
                            {"label": "  Heatmap", "value": "heatmap"},
                            {"label": "  Skew Curves", "value": "skew"},
                            {"label": "  Term Structure", "value": "term"},
                        ],
                        value="surface",
                        inline=True,
                        style={"color": COLORS["text_secondary"], "fontSize": "12px",
                               "fontFamily": "'JetBrains Mono', monospace"},
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "20px", "cursor": "pointer"},
                    ),
                ], style={"marginTop": "12px"}),

                html.Div([
                    html.Label("COLOR SCHEME", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="vol-colorscale",
                        options=[
                            {"label": "Viridis", "value": "Viridis"},
                            {"label": "Plasma", "value": "Plasma"},
                            {"label": "Inferno", "value": "Inferno"},
                            {"label": "Turbo", "value": "Turbo"},
                            {"label": "Thermal", "value": "thermal"},
                            {"label": "Electric", "value": "Electric"},
                            {"label": "Ice", "value": "ice"},
                            {"label": "RdBu (Diverging)", "value": "RdBu"},
                        ],
                        value="Plasma",
                        clearable=False,
                        style={"fontSize": "12px", "marginTop": "4px"},
                    ),
                ], style={"marginTop": "12px", "maxWidth": "220px"}),
            ], style=CARD_STYLE, className="dashboard-card"),
        ]),

        # ── Stats Row ─────────────────────────────────────────────────
        html.Div(id="vol-stats-row", style={
            "display": "flex", "gap": "12px", "marginBottom": "16px", "flexWrap": "wrap",
        }),

        # ── Chart ─────────────────────────────────────────────────────
        html.Div([
            dcc.Loading(
                dcc.Graph(
                    id="vol-surface-chart",
                    config={"displayModeBar": True, "scrollZoom": True,
                            "modeBarButtonsToAdd": ["toImage"]},
                    style={"height": "560px"},
                ),
                type="dot", color=COLORS["accent_cyan"],
            ),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Smile Snapshot Table ──────────────────────────────────────
        html.Div([
            html.Div("ATM VOLATILITY TERM STRUCTURE", style=CARD_HEADER_STYLE),
            html.Div(id="vol-term-table"),
        ], style=CARD_STYLE, className="dashboard-card"),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("vol-surface-chart", "figure"),
         Output("vol-stats-row", "children"),
         Output("vol-term-table", "children")],
        [Input("vol-spot", "value"),
         Input("vol-base", "value"),
         Input("vol-skew", "value"),
         Input("vol-convex", "value"),
         Input("vol-term", "value"),
         Input("vol-strike-range", "value"),
         Input("vol-viz-mode", "value"),
         Input("vol-colorscale", "value")],
    )
    def update_vol_surface(spot, base_vol, skew, convexity, term_slope,
                           strike_range_pct, viz_mode, colorscale):
        spot = spot or 100
        base_vol = base_vol or 0.20
        skew = skew if skew is not None else -0.15
        convexity = convexity or 0.10
        term_slope = term_slope if term_slope is not None else 0.02
        strike_range_pct = (strike_range_pct or 30) / 100.0

        strikes, expiries, vol_matrix = generate_vol_surface(
            S=spot, base_vol=base_vol, skew_slope=skew,
            skew_convexity=convexity, term_slope=term_slope,
            num_strikes=50, num_expiries=25,
            strike_range=(1 - strike_range_pct, 1 + strike_range_pct),
        )

        # ── Build Figure ──────────────────────────────────────────
        fig = go.Figure()
        tpl = CHART_TEMPLATE["layout"]

        if viz_mode == "surface":
            fig.add_trace(go.Surface(
                x=strikes, y=expiries, z=vol_matrix * 100,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(text="IV %", font=dict(color=COLORS["text_secondary"], size=11)),
                    tickfont=dict(color=COLORS["text_secondary"], size=10),
                    bgcolor=COLORS["bg_card"],
                    bordercolor=COLORS["border_subtle"],
                    len=0.6,
                ),
                opacity=0.92,
                hovertemplate=(
                    "<b>Strike:</b> %{x:.1f}<br>"
                    "<b>Expiry:</b> %{y:.2f}y<br>"
                    "<b>IV:</b> %{z:.2f}%<extra></extra>"
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, project_z=True, highlightcolor="#fff", width=1),
                ),
            ))
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title="Strike", backgroundcolor=COLORS["bg_card"],
                               gridcolor=COLORS["border_subtle"], color=COLORS["text_secondary"]),
                    yaxis=dict(title="Expiry (yrs)", backgroundcolor=COLORS["bg_card"],
                               gridcolor=COLORS["border_subtle"], color=COLORS["text_secondary"]),
                    zaxis=dict(title="IV (%)", backgroundcolor=COLORS["bg_card"],
                               gridcolor=COLORS["border_subtle"], color=COLORS["text_secondary"]),
                    bgcolor=COLORS["bg_card"],
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
                ),
                title=dict(text="Implied Volatility Surface", font=dict(
                    color=COLORS["text_primary"], size=14, family="JetBrains Mono")),
            )

        elif viz_mode == "heatmap":
            fig.add_trace(go.Heatmap(
                x=strikes, y=expiries, z=vol_matrix * 100,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(text="IV %", font=dict(color=COLORS["text_secondary"])),
                    tickfont=dict(color=COLORS["text_secondary"]),
                ),
                hovertemplate=(
                    "<b>K:</b> %{x:.1f}<br><b>T:</b> %{y:.2f}y<br>"
                    "<b>IV:</b> %{z:.2f}%<extra></extra>"
                ),
            ))
            fig.update_layout(
                xaxis_title="Strike", yaxis_title="Expiry (years)",
                title=dict(text="IV Heatmap", font=dict(color=COLORS["text_primary"], size=14)),
            )

        elif viz_mode == "skew":
            expiry_indices = np.linspace(0, len(expiries) - 1, 6, dtype=int)
            colors = [COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_purple"],
                      COLORS["accent_green"], COLORS["accent_orange"], COLORS["accent_pink"]]
            for idx, ei in enumerate(expiry_indices):
                fig.add_trace(go.Scatter(
                    x=strikes, y=vol_matrix[ei, :] * 100,
                    mode="lines", name=f"{expiries[ei]:.2f}y",
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate=f"<b>{expiries[ei]:.2f}y</b><br>K: %{{x:.1f}}<br>IV: %{{y:.2f}}%<extra></extra>",
                ))
            fig.update_layout(
                xaxis_title="Strike", yaxis_title="Implied Vol (%)",
                title=dict(text="Volatility Skew by Expiry", font=dict(
                    color=COLORS["text_primary"], size=14)),
                legend=dict(font=dict(color=COLORS["text_secondary"], size=11)),
            )

        elif viz_mode == "term":
            atm_idx = np.argmin(np.abs(strikes - spot))
            offsets = [-5, -2, 0, 2, 5]
            colors = [COLORS["accent_red"], COLORS["accent_orange"], COLORS["accent_cyan"],
                      COLORS["accent_green"], COLORS["accent_purple"]]
            for i, off in enumerate(offsets):
                si = max(0, min(len(strikes) - 1, atm_idx + off))
                fig.add_trace(go.Scatter(
                    x=expiries, y=vol_matrix[:, si] * 100,
                    mode="lines+markers", name=f"K={strikes[si]:.0f}",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4),
                ))
            fig.update_layout(
                xaxis_title="Expiry (years)", yaxis_title="Implied Vol (%)",
                title=dict(text="Term Structure of Volatility", font=dict(
                    color=COLORS["text_primary"], size=14)),
                legend=dict(font=dict(color=COLORS["text_secondary"], size=11)),
            )

        fig.update_layout(
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            margin=dict(l=50, r=20, t=50, b=40),
            hoverlabel=tpl["hoverlabel"],
        )

        # ── Stats Boxes ───────────────────────────────────────────
        atm_idx = np.argmin(np.abs(strikes - spot))
        short_idx = 0
        long_idx = -1
        min_vol = vol_matrix.min() * 100
        max_vol = vol_matrix.max() * 100
        atm_short = vol_matrix[short_idx, atm_idx] * 100
        atm_long = vol_matrix[long_idx, atm_idx] * 100
        skew_25d = (vol_matrix[len(expiries) // 4, 0] - vol_matrix[len(expiries) // 4, -1]) * 100

        def stat_box(label, value, color=COLORS["accent_cyan"]):
            return html.Div([
                html.Div(f"{value:.2f}%", className="stat-value",
                         style={"color": color}),
                html.Div(label, className="stat-label"),
            ], style={**STAT_BOX_STYLE, "flex": "1"})

        stats = [
            stat_box("ATM SHORT-DATED", atm_short, COLORS["accent_cyan"]),
            stat_box("ATM LONG-DATED", atm_long, COLORS["accent_blue"]),
            stat_box("MIN IV", min_vol, COLORS["accent_green"]),
            stat_box("MAX IV", max_vol, COLORS["accent_red"]),
            stat_box("25D SKEW", skew_25d, COLORS["accent_purple"]),
            stat_box("TERM SPREAD", atm_long - atm_short,
                     COLORS["accent_orange"]),
        ]

        # ── Term Structure Table ──────────────────────────────────
        term_data = []
        for i, T in enumerate(expiries[::max(1, len(expiries) // 8)]):
            row_idx = min(i * max(1, len(expiries) // 8), len(expiries) - 1)
            term_data.append({
                "Expiry": f"{T:.3f}y",
                "ATM IV": f"{vol_matrix[row_idx, atm_idx] * 100:.2f}%",
                "25D Put": f"{vol_matrix[row_idx, max(0, atm_idx - 5)] * 100:.2f}%",
                "25D Call": f"{vol_matrix[row_idx, min(len(strikes) - 1, atm_idx + 5)] * 100:.2f}%",
                "Skew": f"{(vol_matrix[row_idx, max(0, atm_idx - 5)] - vol_matrix[row_idx, min(len(strikes) - 1, atm_idx + 5)]) * 100:.2f}%",
                "Butterfly": f"{((vol_matrix[row_idx, max(0, atm_idx - 5)] + vol_matrix[row_idx, min(len(strikes) - 1, atm_idx + 5)]) / 2 - vol_matrix[row_idx, atm_idx]) * 100:.2f}%",
            })

        term_table = dash_table.DataTable(
            data=term_data,
            columns=[{"name": c, "id": c} for c in term_data[0].keys()],
            style_header={
                "backgroundColor": COLORS["bg_secondary"],
                "color": COLORS["text_secondary"],
                "fontWeight": "600", "fontSize": "11px",
                "textTransform": "uppercase", "letterSpacing": "1px",
                "border": f"1px solid {COLORS['border_subtle']}",
                "padding": "10px",
            },
            style_cell={
                "backgroundColor": COLORS["bg_card"],
                "color": COLORS["text_primary"],
                "fontSize": "12px",
                "fontFamily": "'JetBrains Mono', monospace",
                "border": f"1px solid {COLORS['border_subtle']}",
                "padding": "8px 12px",
                "textAlign": "center",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"},
                 "backgroundColor": COLORS["bg_secondary"]},
            ],
        )

        return fig, stats, term_table
