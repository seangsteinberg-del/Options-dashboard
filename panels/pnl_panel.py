"""
P&L Attribution Panel
=====================
Greeks-based P&L decomposition (delta, gamma, theta, vega, residual),
waterfall charts, by-ticker breakdown, and simulated daily P&L history.
"""

from dash import html, dcc, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from core.portfolio import get_books, get_all_positions, get_positions, pnl_attribution
from core.bloomberg import get_spot_prices
from core.simulator import generate_portfolio_history
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, make_stat_style,
)


def layout():
    books = get_books()
    return html.Div([
        # ── Controls ─────────────────────────────────────────────
        html.Div([
            html.Div("P&L ATTRIBUTION", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Label("BOOK", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="pnl-book",
                        options=[{"label": b, "value": b} for b in books] + [{"label": "ALL BOOKS", "value": "ALL"}],
                        value="MAIN", clearable=False, style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "140px", "marginRight": "14px"}),
                html.Div([
                    html.Label("SPOT SHOCK (%)", style=LABEL_STYLE),
                    dcc.Input(id="pnl-shock", type="number", value=1.0, step=0.5,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "120px", "marginRight": "14px"}),
                html.Div([
                    html.Label("RATE (%)", style=LABEL_STYLE),
                    dcc.Input(id="pnl-rate", type="number", value=5.0, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "100px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Stats Row ────────────────────────────────────────────
        html.Div(id="pnl-stats", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap",
        }),

        # ── Row: Waterfall + By-Ticker ───────────────────────────
        html.Div([
            html.Div([dcc.Graph(id="pnl-waterfall", style={"height": "400px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
            html.Div([dcc.Graph(id="pnl-ticker-decomp", style={"height": "400px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Row: Daily P&L + Cumulative ──────────────────────────
        html.Div([
            html.Div([dcc.Graph(id="pnl-daily", style={"height": "380px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
            html.Div([dcc.Graph(id="pnl-cumulative", style={"height": "380px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("pnl-stats", "children"),
         Output("pnl-waterfall", "figure"),
         Output("pnl-ticker-decomp", "figure"),
         Output("pnl-daily", "figure"),
         Output("pnl-cumulative", "figure")],
        [Input("pnl-book", "value"),
         Input("pnl-shock", "value"),
         Input("pnl-rate", "value")],
    )
    def update_pnl(book, shock_pct, rate):
        shock_pct = shock_pct or 1.0
        rate = (rate or 5.0) / 100
        q = 0.015

        # Get positions
        if book == "ALL":
            positions = get_all_positions()
        else:
            positions = get_positions(book or "MAIN")

        if not positions:
            empty_fig = go.Figure(layout={**CHART_TEMPLATE["layout"], "title": "No positions"})
            empty_stats = [html.Div("No positions in book", style={"color": COLORS["text_muted"]})]
            return empty_stats, empty_fig, empty_fig, empty_fig, empty_fig

        # Spot prices
        tickers = list(set(p.get("ticker", "SPY") for p in positions))
        spot_data = get_spot_prices(tickers)
        S_old = {t: d.get("price", 100) for t, d in spot_data.items()}
        S_new = {t: p * (1 + shock_pct / 100) for t, p in S_old.items()}

        # P&L attribution
        attr = pnl_attribution(positions, S_old, S_new, rate, q)

        # ── Stat boxes ───────────────────────────────────────────
        def sbox(label, value, color):
            return html.Div([
                html.Div(f"${value:+,.0f}", style={
                    "color": color, "fontSize": "20px", "fontWeight": "700",
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
                html.Div(label, style={
                    "color": COLORS["text_muted"], "fontSize": "9px",
                    "textTransform": "uppercase", "letterSpacing": "1px", "marginTop": "4px",
                }),
            ], style={**make_stat_style(color), "flex": "1", "minWidth": "100px"})

        total = attr["total_pnl"]
        total_color = COLORS["pnl_profit"] if total >= 0 else COLORS["pnl_loss"]

        stats = [
            sbox("Total P&L", total, total_color),
            sbox("Delta P&L", attr["delta_pnl"], COLORS["accent_cyan"]),
            sbox("Gamma P&L", attr["gamma_pnl"], COLORS["accent_blue"]),
            sbox("Theta P&L", attr["theta_pnl"], COLORS["accent_orange"]),
            sbox("Vega P&L", attr["vega_pnl"], COLORS["accent_purple"]),
            sbox("Residual", attr["cross_pnl"], COLORS["text_secondary"]),
        ]

        # ── Waterfall chart ──────────────────────────────────────
        components = ["Delta", "Gamma", "Theta", "Vega", "Cross", "Total"]
        values = [attr["delta_pnl"], attr["gamma_pnl"], attr["theta_pnl"],
                  attr["vega_pnl"], attr["cross_pnl"], attr["total_pnl"]]
        colors = [COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_orange"],
                  COLORS["accent_purple"], COLORS["text_secondary"], total_color]

        wf = go.Figure()
        wf.add_trace(go.Waterfall(
            x=components,
            y=values,
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            connector={"line": {"color": COLORS["border"], "width": 1}},
            increasing={"marker": {"color": COLORS["pnl_profit"]}},
            decreasing={"marker": {"color": COLORS["pnl_loss"]}},
            totals={"marker": {"color": total_color}},
            textposition="outside",
            text=[f"${v:+,.0f}" for v in values],
            textfont={"size": 10, "color": COLORS["text_secondary"]},
        ))
        wf.update_layout(
            **CHART_TEMPLATE["layout"],
            title=f"P&L WATERFALL — {shock_pct:+.1f}% Spot Shock",
            showlegend=False,
        )

        # ── By-ticker decomposition ──────────────────────────────
        by_tk = attr.get("by_ticker", {})
        tk_names = list(by_tk.keys())

        td = go.Figure()
        for comp, key, color in [
            ("Delta", "delta", COLORS["accent_cyan"]),
            ("Gamma", "gamma", COLORS["accent_blue"]),
            ("Theta", "theta", COLORS["accent_orange"]),
            ("Vega", "vega", COLORS["accent_purple"]),
        ]:
            vals = [by_tk[t].get(key, 0) for t in tk_names]
            td.add_trace(go.Bar(
                name=comp, x=tk_names, y=vals,
                marker_color=color, opacity=0.85,
            ))

        td.update_layout(
            **CHART_TEMPLATE["layout"],
            title="P&L BY TICKER — GREEKS DECOMPOSITION",
            barmode="group",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center",
                       font={"size": 10, "color": COLORS["text_secondary"]}),
        )

        # ── Daily P&L (simulated) ────────────────────────────────
        history = generate_portfolio_history()
        daily_pnl = history["daily_pnl"]
        days = list(range(len(daily_pnl)))

        dpf = go.Figure()
        bar_colors = [COLORS["pnl_profit"] if v >= 0 else COLORS["pnl_loss"] for v in daily_pnl]
        dpf.add_trace(go.Bar(
            x=days, y=daily_pnl,
            marker_color=bar_colors,
            name="Daily P&L",
        ))
        dpf.update_layout(
            **CHART_TEMPLATE["layout"],
            title="DAILY P&L (60-DAY SIMULATION)",
            xaxis_title="Trading Day",
            yaxis_title="P&L ($)",
        )

        # ── Cumulative P&L ───────────────────────────────────────
        cum_pnl = history["cumulative_pnl"]

        cf = go.Figure()
        cum_color = COLORS["pnl_profit"] if cum_pnl[-1] >= 0 else COLORS["pnl_loss"]
        cf.add_trace(go.Scatter(
            x=days, y=cum_pnl,
            mode="lines", name="Cumulative P&L",
            line=dict(color=cum_color, width=2),
            fill="tozeroy",
            fillcolor=f"{cum_color}15",
        ))
        cf.add_hline(y=0, line_dash="dash", line_color=COLORS["border"])
        cf.update_layout(
            **CHART_TEMPLATE["layout"],
            title="CUMULATIVE P&L",
            xaxis_title="Trading Day",
            yaxis_title="Cumulative P&L ($)",
        )

        return stats, wf, td, dpf, cf
