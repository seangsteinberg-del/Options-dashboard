"""
Risk & Greeks Panel
===================
Portfolio-level risk dashboard with scenario analysis, Greeks grid,
P&L attribution, and exposure breakdowns.
"""

from dash import html, dcc, dash_table, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from core.pricing import (
    compute_all_greeks, bs_price, scenario_grid, aggregate_portfolio_greeks,
)
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE, BUTTON_STYLE,
)


# ── Default sample portfolio ─────────────────────────────────────────────
DEFAULT_POSITIONS = [
    {"strike": 95,  "expiry": 0.25, "vol": 0.22, "option_type": "put",  "quantity": -10, "entry_price": 1.80, "multiplier": 100},
    {"strike": 100, "expiry": 0.25, "vol": 0.20, "option_type": "call", "quantity": 20,  "entry_price": 3.50, "multiplier": 100},
    {"strike": 105, "expiry": 0.25, "vol": 0.19, "option_type": "call", "quantity": -15, "entry_price": 1.60, "multiplier": 100},
    {"strike": 100, "expiry": 0.50, "vol": 0.21, "option_type": "put",  "quantity": 5,   "entry_price": 4.20, "multiplier": 100},
    {"strike": 110, "expiry": 0.50, "vol": 0.18, "option_type": "call", "quantity": -5,  "entry_price": 2.10, "multiplier": 100},
    {"strike": 90,  "expiry": 0.75, "vol": 0.25, "option_type": "put",  "quantity": 8,   "entry_price": 3.90, "multiplier": 100},
    {"strike": 100, "expiry": 1.00, "vol": 0.20, "option_type": "call", "quantity": 10,  "entry_price": 7.50, "multiplier": 100},
    {"strike": 95,  "expiry": 0.10, "vol": 0.24, "option_type": "call", "quantity": -8,  "entry_price": 6.20, "multiplier": 100},
]


def layout():
    return html.Div([
        # ── Market Parameters ─────────────────────────────────────
        html.Div([
            html.Div("PORTFOLIO RISK PARAMETERS", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Label("SPOT PRICE", style=LABEL_STYLE),
                    dcc.Input(id="risk-spot", type="number", value=100, step=0.5,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "140px", "marginRight": "16px"}),
                html.Div([
                    html.Label("RISK-FREE RATE (%)", style=LABEL_STYLE),
                    dcc.Input(id="risk-rate", type="number", value=5.0, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "140px", "marginRight": "16px"}),
                html.Div([
                    html.Label("DIVIDEND YIELD (%)", style=LABEL_STYLE),
                    dcc.Input(id="risk-div", type="number", value=1.5, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "140px", "marginRight": "16px"}),
                html.Div([
                    html.Label("SCENARIO METRIC", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="risk-metric",
                        options=[
                            {"label": "P&L", "value": "pnl"},
                            {"label": "Price", "value": "price"},
                            {"label": "Delta", "value": "delta"},
                            {"label": "Gamma", "value": "gamma"},
                            {"label": "Theta", "value": "theta"},
                            {"label": "Vega", "value": "vega"},
                        ],
                        value="pnl", clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "160px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Portfolio Summary Stats ───────────────────────────────
        html.Div(id="risk-summary-stats", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap",
        }),

        # ── Position Grid ─────────────────────────────────────────
        html.Div([
            html.Div("POSITION GRID", style=CARD_HEADER_STYLE),
            html.Div(id="risk-position-grid"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Charts Row ────────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Graph(id="risk-exposure-chart", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"},
                className="dashboard-card"),
            html.Div([
                dcc.Graph(id="risk-pnl-chart", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"},
                className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Scenario Heatmap ──────────────────────────────────────
        html.Div([
            html.Div("SCENARIO ANALYSIS — PORTFOLIO LEVEL", style=CARD_HEADER_STYLE),
            dcc.Graph(id="risk-scenario-heatmap", style={"height": "420px"},
                      config={"displayModeBar": True}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Greeks Sensitivity ────────────────────────────────────
        html.Div([
            html.Div("GREEKS SENSITIVITY ACROSS SPOT", style=CARD_HEADER_STYLE),
            dcc.Graph(id="risk-greeks-sensitivity", style={"height": "380px"},
                      config={"displayModeBar": True}),
        ], style=CARD_STYLE, className="dashboard-card"),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("risk-summary-stats", "children"),
         Output("risk-position-grid", "children"),
         Output("risk-exposure-chart", "figure"),
         Output("risk-pnl-chart", "figure"),
         Output("risk-scenario-heatmap", "figure"),
         Output("risk-greeks-sensitivity", "figure")],
        [Input("risk-spot", "value"),
         Input("risk-rate", "value"),
         Input("risk-div", "value"),
         Input("risk-metric", "value")],
    )
    def update_risk(spot, rate, div_yield, metric):
        S = spot or 100
        r = (rate or 5.0) / 100
        q = (div_yield or 1.5) / 100
        positions = DEFAULT_POSITIONS
        tpl = CHART_TEMPLATE["layout"]

        # ── Aggregate Greeks ──────────────────────────────────────
        totals = aggregate_portfolio_greeks(positions, S, r, q)

        # ── Summary Stat Boxes ────────────────────────────────────
        def stat_box(label, value, fmt=".2f", prefix="", color=COLORS["accent_cyan"]):
            txt = f"{prefix}{value:{fmt}}"
            return html.Div([
                html.Div(txt, className="stat-value",
                         style={"color": color, "fontSize": "18px"}),
                html.Div(label, className="stat-label"),
            ], style={**STAT_BOX_STYLE, "flex": "1", "minWidth": "110px"})

        pnl_color = COLORS["pnl_profit"] if totals["pnl"] >= 0 else COLORS["pnl_loss"]
        stats = [
            stat_box("TOTAL P&L", totals["pnl"], ".2f", "$", pnl_color),
            stat_box("NET DELTA", totals["delta"], ".1f", "", COLORS["accent_cyan"]),
            stat_box("NET GAMMA", totals["gamma"], ".2f", "", COLORS["accent_blue"]),
            stat_box("NET THETA", totals["theta"], ".2f", "$", COLORS["accent_orange"]),
            stat_box("NET VEGA", totals["vega"], ".2f", "$", COLORS["accent_purple"]),
            stat_box("NET VANNA", totals["vanna"], ".2f", "", COLORS["accent_pink"]),
            stat_box("NOTIONAL", totals["notional"], ",.0f", "$", COLORS["text_secondary"]),
        ]

        # ── Position Grid ─────────────────────────────────────────
        grid_data = []
        for pos in positions:
            greeks = compute_all_greeks(S, pos["strike"], pos["expiry"], r, q,
                                        pos["vol"], pos["option_type"])
            qty = pos["quantity"]
            mult = pos["multiplier"]
            mkt_val = greeks["price"] * qty * mult
            pnl = (greeks["price"] - pos["entry_price"]) * qty * mult
            grid_data.append({
                "Type": pos["option_type"].upper(),
                "Strike": pos["strike"],
                "Expiry": f"{pos['expiry']:.2f}y",
                "IV": f"{pos['vol'] * 100:.1f}%",
                "Qty": qty,
                "Entry": f"${pos['entry_price']:.2f}",
                "Mark": f"${greeks['price']:.2f}",
                "Mkt Val": f"${mkt_val:,.0f}",
                "P&L": f"${pnl:,.0f}",
                "Delta": f"{greeks['delta'] * qty * mult:.1f}",
                "Gamma": f"{greeks['gamma'] * qty * mult:.2f}",
                "Theta": f"${greeks['theta'] * qty * mult:.2f}",
                "Vega": f"${greeks['vega'] * qty * mult:.2f}",
            })

        pos_table = dash_table.DataTable(
            data=grid_data,
            columns=[{"name": c, "id": c} for c in grid_data[0].keys()],
            style_header={
                "backgroundColor": COLORS["bg_secondary"],
                "color": COLORS["text_secondary"],
                "fontWeight": "600", "fontSize": "10px",
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
                "padding": "6px 8px",
                "textAlign": "center",
                "minWidth": "70px",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg_secondary"]},
                {"if": {"filter_query": "{Qty} > 0", "column_id": "Qty"},
                 "color": COLORS["accent_green"]},
                {"if": {"filter_query": "{Qty} < 0", "column_id": "Qty"},
                 "color": COLORS["accent_red"]},
            ],
            sort_action="native",
            page_size=20,
        )

        # ── Exposure Breakdown (bar chart) ────────────────────────
        exp_fig = make_subplots(rows=1, cols=2, subplot_titles=("Delta by Strike", "Vega by Expiry"))

        # Delta by strike
        strike_delta = {}
        strike_gamma = {}
        for pos in positions:
            g = compute_all_greeks(S, pos["strike"], pos["expiry"], r, q,
                                    pos["vol"], pos["option_type"])
            k = pos["strike"]
            strike_delta[k] = strike_delta.get(k, 0) + g["delta"] * pos["quantity"] * pos["multiplier"]
            strike_gamma[k] = strike_gamma.get(k, 0) + g["gamma"] * pos["quantity"] * pos["multiplier"]

        strikes_sorted = sorted(strike_delta.keys())
        delta_vals = [strike_delta[k] for k in strikes_sorted]
        delta_colors = [COLORS["accent_green"] if v >= 0 else COLORS["accent_red"] for v in delta_vals]

        exp_fig.add_trace(go.Bar(
            x=[str(k) for k in strikes_sorted], y=delta_vals,
            marker_color=delta_colors, name="Delta",
            hovertemplate="K=%{x}<br>Delta=%{y:.1f}<extra></extra>",
        ), row=1, col=1)

        # Vega by expiry
        expiry_vega = {}
        for pos in positions:
            g = compute_all_greeks(S, pos["strike"], pos["expiry"], r, q,
                                    pos["vol"], pos["option_type"])
            t_key = f"{pos['expiry']:.2f}y"
            expiry_vega[t_key] = expiry_vega.get(t_key, 0) + g["vega"] * pos["quantity"] * pos["multiplier"]

        exp_sorted = sorted(expiry_vega.keys())
        vega_vals = [expiry_vega[t] for t in exp_sorted]
        vega_colors = [COLORS["accent_purple"] if v >= 0 else COLORS["accent_orange"] for v in vega_vals]

        exp_fig.add_trace(go.Bar(
            x=exp_sorted, y=vega_vals,
            marker_color=vega_colors, name="Vega",
            hovertemplate="T=%{x}<br>Vega=$%{y:.2f}<extra></extra>",
        ), row=1, col=2)

        exp_fig.update_layout(
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            showlegend=False,
            margin=dict(l=50, r=20, t=40, b=40),
            hoverlabel=tpl["hoverlabel"],
        )
        exp_fig.update_xaxes(gridcolor=COLORS["border_subtle"])
        exp_fig.update_yaxes(gridcolor=COLORS["border_subtle"])
        for ann in exp_fig.layout.annotations:
            ann.font.color = COLORS["text_primary"]
            ann.font.size = 12

        # ── Portfolio P&L vs Spot ─────────────────────────────────
        spot_range = np.linspace(S * 0.75, S * 1.25, 200)
        portfolio_pnl = np.zeros_like(spot_range)

        for pos in positions:
            for j, s in enumerate(spot_range):
                price = bs_price(s, pos["strike"], pos["expiry"], r, q,
                                 pos["vol"], pos["option_type"])
                portfolio_pnl[j] += (price - pos["entry_price"]) * pos["quantity"] * pos["multiplier"]

        pnl_fig = go.Figure()
        pnl_fig.add_trace(go.Scatter(
            x=spot_range, y=portfolio_pnl, mode="lines",
            line=dict(color=COLORS["accent_cyan"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.06)",
            hovertemplate="Spot: %{x:.1f}<br>P&L: $%{y:,.0f}<extra></extra>",
        ))
        pnl_fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
        pnl_fig.add_vline(x=S, line=dict(color=COLORS["accent_orange"], width=1, dash="dash"))

        pnl_fig.update_layout(
            title=dict(text="Portfolio P&L vs Spot", font=dict(
                color=COLORS["text_primary"], size=13)),
            xaxis_title="Spot Price",
            yaxis_title="P&L ($)",
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            margin=dict(l=60, r=20, t=40, b=40),
            hoverlabel=tpl["hoverlabel"],
            xaxis=dict(gridcolor=COLORS["border_subtle"]),
            yaxis=dict(gridcolor=COLORS["border_subtle"]),
        )

        # ── Scenario Heatmap ──────────────────────────────────────
        spot_shocks = np.linspace(-0.20, 0.20, 17)
        vol_shocks = np.linspace(-0.10, 0.10, 17)

        scen_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))
        for vi, dv in enumerate(vol_shocks):
            for si, ds in enumerate(spot_shocks):
                s_new = S * (1 + ds)
                val = 0
                for pos in positions:
                    sig_new = max(pos["vol"] + dv, 0.01)
                    if metric in ("pnl", "price"):
                        p = bs_price(s_new, pos["strike"], pos["expiry"], r, q,
                                     sig_new, pos["option_type"])
                        if metric == "pnl":
                            val += (p - pos["entry_price"]) * pos["quantity"] * pos["multiplier"]
                        else:
                            val += p * pos["quantity"] * pos["multiplier"]
                    else:
                        g = compute_all_greeks(s_new, pos["strike"], pos["expiry"],
                                               r, q, sig_new, pos["option_type"])
                        val += g.get(metric, 0) * pos["quantity"] * pos["multiplier"]
                scen_matrix[vi, si] = val

        scen_fig = go.Figure(go.Heatmap(
            x=[f"{ds:+.0%}" for ds in spot_shocks],
            y=[f"{dv:+.0%}" for dv in vol_shocks],
            z=scen_matrix,
            colorscale=[[0, COLORS["accent_red"]], [0.5, COLORS["bg_card"]],
                        [1, COLORS["accent_green"]]],
            zmid=0,
            colorbar=dict(
                title=dict(text=metric.upper(), font=dict(color=COLORS["text_secondary"])),
                tickfont=dict(color=COLORS["text_secondary"]),
            ),
            hovertemplate="Spot: %{x}<br>Vol: %{y}<br>Value: %{z:,.2f}<extra></extra>",
        ))
        scen_fig.update_layout(
            xaxis_title="Spot Shock",
            yaxis_title="Vol Shock",
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            margin=dict(l=60, r=20, t=20, b=50),
            hoverlabel=tpl["hoverlabel"],
        )

        # ── Greeks Sensitivity ────────────────────────────────────
        sens_fig = make_subplots(rows=2, cols=2,
                                 subplot_titles=("Portfolio Delta", "Portfolio Gamma",
                                                 "Portfolio Theta ($/day)", "Portfolio Vega ($)"))

        spot_grid = np.linspace(S * 0.8, S * 1.2, 150)
        port_delta = np.zeros_like(spot_grid)
        port_gamma = np.zeros_like(spot_grid)
        port_theta = np.zeros_like(spot_grid)
        port_vega = np.zeros_like(spot_grid)

        for pos in positions:
            for j, s in enumerate(spot_grid):
                g = compute_all_greeks(s, pos["strike"], pos["expiry"], r, q,
                                        pos["vol"], pos["option_type"])
                m = pos["quantity"] * pos["multiplier"]
                port_delta[j] += g["delta"] * m
                port_gamma[j] += g["gamma"] * m
                port_theta[j] += g["theta"] * m
                port_vega[j] += g["vega"] * m

        for row, col, data, name, color in [
            (1, 1, port_delta, "Delta", COLORS["accent_cyan"]),
            (1, 2, port_gamma, "Gamma", COLORS["accent_blue"]),
            (2, 1, port_theta, "Theta", COLORS["accent_orange"]),
            (2, 2, port_vega, "Vega", COLORS["accent_purple"]),
        ]:
            sens_fig.add_trace(go.Scatter(
                x=spot_grid, y=data, mode="lines",
                line=dict(color=color, width=2), name=name,
                showlegend=False,
            ), row=row, col=col)
            sens_fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"),
                               row=row, col=col)

        sens_fig.update_layout(
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            margin=dict(l=50, r=20, t=40, b=40),
            height=380,
            hoverlabel=tpl["hoverlabel"],
        )
        sens_fig.update_xaxes(gridcolor=COLORS["border_subtle"])
        sens_fig.update_yaxes(gridcolor=COLORS["border_subtle"])
        for ann in sens_fig.layout.annotations:
            ann.font.color = COLORS["text_primary"]
            ann.font.size = 11

        return stats, pos_table, exp_fig, pnl_fig, scen_fig, sens_fig
