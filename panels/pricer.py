"""
Options Pricer Panel
====================
Single-leg and multi-leg strategy pricer with payoff diagrams,
full Greeks display, and implied vol solver.
"""

from dash import html, dcc, dash_table, callback, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np

from core.pricing import bs_price, compute_all_greeks, implied_vol
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE, BUTTON_STYLE, BUTTON_SUCCESS_STYLE,
)


# ── Pre-defined strategies ────────────────────────────────────────────────
STRATEGIES = {
    "Single Call": [{"type": "call", "strike_offset": 0, "qty": 1}],
    "Single Put": [{"type": "put", "strike_offset": 0, "qty": 1}],
    "Bull Call Spread": [
        {"type": "call", "strike_offset": -5, "qty": 1},
        {"type": "call", "strike_offset": 5, "qty": -1},
    ],
    "Bear Put Spread": [
        {"type": "put", "strike_offset": 5, "qty": 1},
        {"type": "put", "strike_offset": -5, "qty": -1},
    ],
    "Long Straddle": [
        {"type": "call", "strike_offset": 0, "qty": 1},
        {"type": "put", "strike_offset": 0, "qty": 1},
    ],
    "Long Strangle": [
        {"type": "call", "strike_offset": 5, "qty": 1},
        {"type": "put", "strike_offset": -5, "qty": 1},
    ],
    "Iron Condor": [
        {"type": "put", "strike_offset": -10, "qty": 1},
        {"type": "put", "strike_offset": -5, "qty": -1},
        {"type": "call", "strike_offset": 5, "qty": -1},
        {"type": "call", "strike_offset": 10, "qty": 1},
    ],
    "Butterfly": [
        {"type": "call", "strike_offset": -5, "qty": 1},
        {"type": "call", "strike_offset": 0, "qty": -2},
        {"type": "call", "strike_offset": 5, "qty": 1},
    ],
    "Calendar Spread": [
        {"type": "call", "strike_offset": 0, "qty": -1},  # short near
        {"type": "call", "strike_offset": 0, "qty": 1},   # long far (different T)
    ],
    "Ratio Spread 1x2": [
        {"type": "call", "strike_offset": 0, "qty": 1},
        {"type": "call", "strike_offset": 10, "qty": -2},
    ],
    "Jade Lizard": [
        {"type": "put", "strike_offset": -10, "qty": -1},
        {"type": "call", "strike_offset": 5, "qty": -1},
        {"type": "call", "strike_offset": 10, "qty": 1},
    ],
}


def layout():
    return html.Div([
        # ── Input Controls ────────────────────────────────────────
        html.Div([
            html.Div("OPTIONS PRICER", style=CARD_HEADER_STYLE),

            html.Div([
                # Market Data Column
                html.Div([
                    html.Div("MARKET DATA", style={
                        **LABEL_STYLE, "fontSize": "12px", "color": COLORS["accent_cyan"],
                        "marginBottom": "10px",
                    }),
                    html.Label("SPOT PRICE", style=LABEL_STYLE),
                    dcc.Input(id="pricer-spot", type="number", value=100, step=0.5,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("RISK-FREE RATE (%)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-rate", type="number", value=5.0, step=0.1,
                              min=0, max=50,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("DIVIDEND YIELD (%)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-div", type="number", value=1.5, step=0.1,
                              min=0, max=50,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("VOLATILITY (%)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-vol", type="number", value=20.0, step=0.5,
                              min=0.1, max=500,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("TIME TO EXPIRY (DAYS)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-dte", type="number", value=30, step=1,
                              min=0, max=3650,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "200px", "marginRight": "24px"}),

                # Strategy Column
                html.Div([
                    html.Div("STRATEGY", style={
                        **LABEL_STYLE, "fontSize": "12px", "color": COLORS["accent_cyan"],
                        "marginBottom": "10px",
                    }),
                    html.Label("PRESET", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="pricer-strategy",
                        options=[{"label": k, "value": k} for k in STRATEGIES],
                        value="Single Call",
                        clearable=False,
                        style={"marginBottom": "10px", "fontSize": "12px"},
                    ),

                    html.Label("BASE STRIKE", style=LABEL_STYLE),
                    dcc.Input(id="pricer-strike", type="number", value=100, step=0.5,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("CONTRACT MULTIPLIER", style=LABEL_STYLE),
                    dcc.Input(id="pricer-multiplier", type="number", value=100, step=1,
                              min=1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("NUMBER OF CONTRACTS", style=LABEL_STYLE),
                    dcc.Input(id="pricer-contracts", type="number", value=1, step=1,
                              min=1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("PAYOFF RANGE (%)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-payoff-range", type="number", value=30, step=5,
                              min=5, max=100,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "200px"}),
            ], style={"display": "flex", "flexWrap": "wrap"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Greeks Display ────────────────────────────────────────
        html.Div(id="pricer-greeks-row", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap",
        }),

        # ── Leg Breakdown Table ───────────────────────────────────
        html.Div([
            html.Div("LEG BREAKDOWN", style=CARD_HEADER_STYLE),
            html.Div(id="pricer-legs-table"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Charts Row ────────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Graph(id="pricer-payoff-chart", style={"height": "400px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"},
                className="dashboard-card"),

            html.Div([
                dcc.Graph(id="pricer-greeks-chart", style={"height": "400px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"},
                className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── P&L Heatmap (Spot vs Vol) ────────────────────────────
        html.Div([
            html.Div("P&L HEATMAP — SPOT vs VOLATILITY", style=CARD_HEADER_STYLE),
            dcc.Graph(id="pricer-pnl-heatmap", style={"height": "400px"},
                      config={"displayModeBar": True}),
        ], style=CARD_STYLE, className="dashboard-card"),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("pricer-greeks-row", "children"),
         Output("pricer-legs-table", "children"),
         Output("pricer-payoff-chart", "figure"),
         Output("pricer-greeks-chart", "figure"),
         Output("pricer-pnl-heatmap", "figure")],
        [Input("pricer-spot", "value"),
         Input("pricer-rate", "value"),
         Input("pricer-div", "value"),
         Input("pricer-vol", "value"),
         Input("pricer-dte", "value"),
         Input("pricer-strategy", "value"),
         Input("pricer-strike", "value"),
         Input("pricer-multiplier", "value"),
         Input("pricer-contracts", "value"),
         Input("pricer-payoff-range", "value")],
    )
    def update_pricer(spot, rate, div_yield, vol, dte, strategy, strike,
                      multiplier, contracts, payoff_range):
        S = spot or 100
        r = (rate or 5.0) / 100
        q = (div_yield or 1.5) / 100
        sigma = (vol or 20.0) / 100
        T = (dte or 30) / 365.0
        K_base = strike or 100
        mult = multiplier or 100
        n_contracts = contracts or 1
        prange = (payoff_range or 30) / 100.0

        legs = STRATEGIES.get(strategy, STRATEGIES["Single Call"])
        tpl = CHART_TEMPLATE["layout"]

        # ── Compute per-leg Greeks ────────────────────────────────
        leg_data = []
        total_greeks = {"price": 0, "delta": 0, "gamma": 0, "theta": 0,
                        "vega": 0, "rho": 0, "vanna": 0, "volga": 0}

        for i, leg in enumerate(legs):
            K = K_base + leg["strike_offset"]
            qty = leg["qty"] * n_contracts
            otype = leg["type"]
            greeks = compute_all_greeks(S, K, T, r, q, sigma, otype)

            for g in total_greeks:
                total_greeks[g] += greeks[g] * qty * (mult if g != "price" else mult)

            direction = "LONG" if qty > 0 else "SHORT"
            leg_data.append({
                "Leg": f"#{i+1}",
                "Direction": direction,
                "Type": otype.upper(),
                "Strike": f"{K:.1f}",
                "Qty": f"{abs(qty)}",
                "Price": f"${greeks['price']:.4f}",
                "Delta": f"{greeks['delta'] * qty:.4f}",
                "Gamma": f"{greeks['gamma'] * qty:.6f}",
                "Theta": f"{greeks['theta'] * qty:.4f}",
                "Vega": f"{greeks['vega'] * qty:.4f}",
            })

        # Net premium
        net_premium = total_greeks["price"]

        # ── Greeks Stat Boxes ─────────────────────────────────────
        def greek_box(label, value, fmt=".4f", color=COLORS["accent_cyan"]):
            return html.Div([
                html.Div(f"{value:{fmt}}", className="stat-value",
                         style={"color": color, "fontSize": "18px"}),
                html.Div(label, className="stat-label"),
            ], style={**STAT_BOX_STYLE, "flex": "1", "minWidth": "100px"})

        pnl_color = COLORS["pnl_profit"] if net_premium >= 0 else COLORS["pnl_loss"]
        greeks_row = [
            greek_box("NET PREMIUM", net_premium, ".2f", pnl_color),
            greek_box("DELTA", total_greeks["delta"], ".4f", COLORS["accent_cyan"]),
            greek_box("GAMMA", total_greeks["gamma"], ".6f", COLORS["accent_blue"]),
            greek_box("THETA", total_greeks["theta"], ".4f", COLORS["accent_orange"]),
            greek_box("VEGA", total_greeks["vega"], ".4f", COLORS["accent_purple"]),
            greek_box("RHO", total_greeks["rho"], ".4f", COLORS["text_secondary"]),
        ]

        # ── Legs Table ────────────────────────────────────────────
        legs_table = dash_table.DataTable(
            data=leg_data,
            columns=[{"name": c, "id": c} for c in leg_data[0].keys()],
            style_header={
                "backgroundColor": COLORS["bg_secondary"],
                "color": COLORS["text_secondary"],
                "fontWeight": "600", "fontSize": "11px",
                "textTransform": "uppercase", "letterSpacing": "1px",
                "border": f"1px solid {COLORS['border_subtle']}",
            },
            style_cell={
                "backgroundColor": COLORS["bg_card"],
                "color": COLORS["text_primary"],
                "fontSize": "12px",
                "fontFamily": "'JetBrains Mono', monospace",
                "border": f"1px solid {COLORS['border_subtle']}",
                "padding": "8px 10px",
                "textAlign": "center",
            },
            style_data_conditional=[
                {"if": {"filter_query": "{Direction} = LONG"},
                 "color": COLORS["accent_green"]},
                {"if": {"filter_query": "{Direction} = SHORT"},
                 "color": COLORS["accent_red"]},
                {"if": {"row_index": "odd"},
                 "backgroundColor": COLORS["bg_secondary"]},
            ],
        )

        # ── Payoff Diagram ────────────────────────────────────────
        spot_range = np.linspace(S * (1 - prange), S * (1 + prange), 300)
        payoff_expiry = np.zeros_like(spot_range)
        payoff_now = np.zeros_like(spot_range)

        for leg in legs:
            K = K_base + leg["strike_offset"]
            qty = leg["qty"] * n_contracts * mult
            otype = leg["type"]

            for j, s in enumerate(spot_range):
                # At expiry
                if otype == "call":
                    payoff_expiry[j] += max(s - K, 0) * qty
                else:
                    payoff_expiry[j] += max(K - s, 0) * qty

                # Current (with time value)
                payoff_now[j] += bs_price(s, K, T, r, q, sigma, otype) * qty

        # Subtract initial cost
        payoff_expiry -= net_premium
        payoff_now -= net_premium

        payoff_fig = go.Figure()
        payoff_fig.add_trace(go.Scatter(
            x=spot_range, y=payoff_expiry, mode="lines",
            name="At Expiry", line=dict(color=COLORS["accent_cyan"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.08)",
        ))
        payoff_fig.add_trace(go.Scatter(
            x=spot_range, y=payoff_now, mode="lines",
            name="Current", line=dict(color=COLORS["accent_purple"], width=2, dash="dash"),
        ))
        # Breakeven line
        payoff_fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
        # Spot marker
        payoff_fig.add_vline(x=S, line=dict(color=COLORS["accent_orange"], width=1, dash="dash"),
                             annotation_text=f"Spot {S:.0f}",
                             annotation_font=dict(color=COLORS["accent_orange"], size=10))

        payoff_fig.update_layout(
            title=dict(text=f"Payoff Diagram — {strategy}", font=dict(
                color=COLORS["text_primary"], size=14)),
            xaxis_title="Underlying Price",
            yaxis_title="P&L ($)",
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            margin=dict(l=50, r=20, t=45, b=40),
            legend=dict(font=dict(color=COLORS["text_secondary"], size=11),
                        bgcolor="rgba(0,0,0,0)"),
            hoverlabel=tpl["hoverlabel"],
        )

        # ── Greeks vs Spot Chart ──────────────────────────────────
        greeks_chart = go.Figure()
        greek_names = ["delta", "gamma", "theta", "vega"]
        greek_colors = [COLORS["accent_cyan"], COLORS["accent_blue"],
                        COLORS["accent_orange"], COLORS["accent_purple"]]

        spot_grid = np.linspace(S * (1 - prange), S * (1 + prange), 200)
        for gi, gname in enumerate(greek_names):
            vals = np.zeros_like(spot_grid)
            for leg in legs:
                K = K_base + leg["strike_offset"]
                qty = leg["qty"] * n_contracts * mult
                otype = leg["type"]
                for j, s in enumerate(spot_grid):
                    g = compute_all_greeks(s, K, T, r, q, sigma, otype)
                    vals[j] += g[gname] * qty
            greeks_chart.add_trace(go.Scatter(
                x=spot_grid, y=vals, mode="lines",
                name=gname.capitalize(),
                line=dict(color=greek_colors[gi], width=2),
                visible=True if gname == "delta" else "legendonly",
            ))

        greeks_chart.update_layout(
            title=dict(text="Greeks vs Spot", font=dict(
                color=COLORS["text_primary"], size=14)),
            xaxis_title="Underlying Price",
            yaxis_title="Greek Value",
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            margin=dict(l=50, r=20, t=45, b=40),
            legend=dict(font=dict(color=COLORS["text_secondary"], size=11),
                        bgcolor="rgba(0,0,0,0)"),
            hoverlabel=tpl["hoverlabel"],
        )

        # ── P&L Heatmap ──────────────────────────────────────────
        spot_shocks = np.linspace(-prange, prange, 21)
        vol_shocks = np.linspace(-0.15, 0.15, 21)
        pnl_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))

        for vi, dv in enumerate(vol_shocks):
            for si, ds in enumerate(spot_shocks):
                s_new = S * (1 + ds)
                sig_new = max(sigma + dv, 0.01)
                pnl = 0
                for leg in legs:
                    K = K_base + leg["strike_offset"]
                    qty = leg["qty"] * n_contracts * mult
                    otype = leg["type"]
                    pnl += bs_price(s_new, K, T, r, q, sig_new, otype) * qty
                pnl_matrix[vi, si] = pnl - net_premium

        heatmap_fig = go.Figure(go.Heatmap(
            x=[f"{ds:+.0%}" for ds in spot_shocks],
            y=[f"{dv:+.0%}" for dv in vol_shocks],
            z=pnl_matrix,
            colorscale=[
                [0, COLORS["accent_red"]],
                [0.5, COLORS["bg_card"]],
                [1, COLORS["accent_green"]],
            ],
            zmid=0,
            colorbar=dict(
                title=dict(text="P&L", font=dict(color=COLORS["text_secondary"])),
                tickfont=dict(color=COLORS["text_secondary"]),
            ),
            hovertemplate="Spot: %{x}<br>Vol: %{y}<br>P&L: $%{z:.2f}<extra></extra>",
        ))
        heatmap_fig.update_layout(
            xaxis_title="Spot Shock",
            yaxis_title="Vol Shock",
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            margin=dict(l=60, r=20, t=30, b=50),
            hoverlabel=tpl["hoverlabel"],
        )

        return greeks_row, legs_table, payoff_fig, greeks_chart, heatmap_fig
