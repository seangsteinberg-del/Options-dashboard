"""
Options Pricer Panel v2
========================
Multi-leg strategy pricer with Monte Carlo simulation paths,
probability of profit analysis, Bloomberg market data, and
comprehensive payoff/Greeks visualization.
"""

from dash import html, dcc, dash_table, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from core.pricing import (
    bs_price, compute_all_greeks, monte_carlo_price, binomial_tree_price,
    probability_itm, probability_of_profit, expected_move, probability_touch,
)
from core.bloomberg import get_spot_prices, is_connected
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE, BUTTON_STYLE, make_stat_style,
)


STRATEGIES = {
    "Single Call": [{"type": "call", "strike_offset": 0, "qty": 1}],
    "Single Put": [{"type": "put", "strike_offset": 0, "qty": 1}],
    "Bull Call Spread": [
        {"type": "call", "strike_offset": -5, "qty": 1},
        {"type": "call", "strike_offset": 5, "qty": -1}],
    "Bear Put Spread": [
        {"type": "put", "strike_offset": 5, "qty": 1},
        {"type": "put", "strike_offset": -5, "qty": -1}],
    "Long Straddle": [
        {"type": "call", "strike_offset": 0, "qty": 1},
        {"type": "put", "strike_offset": 0, "qty": 1}],
    "Long Strangle": [
        {"type": "call", "strike_offset": 5, "qty": 1},
        {"type": "put", "strike_offset": -5, "qty": 1}],
    "Short Strangle": [
        {"type": "call", "strike_offset": 5, "qty": -1},
        {"type": "put", "strike_offset": -5, "qty": -1}],
    "Iron Condor": [
        {"type": "put", "strike_offset": -10, "qty": 1},
        {"type": "put", "strike_offset": -5, "qty": -1},
        {"type": "call", "strike_offset": 5, "qty": -1},
        {"type": "call", "strike_offset": 10, "qty": 1}],
    "Butterfly": [
        {"type": "call", "strike_offset": -5, "qty": 1},
        {"type": "call", "strike_offset": 0, "qty": -2},
        {"type": "call", "strike_offset": 5, "qty": 1}],
    "Iron Butterfly": [
        {"type": "put", "strike_offset": -5, "qty": 1},
        {"type": "put", "strike_offset": 0, "qty": -1},
        {"type": "call", "strike_offset": 0, "qty": -1},
        {"type": "call", "strike_offset": 5, "qty": 1}],
    "Ratio Spread 1x2": [
        {"type": "call", "strike_offset": 0, "qty": 1},
        {"type": "call", "strike_offset": 10, "qty": -2}],
    "Jade Lizard": [
        {"type": "put", "strike_offset": -10, "qty": -1},
        {"type": "call", "strike_offset": 5, "qty": -1},
        {"type": "call", "strike_offset": 10, "qty": 1}],
    "Broken Wing Butterfly": [
        {"type": "call", "strike_offset": -5, "qty": 1},
        {"type": "call", "strike_offset": 0, "qty": -2},
        {"type": "call", "strike_offset": 10, "qty": 1}],
}


def layout():
    return html.Div([
        # ── Controls ──────────────────────────────────────────
        html.Div([
            html.Div("OPTIONS PRICER", style=CARD_HEADER_STYLE),
            html.Div([
                # Market Data
                html.Div([
                    html.Div("MARKET DATA", style={
                        **LABEL_STYLE, "fontSize": "11px", "color": COLORS["accent_cyan"], "marginBottom": "10px"}),
                    html.Label("UNDERLYING", style=LABEL_STYLE),
                    dcc.Dropdown(id="pricer-ticker", options=[
                        {"label": t, "value": t} for t in
                        ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META"]
                    ], value="SPY", clearable=False, style={"fontSize": "12px", "marginBottom": "10px"}),

                    html.Label("SPOT PRICE", style=LABEL_STYLE),
                    dcc.Input(id="pricer-spot", type="number", value=521.40, step=0.5,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("RISK-FREE RATE (%)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-rate", type="number", value=5.0, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("DIVIDEND YIELD (%)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-div", type="number", value=1.5, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("VOLATILITY (%)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-vol", type="number", value=20.0, step=0.5,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("DTE (DAYS)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-dte", type="number", value=30, step=1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "200px", "marginRight": "24px"}),

                # Strategy
                html.Div([
                    html.Div("STRATEGY", style={
                        **LABEL_STYLE, "fontSize": "11px", "color": COLORS["accent_cyan"], "marginBottom": "10px"}),
                    html.Label("PRESET", style=LABEL_STYLE),
                    dcc.Dropdown(id="pricer-strategy", options=[
                        {"label": k, "value": k} for k in STRATEGIES
                    ], value="Single Call", clearable=False, style={"fontSize": "12px", "marginBottom": "10px"}),

                    html.Label("BASE STRIKE", style=LABEL_STYLE),
                    dcc.Input(id="pricer-strike", type="number", value=520, step=1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("CONTRACTS", style=LABEL_STYLE),
                    dcc.Input(id="pricer-contracts", type="number", value=1, step=1, min=1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("MULTIPLIER", style=LABEL_STYLE),
                    dcc.Input(id="pricer-multiplier", type="number", value=100, step=1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Label("PRICING MODEL", style=LABEL_STYLE),
                    dcc.Dropdown(id="pricer-model", options=[
                        {"label": "Black-Scholes (Analytical)", "value": "bs"},
                        {"label": "Monte Carlo (50K paths)", "value": "mc"},
                        {"label": "Binomial Tree (200 steps)", "value": "binom"},
                    ], value="bs", clearable=False, style={"fontSize": "12px", "marginBottom": "10px"}),

                    html.Label("PAYOFF RANGE (%)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-payoff-range", type="number", value=25, step=5,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "200px"}),
            ], style={"display": "flex", "flexWrap": "wrap"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Greeks + Probability Row ──────────────────────────
        html.Div(id="pricer-greeks-row", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap"}),

        # ── Leg Breakdown ─────────────────────────────────────
        html.Div([
            html.Div("LEG BREAKDOWN", style=CARD_HEADER_STYLE),
            html.Div(id="pricer-legs-table"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Charts Row 1: Payoff + Greeks ─────────────────────
        html.Div([
            html.Div([
                dcc.Graph(id="pricer-payoff-chart", style={"height": "420px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"}, className="dashboard-card"),
            html.Div([
                dcc.Graph(id="pricer-greeks-chart", style={"height": "420px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Charts Row 2: MC Paths + P&L Heatmap ─────────────
        html.Div([
            html.Div([
                dcc.Graph(id="pricer-mc-chart", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"}, className="dashboard-card"),
            html.Div([
                dcc.Graph(id="pricer-pnl-heatmap", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("pricer-greeks-row", "children"),
         Output("pricer-legs-table", "children"),
         Output("pricer-payoff-chart", "figure"),
         Output("pricer-greeks-chart", "figure"),
         Output("pricer-mc-chart", "figure"),
         Output("pricer-pnl-heatmap", "figure"),
         Output("pricer-spot", "value")],
        [Input("pricer-ticker", "value"),
         Input("pricer-spot", "value"),
         Input("pricer-rate", "value"),
         Input("pricer-div", "value"),
         Input("pricer-vol", "value"),
         Input("pricer-dte", "value"),
         Input("pricer-strategy", "value"),
         Input("pricer-strike", "value"),
         Input("pricer-contracts", "value"),
         Input("pricer-multiplier", "value"),
         Input("pricer-model", "value"),
         Input("pricer-payoff-range", "value")],
    )
    def update_pricer(ticker, spot, rate, div_yield, vol, dte, strategy, strike,
                      contracts, multiplier, pricing_model, payoff_range):
        # Auto-fill spot from Bloomberg
        spots = get_spot_prices([ticker or "SPY"])
        bbg_spot = spots.get(ticker or "SPY", {}).get("price", spot or 100)
        S = spot or bbg_spot

        r = (rate or 5.0) / 100
        q = (div_yield or 1.5) / 100
        sigma = (vol or 20.0) / 100
        T = (dte or 30) / 365.0
        K_base = strike or round(S)
        mult = multiplier or 100
        n_con = contracts or 1
        prange = (payoff_range or 25) / 100.0

        legs = STRATEGIES.get(strategy, STRATEGIES["Single Call"])
        tpl = CHART_TEMPLATE["layout"]

        # ── Per-leg Greeks ────────────────────────────────────
        leg_data = []
        total = {"price": 0, "delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

        for i, leg in enumerate(legs):
            K = K_base + leg["strike_offset"]
            qty = leg["qty"] * n_con
            g = compute_all_greeks(S, K, T, r, q, sigma, leg["type"])
            for k in total:
                total[k] += g[k] * qty * mult

            leg_data.append({
                "Leg": f"#{i+1}", "Dir": "LONG" if qty > 0 else "SHORT",
                "Type": leg["type"].upper(), "Strike": f"{K:.1f}", "Qty": f"{abs(qty)}",
                "Price": f"${g['price']:.4f}", "Delta": f"{g['delta'] * qty:.4f}",
                "Gamma": f"{g['gamma'] * qty:.6f}", "Theta": f"{g['theta'] * qty:.4f}",
                "Vega": f"{g['vega'] * qty:.4f}",
            })

        net = total["price"]

        # ── Probability analytics ─────────────────────────────
        if len(legs) == 1:
            p_itm = probability_itm(S, K_base + legs[0]["strike_offset"], T, r, q, sigma, legs[0]["type"]) * 100
            p_profit = probability_of_profit(S, K_base + legs[0]["strike_offset"], T, r, q, sigma, legs[0]["type"]) * 100
            p_touch_val = probability_touch(S, K_base + legs[0]["strike_offset"], T, r, q, sigma) * 100
        else:
            p_itm = 0
            p_profit = 0
            p_touch_val = 0

        exp_mv = expected_move(S, T, sigma)

        # ── Stat Boxes ────────────────────────────────────────
        def sbox(label, value, fmt, color):
            return html.Div([
                html.Div(f"{value:{fmt}}", className="stat-value", style={"color": color, "fontSize": "18px"}),
                html.Div(label, className="stat-label"),
            ], style={**make_stat_style(color), "flex": "1", "minWidth": "100px"}, className="stat-box")

        pcolor = COLORS["pnl_profit"] if net >= 0 else COLORS["pnl_loss"]
        greeks_row = [
            sbox("NET PREMIUM", net, ",.2f", pcolor),
            sbox("DELTA", total["delta"], "+.2f", COLORS["accent_cyan"]),
            sbox("GAMMA", total["gamma"], ".4f", COLORS["accent_blue"]),
            sbox("THETA", total["theta"], ".2f", COLORS["accent_orange"]),
            sbox("VEGA", total["vega"], ".2f", COLORS["accent_purple"]),
            sbox("P(ITM)", p_itm, ".1f%", COLORS["accent_teal"]),
            sbox("P(PROFIT)", p_profit, ".1f%", COLORS["accent_green"]),
            sbox("EXP MOVE", exp_mv, ",.1f", COLORS["accent_pink"]),
        ]

        # ── Legs Table ────────────────────────────────────────
        legs_table = dash_table.DataTable(
            data=leg_data,
            columns=[{"name": c, "id": c} for c in leg_data[0].keys()],
            style_header={"backgroundColor": COLORS["bg_secondary"], "color": COLORS["text_secondary"],
                          "fontWeight": "700", "fontSize": "10px", "textTransform": "uppercase",
                          "letterSpacing": "1px", "border": f"1px solid {COLORS['border_subtle']}"},
            style_cell={"backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"],
                        "fontSize": "12px", "fontFamily": "'JetBrains Mono', monospace",
                        "border": f"1px solid {COLORS['border_subtle']}", "padding": "8px 10px", "textAlign": "center"},
            style_data_conditional=[
                {"if": {"filter_query": "{Dir} = LONG"}, "color": COLORS["accent_green"]},
                {"if": {"filter_query": "{Dir} = SHORT"}, "color": COLORS["accent_red"]},
                {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg_secondary"]}],
        )

        # ── Payoff Diagram ────────────────────────────────────
        spot_range = np.linspace(S * (1 - prange), S * (1 + prange), 400)
        payoff_exp = np.zeros_like(spot_range)
        payoff_now = np.zeros_like(spot_range)
        payoff_half = np.zeros_like(spot_range)

        for leg in legs:
            K = K_base + leg["strike_offset"]
            qty_total = leg["qty"] * n_con * mult
            for j, s in enumerate(spot_range):
                if leg["type"] == "call":
                    payoff_exp[j] += max(s - K, 0) * qty_total
                else:
                    payoff_exp[j] += max(K - s, 0) * qty_total
                payoff_now[j] += bs_price(s, K, T, r, q, sigma, leg["type"]) * qty_total
                payoff_half[j] += bs_price(s, K, T / 2, r, q, sigma, leg["type"]) * qty_total

        payoff_exp -= net
        payoff_now -= net
        payoff_half -= net

        pf = go.Figure()
        pf.add_trace(go.Scatter(x=spot_range, y=payoff_exp, mode="lines", name="At Expiry",
                                line=dict(color=COLORS["accent_cyan"], width=3),
                                fill="tozeroy", fillcolor="rgba(6,182,212,0.06)"))
        pf.add_trace(go.Scatter(x=spot_range, y=payoff_half, mode="lines", name="Half DTE",
                                line=dict(color=COLORS["accent_purple"], width=2, dash="dash")))
        pf.add_trace(go.Scatter(x=spot_range, y=payoff_now, mode="lines", name="Now",
                                line=dict(color=COLORS["accent_blue"], width=2, dash="dot")))
        pf.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
        pf.add_vline(x=S, line=dict(color=COLORS["accent_orange"], width=1.5, dash="dash"),
                      annotation_text=f"Spot {S:.0f}", annotation_font=dict(color=COLORS["accent_orange"], size=10))

        # Expected move shading
        pf.add_vrect(x0=S - exp_mv, x1=S + exp_mv,
                      fillcolor="rgba(139,92,246,0.05)", line_width=0,
                      annotation_text="1σ Move", annotation_position="top left",
                      annotation_font=dict(color=COLORS["accent_purple"], size=9))

        pf.update_layout(title=dict(text=f"Payoff — {strategy}", font=dict(color=COLORS["text_primary"], size=13)),
                         xaxis_title="Underlying", yaxis_title="P&L ($)",
                         paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], margin=dict(l=55, r=20, t=45, b=40),
                         legend=dict(font=dict(color=COLORS["text_secondary"], size=10), bgcolor="rgba(0,0,0,0)"),
                         hoverlabel=tpl["hoverlabel"],
                         xaxis=dict(gridcolor="rgba(30,42,69,0.5)"), yaxis=dict(gridcolor="rgba(30,42,69,0.5)"))

        # ── Greeks Chart ──────────────────────────────────────
        gc = make_subplots(rows=2, cols=2, subplot_titles=("Delta", "Gamma", "Theta ($/day)", "Vega ($)"))
        gnames = ["delta", "gamma", "theta", "vega"]
        gcolors = [COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_orange"], COLORS["accent_purple"]]
        spot_grid = np.linspace(S * (1 - prange), S * (1 + prange), 200)

        for gi, (gn, gc_color) in enumerate(zip(gnames, gcolors)):
            vals = np.zeros_like(spot_grid)
            for leg in legs:
                K = K_base + leg["strike_offset"]
                qty_total = leg["qty"] * n_con * mult
                for j, s in enumerate(spot_grid):
                    vals[j] += compute_all_greeks(s, K, T, r, q, sigma, leg["type"])[gn] * qty_total
            r_idx, c_idx = divmod(gi, 2)
            gc.add_trace(go.Scatter(x=spot_grid, y=vals, mode="lines", showlegend=False,
                                    line=dict(color=gc_color, width=2)), row=r_idx+1, col=c_idx+1)
            gc.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"), row=r_idx+1, col=c_idx+1)

        gc.update_layout(paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], margin=dict(l=50, r=20, t=40, b=30), height=420,
                         hoverlabel=tpl["hoverlabel"])
        gc.update_xaxes(gridcolor="rgba(30,42,69,0.5)")
        gc.update_yaxes(gridcolor="rgba(30,42,69,0.5)")
        for ann in gc.layout.annotations:
            ann.font.color = COLORS["text_primary"]
            ann.font.size = 11

        # ── Monte Carlo Paths ─────────────────────────────────
        mc_fig = go.Figure()
        mc_result = monte_carlo_price(S, K_base, T, r, q, sigma,
                                       legs[0]["type"] if len(legs) == 1 else "call",
                                       n_paths=20000, n_steps=60, return_paths=True, seed=42)

        if "paths" in mc_result:
            paths = mc_result["paths"]
            times = mc_result["times"]
            # Color paths by terminal P&L
            for p in paths[:100]:
                terminal = p[-1]
                c = COLORS["accent_green"] if terminal > S else COLORS["accent_red"]
                mc_fig.add_trace(go.Scatter(x=times, y=p, mode="lines",
                                            line=dict(color=c, width=0.5), opacity=0.15,
                                            showlegend=False, hoverinfo="skip"))
            # Mean path
            mean_path = np.mean(paths, axis=0)
            mc_fig.add_trace(go.Scatter(x=times, y=mean_path, mode="lines",
                                        name="Mean Path", line=dict(color=COLORS["accent_cyan"], width=2.5)))
            # Confidence bands
            p5 = np.percentile(paths, 5, axis=0)
            p95 = np.percentile(paths, 95, axis=0)
            mc_fig.add_trace(go.Scatter(x=times, y=p95, mode="lines", name="95th pct",
                                        line=dict(color=COLORS["accent_purple"], width=1, dash="dash")))
            mc_fig.add_trace(go.Scatter(x=times, y=p5, mode="lines", name="5th pct",
                                        line=dict(color=COLORS["accent_purple"], width=1, dash="dash"),
                                        fill="tonexty", fillcolor="rgba(139,92,246,0.05)"))

        mc_title = f"Monte Carlo Paths — BS: ${mc_result['price']:.2f} (±${mc_result['std_error']:.4f})"
        mc_fig.update_layout(title=dict(text=mc_title, font=dict(color=COLORS["text_primary"], size=12)),
                             xaxis_title="Time (years)", yaxis_title="Price",
                             paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                             font=tpl["font"], margin=dict(l=55, r=20, t=45, b=40),
                             legend=dict(font=dict(color=COLORS["text_secondary"], size=10), bgcolor="rgba(0,0,0,0)"),
                             hoverlabel=tpl["hoverlabel"],
                             xaxis=dict(gridcolor="rgba(30,42,69,0.5)"), yaxis=dict(gridcolor="rgba(30,42,69,0.5)"))

        # ── P&L Heatmap ──────────────────────────────────────
        spot_shocks = np.linspace(-prange, prange, 25)
        vol_shocks = np.linspace(-0.15, 0.15, 25)
        pnl_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))

        for vi, dv in enumerate(vol_shocks):
            for si, ds in enumerate(spot_shocks):
                s_new = S * (1 + ds)
                sig_new = max(sigma + dv, 0.01)
                pnl = 0
                for leg in legs:
                    K = K_base + leg["strike_offset"]
                    qty_total = leg["qty"] * n_con * mult
                    pnl += bs_price(s_new, K, T, r, q, sig_new, leg["type"]) * qty_total
                pnl_matrix[vi, si] = pnl - net

        hm = go.Figure(go.Heatmap(
            x=[f"{ds:+.0%}" for ds in spot_shocks],
            y=[f"{dv:+.0%}" for dv in vol_shocks],
            z=pnl_matrix, zmid=0,
            colorscale=[[0, COLORS["accent_red"]], [0.5, "#111827"], [1, COLORS["accent_green"]]],
            colorbar=dict(title=dict(text="P&L", font=dict(color=COLORS["text_muted"])),
                          tickfont=dict(color=COLORS["text_muted"])),
            hovertemplate="Spot: %{x}<br>Vol: %{y}<br>P&L: $%{z:,.0f}<extra></extra>"))
        hm.update_layout(title=dict(text="P&L Heatmap — Spot vs Vol", font=dict(color=COLORS["text_primary"], size=12)),
                         xaxis_title="Spot Shock", yaxis_title="Vol Shock",
                         paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], margin=dict(l=60, r=20, t=40, b=50), hoverlabel=tpl["hoverlabel"])

        return greeks_row, legs_table, pf, gc, mc_fig, hm, S
