"""
Options Pricer Panel v2
========================
Bloomberg-powered pricer with multi-leg strategies, Monte Carlo simulation,
probability analytics, payoff diagrams, and Greeks visualization.
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
    CHART_TEMPLATE, STAT_BOX_STYLE, make_stat_style,
)

STRATEGIES = {
    "Single Call": [{"type": "call", "strike_offset": 0, "qty": 1}],
    "Single Put": [{"type": "put", "strike_offset": 0, "qty": 1}],
    "Bull Call Spread": [{"type": "call", "strike_offset": -5, "qty": 1},
                         {"type": "call", "strike_offset": 5, "qty": -1}],
    "Bear Put Spread": [{"type": "put", "strike_offset": 5, "qty": 1},
                         {"type": "put", "strike_offset": -5, "qty": -1}],
    "Long Straddle": [{"type": "call", "strike_offset": 0, "qty": 1},
                      {"type": "put", "strike_offset": 0, "qty": 1}],
    "Long Strangle": [{"type": "call", "strike_offset": 5, "qty": 1},
                      {"type": "put", "strike_offset": -5, "qty": 1}],
    "Short Straddle": [{"type": "call", "strike_offset": 0, "qty": -1},
                       {"type": "put", "strike_offset": 0, "qty": -1}],
    "Iron Condor": [{"type": "put", "strike_offset": -10, "qty": 1},
                    {"type": "put", "strike_offset": -5, "qty": -1},
                    {"type": "call", "strike_offset": 5, "qty": -1},
                    {"type": "call", "strike_offset": 10, "qty": 1}],
    "Butterfly": [{"type": "call", "strike_offset": -5, "qty": 1},
                  {"type": "call", "strike_offset": 0, "qty": -2},
                  {"type": "call", "strike_offset": 5, "qty": 1}],
    "Iron Butterfly": [{"type": "put", "strike_offset": 0, "qty": -1},
                       {"type": "call", "strike_offset": 0, "qty": -1},
                       {"type": "put", "strike_offset": -5, "qty": 1},
                       {"type": "call", "strike_offset": 5, "qty": 1}],
    "Ratio Spread 1x2": [{"type": "call", "strike_offset": 0, "qty": 1},
                         {"type": "call", "strike_offset": 10, "qty": -2}],
    "Jade Lizard": [{"type": "put", "strike_offset": -10, "qty": -1},
                    {"type": "call", "strike_offset": 5, "qty": -1},
                    {"type": "call", "strike_offset": 10, "qty": 1}],
    "Broken Wing Butterfly": [{"type": "put", "strike_offset": -10, "qty": 1},
                               {"type": "put", "strike_offset": 0, "qty": -2},
                               {"type": "put", "strike_offset": 5, "qty": 1}],
}


def layout():
    return html.Div([
        # ── Controls ──────────────────────────────────────────────
        html.Div([
            html.Div("OPTIONS PRICER", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Div("MARKET DATA", style={**LABEL_STYLE, "fontSize": "11px",
                             "color": COLORS["accent_cyan"], "marginBottom": "10px"}),
                    html.Label("UNDERLYING", style=LABEL_STYLE),
                    dcc.Dropdown(id="pricer-ticker",
                        options=[{"label": t, "value": t} for t in
                                 ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META"]],
                        value="SPY", clearable=False,
                        style={"fontSize": "12px", "marginBottom": "10px"}),
                    html.Label("RISK-FREE RATE %", style=LABEL_STYLE),
                    dcc.Input(id="pricer-rate", type="number", value=5.0, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                    html.Label("DIVIDEND YIELD %", style=LABEL_STYLE),
                    dcc.Input(id="pricer-div", type="number", value=1.5, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                    html.Label("VOLATILITY %", style=LABEL_STYLE),
                    dcc.Input(id="pricer-vol", type="number", value=20.0, step=0.5, min=0.1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                    html.Label("DTE (DAYS)", style=LABEL_STYLE),
                    dcc.Input(id="pricer-dte", type="number", value=30, step=1, min=0,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "180px"}),

                html.Div([
                    html.Div("STRATEGY", style={**LABEL_STYLE, "fontSize": "11px",
                             "color": COLORS["accent_cyan"], "marginBottom": "10px"}),
                    html.Label("PRESET", style=LABEL_STYLE),
                    dcc.Dropdown(id="pricer-strategy",
                        options=[{"label": k, "value": k} for k in STRATEGIES],
                        value="Single Call", clearable=False,
                        style={"fontSize": "12px", "marginBottom": "10px"}),
                    html.Label("BASE STRIKE", style=LABEL_STYLE),
                    dcc.Input(id="pricer-strike", type="number", value=521, step=1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                    html.Label("CONTRACTS", style=LABEL_STYLE),
                    dcc.Input(id="pricer-contracts", type="number", value=1, step=1, min=1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
                    html.Label("MULTIPLIER", style=LABEL_STYLE),
                    dcc.Input(id="pricer-multiplier", type="number", value=100, step=1,
                              style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),

                    html.Div("PRICING MODEL", style={**LABEL_STYLE, "fontSize": "11px",
                             "color": COLORS["accent_cyan"], "marginBottom": "10px", "marginTop": "12px"}),
                    dcc.RadioItems(id="pricer-model",
                        options=[{"label": "  Black-Scholes", "value": "bs"},
                                 {"label": "  Monte Carlo", "value": "mc"},
                                 {"label": "  Binomial Tree", "value": "binom"}],
                        value="bs", inline=False,
                        style={"color": COLORS["text_secondary"], "fontSize": "12px"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "block", "marginBottom": "6px", "cursor": "pointer"}),
                ], style={"flex": "1", "minWidth": "180px"}),
            ], style={"display": "flex", "gap": "32px", "flexWrap": "wrap"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Greeks + Probability Stats ────────────────────────────
        html.Div(id="pricer-stats-row", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap"}),

        # ── Leg Breakdown ─────────────────────────────────────────
        html.Div([
            html.Div("LEG BREAKDOWN", style=CARD_HEADER_STYLE),
            html.Div(id="pricer-legs-table"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Charts Row 1: Payoff + Greeks ─────────────────────────
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

        # ── Charts Row 2: Monte Carlo + P&L Heatmap ──────────────
        html.Div([
            html.Div([
                html.Div("MONTE CARLO SIMULATION", style=CARD_HEADER_STYLE),
                dcc.Graph(id="pricer-mc-chart", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"}, className="dashboard-card"),
            html.Div([
                html.Div("P&L HEATMAP — SPOT vs VOL", style=CARD_HEADER_STYLE),
                dcc.Graph(id="pricer-pnl-heatmap", style={"height": "380px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "420px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("pricer-stats-row", "children"),
         Output("pricer-legs-table", "children"),
         Output("pricer-payoff-chart", "figure"),
         Output("pricer-greeks-chart", "figure"),
         Output("pricer-mc-chart", "figure"),
         Output("pricer-pnl-heatmap", "figure")],
        [Input("pricer-ticker", "value"), Input("pricer-rate", "value"),
         Input("pricer-div", "value"), Input("pricer-vol", "value"),
         Input("pricer-dte", "value"), Input("pricer-strategy", "value"),
         Input("pricer-strike", "value"), Input("pricer-contracts", "value"),
         Input("pricer-multiplier", "value"), Input("pricer-model", "value")],
    )
    def update_pricer(ticker, rate, div_yield, vol, dte, strategy, strike,
                      contracts, multiplier, model):
        spots = get_spot_prices([ticker or "SPY"])
        S = spots.get(ticker or "SPY", {"price": 521})["price"]
        r = (rate or 5.0) / 100
        q = (div_yield or 1.5) / 100
        sigma = (vol or 20.0) / 100
        T = (dte or 30) / 365.0
        K_base = strike or round(S)
        mult = multiplier or 100
        n_con = contracts or 1
        legs = STRATEGIES.get(strategy, STRATEGIES["Single Call"])
        tpl = CHART_TEMPLATE["layout"]
        prange = 0.30

        # ── Per-leg Greeks ────────────────────────────────────────
        leg_data = []
        total_greeks = {k: 0 for k in ["price", "delta", "gamma", "theta", "vega", "rho"]}

        for i, leg in enumerate(legs):
            K = K_base + leg["strike_offset"]
            qty = leg["qty"] * n_con
            otype = leg["type"]

            if model == "mc":
                mc = monte_carlo_price(S, K, T, r, q, sigma, otype, n_paths=20000, seed=42)
                price = mc["price"]
            elif model == "binom":
                price = binomial_tree_price(S, K, T, r, q, sigma, otype, n_steps=200)
            else:
                price = bs_price(S, K, T, r, q, sigma, otype)

            greeks = compute_all_greeks(S, K, T, r, q, sigma, otype)
            greeks["price"] = price  # override with chosen model

            for g in total_greeks:
                total_greeks[g] += greeks[g] * qty * mult

            leg_data.append({
                "Leg": f"#{i+1}", "Dir": "LONG" if qty > 0 else "SHORT",
                "Type": otype.upper(), "Strike": f"{K:.1f}", "Qty": f"{abs(qty)}",
                "Price": f"${price:.4f}",
                "Delta": f"{greeks['delta'] * qty:.4f}", "Gamma": f"{greeks['gamma'] * qty:.6f}",
                "Theta": f"{greeks['theta'] * qty:.4f}", "Vega": f"{greeks['vega'] * qty:.4f}",
                "P(ITM)": f"{probability_itm(S, K, T, r, q, sigma, otype) * 100:.1f}%",
            })

        net_prem = total_greeks["price"]

        # ── Probability analytics for the strategy ────────────────
        # Compute probability of profit for the whole structure via MC
        spot_range_mc = np.linspace(S * 0.5, S * 1.5, 1000)
        expiry_pnl = np.zeros_like(spot_range_mc)
        for leg in legs:
            K = K_base + leg["strike_offset"]
            qty = leg["qty"] * n_con * mult
            for j, s in enumerate(spot_range_mc):
                payoff = max(s - K, 0) if leg["type"] == "call" else max(K - s, 0)
                expiry_pnl[j] += payoff * qty
        expiry_pnl -= net_prem

        # Probability of profit using log-normal distribution
        log_spots = np.log(spot_range_mc / S)
        mu_T = (r - q - 0.5 * sigma ** 2) * T
        sigma_T = sigma * np.sqrt(T) if T > 0 else 0.001
        from scipy.stats import norm as norm_dist
        pdf_vals = norm_dist.pdf(log_spots, mu_T, sigma_T) / spot_range_mc
        pdf_vals /= np.trapz(pdf_vals, spot_range_mc)

        profitable = expiry_pnl > 0
        pop = np.trapz(pdf_vals * profitable, spot_range_mc) * 100
        expected_pnl = np.trapz(pdf_vals * expiry_pnl, spot_range_mc)
        max_profit = expiry_pnl.max()
        max_loss = expiry_pnl.min()
        exp_move = expected_move(S, T, sigma) if T > 0 else 0

        # ── Stats Row ─────────────────────────────────────────────
        def sbox(label, value, fmt, color=COLORS["accent_cyan"], prefix=""):
            return html.Div([
                html.Div(f"{prefix}{value:{fmt}}", className="stat-value",
                         style={"color": color, "fontSize": "18px"}),
                html.Div(label, className="stat-label"),
            ], style={**make_stat_style(color), "flex": "1", "minWidth": "100px"}, className="stat-box")

        pnl_c = COLORS["pnl_profit"] if net_prem >= 0 else COLORS["pnl_loss"]
        stats = [
            sbox("NET PREMIUM", net_prem, ".2f", pnl_c, "$"),
            sbox("DELTA", total_greeks["delta"], ".2f", COLORS["accent_cyan"]),
            sbox("GAMMA", total_greeks["gamma"], ".4f", COLORS["accent_blue"]),
            sbox("THETA", total_greeks["theta"], ".2f", COLORS["accent_orange"], "$"),
            sbox("VEGA", total_greeks["vega"], ".2f", COLORS["accent_purple"], "$"),
            sbox("P(PROFIT)", pop, ".1f", COLORS["accent_green"] if pop > 50 else COLORS["accent_red"]),
            sbox("EXP P&L", expected_pnl, ",.0f", COLORS["pnl_profit"] if expected_pnl > 0 else COLORS["pnl_loss"], "$"),
            sbox("EXP MOVE", exp_move, ".2f", COLORS["accent_teal"], "±$"),
        ]

        # ── Legs Table ────────────────────────────────────────────
        legs_table = dash_table.DataTable(
            data=leg_data,
            columns=[{"name": c, "id": c} for c in leg_data[0].keys()],
            style_header={"backgroundColor": COLORS["bg_secondary"], "color": COLORS["text_secondary"],
                          "fontWeight": "700", "fontSize": "10px", "textTransform": "uppercase",
                          "letterSpacing": "0.5px", "border": f"1px solid {COLORS['border_subtle']}"},
            style_cell={"backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"],
                        "fontSize": "12px", "fontFamily": "'JetBrains Mono', monospace",
                        "border": f"1px solid {COLORS['border_subtle']}", "padding": "8px 10px",
                        "textAlign": "center"},
            style_data_conditional=[
                {"if": {"filter_query": "{Dir} = LONG"}, "color": COLORS["accent_green"]},
                {"if": {"filter_query": "{Dir} = SHORT"}, "color": COLORS["accent_red"]},
                {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg_secondary"]}],
        )

        # ── Payoff Chart ──────────────────────────────────────────
        spot_range = np.linspace(S * (1 - prange), S * (1 + prange), 300)
        payoff_exp = np.zeros_like(spot_range)
        payoff_now = np.zeros_like(spot_range)
        payoff_mid = np.zeros_like(spot_range)

        for leg in legs:
            K = K_base + leg["strike_offset"]
            qty = leg["qty"] * n_con * mult
            otype = leg["type"]
            for j, s in enumerate(spot_range):
                payoff_exp[j] += (max(s - K, 0) if otype == "call" else max(K - s, 0)) * qty
                payoff_now[j] += bs_price(s, K, T, r, q, sigma, otype) * qty
                payoff_mid[j] += bs_price(s, K, T / 2, r, q, sigma, otype) * qty

        payoff_exp -= net_prem
        payoff_now -= net_prem
        payoff_mid -= net_prem

        payoff_fig = go.Figure()
        # Profit/loss fill
        payoff_fig.add_trace(go.Scatter(x=spot_range, y=payoff_exp, mode="lines",
            name="At Expiry", line=dict(color=COLORS["accent_cyan"], width=3),
            fill="tozeroy", fillcolor="rgba(6,182,212,0.08)"))
        payoff_fig.add_trace(go.Scatter(x=spot_range, y=payoff_mid, mode="lines",
            name="T/2", line=dict(color=COLORS["accent_purple"], width=2, dash="dash")))
        payoff_fig.add_trace(go.Scatter(x=spot_range, y=payoff_now, mode="lines",
            name="Current", line=dict(color=COLORS["accent_orange"], width=2, dash="dot")))
        payoff_fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
        payoff_fig.add_vline(x=S, line=dict(color=COLORS["accent_blue"], width=1, dash="dash"),
                             annotation_text=f"Spot ${S:.0f}",
                             annotation_font=dict(color=COLORS["accent_blue"], size=10))
        # Expected move range
        if exp_move > 0:
            payoff_fig.add_vrect(x0=S - exp_move, x1=S + exp_move,
                                 fillcolor="rgba(59,130,246,0.05)", line_width=0,
                                 annotation_text="1σ move", annotation_position="top left",
                                 annotation_font=dict(color=COLORS["text_muted"], size=9))

        payoff_fig.update_layout(
            title=dict(text=f"{strategy} — Payoff Diagram", font=dict(color=COLORS["text_primary"], size=14)),
            xaxis_title="Underlying Price", yaxis_title="P&L ($)",
            paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"], margin=dict(l=50, r=20, t=45, b=40),
            legend=dict(font=dict(color=COLORS["text_secondary"], size=11), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=tpl["hoverlabel"],
            xaxis=dict(gridcolor="rgba(30,42,69,0.5)"), yaxis=dict(gridcolor="rgba(30,42,69,0.5)"))

        # ── Greeks Chart ──────────────────────────────────────────
        greeks_fig = make_subplots(rows=2, cols=2,
                                    subplot_titles=("Delta", "Gamma", "Theta ($/day)", "Vega ($)"))
        spot_grid = np.linspace(S * (1 - prange), S * (1 + prange), 200)
        greek_names = ["delta", "gamma", "theta", "vega"]
        greek_colors = [COLORS["accent_cyan"], COLORS["accent_blue"],
                        COLORS["accent_orange"], COLORS["accent_purple"]]
        positions = [(1,1), (1,2), (2,1), (2,2)]

        for gi, (gname, color, (row, col)) in enumerate(zip(greek_names, greek_colors, positions)):
            vals = np.zeros_like(spot_grid)
            for leg in legs:
                K = K_base + leg["strike_offset"]
                qty = leg["qty"] * n_con * mult
                for j, s in enumerate(spot_grid):
                    g = compute_all_greeks(s, K, T, r, q, sigma, leg["type"])
                    vals[j] += g[gname] * qty
            greeks_fig.add_trace(go.Scatter(x=spot_grid, y=vals, mode="lines",
                line=dict(color=color, width=2), name=gname.capitalize(), showlegend=False),
                row=row, col=col)
            greeks_fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"),
                                 row=row, col=col)

        greeks_fig.update_layout(
            paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"], margin=dict(l=50, r=20, t=40, b=40), height=420,
            hoverlabel=tpl["hoverlabel"])
        greeks_fig.update_xaxes(gridcolor="rgba(30,42,69,0.5)")
        greeks_fig.update_yaxes(gridcolor="rgba(30,42,69,0.5)")
        for ann in greeks_fig.layout.annotations:
            ann.font.color = COLORS["text_primary"]; ann.font.size = 11

        # ── Monte Carlo Chart ─────────────────────────────────────
        mc_fig = go.Figure()
        # Use first leg for MC paths visualization
        first_leg = legs[0]
        K_mc = K_base + first_leg["strike_offset"]
        mc_result = monte_carlo_price(S, K_mc, T, r, q, sigma, first_leg["type"],
                                       n_paths=10000, n_steps=80, return_paths=True, seed=42)

        if "paths" in mc_result:
            for path in mc_result["paths"][:100]:
                mc_fig.add_trace(go.Scatter(x=mc_result["times"], y=path, mode="lines",
                    line=dict(width=0.5, color="rgba(6,182,212,0.15)"), showlegend=False,
                    hoverinfo="skip"))
            # Mean path
            mean_path = np.mean(mc_result["paths"], axis=0)
            mc_fig.add_trace(go.Scatter(x=mc_result["times"], y=mean_path, mode="lines",
                name="Mean Path", line=dict(color=COLORS["accent_cyan"], width=2.5)))
            # Strike line
            mc_fig.add_hline(y=K_mc, line=dict(color=COLORS["accent_red"], width=1, dash="dash"),
                             annotation_text=f"K={K_mc}", annotation_font=dict(color=COLORS["accent_red"], size=10))
            # Terminal distribution (as marginal histogram)
            mc_fig.add_annotation(
                text=f"MC Price: ${mc_result['price']:.4f} ± ${mc_result['std_error']:.4f}",
                xref="paper", yref="paper", x=0.02, y=0.98,
                showarrow=False, font=dict(color=COLORS["accent_cyan"], size=11))

        mc_fig.update_layout(
            paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"], margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(title="Time (years)", gridcolor="rgba(30,42,69,0.5)"),
            yaxis=dict(title="Price", gridcolor="rgba(30,42,69,0.5)"),
            legend=dict(font=dict(color=COLORS["text_secondary"]), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=tpl["hoverlabel"])

        # ── P&L Heatmap ──────────────────────────────────────────
        spot_shocks = np.linspace(-prange, prange, 21)
        vol_shocks = np.linspace(-0.15, 0.15, 21)
        pnl_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))

        for vi, dv in enumerate(vol_shocks):
            for si, ds in enumerate(spot_shocks):
                pnl = 0
                for leg in legs:
                    K = K_base + leg["strike_offset"]
                    qty = leg["qty"] * n_con * mult
                    pnl += bs_price(S * (1 + ds), K, T, r, q, max(sigma + dv, 0.01), leg["type"]) * qty
                pnl_matrix[vi, si] = pnl - net_prem

        hm_fig = go.Figure(go.Heatmap(
            x=[f"{ds:+.0%}" for ds in spot_shocks],
            y=[f"{dv:+.0%}" for dv in vol_shocks],
            z=pnl_matrix, zmid=0,
            colorscale=[[0, COLORS["accent_red"]], [0.5, "#111827"], [1, COLORS["accent_green"]]],
            colorbar=dict(title=dict(text="P&L", font=dict(color=COLORS["text_secondary"])),
                          tickfont=dict(color=COLORS["text_secondary"])),
            hovertemplate="Spot: %{x}<br>Vol: %{y}<br>P&L: $%{z:,.0f}<extra></extra>"))
        hm_fig.update_layout(
            xaxis_title="Spot Shock", yaxis_title="Vol Shock",
            paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"], margin=dict(l=60, r=20, t=20, b=50),
            hoverlabel=tpl["hoverlabel"])

        return stats, legs_table, payoff_fig, greeks_fig, mc_fig, hm_fig
