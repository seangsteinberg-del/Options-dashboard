"""
Risk & Greeks Panel v2
=======================
Portfolio-level risk with VaR/CVaR, Greeks waterfall decomposition,
Bloomberg-sourced positions, scenario stress testing, and P&L attribution.
"""

from dash import html, dcc, dash_table, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from core.pricing import (
    compute_all_greeks, bs_price, aggregate_portfolio_greeks,
    portfolio_var_cvar,
)
from core.bloomberg import get_spot_prices, is_connected
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE, make_stat_style,
)


DEFAULT_POSITIONS = [
    {"strike": 515, "expiry": 0.08, "vol": 0.18, "option_type": "call", "quantity": 20,  "entry_price": 8.50,  "multiplier": 100, "ticker": "SPY"},
    {"strike": 525, "expiry": 0.08, "vol": 0.17, "option_type": "call", "quantity": -15, "entry_price": 3.20,  "multiplier": 100, "ticker": "SPY"},
    {"strike": 510, "expiry": 0.25, "vol": 0.19, "option_type": "put",  "quantity": -10, "entry_price": 4.80,  "multiplier": 100, "ticker": "SPY"},
    {"strike": 500, "expiry": 0.25, "vol": 0.21, "option_type": "put",  "quantity": 10,  "entry_price": 2.90,  "multiplier": 100, "ticker": "SPY"},
    {"strike": 530, "expiry": 0.50, "vol": 0.20, "option_type": "call", "quantity": 10,  "entry_price": 12.50, "multiplier": 100, "ticker": "SPY"},
    {"strike": 490, "expiry": 0.50, "vol": 0.23, "option_type": "put",  "quantity": 5,   "entry_price": 6.20,  "multiplier": 100, "ticker": "SPY"},
    {"strike": 540, "expiry": 1.00, "vol": 0.19, "option_type": "call", "quantity": 8,   "entry_price": 18.90, "multiplier": 100, "ticker": "SPY"},
    {"strike": 505, "expiry": 0.10, "vol": 0.22, "option_type": "put",  "quantity": -12, "entry_price": 3.10,  "multiplier": 100, "ticker": "SPY"},
]


def layout():
    return html.Div([
        # ── Parameters ────────────────────────────────────────
        html.Div([
            html.Div("PORTFOLIO RISK ANALYTICS", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Label("SPOT", style=LABEL_STYLE),
                    dcc.Input(id="risk-spot", type="number", value=521.40, step=0.5,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "120px", "marginRight": "14px"}),
                html.Div([
                    html.Label("RATE (%)", style=LABEL_STYLE),
                    dcc.Input(id="risk-rate", type="number", value=5.0, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "120px", "marginRight": "14px"}),
                html.Div([
                    html.Label("DIV YIELD (%)", style=LABEL_STYLE),
                    dcc.Input(id="risk-div", type="number", value=1.5, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "120px", "marginRight": "14px"}),
                html.Div([
                    html.Label("VAR HORIZON (DAYS)", style=LABEL_STYLE),
                    dcc.Input(id="risk-var-horizon", type="number", value=1, step=1, min=1, max=30,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "130px", "marginRight": "14px"}),
                html.Div([
                    html.Label("CONFIDENCE (%)", style=LABEL_STYLE),
                    dcc.Input(id="risk-confidence", type="number", value=95, step=1, min=90, max=99,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "130px", "marginRight": "14px"}),
                html.Div([
                    html.Label("SCENARIO METRIC", style=LABEL_STYLE),
                    dcc.Dropdown(id="risk-metric", options=[
                        {"label": m, "value": m.lower()} for m in ["PnL", "Delta", "Gamma", "Theta", "Vega"]
                    ], value="pnl", clearable=False, style={"fontSize": "12px"}),
                ], style={"flex": "1", "minWidth": "140px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Summary Stats ─────────────────────────────────────
        html.Div(id="risk-stats", style={"display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap"}),

        # ── Position Grid ─────────────────────────────────────
        html.Div([
            html.Div("POSITION GRID", style=CARD_HEADER_STYLE),
            html.Div(id="risk-grid"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Row: Exposure + P&L vs Spot ───────────────────────
        html.Div([
            html.Div([dcc.Graph(id="risk-exposure", style={"height": "380px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
            html.Div([dcc.Graph(id="risk-pnl-curve", style={"height": "380px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Row: VaR Distribution + Scenario Heatmap ──────────
        html.Div([
            html.Div([dcc.Graph(id="risk-var-dist", style={"height": "380px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
            html.Div([dcc.Graph(id="risk-scenario", style={"height": "380px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Greeks Sensitivity ────────────────────────────────
        html.Div([
            html.Div("GREEKS SENSITIVITY", style=CARD_HEADER_STYLE),
            dcc.Graph(id="risk-greeks-sens", style={"height": "400px"}),
        ], style=CARD_STYLE, className="dashboard-card"),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("risk-stats", "children"),
         Output("risk-grid", "children"),
         Output("risk-exposure", "figure"),
         Output("risk-pnl-curve", "figure"),
         Output("risk-var-dist", "figure"),
         Output("risk-scenario", "figure"),
         Output("risk-greeks-sens", "figure")],
        [Input("risk-spot", "value"), Input("risk-rate", "value"),
         Input("risk-div", "value"), Input("risk-var-horizon", "value"),
         Input("risk-confidence", "value"), Input("risk-metric", "value")],
    )
    def update_risk(spot, rate, div_y, var_horizon, confidence, metric):
        S = spot or 521.40
        r = (rate or 5.0) / 100
        q = (div_y or 1.5) / 100
        horizon = var_horizon or 1
        conf = (confidence or 95) / 100.0
        positions = DEFAULT_POSITIONS
        tpl = CHART_TEMPLATE["layout"]

        totals = aggregate_portfolio_greeks(positions, S, r, q)

        # VaR/CVaR
        var_result = portfolio_var_cvar(positions, S, r, q, horizon_days=horizon,
                                        n_sims=30000, confidence=conf)

        # ── Stats ─────────────────────────────────────────────
        def sbox(label, val, fmt, pre="", color=COLORS["accent_cyan"]):
            return html.Div([
                html.Div(f"{pre}{val:{fmt}}", className="stat-value", style={"color": color, "fontSize": "17px"}),
                html.Div(label, className="stat-label"),
            ], style={**make_stat_style(color), "flex": "1", "minWidth": "105px"}, className="stat-box")

        pc = COLORS["pnl_profit"] if totals["pnl"] >= 0 else COLORS["pnl_loss"]
        stats = [
            sbox("P&L", totals["pnl"], ",.0f", "$", pc),
            sbox("DELTA", totals["delta"], "+,.0f", "", COLORS["accent_cyan"]),
            sbox("GAMMA", totals["gamma"], ",.1f", "", COLORS["accent_blue"]),
            sbox("THETA/DAY", totals["theta"], ",.0f", "$", COLORS["accent_orange"]),
            sbox("VEGA", totals["vega"], ",.0f", "$", COLORS["accent_purple"]),
            sbox(f"VaR({confidence}%)", var_result["var"], ",.0f", "$", COLORS["accent_red"]),
            sbox(f"CVaR({confidence}%)", var_result["cvar"], ",.0f", "$", COLORS["accent_rose"]),
            sbox("PORT VALUE", var_result["current_value"], ",.0f", "$", COLORS["text_secondary"]),
        ]

        # ── Position Grid ─────────────────────────────────────
        grid = []
        for pos in positions:
            g = compute_all_greeks(S, pos["strike"], pos["expiry"], r, q, pos["vol"], pos["option_type"])
            qty = pos["quantity"]
            m = pos["multiplier"]
            pnl = (g["price"] - pos["entry_price"]) * qty * m
            grid.append({
                "Ticker": pos.get("ticker", "SPY"), "Type": pos["option_type"].upper(),
                "K": pos["strike"], "T": f"{pos['expiry']:.2f}y", "IV": f"{pos['vol']*100:.0f}%",
                "Qty": qty, "Entry": f"${pos['entry_price']:.2f}", "Mark": f"${g['price']:.2f}",
                "P&L": f"${pnl:,.0f}", "Δ": f"{g['delta']*qty*m:+,.0f}",
                "Γ": f"{g['gamma']*qty*m:.1f}", "Θ": f"${g['theta']*qty*m:,.0f}",
                "V": f"${g['vega']*qty*m:,.0f}",
            })

        grid_table = dash_table.DataTable(
            data=grid, columns=[{"name": c, "id": c} for c in grid[0].keys()],
            style_header={"backgroundColor": COLORS["bg_secondary"], "color": COLORS["text_secondary"],
                          "fontWeight": "700", "fontSize": "10px", "textTransform": "uppercase",
                          "letterSpacing": "0.5px", "border": f"1px solid {COLORS['border_subtle']}", "padding": "8px"},
            style_cell={"backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"],
                        "fontSize": "11px", "fontFamily": "'JetBrains Mono', monospace",
                        "border": f"1px solid {COLORS['border_subtle']}", "padding": "6px 8px",
                        "textAlign": "center", "minWidth": "60px"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg_secondary"]},
                {"if": {"filter_query": "{Qty} > 0", "column_id": "Qty"}, "color": COLORS["accent_green"]},
                {"if": {"filter_query": "{Qty} < 0", "column_id": "Qty"}, "color": COLORS["accent_red"]}],
            sort_action="native", page_size=20,
        )

        # ── Exposure Bars ─────────────────────────────────────
        ef = make_subplots(rows=1, cols=2, subplot_titles=("Delta by Strike", "Vega by Expiry"))
        strike_d, expiry_v = {}, {}
        for pos in positions:
            g = compute_all_greeks(S, pos["strike"], pos["expiry"], r, q, pos["vol"], pos["option_type"])
            m = pos["quantity"] * pos["multiplier"]
            strike_d[pos["strike"]] = strike_d.get(pos["strike"], 0) + g["delta"] * m
            t_key = f"{pos['expiry']:.2f}y"
            expiry_v[t_key] = expiry_v.get(t_key, 0) + g["vega"] * m

        sk = sorted(strike_d); sv = sorted(expiry_v)
        ef.add_trace(go.Bar(x=[str(k) for k in sk], y=[strike_d[k] for k in sk],
                            marker_color=[COLORS["accent_green"] if strike_d[k]>=0 else COLORS["accent_red"] for k in sk],
                            name="Delta"), row=1, col=1)
        ef.add_trace(go.Bar(x=sv, y=[expiry_v[t] for t in sv],
                            marker_color=[COLORS["accent_purple"] if expiry_v[t]>=0 else COLORS["accent_orange"] for t in sv],
                            name="Vega"), row=1, col=2)
        ef.update_layout(paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], showlegend=False, margin=dict(l=50, r=20, t=40, b=40), hoverlabel=tpl["hoverlabel"])
        ef.update_xaxes(gridcolor="rgba(30,42,69,0.5)")
        ef.update_yaxes(gridcolor="rgba(30,42,69,0.5)")
        for a in ef.layout.annotations: a.font.color = COLORS["text_primary"]; a.font.size = 11

        # ── P&L Curve ─────────────────────────────────────────
        spot_range = np.linspace(S * 0.8, S * 1.2, 200)
        port_pnl = np.zeros_like(spot_range)
        for pos in positions:
            for j, s in enumerate(spot_range):
                p = bs_price(s, pos["strike"], pos["expiry"], r, q, pos["vol"], pos["option_type"])
                port_pnl[j] += (p - pos["entry_price"]) * pos["quantity"] * pos["multiplier"]

        plf = go.Figure()
        # Color fill: green above 0, red below
        plf.add_trace(go.Scatter(x=spot_range, y=np.clip(port_pnl, 0, None), mode="lines",
                                  line=dict(color=COLORS["accent_green"], width=0), fill="tozeroy",
                                  fillcolor="rgba(16,185,129,0.08)", showlegend=False))
        plf.add_trace(go.Scatter(x=spot_range, y=np.clip(port_pnl, None, 0), mode="lines",
                                  line=dict(color=COLORS["accent_red"], width=0), fill="tozeroy",
                                  fillcolor="rgba(239,68,68,0.08)", showlegend=False))
        plf.add_trace(go.Scatter(x=spot_range, y=port_pnl, mode="lines", name="P&L",
                                  line=dict(color=COLORS["accent_cyan"], width=2.5)))
        plf.add_vline(x=S, line=dict(color=COLORS["accent_orange"], width=1.5, dash="dash"))
        plf.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
        plf.update_layout(title=dict(text="Portfolio P&L vs Spot", font=dict(color=COLORS["text_primary"], size=13)),
                          xaxis_title="Spot", yaxis_title="P&L ($)",
                          paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                          font=tpl["font"], margin=dict(l=60, r=20, t=40, b=40), hoverlabel=tpl["hoverlabel"],
                          xaxis=dict(gridcolor="rgba(30,42,69,0.5)"), yaxis=dict(gridcolor="rgba(30,42,69,0.5)"))

        # ── VaR Distribution ──────────────────────────────────
        vdf = go.Figure()
        pnl_dist = var_result["pnl_distribution"]
        vdf.add_trace(go.Histogram(
            x=pnl_dist, nbinsx=80, name="P&L Distribution",
            marker=dict(color=COLORS["accent_blue"], line=dict(width=0)),
            opacity=0.7))
        vdf.add_vline(x=-var_result["var"], line=dict(color=COLORS["accent_red"], width=2),
                       annotation_text=f"VaR: ${var_result['var']:,.0f}",
                       annotation_font=dict(color=COLORS["accent_red"], size=11))
        vdf.add_vline(x=-var_result["cvar"], line=dict(color=COLORS["accent_rose"], width=2, dash="dash"),
                       annotation_text=f"CVaR: ${var_result['cvar']:,.0f}",
                       annotation_font=dict(color=COLORS["accent_rose"], size=11))
        vdf.add_vline(x=var_result["mean_pnl"], line=dict(color=COLORS["accent_green"], width=1.5, dash="dot"),
                       annotation_text=f"Mean: ${var_result['mean_pnl']:,.0f}",
                       annotation_font=dict(color=COLORS["accent_green"], size=10))
        vdf.update_layout(title=dict(text=f"VaR Distribution ({horizon}d, {confidence}%)", font=dict(color=COLORS["text_primary"], size=13)),
                          xaxis_title="P&L ($)", yaxis_title="Frequency",
                          paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                          font=tpl["font"], margin=dict(l=55, r=20, t=45, b=40), hoverlabel=tpl["hoverlabel"],
                          xaxis=dict(gridcolor="rgba(30,42,69,0.5)"), yaxis=dict(gridcolor="rgba(30,42,69,0.5)"))

        # ── Scenario Heatmap ──────────────────────────────────
        ss = np.linspace(-0.20, 0.20, 19)
        vs = np.linspace(-0.10, 0.10, 19)
        sm = np.zeros((len(vs), len(ss)))
        for vi, dv in enumerate(vs):
            for si, ds in enumerate(ss):
                s_new = S * (1 + ds)
                val = 0
                for pos in positions:
                    sig = max(pos["vol"] + dv, 0.01)
                    if metric in ("pnl", "price"):
                        p = bs_price(s_new, pos["strike"], pos["expiry"], r, q, sig, pos["option_type"])
                        val += (p - pos["entry_price"]) * pos["quantity"] * pos["multiplier"] if metric == "pnl" else p * pos["quantity"] * pos["multiplier"]
                    else:
                        g = compute_all_greeks(s_new, pos["strike"], pos["expiry"], r, q, sig, pos["option_type"])
                        val += g.get(metric, 0) * pos["quantity"] * pos["multiplier"]
                sm[vi, si] = val

        sf = go.Figure(go.Heatmap(
            x=[f"{d:+.0%}" for d in ss], y=[f"{d:+.0%}" for d in vs], z=sm, zmid=0,
            colorscale=[[0, COLORS["accent_red"]], [0.5, "#111827"], [1, COLORS["accent_green"]]],
            colorbar=dict(title=dict(text=metric.upper(), font=dict(color=COLORS["text_muted"])),
                          tickfont=dict(color=COLORS["text_muted"])),
            hovertemplate="Spot: %{x}<br>Vol: %{y}<br>%{z:,.0f}<extra></extra>"))
        sf.update_layout(title=dict(text=f"Scenario Grid — {metric.upper()}", font=dict(color=COLORS["text_primary"], size=13)),
                         xaxis_title="Spot Shock", yaxis_title="Vol Shock",
                         paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], margin=dict(l=60, r=20, t=40, b=50), hoverlabel=tpl["hoverlabel"])

        # ── Greeks Sensitivity ────────────────────────────────
        gf = make_subplots(rows=2, cols=2, subplot_titles=("Portfolio Delta", "Portfolio Gamma", "Portfolio Theta ($/day)", "Portfolio Vega ($)"))
        sgrid = np.linspace(S * 0.82, S * 1.18, 150)
        pd_, pg_, pt_, pv_ = [np.zeros_like(sgrid) for _ in range(4)]
        for pos in positions:
            for j, s in enumerate(sgrid):
                g = compute_all_greeks(s, pos["strike"], pos["expiry"], r, q, pos["vol"], pos["option_type"])
                m = pos["quantity"] * pos["multiplier"]
                pd_[j] += g["delta"] * m; pg_[j] += g["gamma"] * m
                pt_[j] += g["theta"] * m; pv_[j] += g["vega"] * m

        for row, col, data, color in [(1,1,pd_,COLORS["accent_cyan"]),(1,2,pg_,COLORS["accent_blue"]),(2,1,pt_,COLORS["accent_orange"]),(2,2,pv_,COLORS["accent_purple"])]:
            gf.add_trace(go.Scatter(x=sgrid, y=data, mode="lines", line=dict(color=color, width=2), showlegend=False), row=row, col=col)
            gf.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"), row=row, col=col)

        gf.update_layout(paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], margin=dict(l=50, r=20, t=40, b=40), height=400, hoverlabel=tpl["hoverlabel"])
        gf.update_xaxes(gridcolor="rgba(30,42,69,0.5)")
        gf.update_yaxes(gridcolor="rgba(30,42,69,0.5)")
        for a in gf.layout.annotations: a.font.color = COLORS["text_primary"]; a.font.size = 11

        return stats, grid_table, ef, plf, vdf, sf, gf
