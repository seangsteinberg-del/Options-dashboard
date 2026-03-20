"""
Options Chain Panel
===================
Full options chain viewer pulling from Bloomberg (or synthetic fallback).
Shows calls/puts side-by-side with IV skew, volume heatmap, and Greeks.
"""

from dash import html, dcc, dash_table, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from core.pricing import generate_options_chain
from core.bloomberg import get_options_chain, get_spot_prices, is_connected
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE, make_stat_style,
)


EXPIRY_OPTIONS = [
    {"label": "1W — Mar 27", "value": "7"},
    {"label": "2W — Apr 03", "value": "14"},
    {"label": "1M — Apr 17", "value": "30"},
    {"label": "2M — May 15", "value": "60"},
    {"label": "3M — Jun 19", "value": "90"},
    {"label": "6M — Sep 18", "value": "180"},
    {"label": "1Y — Mar 19 '27", "value": "365"},
]


def layout():
    return html.Div([
        # ── Controls ──────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("OPTIONS CHAIN", style={**CARD_HEADER_STYLE, "display": "inline",
                           "borderBottom": "none", "paddingBottom": "0", "marginBottom": "0"}),
                html.Span("  BLOOMBERG" if is_connected() else "  SYNTHETIC", style={
                    "fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px",
                    "color": COLORS["accent_green"] if is_connected() else COLORS["accent_orange"],
                    "marginLeft": "12px"}),
            ], style={**CARD_HEADER_STYLE}),

            html.Div([
                html.Div([
                    html.Label("UNDERLYING", style=LABEL_STYLE),
                    dcc.Dropdown(id="chain-ticker", options=[
                        {"label": t, "value": t} for t in
                        ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "GS"]
                    ], value="SPY", clearable=False, style={"fontSize": "12px"}),
                ], style={"flex": "1", "minWidth": "130px", "marginRight": "14px"}),
                html.Div([
                    html.Label("EXPIRY", style=LABEL_STYLE),
                    dcc.Dropdown(id="chain-expiry", options=EXPIRY_OPTIONS,
                                 value="30", clearable=False, style={"fontSize": "12px"}),
                ], style={"flex": "1", "minWidth": "170px", "marginRight": "14px"}),
                html.Div([
                    html.Label("STRIKES TO SHOW", style=LABEL_STYLE),
                    dcc.Slider(id="chain-num-strikes", min=10, max=40, step=5, value=20,
                               marks={i: str(i) for i in range(10, 45, 10)}),
                ], style={"flex": "2", "minWidth": "200px", "marginRight": "14px"}),
                html.Div([
                    html.Label("HIGHLIGHT", style=LABEL_STYLE),
                    dcc.Dropdown(id="chain-highlight", options=[
                        {"label": "None", "value": "none"},
                        {"label": "High IV", "value": "iv"},
                        {"label": "High Volume", "value": "volume"},
                        {"label": "ITM", "value": "itm"},
                    ], value="none", clearable=False, style={"fontSize": "12px"}),
                ], style={"flex": "1", "minWidth": "130px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Summary ───────────────────────────────────────────
        html.Div(id="chain-stats", style={"display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap"}),

        # ── Calls Table ───────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("CALLS", style={**CARD_HEADER_STYLE, "display": "inline", "color": COLORS["accent_cyan"],
                           "borderBottom": "none", "paddingBottom": "0", "marginBottom": "0"}),
            ], style={**CARD_HEADER_STYLE}),
            html.Div(id="chain-calls-table"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Puts Table ────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("PUTS", style={**CARD_HEADER_STYLE, "display": "inline", "color": COLORS["accent_purple"],
                           "borderBottom": "none", "paddingBottom": "0", "marginBottom": "0"}),
            ], style={**CARD_HEADER_STYLE}),
            html.Div(id="chain-puts-table"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Charts ────────────────────────────────────────────
        html.Div([
            html.Div([dcc.Graph(id="chain-iv-skew", style={"height": "360px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
            html.Div([dcc.Graph(id="chain-oi-chart", style={"height": "360px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Volume Heatmap ────────────────────────────────────
        html.Div([
            html.Div("VOLUME / OPEN INTEREST HEATMAP", style=CARD_HEADER_STYLE),
            dcc.Graph(id="chain-vol-heatmap", style={"height": "350px"}),
        ], style=CARD_STYLE, className="dashboard-card"),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("chain-stats", "children"),
         Output("chain-calls-table", "children"),
         Output("chain-puts-table", "children"),
         Output("chain-iv-skew", "figure"),
         Output("chain-oi-chart", "figure"),
         Output("chain-vol-heatmap", "figure")],
        [Input("chain-ticker", "value"),
         Input("chain-expiry", "value"),
         Input("chain-num-strikes", "value"),
         Input("chain-highlight", "value")],
    )
    def update_chain(ticker, expiry_days, num_strikes, highlight):
        ticker = ticker or "SPY"
        dte = int(expiry_days or 30)
        n_strikes = num_strikes or 20
        tpl = CHART_TEMPLATE["layout"]

        spots = get_spot_prices([ticker])
        S = spots.get(ticker, {}).get("price", 100)

        # Get chain (Bloomberg or synthetic)
        chain_df = get_options_chain(ticker)
        if chain_df.empty or "strike" not in chain_df.columns:
            chain_data = generate_options_chain(
                S=S, r=0.05, q=0.015, base_vol=0.20,
                expiry_days=dte, strike_step=max(1, round(S / 40)),
                num_strikes=n_strikes,
            )
            chain_df = pd.DataFrame(chain_data)

        calls = chain_df[chain_df["type"] == "call"].copy()
        puts = chain_df[chain_df["type"] == "put"].copy()

        # ── Stats ─────────────────────────────────────────────
        total_call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
        total_put_vol = puts["volume"].sum() if "volume" in puts.columns else 0
        pc_ratio = total_put_vol / max(total_call_vol, 1)
        total_oi = chain_df["open_interest"].sum() if "open_interest" in chain_df.columns else 0
        max_pain = calls["strike"].iloc[len(calls)//2] if len(calls) > 0 else S
        avg_iv = chain_df["iv"].mean() if "iv" in chain_df.columns else 0

        def sb(l, v, c):
            return html.Div([html.Div(str(v), className="stat-value", style={"color": c, "fontSize": "18px"}),
                             html.Div(l, className="stat-label")],
                            style={**make_stat_style(c), "flex": "1", "minWidth": "100px"}, className="stat-box")

        stats = [
            sb("SPOT", f"${S:,.2f}", COLORS["text_primary"]),
            sb("CALL VOL", f"{total_call_vol:,}", COLORS["accent_cyan"]),
            sb("PUT VOL", f"{total_put_vol:,}", COLORS["accent_purple"]),
            sb("P/C RATIO", f"{pc_ratio:.2f}", COLORS["accent_orange"]),
            sb("TOTAL OI", f"{total_oi:,}", COLORS["accent_blue"]),
            sb("AVG IV", f"{avg_iv:.1f}%", COLORS["accent_green"]),
            sb("DTE", f"{dte}", COLORS["text_secondary"]),
        ]

        # ── Tables ────────────────────────────────────────────
        def make_chain_table(df, opt_type):
            cols = ["strike", "bid", "ask", "mid", "iv", "delta", "gamma", "theta", "vega", "volume", "open_interest"]
            avail = [c for c in cols if c in df.columns]
            tdata = df[avail].round(4).to_dict("records")

            cond = [{"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg_secondary"]}]
            if highlight == "itm":
                if opt_type == "call":
                    cond.append({"if": {"filter_query": f"{{strike}} < {S}"}, "backgroundColor": "rgba(6,182,212,0.08)"})
                else:
                    cond.append({"if": {"filter_query": f"{{strike}} > {S}"}, "backgroundColor": "rgba(139,92,246,0.08)"})
            elif highlight == "iv":
                cond.append({"if": {"filter_query": f"{{iv}} > {avg_iv * 1.2}"}, "backgroundColor": "rgba(245,158,11,0.08)"})

            return dash_table.DataTable(
                data=tdata, columns=[{"name": c.upper(), "id": c} for c in avail],
                style_header={"backgroundColor": COLORS["bg_secondary"], "color": COLORS["text_secondary"],
                              "fontWeight": "700", "fontSize": "10px", "textTransform": "uppercase",
                              "letterSpacing": "0.5px", "border": f"1px solid {COLORS['border_subtle']}", "padding": "8px"},
                style_cell={"backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"],
                            "fontSize": "11px", "fontFamily": "'JetBrains Mono', monospace",
                            "border": f"1px solid {COLORS['border_subtle']}", "padding": "6px 8px",
                            "textAlign": "center", "minWidth": "60px"},
                style_data_conditional=cond,
                sort_action="native", page_size=20, page_action="native",
            )

        calls_table = make_chain_table(calls, "call")
        puts_table = make_chain_table(puts, "put")

        # ── IV Skew Chart ─────────────────────────────────────
        skew_fig = go.Figure()
        if "iv" in calls.columns and "strike" in calls.columns:
            skew_fig.add_trace(go.Scatter(
                x=calls["strike"], y=calls["iv"], mode="lines+markers",
                name="Call IV", line=dict(color=COLORS["accent_cyan"], width=2.5), marker=dict(size=4)))
        if "iv" in puts.columns and "strike" in puts.columns:
            skew_fig.add_trace(go.Scatter(
                x=puts["strike"], y=puts["iv"], mode="lines+markers",
                name="Put IV", line=dict(color=COLORS["accent_purple"], width=2.5), marker=dict(size=4)))
        skew_fig.add_vline(x=S, line=dict(color=COLORS["accent_orange"], width=1.5, dash="dash"),
                           annotation_text=f"Spot {S:.0f}", annotation_font=dict(color=COLORS["accent_orange"], size=10))
        skew_fig.update_layout(
            title=dict(text="IV Skew", font=dict(color=COLORS["text_primary"], size=13)),
            xaxis_title="Strike", yaxis_title="IV (%)",
            paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"], margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(font=dict(color=COLORS["text_secondary"], size=10), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=tpl["hoverlabel"],
            xaxis=dict(gridcolor="rgba(30,42,69,0.5)"), yaxis=dict(gridcolor="rgba(30,42,69,0.5)"))

        # ── OI / Volume Chart ─────────────────────────────────
        oi_fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "open_interest" in calls.columns:
            oi_fig.add_trace(go.Bar(x=calls["strike"], y=calls["open_interest"], name="Call OI",
                                     marker_color=COLORS["accent_cyan"], opacity=0.6), secondary_y=False)
            oi_fig.add_trace(go.Bar(x=puts["strike"], y=puts["open_interest"], name="Put OI",
                                     marker_color=COLORS["accent_purple"], opacity=0.6), secondary_y=False)
        if "volume" in calls.columns:
            oi_fig.add_trace(go.Scatter(x=calls["strike"], y=calls["volume"], name="Call Vol",
                                         mode="lines", line=dict(color=COLORS["accent_green"], width=2)), secondary_y=True)
            oi_fig.add_trace(go.Scatter(x=puts["strike"], y=puts["volume"], name="Put Vol",
                                         mode="lines", line=dict(color=COLORS["accent_orange"], width=2)), secondary_y=True)
        oi_fig.add_vline(x=S, line=dict(color=COLORS["accent_orange"], width=1, dash="dash"))
        oi_fig.update_layout(
            title=dict(text="Open Interest & Volume", font=dict(color=COLORS["text_primary"], size=13)),
            paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"], margin=dict(l=50, r=50, t=40, b=40), barmode="group",
            legend=dict(font=dict(color=COLORS["text_secondary"], size=10), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=tpl["hoverlabel"],
            xaxis=dict(gridcolor="rgba(30,42,69,0.5)"))
        oi_fig.update_yaxes(title_text="Open Interest", gridcolor="rgba(30,42,69,0.5)", secondary_y=False)
        oi_fig.update_yaxes(title_text="Volume", gridcolor="rgba(30,42,69,0.5)", secondary_y=True)

        # ── Volume Heatmap ────────────────────────────────────
        all_strikes = sorted(chain_df["strike"].unique())
        types = ["call", "put"]
        vol_matrix = np.zeros((2, len(all_strikes)))
        for j, K in enumerate(all_strikes):
            for i, otype in enumerate(types):
                row = chain_df[(chain_df["strike"] == K) & (chain_df["type"] == otype)]
                if not row.empty and "volume" in row.columns:
                    vol_matrix[i, j] = row.iloc[0]["volume"]

        hm = go.Figure(go.Heatmap(
            x=[f"{K:.0f}" for K in all_strikes], y=["CALL", "PUT"], z=vol_matrix,
            colorscale=[[0, COLORS["bg_card"]], [0.5, COLORS["accent_blue"]], [1, COLORS["accent_cyan"]]],
            colorbar=dict(title=dict(text="Volume", font=dict(color=COLORS["text_muted"])),
                          tickfont=dict(color=COLORS["text_muted"])),
            hovertemplate="K=%{x}<br>%{y}<br>Vol=%{z:,.0f}<extra></extra>"))
        hm.update_layout(paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], margin=dict(l=50, r=20, t=20, b=40), hoverlabel=tpl["hoverlabel"])

        return stats, calls_table, puts_table, skew_fig, oi_fig, hm
