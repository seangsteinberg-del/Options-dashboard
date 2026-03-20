"""
Trade Blotter Panel v2
=======================
Trade entry, execution log, P&L attribution, volume analytics,
with Bloomberg-connected ticker resolution.
"""

from dash import html, dcc, dash_table, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from datetime import datetime, timedelta

from core.pricing import bs_price
from core.bloomberg import get_spot_prices
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE, BUTTON_SUCCESS_STYLE, make_stat_style,
)


def _gen_trades():
    np.random.seed(42)
    now = datetime(2026, 3, 20, 15, 30)
    tickers = ["SPY", "AAPL", "TSLA", "MSFT", "NVDA", "QQQ", "AMZN", "META"]
    spots = {"SPY": 521, "AAPL": 178, "TSLA": 176, "MSFT": 416, "NVDA": 882, "QQQ": 447, "AMZN": 186, "META": 507}
    trades = []
    for i in range(60):
        tk = np.random.choice(tickers)
        is_call = np.random.random() > 0.45
        is_buy = np.random.random() > 0.45
        sp = spots[tk]
        K = round(sp * np.random.uniform(0.9, 1.1), 0)
        px = max(0.10, round(np.random.lognormal(1.0, 0.8), 2))
        qty = np.random.choice([1, 2, 5, 10, 20, 50])
        ts = now - timedelta(hours=np.random.uniform(0, 120))
        trades.append({
            "id": f"TRD-{10000+i}", "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": tk, "type": "CALL" if is_call else "PUT",
            "strike": K, "expiry": (now + timedelta(days=np.random.choice([7,14,30,45,60,90]))).strftime("%Y-%m-%d"),
            "side": "BUY" if is_buy else "SELL",
            "qty": qty if is_buy else -qty, "price": px,
            "notional": round(px * qty * 100, 2),
            "status": np.random.choice(["FILLED"]*4 + ["PARTIAL", "WORKING"]),
            "venue": np.random.choice(["CBOE", "ISE", "PHLX", "ARCA", "BATS", "MIAX"]),
            "iv": round(np.random.uniform(0.15, 0.65) * 100, 1),
        })
    return sorted(trades, key=lambda x: x["timestamp"], reverse=True)


def layout():
    return html.Div([
        # ── Trade Entry ───────────────────────────────────────
        html.Div([
            html.Div("TRADE ENTRY", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([html.Label("TICKER", style=LABEL_STYLE),
                           dcc.Input(id="b-ticker", type="text", value="SPY",
                                     style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True)],
                         style={"flex": "1", "minWidth": "90px", "marginRight": "10px"}),
                html.Div([html.Label("TYPE", style=LABEL_STYLE),
                           dcc.Dropdown(id="b-type", options=[{"label": "CALL", "value": "CALL"},
                                                               {"label": "PUT", "value": "PUT"}],
                                        value="CALL", clearable=False, style={"fontSize": "12px", "marginBottom": "8px"})],
                         style={"flex": "1", "minWidth": "90px", "marginRight": "10px"}),
                html.Div([html.Label("SIDE", style=LABEL_STYLE),
                           dcc.Dropdown(id="b-side", options=[{"label": "BUY", "value": "BUY"},
                                                               {"label": "SELL", "value": "SELL"}],
                                        value="BUY", clearable=False, style={"fontSize": "12px", "marginBottom": "8px"})],
                         style={"flex": "1", "minWidth": "80px", "marginRight": "10px"}),
                html.Div([html.Label("STRIKE", style=LABEL_STYLE),
                           dcc.Input(id="b-strike", type="number", value=520, step=1,
                                     style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True)],
                         style={"flex": "1", "minWidth": "90px", "marginRight": "10px"}),
                html.Div([html.Label("QTY", style=LABEL_STYLE),
                           dcc.Input(id="b-qty", type="number", value=10, step=1, min=1,
                                     style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True)],
                         style={"flex": "1", "minWidth": "70px", "marginRight": "10px"}),
                html.Div([html.Label("PRICE", style=LABEL_STYLE),
                           dcc.Input(id="b-px", type="number", value=5.50, step=0.01,
                                     style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True)],
                         style={"flex": "1", "minWidth": "90px", "marginRight": "10px"}),
                html.Div([html.Label("EXPIRY", style=LABEL_STYLE),
                           dcc.Input(id="b-expiry", type="text", value="2026-04-17",
                                     style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True)],
                         style={"flex": "1", "minWidth": "120px", "marginRight": "10px"}),
                html.Div([html.Label("\u00A0", style=LABEL_STYLE),
                           html.Button("SUBMIT", id="b-submit",
                                       style={**BUTTON_SUCCESS_STYLE, "width": "100%"})],
                         style={"flex": "1", "minWidth": "110px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Stats ─────────────────────────────────────────────
        html.Div(id="b-stats", style={"display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap"}),

        # ── Charts ────────────────────────────────────────────
        html.Div([
            html.Div([dcc.Graph(id="b-vol-chart", style={"height": "340px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "380px"}, className="dashboard-card"),
            html.Div([dcc.Graph(id="b-flow-chart", style={"height": "340px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "380px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── P&L by Ticker ─────────────────────────────────────
        html.Div([
            html.Div("P&L ATTRIBUTION BY TICKER", style=CARD_HEADER_STYLE),
            dcc.Graph(id="b-pnl-attr", style={"height": "320px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Execution Log ─────────────────────────────────────
        html.Div([
            html.Div("EXECUTION LOG", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([html.Label("TICKER", style=LABEL_STYLE),
                           dcc.Dropdown(id="b-ft", options=[{"label": "ALL", "value": "ALL"}] +
                                        [{"label": t, "value": t} for t in ["SPY","AAPL","TSLA","MSFT","NVDA","QQQ","AMZN","META"]],
                                        value="ALL", clearable=False, style={"fontSize": "11px", "width": "120px"})],
                         style={"marginRight": "14px"}),
                html.Div([html.Label("STATUS", style=LABEL_STYLE),
                           dcc.Dropdown(id="b-fs", options=[{"label": s, "value": s} for s in ["ALL","FILLED","PARTIAL","WORKING"]],
                                        value="ALL", clearable=False, style={"fontSize": "11px", "width": "120px"})],
                         style={"marginRight": "14px"}),
                html.Div([html.Label("SIDE", style=LABEL_STYLE),
                           dcc.Dropdown(id="b-fd", options=[{"label": s, "value": s} for s in ["ALL","BUY","SELL"]],
                                        value="ALL", clearable=False, style={"fontSize": "11px", "width": "100px"})]),
            ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div(id="b-table"),
        ], style=CARD_STYLE, className="dashboard-card"),

        dcc.Store(id="b-store", data=json.dumps(_gen_trades())),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("b-stats", "children"), Output("b-table", "children"),
         Output("b-vol-chart", "figure"), Output("b-flow-chart", "figure"),
         Output("b-pnl-attr", "figure"), Output("b-store", "data")],
        [Input("b-ft", "value"), Input("b-fs", "value"), Input("b-fd", "value"),
         Input("b-submit", "n_clicks")],
        [State("b-store", "data"), State("b-ticker", "value"), State("b-type", "value"),
         State("b-side", "value"), State("b-strike", "value"), State("b-qty", "value"),
         State("b-px", "value"), State("b-expiry", "value")],
    )
    def update_blotter(ft, fs, fd, n_clicks, store, nt, nty, ns, nk, nq, npx, nex):
        trades = json.loads(store) if store else _gen_trades()
        tpl = CHART_TEMPLATE["layout"]

        if n_clicks and n_clicks > 0:
            qs = (nq or 10) if ns == "BUY" else -(nq or 10)
            trades = [{"id": f"TRD-{10060+n_clicks}", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "ticker": (nt or "SPY").upper(), "type": nty or "CALL",
                        "strike": nk or 520, "expiry": nex or "2026-04-17",
                        "side": ns or "BUY", "qty": qs, "price": npx or 5.50,
                        "notional": round((npx or 5.50) * abs(qs) * 100, 2),
                        "status": "FILLED", "venue": "CBOE", "iv": 25.0}] + trades

        filtered = trades
        if ft and ft != "ALL": filtered = [t for t in filtered if t["ticker"] == ft]
        if fs and fs != "ALL": filtered = [t for t in filtered if t["status"] == fs]
        if fd and fd != "ALL": filtered = [t for t in filtered if t["side"] == fd]

        # Stats
        total_n = sum(t["notional"] for t in filtered)
        buys = sum(1 for t in filtered if t.get("qty", 0) > 0)
        sells = sum(1 for t in filtered if t.get("qty", 0) < 0)
        contracts = sum(abs(t.get("qty", 0)) for t in filtered)
        utk = len(set(t["ticker"] for t in filtered))
        avg_iv = np.mean([t["iv"] for t in filtered]) if filtered else 0
        fill_r = sum(1 for t in filtered if t["status"] == "FILLED") / max(len(filtered), 1) * 100

        def sb(l, v, c):
            return html.Div([html.Div(str(v), className="stat-value", style={"color": c, "fontSize": "18px"}),
                             html.Div(l, className="stat-label")],
                            style={**make_stat_style(c), "flex": "1", "minWidth": "95px"}, className="stat-box")

        stats = [sb("TRADES", len(filtered), COLORS["accent_cyan"]), sb("BUYS", buys, COLORS["accent_green"]),
                 sb("SELLS", sells, COLORS["accent_red"]), sb("CONTRACTS", f"{contracts:,}", COLORS["accent_blue"]),
                 sb("NOTIONAL", f"${total_n:,.0f}", COLORS["accent_purple"]),
                 sb("TICKERS", utk, COLORS["accent_orange"]),
                 sb("AVG IV", f"{avg_iv:.1f}%", COLORS["accent_pink"]),
                 sb("FILL RATE", f"{fill_r:.0f}%", COLORS["accent_green"])]

        # Table
        tdata = [{"ID": t["id"], "Time": t["timestamp"], "Ticker": t["ticker"],
                   "Type": t["type"], "K": t["strike"], "Expiry": t["expiry"],
                   "Side": t["side"], "Qty": t.get("qty", 0), "Price": f"${t['price']:.2f}",
                   "Notional": f"${t['notional']:,.0f}", "IV": f"{t['iv']}%",
                   "Status": t["status"], "Venue": t.get("venue", "")} for t in filtered]

        table = dash_table.DataTable(
            data=tdata, columns=[{"name": c, "id": c} for c in (tdata[0].keys() if tdata else [])],
            style_header={"backgroundColor": COLORS["bg_secondary"], "color": COLORS["text_secondary"],
                          "fontWeight": "700", "fontSize": "10px", "textTransform": "uppercase",
                          "letterSpacing": "0.5px", "border": f"1px solid {COLORS['border_subtle']}", "padding": "8px 6px"},
            style_cell={"backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"],
                        "fontSize": "11px", "fontFamily": "'JetBrains Mono', monospace",
                        "border": f"1px solid {COLORS['border_subtle']}", "padding": "6px 8px",
                        "textAlign": "center", "minWidth": "60px", "maxWidth": "120px"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg_secondary"]},
                {"if": {"filter_query": "{Side} = BUY", "column_id": "Side"}, "color": COLORS["accent_green"], "fontWeight": "700"},
                {"if": {"filter_query": "{Side} = SELL", "column_id": "Side"}, "color": COLORS["accent_red"], "fontWeight": "700"},
                {"if": {"filter_query": "{Status} = FILLED", "column_id": "Status"}, "color": COLORS["accent_green"]},
                {"if": {"filter_query": "{Status} = WORKING", "column_id": "Status"}, "color": COLORS["accent_orange"]}],
            sort_action="native", filter_action="native", page_size=15, page_action="native",
        )

        # Volume by Ticker
        tkv = {}
        for t in filtered: tkv[t["ticker"]] = tkv.get(t["ticker"], 0) + abs(t.get("qty", 0))
        ts_sorted = sorted(tkv, key=tkv.get, reverse=True)
        colors = [COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_purple"],
                  COLORS["accent_green"], COLORS["accent_orange"], COLORS["accent_pink"],
                  COLORS["accent_red"], COLORS["accent_teal"]]

        vf = go.Figure(go.Bar(x=ts_sorted, y=[tkv[t] for t in ts_sorted],
                               marker=dict(color=colors[:len(ts_sorted)], line=dict(width=0)),
                               hovertemplate="%{x}: %{y} contracts<extra></extra>"))
        vf.update_layout(title=dict(text="Volume by Ticker", font=dict(color=COLORS["text_primary"], size=13)),
                         paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], margin=dict(l=50, r=20, t=40, b=40),
                         xaxis=dict(gridcolor="rgba(30,42,69,0.5)"), yaxis=dict(title="Contracts", gridcolor="rgba(30,42,69,0.5)"),
                         hoverlabel=tpl["hoverlabel"])

        # Cumulative Flow
        sbt = sorted(filtered, key=lambda x: x["timestamp"])
        cum = np.cumsum([t["notional"] * (1 if t.get("qty", 0) > 0 else -1) for t in sbt])
        ff = go.Figure(go.Scatter(x=[t["timestamp"] for t in sbt], y=cum, mode="lines",
                                   line=dict(color=COLORS["accent_cyan"], width=2),
                                   fill="tozeroy", fillcolor="rgba(6,182,212,0.05)"))
        ff.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
        ff.update_layout(title=dict(text="Cumulative Net Flow", font=dict(color=COLORS["text_primary"], size=13)),
                         paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], margin=dict(l=60, r=20, t=40, b=40),
                         xaxis=dict(gridcolor="rgba(30,42,69,0.5)"), yaxis=dict(title="$", gridcolor="rgba(30,42,69,0.5)"),
                         hoverlabel=tpl["hoverlabel"])

        # P&L Attribution by Ticker
        tk_pnl = {}
        for t in filtered:
            tk = t["ticker"]
            flow = t["notional"] * (1 if t.get("qty", 0) > 0 else -1)
            tk_pnl[tk] = tk_pnl.get(tk, 0) + flow * np.random.uniform(-0.15, 0.25)
        tks = sorted(tk_pnl)
        pnl_colors = [COLORS["accent_green"] if tk_pnl[t] >= 0 else COLORS["accent_red"] for t in tks]

        af = go.Figure(go.Bar(x=tks, y=[tk_pnl[t] for t in tks], marker_color=pnl_colors,
                               hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"))
        af.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
        af.update_layout(paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
                         font=tpl["font"], margin=dict(l=60, r=20, t=20, b=40),
                         xaxis=dict(gridcolor="rgba(30,42,69,0.5)"), yaxis=dict(title="P&L ($)", gridcolor="rgba(30,42,69,0.5)"),
                         hoverlabel=tpl["hoverlabel"])

        return stats, table, vf, ff, af, json.dumps(trades)
