"""
Trade Blotter Panel
===================
Trade entry, position management, execution log, and portfolio P&L tracking.
"""

from dash import html, dcc, dash_table, Input, Output, State, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

from core.pricing import bs_price, compute_all_greeks
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE, BUTTON_STYLE, BUTTON_DANGER_STYLE,
    BUTTON_SUCCESS_STYLE,
)


# ── Generate sample trade history ─────────────────────────────────────────
def generate_sample_trades():
    np.random.seed(42)
    now = datetime(2026, 3, 20, 15, 30)
    trades = []
    tickers = ["SPY", "AAPL", "TSLA", "MSFT", "NVDA", "QQQ", "AMZN", "META"]

    for i in range(50):
        ticker = np.random.choice(tickers)
        is_call = np.random.random() > 0.45
        is_buy = np.random.random() > 0.45
        spot = {"SPY": 520, "AAPL": 178, "TSLA": 175, "MSFT": 415,
                "NVDA": 880, "QQQ": 445, "AMZN": 185, "META": 505}[ticker]
        strike = round(spot * np.random.uniform(0.9, 1.1), 0)
        price = max(0.10, round(np.random.lognormal(1.0, 0.8), 2))
        qty = np.random.choice([1, 2, 5, 10, 20, 50])
        timestamp = now - timedelta(hours=np.random.uniform(0, 72))

        trades.append({
            "id": f"TRD-{10000 + i}",
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "type": "CALL" if is_call else "PUT",
            "strike": strike,
            "expiry": (now + timedelta(days=np.random.choice([7, 14, 30, 45, 60, 90]))).strftime("%Y-%m-%d"),
            "side": "BUY" if is_buy else "SELL",
            "qty": qty if is_buy else -qty,
            "price": price,
            "notional": round(price * qty * 100, 2),
            "status": np.random.choice(["FILLED", "FILLED", "FILLED", "PARTIAL", "WORKING"]),
            "venue": np.random.choice(["CBOE", "ISE", "PHLX", "ARCA", "BATS", "MIAX"]),
            "iv": round(np.random.uniform(0.15, 0.60) * 100, 1),
        })

    return sorted(trades, key=lambda x: x["timestamp"], reverse=True)


def layout():
    return html.Div([
        # ── Trade Entry Form ──────────────────────────────────────
        html.Div([
            html.Div("TRADE ENTRY", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Label("TICKER", style=LABEL_STYLE),
                    dcc.Input(id="blotter-ticker", type="text", value="SPY",
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "100px", "marginRight": "12px"}),

                html.Div([
                    html.Label("OPTION TYPE", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="blotter-type",
                        options=[{"label": "CALL", "value": "CALL"},
                                 {"label": "PUT", "value": "PUT"}],
                        value="CALL", clearable=False,
                        style={"fontSize": "12px", "marginBottom": "8px"},
                    ),
                ], style={"flex": "1", "minWidth": "100px", "marginRight": "12px"}),

                html.Div([
                    html.Label("SIDE", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="blotter-side",
                        options=[{"label": "BUY", "value": "BUY"},
                                 {"label": "SELL", "value": "SELL"}],
                        value="BUY", clearable=False,
                        style={"fontSize": "12px", "marginBottom": "8px"},
                    ),
                ], style={"flex": "1", "minWidth": "100px", "marginRight": "12px"}),

                html.Div([
                    html.Label("STRIKE", style=LABEL_STYLE),
                    dcc.Input(id="blotter-strike", type="number", value=520, step=1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "100px", "marginRight": "12px"}),

                html.Div([
                    html.Label("QTY", style=LABEL_STYLE),
                    dcc.Input(id="blotter-qty", type="number", value=10, step=1, min=1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "80px", "marginRight": "12px"}),

                html.Div([
                    html.Label("PRICE", style=LABEL_STYLE),
                    dcc.Input(id="blotter-price", type="number", value=5.50, step=0.01,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "100px", "marginRight": "12px"}),

                html.Div([
                    html.Label("EXPIRY", style=LABEL_STYLE),
                    dcc.Input(id="blotter-expiry", type="text", value="2026-04-17",
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "130px", "marginRight": "12px"}),

                html.Div([
                    html.Label("\u00A0", style=LABEL_STYLE),
                    html.Button("SUBMIT ORDER", id="blotter-submit",
                                style={**BUTTON_SUCCESS_STYLE, "width": "100%", "marginTop": "0px"}),
                ], style={"flex": "1", "minWidth": "140px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Summary Stats ─────────────────────────────────────────
        html.Div(id="blotter-stats", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap",
        }),

        # ── Charts Row ────────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Graph(id="blotter-volume-chart", style={"height": "350px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"},
                className="dashboard-card"),
            html.Div([
                dcc.Graph(id="blotter-pnl-chart", style={"height": "350px"},
                          config={"displayModeBar": True}),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px"},
                className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Execution Log ─────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("EXECUTION LOG", style={**CARD_HEADER_STYLE, "display": "inline", "borderBottom": "none", "paddingBottom": "0"}),
                html.Span(f"  ({len(generate_sample_trades())} trades)", style={
                    "color": COLORS["text_muted"], "fontSize": "12px",
                    "fontFamily": "monospace",
                }),
            ], style={**CARD_HEADER_STYLE}),

            html.Div([
                html.Div([
                    html.Label("FILTER TICKER", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="blotter-filter-ticker",
                        options=[{"label": "ALL", "value": "ALL"}] +
                                [{"label": t, "value": t} for t in
                                 ["SPY", "AAPL", "TSLA", "MSFT", "NVDA", "QQQ", "AMZN", "META"]],
                        value="ALL", clearable=False,
                        style={"fontSize": "12px", "width": "140px"},
                    ),
                ], style={"marginRight": "16px"}),
                html.Div([
                    html.Label("FILTER STATUS", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="blotter-filter-status",
                        options=[{"label": "ALL", "value": "ALL"},
                                 {"label": "FILLED", "value": "FILLED"},
                                 {"label": "PARTIAL", "value": "PARTIAL"},
                                 {"label": "WORKING", "value": "WORKING"}],
                        value="ALL", clearable=False,
                        style={"fontSize": "12px", "width": "140px"},
                    ),
                ], style={"marginRight": "16px"}),
                html.Div([
                    html.Label("FILTER SIDE", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="blotter-filter-side",
                        options=[{"label": "ALL", "value": "ALL"},
                                 {"label": "BUY", "value": "BUY"},
                                 {"label": "SELL", "value": "SELL"}],
                        value="ALL", clearable=False,
                        style={"fontSize": "12px", "width": "120px"},
                    ),
                ]),
            ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "12px"}),

            html.Div(id="blotter-table"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # Hidden store for trade data
        dcc.Store(id="blotter-trades-store",
                  data=json.dumps(generate_sample_trades())),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("blotter-stats", "children"),
         Output("blotter-table", "children"),
         Output("blotter-volume-chart", "figure"),
         Output("blotter-pnl-chart", "figure"),
         Output("blotter-trades-store", "data")],
        [Input("blotter-filter-ticker", "value"),
         Input("blotter-filter-status", "value"),
         Input("blotter-filter-side", "value"),
         Input("blotter-submit", "n_clicks")],
        [State("blotter-trades-store", "data"),
         State("blotter-ticker", "value"),
         State("blotter-type", "value"),
         State("blotter-side", "value"),
         State("blotter-strike", "value"),
         State("blotter-qty", "value"),
         State("blotter-price", "value"),
         State("blotter-expiry", "value")],
    )
    def update_blotter(filter_ticker, filter_status, filter_side, n_clicks,
                       trades_json, new_ticker, new_type, new_side,
                       new_strike, new_qty, new_price, new_expiry):
        trades = json.loads(trades_json) if trades_json else generate_sample_trades()
        tpl = CHART_TEMPLATE["layout"]

        # ── Add new trade if submitted ────────────────────────────
        if n_clicks and n_clicks > 0:
            qty_signed = (new_qty or 10) if new_side == "BUY" else -(new_qty or 10)
            new_trade = {
                "id": f"TRD-{10050 + n_clicks}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": (new_ticker or "SPY").upper(),
                "type": new_type or "CALL",
                "strike": new_strike or 520,
                "expiry": new_expiry or "2026-04-17",
                "side": new_side or "BUY",
                "qty": qty_signed,
                "price": new_price or 5.50,
                "notional": round((new_price or 5.50) * abs(qty_signed) * 100, 2),
                "status": "FILLED",
                "venue": "CBOE",
                "iv": 25.0,
            }
            trades = [new_trade] + trades

        # ── Filter ────────────────────────────────────────────────
        filtered = trades
        if filter_ticker and filter_ticker != "ALL":
            filtered = [t for t in filtered if t["ticker"] == filter_ticker]
        if filter_status and filter_status != "ALL":
            filtered = [t for t in filtered if t["status"] == filter_status]
        if filter_side and filter_side != "ALL":
            filtered = [t for t in filtered if t["side"] == filter_side]

        # ── Summary Stats ─────────────────────────────────────────
        total_notional = sum(t["notional"] for t in filtered)
        total_buys = sum(1 for t in filtered if t["qty"] > 0)
        total_sells = sum(1 for t in filtered if t["qty"] < 0)
        total_contracts = sum(abs(t["qty"]) for t in filtered)
        unique_tickers = len(set(t["ticker"] for t in filtered))
        avg_iv = np.mean([t["iv"] for t in filtered]) if filtered else 0
        filled_pct = sum(1 for t in filtered if t["status"] == "FILLED") / max(len(filtered), 1) * 100

        def stat_box(label, value, color=COLORS["accent_cyan"]):
            return html.Div([
                html.Div(str(value), className="stat-value",
                         style={"color": color, "fontSize": "18px"}),
                html.Div(label, className="stat-label"),
            ], style={**STAT_BOX_STYLE, "flex": "1", "minWidth": "100px"})

        stats = [
            stat_box("TOTAL TRADES", len(filtered), COLORS["accent_cyan"]),
            stat_box("BUYS", total_buys, COLORS["accent_green"]),
            stat_box("SELLS", total_sells, COLORS["accent_red"]),
            stat_box("CONTRACTS", f"{total_contracts:,}", COLORS["accent_blue"]),
            stat_box("NOTIONAL", f"${total_notional:,.0f}", COLORS["accent_purple"]),
            stat_box("TICKERS", unique_tickers, COLORS["accent_orange"]),
            stat_box("AVG IV", f"{avg_iv:.1f}%", COLORS["accent_pink"]),
            stat_box("FILL RATE", f"{filled_pct:.0f}%", COLORS["accent_green"]),
        ]

        # ── Table Data ────────────────────────────────────────────
        table_data = []
        for t in filtered:
            table_data.append({
                "ID": t["id"],
                "Time": t["timestamp"],
                "Ticker": t["ticker"],
                "Type": t["type"],
                "Strike": t["strike"],
                "Expiry": t["expiry"],
                "Side": t["side"],
                "Qty": t["qty"],
                "Price": f"${t['price']:.2f}",
                "Notional": f"${t['notional']:,.0f}",
                "IV": f"{t['iv']}%",
                "Status": t["status"],
                "Venue": t["venue"],
            })

        trade_table = dash_table.DataTable(
            data=table_data,
            columns=[{"name": c, "id": c} for c in table_data[0].keys()] if table_data else [],
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
                "minWidth": "65px",
                "maxWidth": "130px",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg_secondary"]},
                {"if": {"filter_query": "{Side} = BUY", "column_id": "Side"},
                 "color": COLORS["accent_green"], "fontWeight": "600"},
                {"if": {"filter_query": "{Side} = SELL", "column_id": "Side"},
                 "color": COLORS["accent_red"], "fontWeight": "600"},
                {"if": {"filter_query": "{Status} = FILLED", "column_id": "Status"},
                 "color": COLORS["accent_green"]},
                {"if": {"filter_query": "{Status} = WORKING", "column_id": "Status"},
                 "color": COLORS["accent_orange"]},
                {"if": {"filter_query": "{Status} = PARTIAL", "column_id": "Status"},
                 "color": COLORS["accent_blue"]},
                {"if": {"filter_query": "{Type} = CALL", "column_id": "Type"},
                 "color": COLORS["accent_cyan"]},
                {"if": {"filter_query": "{Type} = PUT", "column_id": "Type"},
                 "color": COLORS["accent_purple"]},
            ],
            sort_action="native",
            filter_action="native",
            page_size=15,
            page_action="native",
        )

        # ── Volume by Ticker Chart ────────────────────────────────
        ticker_vol = {}
        for t in filtered:
            ticker_vol[t["ticker"]] = ticker_vol.get(t["ticker"], 0) + abs(t["qty"])
        tickers_sorted = sorted(ticker_vol, key=ticker_vol.get, reverse=True)

        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(
            x=tickers_sorted,
            y=[ticker_vol[t] for t in tickers_sorted],
            marker=dict(
                color=[COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_purple"],
                       COLORS["accent_green"], COLORS["accent_orange"], COLORS["accent_pink"],
                       COLORS["accent_red"], COLORS["text_secondary"]][:len(tickers_sorted)],
                line=dict(width=0),
            ),
            hovertemplate="%{x}: %{y} contracts<extra></extra>",
        ))
        vol_fig.update_layout(
            title=dict(text="Volume by Ticker", font=dict(color=COLORS["text_primary"], size=13)),
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            margin=dict(l=50, r=20, t=40, b=40),
            xaxis=dict(gridcolor=COLORS["border_subtle"]),
            yaxis=dict(title="Contracts", gridcolor=COLORS["border_subtle"]),
            hoverlabel=tpl["hoverlabel"],
        )

        # ── Cumulative Notional Chart ─────────────────────────────
        sorted_by_time = sorted(filtered, key=lambda x: x["timestamp"])
        cum_notional = np.cumsum([t["notional"] * (1 if t["qty"] > 0 else -1)
                                  for t in sorted_by_time])
        timestamps = [t["timestamp"] for t in sorted_by_time]

        pnl_fig = go.Figure()
        pnl_fig.add_trace(go.Scatter(
            x=timestamps, y=cum_notional, mode="lines",
            line=dict(color=COLORS["accent_cyan"], width=2),
            fill="tozeroy", fillcolor="rgba(34,211,238,0.06)",
            hovertemplate="%{x}<br>Cum. Flow: $%{y:,.0f}<extra></extra>",
        ))
        pnl_fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
        pnl_fig.update_layout(
            title=dict(text="Cumulative Net Flow", font=dict(
                color=COLORS["text_primary"], size=13)),
            paper_bgcolor=tpl["paper_bgcolor"],
            plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"],
            margin=dict(l=60, r=20, t=40, b=40),
            xaxis=dict(gridcolor=COLORS["border_subtle"]),
            yaxis=dict(title="Net Flow ($)", gridcolor=COLORS["border_subtle"]),
            hoverlabel=tpl["hoverlabel"],
        )

        return stats, trade_table, vol_fig, pnl_fig, json.dumps(trades)
