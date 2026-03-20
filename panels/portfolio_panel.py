"""
Portfolio Management Panel
==========================
Multi-book position management with live Greeks, P&L tracking,
risk limit monitoring, and position lifecycle (add/close).
"""

from dash import html, dcc, dash_table, Input, Output, State, no_update, ctx
import numpy as np

from core.portfolio import (
    get_books, get_positions, compute_book_risk, check_risk_limits,
    add_position, close_position, get_risk_limits, get_trade_history,
)
from core.bloomberg import get_spot_prices
from core.pricing import bs_price, compute_all_greeks
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE, make_stat_style, BUTTON_STYLE,
    BUTTON_DANGER_STYLE, BUTTON_SUCCESS_STYLE, TABLE_HEADER_STYLE, TABLE_CELL_STYLE,
)


def layout():
    books = get_books()
    return html.Div([
        # ── Book Selector + Summary ─────────────────────────────
        html.Div([
            html.Div("PORTFOLIO MANAGEMENT", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Label("BOOK", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="port-book",
                        options=[{"label": b, "value": b} for b in books],
                        value=books[0] if books else "MAIN",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "140px", "marginRight": "14px"}),
                html.Div([
                    html.Label("RATE (%)", style=LABEL_STYLE),
                    dcc.Input(id="port-rate", type="number", value=5.0, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "100px", "marginRight": "14px"}),
                html.Div([
                    html.Label("DIV YIELD (%)", style=LABEL_STYLE),
                    dcc.Input(id="port-div", type="number", value=1.5, step=0.1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "100px", "marginRight": "14px"}),
                html.Div([
                    html.Button("REFRESH", id="port-refresh-btn", n_clicks=0,
                                style=BUTTON_STYLE),
                ], style={"display": "flex", "alignItems": "flex-end", "paddingBottom": "8px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Stats Row ────────────────────────────────────────────
        html.Div(id="port-stats", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap",
        }),

        # ── Risk Limit Alerts ────────────────────────────────────
        html.Div(id="port-alerts", style={"marginBottom": "16px"}),

        # ── Position Grid ────────────────────────────────────────
        html.Div([
            html.Div("POSITIONS", style=CARD_HEADER_STYLE),
            html.Div(id="port-grid"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Add Position Form ────────────────────────────────────
        html.Div([
            html.Div("ADD POSITION", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Label("TICKER", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="port-add-ticker",
                        options=[{"label": t, "value": t} for t in
                                 ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "GS", "XOM", "GLD"]],
                        value="SPY", clearable=False, style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "100px"}),
                html.Div([
                    html.Label("STRIKE", style=LABEL_STYLE),
                    dcc.Input(id="port-add-strike", type="number", value=520, step=1,
                              style=INPUT_STYLE, debounce=True),
                ], style={"flex": "1", "minWidth": "90px"}),
                html.Div([
                    html.Label("EXPIRY (YRS)", style=LABEL_STYLE),
                    dcc.Input(id="port-add-expiry", type="number", value=0.25, step=0.01, min=0.01,
                              style=INPUT_STYLE, debounce=True),
                ], style={"flex": "1", "minWidth": "100px"}),
                html.Div([
                    html.Label("TYPE", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="port-add-type",
                        options=[{"label": "CALL", "value": "call"}, {"label": "PUT", "value": "put"}],
                        value="call", clearable=False, style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "80px"}),
                html.Div([
                    html.Label("QUANTITY", style=LABEL_STYLE),
                    dcc.Input(id="port-add-qty", type="number", value=10, step=1,
                              style=INPUT_STYLE, debounce=True),
                ], style={"flex": "1", "minWidth": "90px"}),
                html.Div([
                    html.Label("IV (%)", style=LABEL_STYLE),
                    dcc.Input(id="port-add-vol", type="number", value=20, step=1, min=1,
                              style=INPUT_STYLE, debounce=True),
                ], style={"flex": "1", "minWidth": "80px"}),
                html.Div([
                    html.Button("ADD", id="port-add-btn", n_clicks=0, style=BUTTON_SUCCESS_STYLE),
                ], style={"display": "flex", "alignItems": "flex-end", "paddingBottom": "0px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Close Position ───────────────────────────────────────
        html.Div([
            html.Div("CLOSE POSITION", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Label("POSITION ID", style=LABEL_STYLE),
                    dcc.Input(id="port-close-id", type="text", placeholder="POS-001",
                              style=INPUT_STYLE, debounce=True),
                ], style={"flex": "2", "minWidth": "140px"}),
                html.Div([
                    html.Button("CLOSE", id="port-close-btn", n_clicks=0, style=BUTTON_DANGER_STYLE),
                ], style={"display": "flex", "alignItems": "flex-end", "paddingBottom": "0px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Trade History ────────────────────────────────────────
        html.Div([
            html.Div("RECENT TRADES", style=CARD_HEADER_STYLE),
            html.Div(id="port-trades"),
        ], style=CARD_STYLE, className="dashboard-card"),

        # Hidden div for action feedback
        html.Div(id="port-action-output", style={"display": "none"}),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("port-stats", "children"),
         Output("port-alerts", "children"),
         Output("port-grid", "children"),
         Output("port-trades", "children"),
         Output("port-action-output", "children")],
        [Input("port-refresh-btn", "n_clicks"),
         Input("port-add-btn", "n_clicks"),
         Input("port-close-btn", "n_clicks")],
        [State("port-book", "value"),
         State("port-rate", "value"),
         State("port-div", "value"),
         State("port-add-ticker", "value"),
         State("port-add-strike", "value"),
         State("port-add-expiry", "value"),
         State("port-add-type", "value"),
         State("port-add-qty", "value"),
         State("port-add-vol", "value"),
         State("port-close-id", "value")],
    )
    def update_portfolio(n_refresh, n_add, n_close,
                         book, rate, div_y,
                         add_ticker, add_strike, add_expiry, add_type, add_qty, add_vol,
                         close_id):
        rate = (rate or 5.0) / 100
        div_y = (div_y or 1.5) / 100
        book = book or "MAIN"
        triggered = ctx.triggered_id

        # ── Handle add position ──────────────────────────────────
        if triggered == "port-add-btn" and add_strike and add_expiry and add_qty and add_vol:
            tickers_for_spot = [add_ticker]
            spot_data = get_spot_prices(tickers_for_spot)
            S = spot_data.get(add_ticker, {}).get("price", 100)
            entry_price = bs_price(S, add_strike, add_expiry, rate, div_y, add_vol / 100, add_type)

            add_position(book, {
                "ticker": add_ticker,
                "strike": add_strike,
                "expiry": add_expiry,
                "option_type": add_type,
                "quantity": add_qty,
                "vol": add_vol / 100,
                "entry_price": round(entry_price, 2),
            })

        # ── Handle close position ────────────────────────────────
        if triggered == "port-close-btn" and close_id:
            close_position(book, close_id.strip())

        # ── Fetch current data ───────────────────────────────────
        positions = get_positions(book)
        tickers = list(set(p.get("ticker", "SPY") for p in positions))
        spot_data = get_spot_prices(tickers) if tickers else {}
        spot_prices = {t: d.get("price", 100) for t, d in spot_data.items()}

        risk = compute_book_risk(book, spot_prices, rate, div_y)
        breaches = check_risk_limits(risk)

        # ── Stat boxes ───────────────────────────────────────────
        def sbox(label, value, fmt, color):
            return html.Div([
                html.Div(f"{value:{fmt}}", style={
                    "color": color, "fontSize": "20px", "fontWeight": "700",
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
                html.Div(label, style={
                    "color": COLORS["text_muted"], "fontSize": "9px",
                    "textTransform": "uppercase", "letterSpacing": "1px", "marginTop": "4px",
                }),
            ], style={**make_stat_style(color), "flex": "1", "minWidth": "100px"})

        pnl = risk.get("pnl", 0)
        pnl_color = COLORS["pnl_profit"] if pnl >= 0 else COLORS["pnl_loss"]

        stats = [
            sbox("Positions", len(positions), "d", COLORS["accent_cyan"]),
            sbox("Notional", risk.get("notional", 0), "$,.0f", COLORS["accent_blue"]),
            sbox("Delta", risk.get("delta", 0), "+,.0f", COLORS["accent_green"]),
            sbox("Gamma", risk.get("gamma", 0), "+,.1f", COLORS["accent_purple"]),
            sbox("Theta", risk.get("theta", 0), "+,.0f", COLORS["accent_orange"]),
            sbox("Vega", risk.get("vega", 0), "+,.0f", COLORS["accent_teal"]),
            sbox("P&L", pnl, "+$,.0f", pnl_color),
        ]

        # ── Breach alerts ────────────────────────────────────────
        alerts = []
        for b in breaches:
            sev_color = COLORS["accent_red"] if b["severity"] == "CRITICAL" else (
                COLORS["accent_orange"] if b["severity"] == "WARNING" else COLORS["accent_cyan"])
            alerts.append(html.Div([
                html.Span(f"[{b['severity']}] ", style={"color": sev_color, "fontWeight": "700"}),
                html.Span(f"{b['metric']}: ", style={"color": COLORS["text_primary"]}),
                html.Span(f"{b['current']:,.0f}", style={"color": sev_color, "fontWeight": "600"}),
                html.Span(f" / {b['limit']:,.0f} limit ", style={"color": COLORS["text_muted"]}),
                html.Span(f"({b['utilization']:.0f}%)", style={"color": sev_color}),
            ], style={
                "padding": "10px 16px", "marginBottom": "4px",
                "backgroundColor": COLORS["bg_secondary"],
                "border": f"1px solid {sev_color}40",
                "borderLeft": f"3px solid {sev_color}",
                "borderRadius": "8px",
                "fontFamily": "'JetBrains Mono', monospace", "fontSize": "12px",
            }))

        if not alerts:
            alerts = [html.Div("All risk limits within bounds", style={
                "color": COLORS["accent_green"], "fontSize": "12px",
                "fontFamily": "'JetBrains Mono', monospace",
                "padding": "10px 16px",
                "backgroundColor": COLORS["bg_secondary"],
                "border": f"1px solid {COLORS['accent_green']}30",
                "borderLeft": f"3px solid {COLORS['accent_green']}",
                "borderRadius": "8px",
            })]

        # ── Position table ───────────────────────────────────────
        rows = []
        for pos in positions:
            ticker = pos.get("ticker", "SPY")
            S = spot_prices.get(ticker, 100)
            K = pos["strike"]
            T = pos["expiry"]
            sig = pos["vol"]
            otype = pos["option_type"]
            qty = pos["quantity"]
            mult = pos.get("multiplier", 100)
            entry = pos.get("entry_price", 0)

            g = compute_all_greeks(S, K, T, rate, div_y, sig, otype)
            cur_price = g["price"]
            pos_pnl = (cur_price - entry) * qty * mult

            rows.append({
                "ID": pos.get("id", ""),
                "Ticker": ticker,
                "Type": otype.upper(),
                "K": K,
                "T": f"{T:.2f}",
                "Qty": qty,
                "Entry": f"{entry:.2f}",
                "Current": f"{cur_price:.2f}",
                "P&L": f"{pos_pnl:+,.0f}",
                "Delta": f"{g['delta'] * qty * mult:+,.0f}",
                "Gamma": f"{g['gamma'] * qty * mult:+,.1f}",
                "Theta": f"{g['theta'] * qty * mult:+,.0f}",
                "Vega": f"{g['vega'] * qty * mult:+,.0f}",
            })

        grid = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in
                     ["ID", "Ticker", "Type", "K", "T", "Qty", "Entry", "Current", "P&L",
                      "Delta", "Gamma", "Theta", "Vega"]],
            data=rows,
            style_header=TABLE_HEADER_STYLE,
            style_cell=TABLE_CELL_STYLE,
            style_data_conditional=[
                {"if": {"filter_query": '{P&L} contains "+"'}, "color": COLORS["pnl_profit"]},
                {"if": {"filter_query": '{P&L} contains "-"'}, "color": COLORS["pnl_loss"]},
                {"if": {"filter_query": '{Qty} contains "-"'}, "backgroundColor": "#1a1020"},
            ],
            sort_action="native",
            page_size=20,
            style_table={"overflowX": "auto"},
        )

        # ── Trade history ────────────────────────────────────────
        history = get_trade_history()[-10:]  # Last 10 trades
        trade_rows = []
        for t in reversed(history):
            det = t.get("details", {})
            trade_rows.append({
                "Time": t.get("timestamp", "")[:19],
                "Action": t.get("action", ""),
                "Book": t.get("book", ""),
                "Ticker": det.get("ticker", ""),
                "Type": det.get("option_type", "").upper(),
                "K": det.get("strike", ""),
                "Qty": det.get("quantity", ""),
            })

        trades_table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in ["Time", "Action", "Book", "Ticker", "Type", "K", "Qty"]],
            data=trade_rows,
            style_header=TABLE_HEADER_STYLE,
            style_cell=TABLE_CELL_STYLE,
            style_data_conditional=[
                {"if": {"filter_query": '{Action} = "OPEN"'}, "color": COLORS["accent_green"]},
                {"if": {"filter_query": '{Action} = "CLOSE"'}, "color": COLORS["accent_red"]},
            ],
            page_size=10,
            style_table={"overflowX": "auto"},
        ) if trade_rows else html.Div("No trades yet", style={"color": COLORS["text_muted"], "fontSize": "12px"})

        return stats, alerts, grid, trades_table, ""
