"""
Advanced Analytics Panel
========================
Volatility cones, regime detection, cross-asset correlation,
margin estimation, and earnings/events calendar.
"""

from dash import html, dcc, dash_table, Input, Output
import plotly.graph_objects as go
import numpy as np

from core.analytics import (
    volatility_cone, detect_vol_regime, compute_correlation_matrix,
    generate_multi_asset_prices, estimate_margin, get_upcoming_events,
)
from core.pricing import generate_price_history
from core.bloomberg import get_spot_prices
from core.portfolio import get_all_positions
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, make_stat_style, TABLE_HEADER_STYLE, TABLE_CELL_STYLE,
)


TICKER_LIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "GS", "XOM", "GLD"]


def layout():
    return html.Div([
        # ── Controls ─────────────────────────────────────────────
        html.Div([
            html.Div("ADVANCED ANALYTICS", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Label("TICKER", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="ana-ticker",
                        options=[{"label": t, "value": t} for t in TICKER_LIST],
                        value="SPY", clearable=False, style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "120px", "marginRight": "14px"}),
                html.Div([
                    html.Label("HISTORY (DAYS)", style=LABEL_STYLE),
                    dcc.Input(id="ana-days", type="number", value=504, step=10, min=100,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "120px", "marginRight": "14px"}),
                html.Div([
                    html.Label("CORR ASSETS", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="ana-corr-tickers",
                        options=[{"label": t, "value": t} for t in TICKER_LIST],
                        value=["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "GLD"],
                        multi=True, style={"fontSize": "12px"},
                    ),
                ], style={"flex": "2", "minWidth": "200px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Vol Regime Badge ─────────────────────────────────────
        html.Div(id="ana-regime", style={"marginBottom": "16px"}),

        # ── Row: Vol Cone + Correlation ──────────────────────────
        html.Div([
            html.Div([dcc.Graph(id="ana-vol-cone", style={"height": "420px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
            html.Div([dcc.Graph(id="ana-correlation", style={"height": "420px"})],
                     style={**CARD_STYLE, "flex": "1", "minWidth": "400px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Row: Margin + Events ─────────────────────────────────
        html.Div([
            html.Div([
                html.Div("MARGIN ESTIMATION", style=CARD_HEADER_STYLE),
                html.Div(id="ana-margin"),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "350px"}, className="dashboard-card"),
            html.Div([
                html.Div("EARNINGS & EVENTS CALENDAR", style=CARD_HEADER_STYLE),
                html.Div(id="ana-events"),
            ], style={**CARD_STYLE, "flex": "1.5", "minWidth": "450px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("ana-regime", "children"),
         Output("ana-vol-cone", "figure"),
         Output("ana-correlation", "figure"),
         Output("ana-margin", "children"),
         Output("ana-events", "children")],
        [Input("ana-ticker", "value"),
         Input("ana-days", "value"),
         Input("ana-corr-tickers", "value")],
    )
    def update_analytics(ticker, days, corr_tickers):
        ticker = ticker or "SPY"
        days = days or 504
        corr_tickers = corr_tickers or ["SPY", "QQQ"]

        # ── Generate price history ───────────────────────────────
        from core.bloomberg import _FALLBACK_TICKERS
        info = _FALLBACK_TICKERS.get(ticker, {"price": 100, "vol": 0.20})
        prices = generate_price_history(S=info["price"], sigma=info["vol"],
                                        days=days, seed=hash(ticker) % 2**31)

        # ── Vol Regime ───────────────────────────────────────────
        regime = detect_vol_regime(prices)
        regime_display = html.Div([
            html.Div([
                html.Div([
                    html.Span("VOL REGIME", style={
                        "color": COLORS["text_muted"], "fontSize": "10px",
                        "letterSpacing": "1.5px", "marginRight": "16px",
                    }),
                    html.Span(regime["regime"], style={
                        "color": regime["color"], "fontSize": "18px", "fontWeight": "800",
                        "letterSpacing": "2px",
                        "textShadow": f"0 0 20px {regime['color']}60",
                    }),
                    html.Span(f"  {regime['trend']}", style={
                        "color": COLORS["text_secondary"], "fontSize": "12px",
                        "marginLeft": "14px",
                    }),
                ], style={"display": "flex", "alignItems": "center"}),
                html.Div([
                    html.Span(f"RV {regime['rv_short']:.1f}% ({int(10)}d)", style={
                        "color": COLORS["text_secondary"], "fontSize": "11px", "marginRight": "20px",
                    }),
                    html.Span(f"RV {regime['rv_long']:.1f}% ({int(60)}d)", style={
                        "color": COLORS["text_secondary"], "fontSize": "11px", "marginRight": "20px",
                    }),
                    html.Span(f"Ratio {regime['ratio']:.2f}x", style={
                        "color": COLORS["accent_cyan"], "fontSize": "11px", "marginRight": "20px",
                    }),
                    html.Span(f"Percentile {regime['percentile']:.0f}%", style={
                        "color": COLORS["accent_blue"], "fontSize": "11px",
                    }),
                ], style={"marginTop": "6px"}),
            ], style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "flexWrap": "wrap",
            }),
        ], style={
            **CARD_STYLE,
            "borderLeft": f"4px solid {regime['color']}",
            "boxShadow": f"0 0 20px {regime['color']}15",
        })

        # ── Volatility Cone ──────────────────────────────────────
        cone_df = volatility_cone(prices)

        vcf = go.Figure()
        if not cone_df.empty:
            windows = cone_df["window"].values
            # Percentile bands
            for pct, name, color, opacity in [
                ("max", "Max", COLORS["accent_red"], 0.1),
                ("p90", "P90", COLORS["accent_orange"], 0.12),
                ("p75", "P75", COLORS["accent_purple"], 0.12),
            ]:
                vcf.add_trace(go.Scatter(
                    x=windows, y=cone_df[pct], mode="lines",
                    name=name, line=dict(color=color, width=1, dash="dot"),
                ))

            vcf.add_trace(go.Scatter(
                x=windows, y=cone_df["median"], mode="lines",
                name="Median", line=dict(color=COLORS["accent_blue"], width=2),
            ))

            for pct, name, color in [
                ("p25", "P25", COLORS["accent_purple"]),
                ("p10", "P10", COLORS["accent_orange"]),
                ("min", "Min", COLORS["accent_red"]),
            ]:
                vcf.add_trace(go.Scatter(
                    x=windows, y=cone_df[pct], mode="lines",
                    name=name, line=dict(color=color, width=1, dash="dot"),
                ))

            # Current RV
            vcf.add_trace(go.Scatter(
                x=windows, y=cone_df["current"], mode="lines+markers",
                name="Current RV", line=dict(color=COLORS["accent_cyan"], width=3),
                marker=dict(size=8, symbol="diamond"),
            ))

            # Fill between p25-p75
            vcf.add_trace(go.Scatter(
                x=np.concatenate([windows, windows[::-1]]),
                y=np.concatenate([cone_df["p75"].values, cone_df["p25"].values[::-1]]),
                fill="toself", fillcolor=f"{COLORS['accent_blue']}15",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

        vcf.update_layout(
            **CHART_TEMPLATE["layout"],
            title=f"VOLATILITY CONE — {ticker}",
            xaxis_title="Window (days)",
            yaxis_title="Realized Vol (%)",
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center",
                       font={"size": 9, "color": COLORS["text_secondary"]}),
        )

        # ── Correlation Matrix ───────────────────────────────────
        multi_prices = generate_multi_asset_prices(corr_tickers, days=min(days, 252))
        corr_df, rv_df = compute_correlation_matrix(multi_prices)

        cf = go.Figure()
        if not corr_df.empty:
            cf.add_trace(go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale=[
                    [0, COLORS["accent_red"]],
                    [0.5, COLORS["bg_card"]],
                    [1, COLORS["accent_green"]],
                ],
                zmin=-1, zmax=1,
                text=np.round(corr_df.values, 2),
                texttemplate="%{text:.2f}",
                textfont={"size": 11, "color": COLORS["text_primary"]},
                hoverongaps=False,
                colorbar=dict(
                    title="Corr", titleside="right",
                    tickfont={"color": COLORS["text_muted"], "size": 10},
                ),
            ))

        cf.update_layout(
            **CHART_TEMPLATE["layout"],
            title="CROSS-ASSET CORRELATION",
            xaxis=dict(side="bottom", tickfont={"size": 11}),
            yaxis=dict(autorange="reversed", tickfont={"size": 11}),
        )

        # ── Margin Estimation ────────────────────────────────────
        positions = get_all_positions()
        tickers_in_port = list(set(p.get("ticker", "SPY") for p in positions))
        spot_data = get_spot_prices(tickers_in_port) if tickers_in_port else {}
        spot_prices = {t: d.get("price", 100) for t, d in spot_data.items()}

        if positions:
            margin = estimate_margin(positions, spot_prices)
        else:
            margin = {"initial_margin": 0, "maintenance_margin": 0, "worst_case_loss": 0,
                      "worst_scenario": "N/A", "short_option_margin": 0, "margin_utilization": 0}

        def mrow(label, value, fmt="$,.0f", color=COLORS["text_primary"]):
            return html.Div([
                html.Span(label, style={"color": COLORS["text_secondary"], "fontSize": "11px", "flex": "1"}),
                html.Span(f"{value:{fmt}}" if isinstance(value, (int, float)) else value, style={
                    "color": color, "fontSize": "13px", "fontWeight": "600",
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "padding": "8px 0", "borderBottom": f"1px solid {COLORS['border_subtle']}"})

        margin_display = html.Div([
            mrow("Initial Margin", margin["initial_margin"], color=COLORS["accent_orange"]),
            mrow("Maintenance Margin", margin["maintenance_margin"], color=COLORS["accent_cyan"]),
            mrow("Worst-Case Loss", margin["worst_case_loss"], color=COLORS["accent_red"]),
            mrow("Worst Scenario", margin.get("worst_scenario", "N/A"), fmt="s", color=COLORS["text_primary"]),
            mrow("Short Option Margin", margin["short_option_margin"], color=COLORS["accent_purple"]),
            mrow("Margin Utilization", margin.get("margin_utilization", 0), fmt=".1f", color=COLORS["accent_blue"]),
        ])

        # ── Events Calendar ──────────────────────────────────────
        events = get_upcoming_events(corr_tickers[:6])
        event_rows = []
        for ev in events[:15]:
            event_rows.append({
                "Ticker": ev["ticker"],
                "Event": ev["event"],
                "Date": ev["date"],
                "Days": ev["days_away"],
                "Move %": f"{ev['expected_move']:.1f}" if ev["expected_move"] else "-",
                "Impact": ev["impact"],
            })

        events_table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in ["Ticker", "Event", "Date", "Days", "Move %", "Impact"]],
            data=event_rows,
            style_header=TABLE_HEADER_STYLE,
            style_cell=TABLE_CELL_STYLE,
            style_data_conditional=[
                {"if": {"filter_query": '{Impact} = "HIGH"'}, "color": COLORS["accent_red"], "fontWeight": "700"},
                {"if": {"filter_query": '{Impact} = "MEDIUM"'}, "color": COLORS["accent_orange"]},
                {"if": {"filter_query": '{Impact} = "LOW"'}, "color": COLORS["text_muted"]},
                {"if": {"filter_query": '{Event} = "Earnings"'}, "backgroundColor": f"{COLORS['accent_orange']}10"},
            ],
            page_size=15,
            sort_action="native",
            style_table={"overflowX": "auto"},
        )

        return regime_display, vcf, cf, margin_display, events_table
