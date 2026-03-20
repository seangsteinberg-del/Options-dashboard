"""
Stress Testing Panel
====================
Named historical scenarios, custom shocks, position-level impact,
and multi-scenario comparison charts.
"""

from dash import html, dcc, dash_table, Input, Output, State
import plotly.graph_objects as go
import numpy as np

from core.stress import SCENARIOS, run_stress_test, run_custom_stress, compare_scenarios
from core.portfolio import get_all_positions
from core.bloomberg import get_spot_prices
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, make_stat_style, BUTTON_STYLE,
    TABLE_HEADER_STYLE, TABLE_CELL_STYLE,
)


def layout():
    scenario_opts = [{"label": f"{name} — {s['severity']}", "value": name}
                     for name, s in SCENARIOS.items()]
    scenario_opts.append({"label": "CUSTOM SCENARIO", "value": "CUSTOM"})

    return html.Div([
        # ── Controls ─────────────────────────────────────────────
        html.Div([
            html.Div("STRESS TESTING", style=CARD_HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Label("SCENARIO", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="stress-scenario",
                        options=scenario_opts,
                        value="2008 GFC",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "2", "minWidth": "200px", "marginRight": "14px"}),
                html.Div([
                    html.Label("SPOT SHOCK (%)", style=LABEL_STYLE),
                    dcc.Input(id="stress-spot", type="number", value=-20, step=1,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "110px", "marginRight": "14px"}),
                html.Div([
                    html.Label("VOL SHOCK (%)", style=LABEL_STYLE),
                    dcc.Input(id="stress-vol", type="number", value=100, step=5,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "110px", "marginRight": "14px"}),
                html.Div([
                    html.Label("RATE SHOCK (%)", style=LABEL_STYLE),
                    dcc.Input(id="stress-rate", type="number", value=-1.0, step=0.25,
                              style={**INPUT_STYLE, "marginBottom": "8px"}, debounce=True),
                ], style={"flex": "1", "minWidth": "110px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Summary Stats ────────────────────────────────────────
        html.Div(id="stress-stats", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap",
        }),

        # ── Scenario Description ─────────────────────────────────
        html.Div(id="stress-desc", style={"marginBottom": "16px"}),

        # ── Row: Position Impact + Scenario Comparison ───────────
        html.Div([
            html.Div([
                html.Div("POSITION-LEVEL IMPACT", style=CARD_HEADER_STYLE),
                html.Div(id="stress-impact-table"),
            ], style={**CARD_STYLE, "flex": "1", "minWidth": "500px"}, className="dashboard-card"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ── Multi-Scenario Comparison ────────────────────────────
        html.Div([
            dcc.Graph(id="stress-comparison", style={"height": "420px"}),
        ], style=CARD_STYLE, className="dashboard-card"),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("stress-stats", "children"),
         Output("stress-desc", "children"),
         Output("stress-impact-table", "children"),
         Output("stress-comparison", "figure")],
        [Input("stress-scenario", "value"),
         Input("stress-spot", "value"),
         Input("stress-vol", "value"),
         Input("stress-rate", "value")],
    )
    def update_stress(scenario_name, custom_spot, custom_vol, custom_rate):
        positions = get_all_positions()
        if not positions:
            empty = go.Figure(layout={**CHART_TEMPLATE["layout"], "title": "No positions"})
            return (
                [html.Div("No positions", style={"color": COLORS["text_muted"]})],
                html.Div(),
                html.Div("Add positions in the Portfolio tab", style={"color": COLORS["text_muted"]}),
                empty,
            )

        tickers = list(set(p.get("ticker", "SPY") for p in positions))
        spot_data = get_spot_prices(tickers)
        spot_prices = {t: d.get("price", 100) for t, d in spot_data.items()}

        # Run selected scenario
        if scenario_name == "CUSTOM":
            result = run_custom_stress(
                positions, spot_prices,
                (custom_spot or -20) / 100,
                (custom_vol or 100) / 100,
                (custom_rate or -1.0) / 100,
            )
        else:
            result = run_stress_test(positions, spot_prices, scenario_name)

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
            ], style={**make_stat_style(color), "flex": "1", "minWidth": "110px"})

        total_pnl = result["total_pnl_impact"]
        pnl_color = COLORS["pnl_profit"] if total_pnl >= 0 else COLORS["pnl_loss"]
        worst = result.get("worst_position")
        worst_pnl = worst["pnl_impact"] if worst else 0

        stats = [
            sbox("Total P&L Impact", total_pnl, "+$,.0f", pnl_color),
            sbox("Base Value", result["total_base_value"], "$,.0f", COLORS["accent_cyan"]),
            sbox("Stressed Value", result["total_stressed_value"], "$,.0f", COLORS["accent_blue"]),
            sbox("Worst Position", worst_pnl, "+$,.0f", COLORS["accent_red"]),
            sbox("Winners", result["n_winners"], "d", COLORS["accent_green"]),
            sbox("Losers", result["n_losers"], "d", COLORS["accent_red"]),
        ]

        # ── Description ──────────────────────────────────────────
        desc_text = result.get("description", "")
        severity = result.get("severity", "")
        sev_color = result.get("color", COLORS["text_muted"])

        desc = html.Div([
            html.Div([
                html.Span(result.get("scenario_name", ""), style={
                    "color": COLORS["text_primary"], "fontSize": "14px", "fontWeight": "700",
                    "marginRight": "12px",
                }),
                html.Span(severity, style={
                    "color": sev_color, "fontSize": "11px", "fontWeight": "700",
                    "padding": "2px 8px", "borderRadius": "4px",
                    "border": f"1px solid {sev_color}",
                }),
            ], style={"marginBottom": "6px"}),
            html.Div(desc_text, style={
                "color": COLORS["text_secondary"], "fontSize": "12px",
            }),
            html.Div([
                html.Span(f"Spot: {result['spot_shock']:+.0%}", style={
                    "color": COLORS["accent_cyan"], "marginRight": "16px", "fontSize": "11px",
                }),
                html.Span(f"Vol: {result['vol_shock']:+.0%}", style={
                    "color": COLORS["accent_purple"], "marginRight": "16px", "fontSize": "11px",
                }),
                html.Span(f"Rate: {result['rate_shock']:+.0%}", style={
                    "color": COLORS["accent_orange"], "fontSize": "11px",
                }),
            ], style={"marginTop": "8px"}),
        ], style={
            **CARD_STYLE,
            "borderLeft": f"4px solid {sev_color}",
        })

        # ── Position impact table ────────────────────────────────
        impact_rows = []
        for pos in result.get("positions", []):
            impact_rows.append({
                "ID": pos["id"],
                "Ticker": pos["ticker"],
                "Type": pos["option_type"].upper(),
                "K": pos["strike"],
                "Qty": pos["quantity"],
                "Base": f"${pos['base_price']:.2f}",
                "Stressed": f"${pos['stressed_price']:.2f}",
                "P&L Impact": f"${pos['pnl_impact']:+,.0f}",
                "Chg %": f"{pos['pct_change']:+.1f}%",
            })

        impact_table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in
                     ["ID", "Ticker", "Type", "K", "Qty", "Base", "Stressed", "P&L Impact", "Chg %"]],
            data=impact_rows,
            style_header=TABLE_HEADER_STYLE,
            style_cell=TABLE_CELL_STYLE,
            style_data_conditional=[
                {"if": {"filter_query": '{P&L Impact} contains "+"'}, "color": COLORS["pnl_profit"]},
                {"if": {"filter_query": '{P&L Impact} contains "-"'}, "color": COLORS["pnl_loss"]},
            ],
            sort_action="native",
            page_size=20,
            style_table={"overflowX": "auto"},
        )

        # ── Multi-scenario comparison ────────────────────────────
        comparisons = compare_scenarios(positions, spot_prices)

        comp_fig = go.Figure()
        if comparisons:
            names = [c["scenario"] for c in comparisons]
            pnls = [c["total_pnl"] for c in comparisons]
            bar_colors = [COLORS["pnl_profit"] if p >= 0 else COLORS["pnl_loss"] for p in pnls]

            comp_fig.add_trace(go.Bar(
                x=names, y=pnls,
                marker_color=bar_colors,
                text=[f"${p:+,.0f}" for p in pnls],
                textposition="outside",
                textfont={"size": 10, "color": COLORS["text_secondary"]},
            ))

            # Add severity badges as annotations
            for i, c in enumerate(comparisons):
                comp_fig.add_annotation(
                    x=c["scenario"], y=0,
                    text=c["severity"],
                    showarrow=False,
                    font={"size": 8, "color": c["color"]},
                    yshift=-15,
                )

        comp_fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title="ALL SCENARIOS — PORTFOLIO P&L IMPACT",
            xaxis_title="",
            yaxis_title="P&L Impact ($)",
            xaxis_tickangle=-30,
            showlegend=False,
        )

        return stats, desc, impact_table, comp_fig
