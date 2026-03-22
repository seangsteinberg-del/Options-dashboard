"""
Macro Events & Positioning Panel
=================================
Central bank calendar, economic data releases, event impact analysis,
and CFTC speculative positioning dashboard for FX options workstation.

Provides a Gantt-style event timeline, click-through impact analysis
with historical vol behaviour around events, and a full positioning
dashboard with net speculative bars, z-scores, and extreme flags.
"""

import dash
from dash import html, dcc, Input, Output, State, no_update, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.theme import COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE, LABEL_STYLE, DROPDOWN_STYLE
from core.bloomberg_fx import get_cftc_positioning, get_central_bank_dates, get_fx_spots, get_all_pairs
from core.fx_analytics import positioning_zscore, positioning_extremes, cftc_positioning_data, vol_change
from core.fx_conventions import FX_PAIR_REGISTRY


# ============================================================================
#  Constants & Styles
# ============================================================================

CARD_HEADER_STYLE = {
    "color": COLORS["text_primary"],
    "fontSize": "13px",
    "fontWeight": "700",
    "fontFamily": "'JetBrains Mono', monospace",
    "marginBottom": "18px",
    "paddingBottom": "12px",
    "borderBottom": f"1px solid {COLORS['border_subtle']}",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": COLORS["bg_secondary"],
    "color": COLORS["text_secondary"],
    "fontWeight": "700",
    "fontSize": "10px",
    "textTransform": "uppercase",
    "letterSpacing": "1.2px",
    "border": f"1px solid {COLORS['border_subtle']}",
    "padding": "10px 12px",
}

TABLE_CELL_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["text_primary"],
    "fontSize": "12px",
    "fontFamily": "'JetBrains Mono', monospace",
    "border": f"1px solid {COLORS['border_subtle']}",
    "padding": "8px 12px",
}

IMPACT_COLORS = {
    "HIGH":   COLORS["accent_rose"],
    "MEDIUM": COLORS["accent_amber"],
    "LOW":    COLORS["accent_cyan"],
}

IMPACT_RGBA = {
    "HIGH":   "rgba(244, 63, 94, 0.55)",
    "MEDIUM": "rgba(245, 158, 11, 0.55)",
    "LOW":    "rgba(6, 182, 212, 0.45)",
}

IMPACT_RGBA_BORDER = {
    "HIGH":   "rgba(244, 63, 94, 0.9)",
    "MEDIUM": "rgba(245, 158, 11, 0.9)",
    "LOW":    "rgba(6, 182, 212, 0.9)",
}

POSITIONING_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
]


# ============================================================================
#  Events Database (Synthetic)
# ============================================================================

def _fwd(days):
    """Return ISO date string N calendar days from today.

    Skips weekends so that the resulting date always falls on a business day,
    matching real-world market event scheduling.
    """
    dt = datetime.now() + timedelta(days=days)
    # Push Saturday -> Monday, Sunday -> Monday
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


# Events are deliberately spread across a ~45-day forward window so that the
# calendar does not cluster everything into one or two weeks.  Central bank
# meetings and data releases are staggered in a way that roughly mirrors
# real-world scheduling cadences.

EVENTS_DB = {
    "central_banks": [
        {"bank": "FOMC", "date": _fwd(7), "current_rate": 5.50,
         "expected": "HOLD", "pairs": ["EURUSD", "USDJPY", "GBPUSD"], "impact": "HIGH"},
        {"bank": "ECB", "date": _fwd(14), "current_rate": 4.50,
         "expected": "HOLD", "pairs": ["EURUSD", "EURGBP", "EURJPY"], "impact": "HIGH"},
        {"bank": "BOJ", "date": _fwd(10), "current_rate": 0.25,
         "expected": "HOLD", "pairs": ["USDJPY", "EURJPY", "GBPJPY"], "impact": "HIGH"},
        {"bank": "BOE", "date": _fwd(21), "current_rate": 5.25,
         "expected": "HOLD", "pairs": ["GBPUSD", "EURGBP", "GBPJPY"], "impact": "MEDIUM"},
        {"bank": "RBA", "date": _fwd(3), "current_rate": 4.35,
         "expected": "HOLD", "pairs": ["AUDUSD", "AUDJPY", "AUDNZD"], "impact": "MEDIUM"},
        {"bank": "BOC", "date": _fwd(28), "current_rate": 5.00,
         "expected": "HOLD", "pairs": ["USDCAD", "CADJPY"], "impact": "MEDIUM"},
        {"bank": "RBNZ", "date": _fwd(35), "current_rate": 5.50,
         "expected": "HOLD", "pairs": ["NZDUSD", "AUDNZD", "NZDJPY"], "impact": "LOW"},
        {"bank": "Riksbank", "date": _fwd(42), "current_rate": 4.00,
         "expected": "POSSIBLE CUT", "pairs": ["EURSEK", "USDSEK"], "impact": "LOW"},
        {"bank": "Norges Bank", "date": _fwd(24), "current_rate": 4.50,
         "expected": "HOLD", "pairs": ["EURNOK", "USDNOK"], "impact": "LOW"},
        {"bank": "SNB", "date": _fwd(31), "current_rate": 1.75,
         "expected": "POSSIBLE CUT", "pairs": ["USDCHF", "EURCHF"], "impact": "MEDIUM"},
        {"bank": "PBOC", "date": _fwd(17), "current_rate": 3.45,
         "expected": "HOLD", "pairs": ["USDCNH"], "impact": "MEDIUM"},
    ],
    "data_releases": [
        {"event": "US NFP", "date": _fwd(2), "consensus": "200K",
         "prior": "275K", "pairs": ["EURUSD", "USDJPY", "GBPUSD"], "impact": "HIGH"},
        {"event": "US CPI", "date": _fwd(12), "consensus": "3.1%",
         "prior": "3.1%", "pairs": ["EURUSD", "USDJPY"], "impact": "HIGH"},
        {"event": "US Core CPI", "date": _fwd(12), "consensus": "3.7%",
         "prior": "3.9%", "pairs": ["EURUSD", "USDJPY"], "impact": "HIGH"},
        {"event": "EU CPI", "date": _fwd(19), "consensus": "2.6%",
         "prior": "2.8%", "pairs": ["EURUSD", "EURGBP"], "impact": "MEDIUM"},
        {"event": "UK CPI", "date": _fwd(8), "consensus": "4.0%",
         "prior": "4.0%", "pairs": ["GBPUSD", "EURGBP"], "impact": "MEDIUM"},
        {"event": "US GDP Q4", "date": _fwd(33), "consensus": "3.2%",
         "prior": "3.2%", "pairs": ["EURUSD", "USDJPY"], "impact": "MEDIUM"},
        {"event": "US Retail Sales", "date": _fwd(16), "consensus": "0.3%",
         "prior": "-0.8%", "pairs": ["EURUSD", "USDJPY"], "impact": "MEDIUM"},
        {"event": "ISM Manufacturing", "date": _fwd(5), "consensus": "49.5",
         "prior": "49.1", "pairs": ["EURUSD", "USDJPY"], "impact": "MEDIUM"},
        {"event": "CFTC Positioning", "date": _fwd(1),
         "pairs": ["ALL"], "impact": "LOW"},
    ],
}


# ============================================================================
#  Synthetic Data Helpers
# ============================================================================

def _build_event_list():
    """Flatten both categories into a single sorted event list."""
    events = []
    for cb in EVENTS_DB["central_banks"]:
        events.append({
            "id": f"CB_{cb['bank']}",
            "name": f"{cb['bank']} Decision",
            "category": "Central Bank",
            "date": cb["date"],
            "impact": cb["impact"],
            "pairs": cb["pairs"],
            "detail": {
                "current_rate": cb["current_rate"],
                "expected": cb["expected"],
            },
        })
    for dr in EVENTS_DB["data_releases"]:
        detail = {}
        if "consensus" in dr:
            detail["consensus"] = dr["consensus"]
        if "prior" in dr:
            detail["prior"] = dr["prior"]
        events.append({
            "id": f"DR_{dr['event'].replace(' ', '_')}",
            "name": dr["event"],
            "category": "Data Release",
            "date": dr["date"],
            "impact": dr["impact"],
            "pairs": dr["pairs"],
            "detail": detail,
        })
    events.sort(key=lambda e: e["date"])
    return events


ALL_EVENTS = _build_event_list()


def _historical_vol_impact(event_name):
    """
    Synthetic historical vol impact data for a given event type.
    Returns dict with typical ATM 1W vol change pre/post event.
    """
    rng = np.random.RandomState(hash(event_name) % 2**31)
    is_high = any(
        (e["name"] == event_name and e["impact"] == "HIGH") for e in ALL_EVENTS
    )
    base_move = rng.uniform(1.2, 3.5) if is_high else rng.uniform(0.3, 1.5)
    return {
        "avg_1w_vol_rise_pre":  round(base_move, 2),
        "avg_1w_vol_drop_post": round(base_move * rng.uniform(0.6, 0.9), 2),
        "max_1d_spot_move_bps": int(rng.uniform(30, 180) if is_high else rng.uniform(10, 60)),
        "event_premium_vol":    round(base_move * rng.uniform(0.3, 0.7), 2),
        "typical_straddle_pnl_pips": int(rng.uniform(15, 90) if is_high else rng.uniform(5, 30)),
        "n_observations":       int(rng.randint(12, 40)),
    }


def _what_if_scenarios(event):
    """Generate better-than / worse-than text for an event."""
    name = event["name"]
    cat = event["category"]
    detail = event.get("detail", {})

    if cat == "Central Bank":
        rate = detail.get("current_rate", "N/A")
        exp = detail.get("expected", "HOLD")
        better = (
            f"Dovish surprise: rate cut or dovish guidance. "
            f"Current rate {rate}%. Market expects {exp}. "
            f"A cut could weaken the currency 50-150 pips on major pairs."
        )
        worse = (
            f"Hawkish surprise: unexpected hike or hawkish shift. "
            f"Would strengthen the currency significantly. "
            f"Vol spike of 2-4 vols in 1W tenors likely."
        )
    else:
        consensus = detail.get("consensus", "N/A")
        prior = detail.get("prior", "N/A")
        better = (
            f"Better than expected: above consensus ({consensus}), prior was {prior}. "
            f"Strong data supports USD, weakens risk pairs. Expect 30-80 pip moves."
        )
        worse = (
            f"Worse than expected: below consensus ({consensus}), prior was {prior}. "
            f"Weak data weighs on USD, supports EUR, JPY. Vol premium collapses post-release."
        )
    return better, worse


def _generate_positioning_data():
    """Generate full positioning dataset for the dashboard."""
    records = []
    for pair in POSITIONING_PAIRS:
        pos = cftc_positioning_data(pair)
        z = positioning_zscore(pair)
        records.append({
            "pair": pair,
            "net_speculative": pos["net_speculative"],
            "spec_long": pos["spec_long"],
            "spec_short": pos["spec_short"],
            "open_interest": pos["open_interest"],
            "zscore": round(z["zscore"], 2),
            "signal": z["signal"],
        })
    return pd.DataFrame(records)


# ============================================================================
#  Layout
# ============================================================================

def layout():
    event_options = [
        {"label": f"{e['name']}  ({e['date']})", "value": e["id"]}
        for e in ALL_EVENTS
    ]
    default_event = ALL_EVENTS[0]["id"] if ALL_EVENTS else None

    pair_filter_options = [{"label": "ALL PAIRS", "value": "ALL"}] + [
        {"label": p, "value": p} for p in POSITIONING_PAIRS
    ]

    return html.Div([

        # ---- Title bar -------------------------------------------------------
        html.Div([
            html.Div("MACRO EVENTS & POSITIONING", style=CARD_HEADER_STYLE),
            html.Div([
                html.Span("EVENTS: ", style={
                    "color": COLORS["text_muted"], "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
                html.Span(f"{len(ALL_EVENTS)}", style={
                    "color": COLORS["accent_cyan"], "fontSize": "13px",
                    "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace",
                    "marginRight": "24px",
                }),
                html.Span("HIGH IMPACT: ", style={
                    "color": COLORS["text_muted"], "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
                html.Span(
                    f"{sum(1 for e in ALL_EVENTS if e['impact'] == 'HIGH')}",
                    style={
                        "color": COLORS["accent_rose"], "fontSize": "13px",
                        "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace",
                    },
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            **CARD_STYLE,
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "padding": "16px 24px",
        }, className="dashboard-card"),

        # ---- Top Row: Calendar (60%) + Impact Analysis (40%) ----------------
        html.Div([

            # -- Calendar View (left 60%) --
            html.Div([
                html.Div([
                    html.Div("EVENT CALENDAR", style=CARD_HEADER_STYLE),
                    html.Div([
                        html.Div([
                            html.Label("IMPACT FILTER", style=LABEL_STYLE),
                            dcc.Dropdown(
                                id="events-impact-filter",
                                options=[
                                    {"label": "ALL", "value": "ALL"},
                                    {"label": "HIGH", "value": "HIGH"},
                                    {"label": "MEDIUM", "value": "MEDIUM"},
                                    {"label": "LOW", "value": "LOW"},
                                ],
                                value="ALL",
                                clearable=False,
                                style={"fontSize": "12px", "width": "140px"},
                            ),
                        ], style={"marginRight": "14px"}),
                        html.Div([
                            html.Label("CATEGORY", style=LABEL_STYLE),
                            dcc.Dropdown(
                                id="events-category-filter",
                                options=[
                                    {"label": "ALL", "value": "ALL"},
                                    {"label": "CENTRAL BANKS", "value": "Central Bank"},
                                    {"label": "DATA RELEASES", "value": "Data Release"},
                                ],
                                value="ALL",
                                clearable=False,
                                style={"fontSize": "12px", "width": "180px"},
                            ),
                        ], style={"marginRight": "14px"}),
                        html.Div([
                            html.Label("SELECT EVENT", style=LABEL_STYLE),
                            dcc.Dropdown(
                                id="events-event-select",
                                options=event_options,
                                value=default_event,
                                clearable=False,
                                style={"fontSize": "12px", "width": "280px"},
                            ),
                        ]),
                    ], style={"display": "flex", "flexWrap": "wrap",
                              "gap": "4px", "marginBottom": "12px"}),
                    dcc.Graph(
                        id="events-calendar-chart",
                        style={"height": "440px"},
                        config={"displayModeBar": False},
                    ),
                ], style=CARD_STYLE, className="dashboard-card"),
            ], style={"flex": "3", "minWidth": "550px"}),

            # -- Impact Analysis (right 40%) --
            html.Div([
                html.Div([
                    html.Div("EVENT IMPACT ANALYSIS", style=CARD_HEADER_STYLE),
                    html.Div(id="events-impact-summary", style={
                        "marginBottom": "14px",
                    }),
                    html.Div(id="events-whatif-section", style={
                        "marginBottom": "14px",
                    }),
                    dcc.Graph(
                        id="events-vol-impact-chart",
                        style={"height": "260px"},
                        config={"displayModeBar": False},
                    ),
                ], style=CARD_STYLE, className="dashboard-card"),
            ], style={"flex": "2", "minWidth": "380px"}),

        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "16px"}),

        # ---- Bottom Row: Positioning Dashboard (full width) -----------------
        html.Div([
            html.Div("CFTC SPECULATIVE POSITIONING", style=CARD_HEADER_STYLE),

            # Controls row
            html.Div([
                html.Div([
                    html.Label("PAIR FILTER", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="events-pos-pair-filter",
                        options=pair_filter_options,
                        value="ALL",
                        clearable=False,
                        style={"fontSize": "12px", "width": "160px"},
                    ),
                ], style={"marginRight": "14px"}),
                html.Div([
                    html.Label("Z-SCORE LOOKBACK (WEEKS)", style=LABEL_STYLE),
                    dcc.Input(
                        id="events-pos-lookback",
                        type="number", value=156, min=26, max=520, step=26,
                        style={
                            "backgroundColor": COLORS["bg_input"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "10px",
                            "color": COLORS["text_primary"],
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "13px",
                            "padding": "10px 14px",
                            "width": "120px",
                        },
                        debounce=True,
                    ),
                ]),
            ], style={"display": "flex", "flexWrap": "wrap",
                      "gap": "4px", "marginBottom": "16px"}),

            # Charts row: Net Speculative + Z-Score
            html.Div([
                html.Div([
                    dcc.Graph(
                        id="events-pos-net-chart",
                        style={"height": "360px"},
                        config={"displayModeBar": False},
                    ),
                ], style={"flex": "1", "minWidth": "420px"}),
                html.Div([
                    dcc.Graph(
                        id="events-pos-zscore-chart",
                        style={"height": "360px"},
                        config={"displayModeBar": False},
                    ),
                ], style={"flex": "1", "minWidth": "420px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

            # Extreme positioning table
            html.Div([
                html.Div("EXTREME POSITIONING FLAGS  ( |Z| > 1.5 )",
                         style={**CARD_HEADER_STYLE, "marginTop": "20px"}),
                html.Div(id="events-pos-extremes-table"),
            ]),

        ], style=CARD_STYLE, className="dashboard-card"),

        # Hidden stores
        dcc.Store(id="events-selected-event-store", data=default_event),
        dcc.Interval(id="events-interval", interval=120_000, n_intervals=0),

    ])


# ============================================================================
#  Callbacks
# ============================================================================

def register_callbacks(app):
    """Register all callbacks for the events & positioning panel."""

    # ------------------------------------------------------------------
    # 1. Calendar Gantt Chart
    # ------------------------------------------------------------------
    @app.callback(
        Output("events-calendar-chart", "figure"),
        [Input("events-impact-filter", "value"),
         Input("events-category-filter", "value"),
         Input("events-interval", "n_intervals")],
    )
    def render_calendar(impact_filter, category_filter, _n):
        filtered = ALL_EVENTS[:]

        if impact_filter != "ALL":
            filtered = [e for e in filtered if e["impact"] == impact_filter]
        if category_filter != "ALL":
            filtered = [e for e in filtered if e["category"] == category_filter]

        if not filtered:
            fig = go.Figure()
            fig.update_layout(
                **CHART_TEMPLATE["layout"],
                title="No events match current filters",
                height=420,
            )
            return fig

        # Sort by date (earliest at bottom for readable Gantt)
        filtered.sort(key=lambda e: e["date"], reverse=True)

        names = []
        starts = []
        durations = []
        colors = []
        border_colors = []
        hover_texts = []
        event_ids = []

        for ev in filtered:
            dt = datetime.strptime(ev["date"], "%Y-%m-%d")
            label = ev["name"]
            if ev["category"] == "Central Bank":
                label = f"CB: {label}"
            names.append(label)
            starts.append(dt - timedelta(hours=12))
            durations.append(timedelta(hours=24))
            colors.append(IMPACT_RGBA[ev["impact"]])
            border_colors.append(IMPACT_RGBA_BORDER[ev["impact"]])
            pairs_str = ", ".join(ev["pairs"][:4])
            if len(ev["pairs"]) > 4:
                pairs_str += " ..."
            hover_texts.append(
                f"<b>{ev['name']}</b><br>"
                f"Date: {ev['date']}<br>"
                f"Category: {ev['category']}<br>"
                f"Impact: {ev['impact']}<br>"
                f"Pairs: {pairs_str}"
            )
            event_ids.append(ev["id"])

        fig = go.Figure()

        for i in range(len(names)):
            fig.add_trace(go.Bar(
                y=[names[i]],
                x=[durations[i].total_seconds() * 1000],
                base=[starts[i]],
                orientation="h",
                marker=dict(
                    color=colors[i],
                    line=dict(color=border_colors[i], width=1.5),
                ),
                hovertext=hover_texts[i],
                hoverinfo="text",
                customdata=[event_ids[i]],
                showlegend=False,
                width=0.6,
            ))

        # Determine date range for axis
        all_dates = [datetime.strptime(e["date"], "%Y-%m-%d") for e in filtered]
        min_date = min(all_dates) - timedelta(days=2)
        max_date = max(all_dates) + timedelta(days=2)

        # Add "TODAY" vertical line
        today_ref = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        fig.add_vline(
            x=today_ref.timestamp() * 1000,
            line_width=2,
            line_dash="dash",
            line_color=COLORS["accent_cyan"],
            annotation_text="TODAY",
            annotation_position="top",
            annotation=dict(
                font=dict(
                    color=COLORS["accent_cyan"], size=10,
                    family="'JetBrains Mono', monospace",
                ),
            ),
        )

        # Add legend-like annotations for impact levels
        for level, color in IMPACT_COLORS.items():
            fig.add_trace(go.Bar(
                y=[None], x=[None], orientation="h",
                marker=dict(color=color),
                name=f"{level} IMPACT",
                showlegend=True,
            ))

        fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title=dict(
                text="MACRO EVENT TIMELINE",
                font=dict(color=COLORS["text_primary"], size=14),
            ),
            xaxis=dict(
                type="date",
                range=[min_date, max_date],
                gridcolor="rgba(30,42,69,0.3)",
                tickformat="%b %d",
                tickfont=dict(size=10, color=COLORS["text_muted"]),
                linecolor=COLORS["border_subtle"],
            ),
            yaxis=dict(
                tickfont=dict(
                    size=10, color=COLORS["text_secondary"],
                    family="'JetBrains Mono', monospace",
                ),
                gridcolor="rgba(30,42,69,0.15)",
                linecolor=COLORS["border_subtle"],
            ),
            barmode="overlay",
            height=420,
            margin=dict(l=180, r=30, t=50, b=40),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=10, color=COLORS["text_secondary"]),
            ),
        )
        return fig

    # ------------------------------------------------------------------
    # 2. Event selection sync
    # ------------------------------------------------------------------
    @app.callback(
        Output("events-selected-event-store", "data"),
        [Input("events-event-select", "value"),
         Input("events-calendar-chart", "clickData")],
        [State("events-selected-event-store", "data")],
    )
    def sync_event_selection(dropdown_val, click_data, current_store):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_store or (ALL_EVENTS[0]["id"] if ALL_EVENTS else None)

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "events-calendar-chart" and click_data:
            try:
                point = click_data["points"][0]
                if "customdata" in point and point["customdata"]:
                    return point["customdata"]
            except (KeyError, IndexError):
                pass

        if trigger_id == "events-event-select" and dropdown_val:
            return dropdown_val

        return current_store

    # ------------------------------------------------------------------
    # 3. Impact Analysis: Summary + What-If + Vol Impact chart
    # ------------------------------------------------------------------
    @app.callback(
        [Output("events-impact-summary", "children"),
         Output("events-whatif-section", "children"),
         Output("events-vol-impact-chart", "figure")],
        [Input("events-selected-event-store", "data")],
    )
    def update_impact_analysis(event_id):
        event = None
        for e in ALL_EVENTS:
            if e["id"] == event_id:
                event = e
                break
        if event is None and ALL_EVENTS:
            event = ALL_EVENTS[0]
        if event is None:
            empty_fig = go.Figure()
            empty_fig.update_layout(**CHART_TEMPLATE["layout"], height=240)
            return html.Div("No event selected"), html.Div(), empty_fig

        detail = event.get("detail", {})
        impact_data = _historical_vol_impact(event["name"])
        better_text, worse_text = _what_if_scenarios(event)

        # -- Summary section --
        impact_badge_color = IMPACT_COLORS.get(event["impact"], COLORS["text_muted"])
        summary_children = [
            # Event name
            html.Div(event["name"], style={
                "color": COLORS["text_bright"], "fontSize": "18px",
                "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace",
                "marginBottom": "8px",
            }),
            # Date + Impact badge
            html.Div([
                html.Span(event["date"], style={
                    "color": COLORS["text_secondary"], "fontSize": "13px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginRight": "16px",
                }),
                html.Span(event["impact"], style={
                    "color": impact_badge_color,
                    "fontSize": "11px", "fontWeight": "700",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "padding": "3px 10px",
                    "borderRadius": "6px",
                    "border": f"1px solid {impact_badge_color}",
                    "letterSpacing": "1px",
                }),
                html.Span(f"  {event['category']}", style={
                    "color": COLORS["text_muted"], "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginLeft": "12px",
                }),
            ], style={"marginBottom": "10px"}),
            # Detail line
            html.Div([
                html.Span(f"{k.upper()}: ", style={
                    "color": COLORS["text_muted"], "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', monospace",
                })
                if True else None
                for k, v in detail.items()
                for _ in [None]
            ] if not detail else [
                html.Span([
                    html.Span(f"{k.replace('_', ' ').upper()}: ", style={
                        "color": COLORS["text_muted"], "fontSize": "11px",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                    html.Span(f"{v}  ", style={
                        "color": COLORS["accent_cyan"], "fontSize": "12px",
                        "fontWeight": "600",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "marginRight": "16px",
                    }),
                ]) for k, v in detail.items()
            ], style={"marginBottom": "8px"}),
            # Affected pairs
            html.Div([
                html.Span("PAIRS: ", style={
                    "color": COLORS["text_muted"], "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
            ] + [
                html.Span(p, style={
                    "color": COLORS["accent_blue"], "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "padding": "2px 8px", "marginRight": "6px",
                    "borderRadius": "4px",
                    "backgroundColor": "rgba(59, 130, 246, 0.15)",
                    "border": f"1px solid rgba(59, 130, 246, 0.3)",
                }) for p in event["pairs"]
            ], style={"display": "flex", "flexWrap": "wrap",
                      "alignItems": "center", "gap": "2px"}),
            # Historical stats row
            html.Div([
                _make_mini_stat("1W VOL RISE (PRE)",
                                f"+{impact_data['avg_1w_vol_rise_pre']}v",
                                COLORS["accent_orange"]),
                _make_mini_stat("1W VOL DROP (POST)",
                                f"-{impact_data['avg_1w_vol_drop_post']}v",
                                COLORS["accent_green"]),
                _make_mini_stat("MAX SPOT MOVE",
                                f"{impact_data['max_1d_spot_move_bps']} bps",
                                COLORS["accent_rose"]),
                _make_mini_stat("EVENT PREMIUM",
                                f"+{impact_data['event_premium_vol']}v",
                                COLORS["accent_purple"]),
            ], style={
                "display": "flex", "gap": "8px", "marginTop": "14px",
                "flexWrap": "wrap",
            }),
        ]

        # -- What-If section --
        whatif_children = html.Div([
            html.Div("SCENARIO ANALYSIS", style={
                **CARD_HEADER_STYLE, "fontSize": "11px",
                "marginBottom": "10px", "paddingBottom": "8px",
            }),
            # Better scenario
            html.Div([
                html.Div([
                    html.Span("BETTER", style={
                        "color": COLORS["accent_green"], "fontSize": "10px",
                        "fontWeight": "700", "letterSpacing": "1px",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                ], style={"marginBottom": "4px"}),
                html.Div(better_text, style={
                    "color": COLORS["text_secondary"], "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "lineHeight": "1.5",
                }),
            ], style={
                "backgroundColor": "rgba(16, 185, 129, 0.06)",
                "border": f"1px solid rgba(16, 185, 129, 0.2)",
                "borderRadius": "8px", "padding": "10px 14px",
                "marginBottom": "8px",
            }),
            # Worse scenario
            html.Div([
                html.Div([
                    html.Span("WORSE", style={
                        "color": COLORS["accent_red"], "fontSize": "10px",
                        "fontWeight": "700", "letterSpacing": "1px",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                ], style={"marginBottom": "4px"}),
                html.Div(worse_text, style={
                    "color": COLORS["text_secondary"], "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "lineHeight": "1.5",
                }),
            ], style={
                "backgroundColor": "rgba(239, 68, 68, 0.06)",
                "border": f"1px solid rgba(239, 68, 68, 0.2)",
                "borderRadius": "8px", "padding": "10px 14px",
            }),
        ])

        # -- Vol Impact chart --
        vol_fig = _build_vol_impact_chart(event, impact_data)

        return summary_children, whatif_children, vol_fig

    # ------------------------------------------------------------------
    # 4. Positioning Dashboard: Net Spec + Z-Score + Extremes Table
    # ------------------------------------------------------------------
    @app.callback(
        [Output("events-pos-net-chart", "figure"),
         Output("events-pos-zscore-chart", "figure"),
         Output("events-pos-extremes-table", "children")],
        [Input("events-pos-pair-filter", "value"),
         Input("events-pos-lookback", "value"),
         Input("events-interval", "n_intervals")],
    )
    def update_positioning(pair_filter, lookback, _n):
        lookback = lookback or 156

        if pair_filter and pair_filter != "ALL":
            pairs = [pair_filter]
        else:
            pairs = POSITIONING_PAIRS

        # Gather data
        net_specs = []
        zscores = []
        signals = []
        for p in pairs:
            pos = cftc_positioning_data(p)
            z = positioning_zscore(p, lookback=lookback)
            net_specs.append(pos["net_speculative"])
            zscores.append(z["zscore"])
            signals.append(z["signal"])

        # ---- Net Speculative Bar Chart ----
        net_colors = [
            COLORS["accent_green"] if v >= 0 else COLORS["accent_red"]
            for v in net_specs
        ]

        net_fig = go.Figure()
        net_fig.add_trace(go.Bar(
            x=pairs,
            y=net_specs,
            marker=dict(
                color=net_colors,
                line=dict(
                    color=[
                        "rgba(16,185,129,0.7)" if v >= 0 else "rgba(239,68,68,0.7)"
                        for v in net_specs
                    ],
                    width=1.5,
                ),
            ),
            text=[f"{v:+,.0f}" for v in net_specs],
            textposition="outside",
            textfont=dict(
                size=10, color=COLORS["text_secondary"],
                family="'JetBrains Mono', monospace",
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Net Speculative: %{y:,.0f}<br>"
                "<extra></extra>"
            ),
        ))
        net_fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title=dict(
                text="NET SPECULATIVE POSITIONING (CONTRACTS)",
                font=dict(color=COLORS["text_primary"], size=13),
            ),
            xaxis=dict(
                tickfont=dict(
                    size=11, color=COLORS["text_secondary"],
                    family="'JetBrains Mono', monospace",
                ),
                gridcolor="rgba(30,42,69,0.3)",
                linecolor=COLORS["border_subtle"],
            ),
            yaxis=dict(
                title="Contracts",
                tickfont=dict(size=10, color=COLORS["text_muted"]),
                gridcolor="rgba(30,42,69,0.3)",
                linecolor=COLORS["border_subtle"],
                zerolinecolor=COLORS["accent_blue"],
                zerolinewidth=1,
            ),
            height=340,
            margin=dict(l=60, r=20, t=50, b=50),
            showlegend=False,
        )

        # ---- Z-Score Bar Chart ----
        z_abs = [abs(z) for z in zscores]
        z_colors = []
        for z in zscores:
            az = abs(z)
            if az > 2.0:
                z_colors.append(COLORS["accent_rose"])
            elif az > 1.5:
                z_colors.append(COLORS["accent_orange"])
            elif az > 1.0:
                z_colors.append(COLORS["accent_amber"])
            else:
                z_colors.append(COLORS["accent_cyan"])

        zscore_fig = go.Figure()
        zscore_fig.add_trace(go.Bar(
            x=pairs,
            y=zscores,
            marker=dict(
                color=z_colors,
                line=dict(color=[c for c in z_colors], width=1),
            ),
            text=[f"{z:+.2f}" for z in zscores],
            textposition="outside",
            textfont=dict(
                size=10, color=COLORS["text_secondary"],
                family="'JetBrains Mono', monospace",
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Z-Score: %{y:.2f}<br>"
                "<extra></extra>"
            ),
        ))

        # Add threshold lines
        for threshold, label, color, dash_style in [
            (1.5, "+1.5\u03c3", COLORS["accent_orange"], "dash"),
            (-1.5, "-1.5\u03c3", COLORS["accent_orange"], "dash"),
            (2.0, "+2.0\u03c3", COLORS["accent_rose"], "dot"),
            (-2.0, "-2.0\u03c3", COLORS["accent_rose"], "dot"),
        ]:
            zscore_fig.add_hline(
                y=threshold, line_width=1, line_dash=dash_style,
                line_color=color, opacity=0.6,
                annotation_text=label,
                annotation_position="right",
                annotation=dict(
                    font=dict(size=9, color=color,
                              family="'JetBrains Mono', monospace"),
                ),
            )

        zscore_fig.update_layout(
            **CHART_TEMPLATE["layout"],
            title=dict(
                text=f"POSITIONING Z-SCORE ({lookback}W LOOKBACK)",
                font=dict(color=COLORS["text_primary"], size=13),
            ),
            xaxis=dict(
                tickfont=dict(
                    size=11, color=COLORS["text_secondary"],
                    family="'JetBrains Mono', monospace",
                ),
                gridcolor="rgba(30,42,69,0.3)",
                linecolor=COLORS["border_subtle"],
            ),
            yaxis=dict(
                title="Z-Score (\u03c3)",
                tickfont=dict(size=10, color=COLORS["text_muted"]),
                gridcolor="rgba(30,42,69,0.3)",
                linecolor=COLORS["border_subtle"],
                zerolinecolor=COLORS["accent_blue"],
                zerolinewidth=1,
            ),
            height=340,
            margin=dict(l=60, r=40, t=50, b=50),
            showlegend=False,
        )

        # ---- Extreme Positioning Table ----
        extremes_df = positioning_extremes(pairs)
        extreme_only = extremes_df[extremes_df["extreme"]].copy()

        if extreme_only.empty:
            extremes_table = html.Div(
                "No extreme positioning detected for selected pairs.",
                style={
                    "color": COLORS["text_muted"], "fontSize": "12px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "padding": "16px", "textAlign": "center",
                },
            )
        else:
            extreme_only["direction"] = extreme_only["zscore"].apply(
                lambda z: "LONG" if z > 0 else "SHORT"
            )
            extreme_only["abs_zscore"] = extreme_only["zscore"].abs()
            display_df = extreme_only[
                ["pair", "net_speculative", "zscore", "signal", "direction"]
            ].copy()
            display_df.columns = ["Pair", "Net Spec", "Z-Score", "Signal", "Direction"]

            extremes_table = dash_table.DataTable(
                data=display_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in display_df.columns],
                style_header=TABLE_HEADER_STYLE,
                style_cell={
                    **TABLE_CELL_STYLE,
                    "textAlign": "center",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{Z-Score} > 1.5"},
                        "color": COLORS["accent_green"],
                        "fontWeight": "700",
                    },
                    {
                        "if": {"filter_query": "{Z-Score} < -1.5"},
                        "color": COLORS["accent_red"],
                        "fontWeight": "700",
                    },
                    {
                        "if": {"column_id": "Signal",
                               "filter_query": '{Signal} = "EXTREME_LONG"'},
                        "backgroundColor": "rgba(16, 185, 129, 0.1)",
                        "color": COLORS["accent_green"],
                    },
                    {
                        "if": {"column_id": "Signal",
                               "filter_query": '{Signal} = "EXTREME_SHORT"'},
                        "backgroundColor": "rgba(239, 68, 68, 0.1)",
                        "color": COLORS["accent_red"],
                    },
                ],
                style_table={
                    "overflowX": "auto",
                    "borderRadius": "8px",
                    "border": f"1px solid {COLORS['border_subtle']}",
                },
                page_size=10,
                sort_action="native",
            )

        return net_fig, zscore_fig, extremes_table


# ============================================================================
#  Chart Builder Helpers
# ============================================================================

def _build_vol_impact_chart(event, impact_data):
    """
    Build a bar chart showing historical vol behaviour around this event type.
    Shows pre-event vol rise, post-event vol drop, event premium, and
    typical straddle P&L.
    """
    categories = [
        "1W Vol Rise\n(Pre-Event)",
        "1W Vol Drop\n(Post-Event)",
        "Event Premium\n(Extra Vol)",
        "Max Spot Move\n(bps / 10)",
        "Straddle P&L\n(pips / 10)",
    ]
    values = [
        impact_data["avg_1w_vol_rise_pre"],
        -impact_data["avg_1w_vol_drop_post"],
        impact_data["event_premium_vol"],
        impact_data["max_1d_spot_move_bps"] / 10.0,
        impact_data["typical_straddle_pnl_pips"] / 10.0,
    ]
    bar_colors = [
        COLORS["accent_orange"],
        COLORS["accent_green"],
        COLORS["accent_purple"],
        COLORS["accent_rose"],
        COLORS["accent_cyan"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=bar_colors,
            line=dict(color=bar_colors, width=1.5),
        ),
        text=[f"{v:+.1f}" for v in values],
        textposition="outside",
        textfont=dict(
            size=10, color=COLORS["text_secondary"],
            family="'JetBrains Mono', monospace",
        ),
        hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        **CHART_TEMPLATE["layout"],
        title=dict(
            text=f"HISTORICAL VOL IMPACT: {event['name']}  (n={impact_data['n_observations']})",
            font=dict(color=COLORS["text_primary"], size=12),
        ),
        xaxis=dict(
            tickfont=dict(
                size=9, color=COLORS["text_secondary"],
                family="'JetBrains Mono', monospace",
            ),
            gridcolor="rgba(30,42,69,0.3)",
            linecolor=COLORS["border_subtle"],
        ),
        yaxis=dict(
            title="Vols / Scaled",
            tickfont=dict(size=9, color=COLORS["text_muted"]),
            gridcolor="rgba(30,42,69,0.3)",
            linecolor=COLORS["border_subtle"],
            zerolinecolor=COLORS["border"],
            zerolinewidth=1,
        ),
        height=240,
        margin=dict(l=50, r=20, t=40, b=60),
        showlegend=False,
    )
    return fig


def _make_mini_stat(label, value, color):
    """Small stat box used in the impact analysis summary."""
    return html.Div([
        html.Div(label, style={
            "color": COLORS["text_muted"], "fontSize": "9px",
            "fontFamily": "'JetBrains Mono', monospace",
            "textTransform": "uppercase", "letterSpacing": "0.8px",
            "marginBottom": "4px",
        }),
        html.Div(value, style={
            "color": color, "fontSize": "15px", "fontWeight": "700",
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ], style={
        "backgroundColor": COLORS["bg_secondary"],
        "border": f"1px solid {COLORS['border_subtle']}",
        "borderTop": f"2px solid {color}",
        "borderRadius": "8px",
        "padding": "10px 14px",
        "textAlign": "center",
        "minWidth": "90px",
        "flex": "1",
    })
