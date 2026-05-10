"""
Trade Ideas Panel
=================
Front door to the TRADE workspace. Runs the trade-screener engine across the
FX pair universe, ranks the candidates per-category, and surfaces them in a
single DataTable. Click a row → "Send to VOL" pre-focuses the VOL Surface
panel on the relevant chart bundle, "Send to Structurer" pre-populates the
Trade Workshop with the recommended structure.

Wire-up (handled in app.py):
    - `ideas-to-vol-store`         → vol_surface_fx (sets pair + 4 chart slots)
    - `ideas-to-structurer-store`  → structure_builder (sets pair, tenor, preset)
    Both stores also drive a clientside callback that switches workspace tabs.
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime
from typing import List

import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update, callback_context
from dash.exceptions import PreventUpdate

from core.theme import (
    COLORS, CARD_STYLE, LABEL_STYLE, BUTTON_STYLE,
    BUTTON_SUCCESS_STYLE, DROPDOWN_STYLE,
)
from core.trade_screener import (
    TradeIdea, ALL_CATEGORIES, CATEGORY_DEFAULTS,
    scan_all, top_n_per_category,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Filter options
# ============================================================================

REGION_OPTIONS = [
    {"label": "All Pairs", "value": "ALL"},
    {"label": "G10",       "value": "G10"},
    {"label": "EM",        "value": "EM"},
    {"label": "Majors",    "value": "MAJORS"},
]

TENOR_BUCKET_OPTIONS = [
    {"label": "All Tenors",        "value": "ALL"},
    {"label": "Front (1W-1M)",     "value": "FRONT"},
    {"label": "Belly (1M-3M)",     "value": "BELLY"},
    {"label": "Back  (6M-1Y)",     "value": "BACK"},
]

CATEGORY_OPTIONS = [{"label": c, "value": c} for c in ALL_CATEGORIES]

REFRESH_OPTIONS = [
    {"label": "Manual",           "value": 0},
    {"label": "Every 2 min",      "value": 120_000},
    {"label": "Every 5 min",      "value": 300_000},
]

DEFAULT_TOP_N = 5


# ============================================================================
# Direction colour map (used for the Direction column)
# ============================================================================

DIRECTION_DISPLAY = {
    "long_vol":   ("LONG VOL",   COLORS["accent_green"]),
    "short_vol":  ("SHORT VOL",  COLORS["accent_red"]),
    "long_skew":  ("LONG SKEW",  COLORS["accent_cyan"]),
    "short_skew": ("SHORT SKEW", COLORS["accent_orange"]),
    "calendar":   ("CALENDAR",   COLORS["accent_purple"]),
    "carry":      ("CARRY",      COLORS["accent_yellow"]),
    "tail":       ("TAIL",       COLORS["accent_amber"]),
}


# ============================================================================
# Table builders
# ============================================================================

def _idea_to_row(idea: TradeIdea, idx: int) -> dict:
    """Convert one TradeIdea to a DataTable row dict."""
    direction_label = DIRECTION_DISPLAY.get(idea.direction, (idea.direction.upper(), ""))[0]
    return {
        "_id":       idx,    # used by selection callback
        "Score":     f"{idea.score:.1f}",
        "Category":  idea.category,
        "Pair":      idea.pair,
        "Tenor":     idea.tenor,
        "Direction": direction_label,
        "Structure": idea.structure_preset,
        "Thesis":    idea.thesis,
    }


def _build_table_data(ideas: List[TradeIdea]) -> tuple[list, list]:
    """Return (rows, store_payload) for the DataTable + selection store."""
    rows = [_idea_to_row(idea, i) for i, idea in enumerate(ideas)]
    payload = [
        {
            "pair":             idea.pair,
            "tenor":            idea.tenor,
            "category":         idea.category,
            "direction":        idea.direction,
            "score":            idea.score,
            "structure_preset": idea.structure_preset,
            "vol_view":         idea.vol_view,
            "thesis":           idea.thesis,
            "metrics":          idea.metrics,
        }
        for idea in ideas
    ]
    return rows, payload


# ============================================================================
# Layout
# ============================================================================

_TABLE_COLUMNS = [
    {"name": "Score",     "id": "Score"},
    {"name": "Category",  "id": "Category"},
    {"name": "Pair",      "id": "Pair"},
    {"name": "Tenor",     "id": "Tenor"},
    {"name": "Direction", "id": "Direction"},
    {"name": "Structure", "id": "Structure"},
    {"name": "Thesis",    "id": "Thesis"},
]

_TABLE_HEADER_STYLE = {
    "backgroundColor": COLORS["bg_secondary"],
    "color": COLORS["text_secondary"],
    "fontWeight": "700", "fontSize": "10px",
    "textTransform": "uppercase", "letterSpacing": "0.5px",
    "border": f"1px solid {COLORS['border_subtle']}",
    "padding": "8px 6px",
}

_TABLE_CELL_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["text_primary"],
    "fontSize": "11px",
    "fontFamily": "'JetBrains Mono', monospace",
    "border": f"1px solid {COLORS['border_subtle']}",
    "padding": "6px 8px",
    "textAlign": "left",
}

_DIRECTION_STYLE_RULES = [
    {
        "if": {"filter_query": f'{{Direction}} = "{label}"', "column_id": "Direction"},
        "color": color, "fontWeight": "700",
    }
    for (_d, (label, color)) in DIRECTION_DISPLAY.items()
]


def _make_filter_bar() -> html.Div:
    """The category / region / tenor / refresh control row."""
    return html.Div([
        html.Div([
            html.Label("CATEGORY", style=LABEL_STYLE),
            dcc.Dropdown(
                id="ideas-cat-filter",
                options=CATEGORY_OPTIONS,
                value=ALL_CATEGORIES,
                multi=True, clearable=False,
                style={"fontSize": "11px"},
            ),
        ], style={"flex": "3", "minWidth": "260px"}),

        html.Div([
            html.Label("REGION", style=LABEL_STYLE),
            dcc.Dropdown(
                id="ideas-region-filter",
                options=REGION_OPTIONS, value="ALL", clearable=False,
                style={"fontSize": "11px"},
            ),
        ], style={"flex": "1", "minWidth": "120px"}),

        html.Div([
            html.Label("TENOR BUCKET", style=LABEL_STYLE),
            dcc.Dropdown(
                id="ideas-tenor-filter",
                options=TENOR_BUCKET_OPTIONS, value="ALL", clearable=False,
                style={"fontSize": "11px"},
            ),
        ], style={"flex": "1", "minWidth": "140px"}),

        html.Div([
            html.Label("MIN SCORE", style=LABEL_STYLE),
            dcc.Input(
                id="ideas-min-score",
                type="number", value=5.0, min=0, max=10, step=0.5,
                debounce=True,
                style={
                    "backgroundColor": COLORS["bg_input"],
                    "color": COLORS["text_primary"],
                    "border": f"1px solid {COLORS['border_subtle']}",
                    "padding": "6px 8px", "width": "100%",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontSize": "11px", "borderRadius": "0px",
                },
            ),
        ], style={"flex": "1", "minWidth": "90px"}),

        html.Div([
            html.Label("REFRESH", style=LABEL_STYLE),
            dcc.Dropdown(
                id="ideas-refresh-mode",
                options=REFRESH_OPTIONS, value=120_000, clearable=False,
                style={"fontSize": "11px"},
            ),
        ], style={"flex": "1", "minWidth": "120px"}),

        html.Div([
            html.Label(" ", style=LABEL_STYLE),
            html.Button("RESCAN", id="ideas-refresh-btn", n_clicks=0,
                        style={**BUTTON_STYLE, "padding": "8px 16px",
                               "fontSize": "10px", "width": "100%"}),
        ], style={"flex": "0 0 110px"}),
    ], style={
        "display": "flex", "gap": "10px", "alignItems": "flex-end",
        "marginBottom": "10px",
    })


def _make_action_bar() -> html.Div:
    """Selected-idea readout + 'Send to' buttons."""
    return html.Div([
        html.Div([
            html.Span("SELECTED:  ", style={
                "color": COLORS["text_secondary"], "fontSize": "10px",
                "letterSpacing": "1px", "fontWeight": "700",
            }),
            html.Span("(none — click a row)", id="ideas-selected-readout",
                      style={
                          "color": COLORS["text_primary"], "fontSize": "11px",
                          "fontFamily": "'JetBrains Mono', monospace",
                      }),
        ], style={"flex": "1", "minWidth": "0", "overflow": "hidden",
                   "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),

        html.Button("→ SEND TO VOL", id="ideas-send-vol-btn",
                    n_clicks=0, disabled=True,
                    style={**BUTTON_STYLE, "padding": "8px 16px",
                           "fontSize": "11px",
                           "backgroundColor": COLORS["accent_orange"],
                           "color": COLORS["bg_primary"],
                           "marginRight": "8px"}),

        html.Button("→ SEND TO STRUCTURER", id="ideas-send-stb-btn",
                    n_clicks=0, disabled=True,
                    style={**BUTTON_SUCCESS_STYLE, "padding": "8px 16px",
                           "fontSize": "11px"}),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "8px",
        "padding": "10px 12px",
        "border": f"1px solid {COLORS['border_subtle']}",
        "backgroundColor": COLORS["bg_secondary"],
        "marginBottom": "10px",
    })


def _make_status_bar() -> html.Div:
    return html.Div([
        html.Span("LAST SCAN: ", style={
            "color": COLORS["text_secondary"], "fontSize": "9px",
            "letterSpacing": "1px",
        }),
        html.Span("--", id="ideas-last-scan", style={
            "color": COLORS["text_primary"], "fontSize": "10px",
            "marginRight": "16px",
        }),
        html.Span("IDEAS: ", style={
            "color": COLORS["text_secondary"], "fontSize": "9px",
            "letterSpacing": "1px",
        }),
        html.Span("0", id="ideas-count", style={
            "color": COLORS["text_primary"], "fontSize": "10px",
        }),
    ], style={
        "padding": "6px 12px",
        "borderTop": f"1px solid {COLORS['border_subtle']}",
        "fontFamily": "'JetBrains Mono', monospace",
    })


def layout():
    return html.Div([
        # ── Stores ──
        dcc.Store(id="ideas-data-store", data=[]),         # all scanned ideas
        dcc.Store(id="ideas-selected-store", data=None),   # currently-selected idea payload
        dcc.Store(id="ideas-to-vol-store", data=None),     # handoff: ideas → VOL
        dcc.Store(id="ideas-to-structurer-store", data=None),  # handoff: ideas → Structurer

        # ── Refresh interval (controlled by ideas-refresh-mode) ──
        dcc.Interval(id="ideas-refresh-interval",
                     interval=120_000, n_intervals=0, disabled=False),

        html.Div([
            html.Div("TRADE IDEAS",
                     style={"color": COLORS["text_primary"], "fontSize": "13px",
                            "fontWeight": "700",
                            "fontFamily": "'JetBrains Mono', monospace",
                            "letterSpacing": "1.5px",
                            "textTransform": "uppercase",
                            "marginBottom": "12px",
                            "paddingBottom": "8px",
                            "borderBottom": f"1px solid {COLORS['border_subtle']}"}),

            _make_filter_bar(),
            _make_action_bar(),

            # Main table
            html.Div([
                dash_table.DataTable(
                    id="ideas-table",
                    columns=_TABLE_COLUMNS,
                    data=[],
                    row_selectable="single",
                    selected_rows=[],
                    page_action="native", page_size=25,
                    sort_action="native",
                    style_header=_TABLE_HEADER_STYLE,
                    style_cell=_TABLE_CELL_STYLE,
                    style_cell_conditional=[
                        {"if": {"column_id": "Score"},     "width": "60px",  "textAlign": "center"},
                        {"if": {"column_id": "Category"},  "width": "150px"},
                        {"if": {"column_id": "Pair"},      "width": "75px",  "textAlign": "center"},
                        {"if": {"column_id": "Tenor"},     "width": "75px",  "textAlign": "center"},
                        {"if": {"column_id": "Direction"}, "width": "110px", "textAlign": "center"},
                        {"if": {"column_id": "Structure"}, "width": "150px"},
                        {"if": {"column_id": "Thesis"},
                         "minWidth": "320px", "whiteSpace": "normal",
                         "color": COLORS["text_secondary"], "fontSize": "10px"},
                    ],
                    style_data_conditional=[
                        {"if": {"row_index": "odd"},
                         "backgroundColor": COLORS["bg_secondary"]},
                        {"if": {"state": "selected"},
                         "backgroundColor": COLORS["bg_card_hover"],
                         "border": f"1px solid {COLORS['accent_orange']}"},
                        {"if": {"column_id": "Score"},
                         "color": COLORS["accent_orange"], "fontWeight": "700"},
                        {"if": {"column_id": "Pair"},
                         "color": COLORS["accent_cyan"], "fontWeight": "700"},
                        *_DIRECTION_STYLE_RULES,
                    ],
                    style_table={"overflowX": "auto"},
                ),
            ], style={"marginBottom": "8px"}),

            _make_status_bar(),

        ], style={**CARD_STYLE, "padding": "16px"}),
    ])


# ============================================================================
# Callbacks
# ============================================================================

def register_callbacks(app):

    # ------------------------------------------------------------------
    # 1. Run scan + populate the table (fires on filter change, refresh
    #    button, or interval tick)
    # ------------------------------------------------------------------
    @app.callback(
        [Output("ideas-table", "data"),
         Output("ideas-table", "selected_rows"),
         Output("ideas-data-store", "data"),
         Output("ideas-last-scan", "children"),
         Output("ideas-count", "children")],
        [Input("ideas-refresh-btn", "n_clicks"),
         Input("ideas-refresh-interval", "n_intervals"),
         Input("ideas-cat-filter", "value"),
         Input("ideas-region-filter", "value"),
         Input("ideas-tenor-filter", "value"),
         Input("ideas-min-score", "value")],
    )
    def run_scan(_n_clicks, _n_intervals, categories,
                 region, tenor_bucket, min_score):
        try:
            ideas = scan_all(region=region or "ALL",
                             tenor_bucket=tenor_bucket or "ALL")
        except Exception:
            logger.exception("trade_ideas: scan_all failed")
            # Clear selection so stale row index can't fire on empty data
            return [], [], [], "ERROR", "0"

        # Filter by categories + min score. Preserve ALL_CATEGORIES order
        # so the table rows don't jitter between refreshes — `set()` here
        # would lose ordering and produce non-deterministic output.
        cat_set = set(categories or ALL_CATEGORIES)
        threshold = float(min_score) if min_score is not None else 0.0
        filtered = [i for i in ideas
                    if i.category in cat_set and i.score >= threshold]

        ordered_cats = [c for c in ALL_CATEGORIES if c in cat_set]
        ranked = top_n_per_category(filtered, n=DEFAULT_TOP_N,
                                    categories=ordered_cats)
        rows, payload = _build_table_data(ranked)
        ts = datetime.now().strftime("%H:%M:%S")
        # Clear selection on every refresh — row N may now refer to a
        # different idea, so persisting selection is misleading.
        return rows, [], payload, ts, str(len(ranked))

    # ------------------------------------------------------------------
    # 2. Refresh interval mode — Manual disables, otherwise sets period
    # ------------------------------------------------------------------
    @app.callback(
        [Output("ideas-refresh-interval", "interval"),
         Output("ideas-refresh-interval", "disabled")],
        [Input("ideas-refresh-mode", "value")],
    )
    def configure_interval(mode):
        if not mode or mode == 0:
            return 86_400_000, True   # parked, but a valid value
        return int(mode), False

    # ------------------------------------------------------------------
    # 3. Row selection → enable buttons, update readout
    # ------------------------------------------------------------------
    @app.callback(
        [Output("ideas-selected-store", "data"),
         Output("ideas-selected-readout", "children"),
         Output("ideas-send-vol-btn", "disabled"),
         Output("ideas-send-stb-btn", "disabled")],
        [Input("ideas-table", "selected_rows")],
        [State("ideas-data-store", "data")],
    )
    def select_row(selected_rows, payload):
        if not selected_rows or not payload:
            return None, "(none — click a row)", True, True
        idx = selected_rows[0]
        if idx >= len(payload):
            return None, "(none — click a row)", True, True
        idea = payload[idx]
        readout = (f"{idea['pair']} {idea['tenor']}  │  "
                   f"{idea['category']}  │  "
                   f"{idea['structure_preset']}  │  "
                   f"{idea['thesis']}")
        return idea, readout, False, False

    # ------------------------------------------------------------------
    # 4. Send to VOL — write the handoff store with all the info VOL needs
    # ------------------------------------------------------------------
    @app.callback(
        Output("ideas-to-vol-store", "data"),
        [Input("ideas-send-vol-btn", "n_clicks")],
        [State("ideas-selected-store", "data")],
        prevent_initial_call=True,
    )
    def send_to_vol(n_clicks, idea):
        if not n_clicks or not idea:
            raise PreventUpdate
        # The vol_view key in the idea matches a VIEW_PRESETS key in
        # vol_surface_fx. The destination handler resolves it to 4 chart slots.
        return {
            "pair":     idea["pair"],
            "tenor":    _resolve_tenor_for_vol(idea["tenor"]),
            "vol_view": idea["vol_view"],
            "category": idea["category"],
            "ts":       _time.time(),
        }

    # ------------------------------------------------------------------
    # 5. Send to Structurer — write handoff store with preset + pair + tenor
    # ------------------------------------------------------------------
    @app.callback(
        Output("ideas-to-structurer-store", "data"),
        [Input("ideas-send-stb-btn", "n_clicks")],
        [State("ideas-selected-store", "data")],
        prevent_initial_call=True,
    )
    def send_to_structurer(n_clicks, idea):
        if not n_clicks or not idea:
            raise PreventUpdate
        return {
            "pair":            idea["pair"],
            "tenor":           _resolve_tenor_for_structurer(idea["tenor"]),
            "preset":          idea["structure_preset"],
            "category":        idea["category"],
            "ts":              _time.time(),
        }


# ============================================================================
# Tenor normalisation helpers
# ============================================================================
#
# Some screens emit "1M-6M" or "1M-3M-6M" (term structure trades). The VOL
# panel expects a single tenor; the structurer expects a single tenor too.
# Pick a sensible canonical tenor for each case.

# Tenors actually present in the VOL Surface smile-tenor dropdown.
# Screen tenors that don't exist there (e.g. "2W") get snapped to the
# nearest available tenor so the dropdown doesn't blank out.
_VOL_TENORS = ("ON", "1W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "5Y")


def _snap_to_vol_tenor(tenor: str) -> str:
    """Map any tenor to one that exists in the VOL panel's dropdown."""
    if tenor in _VOL_TENORS:
        return tenor
    # Manual nearest mapping for the few screener tenors that have no
    # direct VOL match.
    return {"2W": "1W"}.get(tenor, "3M")


def _resolve_tenor_for_vol(tenor: str) -> str:
    """Pick a single tenor for VOL panel focus from a screen's tenor label."""
    if not tenor:
        return "3M"
    if "-" in tenor:
        # e.g. "1M-6M" → "3M", "1M-3M-6M" → "3M" (middle horizon)
        return "3M"
    return _snap_to_vol_tenor(tenor)


_STB_TENORS = ("ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y")


def _resolve_tenor_for_structurer(tenor: str) -> str:
    """Pick a single tenor for Structurer from a screen's tenor label."""
    if not tenor:
        return "3M"
    if "-" in tenor:
        # Multi-tenor structures (calendars) — give the structurer the
        # *front* leg, the calendar preset places the second leg via
        # tenor_mult.
        front = tenor.split("-")[0]
    else:
        front = tenor
    return front if front in _STB_TENORS else "3M"
