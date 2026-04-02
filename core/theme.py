"""
Bloomberg Terminal Theme — Pure Terminal Aesthetic.

PURE BLACK. MONOSPACE ONLY. NO ROUNDED CORNERS. NO SHADOWS. NO BLUR.
Grid cells separated by thin borders. Every pixel is information.
"""

import logging
import functools
from core.bloomberg import is_connected

logger = logging.getLogger(__name__)

# ── Color Palette ──────────────────────────────────────────────────────────
COLORS = {
    # Backgrounds — pure black, no navy, no gradients
    "bg_primary":       "#000000",
    "bg_secondary":     "#000000",
    "bg_card":          "#000000",
    "bg_card_hover":    "#0a0a0a",
    "bg_input":         "#000000",
    "bg_header":        "#000000",
    "bg_glass":         "#000000",
    "bg_stat":          "#06060f",

    # Borders — visible blue-gray, thin, structural
    "border":           "#2d2d50",
    "border_focus":     "#ff8800",
    "border_subtle":    "#2d2d50",
    "border_glow":      "rgba(255, 136, 0, 0.30)",

    # Text — clear hierarchy, monospace only
    "text_primary":     "#e0e0e0",   # bright silver — all data (high legibility on black)
    "text_secondary":   "#9a9ab0",   # muted blue-gray — labels, column headers
    "text_muted":       "#9a9ab0",   # dimmed labels
    "text_bright":      "#ffffff",   # pure white — section headers

    # Bloomberg orange — primary accent for selections, highlights
    "accent_blue":      "#4a9eff",   # cool blue — secondary data series
    "accent_cyan":      "#00b4d8",   # cyan — tertiary data, info metrics
    "accent_indigo":    "#7b8cde",   # muted indigo — quaternary series

    # Secondary accent — warm family
    "accent_orange":    "#ff8800",   # Bloomberg orange — primary accent
    "accent_amber":     "#ffaa00",   # warm amber — secondary warm
    "accent_yellow":    "#e6c619",   # muted gold — highlights

    # Semantic — Bloomberg green/red only
    "accent_green":     "#00cc66",   # positive P&L, vol decrease, long
    "accent_red":       "#ff3333",   # negative P&L, vol increase, short
    "accent_purple":    "#a78bfa",   # soft purple — distinct category
    "accent_teal":      "#2dd4bf",   # teal — distinct category
    "accent_pink":      "#ff3333",
    "accent_lime":      "#00cc66",
    "accent_rose":      "#ff3333",

    # P&L
    "pnl_profit":       "#00cc66",
    "pnl_loss":         "#ff3333",
    "pnl_neutral":      "#808080",

    # Gradients — not used in terminal theme but kept for compat
    "grad_start":       "#ff8800",
    "grad_end":         "#ff8800",
    "grad_cyan":        "#ff8800",
    "grad_green":       "#00cc66",

    # Bloomberg orange indicator
    "bbg_orange":       "#ff8800",
}


def status_color():
    return COLORS["accent_green"] if is_connected() else COLORS["accent_orange"]


def status_text():
    return "BLOOMBERG LIVE" if is_connected() else "DISCONNECTED"


# ── Heatmap Colorscales ──────────────────────────────────────────────────
CS_VOL_SURFACE = [
    [0, "#1a1a3e"], [0.2, "#2a2a5e"], [0.4, "#6b4400"],
    [0.6, "#bf6b00"], [0.8, "#ff8800"], [1.0, "#ffcc66"],
]
CS_PNL_DIVERGING = [
    [0, "#ff3333"], [0.3, "#330000"], [0.5, "#0e0e0e"],
    [0.7, "#003300"], [1.0, "#00cc66"],
]
CS_DIVERGING_GR = [  # green-to-red, zero=black
    [0, "#00cc66"], [0.5, "#000000"], [1, "#ff3333"],
]
CS_DIVERGING_RG = [  # red-to-green, zero=black
    [0, "#ff3333"], [0.5, "#000000"], [1, "#00cc66"],
]
CS_RICHNESS = [
    [0.00, "#1565c0"], [0.20, "#0d5a9e"], [0.35, "#1a2a4a"],
    [0.50, "#2a2a40"], [0.65, "#4a2a1a"], [0.80, "#b85c00"],
    [1.00, "#ff8800"],
]
CS_SKEW = [
    [0.00, "#c62828"], [0.25, "#6d2020"], [0.50, "#2a2a40"],
    [0.75, "#1a3a6d"], [1.00, "#1e88e5"],
]


# ── Plotly Chart Template ──────────────────────────────────────────────────
# Base layout values. Used via **CHART_TEMPLATE["layout"] spread.
# IMPORTANT: Only include keys that will NEVER be overridden by callers.
# title, xaxis, yaxis, margin are intentionally excluded to avoid
# "got multiple values for keyword argument" errors.
CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#000000",
        "plot_bgcolor": "#000000",
        "font": {
            "family": "'JetBrains Mono', monospace",
            "color": "#e0e0e0",
            "size": 11,
        },
        "colorway": [
            "#ff8800",   # orange primary
            "#ffffff",   # white secondary
            "#00cc66",   # green
            "#ff3333",   # red
            "#d4d4d4",   # silver
            "#808080",   # gray
            "#ffaa33",   # light orange
            "#1565c0",   # cold blue
        ],
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor": "#0a0a1a",
            "bordercolor": "#ff8800",
            "font": {"color": "#e0e0e0", "family": "'JetBrains Mono', monospace", "size": 11},
        },
    }
}

# ── Layout Constants ───────────────────────────────────────────────────────
# Spacing: 8px base grid. Use GAP for within-group, SECTION_GAP between groups.
GAP = "8px"
SECTION_GAP = "16px"

# Chart heights: use these instead of ad-hoc pixel values.
CHART_SM = 240    # sparklines, secondary charts
CHART_MD = 320    # standard charts (main content)
CHART_LG = 400    # hero charts, heatmaps, 3D surfaces

# Default axis styling — apply separately via xaxis=AXIS_DEFAULTS etc.
AXIS_DEFAULTS = {
    "showgrid": True,
    "gridcolor": "#1e1e38",
    "zerolinecolor": "#3a3a5c",
    "zerolinewidth": 1,
    "tickfont": {"size": 9, "color": "#9a9ab0"},
    "linecolor": "#3a3a5c",
}

TITLE_DEFAULTS = {"font": {"color": "#ffffff", "size": 13}}


def no_data_fig(height=300, msg="NO DATA"):
    """Return a polished empty Plotly figure with skeleton grid lines and centered label."""
    import plotly.graph_objects as go
    fig = go.Figure()
    # Add faux grid lines to give skeleton chart appearance
    for y in [0.2, 0.4, 0.6, 0.8]:
        fig.add_shape(type="line", x0=0, x1=1, y0=y, y1=y,
                      xref="paper", yref="paper",
                      line=dict(color="#1a1a30", width=1))
    fig.update_layout(
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        annotations=[dict(
            text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color="#4a4a6a",
                                        family="'JetBrains Mono', monospace"),
        )],
    )
    return fig


def skeleton_chart(height=300, label="LOADING"):
    """Return an HTML skeleton placeholder with shimmer animation for loading states."""
    from dash import html
    return html.Div(className="skeleton-chart", style={"height": f"{height}px"}, children=[
        html.Div(className="skeleton-grid", children=[
            html.Div(className="skeleton-grid-line") for _ in range(5)
        ]),
        html.Div(label, className="skeleton-label"),
    ])


def chart_layout(**overrides):
    """Build a complete chart layout dict from CHART_TEMPLATE + axis defaults + overrides.

    Usage: fig.update_layout(**chart_layout(height=300, title=dict(text="My Chart")))

    Deep-merges axis/title dicts so callers can pass xaxis=dict(title='X')
    without losing default grid colors, tick fonts, etc.
    """
    import copy as _copy
    base = _copy.deepcopy(CHART_TEMPLATE["layout"])

    # Deep-merge: axis defaults + caller overrides (caller wins on conflict)
    base["xaxis"] = {**AXIS_DEFAULTS, **(overrides.pop("xaxis", {}))}
    base["yaxis"] = {**AXIS_DEFAULTS, **(overrides.pop("yaxis", {}))}
    _title_override = overrides.pop("title", {})
    if isinstance(_title_override, str):
        _title_override = {"text": _title_override}
    base["title"] = {**TITLE_DEFAULTS, **_title_override}

    if "margin" not in overrides:
        base["margin"] = {"l": 50, "r": 20, "t": 40, "b": 40}

    # Handle xaxis_title / yaxis_title shorthand (Plotly convenience)
    if "xaxis_title" in overrides:
        base["xaxis"]["title"] = overrides.pop("xaxis_title")
    if "yaxis_title" in overrides:
        base["yaxis"]["title"] = overrides.pop("yaxis_title")

    base.update(overrides)
    return base


# ── Reusable Component Styles ─────────────────────────────────────────────

# Grid cell — no card, no rounded corners, no shadows, no blur
CARD_STYLE = {
    "backgroundColor": "#000000",
    "border": "1px solid #2d2d50",
    "borderRadius": "0px",
    "padding": "14px",
    "marginBottom": "0px",
}

CARD_HEADER_STYLE = {
    "color": "#ffffff",
    "fontSize": "12px",
    "fontWeight": "700",
    "fontFamily": "'JetBrains Mono', monospace",
    "marginBottom": "8px",
    "paddingBottom": "4px",
    "borderBottom": "1px solid #222240",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
}

INPUT_STYLE = {
    "backgroundColor": "#000000",
    "border": "1px solid #2d2d50",
    "borderRadius": "0px",
    "color": "#e0e0e0",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "11px",
    "padding": "6px 8px",
    "width": "100%",
}

LABEL_STYLE = {
    "color": "#9a9ab0",
    "fontSize": "9px",
    "fontWeight": "600",
    "fontFamily": "'JetBrains Mono', monospace",
    "textTransform": "uppercase",
    "letterSpacing": "1.2px",
    "marginBottom": "4px",
    "display": "block",
}

DROPDOWN_STYLE = {
    "backgroundColor": "#000000",
    "color": "#e0e0e0",
    "border": "1px solid #2d2d50",
    "borderRadius": "0px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "11px",
}

STAT_BOX_STYLE = {
    "backgroundColor": "#06060f",
    "border": "1px solid #2d2d50",
    "borderRadius": "0px",
    "padding": "14px",
    "textAlign": "center",
    "minWidth": "100px",
}

BUTTON_STYLE = {
    "backgroundColor": "#ff8800",
    "color": "#000000",
    "border": "none",
    "borderRadius": "0px",
    "padding": "5px 14px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "10px",
    "fontWeight": "700",
    "cursor": "pointer",
    "letterSpacing": "0.8px",
    "textTransform": "uppercase",
}

BUTTON_DANGER_STYLE = {**BUTTON_STYLE, "backgroundColor": "#ff3333", "color": "#000000"}
BUTTON_SUCCESS_STYLE = {**BUTTON_STYLE, "backgroundColor": "#00cc66", "color": "#000000"}

CSV_BTN_STYLE = {
    "backgroundColor": "transparent",
    "color": "#9a9ab0",
    "border": "1px solid #333",
    "borderRadius": "0px",
    "padding": "1px 6px",
    "fontSize": "8px",
    "fontFamily": "'JetBrains Mono', monospace",
    "cursor": "pointer",
    "letterSpacing": "0.5px",
    "textTransform": "uppercase",
}

TAB_STYLE = {
    "backgroundColor": "transparent",
    "border": "1px solid #2d2d50",
    "borderBottom": "none",
    "borderRadius": "0px",
    "color": "#9a9ab0",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "10px",
    "fontWeight": "600",
    "padding": "8px 16px",
    "letterSpacing": "1.2px",
    "textTransform": "uppercase",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "backgroundColor": "#000000",
    "color": "#ff8800",
    "borderBottom": "none",
    "borderTop": "2px solid #ff8800",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": "#0a0a14",
    "color": "#ffffff",
    "fontWeight": "700",
    "fontSize": "10px",
    "textTransform": "uppercase",
    "letterSpacing": "1.2px",
    "border": "1px solid #2d2d50",
    "padding": "6px 8px",
}

TABLE_CELL_STYLE = {
    "backgroundColor": "#000000",
    "color": "#e0e0e0",
    "fontSize": "11px",
    "fontFamily": "'JetBrains Mono', monospace",
    "border": "1px solid #2d2d50",
    "padding": "5px 8px",
}


# ── Helper: stat box with optional color accent ────────────────────────────
def make_stat_style(color=None):
    style = {**STAT_BOX_STYLE}
    if color:
        style["borderLeft"] = f"3px solid {color}"
    return style


# ── Clickable metric helper ────────────────────────────────────────────────
def clickable_stat(value, label, pair, metric, tenor, color=None):
    """Renders a stat value with color accent."""
    from dash import html
    return html.Div([
        html.Div(str(value), className="stat-value", style={
            "fontSize": "16px", "fontWeight": "700", "color": color or "#d4d4d4",
            "fontFamily": "'JetBrains Mono', monospace",
        }, id={"type": "clickable-metric", "pair": pair, "metric": metric, "tenor": tenor}),
        html.Div(label, className="stat-label", style={
            "fontSize": "10px", "color": "#9a9ab0", "textTransform": "uppercase",
            "letterSpacing": "1px", "fontFamily": "'JetBrains Mono', monospace",
            "marginTop": "4px",
        }),
    ], className="stat-box", style=make_stat_style(color))


# ── Grid cell helper ───────────────────────────────────────────────────────
def grid_cell(children, **kwargs):
    """Wrap content in a terminal grid cell."""
    from dash import html
    style = {
        "backgroundColor": "#000000",
        "border": "1px solid #2d2d50",
        "padding": "14px",
        **kwargs.pop("style", {}),
    }
    return html.Div(children, style=style, **kwargs)


# ── Section header helper ──────────────────────────────────────────────────
def section_header(text):
    """Render a section header in terminal style."""
    from dash import html
    return html.Div(text, style={
        "color": "#ffffff",
        "fontSize": "12px",
        "fontWeight": "700",
        "fontFamily": "'JetBrains Mono', monospace",
        "letterSpacing": "1.5px",
        "textTransform": "uppercase",
        "marginBottom": "8px",
        "paddingBottom": "6px",
        "borderBottom": "1px solid #3a3a5c",
    })


# ── Simple stat box (non-clickable) ─────────────────────────────────────────
def stat_box(label, value, color=None):
    """Render a simple stat box with label + value. Use for KPIs, summaries."""
    from dash import html
    return html.Div([
        html.Div(str(value), style={
            "fontSize": "15px", "fontWeight": "700",
            "color": color or COLORS["text_primary"],
            "fontFamily": "'JetBrains Mono', monospace",
        }),
        html.Div(label, style={
            "fontSize": "9px", "color": COLORS["text_muted"],
            "textTransform": "uppercase", "letterSpacing": "1px",
            "fontFamily": "'JetBrains Mono', monospace",
            "marginTop": "4px",
        }),
    ], style=make_stat_style(color))


# ── Safe chart decorator ────────────────────────────────────────────────────
def safe_chart(fn):
    """Decorator: wrap chart functions so exceptions return a clean empty
    figure with the error message instead of crashing the panel.

    Usage::

        @safe_chart
        def chart_my_thing(pair, sd, spot, r_dom, r_for, **kw):
            ...
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.exception("Chart %s failed", fn.__name__)
            return no_data_fig(msg=f"{fn.__name__}: {exc}")
    return wrapper
