"""
Bloomberg Terminal Theme — Pure Terminal Aesthetic.

PURE BLACK. MONOSPACE ONLY. NO ROUNDED CORNERS. NO SHADOWS. NO BLUR.
Grid cells separated by thin borders. Every pixel is information.
"""

from core.bloomberg import is_connected

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

    # Borders — dark blue-gray, thin, structural
    "border":           "#1a1a2e",
    "border_focus":     "#ff8800",
    "border_subtle":    "#1a1a2e",
    "border_glow":      "rgba(255, 136, 0, 0.25)",

    # Text — clear hierarchy, monospace only
    "text_primary":     "#d4d4d4",   # crisp silver — all data
    "text_secondary":   "#666666",   # muted — labels, column headers
    "text_muted":       "#666666",   # dimmed labels
    "text_bright":      "#ffffff",   # pure white — section headers

    # Bloomberg orange — selections, highlights, active items
    "accent_blue":      "#ff8800",
    "accent_cyan":      "#ff8800",
    "accent_indigo":    "#ff8800",

    # Secondary accent — same orange family
    "accent_orange":    "#ff8800",
    "accent_amber":     "#ff8800",
    "accent_yellow":    "#ff8800",

    # Semantic — Bloomberg green/red only
    "accent_green":     "#00cc66",   # positive P&L, vol decrease, long
    "accent_red":       "#ff3333",   # negative P&L, vol increase, short
    "accent_purple":    "#ff8800",   # map to orange
    "accent_teal":      "#ff8800",
    "accent_pink":      "#ff3333",
    "accent_lime":      "#00cc66",
    "accent_rose":      "#ff3333",

    # P&L
    "pnl_profit":       "#00cc66",
    "pnl_loss":         "#ff3333",
    "pnl_neutral":      "#666666",

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
    return "BLOOMBERG LIVE" if is_connected() else "SYNTHETIC DATA"


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
            "color": "#d4d4d4",
            "size": 11,
        },
        "colorway": [
            "#ff8800",   # orange primary
            "#ffffff",   # white secondary
            "#00cc66",   # green
            "#ff3333",   # red
            "#d4d4d4",   # silver
            "#666666",   # gray
            "#ffaa33",   # light orange
            "#88ff88",   # light green
        ],
        "hoverlabel": {
            "bgcolor": "#000000",
            "bordercolor": "#1a1a2e",
            "font": {"color": "#d4d4d4", "family": "'JetBrains Mono', monospace", "size": 11},
        },
    }
}

# Default axis styling — apply separately via xaxis=AXIS_DEFAULTS etc.
AXIS_DEFAULTS = {
    "gridcolor": "#1a1a2e",
    "zerolinecolor": "#333355",
    "tickfont": {"size": 9, "color": "#888888"},
    "linecolor": "#333355",
}

TITLE_DEFAULTS = {"font": {"color": "#ffffff", "size": 13}}


def chart_layout(**overrides):
    """Build a complete chart layout dict from CHART_TEMPLATE + axis defaults + overrides.

    Usage: fig.update_layout(**chart_layout(height=300, title=dict(text="My Chart")))

    Deep-merges axis/title dicts so callers can pass xaxis=dict(title='X')
    without losing default grid colors, tick fonts, etc.
    """
    base = dict(CHART_TEMPLATE["layout"])

    # Deep-merge: axis defaults + caller overrides (caller wins on conflict)
    base["xaxis"] = {**AXIS_DEFAULTS, **(overrides.pop("xaxis", {}))}
    base["yaxis"] = {**AXIS_DEFAULTS, **(overrides.pop("yaxis", {}))}
    base["title"] = {**TITLE_DEFAULTS, **(overrides.pop("title", {}))}

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
    "border": "1px solid #1a1a2e",
    "borderRadius": "0px",
    "padding": "8px",
    "marginBottom": "0px",
}

CARD_HEADER_STYLE = {
    "color": "#ffffff",
    "fontSize": "16px",
    "fontWeight": "700",
    "fontFamily": "'JetBrains Mono', monospace",
    "marginBottom": "8px",
    "paddingBottom": "4px",
    "borderBottom": "1px solid #1a1a2e",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
}

INPUT_STYLE = {
    "backgroundColor": "#000000",
    "border": "1px solid #1a1a2e",
    "borderRadius": "0px",
    "color": "#d4d4d4",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "11px",
    "padding": "6px 8px",
    "width": "100%",
}

LABEL_STYLE = {
    "color": "#666666",
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
    "color": "#d4d4d4",
    "border": "1px solid #1a1a2e",
    "borderRadius": "0px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "11px",
}

STAT_BOX_STYLE = {
    "backgroundColor": "#000000",
    "border": "1px solid #1a1a2e",
    "borderRadius": "0px",
    "padding": "8px 12px",
    "textAlign": "center",
    "minWidth": "100px",
}

BUTTON_STYLE = {
    "backgroundColor": "#ff8800",
    "color": "#000000",
    "border": "none",
    "borderRadius": "0px",
    "padding": "6px 16px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "10px",
    "fontWeight": "700",
    "cursor": "pointer",
    "letterSpacing": "0.8px",
    "textTransform": "uppercase",
}

BUTTON_DANGER_STYLE = {**BUTTON_STYLE, "backgroundColor": "#ff3333", "color": "#000000"}
BUTTON_SUCCESS_STYLE = {**BUTTON_STYLE, "backgroundColor": "#00cc66", "color": "#000000"}

TAB_STYLE = {
    "backgroundColor": "transparent",
    "border": "1px solid #1a1a2e",
    "borderBottom": "none",
    "borderRadius": "0px",
    "color": "#666666",
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
    "backgroundColor": "#000000",
    "color": "#666666",
    "fontWeight": "700",
    "fontSize": "9px",
    "textTransform": "uppercase",
    "letterSpacing": "1.2px",
    "border": "1px solid #1a1a2e",
    "padding": "6px 8px",
}

TABLE_CELL_STYLE = {
    "backgroundColor": "#000000",
    "color": "#d4d4d4",
    "fontSize": "11px",
    "fontFamily": "'JetBrains Mono', monospace",
    "border": "1px solid #1a1a2e",
    "padding": "4px 8px",
}


# ── Helper: stat box with optional color accent ────────────────────────────
def make_stat_style(color=None):
    style = {**STAT_BOX_STYLE}
    if color:
        style["borderTop"] = f"2px solid {color}"
    return style


# ── Clickable metric helper ────────────────────────────────────────────────
def clickable_stat(value, label, pair, metric, tenor, color=None):
    """Renders a stat value that opens a time series popup when clicked."""
    from dash import html
    return html.Div([
        html.Div(str(value), className="stat-value", style={
            "fontSize": "13px", "fontWeight": "700", "color": color or "#d4d4d4",
            "fontFamily": "'JetBrains Mono', monospace", "cursor": "pointer",
        }, id={"type": "clickable-metric", "pair": pair, "metric": metric, "tenor": tenor}),
        html.Div(label, className="stat-label", style={
            "fontSize": "9px", "color": "#666666", "textTransform": "uppercase",
            "letterSpacing": "1px", "fontFamily": "'JetBrains Mono', monospace",
        }),
    ], className="stat-box", style=make_stat_style(color))


# ── Grid cell helper ───────────────────────────────────────────────────────
def grid_cell(children, **kwargs):
    """Wrap content in a terminal grid cell."""
    from dash import html
    style = {
        "backgroundColor": "#000000",
        "border": "1px solid #1a1a2e",
        "padding": "8px",
        **kwargs.pop("style", {}),
    }
    return html.Div(children, style=style, **kwargs)


# ── Section header helper ──────────────────────────────────────────────────
def section_header(text):
    """Render a section header in terminal style."""
    from dash import html
    return html.Div(text, style={
        "color": "#ffffff",
        "fontSize": "16px",
        "fontWeight": "700",
        "fontFamily": "'JetBrains Mono', monospace",
        "letterSpacing": "1.5px",
        "textTransform": "uppercase",
        "marginBottom": "4px",
        "paddingBottom": "4px",
        "borderBottom": "1px solid #1a1a2e",
    })
