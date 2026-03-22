"""
Dark trading-desk theme configuration — v3.

Refined Bloomberg palette: confident blue + warm amber.
Ultra-clean institutional aesthetic. Two primary accents only.
"""

from core.bloomberg import is_connected

# ── Color Palette ──────────────────────────────────────────────────────────
COLORS = {
    # Backgrounds — deep navy, never pure black
    "bg_primary":       "#070b14",
    "bg_secondary":     "#0c1222",
    "bg_card":          "#0f1729",
    "bg_card_hover":    "#162035",
    "bg_input":         "#0a0f1c",
    "bg_header":        "#080d18",
    "bg_glass":         "rgba(12, 18, 32, 0.90)",

    # Borders — barely visible structure
    "border":           "#1a2540",
    "border_focus":     "#4a9eff",
    "border_subtle":    "#131c30",
    "border_glow":      "rgba(74, 158, 255, 0.25)",

    # Text — clear 4-tier hierarchy
    "text_primary":     "#c8d6e5",   # main data / values
    "text_secondary":   "#6b7d99",   # labels / descriptions
    "text_muted":       "#3d4f68",   # least important
    "text_bright":      "#e8edf4",   # highlighted / headers

    # Primary accent — confident blue (all interactive elements)
    "accent_blue":      "#4a9eff",
    "accent_cyan":      "#4a9eff",   # alias for backwards compat
    "accent_indigo":    "#4a9eff",   # alias

    # Secondary accent — warm amber/gold (highlights, warnings, secondary data)
    "accent_orange":    "#f0a030",
    "accent_amber":     "#f0a030",   # alias
    "accent_yellow":    "#f0a030",   # alias

    # Tertiary — used sparingly for specific semantic meanings only
    "accent_green":     "#22c55e",   # profit, success, positive
    "accent_red":       "#e5484d",   # loss, danger, negative
    "accent_purple":    "#8b7cf6",   # structuring / exotics (muted)
    "accent_teal":      "#4a9eff",   # alias to primary
    "accent_pink":      "#e5484d",   # alias to loss red
    "accent_lime":      "#22c55e",   # alias to profit green
    "accent_rose":      "#e5484d",   # alias to loss red

    # P&L — distinctive, not generic
    "pnl_profit":       "#22c55e",   # money green
    "pnl_loss":         "#e5484d",   # blood red
    "pnl_neutral":      "#5a6a80",

    # Gradients — blue to amber for warmth
    "grad_start":       "#4a9eff",
    "grad_end":         "#f0a030",
    "grad_cyan":        "#4a9eff",
    "grad_green":       "#22c55e",

    # Bloomberg-style orange for the "connected" indicator
    "bbg_orange":       "#f0a030",
}


def status_color():
    return COLORS["accent_green"] if is_connected() else COLORS["accent_orange"]


def status_text():
    return "BLOOMBERG LIVE" if is_connected() else "SYNTHETIC DATA"


# ── Plotly Chart Template ──────────────────────────────────────────────────
CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
            "color": COLORS["text_secondary"],
            "size": 11,
        },
        "title": {"font": {"color": COLORS["text_primary"], "size": 14}},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.04)",
            "zerolinecolor": "rgba(255,255,255,0.08)",
            "tickfont": {"size": 10, "color": COLORS["text_muted"]},
            "linecolor": COLORS["border_subtle"],
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.04)",
            "zerolinecolor": "rgba(255,255,255,0.08)",
            "tickfont": {"size": 10, "color": COLORS["text_muted"]},
            "linecolor": COLORS["border_subtle"],
        },
        "colorway": [
            "#4a9eff",   # primary blue
            "#f0a030",   # amber
            "#8b7cf6",   # muted purple
            "#22c55e",   # green
            "#e5484d",   # red
            "#5ec4d4",   # soft teal
            "#c084fc",   # lavender
            "#fb923c",   # soft orange
        ],
        "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        "hoverlabel": {
            "bgcolor": "#0f1729",
            "bordercolor": "#1a2540",
            "font": {"color": COLORS["text_primary"], "family": "monospace", "size": 12},
        },
    }
}


# ── Reusable Component Styles ─────────────────────────────────────────────

CARD_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "24px",
    "marginBottom": "16px",
    "boxShadow": "0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03)",
    "backdropFilter": "blur(16px)",
    "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    "position": "relative",
    "overflow": "hidden",
}

CARD_HEADER_STYLE = {
    "color": COLORS["text_bright"],
    "fontSize": "12px",
    "fontWeight": "700",
    "fontFamily": "'Inter', -apple-system, sans-serif",
    "marginBottom": "18px",
    "paddingBottom": "12px",
    "borderBottom": f"1px solid {COLORS['border_subtle']}",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
}

INPUT_STYLE = {
    "backgroundColor": COLORS["bg_input"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "color": COLORS["text_primary"],
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "13px",
    "padding": "10px 14px",
    "width": "100%",
    "transition": "border-color 0.2s ease, box-shadow 0.2s ease",
}

LABEL_STYLE = {
    "color": COLORS["text_secondary"],
    "fontSize": "10px",
    "fontWeight": "600",
    "fontFamily": "'Inter', -apple-system, sans-serif",
    "textTransform": "uppercase",
    "letterSpacing": "1.2px",
    "marginBottom": "6px",
    "display": "block",
}

DROPDOWN_STYLE = {
    "backgroundColor": COLORS["bg_input"],
    "color": COLORS["text_primary"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "13px",
}

STAT_BOX_STYLE = {
    "backgroundColor": COLORS["bg_secondary"],
    "border": f"1px solid {COLORS['border_subtle']}",
    "borderRadius": "10px",
    "padding": "16px 20px",
    "textAlign": "center",
    "minWidth": "120px",
    "position": "relative",
    "overflow": "hidden",
    "transition": "all 0.3s ease",
}

BUTTON_STYLE = {
    "backgroundColor": COLORS["accent_blue"],
    "color": COLORS["text_bright"],
    "border": "none",
    "borderRadius": "8px",
    "padding": "11px 28px",
    "fontFamily": "'Inter', -apple-system, sans-serif",
    "fontSize": "12px",
    "fontWeight": "700",
    "cursor": "pointer",
    "letterSpacing": "0.8px",
    "textTransform": "uppercase",
    "transition": "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
    "boxShadow": "0 4px 14px rgba(74,158,255,0.25)",
}

BUTTON_DANGER_STYLE = {**BUTTON_STYLE, "backgroundColor": COLORS["accent_red"],
                       "boxShadow": "0 4px 14px rgba(229,72,77,0.25)"}
BUTTON_SUCCESS_STYLE = {**BUTTON_STYLE, "backgroundColor": COLORS["accent_green"],
                        "boxShadow": "0 4px 14px rgba(34,197,94,0.25)"}

TAB_STYLE = {
    "backgroundColor": "transparent",
    "border": f"1px solid {COLORS['border']}",
    "borderBottom": "none",
    "borderRadius": "8px 8px 0 0",
    "color": COLORS["text_muted"],
    "fontFamily": "'Inter', -apple-system, sans-serif",
    "fontSize": "11px",
    "fontWeight": "600",
    "padding": "12px 22px",
    "letterSpacing": "1.2px",
    "textTransform": "uppercase",
    "transition": "all 0.2s ease",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["accent_blue"],
    "borderBottom": "none",
    "borderTop": f"2px solid {COLORS['accent_blue']}",
    "boxShadow": "0 -2px 12px rgba(74,158,255,0.12)",
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

# ── Helper: make a stat box with optional glow ────────────────────────────
def make_stat_style(color=None):
    style = {**STAT_BOX_STYLE}
    if color:
        style["borderTop"] = f"2px solid {color}"
        style["boxShadow"] = f"0 -2px 12px {color}20"
    return style
