"""
Dark trading-desk theme configuration — v2.

Glassmorphism cards, animated gradients, neon accents,
Bloomberg Terminal-inspired color palette.
"""

from core.bloomberg import is_connected

# ── Color Palette ──────────────────────────────────────────────────────────
COLORS = {
    # Backgrounds (deeper, richer)
    "bg_primary":       "#060a13",
    "bg_secondary":     "#0c1220",
    "bg_card":          "#111827",
    "bg_card_hover":    "#1a2235",
    "bg_input":         "#0a0f1c",
    "bg_header":        "#080d18",
    "bg_glass":         "rgba(17, 24, 39, 0.75)",

    # Borders
    "border":           "#1e2a45",
    "border_focus":     "#3b82f6",
    "border_subtle":    "#151d30",
    "border_glow":      "rgba(59, 130, 246, 0.3)",

    # Text
    "text_primary":     "#e2e8f0",
    "text_secondary":   "#7c8db5",
    "text_muted":       "#475569",
    "text_bright":      "#f8fafc",

    # Accents (neon-bright)
    "accent_blue":      "#3b82f6",
    "accent_cyan":      "#06b6d4",
    "accent_green":     "#10b981",
    "accent_red":       "#ef4444",
    "accent_orange":    "#f59e0b",
    "accent_purple":    "#8b5cf6",
    "accent_pink":      "#ec4899",
    "accent_teal":      "#14b8a6",
    "accent_indigo":    "#6366f1",
    "accent_lime":      "#84cc16",
    "accent_amber":     "#f59e0b",
    "accent_rose":      "#f43f5e",

    # P&L
    "pnl_profit":       "#10b981",
    "pnl_loss":         "#ef4444",
    "pnl_neutral":      "#6b7280",

    # Gradients
    "grad_start":       "#3b82f6",
    "grad_end":         "#8b5cf6",
    "grad_cyan":        "#06b6d4",
    "grad_green":       "#10b981",

    # Bloomberg-style orange for the "connected" indicator
    "bbg_orange":       "#ff6600",
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
            "gridcolor": "rgba(30,42,69,0.5)",
            "zerolinecolor": COLORS["border"],
            "tickfont": {"size": 10, "color": COLORS["text_muted"]},
            "linecolor": COLORS["border_subtle"],
        },
        "yaxis": {
            "gridcolor": "rgba(30,42,69,0.5)",
            "zerolinecolor": COLORS["border"],
            "tickfont": {"size": 10, "color": COLORS["text_muted"]},
            "linecolor": COLORS["border_subtle"],
        },
        "colorway": [
            COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_purple"],
            COLORS["accent_green"], COLORS["accent_orange"], COLORS["accent_pink"],
            COLORS["accent_red"], COLORS["accent_teal"], COLORS["accent_indigo"],
        ],
        "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        "hoverlabel": {
            "bgcolor": COLORS["bg_secondary"],
            "bordercolor": COLORS["border"],
            "font": {"color": COLORS["text_primary"], "family": "monospace", "size": 12},
        },
    }
}


# ── Reusable Component Styles ─────────────────────────────────────────────

CARD_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "16px",
    "padding": "24px",
    "marginBottom": "16px",
    "boxShadow": "0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03)",
    "backdropFilter": "blur(12px)",
    "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    "position": "relative",
    "overflow": "hidden",
}

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

INPUT_STYLE = {
    "backgroundColor": COLORS["bg_input"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
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
    "fontFamily": "'JetBrains Mono', monospace",
    "textTransform": "uppercase",
    "letterSpacing": "1.2px",
    "marginBottom": "6px",
    "display": "block",
}

DROPDOWN_STYLE = {
    "backgroundColor": COLORS["bg_input"],
    "color": COLORS["text_primary"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "13px",
}

STAT_BOX_STYLE = {
    "backgroundColor": COLORS["bg_secondary"],
    "border": f"1px solid {COLORS['border_subtle']}",
    "borderRadius": "12px",
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
    "borderRadius": "10px",
    "padding": "11px 28px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "12px",
    "fontWeight": "700",
    "cursor": "pointer",
    "letterSpacing": "1px",
    "textTransform": "uppercase",
    "transition": "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
    "boxShadow": f"0 4px 14px rgba(59,130,246,0.3)",
}

BUTTON_DANGER_STYLE = {**BUTTON_STYLE, "backgroundColor": COLORS["accent_red"],
                       "boxShadow": "0 4px 14px rgba(239,68,68,0.3)"}
BUTTON_SUCCESS_STYLE = {**BUTTON_STYLE, "backgroundColor": COLORS["accent_green"],
                        "boxShadow": "0 4px 14px rgba(16,185,129,0.3)"}

TAB_STYLE = {
    "backgroundColor": "transparent",
    "border": f"1px solid {COLORS['border']}",
    "borderBottom": "none",
    "borderRadius": "10px 10px 0 0",
    "color": COLORS["text_muted"],
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "11px",
    "fontWeight": "600",
    "padding": "14px 24px",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
    "transition": "all 0.2s ease",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["accent_cyan"],
    "borderBottom": "none",
    "borderTop": f"2px solid {COLORS['accent_cyan']}",
    "boxShadow": f"0 -2px 12px rgba(6,182,212,0.15)",
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
        style["boxShadow"] = f"0 -2px 12px {color}26"
    return style
