"""
Dark trading-desk theme configuration.

Centralised color palette, fonts, and reusable Dash component styles.
"""

# ── Color Palette ──────────────────────────────────────────────────────────
COLORS = {
    # Backgrounds
    "bg_primary":       "#0a0e17",
    "bg_secondary":     "#111827",
    "bg_card":          "#1a1f2e",
    "bg_card_hover":    "#232a3d",
    "bg_input":         "#0d1321",
    "bg_header":        "#0c1220",

    # Borders
    "border":           "#2a3350",
    "border_focus":     "#4a90d9",
    "border_subtle":    "#1e2740",

    # Text
    "text_primary":     "#e8edf5",
    "text_secondary":   "#8892a8",
    "text_muted":       "#5a6580",
    "text_bright":      "#ffffff",

    # Accents
    "accent_blue":      "#4a90d9",
    "accent_cyan":      "#22d3ee",
    "accent_green":     "#10b981",
    "accent_red":       "#ef4444",
    "accent_orange":    "#f59e0b",
    "accent_purple":    "#8b5cf6",
    "accent_pink":      "#ec4899",

    # P&L
    "pnl_profit":       "#10b981",
    "pnl_loss":         "#ef4444",
    "pnl_neutral":      "#6b7280",

    # Gradients (for headers)
    "grad_start":       "#4a90d9",
    "grad_end":         "#8b5cf6",
}

# ── Plotly Chart Template ──────────────────────────────────────────────────
CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["bg_card"],
        "plot_bgcolor": COLORS["bg_card"],
        "font": {
            "family": "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
            "color": COLORS["text_secondary"],
            "size": 11,
        },
        "title": {"font": {"color": COLORS["text_primary"], "size": 14}},
        "xaxis": {
            "gridcolor": COLORS["border_subtle"],
            "zerolinecolor": COLORS["border"],
            "tickfont": {"size": 10},
        },
        "yaxis": {
            "gridcolor": COLORS["border_subtle"],
            "zerolinecolor": COLORS["border"],
            "tickfont": {"size": 10},
        },
        "colorway": [
            COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_purple"],
            COLORS["accent_green"], COLORS["accent_orange"], COLORS["accent_pink"],
            COLORS["accent_red"],
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
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "16px",
    "boxShadow": "0 4px 24px rgba(0,0,0,0.3)",
    "transition": "border-color 0.2s ease",
}

CARD_HEADER_STYLE = {
    "color": COLORS["text_primary"],
    "fontSize": "15px",
    "fontWeight": "600",
    "fontFamily": "'JetBrains Mono', monospace",
    "marginBottom": "16px",
    "paddingBottom": "10px",
    "borderBottom": f"1px solid {COLORS['border_subtle']}",
    "letterSpacing": "0.5px",
}

INPUT_STYLE = {
    "backgroundColor": COLORS["bg_input"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "color": COLORS["text_primary"],
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "13px",
    "padding": "8px 12px",
    "width": "100%",
}

LABEL_STYLE = {
    "color": COLORS["text_secondary"],
    "fontSize": "11px",
    "fontWeight": "500",
    "fontFamily": "'JetBrains Mono', monospace",
    "textTransform": "uppercase",
    "letterSpacing": "1px",
    "marginBottom": "4px",
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
    "borderRadius": "8px",
    "padding": "12px 16px",
    "textAlign": "center",
    "minWidth": "120px",
}

BUTTON_STYLE = {
    "backgroundColor": COLORS["accent_blue"],
    "color": COLORS["text_bright"],
    "border": "none",
    "borderRadius": "8px",
    "padding": "10px 24px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "13px",
    "fontWeight": "600",
    "cursor": "pointer",
    "letterSpacing": "0.5px",
    "transition": "all 0.2s ease",
}

BUTTON_DANGER_STYLE = {**BUTTON_STYLE, "backgroundColor": COLORS["accent_red"]}
BUTTON_SUCCESS_STYLE = {**BUTTON_STYLE, "backgroundColor": COLORS["accent_green"]}

TAB_STYLE = {
    "backgroundColor": COLORS["bg_secondary"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px 8px 0 0",
    "color": COLORS["text_secondary"],
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "12px",
    "fontWeight": "500",
    "padding": "12px 20px",
    "letterSpacing": "0.5px",
    "textTransform": "uppercase",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["accent_cyan"],
    "borderBottom": f"2px solid {COLORS['accent_cyan']}",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": COLORS["bg_secondary"],
    "color": COLORS["text_secondary"],
    "fontWeight": "600",
    "fontSize": "11px",
    "textTransform": "uppercase",
    "letterSpacing": "1px",
    "border": f"1px solid {COLORS['border_subtle']}",
    "padding": "10px 12px",
}

TABLE_CELL_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["text_primary"],
    "fontSize": "13px",
    "fontFamily": "'JetBrains Mono', monospace",
    "border": f"1px solid {COLORS['border_subtle']}",
    "padding": "8px 12px",
}
