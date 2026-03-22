"""
Desk-Wide Vol Signal Scanner
==============================
Scans all 30 FX pairs across vol percentile, z-score, IV-RV spread,
skew, and term-structure metrics. Surfaces actionable signals for the
desk in a sortable / filterable DataTable with drill-down sparklines
on row click.

Exports layout() and register_callbacks(app).
"""

import logging
import traceback

import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, TABLE_HEADER_STYLE, TABLE_CELL_STYLE,
)
from core.bloomberg_fx import (
    get_fx_vol_surface, get_fx_spots, get_all_pairs,
    get_fx_historical_vol, get_fx_rates,
)
from core.fx_analytics import (
    vol_percentile, vol_zscore, iv_rv_percentile, vol_regime_detect,
    rv_scanner, vol_change, carry_per_vol,
)
from core.fx_conventions import FX_PAIR_REGISTRY, tenor_to_years

log = logging.getLogger(__name__)


# ============================================================================
#  Constants
# ============================================================================

_SCANNER_TENORS = ["1M", "3M", "1Y"]

_GROUP_OPTIONS = [
    {"label": "ALL", "value": "ALL"},
    {"label": "G10 MAJOR", "value": "G10_MAJOR"},
    {"label": "G10 CROSS", "value": "G10_CROSS"},
    {"label": "SCANDIE", "value": "SCANDIE"},
    {"label": "EM", "value": "EM"},
]

_SIGNAL_OPTIONS = [
    {"label": "ALL", "value": "ALL"},
    {"label": "VOL CHEAP", "value": "VOL_CHEAP"},
    {"label": "VOL CHEAP-ISH", "value": "VOL_CHEAPISH"},
    {"label": "VOL RICH", "value": "VOL_RICH"},
    {"label": "SKEW EXTREME", "value": "SKEW_EXTREME"},
    {"label": "TERM STEEP", "value": "TERM_STEEP"},
    {"label": "RV OPPORTUNITY", "value": "RV_OPPORTUNITY"},
]

_LOOKBACK_OPTIONS = [
    {"label": "60 d", "value": 60},
    {"label": "120 d", "value": 120},
    {"label": "252 d", "value": 252},
]

# Map filter labels to subgroups / groups in FX_PAIR_REGISTRY
_GROUP_MAP = {
    "G10_MAJOR": lambda spec: spec.group == "G10" and spec.subgroup == "Majors",
    "G10_CROSS": lambda spec: spec.group == "G10" and spec.subgroup == "Crosses",
    "SCANDIE":   lambda spec: spec.group == "G10" and spec.subgroup == "Scandies",
    "EM":        lambda spec: spec.group == "EM",
}

# Column definitions for the scanner DataTable
_TABLE_COLUMNS = [
    {"name": "Pair",          "id": "pair",         "type": "text"},
    {"name": "Group",         "id": "group",        "type": "text"},
    {"name": "Spot",          "id": "spot",         "type": "numeric", "format": dash_table.FormatTemplate.money(4)},
    {"name": "Chg%",          "id": "chg_pct",      "type": "numeric", "format": dash_table.FormatTemplate.percentage(2)},
    {"name": "ATM 1M",        "id": "atm_1m",       "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "ATM 3M",        "id": "atm_3m",       "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "ATM 1Y",        "id": "atm_1y",       "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "25D RR 3M",     "id": "rr25_3m",      "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "25D BF 3M",     "id": "bf25_3m",      "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "IV-RV 3M",      "id": "ivrv_3m",      "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "ATM 1M %ile",   "id": "atm1m_pct",    "type": "numeric", "format": {"specifier": ".0f"}},
    {"name": "ATM 3M %ile",   "id": "atm3m_pct",    "type": "numeric", "format": {"specifier": ".0f"}},
    {"name": "RR 3M %ile",    "id": "rr3m_pct",     "type": "numeric", "format": {"specifier": ".0f"}},
    {"name": "BF 3M %ile",    "id": "bf3m_pct",     "type": "numeric", "format": {"specifier": ".0f"}},
    {"name": "Z(ATM 3M)",     "id": "z_atm3m",      "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Z(RR 3M)",      "id": "z_rr3m",       "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Term Spread",   "id": "term_spread",   "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Signal",        "id": "signal",        "type": "text"},
]

# Neutral default percentile -- 50 means "no opinion", never triggers a signal
_DEFAULT_PERCENTILE = 50.0
# Neutral default z-score -- 0 means nothing unusual
_DEFAULT_ZSCORE = 0.0


# ============================================================================
#  Signal Logic
# ============================================================================

def compute_signal(atm_pct, rr_pct, ivrv_z, term_z):
    """
    Determine desk-level signals from percentile and z-score inputs.

    Parameters
    ----------
    atm_pct : float   ATM 3M percentile (0-100)
    rr_pct  : float   25D RR 3M percentile (0-100)
    ivrv_z  : float   IV-RV spread z-score
    term_z  : float   Term spread z-score (1M-1Y)

    Returns
    -------
    str   Pipe-delimited signal string, or dash when nothing fires.
    """
    signals = []

    # --- Vol level signals (ATM percentile) ---
    if atm_pct < 10:
        signals.append("VOL CHEAP")
    elif atm_pct < 15:
        signals.append("VOL CHEAP-ISH")

    if atm_pct > 90:
        signals.append("VOL RICH")

    # --- Skew signals (risk-reversal percentile) ---
    if rr_pct < 10 or rr_pct > 90:
        signals.append("SKEW EXTREME")

    # --- Term-structure signal ---
    if abs(term_z) > 2:
        signals.append("TERM STEEP")

    # --- IV vs RV divergence ---
    if ivrv_z < -1:
        signals.append("RV BUY")
    if ivrv_z > 1:
        signals.append("RV SELL")

    return " | ".join(signals) if signals else "\u2014"


def _signal_matches_filter(signal_str, filter_value):
    """Return True if *signal_str* contains the token implied by *filter_value*."""
    if filter_value == "ALL":
        return True
    mapping = {
        "VOL_CHEAP": "VOL CHEAP",
        "VOL_CHEAPISH": "VOL CHEAP-ISH",
        "VOL_RICH": "VOL RICH",
        "SKEW_EXTREME": "SKEW EXTREME",
        "TERM_STEEP": "TERM STEEP",
        "RV_OPPORTUNITY": "RV",
    }
    token = mapping.get(filter_value, "")
    return token in signal_str


# ============================================================================
#  Pair Grouping Helpers
# ============================================================================

def _pair_group_label(pair):
    """Short human label for the pair's group."""
    spec = FX_PAIR_REGISTRY.get(pair)
    if spec is None:
        return "OTHER"
    if spec.subgroup == "Majors":
        return "G10 MAJ"
    if spec.subgroup == "Crosses":
        return "G10 X"
    if spec.subgroup == "Scandies":
        return "SCANDIE"
    if spec.group == "EM":
        return spec.subgroup.upper()[:6]
    return spec.group


def _pairs_for_group(group_value):
    """Return the list of pairs that match the group dropdown value."""
    all_pairs = list(FX_PAIR_REGISTRY.keys())
    if group_value == "ALL":
        return all_pairs
    pred = _GROUP_MAP.get(group_value)
    if pred is None:
        return all_pairs
    return [p for p in all_pairs if pred(FX_PAIR_REGISTRY[p])]


# ============================================================================
#  Data-Build Function  (robust, per-pair isolation)
# ============================================================================

def _safe_float(value, default=0.0):
    """Coerce to float, returning *default* on any failure."""
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _safe_atm(surface, tenor):
    """Extract ATM vol from the surface dict returned by get_fx_vol_surface."""
    try:
        if isinstance(surface, dict) and tenor in surface:
            val = surface[tenor].get("atm", None)
            return _safe_float(val, default=np.nan)
    except Exception:
        pass
    return np.nan


def _safe_metric(surface, tenor, key):
    """Extract an arbitrary metric (rr25, bf25, etc.) from the surface dict."""
    try:
        if isinstance(surface, dict) and tenor in surface:
            val = surface[tenor].get(key, None)
            return _safe_float(val, default=0.0)
    except Exception:
        pass
    return 0.0


def _safe_percentile(pair, tenor, metric, lookback):
    """Call vol_percentile with full error isolation."""
    try:
        info = vol_percentile(pair, tenor, metric, lookback)
        if isinstance(info, dict):
            pct = _safe_float(info.get("percentile"), _DEFAULT_PERCENTILE)
            # Clamp to valid range
            return max(0.0, min(100.0, pct))
    except Exception:
        log.debug("vol_percentile failed for %s %s %s", pair, tenor, metric)
    return _DEFAULT_PERCENTILE


def _safe_zscore(pair, tenor, metric, lookback):
    """Call vol_zscore with full error isolation."""
    try:
        info = vol_zscore(pair, tenor, metric, lookback)
        if isinstance(info, dict):
            z = _safe_float(info.get("zscore"), _DEFAULT_ZSCORE)
            # Clamp to a sane range to avoid display issues
            return max(-10.0, min(10.0, z))
    except Exception:
        log.debug("vol_zscore failed for %s %s %s", pair, tenor, metric)
    return _DEFAULT_ZSCORE


def _safe_ivrv(pair, tenor, lookback):
    """Call iv_rv_percentile with full error isolation.

    Returns (spread, z_score) tuple.
    """
    try:
        ivrv = iv_rv_percentile(pair, tenor, lookback=lookback)
        if not isinstance(ivrv, dict):
            return 0.0, _DEFAULT_ZSCORE

        spread = _safe_float(ivrv.get("current_spread"), 0.0)
        mean_s = _safe_float(ivrv.get("mean_spread"), 0.0)
        std_s = _safe_float(ivrv.get("std_spread"), 1.0)
        z = (spread - mean_s) / max(abs(std_s), 1e-6)
        z = max(-10.0, min(10.0, z))
        return spread, z
    except Exception:
        log.debug("iv_rv_percentile failed for %s %s", pair, tenor)
    return 0.0, _DEFAULT_ZSCORE


def _build_default_row(pair):
    """Return a fully-neutral row that will never trigger false signals."""
    return {
        "pair":         pair,
        "group":        _pair_group_label(pair),
        "spot":         np.nan,
        "chg_pct":      0.0,
        "atm_1m":       np.nan,
        "atm_3m":       np.nan,
        "atm_1y":       np.nan,
        "rr25_3m":      0.0,
        "bf25_3m":      0.0,
        "ivrv_3m":      0.0,
        "atm1m_pct":    _DEFAULT_PERCENTILE,
        "atm3m_pct":    _DEFAULT_PERCENTILE,
        "rr3m_pct":     _DEFAULT_PERCENTILE,
        "bf3m_pct":     _DEFAULT_PERCENTILE,
        "z_atm3m":      _DEFAULT_ZSCORE,
        "z_rr3m":       _DEFAULT_ZSCORE,
        "term_spread":  0.0,
        "signal":       "\u2014",
    }


def _build_scanner_data(pairs, lookback):
    """
    Build a list-of-dicts ready for DataTable consumption.

    For every pair, fetches spot, vol surface, percentile/z-score for
    ATM (1M, 3M, 1Y), 25D RR/BF 3M, IV-RV spread, and term structure.

    Each pair is computed independently so one failure never crashes
    the entire scanner.
    """
    # Fetch all spots in one call; wrap in case the whole batch fails
    try:
        spots = get_fx_spots(pairs)
        if not isinstance(spots, dict):
            spots = {}
    except Exception:
        log.warning("get_fx_spots batch call failed; falling back to empty")
        spots = {}

    rows = []

    for pair in pairs:
        try:
            row = _build_single_pair(pair, spots, lookback)
            rows.append(row)
        except Exception:
            log.warning("Scanner failed for %s:\n%s", pair, traceback.format_exc())
            rows.append(_build_default_row(pair))

    return rows


def _build_single_pair(pair, spots, lookback):
    """Build one scanner row.  Raises on total failure; individual fields
    fall back to safe defaults internally."""

    # ---- Spot ----
    spot_info = spots.get(pair)
    if isinstance(spot_info, dict):
        spot_mid = _safe_float(spot_info.get("mid"), np.nan)
        spot_chg = _safe_float(spot_info.get("change_pct"), 0.0)
    else:
        spot_mid = np.nan
        spot_chg = 0.0

    # ---- Vol surface ----
    try:
        surface = get_fx_vol_surface(pair)
        if not isinstance(surface, dict):
            surface = {}
    except Exception:
        surface = {}

    # ---- ATM vols across tenors ----
    atm_1m = _safe_atm(surface, "1M")
    atm_3m = _safe_atm(surface, "3M")
    atm_1y = _safe_atm(surface, "1Y")

    # ---- 25D risk-reversal and butterfly 3M ----
    rr25 = _safe_metric(surface, "3M", "rr25")
    bf25 = _safe_metric(surface, "3M", "bf25")

    # ---- Percentiles (each individually safe) ----
    atm1m_pct = _safe_percentile(pair, "1M", "ATM", lookback)
    atm3m_pct = _safe_percentile(pair, "3M", "ATM", lookback)
    rr3m_pct  = _safe_percentile(pair, "3M", "25D_RR", lookback)
    bf3m_pct  = _safe_percentile(pair, "3M", "25D_BF", lookback)

    # ---- Z-scores (each individually safe) ----
    z_atm3m = _safe_zscore(pair, "3M", "ATM", lookback)
    z_rr3m  = _safe_zscore(pair, "3M", "25D_RR", lookback)

    # ---- IV-RV ----
    ivrv_spread, ivrv_z = _safe_ivrv(pair, "3M", lookback)

    # ---- Term spread (1M - 1Y) ----
    if np.isnan(atm_1m) or np.isnan(atm_1y):
        term_spread = 0.0
        term_z = _DEFAULT_ZSCORE
    else:
        term_spread = atm_1m - atm_1y
        z_atm1m = _safe_zscore(pair, "1M", "ATM", lookback)
        z_atm1y = _safe_zscore(pair, "1Y", "ATM", lookback)
        term_z = z_atm1m - z_atm1y

    # ---- Composite signal ----
    signal = compute_signal(atm3m_pct, rr3m_pct, ivrv_z, term_z)

    return {
        "pair":         pair,
        "group":        _pair_group_label(pair),
        "spot":         round(spot_mid, 5) if not np.isnan(spot_mid) else None,
        "chg_pct":      round(spot_chg / 100.0, 4),
        "atm_1m":       round(atm_1m, 2) if not np.isnan(atm_1m) else None,
        "atm_3m":       round(atm_3m, 2) if not np.isnan(atm_3m) else None,
        "atm_1y":       round(atm_1y, 2) if not np.isnan(atm_1y) else None,
        "rr25_3m":      round(rr25, 2),
        "bf25_3m":      round(bf25, 2),
        "ivrv_3m":      round(ivrv_spread, 2),
        "atm1m_pct":    round(atm1m_pct, 0),
        "atm3m_pct":    round(atm3m_pct, 0),
        "rr3m_pct":     round(rr3m_pct, 0),
        "bf3m_pct":     round(bf3m_pct, 0),
        "z_atm3m":      round(z_atm3m, 2),
        "z_rr3m":       round(z_rr3m, 2),
        "term_spread":  round(term_spread, 2),
        "signal":       signal,
    }


# ============================================================================
#  Conditional Formatting Helpers
# ============================================================================

def _build_conditional_styles():
    """
    Build the style_data_conditional list for the scanner DataTable.

    Five-tier gradient for percentile columns:
      < 20      very blue    bold
      20 - 35   blue-ish
      35 - 65   neutral
      65 - 80   orange-ish
      > 80      very red     bold

    Three-tier intensity for z-score columns:
      |z| < 1   faint color
      1 - 2     medium color
      |z| > 2   strong color  bold

    Signal column gets background badges.
    """
    styles = []

    # ---- Percentile columns: 5-tier gradient ----
    pct_cols = ["atm1m_pct", "atm3m_pct", "rr3m_pct", "bf3m_pct"]
    for col in pct_cols:
        # Tier 1: < 20 -- deep blue, bold, background tint
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} < 20",
                "column_id": col,
            },
            "color": "#60a5fa",  # bright blue text
            "backgroundColor": "rgba(59, 130, 246, 0.18)",
            "fontWeight": "700",
        })
        # Tier 2: 20-35 -- soft blue
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} >= 20 && {{{col}}} < 35",
                "column_id": col,
            },
            "color": COLORS["accent_cyan"],
            "backgroundColor": "rgba(6, 182, 212, 0.08)",
        })
        # Tier 3: 35-65 -- neutral (no override, inherits default)
        # (intentionally omitted so the default cell style applies)

        # Tier 4: 65-80 -- warm orange
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} >= 65 && {{{col}}} < 80",
                "column_id": col,
            },
            "color": COLORS["accent_orange"],
            "backgroundColor": "rgba(245, 158, 11, 0.08)",
        })
        # Tier 5: >= 80 -- deep red, bold, background tint
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} >= 80",
                "column_id": col,
            },
            "color": "#f87171",  # bright red text
            "backgroundColor": "rgba(239, 68, 68, 0.18)",
            "fontWeight": "700",
        })

    # ---- Z-score columns: 3-tier intensity ----
    z_cols = ["z_atm3m", "z_rr3m"]
    for col in z_cols:
        # Faint negative: -1 to 0 (light cyan)
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} >= -1 && {{{col}}} < -0.01",
                "column_id": col,
            },
            "color": "rgba(6, 182, 212, 0.55)",
        })
        # Medium negative: -2 to -1
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} >= -2 && {{{col}}} < -1",
                "column_id": col,
            },
            "color": COLORS["accent_cyan"],
            "backgroundColor": "rgba(6, 182, 212, 0.08)",
        })
        # Strong negative: < -2
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} < -2",
                "column_id": col,
            },
            "color": "#60a5fa",
            "backgroundColor": "rgba(59, 130, 246, 0.18)",
            "fontWeight": "700",
        })
        # Faint positive: 0 to 1 (light orange)
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} > 0.01 && {{{col}}} <= 1",
                "column_id": col,
            },
            "color": "rgba(245, 158, 11, 0.55)",
        })
        # Medium positive: 1 to 2
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} > 1 && {{{col}}} <= 2",
                "column_id": col,
            },
            "color": COLORS["accent_orange"],
            "backgroundColor": "rgba(245, 158, 11, 0.08)",
        })
        # Strong positive: > 2
        styles.append({
            "if": {
                "filter_query": f"{{{col}}} > 2",
                "column_id": col,
            },
            "color": "#f87171",
            "backgroundColor": "rgba(239, 68, 68, 0.18)",
            "fontWeight": "700",
        })

    # ---- Signal column badges (background-colored for prominence) ----
    # Green badge: CHEAP / BUY signals
    styles.append({
        "if": {
            "filter_query": '{signal} contains "VOL CHEAP"',
            "column_id": "signal",
        },
        "color": "#34d399",
        "backgroundColor": "rgba(16, 185, 129, 0.20)",
        "fontWeight": "700",
        "borderLeft": f"3px solid {COLORS['accent_green']}",
    })
    styles.append({
        "if": {
            "filter_query": '{signal} contains "RV BUY"',
            "column_id": "signal",
        },
        "color": "#34d399",
        "backgroundColor": "rgba(16, 185, 129, 0.20)",
        "fontWeight": "700",
        "borderLeft": f"3px solid {COLORS['accent_green']}",
    })
    # Red badge: RICH / SELL signals
    styles.append({
        "if": {
            "filter_query": '{signal} contains "RICH"',
            "column_id": "signal",
        },
        "color": "#f87171",
        "backgroundColor": "rgba(239, 68, 68, 0.20)",
        "fontWeight": "700",
        "borderLeft": f"3px solid {COLORS['accent_red']}",
    })
    styles.append({
        "if": {
            "filter_query": '{signal} contains "RV SELL"',
            "column_id": "signal",
        },
        "color": "#f87171",
        "backgroundColor": "rgba(239, 68, 68, 0.20)",
        "fontWeight": "700",
        "borderLeft": f"3px solid {COLORS['accent_red']}",
    })
    # Amber badge: SKEW EXTREME / TERM STEEP
    styles.append({
        "if": {
            "filter_query": '{signal} contains "SKEW"',
            "column_id": "signal",
        },
        "color": "#fbbf24",
        "backgroundColor": "rgba(245, 158, 11, 0.20)",
        "fontWeight": "700",
        "borderLeft": f"3px solid {COLORS['accent_orange']}",
    })
    styles.append({
        "if": {
            "filter_query": '{signal} contains "TERM"',
            "column_id": "signal",
        },
        "color": "#fbbf24",
        "backgroundColor": "rgba(245, 158, 11, 0.20)",
        "fontWeight": "700",
        "borderLeft": f"3px solid {COLORS['accent_orange']}",
    })
    # Dash (no signal) -- muted
    styles.append({
        "if": {
            "filter_query": '{signal} = "\u2014"',
            "column_id": "signal",
        },
        "color": COLORS["text_muted"],
    })

    # ---- IV-RV colour ----
    styles.append({
        "if": {
            "filter_query": "{ivrv_3m} < -1.5",
            "column_id": "ivrv_3m",
        },
        "color": "#60a5fa",
        "backgroundColor": "rgba(59, 130, 246, 0.12)",
        "fontWeight": "700",
    })
    styles.append({
        "if": {
            "filter_query": "{ivrv_3m} > 1.5",
            "column_id": "ivrv_3m",
        },
        "color": "#f87171",
        "backgroundColor": "rgba(239, 68, 68, 0.12)",
        "fontWeight": "700",
    })

    # ---- Spot chg_pct colour ----
    styles.append({
        "if": {
            "filter_query": "{chg_pct} > 0",
            "column_id": "chg_pct",
        },
        "color": COLORS["accent_green"],
    })
    styles.append({
        "if": {
            "filter_query": "{chg_pct} < 0",
            "column_id": "chg_pct",
        },
        "color": COLORS["accent_red"],
    })

    # ---- Selected row highlight ----
    styles.append({
        "if": {"state": "selected"},
        "backgroundColor": COLORS["bg_card_hover"],
        "border": f"1px solid {COLORS['accent_cyan']}",
    })

    return styles


# ============================================================================
#  Sparkline Chart Builders
# ============================================================================

_SPARK_HEIGHT = 180
_SPARK_MARGIN = dict(l=40, r=12, t=30, b=24)


def _empty_spark(title=""):
    """Return a minimal empty chart matching the dark theme."""
    fig = go.Figure()
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=_SPARK_HEIGHT,
        margin=_SPARK_MARGIN,
        title=dict(text=title, font=dict(size=11, color=COLORS["text_secondary"])),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text="Select a row", showarrow=False,
            font=dict(size=12, color=COLORS["text_muted"]),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )],
    )
    return fig


def _apply_spark_layout(fig, title, yformat=".1f", show_xgrid=False):
    """Apply consistent minimal styling to a sparkline figure."""
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=_SPARK_HEIGHT,
        margin=_SPARK_MARGIN,
        title=dict(text=title, font=dict(size=11, color=COLORS["text_primary"])),
        xaxis=dict(
            showticklabels=False,
            showgrid=show_xgrid,
            zeroline=False,
        ),
        yaxis=dict(
            title=None,
            tickformat=yformat,
            showgrid=False,
            zeroline=False,
        ),
        showlegend=False,
    )
    return fig


def _add_current_annotation(fig, current_val, fmt=".2f", color=None):
    """Add a large current-value annotation in the top-right corner."""
    color = color or COLORS["accent_cyan"]
    fig.add_annotation(
        text=f"<b>{current_val:{fmt}}</b>",
        xref="paper", yref="paper",
        x=0.98, y=0.95,
        showarrow=False,
        font=dict(size=16, color=color, family="'JetBrains Mono', monospace"),
        xanchor="right", yanchor="top",
    )


def _spark_atm_history(pair, lookback=120):
    """ATM 3M implied vol history with +/-1 sigma bands."""
    hist = get_fx_historical_vol(pair, "3M", "ATM", lookback)
    if hist is None or len(hist) < 10:
        return _empty_spark("ATM 3M")

    vals = np.array(hist, dtype=float)
    n = len(vals)
    x = list(range(n))
    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    current = float(vals[-1])

    fig = go.Figure()

    # +/-1 sigma band (filled region)
    fig.add_trace(go.Scatter(
        x=x, y=[mu + sigma] * n,
        mode="lines", line=dict(color=COLORS["accent_orange"], width=1, dash="dot"),
        name="+1\u03c3", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=[mu - sigma] * n,
        mode="lines", line=dict(color=COLORS["accent_orange"], width=1, dash="dot"),
        name="-1\u03c3", showlegend=False,
        fill="tonexty", fillcolor="rgba(245,158,11,0.06)", hoverinfo="skip",
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=x, y=[mu] * n,
        mode="lines", line=dict(color=COLORS["text_muted"], width=1, dash="dash"),
        name="mean", showlegend=False, hoverinfo="skip",
    ))

    # Main series
    fig.add_trace(go.Scatter(
        x=x, y=vals.tolist(),
        mode="lines", line=dict(color=COLORS["accent_cyan"], width=2),
        name="ATM 3M",
        hovertemplate="%{y:.2f}<extra></extra>",
    ))

    # Current level marker
    fig.add_trace(go.Scatter(
        x=[n - 1], y=[current],
        mode="markers", marker=dict(color=COLORS["accent_cyan"], size=8, symbol="circle",
                                     line=dict(color=COLORS["text_bright"], width=1)),
        name="current", showlegend=False,
        hovertemplate=f"Current: {current:.2f}<extra></extra>",
    ))

    _apply_spark_layout(fig, f"ATM 3M  \u2022  {pair}")
    _add_current_annotation(fig, current, fmt=".2f", color=COLORS["accent_cyan"])
    return fig


def _spark_rr_history(pair, lookback=120):
    """25-delta risk reversal 3M history."""
    hist = get_fx_historical_vol(pair, "3M", "25D_RR", lookback)
    if hist is None or len(hist) < 10:
        return _empty_spark("25D RR 3M")

    vals = np.array(hist, dtype=float)
    n = len(vals)
    x = list(range(n))
    current = float(vals[-1])

    fig = go.Figure()

    # Zero line
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["text_muted"])

    # Main series  (rose when negative, cyan when positive)
    fig.add_trace(go.Scatter(
        x=x, y=vals.tolist(),
        mode="lines", line=dict(color=COLORS["accent_purple"], width=2),
        name="25D RR 3M",
        hovertemplate="%{y:.2f}<extra></extra>",
    ))

    # Current marker with colour based on sign
    marker_color = COLORS["accent_rose"] if current < 0 else COLORS["accent_cyan"]
    fig.add_trace(go.Scatter(
        x=[n - 1], y=[current],
        mode="markers", marker=dict(color=marker_color, size=8,
                                     line=dict(color=COLORS["text_bright"], width=1)),
        showlegend=False,
        hovertemplate=f"Current: {current:.2f}<extra></extra>",
    ))

    _apply_spark_layout(fig, f"25D RR 3M  \u2022  {pair}", yformat=".2f")
    ann_color = COLORS["accent_rose"] if current < 0 else COLORS["accent_cyan"]
    _add_current_annotation(fig, current, fmt="+.2f", color=ann_color)
    return fig


def _spark_ivrv_history(pair, lookback=120):
    """IV minus RV spread history as filled area chart."""
    iv_hist = get_fx_historical_vol(pair, "3M", "ATM", lookback)
    if iv_hist is None or len(iv_hist) < 20:
        return _empty_spark("IV-RV Spread")

    from core.fx_analytics import iv_rv_spread as _ivrv_fn
    try:
        df = _ivrv_fn(pair, "3M", rv_window=20, lookback=lookback)
    except Exception:
        return _empty_spark("IV-RV Spread")

    if df is None or df.empty or len(df) < 10:
        return _empty_spark("IV-RV Spread")

    spread = df["spread"].values
    n = len(spread)
    x = list(range(n))
    current = float(spread[-1])

    fig = go.Figure()

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["text_muted"])

    # Positive fill (IV rich) in rose-tint, negative (IV cheap) in cyan-tint
    pos_y = np.where(spread >= 0, spread, 0)
    neg_y = np.where(spread < 0, spread, 0)

    fig.add_trace(go.Scatter(
        x=x, y=pos_y.tolist(),
        fill="tozeroy", fillcolor="rgba(244, 63, 94, 0.15)",
        line=dict(color=COLORS["accent_rose"], width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=neg_y.tolist(),
        fill="tozeroy", fillcolor="rgba(6, 182, 212, 0.15)",
        line=dict(color=COLORS["accent_cyan"], width=0),
        showlegend=False, hoverinfo="skip",
    ))

    # Overlay the main line
    fig.add_trace(go.Scatter(
        x=x, y=spread.tolist(),
        mode="lines", line=dict(color=COLORS["accent_amber"], width=1.5),
        name="IV-RV",
        hovertemplate="%{y:.2f}<extra></extra>",
    ))

    # Current marker
    marker_color = COLORS["accent_rose"] if current > 0 else COLORS["accent_cyan"]
    fig.add_trace(go.Scatter(
        x=[n - 1], y=[current],
        mode="markers", marker=dict(color=marker_color, size=8,
                                     line=dict(color=COLORS["text_bright"], width=1)),
        showlegend=False,
        hovertemplate=f"Current: {current:.2f}<extra></extra>",
    ))

    _apply_spark_layout(fig, f"IV-RV Spread  \u2022  {pair}")
    ann_color = COLORS["accent_rose"] if current > 0 else COLORS["accent_cyan"]
    _add_current_annotation(fig, current, fmt="+.2f", color=ann_color)
    return fig


def _spark_term_structure(pair):
    """ATM term structure snapshot across standard tenors."""
    try:
        surface = get_fx_vol_surface(pair)
    except Exception:
        return _empty_spark("Term Structure")

    if not isinstance(surface, dict) or len(surface) == 0:
        return _empty_spark("Term Structure")

    tenors = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]
    labels = []
    vols = []
    for t in tenors:
        if t in surface:
            try:
                v = surface[t].get("atm", None)
                if v is not None and float(v) > 0:
                    labels.append(t)
                    vols.append(float(v))
            except (TypeError, ValueError):
                continue

    if len(labels) < 2:
        return _empty_spark("Term Structure")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels, y=vols,
        mode="lines+markers",
        line=dict(color=COLORS["accent_cyan"], width=2),
        marker=dict(color=COLORS["accent_cyan"], size=6, symbol="circle",
                    line=dict(color=COLORS["text_bright"], width=1)),
        name="ATM",
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))

    # Highlight the front-end and back-end with amber dots
    if len(vols) >= 2:
        fig.add_trace(go.Scatter(
            x=[labels[0], labels[-1]],
            y=[vols[0], vols[-1]],
            mode="markers",
            marker=dict(color=COLORS["accent_amber"], size=9, symbol="diamond",
                        line=dict(color=COLORS["text_bright"], width=1)),
            showlegend=False,
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        ))

    _apply_spark_layout(fig, f"ATM Term Structure  \u2022  {pair}", yformat=".1f")
    # Show front-back spread as annotation
    if len(vols) >= 2:
        spread = vols[0] - vols[-1]
        spread_color = COLORS["accent_amber"]
        fig.add_annotation(
            text=f"<b>{labels[0]}-{labels[-1]}: {spread:+.1f}</b>",
            xref="paper", yref="paper",
            x=0.98, y=0.95,
            showarrow=False,
            font=dict(size=13, color=spread_color, family="'JetBrains Mono', monospace"),
            xanchor="right", yanchor="top",
        )
    return fig


# ============================================================================
#  Top Movers Summary Row
# ============================================================================

def _stat_box(label, value, sub_text, color):
    """Create a single Top Movers stat box component."""
    return html.Div([
        html.Div(label, style={
            **LABEL_STYLE,
            "marginBottom": "2px",
            "fontSize": "9px",
        }),
        html.Div(value, style={
            "color": color,
            "fontSize": "18px",
            "fontWeight": "700",
            "fontFamily": "'JetBrains Mono', monospace",
            "lineHeight": "1.2",
        }),
        html.Div(sub_text, style={
            "color": COLORS["text_muted"],
            "fontSize": "10px",
            "fontFamily": "'JetBrains Mono', monospace",
            "marginTop": "2px",
        }),
    ], style={
        **STAT_BOX_STYLE,
        "borderTop": f"2px solid {color}",
        "boxShadow": f"0 -2px 12px {color}26",
        "flex": "1",
        "minWidth": "140px",
    })


def _build_top_movers(rows):
    """Build the Top Movers stat boxes from the scanner row data.

    Returns a list of Dash html components for the 5 key boxes:
      - Cheapest Vol
      - Richest Vol
      - Biggest Skew
      - Largest IV-RV Gap
      - Total Signals
    """
    if not rows:
        return []

    df = pd.DataFrame(rows)

    # Filter out default/failed rows (pct == 50 and z == 0 for everything)
    valid = df[df["signal"] != "\u2014"].copy() if len(df) > 0 else df

    boxes = []

    # ---- 1. Cheapest Vol (lowest ATM 3M percentile) ----
    try:
        idx = df["atm3m_pct"].idxmin()
        pair = df.loc[idx, "pair"]
        pct = df.loc[idx, "atm3m_pct"]
        boxes.append(_stat_box(
            "CHEAPEST VOL", pair, f"{pct:.0f}th %ile",
            COLORS["accent_cyan"],
        ))
    except Exception:
        boxes.append(_stat_box("CHEAPEST VOL", "\u2014", "", COLORS["accent_cyan"]))

    # ---- 2. Richest Vol (highest ATM 3M percentile) ----
    try:
        idx = df["atm3m_pct"].idxmax()
        pair = df.loc[idx, "pair"]
        pct = df.loc[idx, "atm3m_pct"]
        boxes.append(_stat_box(
            "RICHEST VOL", pair, f"{pct:.0f}th %ile",
            COLORS["accent_red"],
        ))
    except Exception:
        boxes.append(_stat_box("RICHEST VOL", "\u2014", "", COLORS["accent_red"]))

    # ---- 3. Biggest Skew (most extreme RR z-score by magnitude) ----
    try:
        abs_z = df["z_rr3m"].abs()
        idx = abs_z.idxmax()
        pair = df.loc[idx, "pair"]
        z_val = df.loc[idx, "z_rr3m"]
        boxes.append(_stat_box(
            "BIGGEST SKEW", pair, f"z = {z_val:+.2f}",
            COLORS["accent_purple"],
        ))
    except Exception:
        boxes.append(_stat_box("BIGGEST SKEW", "\u2014", "", COLORS["accent_purple"]))

    # ---- 4. Largest IV-RV Gap ----
    try:
        abs_gap = df["ivrv_3m"].abs()
        idx = abs_gap.idxmax()
        pair = df.loc[idx, "pair"]
        gap_val = df.loc[idx, "ivrv_3m"]
        direction = "IV RICH" if gap_val > 0 else "IV CHEAP"
        boxes.append(_stat_box(
            "LARGEST IV-RV GAP", pair, f"{gap_val:+.1f}v  ({direction})",
            COLORS["accent_amber"],
        ))
    except Exception:
        boxes.append(_stat_box("LARGEST IV-RV GAP", "\u2014", "", COLORS["accent_amber"]))

    # ---- 5. Total Signals ----
    try:
        signal_count = len(df[df["signal"] != "\u2014"])
        total = len(df)
        pct_firing = (signal_count / max(total, 1)) * 100
        boxes.append(_stat_box(
            "TOTAL SIGNALS", f"{signal_count} / {total}",
            f"{pct_firing:.0f}% of pairs",
            COLORS["accent_green"] if signal_count > 0 else COLORS["text_muted"],
        ))
    except Exception:
        boxes.append(_stat_box("TOTAL SIGNALS", "0", "", COLORS["text_muted"]))

    return boxes


# ============================================================================
#  Legacy Summary Stats Row (counts by signal type)
# ============================================================================

def _build_summary_stats(rows):
    """Build breakdown stat boxes from scanner row data."""
    if not rows:
        return []

    df = pd.DataFrame(rows)
    total = len(df)
    cheap = len(df[df["signal"].str.contains("CHEAP", na=False)])
    rich = len(df[df["signal"].str.contains("RICH", na=False)])
    skew = len(df[df["signal"].str.contains("SKEW", na=False)])
    rv_opp = len(df[df["signal"].str.contains("RV", na=False)])
    avg_pct = df["atm3m_pct"].mean()
    avg_z = df["z_atm3m"].mean()

    stats = [
        ("PAIRS SCANNED", str(total), COLORS["accent_cyan"]),
        ("VOL CHEAP", str(cheap), COLORS["accent_green"]),
        ("VOL RICH", str(rich), COLORS["accent_red"]),
        ("SKEW EXTREME", str(skew), COLORS["accent_orange"]),
        ("RV SIGNALS", str(rv_opp), COLORS["accent_purple"]),
        ("AVG %ILE", f"{avg_pct:.0f}", COLORS["accent_blue"]),
        ("AVG Z-SCORE", f"{avg_z:+.2f}", COLORS["accent_amber"]),
    ]

    boxes = []
    for label, value, color in stats:
        boxes.append(
            html.Div([
                html.Div(label, style={
                    **LABEL_STYLE,
                    "marginBottom": "4px",
                    "fontSize": "9px",
                }),
                html.Div(value, style={
                    "color": color,
                    "fontSize": "18px",
                    "fontWeight": "700",
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
            ], style={
                **STAT_BOX_STYLE,
                "borderTop": f"2px solid {color}",
                "boxShadow": f"0 -2px 12px {color}26",
                "flex": "1",
                "minWidth": "100px",
            })
        )
    return boxes


# ============================================================================
#  Layout
# ============================================================================

def layout():
    """Return the top-level Div for the Vol Scanner panel."""
    return html.Div([

        # -- Filters Row -------------------------------------------------------
        html.Div([
            html.Div("DESK-WIDE VOL SIGNAL SCANNER", style={
                "color": COLORS["text_primary"],
                "fontSize": "13px",
                "fontWeight": "700",
                "fontFamily": "'JetBrains Mono', monospace",
                "letterSpacing": "1.5px",
                "textTransform": "uppercase",
                "marginBottom": "16px",
                "paddingBottom": "12px",
                "borderBottom": f"1px solid {COLORS['border_subtle']}",
            }),

            html.Div([
                # Group filter
                html.Div([
                    html.Label("GROUP", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="vol-scanner-group",
                        options=_GROUP_OPTIONS,
                        value="ALL",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "130px", "marginRight": "14px"}),

                # Signal filter
                html.Div([
                    html.Label("SIGNAL", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="vol-scanner-signal",
                        options=_SIGNAL_OPTIONS,
                        value="ALL",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "150px", "marginRight": "14px"}),

                # Min |z-score| slider
                html.Div([
                    html.Label("MIN |Z-SCORE|", style=LABEL_STYLE),
                    dcc.Slider(
                        id="vol-scanner-zscore",
                        min=0, max=3, step=0.1, value=0,
                        marks={0: "0", 0.5: "0.5", 1: "1", 1.5: "1.5",
                               2: "2", 2.5: "2.5", 3: "3"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "2", "minWidth": "220px", "marginRight": "14px"}),

                # Lookback
                html.Div([
                    html.Label("LOOKBACK", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="vol-scanner-lookback",
                        options=_LOOKBACK_OPTIONS,
                        value=252,
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "110px", "marginRight": "14px"}),

                # Refresh button
                html.Div([
                    html.Label("\u00a0", style=LABEL_STYLE),  # spacing placeholder
                    html.Button(
                        "REFRESH",
                        id="vol-scanner-refresh",
                        n_clicks=0,
                        style={
                            "backgroundColor": COLORS["accent_blue"],
                            "color": COLORS["text_bright"],
                            "border": "none",
                            "borderRadius": "10px",
                            "padding": "10px 24px",
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "11px",
                            "fontWeight": "700",
                            "cursor": "pointer",
                            "letterSpacing": "1px",
                            "textTransform": "uppercase",
                            "boxShadow": "0 4px 14px rgba(59,130,246,0.3)",
                            "transition": "all 0.2s ease",
                            "width": "100%",
                        },
                    ),
                ], style={"flex": "0 0 auto", "minWidth": "110px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px",
                       "alignItems": "flex-end"}),
        ], style=CARD_STYLE),

        # -- Top Movers Row (new) -----------------------------------------------
        html.Div(id="vol-scanner-top-movers", style={
            "display": "flex", "gap": "10px", "marginBottom": "10px", "flexWrap": "wrap",
        }),

        # -- Summary stats (signal type counts) ---------------------------------
        html.Div(id="vol-scanner-stats", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap",
        }),

        # -- Loading wrapper ----------------------------------------------------
        dcc.Loading(
            id="vol-scanner-loading",
            type="dot",
            color=COLORS["accent_cyan"],
            children=[

                # -- Scanner DataTable ------------------------------------------
                html.Div([
                    dash_table.DataTable(
                        id="vol-scanner-table",
                        columns=_TABLE_COLUMNS,
                        data=[],
                        sort_action="native",
                        sort_mode="multi",
                        filter_action="native",
                        row_selectable="single",
                        selected_rows=[],
                        page_size=30,
                        style_table={
                            "overflowX": "auto",
                            "borderRadius": "12px",
                            "border": f"1px solid {COLORS['border']}",
                        },
                        style_header={
                            **TABLE_HEADER_STYLE,
                            "borderBottom": f"2px solid {COLORS['accent_cyan']}",
                            "textAlign": "center",
                        },
                        style_cell={
                            **TABLE_CELL_STYLE,
                            "textAlign": "center",
                            "minWidth": "75px",
                            "maxWidth": "130px",
                            "whiteSpace": "nowrap",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                        },
                        style_data={
                            "backgroundColor": COLORS["bg_card"],
                            "color": COLORS["text_primary"],
                            "border": f"1px solid {COLORS['border_subtle']}",
                        },
                        style_filter={
                            "backgroundColor": COLORS["bg_input"],
                            "color": COLORS["text_primary"],
                            "border": f"1px solid {COLORS['border']}",
                        },
                        style_data_conditional=_build_conditional_styles(),
                        style_as_list_view=False,
                        css=[
                            # Force dark dropdown inside filter row
                            {"selector": ".dash-filter input", "rule":
                             f"background-color: {COLORS['bg_input']}; "
                             f"color: {COLORS['text_primary']};"},
                            # Selected row
                            {"selector": "tr.row-selected", "rule":
                             f"background-color: {COLORS['bg_card_hover']} !important;"},
                        ],
                    ),
                ], style=CARD_STYLE),
            ],
        ),

        # -- Drill-Down Panel (hidden until row selected) -----------------------
        html.Div(
            id="vol-scanner-drilldown",
            children=[
                html.Div("DRILL-DOWN", id="vol-scanner-drilldown-title", style={
                    "color": COLORS["text_primary"],
                    "fontSize": "12px",
                    "fontWeight": "700",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "letterSpacing": "1.5px",
                    "textTransform": "uppercase",
                    "marginBottom": "14px",
                    "paddingBottom": "10px",
                    "borderBottom": f"1px solid {COLORS['border_subtle']}",
                }),
                html.Div([
                    html.Div(dcc.Graph(id="vol-scanner-spark-atm",
                                       config={"displayModeBar": False}),
                             style={"flex": "1", "minWidth": "220px"}),
                    html.Div(dcc.Graph(id="vol-scanner-spark-rr",
                                       config={"displayModeBar": False}),
                             style={"flex": "1", "minWidth": "220px"}),
                    html.Div(dcc.Graph(id="vol-scanner-spark-ivrv",
                                       config={"displayModeBar": False}),
                             style={"flex": "1", "minWidth": "220px"}),
                    html.Div(dcc.Graph(id="vol-scanner-spark-term",
                                       config={"displayModeBar": False}),
                             style={"flex": "1", "minWidth": "220px"}),
                ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            ],
            style={**CARD_STYLE, "display": "none"},
        ),

        # Hidden store for full scanner rows (before client-side filtering)
        dcc.Store(id="vol-scanner-store", data=[]),

    ], style={"padding": "0"})


# ============================================================================
#  Callbacks
# ============================================================================

def register_callbacks(app):
    """Register all Dash callbacks for the Vol Scanner panel."""

    # ------------------------------------------------------------------
    # 1. Rebuild scanner data on group / lookback / refresh
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("vol-scanner-store", "data"),
            Output("vol-scanner-table", "data"),
            Output("vol-scanner-top-movers", "children"),
            Output("vol-scanner-stats", "children"),
            Output("vol-scanner-table", "selected_rows"),
        ],
        [
            Input("vol-scanner-group", "value"),
            Input("vol-scanner-signal", "value"),
            Input("vol-scanner-zscore", "value"),
            Input("vol-scanner-lookback", "value"),
            Input("vol-scanner-refresh", "n_clicks"),
        ],
    )
    def update_scanner(group_val, signal_val, min_zscore, lookback, _n):
        """
        Main scanner callback.

        1. Fetch pairs for the selected group.
        2. Build full scanner data for each pair.
        3. Apply signal filter and min z-score filter.
        4. Return data for table, store, top movers, summary stats.
        """
        pairs = _pairs_for_group(group_val)
        all_rows = _build_scanner_data(pairs, lookback)

        # ---- Apply filters ----
        filtered = []
        for row in all_rows:
            # Z-score filter
            if abs(row.get("z_atm3m", 0)) < min_zscore:
                continue
            # Signal filter
            if not _signal_matches_filter(row.get("signal", ""), signal_val):
                continue
            filtered.append(row)

        top_movers = _build_top_movers(filtered)
        stats_children = _build_summary_stats(filtered)
        return all_rows, filtered, top_movers, stats_children, []

    # ------------------------------------------------------------------
    # 2. Drill-down panel on row selection
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("vol-scanner-drilldown", "style"),
            Output("vol-scanner-drilldown-title", "children"),
            Output("vol-scanner-spark-atm", "figure"),
            Output("vol-scanner-spark-rr", "figure"),
            Output("vol-scanner-spark-ivrv", "figure"),
            Output("vol-scanner-spark-term", "figure"),
        ],
        [
            Input("vol-scanner-table", "selected_rows"),
        ],
        [
            State("vol-scanner-table", "data"),
        ],
    )
    def update_drilldown(selected_rows, table_data):
        """
        When the user clicks a row, show the drill-down panel with
        four sparkline charts for the selected pair.
        """
        hidden_style = {**CARD_STYLE, "display": "none"}
        visible_style = {**CARD_STYLE, "display": "block"}

        empty = (
            hidden_style,
            "DRILL-DOWN",
            _empty_spark("ATM 3M"),
            _empty_spark("25D RR 3M"),
            _empty_spark("IV-RV Spread"),
            _empty_spark("Term Structure"),
        )

        if not selected_rows or not table_data:
            return empty

        idx = selected_rows[0]
        if idx >= len(table_data):
            return empty

        row = table_data[idx]
        pair = row.get("pair", "")
        if not pair:
            return empty

        group = row.get("group", "")
        signal = row.get("signal", "\u2014")
        title_text = f"DRILL-DOWN  \u2022  {pair}  ({group})  \u2022  {signal}"

        fig_atm = _spark_atm_history(pair, lookback=120)
        fig_rr = _spark_rr_history(pair, lookback=120)
        fig_ivrv = _spark_ivrv_history(pair, lookback=120)
        fig_term = _spark_term_structure(pair)

        return visible_style, title_text, fig_atm, fig_rr, fig_ivrv, fig_term
