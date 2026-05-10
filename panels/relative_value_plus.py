"""
Relative Value Plus — RISK & ANALYTICS workspace
=================================================
Replaces: Relative Value + Correlation Monitor + Macro Regime + Vol Carry (4 → 1).

Four sub-tabs:
  [CROSS-PAIR]   Vol spread TS, term structure, skew scatter, z-score matrix, IV-RV, correlation
  [CORRELATION]  15×15 matrix, regime badges, breakdown table, rolling chart
  [MACRO]        Rate differentials, DXY proxy, risk sentiment, CB divergence, carry scatter
  [CARRY]        Carry table, term structure overlay, forward vol, calendar spread
"""

import json
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from dash import html, dcc, Input, Output, State, callback_context, ALL, MATCH, dash_table, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, TAB_STYLE, TAB_SELECTED_STYLE,
    TABLE_HEADER_STYLE, TABLE_CELL_STYLE, clickable_stat, make_stat_style,
    GAP, SECTION_GAP, CHART_SM, CHART_MD, CHART_LG, CSV_BTN_STYLE,
    no_data_fig, chart_layout,
)
from core.config import TENORS_HEATMAP_COMPACT, TENORS_LIQUID
from core.csv_export import export_csv
from core.fx_conventions import (
    FX_PAIR_REGISTRY, tenor_to_days, tenor_to_years,
    safe_float as _sf, ALL_PAIRS, G10_PAIRS, EM_PAIRS, MONITOR_PAIRS,
)
from core.fx_analytics import extract_surface_atm as _extract_atm, extract_surface_rr25 as _extract_rr

# ── Constants ────────────────────────────────────────────────────────────────

_P = "rvp"  # prefix
_MONO = "'JetBrains Mono', monospace"

HEATMAP_TENORS = TENORS_HEATMAP_COMPACT
TERM_TENORS = TENORS_LIQUID

LOOKBACK_OPTIONS = [
    {"label": "60D",  "value": 60},
    {"label": "120D", "value": 120},
    {"label": "252D", "value": 252},
    {"label": "504D", "value": 504},
]

_CB_BANKS = {
    "FED":  {"rate": None, "direction": "—"},
    "ECB":  {"rate": None, "direction": "—"},
    "BOE":  {"rate": None, "direction": "—"},
    "BOJ":  {"rate": None, "direction": "—"},
    "SNB":  {"rate": None, "direction": "—"},
    "RBA":  {"rate": None, "direction": "—"},
    "RBNZ": {"rate": None, "direction": "—"},
    "BOC":  {"rate": None, "direction": "—"},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_fig(msg="NO DATA", height=300):
    """Empty figure with message — wraps no_data_fig to accept msg first."""
    return no_data_fig(height=height, msg=msg)


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-PAIR TAB — Charts (from relative_value.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_vol_spread_ts(pair_a, pair_b, tenor, lookback):
    """Vol spread time series with z-score overlay."""
    try:
        from core.bloomberg_fx import get_fx_historical_vol
        from core.fx_analytics import vol_zscore, vol_percentile

        hist_a = get_fx_historical_vol(pair_a, tenor, "ATM", lookback)
        hist_b = get_fx_historical_vol(pair_b, tenor, "ATM", lookback)
        if hist_a is None or hist_b is None:
            return no_data_fig(msg="NO VOL HISTORY")

        a = np.array(hist_a, dtype=float)
        b = np.array(hist_b, dtype=float)
        min_len = min(len(a), len(b))
        a, b = a[-min_len:], b[-min_len:]
        spread = a - b
        x = list(range(min_len))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=x, y=a, mode="lines",
                                 line=dict(color="#ff8800", width=2.5), name=pair_a,
                                 hovertemplate="Day %{x}<br>" + pair_a + ": %{y:.2f}v<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=b, mode="lines",
                                 line=dict(color="#ff3333", width=1.5), name=pair_b,
                                 hovertemplate="Day %{x}<br>" + pair_b + ": %{y:.2f}v<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=a, mode="lines", line=dict(width=0), showlegend=False),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=b, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(255,136,0,0.12)", showlegend=False),
                      secondary_y=False)

        # Z-score line
        if min_len > 20:
            z_line = (spread - np.mean(spread)) / max(np.std(spread), 1e-6)
            fig.add_trace(go.Scatter(x=x, y=z_line, mode="lines",
                                     line=dict(color="#ffffff", width=1, dash="dash"), name="Z-Score",
                                     hovertemplate="Day %{x}<br>Z-Score: %{y:+.2f}\u03c3<extra></extra>"),
                          secondary_y=True)
            fig.add_hline(y=2, line=dict(color="#ff3333", width=0.5, dash="dot"), secondary_y=True)
            fig.add_hline(y=-2, line=dict(color="#1565c0", width=0.5, dash="dot"), secondary_y=True)

        fig.update_layout(**chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=50, t=30, b=20),
                          title=dict(text=f"VOL SPREAD: {pair_a} vs {pair_b} ({tenor})",
                                     font=dict(size=10, color="#9a9ab0")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        fig.update_yaxes(title_text="Vol (%)", secondary_y=False)
        fig.update_yaxes(title_text="Z-Score (\u03c3)", secondary_y=True)
        fig.update_xaxes(title_text="Trading Days")
        return fig
    except Exception:
        logger.debug("Vol spread chart failed", exc_info=True)
        return _empty_fig("VOL SPREAD")


def _build_zscore_matrix(lookback):
    """Z-score heatmap across all pairs × key tenors."""
    try:
        from core.fx_analytics import vol_zscore
        z_matrix = []
        for pair in ALL_PAIRS:
            row = []
            for tenor in HEATMAP_TENORS:
                try:
                    info = vol_zscore(pair, tenor, "ATM", lookback)
                    z = _sf(info.get("zscore", 0) if isinstance(info, dict) else info)
                except Exception:
                    z = 0
                row.append(np.clip(z, -3, 3))
            z_matrix.append(row)

        z = np.array(z_matrix)
        text = [[f"{z[i][j]:+.1f}" for j in range(len(HEATMAP_TENORS))]
                for i in range(len(ALL_PAIRS))]

        fig = go.Figure(go.Heatmap(
            z=z, x=HEATMAP_TENORS, y=ALL_PAIRS, text=text,
            texttemplate="%{text}", textfont=dict(size=9, color="#d0d0d0"),
            colorscale=[[0, "#1565c0"], [0.25, "#0a1628"],
                        [0.50, "#2a2a40"],
                        [0.75, "#2a1200"], [1.0, "#ff3333"]],
            zmin=-3, zmax=3,
            hovertemplate="<b>%{y}</b> %{x}<br>Z: %{z:+.2f}<extra></extra>",
            colorbar=dict(
                title=dict(text="Z", font=dict(size=9, color="#9a9ab0")),
                tickfont=dict(size=8, color="#9a9ab0"),
                len=0.6, thickness=10, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
            ),
            xgap=2, ygap=2,
        ))
        fig.update_layout(**chart_layout(height=CHART_LG,
                          margin=dict(l=65, r=60, t=30, b=20),
                          title=dict(text=f"ATM Z-SCORE MATRIX ({lookback}D)",
                                     font=dict(size=10, color="#9a9ab0")),
                          yaxis=dict(autorange="reversed", tickfont=dict(size=8, color="#9a9ab0"),
                                     showgrid=False),
                          xaxis=dict(tickfont=dict(size=9, color="#9a9ab0"),
                                     showgrid=False,
                                     title=dict(text="Tenor", font=dict(size=9, color="#9a9ab0")))))
        return fig
    except Exception:
        logger.debug("Z-score matrix chart failed", exc_info=True)
        return _empty_fig("Z-SCORE MATRIX")


def _build_ivrv_panel(pair, tenor, lookback):
    """IV-RV spread with filled area."""
    try:
        from core.fx_analytics import iv_rv_spread
        df = iv_rv_spread(pair, tenor, lookback=lookback)
        if df is None or (hasattr(df, '__len__') and len(df) < 10):
            return _empty_fig(f"{pair} IV-RV")

        spread = df["spread"].values if "spread" in df.columns else (df["iv"].values - df["rv"].values)
        x = list(range(len(spread)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=spread, mode="lines",
                                 line=dict(color="#9a9ab0", width=1), name="IV-RV",
                                 hovertemplate="Day %{x}<br>IV-RV: %{y:+.2f}v<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=[max(0, s) for s in spread], mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[0]*len(x), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(255,51,51,0.15)", showlegend=False))
        fig.add_hline(y=0, line=dict(color="#9a9ab0", width=0.5, dash="dash"))

        fig.update_layout(**chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"{pair} IV-RV SPREAD ({tenor})",
                                     font=dict(size=10, color="#9a9ab0")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return fig
    except Exception:
        logger.debug("IV-RV chart failed", exc_info=True)
        return _empty_fig(f"{pair} IV-RV")


def _build_vol_beta_heatmap():
    """Vol beta heatmap: how each pair's vol co-moves with others."""
    try:
        from core.fx_analytics import vol_beta
        pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP"]
        n = len(pairs)
        z = np.full((n, n), np.nan)
        text = [["" for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    z[i][j] = 1.0
                    text[i][j] = "1.00"
                else:
                    vb = vol_beta(pairs[i], pairs[j], "3M", 120)
                    if vb:
                        z[i][j] = vb["beta"]
                        text[i][j] = f"{vb['beta']:.2f}"

        fig = go.Figure(data=go.Heatmap(
            x=pairs, y=pairs, z=z,
            colorscale=[[0, "#1565c0"], [0.5, "#2a2a40"], [1.0, "#ff8800"]],
            text=text, texttemplate="%{text}",
            textfont=dict(size=10, color="#d0d0d0"),
            hovertemplate="Y: %{y}<br>X: %{x}<br>Beta: %{z:.2f}<extra></extra>",
            colorbar=dict(
                title=dict(text="Vol Beta", font=dict(color="#9a9ab0", size=10)),
                tickfont=dict(color="#9a9ab0", size=9),
                len=0.8, thickness=12, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
            ),
            xgap=2, ygap=2,
        ))
        fig.update_layout(
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            font=dict(family=_MONO, color="#e0e0e0", size=11),
            title=dict(text="Vol Beta Matrix (3M ATM, 120d)", font=dict(color="#ffffff", size=13)),
            xaxis=dict(title="", type="category", tickfont=dict(size=9, color="#9a9ab0")),
            yaxis=dict(title="", type="category", autorange="reversed", tickfont=dict(size=9, color="#e0e0e0")),
            margin=dict(l=70, r=20, t=40, b=40),
            height=380,
            hoverlabel=dict(bgcolor="#0a0a14", bordercolor="#2d2d50",
                            font=dict(color="#e0e0e0", family=_MONO, size=11)),
        )
        return fig
    except Exception:
        logger.debug("Vol beta matrix chart failed", exc_info=True)
        return _empty_fig("VOL BETA MATRIX ERROR")


def _build_rr_bf_spreads(pair_a, pair_b, tenor="3M"):
    """Cross-pair RR & BF spread time series with z-score bands."""
    try:
        from core.fx_analytics import cross_pair_rr_spread, cross_pair_bf_spread
        # Try 120-day lookback first (more pairs have this), fallback to 60
        rr = cross_pair_rr_spread(pair_a, pair_b, tenor, 120)
        bf = cross_pair_bf_spread(pair_a, pair_b, tenor, 120)

        if rr is None and bf is None:
            # Try shorter lookback
            rr = cross_pair_rr_spread(pair_a, pair_b, tenor, 60)
            bf = cross_pair_bf_spread(pair_a, pair_b, tenor, 60)

        if rr is None and bf is None:
            return _empty_fig(f"No RR/BF history for {pair_a[:3]}/{pair_a[3:]} vs {pair_b[:3]}/{pair_b[3:]}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=[f"25D RR Spread ({pair_a} - {pair_b})",
                                            f"25D BF Spread ({pair_a} - {pair_b})"])

        for row_idx, (data, name, color) in enumerate([
            (rr, "RR Spread", "#ff8800"),
            (bf, "BF Spread", "#00b4d8"),
        ], 1):
            if data is None:
                continue
            ts = data.get("spread_ts", [])
            mean = data.get("mean", 0)
            std = data.get("std", 1)
            current = data.get("current", 0)
            z_val = data.get("zscore", 0)
            days = list(range(len(ts)))

            # +/- 2 sigma band
            fig.add_trace(go.Scatter(
                x=days + days[::-1],
                y=[mean + 2*std]*len(days) + [mean - 2*std]*len(days),
                fill="toself", fillcolor="rgba(255,136,0,0.12)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ), row=row_idx, col=1)

            # +/- 1 sigma band
            fig.add_trace(go.Scatter(
                x=days + days[::-1],
                y=[mean + std]*len(days) + [mean - std]*len(days),
                fill="toself", fillcolor="rgba(255,136,0,0.10)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ), row=row_idx, col=1)

            # Mean line
            fig.add_hline(y=mean, line=dict(color="#9a9ab0", width=1, dash="dash"), row=row_idx, col=1)

            # Spread time series
            fig.add_trace(go.Scatter(
                x=days, y=ts, mode="lines", name=name,
                line=dict(color=color, width=2),
                hovertemplate=f"Day %{{x}}: %{{y:.2f}}<extra>{name}</extra>",
            ), row=row_idx, col=1)

            # Current marker
            if ts:
                fig.add_trace(go.Scatter(
                    x=[days[-1]], y=[current], mode="markers",
                    marker=dict(color=color, size=8, symbol="diamond"),
                    name=f"Current: {current:.2f} (z={z_val:.1f})", showlegend=True,
                ), row=row_idx, col=1)

        fig.update_layout(
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            font=dict(family=_MONO, color="#e0e0e0", size=11),
            margin=dict(l=50, r=20, t=40, b=30),
            height=450,
            xaxis2=dict(title="Trading Days"),
            yaxis=dict(title="RR Spread", showgrid=True, gridcolor="#1a1a30"),
            yaxis2=dict(title="BF Spread", showgrid=True, gridcolor="#1a1a30"),
            legend=dict(font=dict(color="#9a9ab0", size=9), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor="#0a0a14", bordercolor="#2d2d50",
                            font=dict(color="#e0e0e0", family=_MONO, size=11)),
        )
        return fig
    except Exception:
        logger.debug("RR/BF spread chart failed", exc_info=True)
        return _empty_fig(f"RR/BF SPREAD: {pair_a} vs {pair_b}")


def _build_skew_scatter(pair_a, pair_b, tenor, lookback):
    """Normalised skew scatter (RR/ATM, in % of ATM) with regression."""
    try:
        from core.bloomberg_fx import get_fx_historical_vol
        from scipy.stats import linregress

        rr_a = get_fx_historical_vol(pair_a, tenor, "25D_RR", lookback)
        rr_b = get_fx_historical_vol(pair_b, tenor, "25D_RR", lookback)
        atm_a = get_fx_historical_vol(pair_a, tenor, "ATM", lookback)
        atm_b = get_fx_historical_vol(pair_b, tenor, "ATM", lookback)
        if rr_a is None or rr_b is None or atm_a is None or atm_b is None:
            return _empty_fig("NO SKEW DATA")

        rra = np.array(rr_a, dtype=float)
        rrb = np.array(rr_b, dtype=float)
        ata = np.array(atm_a, dtype=float)
        atb = np.array(atm_b, dtype=float)
        min_len = min(len(rra), len(rrb), len(ata), len(atb))
        rra, rrb = rra[-min_len:], rrb[-min_len:]
        ata, atb = ata[-min_len:], atb[-min_len:]

        # Normalise to skew = RR / ATM * 100 (% of ATM)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(ata > 0, rra / ata * 100, np.nan)
            b = np.where(atb > 0, rrb / atb * 100, np.nan)
        # Drop NaNs paired
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if len(a) < 2:
            return _empty_fig("NO SKEW DATA")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=a[:-1], y=b[:-1], mode="markers",
                                 marker=dict(color="#9a9ab0", size=3, opacity=0.5),
                                 name="History",
                                 hovertemplate=pair_a + " skew: %{x:.1f}%<br>" + pair_b + " skew: %{y:.1f}%<extra>History</extra>"))
        fig.add_trace(go.Scatter(x=[a[-1]], y=[b[-1]], mode="markers",
                                 marker=dict(color="#ff3333", size=10, symbol="diamond"),
                                 name="Current",
                                 hovertemplate=pair_a + " skew: %{x:.1f}%<br>" + pair_b + " skew: %{y:.1f}%<extra>Current</extra>"))
        if len(a) > 5:
            slope, intercept, r, _, _ = linregress(a, b)
            x_line = np.linspace(a.min(), a.max(), 50)
            fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines",
                                     line=dict(color="#ffffff", width=1, dash="dash"),
                                     name=f"R²={r**2:.2f}" if np.isfinite(r) else "R²=—"))

        fig.update_layout(**chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=30),
                          title=dict(text=f"NORMALISED SKEW SCATTER: {pair_a} vs {pair_b} ({tenor})",
                                     font=dict(size=10, color="#9a9ab0")),
                          xaxis=dict(title=dict(text=f"{pair_a} skew (% of ATM)", font=dict(size=9))),
                          yaxis=dict(title=dict(text=f"{pair_b} skew (% of ATM)", font=dict(size=9))),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8))))
        return fig
    except Exception:
        logger.debug("Skew scatter chart failed", exc_info=True)
        return _empty_fig("SKEW SCATTER")


def _build_signal_table(lookback):
    """RV composite signal table."""
    try:
        from core.fx_analytics import vol_zscore, iv_rv_spread, rv_signal_composite
        rows = []
        for pair in ALL_PAIRS:
            try:
                try:
                    atm_info = vol_zscore(pair, "3M", "ATM", lookback)
                    z_atm = _sf(atm_info.get("zscore", 0) if isinstance(atm_info, dict) else 0)
                except Exception:
                    z_atm = 0
                try:
                    rr_info = vol_zscore(pair, "3M", "25D_RR", lookback)
                    z_rr = _sf(rr_info.get("zscore", 0) if isinstance(rr_info, dict) else 0)
                except Exception:
                    z_rr = 0
                try:
                    comp_info = rv_signal_composite(pair, lookback)
                    composite = _sf(comp_info.get("composite_score", 0)
                                    if isinstance(comp_info, dict) else 0)
                except Exception:
                    composite = z_atm * 10 + z_rr * 5

                direction = ("BUY VOL" if composite < -20 else "SELL VOL" if composite > 20
                             else "BUY SKEW" if z_rr < -1.5 else "SELL SKEW" if z_rr > 1.5
                             else "NEUTRAL")
                confidence = "HIGH" if abs(composite) > 30 else "MEDIUM" if abs(composite) > 15 else "LOW"

                rows.append({
                    "pair": pair, "atm_z": round(z_atm, 2), "rr_z": round(z_rr, 2),
                    "composite": round(composite, 1), "direction": direction,
                    "confidence": confidence,
                })
            except Exception:
                rows.append({"pair": pair, "atm_z": 0, "rr_z": 0, "composite": 0,
                             "direction": "NEUTRAL", "confidence": "LOW"})
        return rows
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# CORRELATION TAB — Matrix + regime + rolling (from correlation_monitor.py)
# ═══════════════════════════════════════════════════════════════════════════

def _safe_corr_matrix(window=60):
    """Correlation matrix with fallback. Always returns numpy ndarray."""
    try:
        from core.fx_analytics import spot_correlation_matrix
        matrix = spot_correlation_matrix(MONITOR_PAIRS, window)
        if matrix is not None:
            if hasattr(matrix, 'values'):
                return matrix.values  # DataFrame → ndarray
            return np.array(matrix)
    except Exception:
        pass

    return None


def _build_corr_heatmap(window):
    """15×15 correlation heatmap."""
    corr = _safe_corr_matrix(window)
    if corr is None:
        return no_data_fig(msg="NO CORRELATION DATA")
    n = len(MONITOR_PAIRS)
    # Ensure corr matrix matches expected dimensions
    corr = np.array(corr, dtype=float)
    rows, cols = corr.shape if corr.ndim == 2 else (0, 0)
    if rows < n or cols < n:
        # Pad with NaN if matrix is smaller than expected
        padded = np.full((n, n), np.nan)
        padded[:min(rows, n), :min(cols, n)] = corr[:min(rows, n), :min(cols, n)]
        corr = padded
    else:
        corr = corr[:n, :n]
    labels = [f"{p[:3]}/{p[3:]}" for p in MONITOR_PAIRS]
    text = [[f"{corr[i][j]:.2f}" if np.isfinite(corr[i][j]) else ""
             for j in range(n)]
            for i in range(n)]

    # Convert NaN to None for Plotly compatibility
    z_data = [[None if not np.isfinite(corr[i][j]) else corr[i][j]
                for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=z_data, x=labels, y=labels, text=text,
        texttemplate="%{text}", textfont=dict(size=8, color="#d0d0d0"),
        colorscale=[[0, "#ff3333"], [0.35, "#1a0e0e"], [0.50, "#2a2a40"],
                    [0.65, "#0a1628"], [1.0, "#00cc66"]],
        zmin=-1, zmax=1,
        xgap=2, ygap=2,
        hovertemplate="<b>%{x} vs %{y}</b><br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="ρ", font=dict(size=9, color="#9a9ab0")),
            tickfont=dict(size=8, color="#9a9ab0"),
            len=0.6, thickness=10, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
    ))
    fig.update_layout(**chart_layout(height=CHART_LG,
                      margin=dict(l=60, r=50, t=30, b=50),
                      title=dict(text=f"SPOT CORRELATION ({window}D)",
                                 font=dict(size=10, color="#9a9ab0")),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
                      xaxis=dict(tickfont=dict(size=8), tickangle=45)))
    return fig


def _build_regime_badges():
    """Vol regime badges for 15 pairs."""
    try:
        from core.fx_analytics import vol_regime_detect
    except Exception:
        vol_regime_detect = None

    badges = []
    for pair in MONITOR_PAIRS:
        try:
            if vol_regime_detect:
                info = vol_regime_detect(pair)
                regime = info.get("regime", "NORMAL") if isinstance(info, dict) else "NORMAL"
                color = info.get("color", COLORS["text_primary"]) if isinstance(info, dict) else COLORS["text_primary"]
            else:
                regime = "NORMAL"
                color = COLORS["text_primary"]
        except Exception:
            regime = "NORMAL"
            color = COLORS["text_primary"]

        regime_colors = {"LOW": COLORS["accent_green"], "NORMAL": COLORS["text_primary"], "ELEVATED": COLORS["accent_orange"],
                         "HIGH": COLORS["accent_red"], "CRISIS": COLORS["accent_red"]}
        color = regime_colors.get(regime, color)

        badges.append(html.Div([
            html.Div(f"{pair[:3]}/{pair[3:]}", style={
                "fontSize": "9px", "fontWeight": "700", "color": COLORS["text_primary"], "fontFamily": _MONO,
            }),
            html.Div(regime, style={
                "fontSize": "8px", "fontWeight": "600", "color": color, "fontFamily": _MONO,
            }),
        ], style={
            "border": f"1px solid {color}", "borderRadius": "0px",
            "padding": "4px 6px", "textAlign": "center", "minWidth": "70px",
            "backgroundColor": COLORS["bg_primary"],
        }))
    return badges


def _build_breakdown_table(window_short=20, window_long=120):
    """Pairs with divergent 20d vs 120d correlation."""
    corr_s = _safe_corr_matrix(window_short)
    corr_l = _safe_corr_matrix(window_long)
    if corr_s is None or corr_l is None:
        return html.Div("NO CORRELATION DATA",
                        style={"color": COLORS["text_secondary"], "fontSize": "10px", "padding": "8px"})
    n = len(MONITOR_PAIRS)
    # Ensure matrices match expected dimensions
    corr_s = np.array(corr_s, dtype=float)
    corr_l = np.array(corr_l, dtype=float)
    for arr_name, arr in [("corr_s", corr_s), ("corr_l", corr_l)]:
        if arr.ndim != 2 or arr.shape[0] < n or arr.shape[1] < n:
            return html.Div("CORRELATION DATA DIMENSION MISMATCH",
                            style={"color": COLORS["text_secondary"], "fontSize": "10px", "padding": "8px"})
    corr_s, corr_l = corr_s[:n, :n], corr_l[:n, :n]
    divergences = []
    for i in range(n):
        for j in range(i + 1, n):
            s_val, l_val = corr_s[i][j], corr_l[i][j]
            if not (np.isfinite(s_val) and np.isfinite(l_val)):
                continue
            gap = s_val - l_val
            if abs(gap) > 0.3:
                signal = "DIVERGING" if abs(s_val) < abs(l_val) else "CONVERGING"
                divergences.append({
                    "pair_a": MONITOR_PAIRS[i], "pair_b": MONITOR_PAIRS[j],
                    "corr_20d": round(float(s_val), 2), "corr_120d": round(float(l_val), 2),
                    "gap": round(float(gap), 2), "signal": signal,
                })
    divergences.sort(key=lambda d: abs(d["gap"]), reverse=True)

    if not divergences:
        return html.Div("No significant divergences (|gap| > 0.3)",
                        style={"color": COLORS["text_secondary"], "fontSize": "10px", "padding": "8px"})

    header = html.Tr([html.Th(h, style={**TABLE_HEADER_STYLE, "fontSize": "8px"})
                       for h in ["PAIR A", "PAIR B", "20D", "120D", "GAP", "SIGNAL"]])
    body = []
    for d in divergences[:min(len(divergences), 10)]:
        gap_color = COLORS["accent_red"] if abs(d["gap"]) > 0.4 else COLORS["accent_orange"]
        sig_color = COLORS["accent_red"] if d["signal"] == "DIVERGING" else COLORS["accent_green"]
        body.append(html.Tr([
            html.Td(d["pair_a"], style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
            html.Td(d["pair_b"], style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
            html.Td(f"{d['corr_20d']:+.2f}", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
            html.Td(f"{d['corr_120d']:+.2f}", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
            html.Td(f"{d['gap']:+.2f}", style={**TABLE_CELL_STYLE, "color": gap_color, "fontWeight": "600"}),
            html.Td(d["signal"], style={**TABLE_CELL_STYLE, "color": sig_color, "fontWeight": "600"}),
        ]))
    return html.Table([html.Thead(header), html.Tbody(body)],
                      style={"width": "100%", "borderCollapse": "collapse", "fontFamily": _MONO, "fontSize": "10px"})


def _build_rolling_chart(pair_a, pair_b):
    """Rolling correlation chart for 3 windows."""
    try:
        from core.bloomberg_fx import get_fx_historical_spot
        spot_a = get_fx_historical_spot(pair_a, 252)
        spot_b = get_fx_historical_spot(pair_b, 252)

        def _to_array(s):
            if s is None:
                return None
            if hasattr(s, 'columns') and 'close' in s.columns:
                return s['close'].values.astype(float)
            if hasattr(s, 'values'):
                return s.values.flatten().astype(float)
            return np.array(s, dtype=float)

        a = _to_array(spot_a)
        b = _to_array(spot_b)
        if a is None or b is None:
            return no_data_fig(msg="NO SPOT HISTORY")

        min_len = min(len(a), len(b))
        a, b = a[-min_len:], b[-min_len:]
        ret_a = np.diff(np.log(np.maximum(a, 1e-10)))
        ret_b = np.diff(np.log(np.maximum(b, 1e-10)))

        fig = go.Figure()
        windows = [20, 60, 120]
        colors = ["#ff8800", "#ffffff", "#00cc66"]
        for w, c in zip(windows, colors):
            if len(ret_a) < w:
                continue
            rolling = []
            for i in range(w, len(ret_a)):
                ra = ret_a[i - w:i]
                rb = ret_b[i - w:i]
                if np.std(ra) > 1e-10 and np.std(rb) > 1e-10:
                    rolling.append(np.corrcoef(ra, rb)[0, 1])
                else:
                    rolling.append(0)
            fig.add_trace(go.Scatter(x=list(range(len(rolling))), y=rolling,
                                     mode="lines", line=dict(color=c, width=1.5),
                                     name=f"{w}d",
                                     hovertemplate="Day %{x}<br>\u03c1: %{y:.3f}<extra>" + f"{w}d" + "</extra>"))

        fig.add_hline(y=0, line=dict(color="#9a9ab0", width=0.5, dash="dot"))
        fig.add_hline(y=0.7, line=dict(color="#00cc66", width=0.5, dash="dash"))
        fig.add_hline(y=-0.7, line=dict(color="#00cc66", width=0.5, dash="dash"))

        fig.update_layout(**chart_layout(height=CHART_SM,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"ROLLING CORRELATION: {pair_a} vs {pair_b}",
                                     font=dict(size=10, color="#9a9ab0")),
                          yaxis=dict(range=[-1.05, 1.05]),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8), orientation="h"),
                          xaxis_title="Trading Days", yaxis_title="Correlation (\u03c1)"))
        return fig
    except Exception:
        logger.debug("Rolling correlation chart failed", exc_info=True)
        return _empty_fig(f"ROLLING CORR: {pair_a} vs {pair_b}")


def _build_corr_cone(pair_a, pair_b):
    """Correlation cone: percentile bands across rolling windows."""
    try:
        from core.fx_analytics import correlation_cone
        df = correlation_cone(pair_a, pair_b)
        if df is None or df.empty:
            return no_data_fig(height=320, msg="NO CORRELATION CONE DATA")

        fig = go.Figure()
        windows = df["window"].tolist()

        # p10-p90 band (light fill)
        fig.add_trace(go.Scatter(
            x=windows + windows[::-1],
            y=df["p90"].tolist() + df["p10"].tolist()[::-1],
            fill="toself", fillcolor="rgba(255,136,0,0.14)",
            line=dict(width=0), showlegend=True, name="10th-90th %ile",
            hoverinfo="skip",
        ))

        # p25-p75 band (darker fill)
        fig.add_trace(go.Scatter(
            x=windows + windows[::-1],
            y=df["p75"].tolist() + df["p25"].tolist()[::-1],
            fill="toself", fillcolor="rgba(255,136,0,0.15)",
            line=dict(width=0), showlegend=True, name="25th-75th %ile",
            hoverinfo="skip",
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=windows, y=df["median"], mode="lines",
            name="Median", line=dict(color="#9a9ab0", width=1.5, dash="dash"),
        ))

        # Current values
        fig.add_trace(go.Scatter(
            x=windows, y=df["current"], mode="lines+markers+text",
            name="Current", line=dict(color="#ff8800", width=3),
            marker=dict(size=10, color="#ff8800", line=dict(width=2, color="#ffffff")),
            text=[f"{v:.2f}" for v in df["current"]],
            textposition="top center", textfont=dict(size=10, color="#ff8800"),
        ))

        # Apply theme
        fig.update_layout(
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            font=dict(family=_MONO, color="#e0e0e0", size=11),
            title=dict(text=f"Correlation Cone -- {pair_a} vs {pair_b}",
                       font=dict(color="#ffffff", size=13)),
            xaxis=dict(title="Rolling Window (days)", showgrid=True, gridcolor="#1a1a30",
                       tickfont=dict(size=9, color="#9a9ab0")),
            yaxis=dict(title="Correlation", showgrid=True, gridcolor="#1a1a30",
                       tickfont=dict(size=9, color="#9a9ab0"), range=[-1, 1]),
            margin=dict(l=50, r=20, t=40, b=40),
            height=320,
            legend=dict(font=dict(color="#9a9ab0", size=10), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor="#0a0a14", bordercolor="#2d2d50",
                            font=dict(color="#e0e0e0", family=_MONO, size=11)),
        )
        return fig
    except Exception:
        logger.debug("Correlation cone chart failed", exc_info=True)
        return _empty_fig(f"CORR CONE: {pair_a} vs {pair_b}")


def _build_corr_network(window=60):
    """Force-directed correlation network: clean, readable, only strong links."""
    try:
        from core.fx_analytics import spot_correlation_matrix, vol_percentile

        try:
            import networkx as nx
            _has_nx = True
        except ImportError:
            _has_nx = False

        top_pairs = MONITOR_PAIRS[:12]
        matrix = spot_correlation_matrix(top_pairs, window)
        if matrix is None or (hasattr(matrix, 'empty') and matrix.empty):
            return no_data_fig(msg="NO CORRELATION DATA")

        corr = matrix.values if hasattr(matrix, 'values') else np.array(matrix, dtype=float)
        pairs_used = list(matrix.columns)
        n = len(pairs_used)
        corr = corr[:n, :n]

        # Higher threshold = cleaner chart
        CORR_THRESHOLD = 0.5

        # Layout
        if _has_nx:
            G = nx.Graph()
            for p in pairs_used:
                G.add_node(p)
            for i in range(n):
                for j in range(i + 1, n):
                    c = corr[i][j]
                    if np.isfinite(c) and abs(c) > CORR_THRESHOLD:
                        G.add_edge(pairs_used[i], pairs_used[j], weight=abs(c))
            pos = nx.spring_layout(G, k=5.0 / np.sqrt(max(n, 1)), iterations=200, seed=42)
        else:
            pos = {}
            for i, p in enumerate(pairs_used):
                angle = 2 * np.pi * i / n
                pos[p] = np.array([np.cos(angle), np.sin(angle)])

        # Node vol percentiles
        node_pcts = []
        for p in pairs_used:
            try:
                info = vol_percentile(p, "3M", "ATM", 252)
                pct = _sf(info.get("percentile", 50) if isinstance(info, dict) else 50, 50)
            except Exception:
                pct = 50
            node_pcts.append(pct)

        fig = go.Figure()

        # Edges — batch all positive into one trace, all negative into another
        pos_x, pos_y = [], []
        neg_x, neg_y = [], []
        edge_hovers_pos, edge_hovers_neg = [], []
        for i in range(n):
            for j in range(i + 1, n):
                c = corr[i][j]
                if not np.isfinite(c) or abs(c) <= CORR_THRESHOLD:
                    continue
                x0, y0 = pos[pairs_used[i]]
                x1, y1 = pos[pairs_used[j]]
                target = pos_x if c > 0 else neg_x
                target_y = pos_y if c > 0 else neg_y
                target.extend([x0, x1, None])
                target_y.extend([y0, y1, None])

        if pos_x:
            fig.add_trace(go.Scatter(
                x=pos_x, y=pos_y, mode="lines",
                line=dict(color="rgba(0,204,102,0.35)", width=2),
                name="Positive \u03c1", hoverinfo="skip",
            ))
        if neg_x:
            fig.add_trace(go.Scatter(
                x=neg_x, y=neg_y, mode="lines",
                line=dict(color="rgba(255,51,51,0.35)", width=2),
                name="Negative \u03c1", hoverinfo="skip",
            ))

        # Nodes — large markers with text inside
        node_x = [pos[p][0] for p in pairs_used]
        node_y = [pos[p][1] for p in pairs_used]
        node_labels = [f"{p[:3]}/{p[3:]}" for p in pairs_used]

        # Color scale: green (low vol) → orange → red (high vol)
        node_colors = []
        for pct in node_pcts:
            if pct < 30:
                node_colors.append("#00cc66")
            elif pct < 60:
                node_colors.append("#ff8800")
            else:
                node_colors.append("#ff3333")

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=44, color=node_colors, opacity=0.9,
                        line=dict(width=2, color="#2d2d50")),
            text=node_labels,
            textposition="middle center",
            textfont=dict(size=9, color="#000000", family=_MONO, weight=700),
            hovertext=[
                f"<b>{pairs_used[i][:3]}/{pairs_used[i][3:]}</b><br>"
                f"Vol %ile: {node_pcts[i]:.0f}<br>"
                f"{'LOW' if node_pcts[i] < 30 else 'MID' if node_pcts[i] < 60 else 'HIGH'} VOL"
                for i in range(n)
            ],
            hoverinfo="text", showlegend=False,
        ))

        fig.update_layout(
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            font=dict(family=_MONO, color="#e0e0e0", size=11),
            title=dict(text=f"CORRELATION NETWORK ({window}D, |\u03c1| > {CORR_THRESHOLD})",
                       font=dict(color="#ffffff", size=13)),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=480,
            margin=dict(l=40, r=40, t=45, b=20),
            legend=dict(font=dict(color="#9a9ab0", size=9), bgcolor="rgba(0,0,0,0)",
                        x=0.01, y=0.99),
            hoverlabel=dict(bgcolor="#0a0a14", bordercolor="#2d2d50",
                            font=dict(color="#e0e0e0", family=_MONO, size=11)),
        )
        return fig
    except Exception:
        logger.debug("Correlation network chart failed", exc_info=True)
        return _empty_fig("CORRELATION NETWORK")


# ═══════════════════════════════════════════════════════════════════════════
# MACRO TAB (from macro_regime.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_rate_table():
    """Rate differentials table."""
    from core.bloomberg_fx import get_fx_rates

    header = html.Tr([html.Th(h, style={**TABLE_HEADER_STYLE, "fontSize": "8px"})
                       for h in ["PAIR", "DOM", "FOR", "DIFF", "CARRY"]])
    body = []
    for pair in ALL_PAIRS[:20]:
        try:
            r = get_fx_rates(pair) or {}
        except Exception:
            r = {}
        dom = _sf(r.get("r_dom", 0)) * 100  # convert decimal to percent
        fgn = _sf(r.get("r_for", 0)) * 100
        diff = dom - fgn
        carry = "RECEIVE" if diff > 0.5 else "PAY" if diff < -0.5 else "FLAT"
        carry_color = COLORS["accent_green"] if carry == "RECEIVE" else COLORS["accent_red"] if carry == "PAY" else COLORS["text_secondary"]
        body.append(html.Tr([
            html.Td(pair, style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
            html.Td(f"{dom:.2f}%", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
            html.Td(f"{fgn:.2f}%", style={**TABLE_CELL_STYLE, "textAlign": "right"}),
            html.Td(f"{diff:+.2f}", style={**TABLE_CELL_STYLE, "textAlign": "right",
                     "color": COLORS["accent_green"] if diff > 0 else COLORS["accent_red"]}),
            html.Td(carry, style={**TABLE_CELL_STYLE, "color": carry_color, "fontWeight": "600"}),
        ]))
    return html.Table([html.Thead(header), html.Tbody(body)],
                      style={"width": "100%", "borderCollapse": "collapse",
                             "fontFamily": _MONO, "fontSize": "10px"})


def _build_dxy_chart(lookback):
    """DXY proxy chart — trade-weighted USD index from spot data."""
    try:
        from core.bloomberg_fx import get_fx_historical_spot
        # Build a trade-weighted USD index from spot history
        weights = {"EURUSD": -0.576, "USDJPY": 0.136, "GBPUSD": -0.119,
                   "USDCHF": 0.036, "USDCAD": 0.091, "NZDUSD": -0.021, "AUDUSD": -0.021}

        all_series = {}
        min_len = lookback
        for pair, w in weights.items():
            hist = get_fx_historical_spot(pair, lookback + 10)
            if hist is not None and len(hist) > 20:
                if hasattr(hist, 'columns') and 'close' in hist.columns:
                    arr = hist['close'].values.astype(float)
                elif hasattr(hist, 'values'):
                    arr = hist.values.flatten().astype(float)
                else:
                    arr = np.array(hist, dtype=float)
                all_series[pair] = arr
                min_len = min(min_len, len(arr))

        if len(all_series) < 3:
            return no_data_fig(msg="INSUFFICIENT SPOT DATA")
        else:
            # Compute weighted log returns
            n = min_len
            index = np.ones(n) * 100
            # Accumulate weighted log returns from all pairs, then compound
            total_log_returns = np.zeros(n - 1)
            for pair, w in weights.items():
                if pair not in all_series:
                    continue
                arr = all_series[pair][-n:]
                total_log_returns += np.diff(np.log(arr)) * w
            # Compound into index
            for i in range(len(total_log_returns)):
                index[i + 1] = index[i] * np.exp(total_log_returns[i])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(index))), y=index, mode="lines",
                                 line=dict(color="#ff8800", width=2.5), name="DXY Proxy",
                                 hovertemplate="Day %{x}<br>DXY: %{y:.2f}<extra></extra>"))
        fig.update_layout(**chart_layout(height=CHART_SM,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"DXY PROXY ({lookback}D)",
                                     font=dict(size=10, color="#9a9ab0")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8)),
                          xaxis_title="Trading Days", yaxis_title="Index Level"))
        return fig
    except Exception:
        logger.debug("DXY proxy chart failed", exc_info=True)
        return _empty_fig("DXY PROXY")


def _build_risk_sentiment():
    """Risk sentiment composite."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface, get_fx_spots
        spots = get_fx_spots() or {}
        factors = []

        # G10 avg vol
        g10_vols = []
        for pair in G10_PAIRS[:8]:
            surf = get_fx_vol_surface(pair) or {}
            atm = _extract_atm(surf, "1M")
            if atm > 0:
                g10_vols.append(atm)
        g10_avg = np.mean(g10_vols) if g10_vols else 8.0
        vol_score = min(100, max(0, (g10_avg - 5) * 8))
        factors.append(("G10 AVG VOL", f"{g10_avg:.1f}v", vol_score))

        # Haven bid
        jpy_chg = _sf(spots.get("USDJPY", {}).get("change_pct", 0))
        chf_chg = _sf(spots.get("USDCHF", {}).get("change_pct", 0))
        haven_score = min(100, max(0, 50 + (jpy_chg + chf_chg) * 10))
        factors.append(("HAVEN BID", f"JPY{jpy_chg:+.1f}% CHF{chf_chg:+.1f}%", haven_score))

        # Composite
        composite = np.mean([f[2] for f in factors])
        label = "RISK-OFF" if composite > 65 else "RISK-ON" if composite < 35 else "NEUTRAL"
        color = COLORS["accent_red"] if composite > 65 else COLORS["accent_green"] if composite < 35 else COLORS["accent_orange"]

        items = [html.Div([
            html.Div(label, style={"fontSize": "16px", "fontWeight": "700",
                                   "color": color, "fontFamily": _MONO}),
            html.Div(f"COMPOSITE: {composite:.0f}" if np.isfinite(composite) else "COMPOSITE: —", style={
                "fontSize": "9px", "color": COLORS["text_secondary"], "fontFamily": _MONO}),
        ], style={**STAT_BOX_STYLE, "borderTop": f"2px solid {color}"})]

        for name, val, score in factors:
            items.append(html.Div([
                html.Div(val, style={"fontSize": "11px", "color": COLORS["text_primary"], "fontFamily": _MONO}),
                html.Div(name, style={"fontSize": "8px", "color": COLORS["text_secondary"], "fontFamily": _MONO}),
            ], style={**STAT_BOX_STYLE}))

        return items
    except Exception:
        logger.debug("Risk sentiment build failed", exc_info=True)
        return [html.Div("Error", style={"color": COLORS["text_secondary"]})]


def _build_cb_chart():
    """Central bank rates bar chart — tries to update from Bloomberg rates."""
    # Try to refresh rates from Bloomberg data
    try:
        from core.bloomberg_fx import get_fx_rates
        # Map each central bank to a currency pair and which leg to use
        bank_map = {
            "FED":  {"pair": "EURUSD", "key": "r_dom"},   # USD = domestic in EURUSD
            "ECB":  {"pair": "EURUSD", "key": "r_for"},   # EUR = foreign in EURUSD
            "BOE":  {"pair": "GBPUSD", "key": "r_for"},   # GBP = foreign in GBPUSD
            "BOJ":  {"pair": "USDJPY", "key": "r_for"},   # JPY = quote (foreign) in USDJPY
            "SNB":  {"pair": "USDCHF", "key": "r_for"},   # CHF = quote (foreign) in USDCHF
            "RBA":  {"pair": "AUDUSD", "key": "r_for"},   # AUD = foreign in AUDUSD
            "RBNZ": {"pair": "NZDUSD", "key": "r_for"},   # NZD = foreign in NZDUSD
            "BOC":  {"pair": "USDCAD", "key": "r_for"},   # CAD = quote (foreign) in USDCAD
        }
        for bank, info in bank_map.items():
            try:
                rates_data = get_fx_rates(info["pair"])
                if rates_data and isinstance(rates_data, dict):
                    actual = _sf(rates_data.get(info["key"], 0))
                    if actual > 0:
                        # Convert from decimal to percentage if needed
                        rate_pct = actual * 100 if actual < 1 else actual
                        # Use a local copy to avoid mutating module-level state
                        from copy import copy as _cb_copy
                        local_banks = {k: dict(v) for k, v in _CB_BANKS.items()}
                        local_banks[bank]["rate"] = round(rate_pct, 2)
                        _CB_BANKS[bank] = local_banks[bank]
            except Exception:
                pass  # Bloomberg rate fetch failed for this bank
    except Exception:
        pass  # Bloomberg rate fetch failed

    banks = list(_CB_BANKS.keys())
    rates = [_CB_BANKS[b]["rate"] or 0.0 for b in banks]
    dirs = [_CB_BANKS[b]["direction"] for b in banks]
    colors = ["#ff3333" if d == "HIKING" else "#00cc66" if d == "CUTTING" else "#ff8800" for d in dirs]

    fig = go.Figure(go.Bar(y=banks, x=rates, orientation="h",
                            marker_color=colors,
                            text=[f"{r:.2f}% ({d})" for r, d in zip(rates, dirs)],
                            textposition="outside", textfont=dict(size=8)))
    fig.update_layout(**chart_layout(height=CHART_SM,
                      margin=dict(l=40, r=80, t=30, b=10), showlegend=False,
                      title=dict(text="CENTRAL BANK POLICY RATES",
                                 font=dict(size=10, color="#9a9ab0")),
                      xaxis=dict(tickfont=dict(size=8),
                                 title=dict(text="Policy Rate (%)", font=dict(size=9, color="#9a9ab0"))),
                      yaxis=dict(tickfont=dict(size=9))))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CARRY TAB (from vol_carry.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_carry_data():
    """Carry table data."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
    except Exception:
        pass

    rows = []
    for pair in ALL_PAIRS:
        try:
            surf = get_fx_vol_surface(pair) or {}
            atm_1m = _extract_atm(surf, "1M")
            atm_3m = _extract_atm(surf, "3M")
            atm_1y = _extract_atm(surf, "1Y")
            term_spread = atm_1m - atm_1y if atm_1m > 0 and atm_1y > 0 else 0

            # Proper variance-based forward vol: σ_fwd = sqrt((σ₂²T₂ - σ₁²T₁) / (T₂ - T₁))
            t1 = 1.0 / 12.0  # 1M in years
            t3 = 3.0 / 12.0  # 3M in years
            v1 = atm_1m / 100.0 if atm_1m > 0 else 0
            v3 = atm_3m / 100.0 if atm_3m > 0 else 0
            fwd_var = (v3 ** 2 * t3 - v1 ** 2 * t1) / max(t3 - t1, 1e-6)
            fwd_3x3 = np.sqrt(max(fwd_var, 0)) * 100 if fwd_var > 0 else atm_3m

            # Carry (computed from term structure)
            carry_day = (atm_3m - fwd_3x3) / 90 if atm_3m > 0 else 0
            carry_vol = carry_day / max(atm_3m, 0.01) * 100 if atm_3m > 0 else 0

            signal = ("SELL VOL" if carry_vol > 2 else "BUY VOL" if carry_vol < -1 else "—")

            rows.append({
                "pair": pair, "atm_1m": round(atm_1m, 2), "atm_3m": round(atm_3m, 2),
                "atm_1y": round(atm_1y, 2), "term_spread": round(term_spread, 2),
                "fwd_3x3": round(fwd_3x3, 2), "carry_day": round(carry_day, 3),
                "carry_vol": round(carry_vol, 2), "signal": signal,
            })
        except Exception:
            rows.append({"pair": pair, "atm_1m": 0, "atm_3m": 0, "atm_1y": 0,
                         "term_spread": 0, "fwd_3x3": 0, "carry_day": 0, "carry_vol": 0, "signal": "—"})
    rows.sort(key=lambda r: r.get("carry_vol", 0), reverse=True)
    return rows


def _build_term_chart(pair, comp_pair=None):
    """Term structure overlay."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        surf = get_fx_vol_surface(pair) or {}
        vols = [_extract_atm(surf, t) for t in TERM_TENORS]
        labels = TERM_TENORS

        if all(v == 0 or v is None for v in vols):
            return no_data_fig(msg="NO VOL DATA")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=labels, y=vols, mode="lines+markers",
                                 line=dict(color="#ff8800", width=2), marker=dict(size=5),
                                 name=pair,
                                 hovertemplate="%{x}: %{y:.2f}v<extra>" + pair + "</extra>"))
        if comp_pair and comp_pair != pair:
            surf2 = get_fx_vol_surface(comp_pair) or {}
            vols2 = [_extract_atm(surf2, t) for t in TERM_TENORS]
            fig.add_trace(go.Scatter(x=labels, y=vols2, mode="lines+markers",
                                     line=dict(color="#ffffff", width=1.5, dash="dash"),
                                     marker=dict(size=4), name=comp_pair,
                                     hovertemplate="%{x}: %{y:.2f}v<extra>" + comp_pair + "</extra>"))

        fig.update_layout(**chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"TERM STRUCTURE: {pair}",
                                     font=dict(size=10, color="#9a9ab0")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8)),
                          xaxis_title="Tenor", yaxis_title="ATM Vol (%)"))
        return fig
    except Exception:
        logger.debug("Term structure chart failed", exc_info=True)
        return _empty_fig(f"{pair} TERM")


def _build_fwd_vol(pair):
    """Forward vol bar chart."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        surf = get_fx_vol_surface(pair) or {}
        tenors = ["1M", "2M", "3M", "6M", "1Y"]
        spot_vols = [_extract_atm(surf, t) for t in tenors]

        if all(v == 0 or v is None for v in spot_vols):
            return no_data_fig(msg="NO VOL DATA")

        # Calculate forward vols (variance-based)
        fwd_vols = [spot_vols[0]]
        for i in range(1, len(spot_vols)):
            if spot_vols[i] > 0 and spot_vols[i - 1] > 0:
                t_prev = tenor_to_years(tenors[i - 1])
                t_curr = tenor_to_years(tenors[i])
                var_prev = (spot_vols[i - 1] / 100.0) ** 2 * t_prev
                var_curr = (spot_vols[i] / 100.0) ** 2 * t_curr
                fwd_var = (var_curr - var_prev) / max(t_curr - t_prev, 1e-6)
                fwd = np.sqrt(max(fwd_var, 0)) * 100
                fwd_vols.append(fwd)
            else:
                fwd_vols.append(0)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=tenors, y=fwd_vols, marker_color="#ff8800",
                              text=[f"{v:.1f}" for v in fwd_vols],
                              textposition="outside", textfont=dict(size=8),
                              name="Forward Vol",
                              hovertemplate="%{x}: %{y:.2f}v<extra>Forward</extra>"))
        fig.add_trace(go.Scatter(x=tenors, y=spot_vols, mode="lines+markers",
                                  line=dict(color="#ffffff", width=1.5, dash="dot"),
                                  marker=dict(size=4), name="Spot Vol",
                                  hovertemplate="%{x}: %{y:.2f}v<extra>Spot</extra>"))
        fig.update_layout(**chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=20),
                          title=dict(text=f"FORWARD VOL: {pair}",
                                     font=dict(size=10, color="#9a9ab0")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8)),
                          xaxis_title="Tenor", yaxis_title="Vol (%)"))
        return fig
    except Exception:
        logger.debug("Forward vol chart failed", exc_info=True)
        return _empty_fig(f"{pair} FWD VOL")


def _build_calendar_spread(pair):
    """Calendar spread: difference between near (1M) and far (3M) tenor ATM vols over time."""
    try:
        from core.bloomberg_fx import get_fx_historical_vol
        near = get_fx_historical_vol(pair, "1M", "atm", 252)
        far = get_fx_historical_vol(pair, "3M", "atm", 252)

        if near is None or far is None or len(near) < 10 or len(far) < 10:
            raise ValueError("Insufficient vol history")

        # Align lengths
        n = min(len(near), len(far))
        near_vals = near.values[-n:].astype(float) if hasattr(near, 'values') else np.array(near[-n:], dtype=float)
        far_vals = far.values[-n:].astype(float) if hasattr(far, 'values') else np.array(far[-n:], dtype=float)
        spread = near_vals - far_vals

        fig = go.Figure()
        x = list(range(n))
        # Shade positive (backwardation) vs negative (contango)
        pos_y = [s if s >= 0 else 0 for s in spread]
        neg_y = [s if s < 0 else 0 for s in spread]
        fig.add_trace(go.Bar(x=x, y=pos_y, marker_color="rgba(255,51,51,0.5)",
                              name="Backwardation", showlegend=True,
                              hovertemplate="Day %{x}<br>Backwardation: %{y:.2f}v<extra></extra>"))
        fig.add_trace(go.Bar(x=x, y=neg_y, marker_color="rgba(21,101,192,0.5)",
                              name="Contango", showlegend=True,
                              hovertemplate="Day %{x}<br>Contango: %{y:.2f}v<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=spread, mode="lines",
                                  line=dict(color="#ff8800", width=2.5), name="1M-3M Spread",
                                  hovertemplate="Day %{x}<br>Spread: %{y:+.2f}v<extra></extra>"))
        fig.add_hline(y=0, line_dash="dot", line_color="#444444", line_width=0.8)
        fig.update_layout(**chart_layout(height=CHART_MD,
                          margin=dict(l=50, r=20, t=30, b=20),
                          barmode="overlay",
                          title=dict(text=f"CALENDAR SPREAD (1M-3M): {pair}",
                                     font=dict(size=10, color="#9a9ab0")),
                          legend=dict(x=0.02, y=0.98, font=dict(size=8)),
                          xaxis_title="Trading Days", yaxis_title="Spread (vol pts)"))
        return fig
    except Exception:
        logger.debug("Calendar spread chart failed", exc_info=True)
        return _empty_fig(f"{pair} CALENDAR SPREAD")


def _build_carry_vol_bubble():
    """Carry vs Vol percentile bubble chart with quadrant analysis."""
    try:
        from core.fx_analytics import carry_per_vol, vol_percentile, carry_momentum

        top_pairs = MONITOR_PAIRS[:15]
        carry_df = carry_per_vol(top_pairs)

        if carry_df is None or carry_df.empty:
            return no_data_fig(msg="NO CARRY DATA")

        # Build data for each pair
        x_vals, y_vals, sizes, colors, texts, hovers = [], [], [], [], [], []
        for _, row in carry_df.iterrows():
            pair = row["pair"]
            carry_bps = _sf(row.get("carry_bps", 0))
            sharpe = _sf(row.get("sharpe_proxy", 0))

            # Vol percentile (Y axis)
            try:
                vp_info = vol_percentile(pair, "3M", "ATM", 252)
                pct = _sf(vp_info.get("percentile", 50) if isinstance(vp_info, dict) else 50, 50)
            except Exception:
                pct = 50

            # Momentum → color
            try:
                mom_info = carry_momentum(pair)
                momentum = mom_info.get("momentum", "STABLE") if isinstance(mom_info, dict) else "STABLE"
            except Exception:
                momentum = "STABLE"

            if momentum == "IMPROVING":
                color = "#00cc66"
            elif momentum == "DETERIORATING":
                color = "#ff3333"
            else:
                color = "#9a9ab0"

            bubble_size = max(10, min(28, abs(sharpe) * 12 + 8))

            x_vals.append(carry_bps)
            y_vals.append(pct)
            sizes.append(bubble_size)
            colors.append(color)
            texts.append(f"{pair[:3]}/{pair[3:]}")
            hovers.append(
                f"<b>{pair[:3]}/{pair[3:]}</b><br>"
                f"Carry: {carry_bps:.0f} bps<br>"
                f"Vol Pctile: {pct:.0f}%<br>"
                f"Sharpe Proxy: {sharpe:.2f}<br>"
                f"Momentum: {momentum}"
            )

        if not x_vals:
            return no_data_fig(msg="NO CARRY BUBBLE DATA")

        fig = go.Figure()

        # ── Clip x-axis to exclude extreme outliers for readability ──
        # Use interquartile range to set sensible bounds
        x_arr = np.array(x_vals)
        q1, q3 = np.percentile(x_arr, 15), np.percentile(x_arr, 85)
        iqr = q3 - q1
        x_lo = min(q1 - 2 * max(iqr, 50), min(x_arr) * 0.9)
        x_hi = max(q3 + 2 * max(iqr, 50), max(x_arr) * 1.1)
        # Ensure 0 is visible and symmetric-ish
        x_lo = min(x_lo, -100)
        x_hi = max(x_hi, 100)

        # ── Quadrant divider lines ──
        fig.add_vline(x=0, line=dict(color="#2d2d50", width=1, dash="dash"))
        fig.add_hline(y=50, line=dict(color="#2d2d50", width=1, dash="dash"))

        # ── Quadrant labels (corner positioned) ──
        ann_font = dict(size=9, family=_MONO)
        fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper",
                           text="HIGH CARRY + RICH VOL", xanchor="right",
                           showarrow=False, font=dict(**ann_font, color="#ff8800"), opacity=0.6)
        fig.add_annotation(x=0.98, y=0.02, xref="paper", yref="paper",
                           text="HIGH CARRY + CHEAP VOL  \u2605", xanchor="right",
                           showarrow=False, font=dict(**ann_font, color="#00cc66"), opacity=0.8)
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                           text="LOW CARRY + RICH VOL", xanchor="left",
                           showarrow=False, font=dict(**ann_font, color="#ff3333"), opacity=0.6)
        fig.add_annotation(x=0.02, y=0.02, xref="paper", yref="paper",
                           text="LOW CARRY + CHEAP VOL", xanchor="left",
                           showarrow=False, font=dict(**ann_font, color="#9a9ab0"), opacity=0.4)

        # ── Bubble traces (one per momentum group for legend) ──
        groups = {
            "IMPROVING": ("#00cc66", []),
            "DETERIORATING": ("#ff3333", []),
            "STABLE": ("#9a9ab0", []),
        }
        for i in range(len(x_vals)):
            mom_key = "IMPROVING" if colors[i] == "#00cc66" else (
                "DETERIORATING" if colors[i] == "#ff3333" else "STABLE")
            groups[mom_key][1].append(i)

        for mom_label, (color, indices) in groups.items():
            if not indices:
                continue
            fig.add_trace(go.Scatter(
                x=[x_vals[i] for i in indices],
                y=[y_vals[i] for i in indices],
                mode="markers+text",
                marker=dict(
                    size=[sizes[i] for i in indices],
                    color=color,
                    opacity=0.85,
                    line=dict(width=1.5, color="#ff8800"),
                ),
                text=[texts[i] for i in indices],
                textposition="top center",
                textfont=dict(size=8, color="#e0e0e0", family=_MONO),
                hovertext=[hovers[i] for i in indices],
                hoverinfo="text",
                name=mom_label,
            ))

        fig.update_layout(
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            font=dict(family=_MONO, color="#e0e0e0", size=11),
            title=dict(
                text="CARRY vs VOL REGIME  (bubble = Sharpe proxy, color = momentum)",
                font=dict(color="#ffffff", size=12),
            ),
            xaxis=dict(
                title=dict(text="Carry (bps)", font=dict(size=10, color="#9a9ab0")),
                showgrid=True, gridcolor="#1a1a30", zeroline=False,
                tickfont=dict(size=9, color="#9a9ab0"),
            ),
            yaxis=dict(
                title=dict(text="ATM 3M Vol Percentile (%)", font=dict(size=10, color="#9a9ab0")),
                showgrid=True, gridcolor="#1a1a30", zeroline=False,
                range=[-5, 105],
                tickfont=dict(size=9, color="#9a9ab0"),
            ),
            height=400,
            margin=dict(l=50, r=30, t=40, b=40),
            legend=dict(
                title=dict(text="MOMENTUM", font=dict(size=9, color="#ff8800")),
                font=dict(color="#9a9ab0", size=9), bgcolor="rgba(0,0,0,0)",
                x=1.0, y=1.0, xanchor="right",
                bordercolor="#2d2d50", borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor="#0a0a14", bordercolor="#2d2d50",
                font=dict(color="#e0e0e0", family=_MONO, size=11),
            ),
        )
        return fig
    except Exception:
        logger.debug("Carry vs vol bubble chart failed", exc_info=True)
        return _empty_fig("CARRY vs VOL BUBBLE")


# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

def layout():
    return html.Div([
        # CSV download
        dcc.Download(id=f"{_P}-csv-download"),
        # Stores
        dcc.Store(id=f"{_P}-corr-pair-a", data="EURUSD"),
        dcc.Store(id=f"{_P}-corr-pair-b", data="USDJPY"),
        dcc.Store(id=f"{_P}-carry-pair", data="EURUSD"),
        dcc.Interval(id=f"{_P}-interval", interval=300_000, n_intervals=0),

        # ── Title + Controls ──
        html.Div([
            html.Span("RELATIVE VALUE", style={
                "color": COLORS["text_bright"], "fontSize": "13px", "fontWeight": "700",
                "letterSpacing": "2px", "fontFamily": _MONO,
            }),
            html.Div([
                dcc.Dropdown(id=f"{_P}-pair-a", options=[{"label": p, "value": p} for p in ALL_PAIRS],
                             value="EURUSD", clearable=False, style={**DROPDOWN_STYLE, "width": "110px"}),
                dcc.Dropdown(id=f"{_P}-pair-b", options=[{"label": p, "value": p} for p in ALL_PAIRS],
                             value="USDJPY", clearable=False, style={**DROPDOWN_STYLE, "width": "110px"}),
                dcc.Dropdown(id=f"{_P}-tenor", options=[{"label": t, "value": t} for t in TERM_TENORS],
                             value="3M", clearable=False, style={**DROPDOWN_STYLE, "width": "70px"}),
                dcc.Dropdown(id=f"{_P}-lookback", options=LOOKBACK_OPTIONS, value=252,
                             clearable=False, style={**DROPDOWN_STYLE, "width": "80px"}),
            ], style={"display": "flex", "gap": GAP}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "padding": f"{GAP} 0",
                  "borderBottom": f"1px solid {COLORS['border']}"}),

        # ── Sub-Tabs ──
        dcc.Tabs(id=f"{_P}-tabs", value="cross-pair", children=[
            dcc.Tab(label="CROSS-PAIR", value="cross-pair", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="CORRELATION", value="correlation", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="MACRO", value="macro", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="CARRY", value="carry", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ], style={"marginTop": GAP, "marginBottom": "0"}),

        # ══════════ CROSS-PAIR TAB ══════════
        html.Div(id=f"{_P}-cross-container", children=[
            html.Div([
                html.Div([
                    html.Button("CSV", id=f"{_P}-csv-vol-spread", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-vol-spread", config={"displayModeBar": False, "responsive": True},
                              style={"height": "320px"}),
                ], style={"flex": "1", "minWidth": "350px"}),
                html.Div([
                    html.Button("CSV", id=f"{_P}-csv-skew-scatter", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-skew-scatter", config={"displayModeBar": False, "responsive": True},
                              style={"height": "320px"}),
                ], style={"flex": "1", "minWidth": "350px"}),
            ], style={"display": "flex", "gap": GAP}),
            html.Div([
                html.Div([
                    html.Button("CSV", id=f"{_P}-csv-zscore", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-zscore-matrix", config={"displayModeBar": False, "responsive": True},
                              style={"height": "320px"}),
                ], style={"flex": "1", "minWidth": "350px"}),
                html.Div([
                    html.Button("CSV", id=f"{_P}-csv-ivrv", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-ivrv-panel", config={"displayModeBar": False, "responsive": True},
                              style={"height": "320px"}),
                ], style={"flex": "1", "minWidth": "350px"}),
            ], style={"display": "flex", "gap": GAP, "marginTop": GAP}),
            # Signal table
            html.Div(id=f"{_P}-signal-table", style={"marginTop": SECTION_GAP}),
            # RR/BF spread charts
            dcc.Graph(id=f"{_P}-rr-bf-spread", config={"displayModeBar": False, "responsive": True},
                      style={"height": "450px", "marginTop": GAP}),
            # Vol beta heatmap
            dcc.Graph(id=f"{_P}-vol-beta", config={"displayModeBar": False, "responsive": True},
                      style={"height": "380px", "marginTop": GAP}),
        ]),

        # ══════════ CORRELATION TAB ══════════
        html.Div(id=f"{_P}-corr-container", style={"display": "none"}, children=[
            # Controls + regime badges row
            html.Div([
                html.Div([
                    html.Span("WINDOW", style=LABEL_STYLE),
                    dcc.Dropdown(id=f"{_P}-corr-window",
                                 options=[{"label": f"{w}D", "value": w} for w in [20, 60, 120]],
                                 value=60, clearable=False,
                                 style={**DROPDOWN_STYLE, "width": "80px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": GAP}),
                html.Div([
                    html.Span("VOL REGIME", style={**LABEL_STYLE, "marginBottom": "0", "marginRight": GAP}),
                    html.Div(id=f"{_P}-regime-badges", style={
                        "display": "flex", "flexWrap": "wrap", "gap": "4px",
                    }),
                ], style={"display": "flex", "alignItems": "center", "flex": "1"}),
                html.Button("CSV", id=f"{_P}-csv-corr", n_clicks=0, style=CSV_BTN_STYLE),
            ], style={"display": "flex", "gap": SECTION_GAP, "alignItems": "center",
                      "marginBottom": GAP}),
            # Full-width correlation heatmap
            dcc.Graph(id=f"{_P}-corr-heatmap", config={"displayModeBar": False, "responsive": True},
                      style={"height": "600px"}),
            # Breakdown table
            html.Div(id=f"{_P}-corr-breakdown", style={"marginTop": SECTION_GAP}),
            # Rolling chart + Correlation cone side by side
            html.Div([
                html.Div([
                    html.Button("CSV", id=f"{_P}-csv-rolling", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-rolling-chart", config={"displayModeBar": False, "responsive": True},
                              style={"height": "280px"}),
                ], style={"flex": "1"}),
                html.Div([
                    dcc.Graph(id=f"{_P}-corr-cone", config={"displayModeBar": False, "responsive": True},
                              style={"height": "280px"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": GAP, "marginTop": GAP}),
            # Correlation network (force-directed graph)
            dcc.Graph(id=f"{_P}-corr-network", config={"displayModeBar": False, "responsive": True},
                      style={"height": "420px", "marginTop": GAP}),
        ]),

        # ══════════ MACRO TAB ══════════
        html.Div(id=f"{_P}-macro-container", style={"display": "none"}, children=[
            html.Div([
                html.Div([
                    html.Div("RATE DIFFERENTIALS", style={
                        "color": COLORS["text_secondary"], "fontSize": "9px", "fontWeight": "700",
                        "letterSpacing": "1.5px", "marginBottom": GAP, "fontFamily": _MONO,
                    }),
                    html.Div(id=f"{_P}-rate-table", style={"overflowY": "auto", "maxHeight": "350px"}),
                ], style={"flex": "1", "border": f"1px solid {COLORS['border']}", "padding": GAP}),
                html.Div([
                    html.Div(id=f"{_P}-risk-sentiment", style={"marginBottom": GAP}),
                    html.Button("CSV", id=f"{_P}-csv-cb", n_clicks=0, style=CSV_BTN_STYLE),
                    dcc.Graph(id=f"{_P}-cb-chart", config={"displayModeBar": False, "responsive": True},
                              style={"height": f"{CHART_SM}px"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": GAP}),
            html.Div([
                html.Div([
                    html.Span("LOOKBACK", style=LABEL_STYLE),
                    dcc.Dropdown(id=f"{_P}-dxy-lookback", options=LOOKBACK_OPTIONS, value=252,
                                 clearable=False, style={**DROPDOWN_STYLE, "width": "80px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": GAP, "padding": f"{GAP} 0"}),
                html.Button("CSV", id=f"{_P}-csv-dxy", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id=f"{_P}-dxy-chart", config={"displayModeBar": False, "responsive": True},
                          style={"height": f"{CHART_SM}px"}),
            ], style={"marginTop": SECTION_GAP}),
        ]),

        # ══════════ CARRY TAB ══════════
        html.Div(id=f"{_P}-carry-container", style={"display": "none"}, children=[
            # Carry vs Vol bubble chart (at the top)
            dcc.Graph(id=f"{_P}-carry-bubble", config={"displayModeBar": False, "responsive": True},
                      style={"height": "400px", "marginBottom": GAP}),
            html.Div(id=f"{_P}-carry-table-wrapper", style={
                "overflowY": "auto", "maxHeight": "350px", "border": f"1px solid {COLORS['border']}",
            }),
            html.Div([
                html.Div([
                    html.Span("PAIR", style=LABEL_STYLE),
                    dcc.Dropdown(id=f"{_P}-carry-pair-dd",
                                 options=[{"label": p, "value": p} for p in ALL_PAIRS],
                                 value="EURUSD", clearable=False,
                                 style={**DROPDOWN_STYLE, "width": "110px"}),
                    html.Span("COMPARE", style={**LABEL_STYLE, "marginLeft": "12px"}),
                    dcc.Dropdown(id=f"{_P}-carry-comp-dd",
                                 options=[{"label": p, "value": p} for p in ALL_PAIRS],
                                 value="USDJPY", clearable=False,
                                 style={**DROPDOWN_STYLE, "width": "110px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": GAP, "padding": f"{GAP} 0"}),
                html.Div([
                    html.Div([
                        html.Button("CSV", id=f"{_P}-csv-carry-term", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id=f"{_P}-carry-term", config={"displayModeBar": False, "responsive": True},
                                  style={"height": "300px"}),
                    ], style={"flex": "1", "minWidth": "350px"}),
                    html.Div([
                        html.Button("CSV", id=f"{_P}-csv-carry-fwd", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id=f"{_P}-carry-fwd", config={"displayModeBar": False, "responsive": True},
                                  style={"height": "300px"}),
                    ], style={"flex": "1", "minWidth": "350px"}),
                ], style={"display": "flex", "gap": GAP}),
                html.Button("CSV", id=f"{_P}-csv-carry-cal", n_clicks=0, style=CSV_BTN_STYLE),
                dcc.Graph(id=f"{_P}-carry-calendar", config={"displayModeBar": False, "responsive": True},
                          style={"height": f"{CHART_MD}px", "marginTop": GAP}),
            ], style={"marginTop": SECTION_GAP}),
        ]),

    ], style={"fontFamily": _MONO})


# ═══════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app):

    # ── Tab visibility ──
    @app.callback(
        [
            Output(f"{_P}-cross-container", "style"),
            Output(f"{_P}-corr-container", "style"),
            Output(f"{_P}-macro-container", "style"),
            Output(f"{_P}-carry-container", "style"),
        ],
        Input(f"{_P}-tabs", "value"),
    )
    def toggle_tabs(tab):
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if tab == "cross-pair" else hide,
            show if tab == "correlation" else hide,
            show if tab == "macro" else hide,
            show if tab == "carry" else hide,
        )

    # ══════════ CROSS-PAIR CALLBACKS ══════════

    @app.callback(
        [
            Output(f"{_P}-vol-spread", "figure"),
            Output(f"{_P}-skew-scatter", "figure"),
            Output(f"{_P}-zscore-matrix", "figure"),
            Output(f"{_P}-ivrv-panel", "figure"),
        ],
        [
            Input(f"{_P}-pair-a", "value"),
            Input(f"{_P}-pair-b", "value"),
            Input(f"{_P}-tenor", "value"),
            Input(f"{_P}-lookback", "value"),
        ],
    )
    def update_cross_pair(pair_a, pair_b, tenor, lookback):
        try:
            pa, pb = pair_a or "EURUSD", pair_b or "USDJPY"
            t = tenor or "3M"
            lb = lookback or 252
            return (
                _build_vol_spread_ts(pa, pb, t, lb),
                _build_skew_scatter(pa, pb, t, lb),
                _build_zscore_matrix(lb),
                _build_ivrv_panel(pa, t, lb),
            )
        except Exception as exc:
            logging.exception("update_cross_pair failed")
            empty = no_data_fig("Error")
            return empty, empty, empty, empty

    @app.callback(
        Output(f"{_P}-signal-table", "children"),
        Input(f"{_P}-lookback", "value"),
    )
    def update_signal_table(lookback):
        rows = _build_signal_table(lookback or 252)
        if not rows:
            return html.Div("No signal data", style={"color": COLORS["text_secondary"], "fontSize": "10px"})

        columns = [
            {"name": "PAIR", "id": "pair"},
            {"name": "ATM Z", "id": "atm_z", "type": "numeric"},
            {"name": "RR Z", "id": "rr_z", "type": "numeric"},
            {"name": "COMPOSITE", "id": "composite", "type": "numeric"},
            {"name": "SIGNAL", "id": "direction"},
            {"name": "CONF", "id": "confidence"},
        ]
        return dash_table.DataTable(
            columns=columns, data=rows, sort_action="native", page_size=30,
            style_table={"overflowX": "auto", "maxHeight": "400px"},
            style_header={
                "backgroundColor": COLORS["bg_primary"], "color": COLORS["text_secondary"],
                "fontWeight": "700", "fontSize": "9px", "textTransform": "uppercase",
                "border": f"1px solid {COLORS['border']}", "fontFamily": _MONO,
            },
            style_cell={
                "backgroundColor": COLORS["bg_primary"], "color": COLORS["text_primary"],
                "fontSize": "11px", "fontFamily": _MONO,
                "border": f"1px solid {COLORS['border']}", "padding": "4px 6px",
                "textAlign": "right",
            },
            style_cell_conditional=[
                {"if": {"column_id": "pair"}, "textAlign": "left", "fontWeight": "700"},
                {"if": {"column_id": "direction"}, "textAlign": "center"},
                {"if": {"column_id": "confidence"}, "textAlign": "center"},
            ],
            style_data_conditional=[
                {"if": {"filter_query": '{direction} = "BUY VOL"', "column_id": "direction"},
                 "color": "#1565c0", "fontWeight": "bold"},
                {"if": {"filter_query": '{direction} = "SELL VOL"', "column_id": "direction"},
                 "color": COLORS["accent_red"], "fontWeight": "bold"},
                {"if": {"filter_query": "{composite} > 30", "column_id": "composite"},
                 "color": COLORS["accent_red"], "fontWeight": "bold"},
                {"if": {"filter_query": "{composite} < -30", "column_id": "composite"},
                 "color": "#1565c0", "fontWeight": "bold"},
                {"if": {"filter_query": '{confidence} = "HIGH"', "column_id": "confidence"},
                 "color": COLORS["accent_green"], "fontWeight": "bold"},
            ],
        )

    # ── Vol Beta Heatmap ──
    @app.callback(
        Output(f"{_P}-vol-beta", "figure"),
        [Input(f"{_P}-tabs", "value"), Input(f"{_P}-interval", "n_intervals")],
    )
    def update_vol_beta(tab, n):
        if tab != "cross-pair":
            raise PreventUpdate
        return _build_vol_beta_heatmap()

    # ── RR/BF Spread Charts ──
    @app.callback(
        Output(f"{_P}-rr-bf-spread", "figure"),
        [Input(f"{_P}-pair-a", "value"), Input(f"{_P}-pair-b", "value"),
         Input(f"{_P}-tenor", "value"), Input(f"{_P}-tabs", "value")],
    )
    def update_rr_bf(pair_a, pair_b, tenor, tab):
        if tab != "cross-pair":
            raise PreventUpdate
        pa = pair_a or "EURUSD"
        pb = pair_b or "USDJPY"
        t = tenor or "3M"
        if pa == pb:
            return _empty_fig("Select two different pairs")
        return _build_rr_bf_spreads(pa, pb, t)

    # ══════════ CORRELATION CALLBACKS ══════════

    @app.callback(
        [
            Output(f"{_P}-corr-heatmap", "figure"),
            Output(f"{_P}-regime-badges", "children"),
            Output(f"{_P}-corr-breakdown", "children"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-corr-window", "value"),
            Input(f"{_P}-tabs", "value"),
        ],
    )
    def update_correlation(n, window, tab):
        if tab != "correlation":
            raise PreventUpdate
        w = window or 60
        return (
            _build_corr_heatmap(w),
            _build_regime_badges(),
            _build_breakdown_table(20, 120),
        )

    @app.callback(
        [Output(f"{_P}-corr-pair-a", "data"), Output(f"{_P}-corr-pair-b", "data")],
        Input(f"{_P}-corr-heatmap", "clickData"),
        prevent_initial_call=True,
    )
    def corr_heatmap_click(click_data):
        if not click_data or not click_data.get("points") or len(click_data["points"]) == 0:
            raise PreventUpdate
        pt = click_data["points"][0]
        x_label = pt.get("x", "")
        y_label = pt.get("y", "")
        # Parse "EUR/USD" → "EURUSD"
        pair_a = y_label.replace("/", "")
        pair_b = x_label.replace("/", "")
        if pair_a in MONITOR_PAIRS and pair_b in MONITOR_PAIRS:
            return pair_a, pair_b
        raise PreventUpdate

    @app.callback(
        Output(f"{_P}-rolling-chart", "figure"),
        [Input(f"{_P}-corr-pair-a", "data"), Input(f"{_P}-corr-pair-b", "data")],
    )
    def update_rolling(pair_a, pair_b):
        pa = pair_a or "EURUSD"
        pb = pair_b or "USDJPY"
        if pa == pb:
            return _empty_fig("Select two different pairs")
        return _build_rolling_chart(pa, pb)

    @app.callback(
        Output(f"{_P}-corr-cone", "figure"),
        [Input(f"{_P}-corr-pair-a", "data"), Input(f"{_P}-corr-pair-b", "data"),
         Input(f"{_P}-tabs", "value")],
    )
    def update_corr_cone(pair_a, pair_b, tab):
        if tab != "correlation":
            raise PreventUpdate
        pa = pair_a or "EURUSD"
        pb = pair_b or "USDJPY"
        if pa == pb:
            return _empty_fig("Select two different pairs")
        return _build_corr_cone(pa, pb)

    @app.callback(
        Output(f"{_P}-corr-network", "figure"),
        [Input(f"{_P}-corr-window", "value"), Input(f"{_P}-tabs", "value")],
    )
    def update_corr_network(window, tab):
        if tab != "correlation":
            raise PreventUpdate
        return _build_corr_network(window or 60)

    # ══════════ MACRO CALLBACKS ══════════

    @app.callback(
        [
            Output(f"{_P}-rate-table", "children"),
            Output(f"{_P}-risk-sentiment", "children"),
            Output(f"{_P}-cb-chart", "figure"),
            Output(f"{_P}-dxy-chart", "figure"),
        ],
        [
            Input(f"{_P}-interval", "n_intervals"),
            Input(f"{_P}-dxy-lookback", "value"),
            Input(f"{_P}-tabs", "value"),
        ],
    )
    def update_macro(n, lookback, tab):
        if tab != "macro":
            raise PreventUpdate
        try:
            return (
                _build_rate_table(),
                _build_risk_sentiment(),
                _build_cb_chart(),
                _build_dxy_chart(lookback or 252),
            )
        except Exception as exc:
            logging.exception("update_macro failed")
            empty = no_data_fig("Error")
            err_div = html.Div("Macro data error", style={"color": COLORS["accent_red"]})
            return err_div, [err_div], empty, empty

    # ══════════ CARRY CALLBACKS ══════════

    @app.callback(
        Output(f"{_P}-carry-bubble", "figure"),
        [Input(f"{_P}-tabs", "value"), Input(f"{_P}-interval", "n_intervals")],
    )
    def update_carry_bubble(tab, n):
        if tab != "carry":
            raise PreventUpdate
        return _build_carry_vol_bubble()

    @app.callback(
        Output(f"{_P}-carry-table-wrapper", "children"),
        [Input(f"{_P}-interval", "n_intervals"), Input(f"{_P}-tabs", "value")],
    )
    def update_carry_table(n, tab):
        if tab != "carry":
            raise PreventUpdate
        try:
            rows = _build_carry_data()
            if not rows:
                return html.Div("No carry data", style={"color": COLORS["text_secondary"]})

            header = html.Tr([html.Th(h, style={**TABLE_HEADER_STYLE, "fontSize": "8px"})
                               for h in ["PAIR", "ATM 1M", "ATM 3M", "ATM 1Y", "TERM", "FWD 3×3",
                                         "CARRY/D", "CARRY/VOL", "SIGNAL"]])
            body = []
            for r in rows:
                sig_color = COLORS["accent_green"] if "SELL" in r.get("signal", "") else COLORS["accent_red"] if "BUY" in r.get("signal", "") else COLORS["text_secondary"]
                def _rv_fmt(v, fmt=".1f", suffix=""):
                    return f"{v:{fmt}}{suffix}" if np.isfinite(v) else "—"
                ts = r['term_spread']
                cv = r['carry_vol']
                ts_color = COLORS["accent_red"] if np.isfinite(ts) and ts > 0.5 else COLORS["accent_green"] if np.isfinite(ts) and ts < -0.5 else COLORS["text_primary"]
                cv_color = COLORS["accent_green"] if np.isfinite(cv) and cv > 2 else COLORS["accent_red"] if np.isfinite(cv) and cv < -1 else COLORS["text_primary"]
                body.append(html.Tr([
                    html.Td(r["pair"], style={**TABLE_CELL_STYLE, "fontWeight": "700"}),
                    html.Td(_rv_fmt(r['atm_1m'], ".1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(r['atm_3m'], ".1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(r['atm_1y'], ".1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(ts, "+.1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right", "color": ts_color}),
                    html.Td(_rv_fmt(r['fwd_3x3'], ".1f", "v"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(r['carry_day'], ".3f"), style={**TABLE_CELL_STYLE, "textAlign": "right"}),
                    html.Td(_rv_fmt(cv, ".1f", "%"), style={**TABLE_CELL_STYLE, "textAlign": "right",
                             "fontWeight": "700", "color": cv_color}),
                    html.Td(r["signal"], style={**TABLE_CELL_STYLE, "color": sig_color, "fontWeight": "600"}),
                ]))

            return html.Table([html.Thead(header), html.Tbody(body)],
                              style={"width": "100%", "borderCollapse": "collapse",
                                     "fontFamily": _MONO, "fontSize": "10px"})
        except Exception as exc:
            logging.exception("update_carry_table failed")
            return html.Div("Carry data error", style={"color": COLORS["accent_red"], "padding": "8px"})

    @app.callback(
        [Output(f"{_P}-carry-term", "figure"), Output(f"{_P}-carry-fwd", "figure"),
         Output(f"{_P}-carry-calendar", "figure")],
        [Input(f"{_P}-carry-pair-dd", "value"), Input(f"{_P}-carry-comp-dd", "value")],
    )
    def update_carry_charts(pair, comp):
        try:
            p = pair or "EURUSD"
            return _build_term_chart(p, comp), _build_fwd_vol(p), _build_calendar_spread(p)
        except Exception as exc:
            logging.exception("update_carry_charts failed")
            empty = no_data_fig("Error")
            return empty, empty, empty

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output(f"{_P}-csv-download", "data"),
        [Input(f"{_P}-csv-vol-spread", "n_clicks"),
         Input(f"{_P}-csv-skew-scatter", "n_clicks"),
         Input(f"{_P}-csv-zscore", "n_clicks"),
         Input(f"{_P}-csv-ivrv", "n_clicks"),
         Input(f"{_P}-csv-corr", "n_clicks"),
         Input(f"{_P}-csv-rolling", "n_clicks"),
         Input(f"{_P}-csv-cb", "n_clicks"),
         Input(f"{_P}-csv-dxy", "n_clicks"),
         Input(f"{_P}-csv-carry-term", "n_clicks"),
         Input(f"{_P}-csv-carry-fwd", "n_clicks"),
         Input(f"{_P}-csv-carry-cal", "n_clicks")],
        [State(f"{_P}-vol-spread", "figure"),
         State(f"{_P}-skew-scatter", "figure"),
         State(f"{_P}-zscore-matrix", "figure"),
         State(f"{_P}-ivrv-panel", "figure"),
         State(f"{_P}-corr-heatmap", "figure"),
         State(f"{_P}-rolling-chart", "figure"),
         State(f"{_P}-cb-chart", "figure"),
         State(f"{_P}-dxy-chart", "figure"),
         State(f"{_P}-carry-term", "figure"),
         State(f"{_P}-carry-fwd", "figure"),
         State(f"{_P}-carry-calendar", "figure")],
        prevent_initial_call=True,
    )
    def rvp_csv_export(*args):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        names = ["vol-spread", "skew-scatter", "zscore", "ivrv",
                 "corr", "rolling", "cb", "dxy",
                 "carry-term", "carry-fwd", "carry-cal"]
        labels = ["VolSpread", "SkewScatter", "ZScoreMatrix", "IVRV",
                  "CorrHeatmap", "RollingCorr", "CBDivergence", "DXYProxy",
                  "CarryTerm", "CarryFwd", "CarryCalendar"]
        n = len(names)
        for i, name in enumerate(names):
            if btn == f"{_P}-csv-{name}":
                fig = args[n + i]
                if fig:
                    return export_csv(fig, "RelValue", labels[i])
                return no_update
        return no_update
