"""
FX Trade Idea Workshop
=======================
Institutional-grade FX options structuring workstation.  Multi-leg builder
with delta- and strike-based entry, Garman-Kohlhagen pricing, full Greeks,
**expected-value calculator** (Breeden-Litzenberger implied density),
vol-regime-aware **suggestion engine**, multi-tenor scan, **solver**
(zero-cost / delta-neutral / target metrics), compare mode, edge analysis,
probability-weighted payoff overlay, and scenario sensitivity.

25 preset structures plus fully custom multi-leg construction.
"""

import logging
from copy import deepcopy
from datetime import datetime

import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, INPUT_STYLE, BUTTON_STYLE,
    BUTTON_SUCCESS_STYLE, BUTTON_DANGER_STYLE,
    make_stat_style, clickable_stat,
    CSV_BTN_STYLE, no_data_fig,
    stat_box as _stat_box, ordinal as _ordinal,
)
from core.config import TENORS_TRADING
from core.csv_export import export_csv
from core.fx_analytics import (
    vol_percentile, iv_rv_percentile, breakeven_vol,
    smile_implied_pdf, vol_regime_detect,
)
from core.bloomberg_fx import get_fx_vol_surface, get_fx_spots, get_fx_rates, get_all_pairs
from core.fx_conventions import (
    tenor_to_years, tenor_to_days, years_to_nearest_tenor,
    delta_to_strike, atm_dns_strike, bf_rr_to_smile,
    FX_PAIR_REGISTRY, forward_points, fx_forward, premium_pips, premium_pct_notional,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Tenor List
# ============================================================================

TENORS = TENORS_TRADING


# ============================================================================
# Preset Structures (25)
# ============================================================================

PRESETS = {
    "Custom": [],
    # ── Vanillas ─────────────────────────────────────────────────────────
    "Call": [
        {"cp": "call", "side": "buy", "delta": 0.50, "ratio": 1},
    ],
    "Put": [
        {"cp": "put", "side": "buy", "delta": 0.50, "ratio": 1},
    ],
    # ── Vertical Spreads ─────────────────────────────────────────────────
    "Call Spread": [
        {"cp": "call", "side": "buy", "delta": 0.40, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.25, "ratio": 1},
    ],
    "Put Spread": [
        {"cp": "put", "side": "buy", "delta": 0.40, "ratio": 1},
        {"cp": "put", "side": "sell", "delta": 0.25, "ratio": 1},
    ],
    # ── Volatility Structures ────────────────────────────────────────────
    "Risk Reversal": [
        {"cp": "put", "side": "sell", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "buy", "delta": 0.25, "ratio": 1},
    ],
    "Straddle": [
        {"cp": "call", "side": "buy", "delta": 0.50, "ratio": 1},
        {"cp": "put", "side": "buy", "delta": 0.50, "ratio": 1},
    ],
    "Strangle": [
        {"cp": "put", "side": "buy", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "buy", "delta": 0.25, "ratio": 1},
    ],
    "25D Risk Reversal": [
        {"cp": "put", "side": "sell", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "buy", "delta": 0.25, "ratio": 1},
    ],
    "25D Strangle": [
        {"cp": "put", "side": "buy", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "buy", "delta": 0.25, "ratio": 1},
    ],
    "10D Strangle": [
        {"cp": "put", "side": "buy", "delta": 0.10, "ratio": 1},
        {"cp": "call", "side": "buy", "delta": 0.10, "ratio": 1},
    ],
    # ── Butterflies & Condors ────────────────────────────────────────────
    "Butterfly": [
        {"cp": "call", "side": "buy", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.50, "ratio": 2},
        {"cp": "call", "side": "buy", "delta": 0.75, "ratio": 1},
    ],
    "Iron Butterfly": [
        {"cp": "put", "side": "sell", "delta": 0.50, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.50, "ratio": 1},
        {"cp": "put", "side": "buy", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "buy", "delta": 0.25, "ratio": 1},
    ],
    "Iron Condor": [
        {"cp": "put", "side": "buy", "delta": 0.10, "ratio": 1},
        {"cp": "put", "side": "sell", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "buy", "delta": 0.10, "ratio": 1},
    ],
    "Broken Wing Butterfly": [
        {"cp": "call", "side": "buy", "delta": 0.40, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.50, "ratio": 2},
        {"cp": "call", "side": "buy", "delta": 0.15, "ratio": 1},
    ],
    # ── Exotic Spreads ───────────────────────────────────────────────────
    "Seagull": [
        {"cp": "call", "side": "buy", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.10, "ratio": 1},
        {"cp": "put", "side": "sell", "delta": 0.25, "ratio": 1},
    ],
    "Collar": [
        {"cp": "put", "side": "buy", "delta": 0.25, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.25, "ratio": 1},
    ],
    "Fence": [
        {"cp": "put", "side": "buy", "delta": 0.30, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.30, "ratio": 1},
    ],
    "Participating Forward": [
        {"cp": "call", "side": "buy", "delta": 0.50, "ratio": 1},
        {"cp": "put", "side": "sell", "delta": 0.50, "ratio": 1},
    ],
    "Leveraged Forward": [
        {"cp": "call", "side": "buy", "delta": 0.50, "ratio": 1},
        {"cp": "put", "side": "sell", "delta": 0.35, "ratio": 2},
    ],
    # ── Ratio Structures ─────────────────────────────────────────────────
    "1x2 Call Spread": [
        {"cp": "call", "side": "buy", "delta": 0.50, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.25, "ratio": 2},
    ],
    "1x2 Put Spread": [
        {"cp": "put", "side": "buy", "delta": 0.50, "ratio": 1},
        {"cp": "put", "side": "sell", "delta": 0.25, "ratio": 2},
    ],
    # ── Calendar & Diagonal ──────────────────────────────────────────────
    "Calendar Spread": [
        {"cp": "call", "side": "buy", "delta": 0.50, "ratio": 1, "tenor_mult": 2.0},
        {"cp": "call", "side": "sell", "delta": 0.50, "ratio": 1, "tenor_mult": 1.0},
    ],
    "Diagonal Spread": [
        {"cp": "call", "side": "buy", "delta": 0.40, "ratio": 1, "tenor_mult": 2.0},
        {"cp": "call", "side": "sell", "delta": 0.30, "ratio": 1, "tenor_mult": 1.0},
    ],
    # ── Multi-Leg Exotics ────────────────────────────────────────────────
    "Christmas Tree": [
        {"cp": "call", "side": "buy", "delta": 0.50, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.35, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.20, "ratio": 1},
    ],
    "Ladder": [
        {"cp": "call", "side": "buy", "delta": 0.50, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.35, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.20, "ratio": 1},
        {"cp": "put", "side": "sell", "delta": 0.25, "ratio": 1},
    ],
    # ── Ratio Plays ─────────────────────────────────────────────────────
    "1x3 Call Spread": [
        {"cp": "call", "side": "buy", "delta": 0.50, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.25, "ratio": 3},
    ],
    "1x3 Put Spread": [
        {"cp": "put", "side": "buy", "delta": 0.50, "ratio": 1},
        {"cp": "put", "side": "sell", "delta": 0.25, "ratio": 3},
    ],
    "Jade Lizard": [
        {"cp": "call", "side": "sell", "delta": 0.30, "ratio": 1},
        {"cp": "call", "side": "buy", "delta": 0.15, "ratio": 1},
        {"cp": "put", "side": "sell", "delta": 0.25, "ratio": 1},
    ],
}


MAX_LEGS = 8

SCAN_TENORS = ["1M", "2M", "3M", "6M", "1Y"]

# ============================================================================
# Structure Classification — maps presets to market views for Trade Analysis
# ============================================================================

STRUCTURE_VIEWS = {
    "Custom":                {"view": "Custom structure", "type": "custom", "vol_view": "unknown"},
    "Call":                  {"view": "Bullish spot", "type": "directional", "vol_view": "long_vol"},
    "Put":                   {"view": "Bearish spot", "type": "directional", "vol_view": "long_vol"},
    "Call Spread":           {"view": "Moderately bullish", "type": "directional", "vol_view": "neutral"},
    "Put Spread":            {"view": "Moderately bearish", "type": "directional", "vol_view": "neutral"},
    "Risk Reversal":         {"view": "Bullish spot + skew", "type": "directional", "vol_view": "skew"},
    "25D Risk Reversal":     {"view": "Bullish spot + skew", "type": "directional", "vol_view": "skew"},
    "Straddle":              {"view": "Long volatility", "type": "vol", "vol_view": "long_vol"},
    "Strangle":              {"view": "Long volatility (wide)", "type": "vol", "vol_view": "long_vol"},
    "25D Strangle":          {"view": "Long volatility (25D wings)", "type": "vol", "vol_view": "long_vol"},
    "10D Strangle":          {"view": "Long tail risk", "type": "vol", "vol_view": "long_vol"},
    "Butterfly":             {"view": "Short vol / range-bound", "type": "vol", "vol_view": "short_vol"},
    "Iron Butterfly":        {"view": "Short vol / defined risk", "type": "vol", "vol_view": "short_vol"},
    "Iron Condor":           {"view": "Range-bound / short vol", "type": "vol", "vol_view": "short_vol"},
    "Broken Wing Butterfly": {"view": "Range-bound with directional bias", "type": "hybrid", "vol_view": "short_vol"},
    "Seagull":               {"view": "Bullish, capped, funded by sold put", "type": "hybrid", "vol_view": "neutral"},
    "Collar":                {"view": "Hedged long / limited range", "type": "hedge", "vol_view": "neutral"},
    "Fence":                 {"view": "Hedged position / tight range", "type": "hedge", "vol_view": "neutral"},
    "Participating Forward": {"view": "Full directional + offset premium", "type": "directional", "vol_view": "neutral"},
    "Leveraged Forward":     {"view": "Leveraged directional (ratio risk)", "type": "directional", "vol_view": "neutral"},
    "1x2 Call Spread":       {"view": "Moderately bullish (ratio risk above)", "type": "hybrid", "vol_view": "short_vol"},
    "1x2 Put Spread":        {"view": "Moderately bearish (ratio risk below)", "type": "hybrid", "vol_view": "short_vol"},
    "Calendar Spread":       {"view": "Long term structure / roll-down", "type": "vol", "vol_view": "term_structure"},
    "Diagonal Spread":       {"view": "Directional + term structure", "type": "hybrid", "vol_view": "term_structure"},
    "Christmas Tree":        {"view": "Moderately bullish, low cost", "type": "directional", "vol_view": "neutral"},
    "Ladder":                {"view": "Directional with distributed risk", "type": "hybrid", "vol_view": "neutral"},
    "1x3 Call Spread":       {"view": "Leveraged bullish (high ratio risk above)", "type": "hybrid", "vol_view": "short_vol"},
    "1x3 Put Spread":        {"view": "Leveraged bearish (high ratio risk below)", "type": "hybrid", "vol_view": "short_vol"},
    "Jade Lizard":           {"view": "Short vol, no upside risk, short downside", "type": "vol", "vol_view": "short_vol"},
}


# ============================================================================
# Inline Garman-Kohlhagen Pricing
# ============================================================================

def _norm_cdf(x):
    """Standard normal CDF using numpy/scipy-free erf approx for speed."""
    return 0.5 * (1.0 + _erf_approx(x / np.sqrt(2.0)))


def _erf_approx(x):
    """Abramowitz & Stegun erf approximation, max error ~1.5e-7."""
    a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                           -1.453152027, 1.061405429)
    p = 0.3275911
    sign = np.where(x >= 0, 1.0, -1.0)
    x_abs = np.abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs)
    return sign * y


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _gk_d1d2(S, K, T, r_d, r_f, sigma):
    """Compute d1 and d2 for Garman-Kohlhagen."""
    T_safe = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-6)
    K = np.maximum(K, 1e-10)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T_safe) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _gk_price(S, K, T, r_d, r_f, sigma, cp):
    """
    Garman-Kohlhagen option price.
    cp: +1 for call, -1 for put.
    Works with scalars or numpy arrays.
    """
    T_safe = np.maximum(T, 1e-10)
    d1, d2 = _gk_d1d2(S, K, T_safe, r_d, r_f, sigma)
    price = cp * (S * np.exp(-r_f * T_safe) * _norm_cdf(cp * d1)
                  - K * np.exp(-r_d * T_safe) * _norm_cdf(cp * d2))
    return np.maximum(price, 0.0)


def _gk_greeks(S, K, T, r_d, r_f, sigma, cp):
    """
    Full Greeks suite for a single Garman-Kohlhagen option.
    Returns dict with delta, gamma, vega, theta, rho_d, rho_f.
    """
    T_safe = max(float(T), 1e-10)
    sigma_safe = max(float(sigma), 1e-6)
    S_f = max(float(S), 1e-10)
    K_f = float(K)
    r_d_f = float(r_d)
    r_f_f = float(r_f)
    cp_f = float(cp)

    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S_f / K_f) + (r_d_f - r_f_f + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T

    nd1 = float(_norm_cdf(cp_f * d1))
    nd2 = float(_norm_cdf(cp_f * d2))
    npd1 = float(_norm_pdf(d1))

    exp_rf = np.exp(-r_f_f * T_safe)
    exp_rd = np.exp(-r_d_f * T_safe)

    delta = cp_f * exp_rf * nd1
    gamma = exp_rf * npd1 / (S_f * sigma_safe * sqrt_T)
    vega = S_f * exp_rf * npd1 * sqrt_T / 100.0  # per 1 vol point
    theta_daily = (
        -S_f * exp_rf * npd1 * sigma_safe / (2.0 * sqrt_T)
        + cp_f * r_f_f * S_f * exp_rf * nd1
        - cp_f * r_d_f * K_f * exp_rd * nd2
    ) / 365.0
    rho_d = cp_f * K_f * T_safe * exp_rd * nd2 / 10000.0
    rho_f = -cp_f * S_f * T_safe * exp_rf * nd1 / 10000.0
    vanna = -exp_rf * npd1 * d2 / sigma_safe
    volga = S_f * exp_rf * npd1 * sqrt_T * d1 * d2 / sigma_safe

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_daily,
        "rho_d": rho_d,
        "rho_f": rho_f,
        "vanna": vanna,
        "volga": volga,
    }


# ============================================================================
# Vol Interpolation Helper
# ============================================================================

def _interp_vol_for_delta(vol_surface_data, tenor, delta_abs, cp_sign):
    """
    Interpolate vol from the surface for a given delta and tenor.
    vol_surface_data: {tenor: {atm, rr25, bf25, rr10, bf10}}
    delta_abs: absolute delta (e.g. 0.25)
    cp_sign: +1 call, -1 put
    Returns vol as a fraction (e.g. 0.07 for 7%).
    """
    if not vol_surface_data:
        return 0.08
    if tenor not in vol_surface_data:
        def _safe_tenor_dist(t):
            try:
                return abs(tenor_to_years(t) - tenor_to_years(tenor))
            except Exception:
                return 999
        available = sorted(vol_surface_data.keys(), key=_safe_tenor_dist)
        if not available:
            return 0.08
        tenor = available[0]

    q = vol_surface_data.get(tenor, {})
    if not q:
        return 0.08
    # bf_rr_to_smile expects vol-point inputs (e.g., 8.5, -0.3, 0.25)
    atm = q.get("atm", 8.0)
    rr25 = q.get("rr25", 0)
    bf25 = q.get("bf25", 0)
    rr10 = q.get("rr10", 0)
    bf10 = q.get("bf10", 0)

    smile = bf_rr_to_smile(atm, rr25, bf25, rr10, bf10)

    # smile values are in vol-points; convert to decimal for GK pricing
    if abs(delta_abs - 0.50) < 0.02:
        return smile["atm"] / 100.0

    if cp_sign > 0:
        if abs(delta_abs - 0.25) < 0.03:
            return smile["c25"] / 100.0
        elif delta_abs <= 0.12:
            return smile.get("c10", smile["c25"]) / 100.0
        else:
            t = (delta_abs - 0.25) / (0.50 - 0.25)
            t = max(0.0, min(1.0, t))
            return (smile["c25"] * (1 - t) + smile["atm"] * t) / 100.0
    else:
        if abs(delta_abs - 0.25) < 0.03:
            return smile["p25"] / 100.0
        elif delta_abs <= 0.12:
            return smile.get("p10", smile["p25"]) / 100.0
        else:
            t = (delta_abs - 0.25) / (0.50 - 0.25)
            t = max(0.0, min(1.0, t))
            return (smile["p25"] * (1 - t) + smile["atm"] * t) / 100.0


def _fmt_strike(K, pair):
    """Format strike with correct decimal places per FX pair convention."""
    if pair.endswith("JPY") or pair.startswith("JPY"):
        return f"{K:.2f}"
    if pair in ("USDMXN", "USDZAR", "USDTRY", "USDBRL", "USDCNH",
                "USDINR", "USDKRW", "EURNOK", "EURSEK", "USDNOK", "USDSEK"):
        return f"{K:.3f}"
    return f"{K:.5f}"


def _get_atm_vol(vol_surface_data, tenor):
    """Get ATM vol for a tenor in decimal form."""
    if not vol_surface_data:
        return 0.08
    if tenor not in vol_surface_data:
        def _safe_td(t):
            try:
                return abs(tenor_to_years(t) - tenor_to_years(tenor))
            except Exception:
                return 999
        available = sorted(vol_surface_data.keys(), key=_safe_td)
        if not available:
            return 0.08
        tenor = available[0]
    return vol_surface_data.get(tenor, {}).get("atm", 8.0) / 100.0


# ============================================================================
# Leg Processing
# ============================================================================

def _process_legs(legs_config, pair, tenor, notional, spot_data, rates, vol_surface):
    """
    For each leg: convert delta to strike, look up vol, price via GK.
    Returns list of leg dicts with all computed fields.
    """
    S = spot_data.get("mid", spot_data.get("bid", 1.0))
    r_d = rates.get("r_dom", 0.03)
    r_f = rates.get("r_for", 0.02)
    T = tenor_to_years(tenor)
    pip_size = 0.0001
    if pair in FX_PAIR_REGISTRY:
        pip_size = FX_PAIR_REGISTRY[pair].pip

    processed = []
    for i, leg in enumerate(legs_config):
        cp_str = leg.get("cp", "call")
        side_str = leg.get("side", "buy")
        delta_abs = max(0.01, min(0.99, float(leg.get("delta", 0.25))))
        ratio = max(1, min(10, int(leg.get("ratio", 1))))
        tenor_mult = max(0.25, min(5.0, float(leg.get("tenor_mult", 1.0))))

        cp_sign = 1 if cp_str == "call" else -1
        side_sign = 1 if side_str == "buy" else -1

        # Apply tenor multiplier for calendar/diagonal spread legs
        leg_T = T * tenor_mult
        leg_tenor = years_to_nearest_tenor(leg_T) if tenor_mult != 1.0 else tenor

        vol = _interp_vol_for_delta(vol_surface, leg_tenor, delta_abs, cp_sign)
        vol = max(vol, 1e-6)  # guard against zero/negative vol from lookup

        target_delta = delta_abs * cp_sign
        K = delta_to_strike(target_delta, S, leg_T, r_d, r_f, vol, cp_sign)
        if np.isnan(K) or K <= 0:
            import logging
            logging.getLogger(__name__).warning(
                "Leg %d: delta_to_strike returned invalid K=%.6f for delta=%.2f, "
                "falling back to forward", i + 1, K if not np.isnan(K) else 0, target_delta)
            F = fx_forward(S, r_d, r_f, leg_T)
            K = F

        price_per_unit = float(_gk_price(S, K, leg_T, r_d, r_f, vol, cp_sign))
        greeks = _gk_greeks(S, K, leg_T, r_d, r_f, vol, cp_sign)

        total_premium = price_per_unit * notional * side_sign * ratio
        prem_pips = price_per_unit / pip_size if pip_size > 0 else 0.0
        prem_pct = price_per_unit / S * 100.0 if S > 0 else 0.0

        processed.append({
            "leg_num": i + 1,
            "cp": cp_str,
            "cp_sign": cp_sign,
            "side": side_str,
            "side_sign": side_sign,
            "delta_input": delta_abs,
            "ratio": ratio,
            "tenor_mult": tenor_mult,
            "strike": K,
            "vol": vol,
            "price_unit": price_per_unit,
            "premium_total": total_premium,
            "premium_pips": prem_pips * side_sign * ratio,
            "premium_pct": prem_pct * side_sign * ratio,
            "delta": greeks["delta"] * side_sign * ratio,
            "gamma": greeks["gamma"] * side_sign * ratio,
            "vega": greeks["vega"] * side_sign * ratio,
            "theta": greeks["theta"] * side_sign * ratio,
            "rho_d": greeks["rho_d"] * side_sign * ratio,
            "rho_f": greeks["rho_f"] * side_sign * ratio,
            "vanna": greeks["vanna"] * side_sign * ratio,
            "volga": greeks["volga"] * side_sign * ratio,
            "S": S, "T": leg_T, "r_d": r_d, "r_f": r_f,
        })

    return processed


# ============================================================================
# Aggregate Calculations
# ============================================================================

def _compute_aggregates(processed_legs, S, T, r_d, r_f, notional, pip_size):
    """Compute net premium, net Greeks, breakevens, max P&L, POP."""
    net_premium = sum(lg["premium_total"] for lg in processed_legs)
    net_premium_pips = sum(lg["premium_pips"] for lg in processed_legs)
    net_delta = sum(lg["delta"] for lg in processed_legs)
    net_gamma = sum(lg["gamma"] for lg in processed_legs)
    net_vega = sum(lg["vega"] for lg in processed_legs)
    net_theta = sum(lg["theta"] for lg in processed_legs)
    net_vanna = sum(lg["vanna"] for lg in processed_legs)
    net_volga = sum(lg["volga"] for lg in processed_legs)

    spot_range = np.linspace(S * 0.70, S * 1.30, 600)
    expiry_pnl = np.zeros_like(spot_range)
    for lg in processed_legs:
        cp = lg["cp_sign"]
        K = lg["strike"]
        qty = lg["side_sign"] * lg["ratio"] * notional
        prem = lg["price_unit"] * abs(qty)
        intrinsic = np.maximum(cp * (spot_range - K), 0.0) * qty
        if lg["side_sign"] > 0:
            intrinsic -= prem
        else:
            intrinsic += prem
        expiry_pnl += intrinsic

    # Breakeven(s)
    breakevens = []
    for j in range(1, len(spot_range)):
        if expiry_pnl[j - 1] * expiry_pnl[j] < 0:
            x0, x1 = spot_range[j - 1], spot_range[j]
            y0, y1 = expiry_pnl[j - 1], expiry_pnl[j]
            be = x0 - y0 * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x0
            breakevens.append(be)

    max_profit = float(np.max(expiry_pnl))
    max_loss = float(np.min(expiry_pnl))

    # Probability of profit (log-normal distribution)
    atm_vol = 0.10
    if processed_legs:
        atm_vol = max(processed_legs[0].get("vol", 0.10), 0.01)
    sigma_T = atm_vol * np.sqrt(max(T, 1e-4))
    mu_T = (r_d - r_f - 0.5 * atm_vol ** 2) * T
    log_spots = np.log(np.maximum(spot_range, 1e-10) / max(S, 1e-10))
    pdf_vals = np.exp(-0.5 * ((log_spots - mu_T) / sigma_T) ** 2) / (sigma_T * np.sqrt(2 * np.pi) * spot_range)
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    total_area = _trapz(pdf_vals, spot_range)
    if total_area > 0:
        pdf_vals /= total_area
    profitable_mask = expiry_pnl > 0
    pop = float(_trapz(pdf_vals * profitable_mask, spot_range) * 100.0)
    pop = max(0.0, min(100.0, pop))

    return {
        "net_premium": net_premium,
        "net_premium_pips": net_premium_pips,
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_vega": net_vega,
        "net_theta": net_theta,
        "net_vanna": net_vanna,
        "net_volga": net_volga,
        "breakevens": breakevens,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "pop": pop,
        "spot_range": spot_range,
        "expiry_pnl": expiry_pnl,
    }


# ============================================================================
# Expected Value Calculator
# ============================================================================

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _compute_expected_value(processed_legs, S, T, r_d, r_f, pair, tenor,
                            notional, pip_size, vol_surface):
    """Compute EV using BOTH implied (risk-neutral) and historical densities.

    - Implied EV uses Breeden-Litzenberger PDF from vol smile (should be ~0).
    - Historical EV uses log-normal with realised vol parameters.
    The historical EV is the meaningful number — "if the future looks like
    the past, what's my expected P&L?"
    """
    try:
        # Structure payoff at a grid of spot levels
        spot_lo = S * 0.70
        spot_hi = S * 1.30
        n_pts = 400
        spot_grid = np.linspace(spot_lo, spot_hi, n_pts)

        net_prem = sum(lg["price_unit"] * lg["side_sign"] * lg["ratio"]
                       for lg in processed_legs)
        payoff = np.zeros(n_pts)
        for lg in processed_legs:
            cp = lg["cp_sign"]
            K = lg["strike"]
            qty = lg["side_sign"] * lg["ratio"]
            payoff += np.maximum(cp * (spot_grid - K), 0.0) * qty
        pnl = (payoff - net_prem) * notional

        # ── Historical density (physical measure) ──
        # Use the first leg's vol as proxy for realised vol.
        # A more accurate approach would use actual RV, but this is fast
        # and available even without Bloomberg historical data.
        atm_vol = max(processed_legs[0].get("vol", 0.08), 0.01) if processed_legs else 0.08
        # Try to get actual realised vol from iv_rv
        try:
            ivrv = iv_rv_percentile(pair, tenor)
            rv = ivrv.get("current_rv", 0) / 100.0 if ivrv else 0
            if rv > 0.005:
                hist_vol = rv
            else:
                hist_vol = atm_vol  # fallback to implied
        except Exception:
            hist_vol = atm_vol

        mu_T = (r_d - r_f - 0.5 * hist_vol ** 2) * T
        sigma_T = hist_vol * np.sqrt(max(T, 1e-6))

        # Log-normal PDF: f(S_T) = (1/(S_T*sigma_T*sqrt(2pi))) * exp(-(ln(S_T/S)-mu_T)^2/(2*sigma_T^2))
        log_s = np.log(np.maximum(spot_grid, 1e-10) / max(S, 1e-10))
        hist_pdf = np.exp(-0.5 * ((log_s - mu_T) / max(sigma_T, 1e-8)) ** 2) / (
            max(sigma_T, 1e-8) * np.sqrt(2 * np.pi) * np.maximum(spot_grid, 1e-10))
        hist_total = _trapz(hist_pdf, spot_grid)
        if hist_total > 0:
            hist_pdf = hist_pdf / hist_total

        hist_ev = float(_trapz(pnl * hist_pdf, spot_grid))
        hist_pop = float(_trapz(hist_pdf * (pnl > 0), spot_grid) * 100)

        premium_abs = abs(net_prem * notional)
        hist_ev_pct = (hist_ev / premium_abs * 100) if premium_abs > 1 else 0.0

        # ── Implied density (for probability overlay on payoff chart) ──
        impl_pdf = None
        impl_strikes = None
        try:
            pdf_data = smile_implied_pdf(pair, tenor)
            if pdf_data is not None and not pdf_data.empty:
                impl_strikes = pdf_data["strike"].values
                impl_pdf_raw = pdf_data["pdf"].values
                impl_total = _trapz(impl_pdf_raw, impl_strikes)
                if impl_total > 0:
                    impl_pdf = impl_pdf_raw / impl_total
        except Exception:
            pass

        return {
            "ev": hist_ev,
            "ev_pips": hist_ev / (notional * pip_size) if (notional * pip_size) > 0 else 0,
            "ev_pct": hist_ev_pct,
            "prob_profit": hist_pop,
            "expected_profit": float(_trapz(pnl * hist_pdf * (pnl > 0), spot_grid)),
            "expected_loss": float(_trapz(pnl * hist_pdf * (pnl < 0), spot_grid)),
            "edge": "POSITIVE" if hist_ev > 0 else "NEGATIVE",
            "edge_color": COLORS["accent_green"] if hist_ev > 0 else COLORS["accent_red"],
            "hist_vol_used": hist_vol,
            "pdf_strikes": impl_strikes if impl_strikes is not None else spot_grid,
            "pdf_vals": impl_pdf if impl_pdf is not None else hist_pdf,
            # Additional data for EV sensitivity & density overlay
            "spot_grid": spot_grid,
            "pnl_grid": pnl,
            "hist_pdf": hist_pdf,
            "hist_pdf_strikes": spot_grid,
            "S": S, "T": T, "r_d": r_d, "r_f": r_f,
        }
    except Exception as exc:
        logger.debug("EV calculation failed: %s", exc)
        return None


# ============================================================================
# Suggestion Engine — scan vol surface and suggest optimal structures
# ============================================================================

def _classify_regime(atm_pct, rr_pct, bf_pct, term_pct):
    """Combine signal percentiles into a named vol regime."""
    a = atm_pct or 50
    r = rr_pct or 50
    b = bf_pct or 50
    t = term_pct or 50

    if a > 80 and abs(r - 50) > 25:
        return {"name": "CRASH FEAR", "color": COLORS["accent_red"],
                "desc": "Vol elevated + skew extreme — risk-off"}
    if a < 20 and b < 25:
        return {"name": "COMPLACENT", "color": COLORS["accent_green"],
                "desc": "Vol + wings historically cheap — opportunity to buy"}
    if a < 30:
        return {"name": "VOL CHEAP", "color": COLORS["accent_green"],
                "desc": "ATM vol near lows — long vol setups attractive"}
    if a > 70:
        return {"name": "VOL RICH", "color": COLORS["accent_orange"],
                "desc": "ATM vol elevated — short vol may have edge"}
    if t > 80:
        return {"name": "EVENT RISK", "color": COLORS["accent_orange"],
                "desc": "Inverted term structure — near-term risk priced"}
    return {"name": "NORMAL", "color": COLORS["text_secondary"],
            "desc": "No extreme signals — standard conditions"}


def _build_suggestions(pair, tenor, vol_surface, spots, rates):
    """Scan current vol surface and generate ranked trade suggestions."""
    suggestions = []

    # ATM vol percentile
    atm_info = vol_percentile(pair, tenor, "ATM", 252)
    atm_pct = atm_info.get("percentile", 50) if atm_info else 50

    # 25D RR percentile
    rr_info = vol_percentile(pair, tenor, "25D_RR", 252)
    rr_pct = rr_info.get("percentile", 50) if rr_info else 50
    rr_val = rr_info.get("current", 0) if rr_info else 0

    # 25D BF percentile
    bf_info = vol_percentile(pair, tenor, "25D_BF", 252)
    bf_pct = bf_info.get("percentile", 50) if bf_info else 50

    # IV-RV spread
    ivrv = iv_rv_percentile(pair, tenor)
    ivrv_pct = ivrv.get("percentile", 50) if ivrv else 50

    # Term structure: 1M vs 1Y ATM
    term_pct = 50
    atm_1m = vol_percentile(pair, "1M", "ATM", 252)
    atm_1y = vol_percentile(pair, "1Y", "ATM", 252)
    if atm_1m and atm_1y:
        spread = (atm_1m.get("current", 0) or 0) - (atm_1y.get("current", 0) or 0)
        term_pct = 80 if spread > 1.0 else (20 if spread < -2.0 else 50)

    # Regime
    regime = _classify_regime(atm_pct, rr_pct, bf_pct, term_pct)

    # Generate suggestions based on signals
    if atm_pct < 25:
        suggestions.append({
            "signal": f"ATM vol at {_ordinal(atm_pct)} %ile — CHEAP",
            "structures": ["Straddle", "Strangle", "25D Strangle"],
            "rationale": "Buy vol when historically cheap",
            "color": COLORS["accent_green"],
        })
    elif atm_pct > 75:
        suggestions.append({
            "signal": f"ATM vol at {_ordinal(atm_pct)} %ile — EXPENSIVE",
            "structures": ["Iron Condor", "Butterfly", "Iron Butterfly"],
            "rationale": "Sell vol when historically rich",
            "color": COLORS["accent_red"],
        })

    if rr_pct > 75:
        direction = "Puts expensive" if rr_val < 0 else "Calls expensive"
        suggestions.append({
            "signal": f"25D RR at {_ordinal(rr_pct)} %ile — {direction}",
            "structures": ["Risk Reversal", "Seagull", "Collar"],
            "rationale": "Sell expensive side of skew",
            "color": COLORS["accent_orange"],
        })
    elif rr_pct < 25:
        suggestions.append({
            "signal": f"25D RR at {_ordinal(rr_pct)} %ile — Skew flat",
            "structures": ["Risk Reversal", "25D Risk Reversal"],
            "rationale": "Buy protection cheaply when skew is flat",
            "color": COLORS["accent_green"],
        })

    if bf_pct < 20:
        suggestions.append({
            "signal": f"Wings (BF25) at {_ordinal(bf_pct)} %ile — CHEAP",
            "structures": ["Strangle", "10D Strangle", "Iron Butterfly"],
            "rationale": "Wings historically cheap — buy convexity",
            "color": COLORS["accent_green"],
        })

    if ivrv_pct > 80:
        suggestions.append({
            "signal": f"IV-RV spread at {_ordinal(ivrv_pct)} %ile — IV RICH",
            "structures": ["Iron Condor", "Butterfly", "Calendar Spread"],
            "rationale": "IV overpriced vs realised — sell premium",
            "color": COLORS["accent_orange"],
        })
    elif ivrv_pct < 20:
        suggestions.append({
            "signal": f"IV-RV spread at {_ordinal(ivrv_pct)} %ile — IV CHEAP",
            "structures": ["Straddle", "Strangle"],
            "rationale": "IV underpriced vs realised — buy premium",
            "color": COLORS["accent_green"],
        })

    if term_pct > 75:
        suggestions.append({
            "signal": "Term structure inverted — near-term event risk",
            "structures": ["Calendar Spread", "Diagonal Spread"],
            "rationale": "Buy back-end / sell front-end to capture normalisation",
            "color": COLORS["accent_orange"],
        })

    return suggestions, regime, {
        "atm_pct": atm_pct, "rr_pct": rr_pct, "bf_pct": bf_pct,
        "ivrv_pct": ivrv_pct, "term_pct": term_pct,
    }


# ============================================================================
# Multi-Tenor Scan
# ============================================================================

def _build_tenor_scan(legs_config, pair, notional, spots, rates, vol_surface):
    """Price the same structure across multiple tenors for comparison."""
    spot_data = spots.get(pair, {"mid": 1.0})
    S = spot_data.get("mid", spot_data.get("bid", 1.0))
    r_d = rates.get("r_dom", 0.03)
    r_f = rates.get("r_for", 0.02)
    pip_size = FX_PAIR_REGISTRY[pair].pip if pair in FX_PAIR_REGISTRY else 0.0001

    rows = []
    for t in SCAN_TENORS:
        try:
            T_val = tenor_to_years(t)
            proc = _process_legs(legs_config, pair, t, notional, spot_data, rates, vol_surface)
            agg = _compute_aggregates(proc, S, T_val, r_d, r_f, notional, pip_size)
            vp = vol_percentile(pair, t, "ATM", 252)
            pct = vp.get("percentile", 50) if vp else 50
            rows.append({
                "tenor": t, "premium_pips": agg["net_premium_pips"],
                "pop": agg["pop"], "theta_day": agg["net_theta"] * notional,
                "theta_pips": agg["net_theta"] / pip_size if pip_size > 0 else 0,
                "vol_pctile": pct,
                "be": agg["breakevens"][0] if agg["breakevens"] else None,
                "max_loss": agg["max_loss"],
            })
        except Exception:
            rows.append({"tenor": t, "premium_pips": 0, "pop": 0,
                         "theta_day": 0, "theta_pips": 0, "vol_pctile": 50,
                         "be": None, "max_loss": 0})
    return rows


# ============================================================================
# Solver — find parameter value that hits a target metric
# ============================================================================

def _solve_for_parameter(target_metric, target_value, solve_leg, solve_param,
                         legs_config, pair, tenor, notional, spot_data, rates,
                         vol_surface):
    """
    Use brentq to find the value of solve_param on solve_leg that makes
    target_metric == target_value.

    Returns: {"success": bool, "value": float, "message": str}
    """
    S = spot_data.get("mid", spot_data.get("bid", 1.0))
    T = tenor_to_years(tenor)
    r_d = rates.get("r_dom", 0.03)
    r_f = rates.get("r_for", 0.02)
    pip_size = FX_PAIR_REGISTRY[pair].pip if pair in FX_PAIR_REGISTRY else 0.0001

    if solve_param == "delta":
        bounds = (0.05, 0.95)
    elif solve_param == "ratio":
        bounds = (1.0, 10.0)
    else:
        return {"success": False, "value": None, "message": f"Unknown param: {solve_param}"}

    def objective(x):
        test_config = deepcopy(legs_config)
        if solve_param == "ratio":
            test_config[solve_leg]["ratio"] = max(1, round(x))
        else:
            test_config[solve_leg][solve_param] = float(x)
        proc = _process_legs(test_config, pair, tenor, notional, spot_data, rates, vol_surface)
        agg = _compute_aggregates(proc, S, T, r_d, r_f, notional, pip_size)
        metric_map = {
            "net_premium_pips": agg["net_premium_pips"],
            "net_premium": agg["net_premium"],
            "net_delta": agg["net_delta"],
            "net_vega": agg["net_vega"] * notional,
            "net_theta": agg["net_theta"] * notional,
            "pop": agg["pop"],
        }
        return metric_map.get(target_metric, 0) - target_value

    try:
        # Ratio is discrete — use brute-force integer search
        if solve_param == "ratio":
            best_r, best_err = 1, float("inf")
            for r in range(1, 11):
                err = abs(objective(r))
                if err < best_err:
                    best_r, best_err = r, err
            tol = max(abs(target_value) * 0.01, 0.5)
            if best_err < tol:
                return {"success": True, "value": best_r,
                        "message": f"Leg {solve_leg+1} ratio = {best_r}"}
            return {"success": False, "value": None,
                    "message": f"No integer ratio achieves target (best: ratio={best_r}, residual={best_err:.2f})"}

        # Delta is continuous — use brentq
        val_lo = objective(bounds[0])
        val_hi = objective(bounds[1])
        if val_lo * val_hi > 0:
            # No sign change — try minimize_scalar as fallback
            res = minimize_scalar(lambda x: abs(objective(x)),
                                  bounds=bounds, method="bounded",
                                  options={"maxiter": 80})
            residual = abs(objective(res.x))
            tol = max(abs(target_value) * 0.01, 0.5)
            if residual < tol:
                v = round(res.x, 4)
                return {"success": True, "value": v,
                        "message": f"Leg {solve_leg+1} {solve_param} = {v} (approx)"}
            return {"success": False, "value": None,
                    "message": f"No exact solution. Range: {val_lo+target_value:.2f} to {val_hi+target_value:.2f}"}

        result = brentq(objective, bounds[0], bounds[1], xtol=1e-5, maxiter=50)
        v = round(result, 4)
        return {"success": True, "value": v,
                "message": f"Leg {solve_leg+1} {solve_param} = {v}"}
    except Exception as exc:
        return {"success": False, "value": None, "message": f"Solver error: {exc}"}


# ============================================================================
# Unified Trade Analysis Panel (edge + scorecard + EV + scenarios)
# ============================================================================

def _detect_view(processed_legs, preset_name):
    """Auto-detect market view from structure or preset."""
    info = STRUCTURE_VIEWS.get(preset_name)
    if info and preset_name != "Custom":
        return info

    # Auto-detect for custom structures
    net_delta = sum(lg["delta"] for lg in processed_legs)
    net_vega = sum(lg["vega"] for lg in processed_legs)
    has_short = any(lg["side_sign"] < 0 for lg in processed_legs)

    if abs(net_delta) > 0.15:
        direction = "Bullish" if net_delta > 0 else "Bearish"
        return {"view": f"{direction} spot (Δ={net_delta:+.2f})", "type": "directional",
                "vol_view": "long_vol" if net_vega > 0 else "neutral"}
    if net_vega > 0.001:
        return {"view": "Long volatility", "type": "vol", "vol_view": "long_vol"}
    if net_vega < -0.001:
        return {"view": "Short volatility", "type": "vol", "vol_view": "short_vol"}
    return {"view": "Delta-neutral / carry", "type": "neutral", "vol_view": "neutral"}


def _generate_risks(processed_legs, agg, notional, S, T, pip_size):
    """Auto-generate risk warnings based on structure analysis."""
    risks = []

    # Check for naked short exposure
    net_call_ratio = sum(lg["ratio"] * lg["side_sign"]
                         for lg in processed_legs if lg["cp_sign"] > 0)
    net_put_ratio = sum(lg["ratio"] * lg["side_sign"]
                        for lg in processed_legs if lg["cp_sign"] < 0)

    if net_call_ratio < 0:
        risks.append(("danger", f"Net short {abs(net_call_ratio)} call(s) — unlimited upside risk"))
    if net_put_ratio < 0:
        risks.append(("danger", f"Net short {abs(net_put_ratio)} put(s) — unlimited downside risk"))

    # Theta bleed
    daily_theta = agg["net_theta"] * notional
    if daily_theta < -10:
        net_prem = abs(agg["net_premium"])
        days_to_bleed = net_prem / abs(daily_theta) if abs(daily_theta) > 1e-6 else 999
        risks.append(("warning",
                       f"Theta bleed: {daily_theta:,.0f}/day. "
                       f"Premium bleeds out in ~{days_to_bleed:.0f} days"))

    # Max loss
    if agg["max_loss"] < -1e12:
        risks.append(("danger", "Potentially unlimited loss"))
    elif agg["max_loss"] < -notional * 0.05:
        risks.append(("warning", f"Max loss: {agg['max_loss']:,.0f}"))

    # Short gamma warning
    net_gamma = agg["net_gamma"]
    if net_gamma < -0.001:
        risks.append(("warning", "Short gamma — vulnerable to sharp spot moves"))

    return risks


def _build_scenario_table(processed_legs, S, T, r_d, r_f, notional):
    """Compute P&L at fixed spot shocks."""
    shocks = [-0.10, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.10]
    net_prem = sum(lg["price_unit"] * lg["side_sign"] * lg["ratio"]
                   for lg in processed_legs)
    results = []
    for ds in shocks:
        s_sh = S * (1.0 + ds)
        pnl = 0.0
        for lg in processed_legs:
            qty = lg["side_sign"] * lg["ratio"]
            pnl += float(_gk_price(s_sh, lg["strike"], T, r_d, r_f,
                                   lg["vol"], lg["cp_sign"])) * qty
        results.append({"shock": ds, "pnl": (pnl - net_prem) * notional})
    return results


# ============================================================================
# EV Sensitivity Curve — EV as a function of realized vol assumption
# ============================================================================

def _build_ev_sensitivity_chart(ev_data):
    """Small chart: EV vs assumed realized vol, with breakeven RV annotated."""
    if not ev_data or "pnl_grid" not in ev_data:
        return html.Div()

    spot_grid = ev_data["spot_grid"]
    pnl = ev_data["pnl_grid"]
    S = ev_data["S"]
    T = ev_data["T"]
    r_d = ev_data["r_d"]
    r_f = ev_data["r_f"]
    hist_vol = ev_data.get("hist_vol_used", 0.08)

    # Sweep RV from 30% of current to 250% of current (at least 2% to 30%)
    rv_lo = max(hist_vol * 0.3, 0.02)
    rv_hi = max(hist_vol * 2.5, 0.30)
    rv_range = np.linspace(rv_lo, rv_hi, 30)
    ev_vals = []

    for rv in rv_range:
        mu_T = (r_d - r_f - 0.5 * rv ** 2) * T
        sigma_T = rv * np.sqrt(max(T, 1e-6))
        log_s = np.log(np.maximum(spot_grid, 1e-10) / max(S, 1e-10))
        pdf = np.exp(-0.5 * ((log_s - mu_T) / max(sigma_T, 1e-8)) ** 2) / (
            max(sigma_T, 1e-8) * np.sqrt(2 * np.pi) * np.maximum(spot_grid, 1e-10))
        total = _trapz(pdf, spot_grid)
        if total > 0:
            pdf = pdf / total
        ev_vals.append(float(_trapz(pnl * pdf, spot_grid)))

    ev_arr = np.array(ev_vals)
    rv_pct = rv_range * 100

    # Find breakeven RV (where EV crosses zero)
    be_rv = None
    for i in range(1, len(ev_arr)):
        if ev_arr[i - 1] * ev_arr[i] < 0:
            # Linear interpolation
            x0, x1 = rv_pct[i - 1], rv_pct[i]
            y0, y1 = ev_arr[i - 1], ev_arr[i]
            be_rv = x0 - y0 * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x0
            break

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rv_pct, y=ev_arr, mode="lines",
        line=dict(color=COLORS["accent_cyan"], width=2),
        fill="tozeroy", fillcolor="rgba(255,136,0,0.04)",
        hovertemplate="RV: %{x:.1f}%<br>EV: %{y:+,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))

    # Mark current RV
    fig.add_vline(x=hist_vol * 100,
                  line=dict(color=COLORS["accent_orange"], width=1, dash="dash"),
                  annotation_text=f"Current RV {hist_vol*100:.1f}%",
                  annotation_font=dict(color=COLORS["accent_orange"], size=8))

    # Mark breakeven
    if be_rv is not None:
        fig.add_vline(x=be_rv,
                      line=dict(color=COLORS["accent_red"], width=1.5, dash="dashdot"),
                      annotation_text=f"BE RV {be_rv:.1f}%",
                      annotation_font=dict(color=COLORS["accent_red"], size=9),
                      annotation_position="top left")

    tpl = CHART_TEMPLATE["layout"]
    fig.update_layout(
        title=dict(text="EV SENSITIVITY TO REALIZED VOL", font=dict(color=COLORS["text_muted"], size=10)),
        xaxis_title="Assumed Realized Vol (%)", yaxis_title="Expected Value",
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=50, r=15, t=30, b=30),
        hoverlabel=tpl["hoverlabel"], height=200, showlegend=False,
        xaxis=dict(gridcolor="#1a1a30"),
        yaxis=dict(gridcolor="#1a1a30"),
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False},
                     style={"marginBottom": "8px"})


# ============================================================================
# Named Scenarios — joint spot + vol + time shocks
# ============================================================================

NAMED_SCENARIOS = [
    {"name": "Risk-off",         "spot": -0.03, "vol_mult": +0.40, "dt": 0},
    {"name": "Risk-on rally",    "spot": +0.02, "vol_mult": -0.15, "dt": 0},
    {"name": "Vol normalization", "spot": 0,     "vol_mult": "mean", "dt": 0},
    {"name": "Post-event crush", "spot": 0,     "vol_mult": -0.25,  "dt": 5 / 365},
    {"name": "Spot stress ↑",    "spot": +0.05, "vol_mult": +0.20, "dt": 0},
    {"name": "Spot stress ↓",    "spot": -0.05, "vol_mult": +0.30, "dt": 0},
    {"name": "Time decay (1W)",  "spot": 0,     "vol_mult": 0,      "dt": 7 / 365},
    {"name": "Time decay (1M)",  "spot": 0,     "vol_mult": 0,      "dt": 30 / 365},
    {"name": "Benign carry",     "spot": +0.005, "vol_mult": -0.05, "dt": 30 / 365},
]


def _build_named_scenarios(processed_legs, S, T, r_d, r_f, notional, atm_vol,
                            pair=None, tenor=None):
    """Compute P&L under named multi-factor scenarios."""
    net_prem = sum(lg["price_unit"] * lg["side_sign"] * lg["ratio"]
                   for lg in processed_legs)

    # Get mean vol for "mean revert" scenario
    mean_vol = atm_vol
    try:
        vp = vol_percentile(pair, tenor or "3M", "ATM", 252)
        if vp and vp.get("mean"):
            mean_vol = vp["mean"] / 100.0
    except Exception:
        pass

    results = []
    for sc in NAMED_SCENARIOS:
        # Skip time-decay scenarios that exceed remaining life
        if sc["dt"] > 0 and sc["dt"] >= T * 0.95:
            continue
        s_new = S * (1.0 + sc["spot"])
        t_new = max(T - sc["dt"], 1e-6)

        # Vol shock: multiplicative or mean-revert
        if sc["vol_mult"] == "mean":
            vol_ratio = mean_vol / max(atm_vol, 1e-6)
        else:
            vol_ratio = 1.0 + sc["vol_mult"]

        pnl_sum = 0.0
        for lg in processed_legs:
            vol_new = max(lg["vol"] * vol_ratio, 0.005)
            qty = lg["side_sign"] * lg["ratio"]
            pnl_sum += float(_gk_price(s_new, lg["strike"], t_new, r_d, r_f,
                                        vol_new, lg["cp_sign"])) * qty
        pnl_total = (pnl_sum - net_prem) * notional

        # Format vol description
        if sc["vol_mult"] == "mean":
            vol_desc = f"→{mean_vol*100:.1f}%"
        elif sc["vol_mult"] == 0:
            vol_desc = "unch"
        else:
            vol_desc = f"{sc['vol_mult']:+.0%}"

        time_desc = f"-{sc['dt']*365:.0f}d" if sc["dt"] > 0 else "now"

        results.append({
            "name": sc["name"],
            "spot_desc": f"{sc['spot']:+.1%}" if sc["spot"] != 0 else "unch",
            "vol_desc": vol_desc,
            "time_desc": time_desc,
            "pnl": pnl_total,
        })
    return results


# ============================================================================
# Structure Efficiency Comparison — rank alternative structures
# ============================================================================

# Map vol views to comparable preset structures
_VIEW_ALTERNATIVES = {
    "long_vol": ["Straddle", "Strangle", "25D Strangle", "10D Strangle", "Calendar Spread"],
    "short_vol": ["Butterfly", "Iron Butterfly", "Iron Condor", "Broken Wing Butterfly"],
    "skew": ["Risk Reversal", "25D Risk Reversal", "Seagull", "Collar"],
    "neutral": ["Call Spread", "Put Spread", "Collar", "Fence", "Iron Condor"],
    "term_structure": ["Calendar Spread", "Diagonal Spread"],
    "unknown": [],
}


def _build_efficiency_table(processed_legs, agg, ev_data, preset_name, view_info,
                             pair, tenor, notional, spot_data, rates, vol_surface,
                             S, T, r_d, r_f, pip_size):
    """Auto-generate alternative structures and compare efficiency metrics."""
    tpl_font = {"fontFamily": "'JetBrains Mono', monospace"}
    vol_view = view_info.get("vol_view", "unknown")
    alternatives = _VIEW_ALTERNATIVES.get(vol_view, [])

    # Remove current preset from alternatives
    alternatives = [a for a in alternatives if a != preset_name][:4]
    if not alternatives:
        return html.Div()

    # Current structure metrics
    current_prem = agg.get("net_premium_pips", 0)
    current_vega = agg.get("net_vega", 0) * notional

    rows = []

    def _safe_vega_per_pip(vega, prem_pips):
        """Vega per pip of premium; 'N/A' for zero-cost structures."""
        if abs(prem_pips) < 0.1:
            return None  # zero-cost structure
        return abs(vega / prem_pips)

    # Current structure row
    rows.append({
        "name": preset_name or "Current",
        "premium": current_prem,
        "max_loss": agg.get("max_loss", 0),
        "be": agg["breakevens"][0] if agg.get("breakevens") and len(agg["breakevens"]) > 0 else None,
        "vega_per_pip": _safe_vega_per_pip(current_vega, current_prem),
        "gamma_theta": min(abs(agg["net_gamma"] / max(abs(agg["net_theta"]), 1e-12)), 999.9),
        "pop": agg.get("pop", 0),
        "ev": ev_data["ev"] if ev_data else 0,
        "is_current": True,
    })

    # Price each alternative
    for alt_name in alternatives:
        try:
            alt_legs = PRESETS.get(alt_name)
            if not alt_legs:
                continue
            alt_proc = _process_legs(alt_legs, pair, tenor, notional, spot_data, rates, vol_surface)
            if not alt_proc:
                continue
            alt_agg = _compute_aggregates(alt_proc, S, T, r_d, r_f, notional, pip_size)
            alt_vega = alt_agg.get("net_vega", 0) * notional
            alt_prem = alt_agg.get("net_premium_pips", 0)

            rows.append({
                "name": alt_name,
                "premium": alt_prem,
                "max_loss": alt_agg.get("max_loss", 0),
                "be": alt_agg["breakevens"][0] if alt_agg.get("breakevens") and len(alt_agg["breakevens"]) > 0 else None,
                "vega_per_pip": _safe_vega_per_pip(alt_vega, alt_prem),
                "gamma_theta": abs(alt_agg["net_gamma"] / max(abs(alt_agg["net_theta"]), 1e-12)),
                "pop": alt_agg.get("pop", 0),
                "ev": 0,  # Skip full EV for performance
                "is_current": False,
            })
        except Exception:
            continue

    if len(rows) < 2:
        return html.Div()

    # Build HTML table
    hdr_s = {"color": COLORS["text_muted"], "fontSize": "8px", "fontWeight": "600",
             "padding": "3px 5px", "textTransform": "uppercase", **tpl_font,
             "borderBottom": f"1px solid {COLORS['border_subtle']}"}
    cell_s = {"color": COLORS["text_primary"], "fontSize": "9px", "padding": "3px 5px",
              **tpl_font, "borderBottom": f"1px solid {COLORS['border_subtle']}"}

    header = html.Tr([html.Th(h, style=hdr_s) for h in
                       ["STRUCTURE", "PREM (p)", "MAX LOSS", "BREAKEVEN",
                        "VEGA/$", "γ/θ", "POP", "EV"]])
    body = []
    for r in rows:
        highlight = {"backgroundColor": "rgba(255,136,0,0.06)"} if r["is_current"] else {}
        be_str = f"{r['be']:.5f}" if r["be"] else "—"
        ev_str = f"{r['ev']:+,.0f}" if r["ev"] != 0 else "—"
        body.append(html.Tr([
            html.Td(r["name"], style={**cell_s, **highlight,
                     "fontWeight": "700" if r["is_current"] else "400",
                     "color": COLORS["accent_cyan"] if r["is_current"] else COLORS["text_primary"]}),
            html.Td(f"{r['premium']:.1f}p", style={**cell_s, **highlight, "textAlign": "right"}),
            html.Td(f"{r['max_loss']:,.0f}" if r["max_loss"] > -1e12 else "UNLIM",
                     style={**cell_s, **highlight, "textAlign": "right"}),
            html.Td(be_str, style={**cell_s, **highlight, "textAlign": "right"}),
            html.Td(f"{r['vega_per_pip']:.0f}" if r['vega_per_pip'] is not None else "N/A",
                     style={**cell_s, **highlight, "textAlign": "right"}),
            html.Td(f"{r['gamma_theta']:.1f}", style={**cell_s, **highlight, "textAlign": "right"}),
            html.Td(f"{r['pop']:.0f}%", style={**cell_s, **highlight, "textAlign": "right"}),
            html.Td(ev_str, style={**cell_s, **highlight, "textAlign": "right"}),
        ]))

    return html.Div([
        html.Div("STRUCTURE COMPARISON", style={"color": COLORS["text_muted"], "fontSize": "9px",
                 "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1px",
                 "marginBottom": "4px", **tpl_font}),
        html.Table([html.Thead(header), html.Tbody(body)],
                   style={"width": "100%", "borderCollapse": "collapse"}),
    ], style={"marginTop": "8px"})


# ============================================================================
# Delta-to-metric mapper for per-leg edge analysis
# ============================================================================

def _delta_to_vol_metric(delta_input, cp_sign):
    """Map a leg's delta to the most relevant vol surface metric for percentile ranking.
    Wings (25D and below) use BF percentile — captures wing richness.
    Near-ATM legs use ATM percentile."""
    d = abs(delta_input)
    if 0.40 <= d <= 0.60:
        return "ATM", "ATM"
    elif 0.20 <= d < 0.40:
        return "25D_BF", "25D wing"
    elif d < 0.20:
        return "25D_BF", "wing"
    return "ATM", "ATM"


def _build_trade_analysis(processed_legs, agg, ev_data, pair, tenor, notional,
                          preset_name, S, T, r_d, r_f, pip_size, vol_surface,
                          spot_data=None, rates=None, atm_vol=0.10):
    """Build the unified Trade Analysis panel (EV + edge + scenarios + risks)."""
    tpl_font = {"fontFamily": "'JetBrains Mono', monospace"}
    section_style = {"marginBottom": "12px"}
    label_s = {"color": COLORS["text_muted"], "fontSize": "9px",
               "textTransform": "uppercase", "letterSpacing": "1px", **tpl_font}
    val_s = {"fontSize": "14px", "fontWeight": "700", **tpl_font}

    # ── View detection ──
    view_info = _detect_view(processed_legs, preset_name)

    # ── EV display ──
    ev_boxes = []
    if ev_data:
        ev_boxes = [
            html.Div([
                html.Div(f"{ev_data['ev']:+,.0f}", style={**val_s, "color": ev_data["edge_color"]}),
                html.Div("EXPECTED VALUE", style=label_s),
            ], style={**make_stat_style(ev_data["edge_color"]), "flex": "1", "minWidth": "100px"}),
            html.Div([
                html.Div(f"{ev_data['ev_pct']:+.1f}%", style={**val_s, "color": ev_data["edge_color"]}),
                html.Div("EV % OF PREMIUM", style=label_s),
            ], style={**make_stat_style(ev_data["edge_color"]), "flex": "1", "minWidth": "100px"}),
            html.Div([
                html.Div(f"{ev_data['prob_profit']:.1f}%" if np.isfinite(ev_data.get('prob_profit', 0)) else "—", style={**val_s, "color": COLORS["accent_cyan"]}),
                html.Div("PROB OF PROFIT (SMILE)", style=label_s),
            ], style={**make_stat_style(COLORS["accent_cyan"]), "flex": "1", "minWidth": "100px"}),
            html.Div([
                html.Div(ev_data["edge"], style={**val_s, "color": ev_data["edge_color"]}),
                html.Div("EDGE", style=label_s),
            ], style={**make_stat_style(ev_data["edge_color"]), "flex": "1", "minWidth": "100px"}),
        ]

    # ── Efficiency metrics ──
    net_vega_scaled = agg["net_vega"] * notional
    net_theta_scaled = agg["net_theta"] * notional
    net_gamma_scaled = agg["net_gamma"] * notional
    prem_paid = abs(agg.get("net_premium_pips", 0))

    cost_per_vega = prem_paid / abs(net_vega_scaled) if abs(net_vega_scaled) > 1e-6 else 0
    theta_gamma = abs(net_theta_scaled) / abs(net_gamma_scaled) if abs(net_gamma_scaled) > 1e-6 else 0

    efficiency_boxes = [
        html.Div([
            html.Div(f"{cost_per_vega:.2f}p" if np.isfinite(cost_per_vega) else "—", style={**val_s, "color": COLORS["accent_blue"]}),
            html.Div("COST / VEGA", style=label_s),
        ], style={**make_stat_style(COLORS["accent_blue"]), "flex": "1", "minWidth": "100px"}),
        html.Div([
            html.Div(f"{theta_gamma:.1f}" if np.isfinite(theta_gamma) else "—", style={**val_s, "color": COLORS["accent_orange"]}),
            html.Div("\u03b8/\u03b3 RATIO", style=label_s),
        ], style={**make_stat_style(COLORS["accent_orange"]), "flex": "1", "minWidth": "100px"}),
    ]

    # ── Per-leg edge (delta-specific percentile) ──
    leg_rows = []
    for lg in processed_legs:
        leg_tenor = years_to_nearest_tenor(lg["T"]) if lg.get("T") else tenor
        metric, metric_label = _delta_to_vol_metric(lg.get("delta_input", 0.5), lg["cp_sign"])
        vp = vol_percentile(pair, leg_tenor, metric, 252)
        if not vp:
            vp = vol_percentile(pair, leg_tenor, "ATM", 252)
            metric_label = "ATM level"
        pct = vp.get("percentile", 50) if vp else 50
        pct_color = (COLORS["accent_green"] if pct < 25
                     else COLORS["accent_red"] if pct > 75
                     else COLORS["text_secondary"])
        cheap_label = ("CHEAP" if pct < 25 else "EXPENSIVE" if pct > 75
                       else "BELOW AVG" if pct < 50 else "ABOVE AVG")
        action = "Buying" if lg["side_sign"] > 0 else "Selling"
        leg_rows.append(html.Div(
            f"L{lg['leg_num']} {action} {lg['cp'].upper()} Δ{lg['delta_input']:.0%} "
            f"| vol {lg['vol']*100:.1f}% | {metric_label} {_ordinal(pct)} %ile ({cheap_label})",
            style={"color": pct_color, "fontSize": "10px", **tpl_font, "marginBottom": "2px"},
        ))

    # ── Breakeven RV ──
    be_section = html.Div()
    try:
        days = tenor_to_days(tenor)
        be = breakeven_vol(pair, tenor, days)
        if be:
            gap = be.get("atm_iv", 0) - be.get("breakeven_rv", 0)
            be_section = html.Div(
                f"Breakeven RV: {be.get('breakeven_rv', 0):.1f}% | ATM IV: {be.get('atm_iv', 0):.1f}% "
                f"| Cushion: {gap:+.1f} vol pts",
                style={"color": COLORS["accent_cyan"] if gap > 0 else COLORS["accent_orange"],
                       "fontSize": "10px", **tpl_font, "marginTop": "4px"},
            )
    except Exception:
        pass

    # ── Carry analysis ──
    carry_section = html.Div()
    daily_theta = agg["net_theta"] * notional
    if abs(daily_theta) > 1e-6:
        theta_pips = agg["net_theta"] / pip_size if pip_size > 0 else 0
        prem_abs = abs(agg["net_premium"])
        days_bleed = prem_abs / abs(daily_theta) if abs(daily_theta) > 1e-6 else 999
        carry_section = html.Div(
            f"Theta: {daily_theta:+,.0f}/day ({theta_pips:+.1f} pips/day) | "
            f"Premium bleeds in ~{days_bleed:.0f} days",
            style={"color": COLORS["accent_orange"] if daily_theta < 0 else COLORS["accent_green"],
                   "fontSize": "10px", **tpl_font, "marginTop": "4px"},
        )

    # ── EV sensitivity chart ──
    ev_sensitivity = None
    try:
        chart = _build_ev_sensitivity_chart(ev_data)
        if chart and not isinstance(chart, html.Div):
            ev_sensitivity = chart
    except Exception:
        pass

    # ── Named scenario table ──
    named_scenarios = []
    try:
        named_scenarios = _build_named_scenarios(
            processed_legs, S, T, r_d, r_f, notional, atm_vol, pair, tenor)
    except Exception:
        pass

    sc_header = html.Tr([html.Th(h, style={**label_s, "padding": "2px 4px", "fontSize": "7px"})
                          for h in ["SCENARIO", "SPOT", "VOL", "TIME", "P&L"]])
    sc_rows = []
    for sc in named_scenarios:
        pnl = sc["pnl"]
        color = COLORS["accent_green"] if pnl > 0 else COLORS["accent_red"] if pnl < 0 else COLORS["text_muted"]
        sc_rows.append(html.Tr([
            html.Td(sc["name"], style={"color": COLORS["text_primary"], "fontSize": "9px",
                     "padding": "2px 4px", **tpl_font}),
            html.Td(sc["spot_desc"], style={"color": COLORS["text_secondary"], "fontSize": "9px",
                     "padding": "2px 4px", "textAlign": "center", **tpl_font}),
            html.Td(sc["vol_desc"], style={"color": COLORS["text_secondary"], "fontSize": "9px",
                     "padding": "2px 4px", "textAlign": "center", **tpl_font}),
            html.Td(sc["time_desc"], style={"color": COLORS["text_secondary"], "fontSize": "9px",
                     "padding": "2px 4px", "textAlign": "center", **tpl_font}),
            html.Td(f"{pnl:+,.0f}", style={"color": color, "fontSize": "9px", "fontWeight": "600",
                     "padding": "2px 4px", "textAlign": "right", **tpl_font}),
        ]))

    # ── Structure efficiency comparison ──
    efficiency_table = html.Div()
    try:
        if spot_data and rates:
            efficiency_table = _build_efficiency_table(
                processed_legs, agg, ev_data, preset_name, view_info,
                pair, tenor, notional, spot_data, rates, vol_surface,
                S, T, r_d, r_f, pip_size)
    except Exception:
        pass

    # ── Risk warnings ──
    risks = _generate_risks(processed_legs, agg, notional, S, T, pip_size)
    risk_items = []
    for level, msg in risks:
        icon = "\u26a0" if level == "warning" else "\u2716"
        color = COLORS["accent_orange"] if level == "warning" else COLORS["accent_red"]
        risk_items.append(html.Div(
            f"{icon} {msg}", style={"color": color, "fontSize": "10px", **tpl_font, "marginBottom": "2px"},
        ))
    if not risks:
        risk_items = [html.Div("\u2713 No major risks detected",
                               style={"color": COLORS["accent_green"], "fontSize": "10px", **tpl_font})]

    # ── Assemble ──
    return html.Div([
        # View + date header
        html.Div([
            html.Span("TRADE ANALYSIS", style={"color": COLORS["text_primary"],
                       "fontSize": "12px", "fontWeight": "700", "letterSpacing": "1.5px", **tpl_font}),
            html.Span(f"  |  {view_info['view']}  |  {datetime.now().strftime('%Y-%m-%d')}",
                       style={"color": COLORS["text_muted"], "fontSize": "10px", **tpl_font}),
        ], style={"marginBottom": "10px"}),

        # EV boxes
        html.Div(ev_boxes + efficiency_boxes, style={"display": "flex", "gap": "8px", "flexWrap": "wrap",
                                   "marginBottom": "10px"}) if ev_boxes else
        html.Div(efficiency_boxes, style={"display": "flex", "gap": "8px", "flexWrap": "wrap",
                                   "marginBottom": "10px"}),

        # Per-leg edge
        html.Div([
            html.Div("PER-LEG EDGE", style={**label_s, "marginBottom": "4px"}),
            *leg_rows,
            be_section,
            carry_section,
        ], style=section_style),

        # EV sensitivity
        html.Div([ev_sensitivity], style=section_style) if ev_sensitivity is not None else html.Div(),

        # Named scenario table
        html.Div([
            html.Div("SCENARIOS (spot + vol + time)", style={**label_s, "marginBottom": "4px"}),
            html.Table([html.Thead(sc_header), html.Tbody(sc_rows)],
                       style={"width": "100%", "borderCollapse": "collapse"})
            if sc_rows else html.Div("No scenario data", style={"color": COLORS["text_muted"],
                                      "fontSize": "9px", **tpl_font}),
        ], style=section_style),

        # Risks
        html.Div([
            html.Div("RISKS", style={**label_s, "marginBottom": "4px"}),
            *risk_items,
        ], style=section_style),

        # Structure efficiency comparison
        efficiency_table,
    ])


# ============================================================================
# Compare Mode Builders
# ============================================================================

def _build_compare_overlay(snapshot_a, snapshot_b, notional):
    """Build overlaid payoff chart + metrics comparison table."""
    tpl = CHART_TEMPLATE["layout"]
    fig = go.Figure()

    for snap, name, color, dash_style in [
        (snapshot_a, "A", COLORS["accent_cyan"], "solid"),
        (snapshot_b, "B", COLORS["accent_purple"], "dash"),
    ]:
        if snap is None:
            continue
        sr = np.array(snap["spot_range"])
        pnl = np.array(snap["expiry_pnl"])
        fig.add_trace(go.Scatter(
            x=sr, y=pnl, mode="lines",
            name=f"{name}: {snap.get('label', '?')}",
            line=dict(color=color, width=2.5 if dash_style == "solid" else 2, dash=dash_style),
        ))
        for be in snap.get("breakevens", []):
            fig.add_vline(x=be, line=dict(color=color, width=1, dash="dot"))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"))
    fig.update_layout(
        title=dict(text="COMPARE: PAYOFF OVERLAY", font=dict(color=COLORS["text_primary"], size=12)),
        xaxis_title="Spot", yaxis_title="P&L ($)",
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=55, r=15, t=35, b=35),
        legend=dict(font=dict(color=COLORS["text_secondary"], size=10), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=tpl["hoverlabel"], height=340,
        xaxis=dict(gridcolor="#1a1a30"),
        yaxis=dict(gridcolor="#1a1a30"),
    )

    # Metrics comparison table
    def _row(label, val_a, val_b, fmt=".1f", better="lower"):
        a = val_a or 0
        b = val_b or 0
        if better == "lower":
            a_better = a < b
        elif better == "lower_abs":
            a_better = abs(a) < abs(b)
        else:
            a_better = a > b
        b_better = not a_better
        ca = COLORS["accent_green"] if a_better else COLORS["text_secondary"]
        cb = COLORS["accent_green"] if b_better else COLORS["text_secondary"]
        cs = {"fontSize": "10px", "fontFamily": "'JetBrains Mono', monospace",
              "padding": "4px 8px", "borderBottom": f"1px solid {COLORS['border_subtle']}"}
        return html.Tr([
            html.Td(label, style={**cs, "color": COLORS["text_muted"], "width": "120px"}),
            html.Td(f"{a:{fmt}}", style={**cs, "color": ca, "textAlign": "right"}),
            html.Td(f"{b:{fmt}}", style={**cs, "color": cb, "textAlign": "right"}),
        ])

    a = snapshot_a or {}
    b = snapshot_b or {}
    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Metric", style={"color": COLORS["text_muted"], "fontSize": "9px", "padding": "4px 8px"}),
            html.Th("A", style={"color": COLORS["accent_cyan"], "fontSize": "9px", "padding": "4px 8px"}),
            html.Th("B", style={"color": COLORS["accent_purple"], "fontSize": "9px", "padding": "4px 8px"}),
        ])),
        html.Tbody([
            _row("Premium (pips)", a.get("prem_pips"), b.get("prem_pips"), ".1f", "lower"),
            _row("Prob of Profit (%)", a.get("pop"), b.get("pop"), ".1f", "higher"),
            _row("Net Delta", a.get("net_delta"), b.get("net_delta"), ".4f", "lower_abs"),
            _row("Net Vega", a.get("net_vega"), b.get("net_vega"), ",.0f", "higher"),
            _row("Daily Theta", a.get("daily_theta"), b.get("daily_theta"), ",.0f", "higher"),
            _row("Max Loss", a.get("max_loss"), b.get("max_loss"), ",.0f", "higher"),
        ]),
    ], style={"width": "100%", "borderCollapse": "collapse"})

    return html.Div([
        html.Div([
            dcc.Graph(figure=fig, style={"height": "340px"},
                      config={"displayModeBar": True, "scrollZoom": False}),
        ], style={"flex": "2", "minWidth": "400px"}),
        html.Div(table, style={"flex": "1", "minWidth": "250px", "padding": "8px"}),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})


# ============================================================================
# Chart Builders
# ============================================================================

def _build_payoff_chart(processed_legs, agg, S, T, r_d, r_f, notional, atm_vol,
                        ev_data=None):
    """Chart 1: Payoff diagram at expiry, T*0.5, and today.
    Vectorised — uses numpy array pricing instead of per-point loop.
    Optional ev_data adds probability density overlay."""
    tpl = CHART_TEMPLATE["layout"]
    # Adaptive range: ±2 implied moves (min ±5%, max ±30%)
    implied_move = atm_vol * np.sqrt(max(T, 1.0 / 365.0))
    half_range = max(0.05, min(2.5 * implied_move, 0.30))
    spot_range = np.linspace(S * (1 - half_range), S * (1 + half_range), 400)
    pnl_expiry = np.zeros_like(spot_range)
    pnl_half = np.zeros_like(spot_range)
    pnl_now = np.zeros_like(spot_range)

    net_prem_per_unit = sum(
        lg["price_unit"] * lg["side_sign"] * lg["ratio"]
        for lg in processed_legs
    )

    for lg in processed_legs:
        cp = lg["cp_sign"]
        K = lg["strike"]
        vol = lg["vol"]
        qty = lg["side_sign"] * lg["ratio"]

        intrinsic = np.maximum(cp * (spot_range - K), 0.0) * qty
        pnl_expiry += intrinsic

        if T > 0.001:
            # Vectorised: _gk_price accepts numpy arrays for S
            pnl_half += _gk_price(spot_range, K, max(T * 0.5, 1e-6), r_d, r_f, vol, cp) * qty
            pnl_now += _gk_price(spot_range, K, T, r_d, r_f, vol, cp) * qty

    pnl_expiry = (pnl_expiry - net_prem_per_unit) * notional
    pnl_half = (pnl_half - net_prem_per_unit) * notional
    pnl_now = (pnl_now - net_prem_per_unit) * notional

    fig = go.Figure()

    # 1-sigma expected move shading
    move_1s = S * atm_vol * np.sqrt(max(T, 1e-4))
    fig.add_vrect(x0=S - move_1s, x1=S + move_1s,
                  fillcolor="rgba(255,136,0,0.04)", line_width=0,
                  annotation_text="1\u03c3", annotation_position="top left",
                  annotation_font=dict(color=COLORS["text_muted"], size=9))

    # Expiry P&L (bold)
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_expiry, mode="lines",
        name="At Expiry", line=dict(color=COLORS["accent_cyan"], width=3),
        fill="tozeroy", fillcolor="rgba(255,136,0,0.04)",
        hovertemplate="Spot: %{x:.4f}<br>P&L: %{y:,.0f}<extra>At Expiry</extra>",
    ))
    # T*0.5
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_half, mode="lines",
        name="T\u00d70.5", line=dict(color=COLORS["accent_purple"], width=2, dash="dash"),
        hovertemplate="Spot: %{x:.4f}<br>P&L: %{y:,.0f}<extra>T\u00d70.5</extra>",
    ))
    # Today (thin)
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_now, mode="lines",
        name="Today", line=dict(color=COLORS["accent_orange"], width=1.5, dash="dot"),
        hovertemplate="Spot: %{x:.4f}<br>P&L: %{y:,.0f}<extra>Today</extra>",
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
    fig.add_vline(x=S, line=dict(color=COLORS["accent_blue"], width=1, dash="dash"),
                  annotation_text=f"Spot {S:.4f}",
                  annotation_font=dict(color=COLORS["accent_blue"], size=9))

    # Breakeven lines with distance from spot
    for be in agg["breakevens"]:
        if not np.isfinite(be):
            continue
        pct_from_spot = (be - S) / S * 100 if S > 0 else 0
        fig.add_vline(x=be, line=dict(color=COLORS["accent_orange"], width=1, dash="dashdot"),
                      annotation_text=f"BE {be:.4f} ({pct_from_spot:+.1f}%)",
                      annotation_font=dict(color=COLORS["accent_orange"], size=8))

    fig.update_layout(
        title=dict(text="PAYOFF DIAGRAM", font=dict(color=COLORS["text_primary"], size=13)),
        xaxis_title="Spot", yaxis_title="P&L ($)",
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=55, r=20, t=40, b=35),
        legend=dict(font=dict(color=COLORS["text_secondary"], size=10),
                    bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
        hoverlabel=tpl["hoverlabel"],
        xaxis=dict(gridcolor="#1a1a30"),
        yaxis=dict(gridcolor="#1a1a30"),
        height=380,
    )
    return fig


def _vectorized_greeks(spot_arr, K, T, r_d, r_f, sigma, cp):
    """Vectorised Greeks over a spot array. Returns dict of arrays."""
    T_s = max(float(T), 1e-10)
    sig = max(float(sigma), 1e-6)
    K_f = max(float(K), 1e-10)
    cp_f = float(cp)
    sqrt_T = np.sqrt(T_s)
    d1 = (np.log(spot_arr / K_f) + (r_d - r_f + 0.5 * sig ** 2) * T_s) / (sig * sqrt_T)
    d2 = d1 - sig * sqrt_T
    nd1 = _norm_cdf(cp_f * d1)
    npd1 = _norm_pdf(d1)
    exp_rf = np.exp(-r_f * T_s)
    exp_rd = np.exp(-r_d * T_s)
    return {
        "delta": cp_f * exp_rf * nd1,
        "gamma": exp_rf * npd1 / (spot_arr * sig * sqrt_T),
        "vega": spot_arr * exp_rf * npd1 * sqrt_T / 100.0,
        "theta": (-spot_arr * exp_rf * npd1 * sig / (2.0 * sqrt_T)
                  + cp_f * r_f * spot_arr * exp_rf * nd1
                  - cp_f * r_d * K_f * exp_rd * _norm_cdf(cp_f * d2)) / 365.0,
    }


def _build_greeks_chart(processed_legs, S, T, r_d, r_f, notional, atm_vol=0.10):
    """Chart 2: 2x2 subplot of Delta, Gamma, Vega, Theta vs spot. Vectorised."""
    tpl = CHART_TEMPLATE["layout"]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Delta", "Gamma", "Vega", "Theta"),
                        vertical_spacing=0.14, horizontal_spacing=0.10)

    implied_move = atm_vol * np.sqrt(max(T, 1.0 / 365.0))
    half_range = max(0.05, min(2.5 * implied_move, 0.30))
    spot_grid = np.linspace(S * (1 - half_range), S * (1 + half_range), 150)
    greek_names = ["delta", "gamma", "vega", "theta"]
    greek_colors = [COLORS["accent_cyan"], COLORS["accent_blue"],
                    COLORS["accent_purple"], COLORS["accent_orange"]]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for gi, (gname, color, (row, col)) in enumerate(zip(greek_names, greek_colors, positions)):
        total_vals = np.zeros(len(spot_grid))

        for li, lg in enumerate(processed_legs):
            qty = lg["side_sign"] * lg["ratio"]
            greeks_arr = _vectorized_greeks(spot_grid, lg["strike"], T,
                                            r_d, r_f, lg["vol"], lg["cp_sign"])
            leg_vals = greeks_arr[gname] * qty
            total_vals += leg_vals

            fig.add_trace(go.Scatter(
                x=spot_grid, y=leg_vals * notional, mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                name=f"L{li+1} {gname}", showlegend=False, opacity=0.4,
                hovertemplate=f"Leg {li+1}<br>Spot: %{{x:.4f}}<br>{gname}: %{{y:,.0f}}<extra></extra>",
            ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=spot_grid, y=total_vals * notional, mode="lines",
            line=dict(color=color, width=2.5),
            name=gname.capitalize(), showlegend=False,
            hovertemplate=f"Spot: %{{x:.4f}}<br>{gname.capitalize()}: %{{y:,.0f}}<extra>Total</extra>",
        ), row=row, col=col)

        fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"),
                      row=row, col=col)

    fig.update_layout(
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=50, r=15, t=35, b=30),
        hoverlabel=tpl["hoverlabel"], height=380,
    )
    fig.update_xaxes(gridcolor="#1a1a30")
    fig.update_yaxes(gridcolor="#1a1a30")
    for ann in fig.layout.annotations:
        ann.font.color = COLORS["text_primary"]
        ann.font.size = 11
    return fig


def _build_pnl_heatmap(processed_legs, S, T, r_d, r_f, notional, atm_vol=0.10):
    """Chart 3: P&L heatmap -- spot shock x vol shock. Vectorised."""
    tpl = CHART_TEMPLATE["layout"]
    implied_move = atm_vol * np.sqrt(max(T, 1.0 / 365.0))
    spot_half = max(0.05, min(2.5 * implied_move, 0.30))
    spot_shocks = np.linspace(-spot_half, spot_half, 25)
    vol_shocks = np.linspace(-0.50, 0.50, 25)

    net_prem_per_unit = sum(
        lg["price_unit"] * lg["side_sign"] * lg["ratio"]
        for lg in processed_legs
    )

    spot_arr = S * (1.0 + spot_shocks)  # shape (25,)
    pnl_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))

    for vi, dv in enumerate(vol_shocks):
        row_pnl = np.zeros(len(spot_shocks))
        for lg in processed_legs:
            vol_shocked = max(lg["vol"] * (1.0 + dv), 0.005)
            qty = lg["side_sign"] * lg["ratio"]
            # _gk_price is vectorised over S (spot_arr)
            row_pnl += _gk_price(spot_arr, lg["strike"], T, r_d, r_f,
                                 vol_shocked, lg["cp_sign"]) * qty
        pnl_matrix[vi, :] = (row_pnl - net_prem_per_unit) * notional

    fig = go.Figure(go.Heatmap(
        x=[f"{ds:+.0%}" for ds in spot_shocks],
        y=[f"{dv:+.0%}" for dv in vol_shocks],
        z=pnl_matrix, zmid=0,
        colorscale=[
            [0, COLORS["accent_red"]],
            [0.5, COLORS["bg_card"]],
            [1, COLORS["accent_green"]],
        ],
        colorbar=dict(
            title=dict(text="P&L", font=dict(color=COLORS["text_secondary"])),
            tickfont=dict(color=COLORS["text_secondary"]),
            thickness=12, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
        hovertemplate="Spot: %{x}<br>Vol: %{y}<br>P&L: %{z:,.0f}<extra></extra>",
        xgap=2, ygap=2,
    ))

    # Current market crosshair
    mid_s = len(spot_shocks) // 2
    mid_v = len(vol_shocks) // 2
    fig.add_trace(go.Scatter(
        x=[f"{spot_shocks[mid_s]:+.0%}"], y=[f"{vol_shocks[mid_v]:+.0%}"],
        mode="markers", marker=dict(size=12, color=COLORS["accent_cyan"],
                                    symbol="cross-thin-open", line=dict(width=2)),
        name="Current", showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="P&L HEATMAP -- SPOT vs VOL SHOCK",
                   font=dict(color=COLORS["text_primary"], size=13)),
        xaxis_title="Spot Shock", yaxis_title="Vol Shock",
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=60, r=15, t=40, b=45),
        hoverlabel=tpl["hoverlabel"], height=380,
    )
    return fig


def _build_3d_surface(processed_legs, S, T, r_d, r_f, notional, atm_vol=0.10):
    """Chart 4: 3D P&L surface -- Spot x Time x P&L."""
    tpl = CHART_TEMPLATE["layout"]
    n_spot = 40
    n_time = 30
    implied_move = atm_vol * np.sqrt(max(T, 1.0 / 365.0))
    half_range = max(0.05, min(2.5 * implied_move, 0.30))
    spot_grid = np.linspace(S * (1 - half_range), S * (1 + half_range), n_spot)
    time_grid = np.linspace(T, max(T * 0.01, 1.0 / 365.0), n_time)

    net_prem_per_unit = sum(
        lg["price_unit"] * lg["side_sign"] * lg["ratio"]
        for lg in processed_legs
    )

    pnl_surface = np.zeros((n_time, n_spot))
    for ti, t_val in enumerate(time_grid):
        row_pnl = np.zeros(n_spot)
        for lg in processed_legs:
            qty = lg["side_sign"] * lg["ratio"]
            # _gk_price vectorised over spot_grid
            row_pnl += _gk_price(spot_grid, lg["strike"], max(t_val, 1e-6),
                                 r_d, r_f, lg["vol"], lg["cp_sign"]) * qty
        pnl_surface[ti, :] = (row_pnl - net_prem_per_unit) * notional

    fig = go.Figure(go.Surface(
        x=spot_grid,
        y=time_grid * 365,  # display in days
        z=pnl_surface,
        colorscale=[
            [0, COLORS["accent_red"]],
            [0.5, COLORS["bg_card"]],
            [1, COLORS["accent_green"]],
        ],
        cmid=0,
        hovertemplate="Spot: %{x:.4f}<br>DTE: %{y:.0f}<br>P&L: %{z:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="3D P&L SURFACE", font=dict(color=COLORS["text_primary"], size=13)),
        scene=dict(
            xaxis=dict(title="Spot", backgroundcolor=COLORS["bg_primary"],
                       gridcolor=COLORS["border"], color=COLORS["text_secondary"]),
            yaxis=dict(title="DTE", backgroundcolor=COLORS["bg_primary"],
                       gridcolor=COLORS["border"], color=COLORS["text_secondary"]),
            zaxis=dict(title="P&L", backgroundcolor=COLORS["bg_primary"],
                       gridcolor=COLORS["border"], color=COLORS["text_secondary"]),
            bgcolor=COLORS["bg_primary"],
        ),
        paper_bgcolor=tpl["paper_bgcolor"],
        font=tpl["font"],
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
    )
    return fig


def _build_premium_table(processed_legs, pair, pip_size, notional):
    """Chart 5: Premium & cost breakdown HTML table."""
    header_style = {
        "backgroundColor": COLORS["bg_secondary"],
        "color": COLORS["text_secondary"],
        "fontWeight": "700", "fontSize": "10px",
        "textTransform": "uppercase", "letterSpacing": "1px",
        "border": f"1px solid {COLORS['border_subtle']}",
        "padding": "8px 10px", "textAlign": "center",
    }
    cell_style = {
        "backgroundColor": COLORS["bg_card"],
        "color": COLORS["text_primary"],
        "fontSize": "11px", "fontFamily": "'JetBrains Mono', monospace",
        "border": f"1px solid {COLORS['border_subtle']}",
        "padding": "6px 10px", "textAlign": "center",
    }
    buy_color = COLORS["accent_green"]
    sell_color = COLORS["accent_red"]

    headers = ["Leg", "C/P", "Side", "Delta", "Strike", "Vol (%)",
               "Prem (pips)", "Prem (%)", "Ratio", "Net Prem",
               "Delta", "Vega", "Theta", "Vanna", "Volga"]
    header_row = html.Tr([html.Th(h, style=header_style) for h in headers])

    rows = []
    net_pips = 0.0
    net_pct = 0.0
    net_total = 0.0
    for lg in processed_legs:
        side_col = buy_color if lg["side"] == "buy" else sell_color
        rows.append(html.Tr([
            html.Td(f"#{lg['leg_num']}", style=cell_style),
            html.Td(lg["cp"].upper(), style=cell_style),
            html.Td(lg["side"].upper(), style={**cell_style, "color": side_col, "fontWeight": "600"}),
            html.Td(f"{lg['delta_input']:.0%}", style=cell_style),
            html.Td(_fmt_strike(lg['strike'], pair), style=cell_style),
            html.Td(f"{lg['vol']*100:.2f}", style=cell_style),
            html.Td(f"{lg['premium_pips']:.1f}", style=cell_style),
            html.Td(f"{lg['premium_pct']:.3f}", style=cell_style),
            html.Td(f"{lg['ratio']}", style=cell_style),
            html.Td(f"{lg['premium_total']:,.0f}", style={
                **cell_style,
                "color": buy_color if lg["premium_total"] < 0 else sell_color,
            }),
            html.Td(f"{lg['delta']:.4f}", style=cell_style),
            html.Td(f"{lg['vega'] * notional:.0f}", style=cell_style),
            html.Td(f"{lg['theta'] * notional:.0f}", style=cell_style),
            html.Td(f"{lg['vanna'] * notional:.4f}", style=cell_style),
            html.Td(f"{lg['volga'] * notional:.0f}", style=cell_style),
        ]))
        net_pips += lg["premium_pips"]
        net_pct += lg["premium_pct"]
        net_total += lg["premium_total"]

    # Net totals row
    total_style = {
        **cell_style,
        "fontWeight": "700",
        "backgroundColor": COLORS["bg_secondary"],
        "color": COLORS["accent_cyan"],
    }
    net_delta = sum(lg["delta"] for lg in processed_legs)
    net_vega = sum(lg["vega"] for lg in processed_legs)
    net_theta = sum(lg["theta"] for lg in processed_legs)
    net_vanna = sum(lg["vanna"] for lg in processed_legs)
    net_volga = sum(lg["volga"] for lg in processed_legs)
    rows.append(html.Tr([
        html.Td("NET", style=total_style),
        html.Td("", style=total_style),
        html.Td("", style=total_style),
        html.Td("", style=total_style),
        html.Td("", style=total_style),
        html.Td("", style=total_style),
        html.Td(f"{net_pips:.1f}", style=total_style),
        html.Td(f"{net_pct:.3f}", style=total_style),
        html.Td("", style=total_style),
        html.Td(f"{net_total:,.0f}", style={
            **total_style,
            "color": COLORS["pnl_profit"] if net_total > 0 else (COLORS["pnl_loss"] if net_total < 0 else COLORS["pnl_neutral"]),
        }),
        html.Td(f"{net_delta:.4f}", style=total_style),
        html.Td(f"{net_vega * notional:.0f}", style=total_style),
        html.Td(f"{net_theta * notional:.0f}", style=total_style),
        html.Td(f"{net_vanna * notional:.4f}", style=total_style),
        html.Td(f"{net_volga * notional:.0f}", style=total_style),
    ]))

    table = html.Table(
        [html.Thead(header_row), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "borderRadius": "0px"},
    )
    return html.Div([
        html.Div("PREMIUM & COST TABLE",
                 style={"color": COLORS["text_primary"], "fontSize": "13px",
                        "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace",
                        "marginBottom": "10px", "letterSpacing": "1.5px",
                        "textTransform": "uppercase"}),
        table,
    ], style={"overflowX": "auto"})


def _build_smile_chart(processed_legs, vol_surface, tenor):
    """Chart 7: Vol smile for current tenor with leg vols marked."""
    tpl = CHART_TEMPLATE["layout"]
    fig = go.Figure()

    delta_labels = ["10P", "25P", "ATM", "25C", "10C"]

    def _extract_smile(vs_data, tnr):
        """Extract 5-point smile from vol surface for a given tenor."""
        if not vs_data or tnr not in vs_data:
            return None
        q = vs_data.get(tnr, {})
        if not q:
            return None
        atm = q.get("atm", 0)
        rr25 = q.get("rr25", 0)
        bf25 = q.get("bf25", 0)
        rr10 = q.get("rr10", 0)
        bf10 = q.get("bf10", 0)
        return [
            atm + bf10 - rr10 / 2.0,   # 10P
            atm + bf25 - rr25 / 2.0,   # 25P
            atm,                         # ATM
            atm + bf25 + rr25 / 2.0,   # 25C
            atm + bf10 + rr10 / 2.0,   # 10C
        ]

    # Overlay tenors: 1M (dotted), current (solid), 1Y (dotted)
    overlay_tenors = []
    if "1M" in vol_surface and "1M" != tenor:
        overlay_tenors.append(("1M", "dot", 1.5, "rgba(128,128,128,0.4)"))
    if "1Y" in vol_surface and "1Y" != tenor:
        overlay_tenors.append(("1Y", "dot", 1.0, "rgba(128,128,128,0.25)"))

    for ot, dash_style, width, color in overlay_tenors:
        smile_pts = _extract_smile(vol_surface, ot)
        if smile_pts:
            fig.add_trace(go.Scatter(
                x=delta_labels, y=smile_pts, mode="lines+markers",
                line=dict(color=color, width=width, dash=dash_style),
                marker=dict(size=4, color=color),
                name=ot, showlegend=True,
                hovertemplate="%{x}: %{y:.2f}%<extra>" + ot + "</extra>",
            ))

    # Current tenor smile (solid orange)
    current_smile = _extract_smile(vol_surface, tenor)
    if current_smile:
        fig.add_trace(go.Scatter(
            x=delta_labels, y=current_smile, mode="lines+markers",
            line=dict(color=COLORS["accent_orange"], width=3),
            marker=dict(size=7, color=COLORS["accent_orange"]),
            name=tenor,
            hovertemplate="%{x}: %{y:.2f}%<extra>" + tenor + "</extra>",
        ))
        # Horizontal ATM reference line
        atm_val = current_smile[2]
        fig.add_hline(y=atm_val, line=dict(color=COLORS["text_muted"], width=1, dash="dash"),
                      annotation_text=f"ATM {atm_val:.2f}",
                      annotation_font=dict(color=COLORS["text_muted"], size=9))

    # Mark each leg's vol on the smile
    for lg in processed_legs:
        vol_pct = lg["vol"] * 100.0
        delta_abs = abs(lg["delta_input"])  # handle negative put deltas
        cp_str = lg["cp"]
        # Map leg delta to nearest smile label
        if delta_abs > 0.45:
            x_label = "ATM"  # deep ITM or ATM — closest to ATM
        elif abs(delta_abs - 0.25) < 0.08:
            x_label = "25C" if cp_str == "call" else "25P"
        elif delta_abs <= 0.17:
            x_label = "10C" if cp_str == "call" else "10P"
        else:
            x_label = "25C" if cp_str == "call" else "25P"

        dot_color = COLORS["accent_cyan"] if cp_str == "call" else COLORS["accent_purple"]
        fig.add_trace(go.Scatter(
            x=[x_label], y=[vol_pct], mode="markers",
            marker=dict(size=12, color=dot_color, symbol="circle",
                        line=dict(width=2, color="white")),
            name=f"L{lg['leg_num']} {cp_str[0].upper()} {delta_abs:.0%}",
            showlegend=True,
            hovertemplate=f"Leg {lg['leg_num']}: {vol_pct:.2f}%<br>{cp_str} {delta_abs:.0%} delta<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="VOL SMILE", font=dict(color=COLORS["text_primary"], size=13)),
        xaxis_title="Delta", yaxis_title="Implied Vol (%)",
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=55, r=15, t=40, b=35),
        legend=dict(font=dict(color=COLORS["text_secondary"], size=10),
                    bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
        hoverlabel=tpl["hoverlabel"],
        xaxis=dict(gridcolor="#1a1a30"),
        yaxis=dict(gridcolor="#1a1a30"),
        height=380,
    )
    return fig


def _build_scenario_chart(processed_legs, S, T, r_d, r_f, notional):
    """Chart 6: Scenario sensitivity -- 3 stacked line charts."""
    tpl = CHART_TEMPLATE["layout"]
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("P&L vs Vol Shift", "P&L vs Time Decay",
                                        "P&L vs Spot Move"),
                        vertical_spacing=0.10)

    net_prem_per_unit = sum(
        lg["price_unit"] * lg["side_sign"] * lg["ratio"]
        for lg in processed_legs
    )

    # --- P&L vs Vol Shift ---
    vol_shifts = np.linspace(-0.50, 0.50, 50)
    pnl_vol = np.zeros(len(vol_shifts))
    for vi, dv in enumerate(vol_shifts):
        pnl = 0.0
        for lg in processed_legs:
            vol_s = max(lg["vol"] * (1.0 + dv), 0.005)
            qty = lg["side_sign"] * lg["ratio"]
            pnl += float(_gk_price(S, lg["strike"], T, r_d, r_f, vol_s, lg["cp_sign"])) * qty
        pnl_vol[vi] = (pnl - net_prem_per_unit) * notional

    fig.add_trace(go.Scatter(
        x=vol_shifts * 100, y=pnl_vol, mode="lines",
        line=dict(color=COLORS["accent_purple"], width=2),
        name="Vol Sensitivity", showlegend=False,
        hovertemplate="Vol shift: %{x:+.0f}%<br>P&L: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"), row=1, col=1)

    # --- P&L vs Time Decay ---
    dte_points = np.linspace(T * 365, 0, 40)
    pnl_time = np.zeros(len(dte_points))
    for ti, dte in enumerate(dte_points):
        t_val = max(dte / 365.0, 1e-6)
        pnl = 0.0
        for lg in processed_legs:
            qty = lg["side_sign"] * lg["ratio"]
            pnl += float(_gk_price(S, lg["strike"], t_val, r_d, r_f, lg["vol"], lg["cp_sign"])) * qty
        pnl_time[ti] = (pnl - net_prem_per_unit) * notional

    fig.add_trace(go.Scatter(
        x=dte_points, y=pnl_time, mode="lines",
        line=dict(color=COLORS["accent_orange"], width=2),
        name="Time Decay", showlegend=False,
        hovertemplate="DTE: %{x:.0f}d<br>P&L: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"), row=2, col=1)

    # --- P&L vs Spot Move ---
    spot_moves = np.linspace(-0.15, 0.15, 60)
    pnl_spot = np.zeros(len(spot_moves))
    for si, ds in enumerate(spot_moves):
        s_sh = S * (1.0 + ds)
        pnl = 0.0
        for lg in processed_legs:
            qty = lg["side_sign"] * lg["ratio"]
            pnl += float(_gk_price(s_sh, lg["strike"], T, r_d, r_f, lg["vol"], lg["cp_sign"])) * qty
        pnl_spot[si] = (pnl - net_prem_per_unit) * notional

    fig.add_trace(go.Scatter(
        x=spot_moves * 100, y=pnl_spot, mode="lines",
        line=dict(color=COLORS["accent_cyan"], width=2),
        name="Spot Sensitivity", showlegend=False,
        hovertemplate="Spot move: %{x:+.1f}%<br>P&L: $%{y:,.0f}<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"), row=3, col=1)

    fig.update_layout(
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=55, r=15, t=35, b=35),
        hoverlabel=tpl["hoverlabel"], height=520,
    )
    fig.update_xaxes(gridcolor="#1a1a30")
    fig.update_xaxes(title_text="Vol Shift (%)", ticksuffix="%", row=1, col=1)
    fig.update_xaxes(title_text="DTE (days)", row=2, col=1)
    fig.update_xaxes(title_text="Spot Move (%)", ticksuffix="%", row=3, col=1)
    fig.update_yaxes(gridcolor="#1a1a30", title_text="P&L ($)")
    for ann in fig.layout.annotations:
        ann.font.color = COLORS["text_primary"]
        ann.font.size = 11
    return fig


# ============================================================================
# Layout
# ============================================================================

def _make_leg_row(idx, cp="call", side="buy", delta=0.25, ratio=1, tenor_mult=1.0, visible=True):
    """Generate one leg configuration row — fully stacked for 350px sidebar."""
    row_bg = "#06060f" if idx % 2 == 0 else COLORS["bg_primary"]
    display = "block" if visible else "none"
    _inp = {**INPUT_STYLE, "padding": "5px 8px", "fontSize": "11px", "textAlign": "center",
            "height": "32px"}
    _lbl = {"fontSize": "8px", "color": COLORS["text_muted"], "textTransform": "uppercase",
            "letterSpacing": "0.5px", "marginBottom": "2px",
            "fontFamily": "'JetBrains Mono', monospace"}
    return html.Div([
        # Row 1: TYPE + SIDE + DELTA — the three things you always set
        html.Div([
            html.Div([
                html.Div("TYPE", style=_lbl),
                dcc.Dropdown(
                    id={"type": "stb-cp", "index": idx},
                    options=[{"label": "CALL", "value": "call"}, {"label": "PUT", "value": "put"}],
                    value=cp, clearable=False, searchable=False,
                    style={"fontSize": "11px", "minHeight": "32px"},
                ),
            ], style={"flex": "2", "minWidth": "0"}),
            html.Div([
                html.Div("SIDE", style=_lbl),
                dcc.Dropdown(
                    id={"type": "stb-side", "index": idx},
                    options=[{"label": "BUY", "value": "buy"}, {"label": "SELL", "value": "sell"}],
                    value=side, clearable=False, searchable=False,
                    style={"fontSize": "11px", "minHeight": "32px"},
                ),
            ], style={"flex": "2", "minWidth": "0"}),
            html.Div([
                html.Div("DELTA", style=_lbl),
                dcc.Input(id={"type": "stb-delta", "index": idx}, type="number",
                          value=delta, min=0.05, max=0.95, step=0.05,
                          style={**_inp, "width": "100%"}, debounce=True),
            ], style={"flex": "1", "minWidth": "55px"}),
        ], style={"display": "flex", "gap": "6px", "alignItems": "flex-end"}),
        # Row 2: RATIO + TNR× + computed readout
        html.Div([
            html.Div([
                html.Div("RATIO", style=_lbl),
                dcc.Input(id={"type": "stb-ratio", "index": idx}, type="number",
                          value=ratio, min=1, max=3, step=1,
                          style={**_inp, "width": "100%"}, debounce=True),
            ], style={"width": "55px", "flexShrink": "0"}),
            html.Div([
                html.Div("TNR\u00d7", style=_lbl),
                dcc.Input(id={"type": "stb-tenor-mult", "index": idx}, type="number",
                          value=tenor_mult, min=0.5, max=4.0, step=0.5,
                          style={**_inp, "width": "100%"}, debounce=True),
            ], style={"width": "55px", "flexShrink": "0"}),
            # Computed readout fills remaining space
            html.Div([
                html.Div("STRIKE / VOL / PREM", style=_lbl),
                html.Div([
                    html.Span(id={"type": "stb-strike-disp", "index": idx},
                              style={"color": COLORS["accent_cyan"], "fontSize": "11px",
                                     "fontWeight": "600"}),
                    html.Span(" \u00b7 ", style={"color": "#3a3a5c"}),
                    html.Span(id={"type": "stb-vol-disp", "index": idx},
                              style={"color": COLORS["text_primary"], "fontSize": "10px"}),
                    html.Span(" \u00b7 ", style={"color": "#3a3a5c"}),
                    html.Span(id={"type": "stb-prem-disp", "index": idx},
                              style={"color": COLORS["accent_orange"], "fontSize": "11px",
                                     "fontWeight": "600"}),
                ], style={"paddingTop": "5px", "whiteSpace": "nowrap"}),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "6px", "marginTop": "6px",
                  "alignItems": "flex-end"}),
    ], id={"type": "stb-leg-row", "index": idx}, style={
        "display": display,
        "padding": "10px 10px", "borderRadius": "0px",
        "backgroundColor": row_bg,
        "borderLeft": f"3px solid {COLORS['accent_orange']}",
        "marginBottom": "4px",
    })


def layout():
    all_pairs = get_all_pairs()
    preset_options = [{"label": k, "value": k} for k in PRESETS.keys()]
    pair_options = [{"label": p, "value": p} for p in all_pairs]
    tenor_options = [{"label": t, "value": t} for t in TENORS]

    # Build initial leg rows (default: Call = 1 leg)
    initial_legs = PRESETS["Call"]
    num_initial = len(initial_legs)
    leg_rows = []
    for i in range(MAX_LEGS):
        if i < num_initial:
            lg = initial_legs[i]
            leg_rows.append(_make_leg_row(
                i, lg["cp"], lg["side"], lg["delta"], lg["ratio"],
                lg.get("tenor_mult", 1.0), visible=True))
        else:
            leg_rows.append(_make_leg_row(i, visible=False))

    return html.Div([
        dcc.Download(id="stb-csv-download"),
        # ── Stores for solver, compare, blotter ──
        dcc.Store(id="stb-solver-store", data=None),
        dcc.Store(id="stb-compare-a", data=None),
        dcc.Store(id="stb-compare-b", data=None),
        dcc.Store(id="stb-suggestions-store", data=[]),
        dcc.Store(id="stb-saved-structures", storage_type="local", data=[]),

        html.Div([
            # ── Left: Input Panel ──────────────────────────────────────
            html.Div([
                html.Div("TRADE IDEA WORKSHOP",
                         style={"color": COLORS["text_primary"], "fontSize": "13px",
                                "fontWeight": "700",
                                "fontFamily": "'JetBrains Mono', monospace",
                                "marginBottom": "16px", "paddingBottom": "10px",
                                "borderBottom": f"1px solid {COLORS['border_subtle']}",
                                "letterSpacing": "1.5px", "textTransform": "uppercase"}),

                # Preset selector
                html.Label("PRESET STRUCTURE", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="stb-preset",
                    options=preset_options, value="Call", clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                # Pair selector
                html.Label("CURRENCY PAIR", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="stb-pair",
                    options=pair_options, value="EURUSD", clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                # Tenor selector
                html.Label("TENOR", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="stb-tenor",
                    options=tenor_options, value="1M", clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                # Notional
                html.Label("NOTIONAL", style=LABEL_STYLE),
                dcc.Input(
                    id="stb-notional",
                    type="number", value=10_000_000, step=1_000_000, min=100_000,
                    style={**INPUT_STYLE, "marginBottom": "14px"}, debounce=True,
                ),

                # Number of active legs
                html.Label("ACTIVE LEGS", style=LABEL_STYLE),
                dcc.Input(
                    id="stb-num-legs",
                    type="number", value=1, min=1, max=MAX_LEGS, step=1,
                    style={**INPUT_STYLE, "marginBottom": "14px"}, debounce=True,
                ),

                # Leg header
                html.Div("OPTION LEGS", style={
                    "color": COLORS["text_bright"], "fontSize": "11px", "fontWeight": "700",
                    "fontFamily": "'JetBrains Mono', monospace", "letterSpacing": "1.5px",
                    "textTransform": "uppercase", "marginBottom": "6px",
                    "paddingBottom": "4px", "borderBottom": f"1px solid {COLORS['border']}",
                }),

                # Leg rows container
                html.Div(leg_rows, id="stb-legs-container"),

                # Add / Remove buttons
                html.Div([
                    html.Button("+ ADD LEG", id="stb-add-leg",
                                style={**BUTTON_STYLE, "fontSize": "10px",
                                       "padding": "7px 16px", "flex": "1"}),
                    html.Button("- REMOVE LAST", id="stb-remove-leg",
                                style={**BUTTON_STYLE, "fontSize": "10px",
                                       "padding": "7px 16px", "flex": "1",
                                       "backgroundColor": COLORS["accent_red"],
                                       "boxShadow": "0 4px 14px rgba(255,51,51,0.25)"}),
                ], style={"display": "flex", "gap": "8px", "marginTop": "10px"}),

                # ── Quick Actions ──────────────────────────────────────
                html.Div([
                    html.Div("QUICK ACTIONS", style={**LABEL_STYLE, "marginTop": "14px",
                             "paddingTop": "10px", "borderTop": f"1px solid {COLORS['border_subtle']}"}),
                    html.Div([
                        html.Button("FLIP", id="stb-flip-btn", n_clicks=0,
                                    style={**BUTTON_STYLE, "fontSize": "9px", "padding": "5px 8px", "flex": "1"}),
                        html.Button("MIRROR", id="stb-mirror-btn", n_clicks=0,
                                    style={**BUTTON_STYLE, "fontSize": "9px", "padding": "5px 8px", "flex": "1"}),
                        html.Button("0-COST", id="stb-zero-cost-btn", n_clicks=0,
                                    style={**BUTTON_SUCCESS_STYLE, "fontSize": "9px", "padding": "5px 8px", "flex": "1"}),
                    ], style={"display": "flex", "gap": "4px"}),
                ]),

                # ── Send to Blotter ───────────────────────────────────
                html.Div([
                    html.Div("EXECUTE", style={**LABEL_STYLE, "marginTop": "14px",
                             "paddingTop": "10px",
                             "borderTop": f"1px solid {COLORS['border_subtle']}"}),
                    html.Button("SEND TO BLOTTER", id="stb-send-to-blotter", n_clicks=0,
                                style={**BUTTON_SUCCESS_STYLE, "width": "100%",
                                       "padding": "12px 16px", "fontSize": "12px",
                                       "fontWeight": "800", "letterSpacing": "2px"}),
                    html.Div(id="stb-send-status", style={
                        "color": COLORS["accent_green"], "fontSize": "9px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "marginTop": "4px", "textAlign": "center", "minHeight": "14px",
                    }),
                ]),

                # ── Solver ─────────────────────────────────────────────
                html.Div([
                    html.Div("SOLVER", style={**LABEL_STYLE, "marginTop": "14px",
                             "paddingTop": "10px", "borderTop": f"1px solid {COLORS['border_subtle']}"}),
                    html.Div([
                        html.Div([
                            html.Label("TARGET", style={**LABEL_STYLE, "fontSize": "8px"}),
                            dcc.Dropdown(id="stb-solver-target",
                                         options=[
                                             {"label": "Net Prem (pips)", "value": "net_premium_pips"},
                                             {"label": "Net Prem (CCY2)", "value": "net_premium"},
                                             {"label": "Net Delta", "value": "net_delta"},
                                             {"label": "Net Vega", "value": "net_vega"},
                                             {"label": "Net Theta", "value": "net_theta"},
                                             {"label": "Prob of Profit", "value": "pop"},
                                         ], value="net_premium_pips", clearable=False,
                                         style={"fontSize": "10px"}),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Label("VALUE", style={**LABEL_STYLE, "fontSize": "8px"}),
                            dcc.Input(id="stb-solver-value", type="number", value=0,
                                      style={**INPUT_STYLE, "fontSize": "10px", "padding": "5px"}, debounce=True),
                        ], style={"width": "65px"}),
                    ], style={"display": "flex", "gap": "4px", "marginBottom": "4px"}),
                    html.Div([
                        html.Div([
                            html.Label("LEG", style={**LABEL_STYLE, "fontSize": "8px"}),
                            dcc.Dropdown(id="stb-solver-leg",
                                         options=[{"label": f"L{i+1}", "value": i} for i in range(MAX_LEGS)],
                                         value=0, clearable=False, style={"fontSize": "10px"}),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Label("PARAM", style={**LABEL_STYLE, "fontSize": "8px"}),
                            dcc.Dropdown(id="stb-solver-param",
                                         options=[{"label": "Delta", "value": "delta"},
                                                  {"label": "Ratio", "value": "ratio"}],
                                         value="delta", clearable=False, style={"fontSize": "10px"}),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "4px", "marginBottom": "4px"}),
                    html.Div([
                        html.Button("SOLVE", id="stb-solve-btn", n_clicks=0,
                                    style={**BUTTON_SUCCESS_STYLE, "fontSize": "9px",
                                           "padding": "5px 12px", "flex": "1"}),
                        html.Button("APPLY", id="stb-solver-apply-btn", n_clicks=0,
                                    style={**BUTTON_STYLE, "fontSize": "9px",
                                           "padding": "5px 12px", "flex": "1",
                                           "display": "none"}),
                    ], style={"display": "flex", "gap": "4px"}),
                    html.Div(id="stb-solver-result", style={
                        "color": COLORS["accent_green"], "fontSize": "9px",
                        "fontFamily": "'JetBrains Mono', monospace", "marginTop": "4px",
                    }),
                ]),

            ], style={
                **CARD_STYLE,
                "width": "350px", "minWidth": "350px", "flexShrink": "0",
                "overflowY": "auto", "maxHeight": "calc(100vh - 100px)",
            }),

            # ── Right: Charts + Stats ─────────────────────────────────
            html.Div([
                # Suggestion bar (intelligence layer)
                html.Div(id="stb-suggestions-bar", style={
                    "marginBottom": "10px",
                }),

                # Summary stat boxes (10 — original 8 + implied move + EV)
                html.Div(id="stb-stats-row", style={
                    "display": "flex", "gap": "8px", "marginBottom": "8px",
                    "flexWrap": "wrap",
                }),

                # Compare toolbar + quick actions
                html.Div([
                    html.Button("SAVE A", id="stb-save-a", n_clicks=0,
                                style={**BUTTON_STYLE, "fontSize": "9px", "padding": "4px 10px"}),
                    html.Button("SAVE B", id="stb-save-b", n_clicks=0,
                                style={**BUTTON_STYLE, "fontSize": "9px", "padding": "4px 10px"}),
                    html.Button("COMPARE", id="stb-compare-toggle", n_clicks=0,
                                style={**BUTTON_STYLE, "fontSize": "9px", "padding": "4px 10px"}),
                    html.Button("CLEAR", id="stb-compare-clear", n_clicks=0,
                                style={**BUTTON_DANGER_STYLE, "fontSize": "9px", "padding": "4px 10px"}),
                    html.Div(id="stb-compare-status", style={
                        "color": COLORS["text_muted"], "fontSize": "9px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "paddingTop": "3px", "flex": "1",
                    }),
                ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                          "marginBottom": "10px"}),

                # Chart grid row 1 (3 charts)
                html.Div([
                    html.Div([
                        html.Button("CSV", id="stb-csv-payoff", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="stb-payoff-chart", style={"height": "380px"},
                                  config={"displayModeBar": True, "scrollZoom": False}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "340px",
                              "padding": "12px"}, className="dashboard-card"),
                    html.Div([
                        html.Button("CSV", id="stb-csv-greeks", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="stb-greeks-chart", style={"height": "380px"},
                                  config={"displayModeBar": True, "scrollZoom": False}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "340px",
                              "padding": "12px"}, className="dashboard-card"),
                    html.Div([
                        html.Button("CSV", id="stb-csv-heatmap", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="stb-heatmap-chart", style={"height": "380px"},
                                  config={"displayModeBar": True, "scrollZoom": False}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "340px",
                              "padding": "12px"}, className="dashboard-card"),
                ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                          "marginBottom": "12px"}),

                # Chart grid row 2 (4 charts: smile, spot×vol surface, premium table, scenario)
                html.Div([
                    html.Div([
                        dcc.Graph(id="struct-smile-chart", style={"height": "380px"},
                                  config={"displayModeBar": True, "scrollZoom": False}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "340px",
                              "padding": "12px"}, className="dashboard-card"),
                    html.Div([
                        dcc.Graph(id="stb-pnl-surface", config={"displayModeBar": True, "scrollZoom": True},
                                  style={"height": "380px"}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "340px",
                              "padding": "12px"}, className="dashboard-card"),
                    html.Div([
                        html.Button("CSV", id="stb-csv-3d", n_clicks=0, style={"display": "none"}),
                        dcc.Graph(id="stb-3d-chart", style={"display": "none"}),
                    ], style={"display": "none"}),
                    html.Div([
                        html.Button("CSV", id="stb-csv-scenario", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="stb-scenario-chart", style={"height": "520px"},
                                  config={"displayModeBar": True, "scrollZoom": False}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "340px",
                              "padding": "12px"}, className="dashboard-card"),
                ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

                # Premium & cost table (full width, scrollable)
                html.Div(id="stb-premium-table-container", style={
                    **CARD_STYLE, "padding": "12px", "marginTop": "12px",
                    "overflowX": "auto",
                }, className="dashboard-card"),

                # Parallel coordinates (full width)
                html.Div([
                    dcc.Graph(id="stb-parallel-coords", config={"displayModeBar": False},
                              style={"height": "350px"}),
                ], style={**CARD_STYLE, "padding": "12px", "marginBottom": "12px",
                          "marginTop": "12px"},
                   className="dashboard-card"),

                # Tenor Scan table
                html.Div(id="stb-tenor-scan", style={
                    **CARD_STYLE, "padding": "12px", "marginTop": "12px",
                }),

                # Trade Analysis (EV + edge + scenarios + risks)
                html.Div(id="stb-trade-analysis", style={
                    **CARD_STYLE, "padding": "12px", "marginTop": "12px",
                }),

                # Compare section (hidden by default)
                html.Div(id="stb-compare-section", style={"display": "none", "marginTop": "12px"}),

                # Historical cost context (kept for backward compat)
                html.Div(id="stb-historical-cost", style={
                    "border": f"1px solid {COLORS['border_subtle']}",
                    "padding": "8px", "marginTop": "4px",
                }),

            ], style={"flex": "1", "minWidth": "0"}),

        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
    ])


# ============================================================================
# P&L Surface (Spot x Vol)
# ============================================================================

def _build_pnl_surface(processed_legs, S, T, r_d, r_f, notional, atm_vol=0.10):
    """Chart: 3D P&L surface -- Spot (x) x Vol (y) -> P&L (z)."""
    tpl = CHART_TEMPLATE["layout"]

    if not processed_legs or T < 1e-6:
        return no_data_fig(height=400, msg="ADD LEGS TO SEE P&L SURFACE")

    n_spots = 40
    n_vols = 30
    atm_vol_safe = max(atm_vol, 0.01)
    implied_move = atm_vol_safe * np.sqrt(max(T, 1.0 / 365.0))
    half_range = max(0.05, min(2.5 * implied_move, 0.30))
    spot_range = np.linspace(S * (1 - half_range), S * (1 + half_range), n_spots)
    vol_range = np.linspace(max(atm_vol_safe * 0.5, 0.01), atm_vol_safe * 1.5, n_vols)

    # Entry premium per unit at current market conditions
    net_prem_per_unit = sum(
        lg["price_unit"] * lg["side_sign"] * lg["ratio"]
        for lg in processed_legs
    )

    pnl_grid = np.zeros((n_vols, n_spots))
    for vi, vol in enumerate(vol_range):
        row_pnl = np.zeros(n_spots)
        for lg in processed_legs:
            qty = lg["side_sign"] * lg["ratio"]
            row_pnl += _gk_price(spot_range, lg["strike"], max(lg["T"], 1e-6),
                                 r_d, r_f, vol, lg["cp_sign"]) * qty
        pnl_grid[vi, :] = (row_pnl - net_prem_per_unit) * notional

    fig = go.Figure(data=go.Surface(
        x=spot_range,
        y=vol_range * 100,
        z=pnl_grid,
        colorscale=[
            [0.0, "#ff3333"],
            [0.3, "#330000"],
            [0.5, "#0e0e0e"],
            [0.7, "#003300"],
            [1.0, "#00cc66"],
        ],
        cmid=0,
        opacity=0.92,
        colorbar=dict(
            title=dict(text="P&L", font=dict(color=COLORS["text_secondary"], size=10)),
            tickfont=dict(color=COLORS["text_secondary"], size=9),
            len=0.6, thickness=12, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
        ),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.3, roughness=0.5),
        hovertemplate="Spot: %{x:.4f}<br>Vol: %{y:.1f}%<br>P&L: %{z:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Spot", backgroundcolor=COLORS["bg_primary"],
                       gridcolor=COLORS["border"], color=COLORS["text_secondary"]),
            yaxis=dict(title="Vol (%)", backgroundcolor=COLORS["bg_primary"],
                       gridcolor=COLORS["border"], color=COLORS["text_secondary"]),
            zaxis=dict(title="P&L", backgroundcolor=COLORS["bg_primary"],
                       gridcolor=COLORS["border"], color=COLORS["text_secondary"]),
            bgcolor=COLORS["bg_primary"],
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.85)),
        ),
        paper_bgcolor=tpl["paper_bgcolor"],
        font=tpl["font"],
        title=dict(text="P&L SURFACE (SPOT \u00d7 VOL)", font=dict(color=COLORS["text_primary"], size=13)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
    )
    return fig


# ============================================================================
# Parallel Coordinates -- Leg Comparison
# ============================================================================

def _build_parallel_coords(processed_legs, S, T, r_d, r_f, notional, atm_vol=0.10):
    """Parallel coordinates: compare current structure's Greeks across legs."""

    if not processed_legs or T < 1e-6:
        return no_data_fig(height=350, msg="ADD LEGS FOR PARALLEL COORDINATES")

    if len(processed_legs) < 2:
        return no_data_fig(height=350, msg="PARALLEL COORDINATES: ADD 2+ LEGS TO COMPARE")

    # Compute per-leg metrics
    leg_data = {"leg": [], "delta": [], "gamma": [], "vega": [], "theta": [],
                "premium": [], "strike": [], "vol": []}

    for li, lg in enumerate(processed_legs):
        leg_data["leg"].append(li + 1)
        qty = lg["side_sign"] * lg["ratio"]
        greeks = _vectorized_greeks(np.array([S]), lg["strike"], max(lg["T"], 1e-6),
                                     r_d, r_f, lg["vol"], lg["cp_sign"])
        leg_data["delta"].append(float(greeks["delta"][0] * qty * notional))
        leg_data["gamma"].append(float(greeks["gamma"][0] * qty * notional))
        leg_data["vega"].append(float(greeks["vega"][0] * qty * notional))
        leg_data["theta"].append(float(greeks["theta"][0] * qty * notional))
        leg_data["premium"].append(float(lg["price_unit"] * qty * notional))
        leg_data["strike"].append(float(lg["strike"]))
        leg_data["vol"].append(float(lg["vol"] * 100))

    if not leg_data["leg"]:
        return no_data_fig(height=350, msg="NO LEG DATA")

    # Build parallel coordinates
    n_legs = len(leg_data["leg"])
    # Color by leg number
    colors = leg_data["leg"]

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=colors,
            colorscale=[[0, "#ff8800"], [0.5, "#00b4d8"], [1.0, "#a78bfa"]],
            showscale=False,
            cmin=1, cmax=max(n_legs, 2),
        ),
        dimensions=[
            dict(label="Leg", values=leg_data["leg"], range=[0.5, n_legs + 0.5]),
            dict(label="Delta ($)", values=leg_data["delta"]),
            dict(label="Gamma ($)", values=leg_data["gamma"]),
            dict(label="Vega ($)", values=leg_data["vega"]),
            dict(label="Theta ($)", values=leg_data["theta"]),
            dict(label="Premium ($)", values=leg_data["premium"]),
            dict(label="Strike", values=leg_data["strike"]),
            dict(label="Vol (%)", values=leg_data["vol"]),
        ],
        labelside="top",
        labelfont=dict(size=10, color="#e0e0e0", family="'JetBrains Mono', monospace"),
        tickfont=dict(size=9, color="#9a9ab0", family="'JetBrains Mono', monospace"),
        rangefont=dict(size=8, color="#9a9ab0", family="'JetBrains Mono', monospace"),
    ))

    fig.update_layout(
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(family="'JetBrains Mono', monospace", color="#e0e0e0"),
        title=dict(text="LEG COMPARISON (PARALLEL COORDINATES)",
                   font=dict(color="#ffffff", size=12),
                   x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=70, b=30),
        height=380,
    )
    return fig


# ============================================================================
# Callbacks
# ============================================================================

def register_callbacks(app):
    # -- Preset selector updates num-legs and leg configs --
    @app.callback(
        [Output("stb-num-legs", "value", allow_duplicate=True)] +
        [Output({"type": "stb-cp", "index": i}, "value") for i in range(MAX_LEGS)] +
        [Output({"type": "stb-side", "index": i}, "value") for i in range(MAX_LEGS)] +
        [Output({"type": "stb-delta", "index": i}, "value") for i in range(MAX_LEGS)] +
        [Output({"type": "stb-ratio", "index": i}, "value") for i in range(MAX_LEGS)] +
        [Output({"type": "stb-tenor-mult", "index": i}, "value") for i in range(MAX_LEGS)],
        [Input("stb-preset", "value")],
        prevent_initial_call=True,
    )
    def update_preset(preset_name):
        legs = PRESETS.get(preset_name, [])
        if not legs:
            return [no_update] * (1 + 5 * MAX_LEGS)

        num_legs = len(legs)
        cp_vals = []
        side_vals = []
        delta_vals = []
        ratio_vals = []
        tenor_mult_vals = []

        for i in range(MAX_LEGS):
            if i < len(legs):
                lg = legs[i]
                cp_vals.append(lg["cp"])
                side_vals.append(lg["side"])
                delta_vals.append(lg["delta"])
                ratio_vals.append(lg["ratio"])
                tenor_mult_vals.append(lg.get("tenor_mult", 1.0))
            else:
                cp_vals.append(no_update)
                side_vals.append(no_update)
                delta_vals.append(no_update)
                ratio_vals.append(no_update)
                tenor_mult_vals.append(no_update)

        return ([num_legs] + cp_vals + side_vals + delta_vals
                + ratio_vals + tenor_mult_vals)

    # -- Add Leg button --
    @app.callback(
        Output("stb-num-legs", "value", allow_duplicate=True),
        Input("stb-add-leg", "n_clicks"),
        State("stb-num-legs", "value"),
        prevent_initial_call=True,
    )
    def add_leg(n_clicks, current):
        if n_clicks is None:
            return no_update
        return min((current or 1) + 1, MAX_LEGS)

    # -- Remove Leg button --
    @app.callback(
        Output("stb-num-legs", "value", allow_duplicate=True),
        Input("stb-remove-leg", "n_clicks"),
        State("stb-num-legs", "value"),
        prevent_initial_call=True,
    )
    def remove_leg(n_clicks, current):
        if n_clicks is None:
            return no_update
        return max((current or 2) - 1, 1)

    # -- Toggle leg row visibility based on num_legs --
    @app.callback(
        [Output({"type": "stb-leg-row", "index": i}, "style") for i in range(MAX_LEGS)],
        [Input("stb-num-legs", "value")],
    )
    def toggle_leg_rows(num_legs):
        num_legs = max(1, min(num_legs or 1, MAX_LEGS))
        styles = []
        for i in range(MAX_LEGS):
            row_bg = COLORS["bg_secondary"] if i % 2 == 0 else COLORS["bg_card"]
            if i < num_legs:
                styles.append({
                    "display": "flex", "gap": "6px", "alignItems": "center",
                    "padding": "5px 8px", "borderRadius": "0px",
                    "backgroundColor": row_bg, "marginBottom": "3px",
                })
            else:
                styles.append({
                    "display": "none", "gap": "6px", "alignItems": "center",
                    "padding": "5px 8px", "borderRadius": "0px",
                    "backgroundColor": row_bg, "marginBottom": "3px",
                })
        return styles

    # -- Main computation callback --
    @app.callback(
        [Output("stb-suggestions-bar", "children"),
         Output("stb-suggestions-store", "data"),
         Output("stb-stats-row", "children"),
         Output("stb-payoff-chart", "figure"),
         Output("stb-greeks-chart", "figure"),
         Output("stb-heatmap-chart", "figure"),
         Output("stb-3d-chart", "figure"),
         Output("stb-premium-table-container", "children"),
         Output("stb-scenario-chart", "figure"),
         Output("struct-smile-chart", "figure"),
         Output("stb-tenor-scan", "children"),
         Output("stb-trade-analysis", "children"),
         Output("stb-pnl-surface", "figure"),
         Output("stb-parallel-coords", "figure")] +
        [Output({"type": "stb-strike-disp", "index": i}, "children") for i in range(MAX_LEGS)] +
        [Output({"type": "stb-vol-disp", "index": i}, "children") for i in range(MAX_LEGS)] +
        [Output({"type": "stb-prem-disp", "index": i}, "children") for i in range(MAX_LEGS)],
        [Input("stb-pair", "value"),
         Input("stb-tenor", "value"),
         Input("stb-notional", "value"),
         Input("stb-num-legs", "value")] +
        [Input({"type": "stb-cp", "index": i}, "value") for i in range(MAX_LEGS)] +
        [Input({"type": "stb-side", "index": i}, "value") for i in range(MAX_LEGS)] +
        [Input({"type": "stb-delta", "index": i}, "value") for i in range(MAX_LEGS)] +
        [Input({"type": "stb-ratio", "index": i}, "value") for i in range(MAX_LEGS)] +
        [Input({"type": "stb-tenor-mult", "index": i}, "value") for i in range(MAX_LEGS)],
        [State("stb-preset", "value")],
    )
    def update_structure(pair, tenor, notional, num_legs,
                         *all_inputs):
        pair = pair or "EURUSD"
        tenor = tenor or "1M"
        notional = max(1_000, min(float(notional or 10_000_000), 1e12))
        num_legs = max(1, min(num_legs or 1, MAX_LEGS))

        # Last element is preset (State), everything before is leg inputs
        leg_inputs = all_inputs[:5 * MAX_LEGS]
        preset_name = all_inputs[5 * MAX_LEGS] if len(all_inputs) > 5 * MAX_LEGS else "Custom"

        # Parse leg inputs (5 groups of MAX_LEGS each)
        cp_vals = list(leg_inputs[0:MAX_LEGS])
        side_vals = list(leg_inputs[MAX_LEGS:2*MAX_LEGS])
        delta_vals = list(leg_inputs[2*MAX_LEGS:3*MAX_LEGS])
        ratio_vals = list(leg_inputs[3*MAX_LEGS:4*MAX_LEGS])
        tenor_mult_vals = list(leg_inputs[4*MAX_LEGS:5*MAX_LEGS])

        legs_config = []
        for i in range(num_legs):
            legs_config.append({
                "cp": cp_vals[i] or "call",
                "side": side_vals[i] or "buy",
                "delta": float(delta_vals[i]) if delta_vals[i] is not None else 0.25,
                "ratio": int(ratio_vals[i]) if ratio_vals[i] is not None else 1,
                "tenor_mult": float(tenor_mult_vals[i]) if tenor_mult_vals[i] is not None else 1.0,
            })

        # Number of new outputs before per-leg displays: 13
        n_new = 14
        empty_div = html.Div()

        # Fetch market data
        spots = get_fx_spots([pair]) or {}
        vol_surface = get_fx_vol_surface(pair) or {}
        rates = get_fx_rates(pair) or {}

        # Guard: only block if we have no spot data at all (vol/rates have fallbacks)
        if not spots or pair not in spots:
            ndf = no_data_fig(height=380, msg="NO MARKET DATA")
            empty_table = html.Div("No market data", style={"color": COLORS["text_muted"],
                                   "fontSize": "11px", "padding": "8px"})
            empty_stats = [html.Div("--", style=STAT_BOX_STYLE) for _ in range(10)]
            ndf_surface = no_data_fig(height=400, msg="NO MARKET DATA")
            return ([empty_div, [], empty_stats, ndf, ndf, ndf, ndf, empty_table, ndf, ndf,
                     empty_div, empty_div, ndf_surface, ndf]
                    + [""] * MAX_LEGS + [""] * MAX_LEGS + [""] * MAX_LEGS)

        try:
            spot_data = spots.get(pair, {"mid": 1.0, "bid": 1.0, "ask": 1.0})

            S = spot_data.get("mid", spot_data.get("bid", 1.0))
            r_d = rates.get("r_dom", 0.03)
            r_f = rates.get("r_for", 0.02)
            T = tenor_to_years(tenor)

            pip_size = 0.0001
            if pair in FX_PAIR_REGISTRY:
                pip_size = FX_PAIR_REGISTRY[pair].pip

            # Process all legs
            processed = _process_legs(legs_config, pair, tenor, notional,
                                      spot_data, rates, vol_surface)

            # Aggregates
            agg = _compute_aggregates(processed, S, T, r_d, r_f, notional, pip_size)
            atm_vol = _get_atm_vol(vol_surface, tenor)

            # ── NEW: Expected Value ──
            ev_data = None
            try:
                ev_data = _compute_expected_value(
                    processed, S, T, r_d, r_f, pair, tenor, notional, pip_size, vol_surface)
            except Exception:
                logger.debug("EV computation failed for %s %s", pair, tenor)

            # ── NEW: Suggestions + Regime ──
            suggestions_div = empty_div
            sugg_store_data = []
            try:
                sugg_list, regime, signal_pcts = _build_suggestions(
                    pair, tenor, vol_surface, spots, rates)
                # Build store: list of best structure names per suggestion
                sugg_store_data = [sg["structures"][0] for sg in sugg_list[:4]
                                   if sg.get("structures")]

                # --- Regime badge with signal percentile context ---
                regime_items = []
                if regime:
                    regime_items.append(html.Div([
                        html.Span(regime["name"], style={
                            "color": regime["color"], "fontWeight": "900",
                            "fontSize": "11px", "letterSpacing": "1px",
                        }),
                    ], style={
                        "backgroundColor": f"{regime['color']}15",
                        "border": f"1px solid {regime['color']}44",
                        "padding": "3px 10px", "marginRight": "8px",
                    }))

                # Signal percentile mini-bars
                if signal_pcts:
                    pct_names = {"atm_pct": "ATM", "rr_pct": "SKEW",
                                 "bf_pct": "WINGS", "ivrv_pct": "IV-RV"}
                    for key, label in pct_names.items():
                        val = signal_pcts.get(key, 50)
                        bar_color = ("#1565c0" if val < 25 else COLORS["accent_orange"]
                                     if val > 75 else "#3a3a5c")
                        regime_items.append(html.Div([
                            html.Span(f"{label} ", style={"color": "#555555",
                                      "fontSize": "8px", "letterSpacing": "0.5px"}),
                            html.Div(style={
                                "width": "40px", "height": "4px", "backgroundColor": "#111122",
                                "display": "inline-block", "verticalAlign": "middle",
                                "position": "relative", "marginRight": "4px",
                            }, children=[
                                html.Div(style={
                                    "width": f"{max(2, val)}%", "height": "100%",
                                    "backgroundColor": bar_color,
                                }),
                            ]),
                            html.Span(f"{val:.0f}", style={
                                "color": bar_color, "fontSize": "9px", "fontWeight": "700",
                            }),
                        ], style={"display": "inline-flex", "alignItems": "center",
                                  "gap": "2px", "marginRight": "10px"}))

                # Suggestion action buttons with rationale
                sugg_btns = []
                for si, sg in enumerate(sugg_list[:4]):
                    best_struct = sg["structures"][0] if sg["structures"] else None
                    rationale = sg.get("rationale", "")
                    sugg_btns.append(html.Div([
                        html.Button(
                            [html.Span(f"{sg['signal']} ", style={"fontWeight": "700"}),
                             html.Span(f"\u2192 {best_struct or '?'}",
                                       style={"color": COLORS["text_primary"]})],
                            id={"type": "stb-suggestion-btn", "index": si},
                            n_clicks=0,
                            style={"color": sg["color"], "fontSize": "9px", "background": "none",
                                   "border": f"1px solid {sg['color']}44",
                                   "cursor": "pointer", "padding": "3px 10px",
                                   "fontFamily": "'JetBrains Mono', monospace",
                                   "width": "100%", "textAlign": "left"}),
                        html.Div(rationale, style={
                            "fontSize": "8px", "color": "#555555", "paddingLeft": "10px",
                            "marginTop": "1px", "fontFamily": "'JetBrains Mono', monospace",
                        }) if rationale else None,
                    ], style={"flex": "1", "minWidth": "180px"}))

                all_items = []
                if regime_items:
                    all_items.append(html.Div(regime_items, style={
                        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
                        "marginBottom": "6px",
                    }))
                if sugg_btns:
                    all_items.append(html.Div(sugg_btns, style={
                        "display": "flex", "gap": "6px", "flexWrap": "wrap",
                    }))

                if all_items:
                    suggestions_div = html.Div(all_items, style={
                        "padding": "8px 10px",
                        "backgroundColor": COLORS["bg_secondary"],
                        "borderLeft": f"3px solid {regime.get('color', COLORS['text_muted'])}",
                    })
            except Exception:
                logger.debug("Suggestions failed for %s", pair)

            # ── Build charts (pass ev_data to payoff for probability overlay) ──
            payoff_fig = _build_payoff_chart(processed, agg, S, T, r_d, r_f,
                                             notional, atm_vol, ev_data=ev_data)
            greeks_fig = _build_greeks_chart(processed, S, T, r_d, r_f, notional, atm_vol)
            heatmap_fig = _build_pnl_heatmap(processed, S, T, r_d, r_f, notional, atm_vol)
            surface_fig = _build_3d_surface(processed, S, T, r_d, r_f, notional, atm_vol)
            pnl_surface_fig = _build_pnl_surface(processed, S, T, r_d, r_f, notional, atm_vol)
            parallel_fig = _build_parallel_coords(processed, S, T, r_d, r_f, notional, atm_vol)
            premium_table = _build_premium_table(processed, pair, pip_size, notional)
            scenario_fig = _build_scenario_chart(processed, S, T, r_d, r_f, notional)
            smile_fig = _build_smile_chart(processed, vol_surface, tenor)

            # ── NEW: Tenor Scan ──
            tenor_scan_div = empty_div
            try:
                ts_rows = _build_tenor_scan(legs_config, pair, notional, spots, rates, vol_surface)
                hdr_s = {"color": COLORS["text_muted"], "fontSize": "9px", "fontWeight": "600",
                         "padding": "4px 6px", "textTransform": "uppercase",
                         "fontFamily": "'JetBrains Mono', monospace",
                         "borderBottom": f"1px solid {COLORS['border_subtle']}"}
                cell_s = {"color": COLORS["text_primary"], "fontSize": "10px",
                          "padding": "4px 6px", "fontFamily": "'JetBrains Mono', monospace",
                          "borderBottom": f"1px solid {COLORS['border_subtle']}"}
                # Find cheapest vol percentile — only highlight if actually cheap (<50th)
                min_pct = min((r["vol_pctile"] for r in ts_rows), default=50)
                t_head = html.Tr([html.Th(h, style=hdr_s) for h in
                                  ["Tenor", "Prem (pips)", "POP", "Θ/day", "Vol %ile", "Breakeven"]])
                t_rows = []
                for r in ts_rows:
                    is_cheapest = r["vol_pctile"] <= min_pct + 1 and min_pct < 50
                    row_color = COLORS["accent_green"] if is_cheapest else COLORS["text_primary"]
                    be_s = f"{r['be']:.5f}" if r["be"] else "--"
                    pct_color = (COLORS["accent_green"] if r["vol_pctile"] < 25
                                 else COLORS["accent_red"] if r["vol_pctile"] > 75
                                 else COLORS["text_secondary"])
                    t_rows.append(html.Tr([
                        html.Td(r["tenor"], style={**cell_s, "color": row_color, "fontWeight": "600" if is_cheapest else "400"}),
                        html.Td(f"{r['premium_pips']:.1f}", style=cell_s),
                        html.Td(f"{r['pop']:.0f}%", style=cell_s),
                        html.Td(f"{r['theta_day']:,.0f}", style=cell_s),
                        html.Td(_ordinal(r['vol_pctile']), style={**cell_s, "color": pct_color}),
                        html.Td(be_s, style=cell_s),
                    ]))
                tenor_scan_div = html.Div([
                    html.Div("TENOR SCAN", style={"color": COLORS["text_primary"],
                             "fontSize": "11px", "fontWeight": "700", "marginBottom": "6px",
                             "fontFamily": "'JetBrains Mono', monospace",
                             "letterSpacing": "1.5px", "textTransform": "uppercase"}),
                    html.Table([html.Thead(t_head), html.Tbody(t_rows)],
                               style={"width": "100%", "borderCollapse": "collapse"}),
                ])
            except Exception:
                logger.debug("Tenor scan failed for %s", pair)

            # ── NEW: Trade Analysis ──
            trade_analysis_div = empty_div
            try:
                trade_analysis_div = _build_trade_analysis(
                    processed, agg, ev_data, pair, tenor, notional,
                    preset_name or "Custom", S, T, r_d, r_f, pip_size, vol_surface,
                    spot_data=spot_data, rates=rates, atm_vol=atm_vol)
            except Exception:
                logger.debug("Trade analysis failed for %s", pair)

            # ── Summary stat boxes (10: original 8 + implied move + EV) ──
            be_str = " / ".join(f"{b:.5f}" for b in agg["breakevens"] if np.isfinite(b)) if agg["breakevens"] else "--"
            prem_color = COLORS["pnl_profit"] if agg["net_premium"] < 0 else COLORS["pnl_loss"]
            pop_color = COLORS["accent_green"] if agg["pop"] > 50 else COLORS["accent_red"]

            # Implied move from ATM straddle
            atm_straddle = (float(_gk_price(S, S, T, r_d, r_f, atm_vol, 1))
                            + float(_gk_price(S, S, T, r_d, r_f, atm_vol, -1)))
            implied_move_pct = atm_straddle / S * 100 if S > 0 else 0
            implied_move_pips = atm_straddle / pip_size if pip_size > 0 else 0

            # Cost vs straddle
            structure_cost = abs(agg["net_premium"])
            straddle_cost = atm_straddle * notional
            cost_vs_straddle = (structure_cost / straddle_cost * 100) if straddle_cost > 0 else 0

            ev_stat_val = f"{ev_data['ev']:+,.0f}" if ev_data else "--"
            ev_stat_color = ev_data["edge_color"] if ev_data else COLORS["text_muted"]

            stats = [
                _stat_box("NET PREMIUM (PIPS)", f"{agg['net_premium_pips']:.1f}", prem_color),
                _stat_box("NET DELTA", f"{agg['net_delta']:.4f}", COLORS["accent_cyan"]),
                _stat_box("NET VEGA", f"{agg['net_vega'] * notional:.0f}", COLORS["accent_purple"]),
                _stat_box("DAILY THETA", f"{agg['net_theta'] * notional:.0f}", COLORS["accent_orange"]),
                _stat_box("BREAK-EVEN", be_str, COLORS["accent_blue"]),
                _stat_box("PROB OF PROFIT", f"{agg['pop']:.1f}%", pop_color),
                _stat_box("IMPLIED MOVE", f"\u00b1{implied_move_pct:.1f}% ({implied_move_pips:.0f}p)",
                          COLORS["accent_blue"]),
                _stat_box("EXPECTED VALUE", ev_stat_val, ev_stat_color),
                _stat_box(f"COST vs STRADDLE", f"{cost_vs_straddle:.0f}%", COLORS["text_secondary"]),
                _stat_box("MAX LOSS",
                         "UNLIMITED" if (
                             sum(lg["ratio"] * lg["side_sign"] for lg in processed if lg["cp_sign"] > 0) < -1e-9
                             or sum(lg["ratio"] * lg["side_sign"] for lg in processed if lg["cp_sign"] < 0) < -1e-9
                         ) else f"{agg['max_loss']:,.0f}",
                         COLORS["accent_red"]),
            ]

            # Per-leg display fields
            strike_disps = []
            vol_disps = []
            prem_disps = []
            for i in range(MAX_LEGS):
                if i < len(processed):
                    lg = processed[i]
                    strike_disps.append(_fmt_strike(lg['strike'], pair))
                    vol_disps.append(f"{lg['vol']*100:.1f}%")
                    prem_disps.append(f"{lg['premium_pips']:.1f}p")
                else:
                    strike_disps.append("")
                    vol_disps.append("")
                    prem_disps.append("")

            return ([suggestions_div, sugg_store_data, stats, payoff_fig, greeks_fig,
                     heatmap_fig, surface_fig, premium_table, scenario_fig, smile_fig,
                     tenor_scan_div, trade_analysis_div, pnl_surface_fig, parallel_fig]
                    + strike_disps + vol_disps + prem_disps)

        except Exception as e:
            logger.error("Structure builder callback failed: %s", e, exc_info=True)
            err_fig = no_data_fig(height=380, msg=f"Error: {type(e).__name__}: {str(e)[:100]}")
            err_surface = no_data_fig(height=400, msg=f"Error: {type(e).__name__}")
            empty_table = html.Div(f"Error: {e}", style={"color": COLORS["accent_red"],
                                   "fontSize": "11px", "padding": "8px"})
            empty_stats = [html.Div("--", style=STAT_BOX_STYLE) for _ in range(10)]
            return ([empty_div, [], empty_stats, err_fig, err_fig, err_fig, err_fig,
                     empty_table, err_fig, err_fig, empty_div, empty_div, err_surface, err_fig]
                    + [""] * MAX_LEGS + [""] * MAX_LEGS + [""] * MAX_LEGS)

    # -- Historical cost context callback --
    @app.callback(
        Output("stb-historical-cost", "children"),
        [Input("stb-pair", "value"),
         Input("stb-tenor", "value"),
         Input("stb-stats-row", "children")],
    )
    def update_historical_cost(pair, tenor, _stats_trigger):
        """
        Show where the current structure premium sits vs its 252-day range.
        Uses ATM vol percentile of the primary pair/tenor as a proxy for
        structure cost percentile.
        """
        pair = pair or "EURUSD"
        tenor = tenor or "1M"

        try:
            vp = vol_percentile(pair, tenor, "ATM", 252)
            if vp is None:
                raise ValueError("No data")
        except Exception:
            return html.Div(
                "Historical cost data unavailable",
                style={"color": COLORS["text_muted"], "fontSize": "11px",
                       "fontStyle": "italic", "padding": "6px"},
            )

        pct = vp.get("percentile", 50)
        current = vp.get("current", 0)
        vol_min = vp.get("min", 0)
        vol_max = vp.get("max", 0)
        vol_mean = vp.get("mean", 0)

        # Determine colour based on percentile (cheap = green, expensive = red)
        if pct <= 25:
            pct_color = COLORS["accent_green"]
            pct_label = "CHEAP"
        elif pct <= 50:
            pct_color = COLORS["accent_cyan"]
            pct_label = "BELOW AVG"
        elif pct <= 75:
            pct_color = COLORS["accent_orange"]
            pct_label = "ABOVE AVG"
        else:
            pct_color = COLORS["accent_red"]
            pct_label = "EXPENSIVE"

        header = html.Div("HISTORICAL COST CONTEXT", style={
            "color": COLORS["text_primary"],
            "fontSize": "11px",
            "fontWeight": "700",
            "fontFamily": "'JetBrains Mono', monospace",
            "marginBottom": "8px",
            "letterSpacing": "1.5px",
            "textTransform": "uppercase",
        })

        subtitle = html.Div(
            f"Current premium sits at the {_ordinal(pct)} percentile of its "
            f"252-day range  ({pct_label})",
            style={
                "color": pct_color,
                "fontSize": "11px",
                "fontWeight": "600",
                "fontFamily": "'JetBrains Mono', monospace",
                "marginBottom": "10px",
            },
        )

        stat_row = html.Div([
            clickable_stat(f"{current:.2f}", "CURRENT ATM VOL",
                           pair, "ATM", tenor, color=COLORS["accent_cyan"]),
            clickable_stat(_ordinal(pct), "PERCENTILE",
                           pair, "ATM", tenor, color=pct_color),
            clickable_stat(f"{vol_min:.2f}", "252D MIN",
                           pair, "ATM", tenor, color=COLORS["accent_green"]),
            clickable_stat(f"{vol_max:.2f}", "252D MAX",
                           pair, "ATM", tenor, color=COLORS["accent_red"]),
            clickable_stat(f"{vol_mean:.2f}", "252D MEAN",
                           pair, "ATM", tenor, color=COLORS["accent_blue"]),
        ], style={
            "display": "flex",
            "gap": "8px",
            "flexWrap": "wrap",
        })

        return html.Div([header, subtitle, stat_row])

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output("stb-csv-download", "data"),
        [Input("stb-csv-payoff", "n_clicks"),
         Input("stb-csv-greeks", "n_clicks"),
         Input("stb-csv-heatmap", "n_clicks"),
         Input("stb-csv-3d", "n_clicks"),
         Input("stb-csv-scenario", "n_clicks")],
        [State("stb-payoff-chart", "figure"),
         State("stb-greeks-chart", "figure"),
         State("stb-heatmap-chart", "figure"),
         State("stb-3d-chart", "figure"),
         State("stb-scenario-chart", "figure")],
        prevent_initial_call=True,
    )
    def stb_csv_export(n1, n2, n3, n4, n5, fig1, fig2, fig3, fig4, fig5):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        mapping = {
            "stb-csv-payoff": (fig1, "StructBuilder", "Payoff"),
            "stb-csv-greeks": (fig2, "StructBuilder", "Greeks"),
            "stb-csv-heatmap": (fig3, "StructBuilder", "Heatmap"),
            "stb-csv-3d": (fig4, "StructBuilder", "Surface3D"),
            "stb-csv-scenario": (fig5, "StructBuilder", "Scenario"),
        }
        if btn not in mapping:
            return no_update
        fig, panel, chart_type = mapping[btn]
        if not fig or not fig.get("data"):
            return no_update
        return export_csv(fig, panel, chart_type)

    # ── Helper: extract legs config from State values ──────────────────
    def _extract_legs(num_legs, leg_states):
        """Parse the flat list of leg State values into legs_config."""
        n = max(1, min(num_legs or 1, MAX_LEGS))
        cp_v = list(leg_states[0:MAX_LEGS])
        side_v = list(leg_states[MAX_LEGS:2*MAX_LEGS])
        delta_v = list(leg_states[2*MAX_LEGS:3*MAX_LEGS])
        ratio_v = list(leg_states[3*MAX_LEGS:4*MAX_LEGS])
        tmult_v = list(leg_states[4*MAX_LEGS:5*MAX_LEGS])
        cfg = []
        for i in range(n):
            cfg.append({
                "cp": cp_v[i] or "call",
                "side": side_v[i] or "buy",
                "delta": float(delta_v[i]) if delta_v[i] is not None else 0.25,
                "ratio": int(ratio_v[i]) if ratio_v[i] is not None else 1,
                "tenor_mult": float(tmult_v[i]) if tmult_v[i] is not None else 1.0,
            })
        return cfg

    # All leg States used by multiple callbacks
    _LEG_STATES = (
        [State({"type": "stb-cp", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State({"type": "stb-side", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State({"type": "stb-delta", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State({"type": "stb-ratio", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State({"type": "stb-tenor-mult", "index": i}, "value") for i in range(MAX_LEGS)]
    )

    # ══════════════════════════════════════════════════════════════════════
    # SOLVER CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("stb-solver-result", "children"),
         Output("stb-solver-apply-btn", "style"),
         Output("stb-solver-store", "data")],
        Input("stb-solve-btn", "n_clicks"),
        [State("stb-solver-target", "value"),
         State("stb-solver-value", "value"),
         State("stb-solver-leg", "value"),
         State("stb-solver-param", "value"),
         State("stb-pair", "value"),
         State("stb-tenor", "value"),
         State("stb-notional", "value"),
         State("stb-num-legs", "value")] + _LEG_STATES,
        prevent_initial_call=True,
    )
    def solve_parameter(n_clicks, target_metric, target_value, solve_leg,
                        solve_param, pair, tenor, notional, num_legs, *leg_states):
        if not n_clicks:
            raise PreventUpdate
        hidden = {"display": "none"}
        shown = {**BUTTON_STYLE, "fontSize": "9px", "padding": "5px 12px", "flex": "1"}

        pair = pair or "EURUSD"
        tenor = tenor or "1M"
        notional = max(1_000, min(float(notional or 10_000_000), 1e12))
        target_value = float(target_value or 0)
        solve_leg = int(solve_leg or 0)

        legs_config = _extract_legs(num_legs, leg_states)
        if solve_leg >= len(legs_config):
            return "Leg out of range", hidden, None

        spots = get_fx_spots([pair]) or {}
        rates = get_fx_rates(pair) or {}
        vol_surface = get_fx_vol_surface(pair) or {}
        spot_data = spots.get(pair, {"mid": 1.0})

        result = _solve_for_parameter(
            target_metric, target_value, solve_leg, solve_param,
            legs_config, pair, tenor, notional, spot_data, rates, vol_surface)

        if result["success"]:
            store = {"leg": solve_leg, "param": solve_param, "value": result["value"]}
            return (
                html.Span(f"\u2713 {result['message']}",
                          style={"color": COLORS["accent_green"]}),
                shown,
                store,
            )
        return (
            html.Span(f"\u2716 {result['message']}",
                      style={"color": COLORS["accent_red"]}),
            hidden,
            None,
        )

    @app.callback(
        [Output({"type": "stb-delta", "index": i}, "value", allow_duplicate=True)
         for i in range(MAX_LEGS)] +
        [Output({"type": "stb-ratio", "index": i}, "value", allow_duplicate=True)
         for i in range(MAX_LEGS)],
        Input("stb-solver-apply-btn", "n_clicks"),
        State("stb-solver-store", "data"),
        prevent_initial_call=True,
    )
    def apply_solver(n_clicks, store):
        if not n_clicks or not store:
            raise PreventUpdate
        leg_idx = store["leg"]
        param = store["param"]
        val = store["value"]
        delta_out = [no_update] * MAX_LEGS
        ratio_out = [no_update] * MAX_LEGS
        if param == "delta" and 0 <= leg_idx < MAX_LEGS:
            delta_out[leg_idx] = val
        elif param == "ratio" and 0 <= leg_idx < MAX_LEGS:
            ratio_out[leg_idx] = val
        return delta_out + ratio_out

    # ══════════════════════════════════════════════════════════════════════
    # QUICK ACTION CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output({"type": "stb-side", "index": i}, "value", allow_duplicate=True)
         for i in range(MAX_LEGS)],
        Input("stb-flip-btn", "n_clicks"),
        [State({"type": "stb-side", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State("stb-num-legs", "value")],
        prevent_initial_call=True,
    )
    def flip_sides(n_clicks, *args):
        if not n_clicks:
            raise PreventUpdate
        sides = args[:MAX_LEGS]
        num_legs = args[MAX_LEGS] or 1
        return [("sell" if sides[i] == "buy" else "buy") if i < num_legs
                else no_update for i in range(MAX_LEGS)]

    @app.callback(
        [Output({"type": "stb-cp", "index": i}, "value", allow_duplicate=True)
         for i in range(MAX_LEGS)],
        Input("stb-mirror-btn", "n_clicks"),
        [State({"type": "stb-cp", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State("stb-num-legs", "value")],
        prevent_initial_call=True,
    )
    def mirror_cp(n_clicks, *args):
        if not n_clicks:
            raise PreventUpdate
        cps = args[:MAX_LEGS]
        num_legs = args[MAX_LEGS] or 1
        return [("put" if cps[i] == "call" else "call") if i < num_legs
                else no_update for i in range(MAX_LEGS)]

    @app.callback(
        [Output({"type": "stb-delta", "index": i}, "value", allow_duplicate=True)
         for i in range(MAX_LEGS)] +
        [Output("stb-solver-result", "children", allow_duplicate=True)],
        Input("stb-zero-cost-btn", "n_clicks"),
        [State("stb-pair", "value"),
         State("stb-tenor", "value"),
         State("stb-notional", "value"),
         State("stb-num-legs", "value")] + _LEG_STATES,
        prevent_initial_call=True,
    )
    def zero_cost(n_clicks, pair, tenor, notional, num_legs, *leg_states):
        if not n_clicks:
            raise PreventUpdate
        pair = pair or "EURUSD"
        tenor = tenor or "1M"
        notional = max(1_000, min(float(notional or 10_000_000), 1e12))

        legs_config = _extract_legs(num_legs, leg_states)

        # Find first short leg
        sell_idx = next((i for i, lg in enumerate(legs_config) if lg["side"] == "sell"), None)
        if sell_idx is None:
            return [no_update] * MAX_LEGS + [
                html.Span("No short leg to adjust", style={"color": COLORS["accent_red"]})]

        spots = get_fx_spots([pair]) or {}
        rates = get_fx_rates(pair) or {}
        vol_surface = get_fx_vol_surface(pair) or {}
        spot_data = spots.get(pair, {"mid": 1.0})

        result = _solve_for_parameter(
            "net_premium_pips", 0.0, sell_idx, "delta",
            legs_config, pair, tenor, notional, spot_data, rates, vol_surface)

        delta_out = [no_update] * MAX_LEGS
        if result["success"]:
            delta_out[sell_idx] = result["value"]
            msg = html.Span(f"\u2713 Zero-cost: L{sell_idx+1} \u0394={result['value']}",
                            style={"color": COLORS["accent_green"]})
        else:
            msg = html.Span(f"\u2716 {result['message']}",
                            style={"color": COLORS["accent_red"]})
        return delta_out + [msg]

    # ══════════════════════════════════════════════════════════════════════
    # COMPARE MODE CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("stb-compare-a", "data"),
         Output("stb-compare-status", "children", allow_duplicate=True)],
        Input("stb-save-a", "n_clicks"),
        [State("stb-pair", "value"), State("stb-tenor", "value"),
         State("stb-notional", "value"), State("stb-preset", "value"),
         State("stb-num-legs", "value")] + _LEG_STATES,
        prevent_initial_call=True,
    )
    def save_compare_a(n_clicks, pair, tenor, notional, preset, num_legs, *leg_states):
        if not n_clicks:
            raise PreventUpdate
        pair = pair or "EURUSD"
        tenor = tenor or "1M"
        notional = max(1_000, min(float(notional or 10_000_000), 1e12))
        legs_config = _extract_legs(num_legs, leg_states)

        spots = get_fx_spots([pair]) or {}
        rates = get_fx_rates(pair) or {}
        vol_surface = get_fx_vol_surface(pair) or {}
        spot_data = spots.get(pair, {"mid": 1.0})
        S = spot_data.get("mid", 1.0)
        T = tenor_to_years(tenor)
        r_d = rates.get("r_dom", 0.03)
        r_f = rates.get("r_for", 0.02)
        pip_size = FX_PAIR_REGISTRY[pair].pip if pair in FX_PAIR_REGISTRY else 0.0001

        proc = _process_legs(legs_config, pair, tenor, notional, spot_data, rates, vol_surface)
        agg = _compute_aggregates(proc, S, T, r_d, r_f, notional, pip_size)

        snap = {
            "label": f"{tenor} {pair} {preset or 'Custom'}",
            "spot_range": agg["spot_range"].tolist(),
            "expiry_pnl": agg["expiry_pnl"].tolist(),
            "breakevens": agg["breakevens"],
            "prem_pips": agg["net_premium_pips"],
            "pop": agg["pop"],
            "net_delta": agg["net_delta"],
            "net_vega": agg["net_vega"] * notional,
            "daily_theta": agg["net_theta"] * notional,
            "max_loss": agg["max_loss"],
        }
        return snap, f"A: {snap['label']}"

    @app.callback(
        [Output("stb-compare-b", "data"),
         Output("stb-compare-status", "children", allow_duplicate=True)],
        Input("stb-save-b", "n_clicks"),
        [State("stb-pair", "value"), State("stb-tenor", "value"),
         State("stb-notional", "value"), State("stb-preset", "value"),
         State("stb-num-legs", "value")] + _LEG_STATES,
        prevent_initial_call=True,
    )
    def save_compare_b(n_clicks, pair, tenor, notional, preset, num_legs, *leg_states):
        if not n_clicks:
            raise PreventUpdate
        pair = pair or "EURUSD"
        tenor = tenor or "1M"
        notional = max(1_000, min(float(notional or 10_000_000), 1e12))
        legs_config = _extract_legs(num_legs, leg_states)

        spots = get_fx_spots([pair]) or {}
        rates = get_fx_rates(pair) or {}
        vol_surface = get_fx_vol_surface(pair) or {}
        spot_data = spots.get(pair, {"mid": 1.0})
        S = spot_data.get("mid", 1.0)
        T = tenor_to_years(tenor)
        r_d = rates.get("r_dom", 0.03)
        r_f = rates.get("r_for", 0.02)
        pip_size = FX_PAIR_REGISTRY[pair].pip if pair in FX_PAIR_REGISTRY else 0.0001

        proc = _process_legs(legs_config, pair, tenor, notional, spot_data, rates, vol_surface)
        agg = _compute_aggregates(proc, S, T, r_d, r_f, notional, pip_size)

        snap = {
            "label": f"{tenor} {pair} {preset or 'Custom'}",
            "spot_range": agg["spot_range"].tolist(),
            "expiry_pnl": agg["expiry_pnl"].tolist(),
            "breakevens": agg["breakevens"],
            "prem_pips": agg["net_premium_pips"],
            "pop": agg["pop"],
            "net_delta": agg["net_delta"],
            "net_vega": agg["net_vega"] * notional,
            "daily_theta": agg["net_theta"] * notional,
            "max_loss": agg["max_loss"],
        }
        return snap, f"B: {snap['label']}"

    @app.callback(
        [Output("stb-compare-section", "style", allow_duplicate=True),
         Output("stb-compare-section", "children", allow_duplicate=True)],
        Input("stb-compare-toggle", "n_clicks"),
        [State("stb-compare-a", "data"),
         State("stb-compare-b", "data"),
         State("stb-notional", "value")],
        prevent_initial_call=True,
    )
    def toggle_compare(n_clicks, snap_a, snap_b, notional):
        if not n_clicks:
            raise PreventUpdate
        # Toggle on odd clicks
        show = (n_clicks % 2) == 1
        if not show or not snap_a or not snap_b:
            return {"display": "none"}, html.Div()
        notional = float(notional or 10_000_000)
        content = _build_compare_overlay(snap_a, snap_b, notional)
        return {**CARD_STYLE, "padding": "12px", "marginTop": "12px"}, content

    @app.callback(
        [Output("stb-compare-a", "data", allow_duplicate=True),
         Output("stb-compare-b", "data", allow_duplicate=True),
         Output("stb-compare-section", "style", allow_duplicate=True),
         Output("stb-compare-status", "children", allow_duplicate=True)],
        Input("stb-compare-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_compare(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return None, None, {"display": "none"}, ""

    # ══════════════════════════════════════════════════════════════════════
    # SUGGESTION CLICK → LOAD PRESET
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("stb-preset", "value", allow_duplicate=True),
        Input({"type": "stb-suggestion-btn", "index": ALL}, "n_clicks"),
        State("stb-suggestions-store", "data"),
        prevent_initial_call=True,
    )
    def load_suggestion(n_clicks_list, sugg_names):
        if not n_clicks_list or all(n == 0 or n is None for n in n_clicks_list):
            raise PreventUpdate
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        # Find which button was clicked
        import json as _json
        prop_id = ctx.triggered[0]["prop_id"]
        try:
            idx = _json.loads(prop_id.split(".")[0])["index"]
        except Exception:
            raise PreventUpdate
        if not sugg_names or idx >= len(sugg_names):
            raise PreventUpdate
        preset_name = sugg_names[idx]
        if preset_name not in PRESETS:
            raise PreventUpdate
        return preset_name

    # ══════════════════════════════════════════════════════════════════════
    # SEND TO BLOTTER
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("stb-to-blotter-store", "data"),
         Output("stb-send-status", "children")],
        [Input("stb-send-to-blotter", "n_clicks")],
        [State("stb-pair", "value"),
         State("stb-tenor", "value"),
         State("stb-notional", "value"),
         State("stb-num-legs", "value"),
         State("stb-preset", "value")] +
        [State({"type": "stb-cp", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State({"type": "stb-side", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State({"type": "stb-delta", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State({"type": "stb-ratio", "index": i}, "value") for i in range(MAX_LEGS)] +
        [State({"type": "stb-tenor-mult", "index": i}, "value") for i in range(MAX_LEGS)],
        prevent_initial_call=True,
    )
    def send_to_blotter(n_clicks, pair, tenor, notional, num_legs, preset_name,
                        *leg_inputs):
        if not n_clicks:
            raise PreventUpdate

        pair = pair or "EURUSD"
        tenor = tenor or "1M"
        notional = max(1_000, min(float(notional or 10_000_000), 1e12))
        num_legs = max(1, min(num_legs or 1, MAX_LEGS))

        # Parse leg inputs (same pattern as update_structure)
        cp_vals = list(leg_inputs[0:MAX_LEGS])
        side_vals = list(leg_inputs[MAX_LEGS:2*MAX_LEGS])
        delta_vals = list(leg_inputs[2*MAX_LEGS:3*MAX_LEGS])
        ratio_vals = list(leg_inputs[3*MAX_LEGS:4*MAX_LEGS])
        tenor_mult_vals = list(leg_inputs[4*MAX_LEGS:5*MAX_LEGS])

        legs_config = []
        for i in range(num_legs):
            legs_config.append({
                "cp": cp_vals[i] or "call",
                "side": side_vals[i] or "buy",
                "delta": float(delta_vals[i]) if delta_vals[i] is not None else 0.25,
                "ratio": int(ratio_vals[i]) if ratio_vals[i] is not None else 1,
                "tenor_mult": float(tenor_mult_vals[i]) if tenor_mult_vals[i] is not None else 1.0,
            })

        # Fetch market data and price legs
        spots = get_fx_spots([pair]) or {}
        vol_surface = get_fx_vol_surface(pair) or {}
        rates = get_fx_rates(pair) or {}
        spot_data = spots.get(pair, {})

        if not spot_data:
            return no_update, "No market data — cannot send"

        try:
            processed = _process_legs(legs_config, pair, tenor, notional,
                                      spot_data, rates, vol_surface)
        except Exception as e:
            return no_update, f"Pricing failed: {str(e)[:60]}"
        if not processed:
            return no_update, "No valid legs to send"

        # Build transfer payload
        T_base = tenor_to_years(tenor)
        transfer_legs = []
        for lg in processed:
            leg_tenor = years_to_nearest_tenor(lg["T"]) if lg.get("tenor_mult", 1.0) != 1.0 else tenor
            transfer_legs.append({
                "cp": lg["cp"],
                "side": lg["side"],
                "strike": round(lg["strike"], 6),
                "delta": round(lg["delta"], 4),
                "vol": lg["vol"],
                "notional": int(notional * lg.get("ratio", 1)),
                "ratio": lg.get("ratio", 1),
                "tenor": leg_tenor,
                "tenor_mult": lg.get("tenor_mult", 1.0),
                "T": lg["T"],
                "r_d": lg["r_d"],
                "r_f": lg["r_f"],
                "price_unit": lg["price_unit"],
                "premium_total": round(lg["premium_total"], 2),
                "S": lg["S"],
                "greeks": {
                    "delta": round(lg["delta"], 4),
                    "gamma": round(lg["gamma"], 6),
                    "vega": round(lg["vega"], 2),
                    "theta": round(lg["theta"], 2),
                },
            })

        payload = {
            "structure_name": preset_name or "Custom",
            "pair": pair,
            "tenor": tenor,
            "notional": int(notional),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "legs": transfer_legs,
        }

        n_legs = len(transfer_legs)
        return payload, f"Sent {n_legs} leg{'s' if n_legs != 1 else ''} to Blotter"

    # ══════════════════════════════════════════════════════════════════════
    # SAVE / RECALL STRUCTURES (local storage)
    # ══════════════════════════════════════════════════════════════════════

    # Save is handled via a clientside callback writing to stb-saved-structures
    # (dcc.Store with storage_type="local").  For now, the save/recall UI is
    # deferred — the store is in place for future use.
