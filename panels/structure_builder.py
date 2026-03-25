"""
Strategy Construction Lab Panel
================================
FX options multi-leg structure builder with delta-based leg configuration,
Garman-Kohlhagen pricing, full Greeks decomposition, payoff diagrams,
P&L heatmaps, 3D surfaces, and scenario sensitivity analysis.

Supports 25 preset structures plus fully custom multi-leg construction
for institutional FX options workflows.
"""

import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update, ALL, MATCH
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, INPUT_STYLE, BUTTON_STYLE,
    make_stat_style, clickable_stat,
    CSV_BTN_STYLE, no_data_fig,
)
from core.csv_export import export_csv
from core.fx_analytics import vol_percentile
from core.bloomberg_fx import get_fx_vol_surface, get_fx_spots, get_fx_rates, get_all_pairs
from core.fx_conventions import (
    tenor_to_years, years_to_nearest_tenor, delta_to_strike, atm_dns_strike, bf_rr_to_smile,
    FX_PAIR_REGISTRY, forward_points, fx_forward, premium_pips, premium_pct_notional,
)


# ============================================================================
# Tenor List
# ============================================================================

TENORS = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]


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
        {"cp": "call", "side": "buy", "delta": 0.35, "ratio": 1},
        {"cp": "call", "side": "sell", "delta": 0.50, "ratio": 2},
        {"cp": "call", "side": "buy", "delta": 0.65, "ratio": 1},
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
}


MAX_LEGS = 8


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
    S_f = float(S)
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
    rho_d = cp_f * K_f * T_safe * exp_rd * nd2 / 100.0
    rho_f = -cp_f * S_f * T_safe * exp_rf * nd1 / 100.0

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_daily,
        "rho_d": rho_d,
        "rho_f": rho_f,
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
        available = sorted(vol_surface_data.keys(),
                           key=lambda t: abs(tenor_to_years(t) - tenor_to_years(tenor)))
        tenor = available[0] if available else list(vol_surface_data.keys())[0]

    q = vol_surface_data[tenor]
    # bf_rr_to_smile expects vol-point inputs (e.g., 8.5, -0.3, 0.25)
    atm = q["atm"]
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


def _get_atm_vol(vol_surface_data, tenor):
    """Get ATM vol for a tenor in decimal form."""
    if not vol_surface_data:
        return 0.08
    if tenor not in vol_surface_data:
        available = sorted(vol_surface_data.keys(),
                           key=lambda t: abs(tenor_to_years(t) - tenor_to_years(tenor)))
        tenor = available[0] if available else list(vol_surface_data.keys())[0]
    return vol_surface_data[tenor]["atm"] / 100.0


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
    r_f = rates.get("r_for", 0.01)
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

        target_delta = delta_abs * cp_sign
        K = delta_to_strike(target_delta, S, leg_T, r_d, r_f, vol, cp_sign)
        if np.isnan(K) or K <= 0:
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
    log_spots = np.log(spot_range / S)
    pdf_vals = np.exp(-0.5 * ((log_spots - mu_T) / sigma_T) ** 2) / (sigma_T * np.sqrt(2 * np.pi) * spot_range)
    total_area = np.trapezoid(pdf_vals, spot_range)
    if total_area > 0:
        pdf_vals /= total_area
    profitable_mask = expiry_pnl > 0
    pop = float(np.trapezoid(pdf_vals * profitable_mask, spot_range) * 100.0)
    pop = max(0.0, min(100.0, pop))

    return {
        "net_premium": net_premium,
        "net_premium_pips": net_premium_pips,
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_vega": net_vega,
        "net_theta": net_theta,
        "breakevens": breakevens,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "pop": pop,
        "spot_range": spot_range,
        "expiry_pnl": expiry_pnl,
    }


# ============================================================================
# Chart Builders
# ============================================================================

def _build_payoff_chart(processed_legs, agg, S, T, r_d, r_f, notional, atm_vol):
    """Chart 1: Payoff diagram at expiry, T*0.5, and today."""
    tpl = CHART_TEMPLATE["layout"]
    spot_range = np.linspace(S * 0.85, S * 1.15, 400)
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
            for j, s in enumerate(spot_range):
                p_half = float(_gk_price(s, K, max(T * 0.5, 1e-6), r_d, r_f, vol, cp))
                p_now = float(_gk_price(s, K, T, r_d, r_f, vol, cp))
                pnl_half[j] += p_half * qty
                pnl_now[j] += p_now * qty

    pnl_expiry = (pnl_expiry - net_prem_per_unit) * notional
    pnl_half = (pnl_half - net_prem_per_unit) * notional
    pnl_now = (pnl_now - net_prem_per_unit) * notional

    fig = go.Figure()

    # 1-sigma / 2-sigma expected move shading
    move_1s = S * atm_vol * np.sqrt(max(T, 1e-4))
    move_2s = 2.0 * move_1s
    fig.add_vrect(x0=S - move_2s, x1=S + move_2s,
                  fillcolor="rgba(59,130,246,0.03)", line_width=0)
    fig.add_vrect(x0=S - move_1s, x1=S + move_1s,
                  fillcolor="rgba(59,130,246,0.06)", line_width=0,
                  annotation_text="1\u03c3", annotation_position="top left",
                  annotation_font=dict(color=COLORS["text_muted"], size=9))

    # Expiry P&L (bold)
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_expiry, mode="lines",
        name="At Expiry", line=dict(color=COLORS["accent_cyan"], width=3),
        fill="tozeroy", fillcolor="rgba(6,182,212,0.06)",
    ))
    # T*0.5
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_half, mode="lines",
        name="T\u00d70.5", line=dict(color=COLORS["accent_purple"], width=2, dash="dash"),
    ))
    # Today (thin)
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_now, mode="lines",
        name="Today", line=dict(color=COLORS["accent_orange"], width=1.5, dash="dot"),
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
    fig.add_vline(x=S, line=dict(color=COLORS["accent_blue"], width=1, dash="dash"),
                  annotation_text=f"Spot {S:.4f}",
                  annotation_font=dict(color=COLORS["accent_blue"], size=9))

    # Breakeven lines
    for be in agg["breakevens"]:
        fig.add_vline(x=be, line=dict(color=COLORS["accent_orange"], width=1, dash="dashdot"),
                      annotation_text=f"BE {be:.4f}",
                      annotation_font=dict(color=COLORS["accent_orange"], size=8))

    fig.update_layout(
        title=dict(text="PAYOFF DIAGRAM", font=dict(color=COLORS["text_primary"], size=13)),
        xaxis_title="Spot", yaxis_title="P&L",
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=55, r=15, t=40, b=35),
        legend=dict(font=dict(color=COLORS["text_secondary"], size=10),
                    bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
        hoverlabel=tpl["hoverlabel"],
        xaxis=dict(gridcolor="rgba(30,42,69,0.5)"),
        yaxis=dict(gridcolor="rgba(30,42,69,0.5)"),
        height=380,
    )
    return fig


def _build_greeks_chart(processed_legs, S, T, r_d, r_f, notional):
    """Chart 2: 2x2 subplot of Delta, Gamma, Vega, Theta vs spot."""
    tpl = CHART_TEMPLATE["layout"]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Delta", "Gamma", "Vega", "Theta"),
                        vertical_spacing=0.14, horizontal_spacing=0.10)

    spot_grid = np.linspace(S * 0.85, S * 1.15, 150)
    greek_names = ["delta", "gamma", "vega", "theta"]
    greek_colors = [COLORS["accent_cyan"], COLORS["accent_blue"],
                    COLORS["accent_purple"], COLORS["accent_orange"]]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for gi, (gname, color, (row, col)) in enumerate(zip(greek_names, greek_colors, positions)):
        total_vals = np.zeros(len(spot_grid))

        for li, lg in enumerate(processed_legs):
            cp = lg["cp_sign"]
            K = lg["strike"]
            vol = lg["vol"]
            qty = lg["side_sign"] * lg["ratio"]
            leg_vals = np.zeros(len(spot_grid))

            for j, s in enumerate(spot_grid):
                g = _gk_greeks(s, K, T, r_d, r_f, vol, cp)
                leg_vals[j] = g[gname] * qty

            total_vals += leg_vals

            # Individual leg traces (thin dotted)
            fig.add_trace(go.Scatter(
                x=spot_grid, y=leg_vals * notional, mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                name=f"L{li+1} {gname}", showlegend=False, opacity=0.4,
            ), row=row, col=col)

        # Structure total (bold)
        fig.add_trace(go.Scatter(
            x=spot_grid, y=total_vals * notional, mode="lines",
            line=dict(color=color, width=2.5),
            name=gname.capitalize(), showlegend=False,
        ), row=row, col=col)

        fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"),
                      row=row, col=col)

    fig.update_layout(
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=50, r=15, t=35, b=30),
        hoverlabel=tpl["hoverlabel"], height=380,
    )
    fig.update_xaxes(gridcolor="rgba(30,42,69,0.5)")
    fig.update_yaxes(gridcolor="rgba(30,42,69,0.5)")
    for ann in fig.layout.annotations:
        ann.font.color = COLORS["text_primary"]
        ann.font.size = 11
    return fig


def _build_pnl_heatmap(processed_legs, S, T, r_d, r_f, notional):
    """Chart 3: P&L heatmap -- spot shock x vol shock."""
    tpl = CHART_TEMPLATE["layout"]
    spot_shocks = np.linspace(-0.15, 0.15, 25)
    vol_shocks = np.linspace(-0.50, 0.50, 25)
    pnl_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))

    net_prem_per_unit = sum(
        lg["price_unit"] * lg["side_sign"] * lg["ratio"]
        for lg in processed_legs
    )

    for vi, dv in enumerate(vol_shocks):
        for si, ds in enumerate(spot_shocks):
            s_shocked = S * (1.0 + ds)
            pnl = 0.0
            for lg in processed_legs:
                cp = lg["cp_sign"]
                K = lg["strike"]
                vol_shocked = max(lg["vol"] * (1.0 + dv), 0.005)
                qty = lg["side_sign"] * lg["ratio"]
                pnl += float(_gk_price(s_shocked, K, T, r_d, r_f, vol_shocked, cp)) * qty
            pnl_matrix[vi, si] = (pnl - net_prem_per_unit) * notional

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
        ),
        hovertemplate="Spot: %{x}<br>Vol: %{y}<br>P&L: %{z:,.0f}<extra></extra>",
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


def _build_3d_surface(processed_legs, S, T, r_d, r_f, notional):
    """Chart 4: 3D P&L surface -- Spot x Time x P&L."""
    tpl = CHART_TEMPLATE["layout"]
    n_spot = 40
    n_time = 30
    spot_grid = np.linspace(S * 0.85, S * 1.15, n_spot)
    time_grid = np.linspace(T, max(T * 0.01, 1.0 / 365.0), n_time)

    net_prem_per_unit = sum(
        lg["price_unit"] * lg["side_sign"] * lg["ratio"]
        for lg in processed_legs
    )

    pnl_surface = np.zeros((n_time, n_spot))
    for ti, t_val in enumerate(time_grid):
        for si, s_val in enumerate(spot_grid):
            pnl = 0.0
            for lg in processed_legs:
                cp = lg["cp_sign"]
                K = lg["strike"]
                vol = lg["vol"]
                qty = lg["side_sign"] * lg["ratio"]
                pnl += float(_gk_price(s_val, K, max(t_val, 1e-6), r_d, r_f, vol, cp)) * qty
            pnl_surface[ti, si] = (pnl - net_prem_per_unit) * notional

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
               "Prem (pips)", "Prem (%)", "Ratio", "Net Prem"]
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
            html.Td(f"{lg['strike']:.5f}", style=cell_style),
            html.Td(f"{lg['vol']*100:.2f}", style=cell_style),
            html.Td(f"{lg['premium_pips']:.1f}", style=cell_style),
            html.Td(f"{lg['premium_pct']:.3f}", style=cell_style),
            html.Td(f"{lg['ratio']}", style=cell_style),
            html.Td(f"{lg['premium_total']:,.0f}", style={
                **cell_style,
                "color": buy_color if lg["premium_total"] < 0 else sell_color,
            }),
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
            "color": COLORS["pnl_profit"] if net_total < 0 else COLORS["pnl_loss"],
        }),
    ]))

    table = html.Table(
        [html.Thead(header_row), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "borderRadius": "8px"},
    )
    return html.Div([
        html.Div("PREMIUM & COST TABLE",
                 style={"color": COLORS["text_primary"], "fontSize": "13px",
                        "fontWeight": "700", "fontFamily": "'JetBrains Mono', monospace",
                        "marginBottom": "10px", "letterSpacing": "1.5px",
                        "textTransform": "uppercase"}),
        table,
    ], style={"overflowX": "auto"})


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
        x=[f"{v:+.0%}" for v in vol_shifts], y=pnl_vol, mode="lines",
        line=dict(color=COLORS["accent_purple"], width=2),
        name="Vol Sensitivity", showlegend=False,
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
        x=[f"{m:+.1%}" for m in spot_moves], y=pnl_spot, mode="lines",
        line=dict(color=COLORS["accent_cyan"], width=2),
        name="Spot Sensitivity", showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=0.5, dash="dot"), row=3, col=1)

    fig.update_layout(
        paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
        font=tpl["font"], margin=dict(l=55, r=15, t=35, b=30),
        hoverlabel=tpl["hoverlabel"], height=380,
    )
    fig.update_xaxes(gridcolor="rgba(30,42,69,0.5)")
    fig.update_yaxes(gridcolor="rgba(30,42,69,0.5)")
    for ann in fig.layout.annotations:
        ann.font.color = COLORS["text_primary"]
        ann.font.size = 11
    return fig


# ============================================================================
# Stat Box Helper
# ============================================================================

def _stat_box(label, value, color=COLORS["accent_cyan"]):
    """Render a single summary stat box."""
    return html.Div([
        html.Div(str(value),
                 style={"color": color, "fontSize": "16px", "fontWeight": "700",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "marginBottom": "4px"}),
        html.Div(label,
                 style={"color": COLORS["text_secondary"], "fontSize": "9px",
                        "fontWeight": "600", "textTransform": "uppercase",
                        "letterSpacing": "1px",
                        "fontFamily": "'JetBrains Mono', monospace"}),
    ], style={**make_stat_style(color), "flex": "1", "minWidth": "110px"})


# ============================================================================
# Layout
# ============================================================================

def _make_leg_row(idx, cp="call", side="buy", delta=0.25, ratio=1, tenor_mult=1.0, visible=True):
    """Generate one leg configuration row with dropdowns and inputs."""
    prefix = f"stb-leg-{idx}"
    row_bg = COLORS["bg_secondary"] if idx % 2 == 0 else COLORS["bg_card"]
    display = "flex" if visible else "none"
    return html.Div([
        html.Div(f"L{idx+1}", style={
            "color": COLORS["text_muted"], "fontSize": "11px", "fontWeight": "700",
            "minWidth": "28px", "textAlign": "center", "paddingTop": "8px",
        }),
        dcc.Dropdown(
            id={"type": "stb-cp", "index": idx},
            options=[{"label": "CALL", "value": "call"}, {"label": "PUT", "value": "put"}],
            value=cp, clearable=False,
            style={"width": "80px", "fontSize": "11px"},
        ),
        dcc.Dropdown(
            id={"type": "stb-side", "index": idx},
            options=[{"label": "BUY", "value": "buy"}, {"label": "SELL", "value": "sell"}],
            value=side, clearable=False,
            style={"width": "75px", "fontSize": "11px"},
        ),
        dcc.Input(
            id={"type": "stb-delta", "index": idx},
            type="number", value=delta, min=0.05, max=0.95, step=0.05,
            style={**INPUT_STYLE, "width": "60px", "padding": "6px 8px", "fontSize": "11px"},
            debounce=True,
        ),
        dcc.Input(
            id={"type": "stb-ratio", "index": idx},
            type="number", value=ratio, min=1, max=3, step=1,
            style={**INPUT_STYLE, "width": "45px", "padding": "6px 8px", "fontSize": "11px"},
            debounce=True,
        ),
        dcc.Input(
            id={"type": "stb-tenor-mult", "index": idx},
            type="number", value=tenor_mult, min=0.5, max=4.0, step=0.5,
            style={**INPUT_STYLE, "width": "45px", "padding": "6px 8px", "fontSize": "11px"},
            debounce=True,
        ),
        html.Div(id={"type": "stb-strike-disp", "index": idx},
                 style={"color": COLORS["text_secondary"], "fontSize": "10px",
                        "minWidth": "70px", "textAlign": "center", "paddingTop": "8px"}),
        html.Div(id={"type": "stb-vol-disp", "index": idx},
                 style={"color": COLORS["text_secondary"], "fontSize": "10px",
                        "minWidth": "50px", "textAlign": "center", "paddingTop": "8px"}),
        html.Div(id={"type": "stb-prem-disp", "index": idx},
                 style={"color": COLORS["text_secondary"], "fontSize": "10px",
                        "minWidth": "65px", "textAlign": "center", "paddingTop": "8px"}),
    ], id={"type": "stb-leg-row", "index": idx}, style={
        "display": display, "gap": "6px", "alignItems": "center",
        "padding": "5px 8px", "borderRadius": "6px",
        "backgroundColor": row_bg,
        "marginBottom": "3px",
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
        html.Div([
            # ── Left: Input Panel ──────────────────────────────────────
            html.Div([
                html.Div("STRATEGY CONSTRUCTION LAB",
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

                # Leg header row
                html.Div([
                    html.Div("", style={"minWidth": "28px"}),
                    html.Div("C/P", style={"width": "80px", **LABEL_STYLE, "marginBottom": "0"}),
                    html.Div("Side", style={"width": "75px", **LABEL_STYLE, "marginBottom": "0"}),
                    html.Div("Delta", style={"width": "60px", **LABEL_STYLE, "marginBottom": "0"}),
                    html.Div("Ratio", style={"width": "45px", **LABEL_STYLE, "marginBottom": "0"}),
                    html.Div("Tnr*", style={"width": "45px", **LABEL_STYLE, "marginBottom": "0"}),
                    html.Div("Strike", style={"minWidth": "70px", **LABEL_STYLE, "marginBottom": "0"}),
                    html.Div("Vol", style={"minWidth": "50px", **LABEL_STYLE, "marginBottom": "0"}),
                    html.Div("Prem", style={"minWidth": "65px", **LABEL_STYLE, "marginBottom": "0"}),
                ], style={
                    "display": "flex", "gap": "6px", "alignItems": "center",
                    "padding": "2px 8px", "marginBottom": "4px",
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
                                       "boxShadow": "0 4px 14px rgba(239,68,68,0.3)"}),
                ], style={"display": "flex", "gap": "8px", "marginTop": "10px"}),

            ], style={
                **CARD_STYLE,
                "width": "350px", "minWidth": "350px", "flexShrink": "0",
                "overflowY": "auto", "maxHeight": "calc(100vh - 100px)",
            }),

            # ── Right: Charts + Stats ─────────────────────────────────
            html.Div([
                # Summary stat boxes (8)
                html.Div(id="stb-stats-row", style={
                    "display": "flex", "gap": "8px", "marginBottom": "12px",
                    "flexWrap": "wrap",
                }),

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

                # Chart grid row 2 (3 charts)
                html.Div([
                    html.Div([
                        html.Button("CSV", id="stb-csv-3d", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="stb-3d-chart", style={"height": "380px"},
                                  config={"displayModeBar": True, "scrollZoom": True}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "340px",
                              "padding": "12px"}, className="dashboard-card"),
                    html.Div(id="stb-premium-table-container", style={
                        **CARD_STYLE, "flex": "1", "minWidth": "340px", "padding": "12px",
                    }, className="dashboard-card"),
                    html.Div([
                        html.Button("CSV", id="stb-csv-scenario", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="stb-scenario-chart", style={"height": "380px"},
                                  config={"displayModeBar": True, "scrollZoom": False}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "340px",
                              "padding": "12px"}, className="dashboard-card"),
                ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

                # Historical cost context
                html.Div(id="stb-historical-cost", style={
                    "border": f"1px solid {COLORS['border_subtle']}",
                    "padding": "8px",
                    "marginTop": "4px",
                }),

            ], style={"flex": "1", "minWidth": "0"}),

        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
    ])


# ============================================================================
# Callbacks
# ============================================================================

def register_callbacks(app):
    # -- Preset selector updates num-legs and leg configs --
    @app.callback(
        [Output("stb-num-legs", "value")] +
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
                    "padding": "5px 8px", "borderRadius": "6px",
                    "backgroundColor": row_bg, "marginBottom": "3px",
                })
            else:
                styles.append({
                    "display": "none", "gap": "6px", "alignItems": "center",
                    "padding": "5px 8px", "borderRadius": "6px",
                    "backgroundColor": row_bg, "marginBottom": "3px",
                })
        return styles

    # -- Main computation callback --
    @app.callback(
        [Output("stb-stats-row", "children"),
         Output("stb-payoff-chart", "figure"),
         Output("stb-greeks-chart", "figure"),
         Output("stb-heatmap-chart", "figure"),
         Output("stb-3d-chart", "figure"),
         Output("stb-premium-table-container", "children"),
         Output("stb-scenario-chart", "figure")] +
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
    )
    def update_structure(pair, tenor, notional, num_legs,
                         *leg_inputs):
        pair = pair or "EURUSD"
        tenor = tenor or "1M"
        notional = notional or 10_000_000
        num_legs = max(1, min(num_legs or 1, MAX_LEGS))

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

        # Fetch market data
        spots = get_fx_spots([pair])
        vol_surface = get_fx_vol_surface(pair)
        rates = get_fx_rates(pair)

        # Guard: if Bloomberg returned no data, show NO DATA on all charts
        if not spots or not vol_surface:
            ndf = no_data_fig(height=380, msg="NO MARKET DATA")
            empty_table = html.Div("No market data", style={"color": COLORS["text_muted"],
                                   "fontSize": "11px", "padding": "8px"})
            empty_stats = [html.Div("--", style=STAT_BOX_STYLE)]
            return ([empty_stats, ndf, ndf, ndf, ndf, empty_table, ndf]
                    + [""] * MAX_LEGS + [""] * MAX_LEGS + [""] * MAX_LEGS)

        spot_data = spots.get(pair, {"mid": 1.0, "bid": 1.0, "ask": 1.0})

        S = spot_data.get("mid", spot_data.get("bid", 1.0))
        r_d = rates.get("r_dom", 0.03)
        r_f = rates.get("r_for", 0.01)
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

        # Build charts
        payoff_fig = _build_payoff_chart(processed, agg, S, T, r_d, r_f, notional, atm_vol)
        greeks_fig = _build_greeks_chart(processed, S, T, r_d, r_f, notional)
        heatmap_fig = _build_pnl_heatmap(processed, S, T, r_d, r_f, notional)
        surface_fig = _build_3d_surface(processed, S, T, r_d, r_f, notional)
        premium_table = _build_premium_table(processed, pair, pip_size, notional)
        scenario_fig = _build_scenario_chart(processed, S, T, r_d, r_f, notional)

        # Summary stat boxes (8)
        be_str = " / ".join(f"{b:.5f}" for b in agg["breakevens"]) if agg["breakevens"] else "--"
        prem_color = COLORS["pnl_profit"] if agg["net_premium"] < 0 else COLORS["pnl_loss"]
        pop_color = COLORS["accent_green"] if agg["pop"] > 50 else COLORS["accent_red"]

        stats = [
            _stat_box("NET PREMIUM (CCY2)", f"{agg['net_premium']:,.0f}", prem_color),
            _stat_box("NET PREMIUM (PIPS)", f"{agg['net_premium_pips']:.1f}", prem_color),
            _stat_box("NET DELTA", f"{agg['net_delta']:.4f}", COLORS["accent_cyan"]),
            _stat_box("NET VEGA", f"{agg['net_vega'] * notional:.0f}", COLORS["accent_purple"]),
            _stat_box("DAILY THETA", f"{agg['net_theta'] * notional:.0f}", COLORS["accent_orange"]),
            _stat_box("BREAK-EVEN", be_str, COLORS["accent_blue"]),
            _stat_box("PROB OF PROFIT", f"{agg['pop']:.1f}%", pop_color),
            _stat_box("MAX LOSS", f"{agg['max_loss']:,.0f}", COLORS["accent_red"]),
        ]

        # Per-leg display fields
        strike_disps = []
        vol_disps = []
        prem_disps = []
        for i in range(MAX_LEGS):
            if i < len(processed):
                lg = processed[i]
                strike_disps.append(f"{lg['strike']:.5f}")
                vol_disps.append(f"{lg['vol']*100:.1f}%")
                prem_disps.append(f"{lg['premium_pips']:.1f}p")
            else:
                strike_disps.append("")
                vol_disps.append("")
                prem_disps.append("")

        return ([stats, payoff_fig, greeks_fig, heatmap_fig, surface_fig,
                 premium_table, scenario_fig]
                + strike_disps + vol_disps + prem_disps)

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

        pct = vp["percentile"]
        current = vp["current"]
        vol_min = vp["min"]
        vol_max = vp["max"]
        vol_mean = vp["mean"]

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
            f"Current premium sits at the {pct:.0f}th percentile of its "
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
            clickable_stat(f"{pct:.0f}th", "PERCENTILE",
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
        if not fig:
            return no_update
        return export_csv(fig, panel, chart_type)
