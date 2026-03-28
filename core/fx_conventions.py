"""
FX Market Conventions Engine.

Institutional-grade FX options conventions covering pair registry, delta systems,
vol quoting/smile construction, forward & carry calculations, premium conversions,
calendar/date utilities, cut times, and pair utilities.

All mathematics follow Garman-Kohlhagen and standard interbank FX conventions.
"""

import logging
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline, RectBivariateSpline
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# FX Pair Data Class
# ============================================================================

@dataclass
class FXPairSpec:
    """Full specification of an FX currency pair."""
    pair: str
    base: str
    quote: str
    bb_spot: str
    bb_vol_prefix: str
    bb_rr25_prefix: str
    bb_bf25_prefix: str
    bb_rr10_prefix: str
    bb_bf10_prefix: str
    bb_fwd_prefix: str
    base_rate_bb: str
    quote_rate_bb: str
    pip: float
    group: str
    subgroup: str
    cut_default: str
    spot_date_rule: int
    premium_ccy: str
    delta_convention: str
    quote_convention: str
    typical_notional: float
    trading_hours: str
    liquidity_tier: int


# ============================================================================
# Cut Times
# ============================================================================

CUT_TIMES = {
    "NY":  {"time": "10:00", "tz": "America/New_York",    "label": "NY Cut 10:00 ET"},
    "TKY": {"time": "15:00", "tz": "Asia/Tokyo",          "label": "Tokyo Cut 15:00 JST"},
    "LDN": {"time": "15:00", "tz": "Europe/London",       "label": "London Cut 15:00 GMT"},
    "ECB": {"time": "14:15", "tz": "Europe/Berlin",       "label": "ECB Fix 14:15 CET"},
    "WMR": {"time": "16:00", "tz": "Europe/London",       "label": "WM/Reuters Fix 16:00 GMT"},
    "BOJ": {"time": "09:55", "tz": "Asia/Tokyo",          "label": "BOJ Fix 09:55 JST"},
}


# ============================================================================
# FX Pair Registry (30 pairs)
# ============================================================================

def _build_pair(pair, base, quote, bb_spot, bb_vol, bb_rr25, bb_bf25,
                bb_rr10, bb_bf10, bb_fwd, base_bb, quote_bb, pip,
                group, subgroup, cut, spot_rule, prem_ccy, delta_conv,
                quote_conv, notional, hours, tier):
    return FXPairSpec(
        pair=pair, base=base, quote=quote, bb_spot=bb_spot,
        bb_vol_prefix=bb_vol, bb_rr25_prefix=bb_rr25, bb_bf25_prefix=bb_bf25,
        bb_rr10_prefix=bb_rr10, bb_bf10_prefix=bb_bf10, bb_fwd_prefix=bb_fwd,
        base_rate_bb=base_bb, quote_rate_bb=quote_bb, pip=pip,
        group=group, subgroup=subgroup, cut_default=cut, spot_date_rule=spot_rule,
        premium_ccy=prem_ccy, delta_convention=delta_conv,
        quote_convention=quote_conv, typical_notional=notional,
        trading_hours=hours, liquidity_tier=tier,
    )


FX_PAIR_REGISTRY: Dict[str, FXPairSpec] = {}

_RAW_PAIRS = [
    # G10 Majors
    ("EURUSD", "EUR", "USD", "EURUSD Curncy", "EURUSDV", "EURUSD25R", "EURUSD25B", "EURUSD10R", "EURUSD10B", "EUR Curncy", "EUDR", "USDR", 0.0001, "G10", "Majors", "NY", 2, "USD", "spot", "pips", 10e6, "24h", 1),
    ("USDJPY", "USD", "JPY", "USDJPY Curncy", "USDJPYV", "USDJPY25R", "USDJPY25B", "USDJPY10R", "USDJPY10B", "JPY Curncy", "USDR", "JPDR", 0.01, "G10", "Majors", "TKY", 2, "JPY", "premium-adjusted", "pips", 10e6, "24h", 1),
    ("GBPUSD", "GBP", "USD", "GBPUSD Curncy", "GBPUSDV", "GBPUSD25R", "GBPUSD25B", "GBPUSD10R", "GBPUSD10B", "GBP Curncy", "BPDR", "USDR", 0.0001, "G10", "Majors", "NY", 2, "USD", "spot", "pips", 10e6, "24h", 1),
    ("USDCHF", "USD", "CHF", "USDCHF Curncy", "USDCHFV", "USDCHF25R", "USDCHF25B", "USDCHF10R", "USDCHF10B", "CHF Curncy", "USDR", "SFDR", 0.0001, "G10", "Majors", "NY", 2, "CHF", "spot", "pips", 10e6, "24h", 1),
    ("AUDUSD", "AUD", "USD", "AUDUSD Curncy", "AUDUSDV", "AUDUSD25R", "AUDUSD25B", "AUDUSD10R", "AUDUSD10B", "AUD Curncy", "ADDR", "USDR", 0.0001, "G10", "Majors", "NY", 2, "USD", "spot", "pips", 10e6, "24h", 2),
    ("NZDUSD", "NZD", "USD", "NZDUSD Curncy", "NZDUSDV", "NZDUSD25R", "NZDUSD25B", "NZDUSD10R", "NZDUSD10B", "NZD Curncy", "NDDR", "USDR", 0.0001, "G10", "Majors", "NY", 2, "USD", "spot", "pips", 5e6, "24h", 2),
    ("USDCAD", "USD", "CAD", "USDCAD Curncy", "USDCADV", "USDCAD25R", "USDCAD25B", "USDCAD10R", "USDCAD10B", "CAD Curncy", "USDR", "CDDR", 0.0001, "G10", "Majors", "NY", 1, "CAD", "spot", "pips", 10e6, "24h", 1),
    # G10 Crosses
    ("EURGBP", "EUR", "GBP", "EURGBP Curncy", "EURGBPV", "EURGBP25R", "EURGBP25B", "EURGBP10R", "EURGBP10B", "EURGBP Curncy", "EUDR", "BPDR", 0.0001, "G10", "Crosses", "NY", 2, "GBP", "spot", "pips", 5e6, "24h", 2),
    ("EURJPY", "EUR", "JPY", "EURJPY Curncy", "EURJPYV", "EURJPY25R", "EURJPY25B", "EURJPY10R", "EURJPY10B", "EURJPY Curncy", "EUDR", "JPDR", 0.01, "G10", "Crosses", "TKY", 2, "JPY", "premium-adjusted", "pips", 5e6, "24h", 2),
    ("GBPJPY", "GBP", "JPY", "GBPJPY Curncy", "GBPJPYV", "GBPJPY25R", "GBPJPY25B", "GBPJPY10R", "GBPJPY10B", "GBPJPY Curncy", "BPDR", "JPDR", 0.01, "G10", "Crosses", "TKY", 2, "JPY", "premium-adjusted", "pips", 5e6, "24h", 2),
    ("AUDJPY", "AUD", "JPY", "AUDJPY Curncy", "AUDJPYV", "AUDJPY25R", "AUDJPY25B", "AUDJPY10R", "AUDJPY10B", "AUDJPY Curncy", "ADDR", "JPDR", 0.01, "G10", "Crosses", "TKY", 2, "JPY", "premium-adjusted", "pips", 5e6, "24h", 2),
    ("EURCHF", "EUR", "CHF", "EURCHF Curncy", "EURCHFV", "EURCHF25R", "EURCHF25B", "EURCHF10R", "EURCHF10B", "EURCHF Curncy", "EUDR", "SFDR", 0.0001, "G10", "Crosses", "NY", 2, "CHF", "spot", "pips", 5e6, "24h", 2),
    ("EURAUD", "EUR", "AUD", "EURAUD Curncy", "EURAUDV", "EURAUD25R", "EURAUD25B", "EURAUD10R", "EURAUD10B", "EURAUD Curncy", "EUDR", "ADDR", 0.0001, "G10", "Crosses", "NY", 2, "AUD", "spot", "pips", 5e6, "24h", 2),
    ("EURNZD", "EUR", "NZD", "EURNZD Curncy", "EURNZDV", "EURNZD25R", "EURNZD25B", "EURNZD10R", "EURNZD10B", "EURNZD Curncy", "EUDR", "NDDR", 0.0001, "G10", "Crosses", "NY", 2, "NZD", "spot", "pips", 5e6, "24h", 3),
    ("NZDJPY", "NZD", "JPY", "NZDJPY Curncy", "NZDJPYV", "NZDJPY25R", "NZDJPY25B", "NZDJPY10R", "NZDJPY10B", "NZDJPY Curncy", "NDDR", "JPDR", 0.01, "G10", "Crosses", "TKY", 2, "JPY", "premium-adjusted", "pips", 3e6, "24h", 3),
    ("AUDNZD", "AUD", "NZD", "AUDNZD Curncy", "AUDNZDV", "AUDNZD25R", "AUDNZD25B", "AUDNZD10R", "AUDNZD10B", "AUDNZD Curncy", "ADDR", "NDDR", 0.0001, "G10", "Crosses", "NY", 2, "NZD", "spot", "pips", 3e6, "24h", 3),
    ("CADCHF", "CAD", "CHF", "CADCHF Curncy", "CADCHFV", "CADCHF25R", "CADCHF25B", "CADCHF10R", "CADCHF10B", "CADCHF Curncy", "CDDR", "SFDR", 0.0001, "G10", "Crosses", "NY", 2, "CHF", "spot", "pips", 3e6, "24h", 3),
    ("CADJPY", "CAD", "JPY", "CADJPY Curncy", "CADJPYV", "CADJPY25R", "CADJPY25B", "CADJPY10R", "CADJPY10B", "CADJPY Curncy", "CDDR", "JPDR", 0.01, "G10", "Crosses", "TKY", 2, "JPY", "premium-adjusted", "pips", 3e6, "24h", 3),
    # Scandies
    ("EURNOK", "EUR", "NOK", "EURNOK Curncy", "EURNOKV", "EURNOK25R", "EURNOK25B", "EURNOK10R", "EURNOK10B", "EURNOK Curncy", "EUDR", "NKDR", 0.0001, "G10", "Scandies", "NY", 2, "NOK", "forward", "pips", 5e6, "EU/NY", 3),
    ("EURSEK", "EUR", "SEK", "EURSEK Curncy", "EURSEKV", "EURSEK25R", "EURSEK25B", "EURSEK10R", "EURSEK10B", "EURSEK Curncy", "EUDR", "SKDR", 0.0001, "G10", "Scandies", "NY", 2, "SEK", "forward", "pips", 5e6, "EU/NY", 3),
    ("USDSEK", "USD", "SEK", "USDSEK Curncy", "USDSEKV", "USDSEK25R", "USDSEK25B", "USDSEK10R", "USDSEK10B", "USDSEK Curncy", "USDR", "SKDR", 0.0001, "G10", "Scandies", "NY", 2, "SEK", "forward", "pips", 5e6, "EU/NY", 3),
    ("USDNOK", "USD", "NOK", "USDNOK Curncy", "USDNOKV", "USDNOK25R", "USDNOK25B", "USDNOK10R", "USDNOK10B", "USDNOK Curncy", "USDR", "NKDR", 0.0001, "G10", "Scandies", "NY", 2, "NOK", "forward", "pips", 5e6, "EU/NY", 3),
    # EM
    ("USDMXN", "USD", "MXN", "USDMXN Curncy", "USDMXNV", "USDMXN25R", "USDMXN25B", "USDMXN10R", "USDMXN10B", "MXN Curncy", "USDR", "MXDR", 0.0001, "EM", "LatAm", "NY", 2, "MXN", "forward", "pips", 5e6, "NY", 3),
    ("USDBRL", "USD", "BRL", "USDBRL Curncy", "USDBRLV", "USDBRL25R", "USDBRL25B", "USDBRL10R", "USDBRL10B", "BRL Curncy", "USDR", "BZDR", 0.0001, "EM", "LatAm", "NY", 2, "BRL", "forward", "pips", 5e6, "NY", 4),
    ("USDTRY", "USD", "TRY", "USDTRY Curncy", "USDTRYV", "USDTRY25R", "USDTRY25B", "USDTRY10R", "USDTRY10B", "TRY Curncy", "USDR", "TYDR", 0.0001, "EM", "CEEMEA", "NY", 2, "TRY", "forward", "pips", 5e6, "EU/NY", 4),
    ("USDZAR", "USD", "ZAR", "USDZAR Curncy", "USDZARV", "USDZAR25R", "USDZAR25B", "USDZAR10R", "USDZAR10B", "ZAR Curncy", "USDR", "SADR", 0.0001, "EM", "CEEMEA", "NY", 2, "ZAR", "forward", "pips", 5e6, "EU/NY", 3),
    ("USDCNH", "USD", "CNH", "USDCNH Curncy", "USDCNHV", "USDCNH25R", "USDCNH25B", "USDCNH10R", "USDCNH10B", "CNH Curncy", "USDR", "CCDR", 0.0001, "EM", "Asia", "TKY", 2, "CNH", "forward", "pips", 5e6, "Asia/EU", 3),
    ("USDINR", "USD", "INR", "USDINR Curncy", "USDINRV", "USDINR25R", "USDINR25B", "USDINR10R", "USDINR10B", "INR Curncy", "USDR", "INDR", 0.01, "EM", "Asia", "NY", 2, "INR", "forward", "pips", 5e6, "Asia", 4),
    ("USDSGD", "USD", "SGD", "USDSGD Curncy", "USDSGDV", "USDSGD25R", "USDSGD25B", "USDSGD10R", "USDSGD10B", "SGD Curncy", "USDR", "SGDR", 0.0001, "EM", "Asia", "NY", 2, "SGD", "forward", "pips", 5e6, "Asia", 3),
    ("USDKRW", "USD", "KRW", "USDKRW Curncy", "USDKRWV", "USDKRW25R", "USDKRW25B", "USDKRW10R", "USDKRW10B", "KRW Curncy", "USDR", "KRDR", 0.01, "EM", "Asia", "TKY", 2, "KRW", "forward", "pips", 5e6, "Asia", 4),
]

for _row in _RAW_PAIRS:
    _p = _build_pair(*_row)
    FX_PAIR_REGISTRY[_p.pair] = _p


def get_pair(pair: str) -> FXPairSpec:
    """Look up a pair specification. Accepts 'EURUSD' or 'EUR/USD'."""
    key = pair.replace("/", "").upper()
    if key in FX_PAIR_REGISTRY:
        return FX_PAIR_REGISTRY[key]
    inverted = key[3:] + key[:3]
    if inverted in FX_PAIR_REGISTRY:
        return FX_PAIR_REGISTRY[inverted]
    raise KeyError(f"Unknown FX pair: {pair}")


def all_pairs() -> List[str]:
    """Return sorted list of all registered pair names."""
    return sorted(FX_PAIR_REGISTRY.keys())


def pair_groups() -> Dict[str, List[str]]:
    """Return pairs grouped by their group/subgroup."""
    groups: Dict[str, List[str]] = {}
    for name, spec in FX_PAIR_REGISTRY.items():
        key = f"{spec.group}/{spec.subgroup}"
        groups.setdefault(key, []).append(name)
    return groups


# ============================================================================
# Garman-Kohlhagen Helpers
# ============================================================================

def _gk_d1(S, K, T, r_d, r_f, sigma):
    """Garman-Kohlhagen d1."""
    sigma = max(sigma, 1e-10)
    T = max(T, 1e-10)
    return (np.log(max(S, 1e-10) / max(K, 1e-10)) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _gk_d2(S, K, T, r_d, r_f, sigma):
    """Garman-Kohlhagen d2."""
    sigma = max(sigma, 1e-10)
    T = max(T, 1e-10)
    return _gk_d1(S, K, T, r_d, r_f, sigma) - sigma * np.sqrt(T)


def _gk_price(S, K, T, r_d, r_f, sigma, cp):
    """Garman-Kohlhagen price. cp = +1 for call, -1 for put."""
    if T <= 0:
        return max(cp * (S - K), 0.0)
    d1 = _gk_d1(S, K, T, r_d, r_f, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return cp * (S * np.exp(-r_f * T) * norm.cdf(cp * d1)
                 - K * np.exp(-r_d * T) * norm.cdf(cp * d2))


# ============================================================================
# Delta System
# ============================================================================

def spot_delta(S, K, T, r_d, r_f, sigma, cp):
    """
    Standard Garman-Kohlhagen spot delta (percentage-of-spot).
    cp: +1 for call, -1 for put.
    """
    d1 = _gk_d1(S, K, T, r_d, r_f, sigma)
    return cp * np.exp(-r_f * T) * norm.cdf(cp * d1)


def forward_delta(S, K, T, r_d, r_f, sigma, cp):
    """
    Forward (undiscounted) delta. Used for Scandies and some EM pairs.
    Removes the foreign discount factor.
    """
    d1 = _gk_d1(S, K, T, r_d, r_f, sigma)
    return cp * norm.cdf(cp * d1)


def premium_adjusted_delta(S, K, T, r_d, r_f, sigma, cp):
    """
    Premium-adjusted delta used for pairs where premium is paid in
    foreign (base) currency, e.g. USDJPY where premium is in JPY.
    Adjusts spot delta by subtracting the premium expressed in base terms.
    """
    d1 = _gk_d1(S, K, T, r_d, r_f, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    spot_d = cp * np.exp(-r_f * T) * norm.cdf(cp * d1)
    # Premium in base currency terms: V/S
    prem_over_S = (cp * S * np.exp(-r_f * T) * norm.cdf(cp * d1)
                   - cp * K * np.exp(-r_d * T) * norm.cdf(cp * d2)) / S
    return spot_d - prem_over_S


def strike_to_delta(K, S, T, r_d, r_f, sigma, cp, convention="spot"):
    """
    Compute delta for a given strike, using the specified convention.
    convention: 'spot', 'forward', or 'premium-adjusted'.
    """
    if convention == "forward":
        return forward_delta(S, K, T, r_d, r_f, sigma, cp)
    elif convention == "premium-adjusted":
        return premium_adjusted_delta(S, K, T, r_d, r_f, sigma, cp)
    return spot_delta(S, K, T, r_d, r_f, sigma, cp)


def delta_to_strike(target_delta, S, T, r_d, r_f, sigma, cp, convention="spot"):
    """
    Invert delta to find strike using Brent root-finding.
    target_delta: positive for calls, negative for puts (in absolute terms
    a 25-delta call has target_delta = 0.25 with cp=+1).
    """
    F = S * np.exp((r_d - r_f) * T)
    K_lo = F * np.exp(-5 * sigma * np.sqrt(T))
    K_hi = F * np.exp(5 * sigma * np.sqrt(T))
    K_lo = max(K_lo, 1e-10)

    def obj(K):
        return strike_to_delta(K, S, T, r_d, r_f, sigma, cp, convention) - target_delta

    try:
        return brentq(obj, K_lo, K_hi, xtol=1e-10, maxiter=200)
    except ValueError:
        logger.warning(
            "delta_to_strike: root-finding failed for target_delta=%.4f, "
            "S=%.4f, T=%.4f, sigma=%.4f, cp=%d, convention=%s; returning NaN",
            target_delta, S, T, sigma, cp, convention,
        )
        return np.nan


def delta_to_strike_vectorized(deltas, S, T, r_d, r_f, sigma, convention="spot"):
    """
    Vectorized Newton-Raphson to convert arrays of deltas to strikes.
    deltas: array of signed deltas (+0.25 for 25d call, -0.25 for 25d put).
    Returns array of strikes.
    """
    deltas = np.asarray(deltas, dtype=float)
    F = S * np.exp((r_d - r_f) * T)
    sqrtT = np.sqrt(T)

    # Infer cp from sign of delta
    cp = np.where(deltas > 0, 1.0, -1.0)
    abs_delta = np.abs(deltas)

    # Initial guess from simple spot delta inversion
    K = F * np.exp(-cp * norm.ppf(abs_delta * np.exp(r_f * T)) * sigma * sqrtT
                   + 0.5 * sigma ** 2 * T)

    for _ in range(50):
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        if convention == "spot":
            calc_delta = cp * np.exp(-r_f * T) * norm.cdf(cp * d1)
            ddelta_dK = -np.exp(-r_f * T) * norm.pdf(cp * d1) / (K * sigma * sqrtT)
        elif convention == "forward":
            calc_delta = cp * norm.cdf(cp * d1)
            ddelta_dK = -norm.pdf(cp * d1) / (K * sigma * sqrtT)
        else:
            d2 = d1 - sigma * sqrtT
            calc_delta = (cp * np.exp(-r_f * T) * norm.cdf(cp * d1)
                          - cp * np.exp(-r_d * T) * (K / S) * norm.cdf(cp * d2))
            ddelta_dK = (-np.exp(-r_f * T) * norm.pdf(cp * d1) / (K * sigma * sqrtT)
                         - cp * np.exp(-r_d * T) * norm.cdf(cp * d2) / S
                         + np.exp(-r_d * T) * norm.pdf(cp * d2) / (S * sigma * sqrtT))

        err = calc_delta - deltas
        dK = -err / np.where(np.abs(ddelta_dK) > 1e-15, ddelta_dK, 1e-15)
        K = np.maximum(K + dK, 1e-10)

        if np.max(np.abs(err)) < 1e-10:
            break

    return K


def atm_dns_strike(S, T, r_d, r_f, sigma):
    """
    ATM Delta-Neutral Straddle (DNS) strike.
    The strike where call delta + put delta = 0 (in absolute terms).
    For spot delta: K_DNS = F * exp(0.5 * sigma^2 * T).
    """
    F = S * np.exp((r_d - r_f) * T)
    return F * np.exp(0.5 * sigma ** 2 * T)


def atm_dns_vol(S, T, r_d, r_f, surface):
    """
    ATM DNS vol from a vol surface dict keyed by delta labels.
    Uses the 'ATM' key directly since that corresponds to DNS convention.
    surface: dict with key 'ATM' -> vol.
    """
    if isinstance(surface, dict):
        return surface.get("ATM", surface.get("atm", None))
    return None


def atm_forward_strike(S, T, r_d, r_f):
    """ATM Forward strike: F = S * exp((r_d - r_f) * T)."""
    return S * np.exp((r_d - r_f) * T)


# ============================================================================
# Vol Quoting and Smile Construction
# ============================================================================

def _nan_to_zero(v):
    """Treat None/NaN as 0.0 for vol surface metrics (missing = no skew/curvature)."""
    if v is None:
        return 0.0
    try:
        import math
        if math.isnan(v):
            return 0.0
    except (TypeError, ValueError):
        pass
    return float(v)


def bf_rr_to_smile(atm, rr25, bf25, rr10=None, bf10=None):
    """
    Convert butterfly/risk-reversal quotes to individual vol pillars.

    Market convention:
        BF25 = 0.5*(C25 + P25) - ATM   =>  C25 + P25 = 2*(ATM + BF25)
        RR25 = C25 - P25

    Returns dict with keys: atm, c25, p25, c10, p10 (vols).
    NaN/None inputs are treated as 0.0 (missing data = flat smile assumption).
    """
    atm = _nan_to_zero(atm) or 8.0  # ATM must be positive; default 8 vol points
    rr25 = _nan_to_zero(rr25)
    bf25 = _nan_to_zero(bf25)
    c25 = atm + bf25 + 0.5 * rr25
    p25 = atm + bf25 - 0.5 * rr25
    result = {"atm": atm, "c25": c25, "p25": p25}

    if rr10 is not None and bf10 is not None:
        rr10 = _nan_to_zero(rr10)
        bf10 = _nan_to_zero(bf10)
        c10 = atm + bf10 + 0.5 * rr10
        p10 = atm + bf10 - 0.5 * rr10
        result["c10"] = c10
        result["p10"] = p10

    return result


def smile_to_bf_rr(atm, c25, p25, c10=None, p10=None):
    """
    Convert individual vol pillars back to butterfly/risk-reversal quotes.

    Returns dict with keys: atm, rr25, bf25, rr10, bf10.
    """
    rr25 = c25 - p25
    bf25 = 0.5 * (c25 + p25) - atm
    result = {"atm": atm, "rr25": rr25, "bf25": bf25}

    if c10 is not None and p10 is not None:
        rr10 = c10 - p10
        bf10 = 0.5 * (c10 + p10) - atm
        result["rr10"] = rr10
        result["bf10"] = bf10

    return result


def build_smile_spline(atm, c25, p25, c10, p10, S, T, r_d, r_f):
    """
    Build a cubic spline smile in delta-space from 5-point vol pillars.

    Maps deltas [10p, 25p, ATM, 25c, 10c] to their respective vols
    and returns a CubicSpline interpolant in delta-space.
    """
    # Delta knots (call delta convention, so puts are 1-|put_delta|)
    # Using signed delta: put deltas negative, call deltas positive
    delta_knots = np.array([-0.10, -0.25, 0.0, 0.25, 0.10])
    vol_knots = np.array([p10, p25, atm, c25, c10])

    # Sort by delta for spline
    sort_idx = np.argsort(delta_knots)
    delta_sorted = delta_knots[sort_idx]
    vol_sorted = vol_knots[sort_idx]

    spline = CubicSpline(delta_sorted, vol_sorted, bc_type="not-a-knot")
    return spline


def build_full_vol_surface(vol_quotes_by_tenor, S, r_d, r_f):
    """
    Build a 2D interpolated vol surface from tenor-keyed market quotes.

    vol_quotes_by_tenor: dict mapping tenor string to dict with keys
        {atm, rr25, bf25, rr10, bf10}.
    S: spot.
    r_d, r_f: domestic and foreign rates (flat for simplicity).

    Returns dict with:
        'tenors': list of tenor strings,
        'T_values': array of year fractions,
        'delta_grid': array of delta knots,
        'vol_matrix': 2D array [n_tenors x n_deltas],
        'splines': list of CubicSpline per tenor,
        'surface_interp': RectBivariateSpline for (T, delta) -> vol.
    """
    tenors = sorted(vol_quotes_by_tenor.keys(), key=lambda t: tenor_to_years(t))
    T_values = np.array([tenor_to_years(t) for t in tenors])
    delta_grid = np.array([-0.10, -0.25, 0.0, 0.25, 0.10])
    sort_idx = np.argsort(delta_grid)
    delta_sorted = delta_grid[sort_idx]

    vol_matrix = np.zeros((len(tenors), len(delta_grid)))
    splines = []

    for i, tenor in enumerate(tenors):
        q = vol_quotes_by_tenor[tenor]
        pillars = bf_rr_to_smile(q["atm"], q["rr25"], q["bf25"],
                                 q.get("rr10"), q.get("bf10"))
        vols = np.array([pillars.get("p10", pillars["p25"]),
                         pillars["p25"], pillars["atm"],
                         pillars["c25"],
                         pillars.get("c10", pillars["c25"])])
        vol_matrix[i, :] = vols[sort_idx]
        sp = CubicSpline(delta_sorted, vols[sort_idx], bc_type="clamped")
        splines.append(sp)

    if len(T_values) >= 2:
        surface_interp = RectBivariateSpline(T_values, delta_sorted,
                                             vol_matrix, kx=min(3, len(T_values) - 1),
                                             ky=min(3, len(delta_sorted) - 1))
    else:
        surface_interp = None

    return {
        "tenors": tenors,
        "T_values": T_values,
        "delta_grid": delta_sorted,
        "vol_matrix": vol_matrix,
        "splines": splines,
        "surface_interp": surface_interp,
    }


def vol_at_delta_tenor(surface, delta_val, tenor):
    """
    Query vol at a specific delta and tenor from the built surface.
    surface: output of build_full_vol_surface.
    """
    T = tenor_to_years(tenor) if isinstance(tenor, str) else tenor
    if surface["surface_interp"] is not None:
        return float(surface["surface_interp"](T, delta_val)[0, 0])
    # Fallback: nearest tenor spline
    idx = np.argmin(np.abs(surface["T_values"] - T))
    return float(surface["splines"][idx](delta_val))


def vol_at_strike_tenor(surface, K, T, S, r_d, r_f, convention="spot"):
    """
    Query vol at a specific strike and time-to-expiry.
    Iteratively finds the implied delta for the given strike, then reads vol.
    """
    if isinstance(T, str):
        T = tenor_to_years(T)

    # Initial guess for vol
    vol_guess = vol_at_delta_tenor(surface, 0.0, T)
    for _ in range(10):
        d_call = strike_to_delta(K, S, T, r_d, r_f, vol_guess, 1, convention)
        F = S * np.exp((r_d - r_f) * T)
        if K >= F:
            delta_query = d_call
        else:
            d_put = strike_to_delta(K, S, T, r_d, r_f, vol_guess, -1, convention)
            delta_query = d_put
        vol_new = vol_at_delta_tenor(surface, delta_query, T)
        if abs(vol_new - vol_guess) < 1e-6:
            break
        vol_guess = vol_new
    return vol_guess


def surface_to_strike_space(surface, S, r_d, r_f, n_strikes=50,
                            strike_range=(0.7, 1.3)):
    """
    Convert a delta-space vol surface to strike-space.
    Returns (strikes, T_values, vol_matrix_strike).
    """
    T_values = surface["T_values"]
    F_values = S * np.exp((r_d - r_f) * T_values)
    moneyness = np.linspace(strike_range[0], strike_range[1], n_strikes)
    vol_matrix = np.zeros((len(T_values), n_strikes))

    tol = 1e-6
    for i, T in enumerate(T_values):
        spline = surface["splines"][i]
        for j, m in enumerate(moneyness):
            K = F_values[i] * m
            sigma_guess = float(spline(0.0))
            for _ in range(8):
                d = strike_to_delta(K, S, T, r_d, r_f, sigma_guess, 1, "spot")
                sigma_new = float(spline(d))
                if abs(sigma_new - sigma_guess) < tol:
                    sigma_guess = sigma_new
                    break
                sigma_guess = sigma_new
            vol_matrix[i, j] = sigma_guess

    # Use forward-based strikes to match the vol_matrix computation
    F0 = S * np.exp((r_d - r_f) * T_values[0]) if len(T_values) > 0 else S
    strikes = F0 * moneyness
    return strikes, T_values, vol_matrix


def implied_pdf(surface, tenor, S, r_d, r_f, n_points=200, convention="spot"):
    """
    Breeden-Litzenberger implied risk-neutral PDF from the vol surface.
    d^2C/dK^2 * exp(r_d * T) gives the risk-neutral density.
    """
    T = tenor_to_years(tenor) if isinstance(tenor, str) else tenor
    F = S * np.exp((r_d - r_f) * T)
    n_points = max(n_points, 2)
    K_arr = np.linspace(F * 0.5, F * 1.5, n_points)
    dK = K_arr[1] - K_arr[0]

    prices = np.zeros(n_points)
    for i, K in enumerate(K_arr):
        vol = vol_at_strike_tenor(surface, K, T, S, r_d, r_f, convention)
        d1 = _gk_d1(S, K, T, r_d, r_f, vol)
        d2 = d1 - vol * np.sqrt(T)
        prices[i] = (S * np.exp(-r_f * T) * norm.cdf(d1)
                     - K * np.exp(-r_d * T) * norm.cdf(d2))

    # Second derivative via finite differences
    pdf = np.zeros(n_points)
    pdf[1:-1] = np.exp(r_d * T) * (prices[2:] - 2 * prices[1:-1] + prices[:-2]) / (dK ** 2)
    pdf = np.maximum(pdf, 0.0)

    return K_arr, pdf


def implied_cdf(surface, tenor, S, r_d, r_f, n_points=200, convention="spot"):
    """
    Implied CDF from Breeden-Litzenberger. Integrates the PDF via trapezoidal rule.
    """
    K_arr, pdf = implied_pdf(surface, tenor, S, r_d, r_f, n_points, convention)
    dK = K_arr[1] - K_arr[0]
    cdf = np.cumsum(pdf) * dK
    # Normalize to [0, 1]
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return K_arr, cdf


def smile_arbitrage_check(surface):
    """
    Check for butterfly and calendar spread arbitrage violations.

    Butterfly: vol smile must produce non-negative butterfly spreads
    (convexity in strike space).
    Calendar: total variance must be non-decreasing in time.

    Returns dict with 'butterfly_violations' and 'calendar_violations' lists.
    """
    bf_violations = []
    cal_violations = []

    T_values = surface["T_values"]
    delta_grid = surface["delta_grid"]
    vol_matrix = surface["vol_matrix"]

    # Butterfly check: second derivative of vol w.r.t. delta should keep prices convex
    for i, T in enumerate(T_values):
        vols = vol_matrix[i, :]
        for j in range(1, len(vols) - 1):
            bf = vols[j - 1] - 2 * vols[j] + vols[j + 1]
            if bf < -0.001 * vols[j]:
                bf_violations.append({
                    "tenor_idx": i, "T": float(T),
                    "delta_idx": j, "delta": float(delta_grid[j]),
                    "butterfly_value": float(bf),
                })

    # Calendar check: total variance must be non-decreasing
    for j in range(len(delta_grid)):
        for i in range(1, len(T_values)):
            tv_prev = vol_matrix[i - 1, j] ** 2 * T_values[i - 1]
            tv_curr = vol_matrix[i, j] ** 2 * T_values[i]
            if tv_curr < tv_prev - 1e-8:
                cal_violations.append({
                    "delta_idx": j, "delta": float(delta_grid[j]),
                    "tenor_idx_prev": i - 1, "tenor_idx_curr": i,
                    "T_prev": float(T_values[i - 1]),
                    "T_curr": float(T_values[i]),
                    "tv_prev": float(tv_prev), "tv_curr": float(tv_curr),
                })

    return {
        "butterfly_violations": bf_violations,
        "calendar_violations": cal_violations,
        "is_arbitrage_free": len(bf_violations) == 0 and len(cal_violations) == 0,
    }


# ============================================================================
# Forward and Carry
# ============================================================================

def fx_forward(S, r_d, r_f, T):
    """FX outright forward: F = S * exp((r_d - r_f) * T)."""
    exponent = (r_d - r_f) * T
    # Guard against float64 overflow (exp(709) ~ 8.2e307)
    if exponent > 500:
        return S * np.exp(500.0)  # Cap at a very large but finite value
    return S * np.exp(exponent)


def forward_points(S, r_d, r_f, T):
    """Forward points: F - S, expressed in pips convention."""
    return fx_forward(S, r_d, r_f, T) - S


def forward_curve(S, r_d_curve, r_f_curve, tenors):
    """
    Build a forward curve from term structures.
    r_d_curve, r_f_curve: arrays of rates corresponding to tenors.
    tenors: list of tenor strings (e.g. ['1W', '1M', '3M', ...]).
    Returns dict mapping tenor -> forward price.
    """
    result = {}
    for i, tenor in enumerate(tenors):
        T = tenor_to_years(tenor)
        rd = r_d_curve[i] if hasattr(r_d_curve, '__getitem__') else r_d_curve
        rf = r_f_curve[i] if hasattr(r_f_curve, '__getitem__') else r_f_curve
        result[tenor] = fx_forward(S, rd, rf, T)
    return result


def implied_rate_from_fwd(S, F, T):
    """
    Implied interest rate differential from spot and forward.
    r_d - r_f = ln(F/S) / T.
    """
    if T <= 0 or S <= 0 or F <= 0:
        return 0.0
    return np.log(F / S) / T


def carry_roll_down(S, r_d, r_f, T, days):
    """
    Carry and roll-down P&L over a holding period.
    Returns dict with carry (interest differential), roll-down (forward convergence),
    and total P&L in pips.
    """
    T_new = max(T - days / 365.0, 0.0)
    F_now = fx_forward(S, r_d, r_f, T)
    F_later = fx_forward(S, r_d, r_f, T_new)

    # Carry is the forward points earned over the holding period
    carry = fx_forward(S, r_d, r_f, days / 365.0) - S
    # Roll-down is the change in forward value
    roll_down = F_later - F_now
    total = carry + roll_down

    return {
        "carry": carry,
        "roll_down": roll_down,
        "total_pnl": total,
        "carry_bps": carry / S * 10000 if S > 0 else 0.0,
        "roll_bps": roll_down / S * 10000 if S > 0 else 0.0,
    }


def swap_points_to_rates(S, fwd_pts, T):
    """
    Convert FX swap points to implied rate differential.
    fwd_pts: forward points (F - S).
    Returns r_d - r_f.
    """
    F = S + fwd_pts
    return implied_rate_from_fwd(S, F, T)


# ============================================================================
# Premium System
# ============================================================================

def premium_ccy1(bs_price_val, S):
    """
    Premium in CCY1 (base currency) terms.
    For a pair like EURUSD, CCY1 is EUR.
    premium_ccy1 = bs_price / S.
    """
    if S == 0:
        return 0.0
    return bs_price_val / S


def premium_ccy2(bs_price_val):
    """
    Premium in CCY2 (quote currency) terms.
    This is the raw Black-Scholes price.
    """
    return bs_price_val


def premium_pips(bs_price_val, pip_size):
    """Premium expressed in pips."""
    if pip_size == 0:
        return 0.0
    return bs_price_val / pip_size


def premium_pct_notional(bs_price_val, S, notional_ccy="base"):
    """
    Premium as percentage of notional.
    notional_ccy: 'base' or 'quote'.
    """
    if notional_ccy == "base":
        return bs_price_val / S * 100.0 if S != 0 else 0.0
    return bs_price_val * 100.0


def premium_usd_equiv(bs_price_val, S, pair, notional):
    """
    Convert premium to USD equivalent.
    pair: FX pair string.
    notional: in base currency.
    """
    spec = get_pair(pair)
    total_premium = bs_price_val * notional

    if spec.quote == "USD":
        return total_premium
    elif spec.base == "USD":
        return total_premium / S if S != 0 else 0.0
    else:
        # For cross pairs, estimate USD equivalent using mid-market cross rate
        pair_spec = get_pair(pair) if pair in FX_PAIR_REGISTRY else None
        quote_ccy = pair_spec.quote if pair_spec else pair[3:]
        if quote_ccy == "EUR":
            usd_rate = 1.0850  # EURUSD
        elif quote_ccy == "GBP":
            usd_rate = 1.2650
        elif quote_ccy == "JPY":
            usd_rate = 1.0 / 149.50
        elif quote_ccy == "CHF":
            usd_rate = 1.0 / 0.8820
        elif quote_ccy == "AUD":
            usd_rate = 0.6520
        elif quote_ccy == "NZD":
            usd_rate = 0.6080
        elif quote_ccy == "CAD":
            usd_rate = 1.0 / 1.3580
        else:
            usd_rate = 1.0
        return total_premium * usd_rate


# ============================================================================
# Calendar and Dates
# ============================================================================

_TENOR_MAP = {
    "ON": 1, "TN": 1, "SN": 1,
    "1D": 1, "2D": 2, "3D": 3,
    "1W": 7, "2W": 14, "3W": 21,
    "1M": 30, "2M": 60, "3M": 91,
    "4M": 122, "5M": 152, "6M": 182,
    "9M": 274, "1Y": 365, "18M": 548,
    "2Y": 730, "3Y": 1095, "4Y": 1461, "5Y": 1826,
}


def tenor_to_days(tenor: str) -> int:
    """Convert a tenor string to approximate number of calendar days."""
    tenor = tenor.upper().strip()
    if tenor in _TENOR_MAP:
        return _TENOR_MAP[tenor]
    # Parse numeric tenors like '10D', '6W', '15M'
    if tenor.endswith("D"):
        return int(tenor[:-1])
    if tenor.endswith("W"):
        return int(tenor[:-1]) * 7
    if tenor.endswith("M"):
        return int(tenor[:-1]) * 30
    if tenor.endswith("Y"):
        return int(tenor[:-1]) * 365
    raise ValueError(f"Cannot parse tenor: {tenor}")


def tenor_to_years(tenor: str) -> float:
    """Convert a tenor string to year fraction (ACT/365)."""
    return tenor_to_days(tenor) / 365.0


def years_to_nearest_tenor(T: float) -> str:
    """Find the nearest standard tenor for a given year fraction."""
    days = T * 365.0
    standard = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "5Y"]
    best = "1M"
    best_dist = float("inf")
    for t in standard:
        d = _TENOR_MAP[t]
        dist = abs(d - days)
        if dist < best_dist:
            best_dist = dist
            best = t
    return best


def fx_spot_date(trade_date, pair=None):
    """
    Calculate the FX spot date (T+2 by default, T+1 for USDCAD).
    trade_date: datetime.date or datetime.datetime.
    """
    if isinstance(trade_date, datetime):
        trade_date = trade_date.date()

    spot_rule = 2
    if pair is not None:
        try:
            spec = get_pair(pair)
            spot_rule = spec.spot_date_rule
        except KeyError:
            import logging
            logging.getLogger(__name__).debug("fx_spot_date: pair %s not in registry, using T+2", pair)

    spot = trade_date
    bdays_added = 0
    while bdays_added < spot_rule:
        spot += timedelta(days=1)
        if spot.weekday() < 5:  # Monday=0 ... Friday=4
            bdays_added += 1
    return spot


def fx_expiry_date(trade_date, tenor, pair=None):
    """
    Calculate the FX option expiry date from trade date and tenor.
    Expiry = spot date of the trade + tenor offset, adjusted to business day.
    """
    if isinstance(trade_date, datetime):
        trade_date = trade_date.date()

    spot = fx_spot_date(trade_date, pair)
    days_offset = tenor_to_days(tenor)
    expiry = spot + timedelta(days=days_offset)

    # Roll to next business day if weekend
    while expiry.weekday() >= 5:
        expiry += timedelta(days=1)

    return expiry


def fx_delivery_date(expiry_date, pair=None):
    """
    FX delivery (settlement) date: typically spot (T+2) from expiry.
    """
    if isinstance(expiry_date, datetime):
        expiry_date = expiry_date.date()

    spot_rule = 2
    if pair is not None:
        try:
            spec = get_pair(pair)
            spot_rule = spec.spot_date_rule
        except KeyError:
            import logging
            logging.getLogger(__name__).debug("fx_expiry_date: pair %s not in registry, using T+2", pair)

    delivery = expiry_date
    bdays_added = 0
    while bdays_added < spot_rule:
        delivery += timedelta(days=1)
        if delivery.weekday() < 5:
            bdays_added += 1
    return delivery


def business_days_between(date1, date2):
    """Count the number of business days (Mon-Fri) between two dates, exclusive."""
    if isinstance(date1, datetime):
        date1 = date1.date()
    if isinstance(date2, datetime):
        date2 = date2.date()

    if date1 > date2:
        date1, date2 = date2, date1

    count = 0
    current = date1 + timedelta(days=1)
    while current < date2:
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    return count


# Major FX holidays (simplified - production would use a full holiday calendar)
_FX_HOLIDAYS = {
    "USD": {(1, 1), (1, 20), (2, 17), (5, 26), (7, 4), (9, 1), (11, 27), (12, 25)},
    "EUR": {(1, 1), (4, 18), (4, 21), (5, 1), (12, 25), (12, 26)},
    "GBP": {(1, 1), (4, 18), (4, 21), (5, 5), (5, 26), (8, 25), (12, 25), (12, 26)},
    "JPY": {(1, 1), (1, 13), (2, 11), (3, 20), (4, 29), (5, 3), (5, 4), (5, 5),
            (7, 21), (9, 15), (9, 23), (10, 13), (11, 3), (11, 23), (12, 23)},
    "CHF": {(1, 1), (1, 2), (4, 18), (4, 21), (5, 1), (5, 29), (6, 9), (8, 1),
            (12, 25), (12, 26)},
    "AUD": {(1, 1), (1, 26), (4, 18), (4, 21), (4, 25), (6, 9), (12, 25), (12, 26)},
    "NZD": {(1, 1), (1, 2), (2, 6), (4, 18), (4, 21), (4, 25), (6, 2), (10, 27),
            (12, 25), (12, 26)},
    "CAD": {(1, 1), (2, 17), (4, 18), (5, 19), (7, 1), (9, 1), (10, 13),
            (11, 11), (12, 25), (12, 26)},
}


def is_fx_holiday(dt, pair=None):
    """
    Check if a date is an FX holiday for the given pair.
    Checks both base and quote currency holiday calendars.
    """
    if isinstance(dt, datetime):
        dt = dt.date()

    # Weekends are always non-trading
    if dt.weekday() >= 5:
        return True

    if pair is None:
        return False

    try:
        spec = get_pair(pair)
    except KeyError:
        return False

    md = (dt.month, dt.day)
    base_hols = _FX_HOLIDAYS.get(spec.base, set())
    quote_hols = _FX_HOLIDAYS.get(spec.quote, set())

    return md in base_hols or md in quote_hols


# ============================================================================
# Pair Utilities
# ============================================================================

def invert_pair(pair: str) -> str:
    """
    Invert a currency pair: EURUSD -> USDEUR.
    Returns the string representation of the inverted pair.
    """
    pair = pair.replace("/", "").upper()
    return pair[3:] + pair[:3]


def is_usd_base(pair: str) -> bool:
    """Check if USD is the base (first) currency."""
    pair = pair.replace("/", "").upper()
    return pair[:3] == "USD"


def is_usd_quote(pair: str) -> bool:
    """Check if USD is the quote (second) currency."""
    pair = pair.replace("/", "").upper()
    return pair[3:] == "USD"


def cross_from_legs(pair_a: str, pair_b: str) -> Optional[str]:
    """
    Derive a cross rate pair from two USD legs.
    E.g. EURUSD + USDJPY -> EURJPY.
    Returns the cross pair string or None if no valid cross exists.
    """
    pair_a = pair_a.replace("/", "").upper()
    pair_b = pair_b.replace("/", "").upper()
    a_base, a_quote = pair_a[:3], pair_a[3:]
    b_base, b_quote = pair_b[:3], pair_b[3:]

    # Case: XXXUSD + USDYYY -> XXXYYY
    if a_quote == "USD" and b_base == "USD":
        return a_base + b_quote
    # Case: USDXXX + YYYUSD -> YYYXXX
    if a_base == "USD" and b_quote == "USD":
        return b_base + a_quote
    # Case: XXXUSD + YYYUSD -> XXXYYY
    if a_quote == "USD" and b_quote == "USD":
        return a_base + b_base
    # Case: USDXXX + USDYYY -> XXXYYY (or YYYXXX)
    if a_base == "USD" and b_base == "USD":
        return a_quote + b_quote

    return None


def triangulate_vol(vol_a, vol_b, corr):
    """
    Implied vol for a cross from two USD legs using triangulation.
    sigma_cross^2 = sigma_a^2 + sigma_b^2 - 2 * corr * sigma_a * sigma_b
    (for pairs where one is inverted relative to the cross).
    """
    var = vol_a ** 2 + vol_b ** 2 - 2 * corr * vol_a * vol_b
    return np.sqrt(max(var, 0.0))


def pip_value(pair: str, notional: float) -> float:
    """
    Calculate the pip value for a given notional in base currency.
    Returns the pip value in quote currency terms.
    """
    spec = get_pair(pair)
    return notional * spec.pip


# ============================================================================
# Additional Pair Query Functions
# ============================================================================

def get_delta_convention(pair: str) -> str:
    """Get the market-standard delta convention for a pair."""
    return get_pair(pair).delta_convention


def get_premium_ccy(pair: str) -> str:
    """Get the premium currency for a pair."""
    return get_pair(pair).premium_ccy


def get_pip_size(pair: str) -> float:
    """Get the pip size for a pair (0.0001 or 0.01 for JPY pairs)."""
    return get_pair(pair).pip


def get_cut_time(pair: str) -> dict:
    """Get the default cut time specification for a pair."""
    spec = get_pair(pair)
    return CUT_TIMES.get(spec.cut_default, CUT_TIMES["NY"])


def get_liquidity_tier(pair: str) -> int:
    """Get liquidity tier (1=most liquid, 4=least)."""
    return get_pair(pair).liquidity_tier


def get_trading_hours(pair: str) -> str:
    """Get typical trading hours description."""
    return get_pair(pair).trading_hours


def pairs_by_liquidity(tier: int) -> List[str]:
    """Return all pairs matching a given liquidity tier."""
    return [name for name, spec in FX_PAIR_REGISTRY.items()
            if spec.liquidity_tier == tier]


def pairs_by_group(group: str) -> List[str]:
    """Return all pairs in a given group (e.g. 'G10', 'EM')."""
    return [name for name, spec in FX_PAIR_REGISTRY.items()
            if spec.group == group]


def pairs_by_subgroup(subgroup: str) -> List[str]:
    """Return all pairs in a given subgroup (e.g. 'Majors', 'Crosses')."""
    return [name for name, spec in FX_PAIR_REGISTRY.items()
            if spec.subgroup == subgroup]
