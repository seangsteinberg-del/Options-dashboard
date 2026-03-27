"""
FX Position & Risk Management
==============================
Institutional FX options portfolio engine: positions, books, Garman-Kohlhagen
Greeks, Taylor P&L attribution, vega/gamma bucketing, risk limits, and hedging.

All pricing uses Garman-Kohlhagen (GK) rather than Black-Scholes because FX
options require separate domestic and foreign interest rates.

Designed for an interbank vol desk workflow:
  - Multi-book position management (flow, prop, hedge, EM)
  - Full Greek computation with GK model
  - P&L attribution (delta/gamma/vega/theta/vanna/cross/unexplained)
  - Risk bucketed by pair x tenor (the key view for any vol desk)
  - Real-time limit monitoring with severity levels
  - Hedge suggestions and what-if analysis
"""

import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

BOOKS = [
    "G10_FLOW", "G10_PROP", "EM_FLOW", "EM_PROP",
    "HEDGE", "CLIENT_FACILITATION",
]

DEFAULT_RISK_LIMITS = {
    "max_delta_per_pair": 5_000_000,
    "max_total_delta": 20_000_000,
    "max_gamma_per_pair": 500_000,
    "max_vega_per_pair": 200_000,
    "max_vega_per_tenor_bucket": 100_000,
    "max_total_vega": 1_000_000,
    "max_daily_theta": -50_000,
    "max_var_95_1d": 500_000,
    "max_notional_per_pair": 100_000_000,
    "max_em_notional_pct": 0.30,
}

TENOR_BUCKETS = ["0-1M", "1-3M", "3-6M", "6-12M", "1-2Y", "2-5Y"]

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "fx_portfolio.json")

_PORTFOLIO: Dict = {
    "books": {b: [] for b in BOOKS},
    "trade_history": [],
    "risk_limits": deepcopy(DEFAULT_RISK_LIMITS),
}
_portfolio_lock = threading.RLock()


# ============================================================================
# Garman-Kohlhagen Pricing
# ============================================================================

def _gk_d1d2(S, K, T, r_d, r_f, sigma):
    """Garman-Kohlhagen d1 and d2."""
    sigma = max(sigma, 1e-6)
    S = max(S, 1e-8)
    K = max(K, 1e-8)
    T = max(T, 1e-8)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def _gk_price(S, K, T, r_d, r_f, sigma, cp):
    """
    Garman-Kohlhagen price.
    cp: +1 for call, -1 for put.
    """
    if T <= 1e-10:
        return max(cp * (S - K), 0.0)
    d1, d2 = _gk_d1d2(S, K, T, r_d, r_f, sigma)
    return cp * (S * np.exp(-r_f * T) * norm.cdf(cp * d1)
                 - K * np.exp(-r_d * T) * norm.cdf(cp * d2))


def _gk_greeks(S, K, T, r_d, r_f, sigma, cp):
    """
    Full Garman-Kohlhagen Greeks.

    Returns dict with: delta, gamma, vega, theta, rho_d, rho_f, vanna, volga.
    All values are per unit of foreign notional:
      - delta: dV/dS  (spot delta, % of spot)
      - gamma: d2V/dS2
      - vega:  dV/d_sigma  (per 1 vol point, i.e. per 0.01)
      - theta: dV/dt  (per calendar day, negative = decay)
      - rho_d: dV/dr_d per 1bp
      - rho_f: dV/dr_f per 1bp
      - vanna: d2V/(dS d_sigma)
      - volga: d2V/d_sigma2
    """
    if T <= 1e-10 or sigma <= 1e-10:
        price = max(cp * (S - K), 0.0)
        itm = (cp * (S - K)) > 0
        return {
            "price": price, "delta": cp * 1.0 if itm else 0.0,
            "gamma": 0.0, "vega": 0.0, "theta": 0.0,
            "rho_d": 0.0, "rho_f": 0.0, "vanna": 0.0, "volga": 0.0,
        }

    d1, d2 = _gk_d1d2(S, K, T, r_d, r_f, sigma)
    sqrtT = np.sqrt(T)
    nd1 = norm.cdf(cp * d1)
    nd2 = norm.cdf(cp * d2)
    npd1 = norm.pdf(d1)
    df_f = np.exp(-r_f * T)
    df_d = np.exp(-r_d * T)

    price = cp * (S * df_f * nd1 - K * df_d * nd2)
    delta = cp * df_f * nd1
    gamma = df_f * npd1 / (S * sigma * sqrtT)
    vega = S * df_f * npd1 * sqrtT / 100.0  # per 1 vol point
    theta_annual = (
        -(S * df_f * npd1 * sigma) / (2.0 * sqrtT)
        + cp * r_f * S * df_f * nd1
        - cp * r_d * K * df_d * nd2
    )
    theta = theta_annual / 365.0  # per calendar day
    rho_d = cp * K * T * df_d * nd2 / 10000.0   # per 1bp
    rho_f = -cp * S * T * df_f * nd1 / 10000.0  # per 1bp
    vanna = -df_f * npd1 * d2 / sigma if sigma > 1e-10 else 0.0
    volga_raw = S * df_f * npd1 * sqrtT * d1 * d2 / sigma if sigma > 1e-10 else 0.0

    return {
        "price": price, "delta": delta, "gamma": gamma, "vega": vega,
        "theta": theta, "rho_d": rho_d, "rho_f": rho_f,
        "vanna": vanna, "volga": volga_raw,
    }


# ============================================================================
# Vol Surface Lookup Helpers
# ============================================================================

def _years_to_expiry(expiry_str):
    """Convert ISO date string to time-to-expiry in years from today."""
    if isinstance(expiry_str, (float, int)):
        return float(expiry_str)
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0.25  # fallback 3M
    today = date.today()
    days = (exp_date - today).days
    return max(days / 365.0, 1e-6)


def _tenor_bucket(T):
    """Map time-to-expiry (years) to a tenor bucket string."""
    days = T * 365.0
    if days <= 31:
        return "0-1M"
    elif days <= 91:
        return "1-3M"
    elif days <= 182:
        return "3-6M"
    elif days <= 365:
        return "6-12M"
    elif days <= 730:
        return "1-2Y"
    else:
        return "2-5Y"


def _lookup_vol(pair, K, T, S, vol_surfaces):
    """
    Look up implied vol from a vol surface dict.

    vol_surfaces can be:
      - dict mapping pair -> float (flat vol)
      - dict mapping pair -> callable(K, T) -> vol
      - dict mapping pair -> dict with 'surface_interp' (RectBivariateSpline)
    Falls back to 0.10 if pair not found.
    """
    surf = vol_surfaces.get(pair)
    if surf is None:
        logger.warning("Vol surface missing for %s, using default 0.10", pair)
        return 0.10
    if isinstance(surf, (int, float)):
        return float(surf)
    if callable(surf):
        return surf(K, T)
    if isinstance(surf, dict):
        interp = surf.get("surface_interp")
        if interp is not None:
            try:
                return float(interp(T, K)[0, 0])
            except Exception:
                pass
        flat = surf.get("atm", surf.get("ATM"))
        if flat is not None:
            return float(flat)
    return 0.10


def _get_rate(pair, rates, which="domestic"):
    """
    Look up interest rate for a pair.

    rates can be:
      - float (same rate for all)
      - dict mapping pair -> (r_d, r_f) tuple
      - dict mapping pair -> dict with 'r_d' and 'r_f'
    """
    if isinstance(rates, (int, float)):
        return float(rates)
    r = rates.get(pair)
    if r is None:
        return 0.04
    if isinstance(r, (list, tuple)):
        return float(r[0]) if which == "domestic" else float(r[1])
    if isinstance(r, dict):
        return float(r.get("r_d" if which == "domestic" else "r_f", 0.04))
    return float(r)


# ============================================================================
# Position Management
# ============================================================================

def _validate_position(pos):
    """Validate position fields. Raises ValueError on bad data."""
    required = ["pair", "option_type", "direction", "strike", "expiry",
                "notional", "book"]
    for fld in required:
        if fld not in pos:
            raise ValueError(f"Missing required field: {fld}")
    if pos["option_type"] not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {pos['option_type']}")
    if pos["direction"] not in ("buy", "sell"):
        raise ValueError(f"direction must be 'buy' or 'sell', got {pos['direction']}")
    if pos["strike"] <= 0:
        raise ValueError("strike must be positive")
    if pos["notional"] <= 0:
        raise ValueError("notional must be positive")


def create_sample_portfolio():
    """
    Create a realistic sample FX options portfolio with ~25 positions
    spread across books, strategies, G10/EM pairs, and expiry tenors.

    Resets the global portfolio state.
    """
    global _PORTFOLIO
    _PORTFOLIO = {
        "books": {b: [] for b in BOOKS},
        "trade_history": [],
        "risk_limits": deepcopy(DEFAULT_RISK_LIMITS),
    }

    today = date.today()
    rng = np.random.RandomState(42)

    positions = [
        # --- G10_FLOW: client hedging activity ---
        {"pair": "EURUSD", "option_type": "call", "direction": "buy", "strike": 1.0900,
         "expiry": (today + timedelta(days=14)).isoformat(), "notional": 10_000_000,
         "notional_ccy": "EUR", "delta_at_entry": 0.52, "entry_vol": 0.072,
         "entry_premium": 72_000, "entry_date": (today - timedelta(days=3)).isoformat(),
         "cut": "NY", "counterparty": "JPMorgan", "strategy": "CLIENT",
         "book": "G10_FLOW", "tags": ["corporate-hedge"], "notes": "Client EUR receivable hedge"},

        {"pair": "EURUSD", "option_type": "put", "direction": "sell", "strike": 1.0650,
         "expiry": (today + timedelta(days=30)).isoformat(), "notional": 10_000_000,
         "notional_ccy": "EUR", "delta_at_entry": -0.30, "entry_vol": 0.074,
         "entry_premium": 42_000, "entry_date": (today - timedelta(days=3)).isoformat(),
         "cut": "NY", "counterparty": "JPMorgan", "strategy": "RR",
         "book": "G10_FLOW", "tags": ["risk-reversal"], "notes": "Short put leg of RR"},

        {"pair": "USDJPY", "option_type": "put", "direction": "buy", "strike": 148.00,
         "expiry": (today + timedelta(days=60)).isoformat(), "notional": 15_000_000,
         "notional_ccy": "USD", "delta_at_entry": -0.35, "entry_vol": 0.095,
         "entry_premium": 225_000, "entry_date": (today - timedelta(days=5)).isoformat(),
         "cut": "TKY", "counterparty": "Nomura", "strategy": "CLIENT",
         "book": "G10_FLOW", "tags": ["importer-hedge"], "notes": "Japanese importer hedge"},

        {"pair": "GBPUSD", "option_type": "call", "direction": "buy", "strike": 1.2750,
         "expiry": (today + timedelta(days=45)).isoformat(), "notional": 8_000_000,
         "notional_ccy": "GBP", "delta_at_entry": 0.45, "entry_vol": 0.082,
         "entry_premium": 88_000, "entry_date": (today - timedelta(days=2)).isoformat(),
         "cut": "LDN", "counterparty": "HSBC", "strategy": "CLIENT",
         "book": "G10_FLOW", "tags": ["flow"], "notes": "UK corporate flow"},

        {"pair": "USDCHF", "option_type": "put", "direction": "sell", "strike": 0.8700,
         "expiry": (today + timedelta(days=21)).isoformat(), "notional": 5_000_000,
         "notional_ccy": "USD", "delta_at_entry": -0.25, "entry_vol": 0.068,
         "entry_premium": 18_000, "entry_date": (today - timedelta(days=1)).isoformat(),
         "cut": "NY", "counterparty": "UBS", "strategy": "CLIENT",
         "book": "G10_FLOW", "tags": ["flow"], "notes": "Swiss pension fund hedge"},

        {"pair": "AUDUSD", "option_type": "call", "direction": "buy", "strike": 0.6600,
         "expiry": (today + timedelta(days=90)).isoformat(), "notional": 12_000_000,
         "notional_ccy": "AUD", "delta_at_entry": 0.40, "entry_vol": 0.105,
         "entry_premium": 95_000, "entry_date": (today - timedelta(days=7)).isoformat(),
         "cut": "NY", "counterparty": "ANZ", "strategy": "CLIENT",
         "book": "G10_FLOW", "tags": ["commodity-linked"], "notes": "AUD exporter hedge"},

        # --- G10_PROP: proprietary vol views ---
        {"pair": "EURUSD", "option_type": "call", "direction": "buy", "strike": 1.0800,
         "expiry": (today + timedelta(days=30)).isoformat(), "notional": 20_000_000,
         "notional_ccy": "EUR", "delta_at_entry": 0.55, "entry_vol": 0.073,
         "entry_premium": 160_000, "entry_date": (today - timedelta(days=4)).isoformat(),
         "cut": "NY", "counterparty": "INTERBANK", "strategy": "STRADDLE",
         "book": "G10_PROP", "tags": ["long-vol", "ECB"], "notes": "Long EURUSD 1M straddle call leg"},

        {"pair": "EURUSD", "option_type": "put", "direction": "buy", "strike": 1.0800,
         "expiry": (today + timedelta(days=30)).isoformat(), "notional": 20_000_000,
         "notional_ccy": "EUR", "delta_at_entry": -0.45, "entry_vol": 0.073,
         "entry_premium": 145_000, "entry_date": (today - timedelta(days=4)).isoformat(),
         "cut": "NY", "counterparty": "INTERBANK", "strategy": "STRADDLE",
         "book": "G10_PROP", "tags": ["long-vol", "ECB"], "notes": "Long EURUSD 1M straddle put leg"},

        {"pair": "USDJPY", "option_type": "call", "direction": "sell", "strike": 155.00,
         "expiry": (today + timedelta(days=7)).isoformat(), "notional": 25_000_000,
         "notional_ccy": "USD", "delta_at_entry": 0.20, "entry_vol": 0.110,
         "entry_premium": 180_000, "entry_date": (today - timedelta(days=2)).isoformat(),
         "cut": "TKY", "counterparty": "INTERBANK", "strategy": "PROP",
         "book": "G10_PROP", "tags": ["short-gamma", "weekly"], "notes": "Short 1W USDJPY topside"},

        {"pair": "GBPUSD", "option_type": "put", "direction": "buy", "strike": 1.2400,
         "expiry": (today + timedelta(days=180)).isoformat(), "notional": 10_000_000,
         "notional_ccy": "GBP", "delta_at_entry": -0.30, "entry_vol": 0.088,
         "entry_premium": 130_000, "entry_date": (today - timedelta(days=10)).isoformat(),
         "cut": "LDN", "counterparty": "INTERBANK", "strategy": "PROP",
         "book": "G10_PROP", "tags": ["long-vol", "Brexit"], "notes": "Long 6M GBP downside vol"},

        {"pair": "USDCAD", "option_type": "call", "direction": "sell", "strike": 1.3800,
         "expiry": (today + timedelta(days=14)).isoformat(), "notional": 8_000_000,
         "notional_ccy": "USD", "delta_at_entry": 0.25, "entry_vol": 0.065,
         "entry_premium": 22_000, "entry_date": (today - timedelta(days=1)).isoformat(),
         "cut": "NY", "counterparty": "INTERBANK", "strategy": "PROP",
         "book": "G10_PROP", "tags": ["short-vol"], "notes": "Selling USDCAD 2W vol"},

        # --- EM_FLOW: EM client hedging ---
        {"pair": "USDMXN", "option_type": "call", "direction": "buy", "strike": 17.50,
         "expiry": (today + timedelta(days=90)).isoformat(), "notional": 5_000_000,
         "notional_ccy": "USD", "delta_at_entry": 0.40, "entry_vol": 0.145,
         "entry_premium": 110_000, "entry_date": (today - timedelta(days=5)).isoformat(),
         "cut": "NY", "counterparty": "Citi", "strategy": "CLIENT",
         "book": "EM_FLOW", "tags": ["nearshoring"], "notes": "MXN hedge for US corp"},

        {"pair": "USDZAR", "option_type": "call", "direction": "buy", "strike": 18.80,
         "expiry": (today + timedelta(days=60)).isoformat(), "notional": 3_000_000,
         "notional_ccy": "USD", "delta_at_entry": 0.45, "entry_vol": 0.165,
         "entry_premium": 75_000, "entry_date": (today - timedelta(days=3)).isoformat(),
         "cut": "NY", "counterparty": "Standard Bank", "strategy": "CLIENT",
         "book": "EM_FLOW", "tags": ["mining"], "notes": "SA mining corp hedge"},

        {"pair": "USDTRY", "option_type": "call", "direction": "sell", "strike": 35.00,
         "expiry": (today + timedelta(days=30)).isoformat(), "notional": 2_000_000,
         "notional_ccy": "USD", "delta_at_entry": 0.55, "entry_vol": 0.250,
         "entry_premium": 95_000, "entry_date": (today - timedelta(days=2)).isoformat(),
         "cut": "NY", "counterparty": "INTERBANK", "strategy": "CLIENT",
         "book": "EM_FLOW", "tags": ["high-carry"], "notes": "TRY call overwrite"},

        # --- EM_PROP: proprietary EM views ---
        {"pair": "USDCNH", "option_type": "put", "direction": "buy", "strike": 7.15,
         "expiry": (today + timedelta(days=90)).isoformat(), "notional": 10_000_000,
         "notional_ccy": "USD", "delta_at_entry": -0.35, "entry_vol": 0.058,
         "entry_premium": 48_000, "entry_date": (today - timedelta(days=6)).isoformat(),
         "cut": "TKY", "counterparty": "INTERBANK", "strategy": "PROP",
         "book": "EM_PROP", "tags": ["CNH-strengthening"], "notes": "Long CNH via put"},

        {"pair": "USDBRL", "option_type": "call", "direction": "buy", "strike": 5.20,
         "expiry": (today + timedelta(days=60)).isoformat(), "notional": 4_000_000,
         "notional_ccy": "USD", "delta_at_entry": 0.50, "entry_vol": 0.155,
         "entry_premium": 68_000, "entry_date": (today - timedelta(days=4)).isoformat(),
         "cut": "NY", "counterparty": "INTERBANK", "strategy": "PROP",
         "book": "EM_PROP", "tags": ["fiscal-risk"], "notes": "Long USDBRL on fiscal concern"},

        {"pair": "USDMXN", "option_type": "put", "direction": "buy", "strike": 16.80,
         "expiry": (today + timedelta(days=180)).isoformat(), "notional": 6_000_000,
         "notional_ccy": "USD", "delta_at_entry": -0.30, "entry_vol": 0.140,
         "entry_premium": 85_000, "entry_date": (today - timedelta(days=8)).isoformat(),
         "cut": "NY", "counterparty": "INTERBANK", "strategy": "RR",
         "book": "EM_PROP", "tags": ["MXN-bull"], "notes": "Long MXN via put, RR leg"},

        # --- HEDGE: delta and vega hedges ---
        {"pair": "EURUSD", "option_type": "put", "direction": "buy", "strike": 1.0500,
         "expiry": (today + timedelta(days=90)).isoformat(), "notional": 15_000_000,
         "notional_ccy": "EUR", "delta_at_entry": -0.20, "entry_vol": 0.078,
         "entry_premium": 55_000, "entry_date": (today - timedelta(days=2)).isoformat(),
         "cut": "NY", "counterparty": "INTERBANK", "strategy": "HEDGE",
         "book": "HEDGE", "tags": ["tail-hedge"], "notes": "EUR tail risk hedge"},

        {"pair": "USDJPY", "option_type": "call", "direction": "buy", "strike": 158.00,
         "expiry": (today + timedelta(days=365)).isoformat(), "notional": 20_000_000,
         "notional_ccy": "USD", "delta_at_entry": 0.30, "entry_vol": 0.100,
         "entry_premium": 380_000, "entry_date": (today - timedelta(days=15)).isoformat(),
         "cut": "TKY", "counterparty": "INTERBANK", "strategy": "HEDGE",
         "book": "HEDGE", "tags": ["intervention-hedge"], "notes": "Hedge vs BOJ intervention"},

        {"pair": "USDMXN", "option_type": "call", "direction": "buy", "strike": 19.00,
         "expiry": (today + timedelta(days=180)).isoformat(), "notional": 8_000_000,
         "notional_ccy": "USD", "delta_at_entry": 0.25, "entry_vol": 0.160,
         "entry_premium": 120_000, "entry_date": (today - timedelta(days=5)).isoformat(),
         "cut": "NY", "counterparty": "INTERBANK", "strategy": "HEDGE",
         "book": "HEDGE", "tags": ["EM-hedge"], "notes": "EM blowout hedge"},

        # --- CLIENT_FACILITATION: market-making residuals ---
        {"pair": "EURJPY", "option_type": "call", "direction": "sell", "strike": 164.00,
         "expiry": (today + timedelta(days=45)).isoformat(), "notional": 7_000_000,
         "notional_ccy": "EUR", "delta_at_entry": 0.40, "entry_vol": 0.098,
         "entry_premium": 105_000, "entry_date": (today - timedelta(days=1)).isoformat(),
         "cut": "TKY", "counterparty": "Mizuho", "strategy": "CLIENT",
         "book": "CLIENT_FACILITATION", "tags": ["cross"], "notes": "EURJPY cross flow"},

        {"pair": "EURGBP", "option_type": "put", "direction": "sell", "strike": 0.8500,
         "expiry": (today + timedelta(days=30)).isoformat(), "notional": 5_000_000,
         "notional_ccy": "EUR", "delta_at_entry": -0.35, "entry_vol": 0.062,
         "entry_premium": 15_000, "entry_date": (today - timedelta(days=1)).isoformat(),
         "cut": "LDN", "counterparty": "Barclays", "strategy": "CLIENT",
         "book": "CLIENT_FACILITATION", "tags": ["cross"], "notes": "EURGBP flow"},

        {"pair": "AUDJPY", "option_type": "call", "direction": "buy", "strike": 98.00,
         "expiry": (today + timedelta(days=60)).isoformat(), "notional": 6_000_000,
         "notional_ccy": "AUD", "delta_at_entry": 0.48, "entry_vol": 0.112,
         "entry_premium": 72_000, "entry_date": (today - timedelta(days=3)).isoformat(),
         "cut": "TKY", "counterparty": "Macquarie", "strategy": "CLIENT",
         "book": "CLIENT_FACILITATION", "tags": ["risk-on"], "notes": "AUDJPY risk-on proxy"},

        {"pair": "NZDUSD", "option_type": "put", "direction": "buy", "strike": 0.5950,
         "expiry": (today + timedelta(days=21)).isoformat(), "notional": 4_000_000,
         "notional_ccy": "NZD", "delta_at_entry": -0.40, "entry_vol": 0.108,
         "entry_premium": 24_000, "entry_date": (today - timedelta(days=2)).isoformat(),
         "cut": "NY", "counterparty": "Westpac", "strategy": "CLIENT",
         "book": "CLIENT_FACILITATION", "tags": ["dairy"], "notes": "NZD dairy exporter"},
    ]

    for pos in positions:
        pos["id"] = str(uuid.uuid4())
        pos.setdefault("notional_ccy", pos["pair"][:3])
        pos.setdefault("tags", [])
        pos.setdefault("notes", "")
        pos["status"] = "open"
        book = pos["book"]
        _PORTFOLIO["books"][book].append(pos)

    return _PORTFOLIO


def add_position(book, position_dict):
    """
    Add a position to a book. Validates all fields, assigns UUID.
    Returns the assigned position ID.
    """
    global _PORTFOLIO
    pos = deepcopy(position_dict)
    pos["book"] = book
    _validate_position(pos)

    pos["id"] = str(uuid.uuid4())
    pos.setdefault("notional_ccy", pos["pair"][:3])
    pos.setdefault("delta_at_entry", 0.0)
    pos.setdefault("entry_vol", 0.10)
    pos.setdefault("entry_premium", 0.0)
    pos.setdefault("entry_date", date.today().isoformat())
    pos.setdefault("cut", "NY")
    pos.setdefault("counterparty", "INTERBANK")
    pos.setdefault("strategy", "PROP")
    pos.setdefault("tags", [])
    pos.setdefault("notes", "")
    pos["status"] = "open"

    with _portfolio_lock:
        if book not in _PORTFOLIO["books"]:
            _PORTFOLIO["books"][book] = []
        _PORTFOLIO["books"][book].append(pos)

        _PORTFOLIO["trade_history"].append({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "action": "OPEN",
            "book": book,
            "position_id": pos["id"],
            "details": deepcopy(pos),
        })
    return pos["id"]


def close_position(book, position_id, close_price=None, close_date=None):
    """Close a position. Updates status, records close details."""
    global _PORTFOLIO
    with _portfolio_lock:
        positions = _PORTFOLIO["books"].get(book, [])
        for pos in positions:
            if pos["id"] == position_id and pos.get("status") == "open":
                pos["status"] = "closed"
                pos["close_price"] = close_price
                pos["close_date"] = close_date or date.today().isoformat()
                _PORTFOLIO["trade_history"].append({
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "action": "CLOSE",
                    "book": book,
                    "position_id": position_id,
                    "close_price": close_price,
                    "close_date": pos["close_date"],
                    "details": deepcopy(pos),
                })
            return True
    return False


def roll_position(book, position_id, new_expiry, new_strike=None):
    """
    Roll a position: close the old one, open a new one with
    updated expiry (and optionally strike).
    Returns the new position ID.
    """
    global _PORTFOLIO
    positions = _PORTFOLIO["books"].get(book, [])
    old_pos = None
    for pos in positions:
        if pos["id"] == position_id and pos.get("status") == "open":
            old_pos = deepcopy(pos)
            break
    if old_pos is None:
        return None

    close_position(book, position_id)

    new_pos = deepcopy(old_pos)
    new_pos["expiry"] = new_expiry
    if new_strike is not None:
        new_pos["strike"] = new_strike
    new_pos["entry_date"] = date.today().isoformat()
    del new_pos["id"]
    del new_pos["status"]
    if "close_price" in new_pos:
        del new_pos["close_price"]
    if "close_date" in new_pos:
        del new_pos["close_date"]

    return add_position(book, new_pos)


def partial_close(book, position_id, close_notional):
    """
    Partially close a position by reducing its notional.
    Keeps the remainder open; logs the partial close.
    Returns True on success.
    """
    global _PORTFOLIO
    with _portfolio_lock:
        positions = _PORTFOLIO["books"].get(book, [])
        for pos in positions:
            if pos["id"] == position_id and pos.get("status") == "open":
                if close_notional >= pos["notional"]:
                    return close_position(book, position_id)
                remaining = pos["notional"] - close_notional
                _PORTFOLIO["trade_history"].append({
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "action": "PARTIAL_CLOSE",
                    "book": book,
                    "position_id": position_id,
                    "closed_notional": close_notional,
                    "remaining_notional": remaining,
                })
                pos["notional"] = remaining
                return True
    return False


def get_positions(book=None, pair=None, status="open"):
    """Query positions with optional filters on book, pair, and status."""
    results = []
    with _portfolio_lock:
        books_to_search = [book] if book else list(_PORTFOLIO["books"].keys())
        for b in books_to_search:
            for pos in _PORTFOLIO["books"].get(b, []):
                if status and pos.get("status") != status:
                    continue
                if pair and pos.get("pair") != pair:
                    continue
                results.append(deepcopy(pos))
    return results


def get_all_positions():
    """Return all open positions across all books."""
    return get_positions(book=None, pair=None, status="open")


# ============================================================================
# Greeks Computation
# ============================================================================

def compute_position_greeks(pos, spot, r_d, r_f, vol_surface):
    """
    Compute full Greeks for a single position using current market data.

    Looks up vol from the surface at the position's strike/expiry.
    Returns dict with all Greeks in USD-equivalent terms,
    accounting for notional and direction (buy/sell).
    """
    pair = pos["pair"]
    S = spot if isinstance(spot, (int, float)) else spot.get(pair, 1.0)
    K = pos["strike"]
    T = _years_to_expiry(pos["expiry"])
    try:
        sigma = _lookup_vol(pair, K, T, S, vol_surface) if isinstance(vol_surface, dict) else float(vol_surface or 0.10)
    except (TypeError, ValueError):
        sigma = 0.10
    cp = 1 if pos["option_type"] == "call" else -1
    sign = 1.0 if pos["direction"] == "buy" else -1.0
    notional = pos["notional"]

    greeks = _gk_greeks(S, K, T, r_d, r_f, sigma, cp)

    # Scale by notional and direction.  GK greeks are per-unit-of-foreign;
    # multiplying by notional (in base ccy) gives USD-equivalent for xxxUSD
    # pairs (approximation for crosses).
    scaled = {}
    for key in ("delta", "gamma", "vega", "theta", "rho_d", "rho_f", "vanna", "volga"):
        scaled[key] = greeks[key] * notional * sign
    scaled["price"] = greeks["price"] * notional * sign
    scaled["pair"] = pair
    scaled["position_id"] = pos.get("id", "")
    scaled["sigma"] = sigma
    scaled["T"] = T
    scaled["bucket"] = _tenor_bucket(T)
    return scaled


def compute_book_risk(book, spots, rates, vol_surfaces):
    """
    Aggregate Greeks for all open positions in a book.

    Returns dict with:
      - 'totals': aggregated Greeks
      - 'by_pair': dict of per-pair aggregated Greeks
      - 'positions': list of per-position Greeks
    """
    positions = get_positions(book=book, status="open")
    totals = {k: 0.0 for k in ("delta", "gamma", "vega", "theta", "rho_d", "rho_f",
                                 "vanna", "volga", "price")}
    by_pair = {}
    pos_greeks = []

    for pos in positions:
        pair = pos["pair"]
        S = spots.get(pair, 1.0) if isinstance(spots, dict) else spots
        r_d = _get_rate(pair, rates, "domestic")
        r_f = _get_rate(pair, rates, "foreign")
        pg = compute_position_greeks(pos, S, r_d, r_f, vol_surfaces)
        pos_greeks.append(pg)

        for k in totals:
            totals[k] += pg.get(k, 0.0)

        if pair not in by_pair:
            by_pair[pair] = {k: 0.0 for k in totals}
        for k in totals:
            by_pair[pair][k] += pg.get(k, 0.0)

    return {"totals": totals, "by_pair": by_pair, "positions": pos_greeks}


def compute_portfolio_risk(spots, rates, vol_surfaces):
    """
    Whole portfolio risk aggregated across all books.

    Returns dict with:
      - 'totals': full portfolio aggregated Greeks
      - 'by_book': per-book risk
      - 'by_pair': per-pair risk across all books
    """
    grand = {k: 0.0 for k in ("delta", "gamma", "vega", "theta", "rho_d", "rho_f",
                                "vanna", "volga", "price")}
    by_book = {}
    by_pair = {}

    for book in BOOKS:
        br = compute_book_risk(book, spots, rates, vol_surfaces)
        by_book[book] = br["totals"]
        for k in grand:
            grand[k] += br["totals"].get(k, 0.0)
        for pair, pair_risk in br["by_pair"].items():
            if pair not in by_pair:
                by_pair[pair] = {k: 0.0 for k in grand}
            for k in grand:
                by_pair[pair][k] += pair_risk.get(k, 0.0)

    return {"totals": grand, "by_book": by_book, "by_pair": by_pair}


# ============================================================================
# P&L Attribution
# ============================================================================

def pnl_attribution(positions, spots_old, spots_new, surfaces_old, surfaces_new,
                    rates, dt=1 / 252):
    """
    Full Taylor P&L decomposition.

    Decomposes into:
      - delta_pnl:   sum(delta * dS)
      - gamma_pnl:   sum(0.5 * gamma * dS^2)
      - vega_pnl:    sum(vega * d_sigma)
      - theta_pnl:   sum(theta * dt)
      - rho_pnl:     sum(rho * dr)   (zero when rates unchanged)
      - vanna_pnl:   sum(vanna * dS * d_sigma)
      - volga_pnl:   sum(0.5 * volga * d_sigma^2)
      - unexplained: actual - explained

    Returns dict with by-component totals and by-position breakdown.
    """
    attr = {k: 0.0 for k in ("delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl",
                               "rho_pnl", "vanna_pnl", "volga_pnl",
                               "unexplained", "total_pnl")}
    by_position = []

    for pos in positions:
        if pos.get("status") != "open":
            continue

        pair = pos["pair"]
        K = pos["strike"]
        T = _years_to_expiry(pos["expiry"])
        cp = 1 if pos["option_type"] == "call" else -1
        sign = 1.0 if pos["direction"] == "buy" else -1.0
        notional = pos["notional"]

        S_old = spots_old.get(pair, 1.0)
        S_new = spots_new.get(pair, S_old)
        r_d = _get_rate(pair, rates, "domestic")
        r_f = _get_rate(pair, rates, "foreign")

        sigma_old = _lookup_vol(pair, K, T, S_old, surfaces_old)
        sigma_new = _lookup_vol(pair, K, T, S_new, surfaces_new)
        dS = S_new - S_old
        d_sigma = sigma_new - sigma_old

        greeks = _gk_greeks(S_old, K, T, r_d, r_f, sigma_old, cp)
        sc = notional * sign  # scale factor

        d_pnl = greeks["delta"] * dS * sc
        g_pnl = 0.5 * greeks["gamma"] * dS ** 2 * sc
        v_pnl = greeks["vega"] * d_sigma * 100.0 * sc  # vega is per vol point
        t_pnl = greeks["theta"] * sc                    # theta is per calendar day
        r_pnl = greeks["rho_d"] * 0.0 * sc             # rates unchanged in this call
        va_pnl = greeks.get("vanna", 0) * dS * d_sigma * sc
        volga_pnl = 0.5 * greeks.get("volga", 0) * d_sigma ** 2 * sc

        # Actual revaluation
        price_old = _gk_price(S_old, K, T, r_d, r_f, sigma_old, cp)
        T_new = max(T - dt, 1e-10)
        price_new = _gk_price(S_new, K, T_new, r_d, r_f, sigma_new, cp)
        actual = (price_new - price_old) * sc

        explained = d_pnl + g_pnl + v_pnl + t_pnl + r_pnl + va_pnl + volga_pnl
        unexpl = actual - explained

        entry = {
            "position_id": pos.get("id", ""),
            "pair": pair,
            "delta_pnl": d_pnl, "gamma_pnl": g_pnl, "vega_pnl": v_pnl,
            "theta_pnl": t_pnl, "rho_pnl": r_pnl, "vanna_pnl": va_pnl,
            "volga_pnl": volga_pnl, "unexplained": unexpl,
            "total_pnl": actual,
        }
        by_position.append(entry)

        for k in ("delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "rho_pnl",
                   "vanna_pnl", "volga_pnl", "unexplained", "total_pnl"):
            attr[k] += entry[k]

    attr["by_position"] = by_position
    return attr


# ============================================================================
# Risk Bucketing
# ============================================================================

def vega_by_bucket(positions, spots, rates, vol_surfaces):
    """
    Vega bucketed by pair x tenor.

    Returns dict: {pair: {"0-1M": vega, "1-3M": vega, ...}}
    The key risk view for any vol desk.
    """
    result = {}
    for pos in positions:
        if pos.get("status") != "open":
            continue
        pair = pos["pair"]
        S = spots.get(pair, 1.0) if isinstance(spots, dict) else spots
        r_d = _get_rate(pair, rates, "domestic")
        r_f = _get_rate(pair, rates, "foreign")
        pg = compute_position_greeks(pos, S, r_d, r_f, vol_surfaces)

        if pair not in result:
            result[pair] = {b: 0.0 for b in TENOR_BUCKETS}
        bucket = pg["bucket"]
        if bucket in result[pair]:
            result[pair][bucket] += pg["vega"]

    return result


def gamma_by_bucket(positions, spots, rates, vol_surfaces):
    """
    Gamma bucketed by pair x tenor.

    Returns dict: {pair: {"0-1M": gamma, "1-3M": gamma, ...}}
    """
    result = {}
    for pos in positions:
        if pos.get("status") != "open":
            continue
        pair = pos["pair"]
        S = spots.get(pair, 1.0) if isinstance(spots, dict) else spots
        r_d = _get_rate(pair, rates, "domestic")
        r_f = _get_rate(pair, rates, "foreign")
        pg = compute_position_greeks(pos, S, r_d, r_f, vol_surfaces)

        if pair not in result:
            result[pair] = {b: 0.0 for b in TENOR_BUCKETS}
        bucket = pg["bucket"]
        if bucket in result[pair]:
            result[pair][bucket] += pg["gamma"]

    return result


def delta_by_pair(positions, spots, rates, vol_surfaces):
    """
    Net delta per pair in USD equivalent.

    Returns dict: {pair: delta_usd}
    """
    result = {}
    for pos in positions:
        if pos.get("status") != "open":
            continue
        pair = pos["pair"]
        S = spots.get(pair, 1.0) if isinstance(spots, dict) else spots
        r_d = _get_rate(pair, rates, "domestic")
        r_f = _get_rate(pair, rates, "foreign")
        pg = compute_position_greeks(pos, S, r_d, r_f, vol_surfaces)

        result[pair] = result.get(pair, 0.0) + pg["delta"]

    return result


def exposure_summary(positions, spots, rates, vol_surfaces):
    """
    Portfolio exposure summary: notional by pair, by book, by strategy.
    Also computes EM vs G10 notional split.
    """
    by_pair = {}
    by_book = {}
    by_strategy = {}
    total_notional = 0.0
    em_notional = 0.0

    em_pairs = {"USDMXN", "USDBRL", "USDTRY", "USDZAR", "USDCNH",
                "USDINR", "USDSGD", "USDKRW"}

    for pos in positions:
        if pos.get("status") != "open":
            continue
        pair = pos["pair"]
        book = pos.get("book", "UNKNOWN")
        strat = pos.get("strategy", "UNKNOWN")
        notional = pos["notional"]

        by_pair[pair] = by_pair.get(pair, 0.0) + notional
        by_book[book] = by_book.get(book, 0.0) + notional
        by_strategy[strat] = by_strategy.get(strat, 0.0) + notional
        total_notional += notional
        if pair in em_pairs:
            em_notional += notional

    g10_notional = total_notional - em_notional
    em_pct = em_notional / max(total_notional, 1.0)

    return {
        "by_pair": by_pair,
        "by_book": by_book,
        "by_strategy": by_strategy,
        "total_notional": total_notional,
        "g10_notional": g10_notional,
        "em_notional": em_notional,
        "em_pct": em_pct,
    }


# ============================================================================
# Limits & Hedging
# ============================================================================

def check_risk_limits(risk_totals, limits=None):
    """
    Check portfolio risk against limits.

    risk_totals: dict from compute_portfolio_risk() -- expects 'totals' and 'by_pair'.
    limits: override dict; uses DEFAULT_RISK_LIMITS if None.

    Returns list of breaches:
      [{"limit": "max_vega_per_pair", "pair": "EURUSD", "current": 250000,
        "limit_value": 200000, "utilization": 1.25, "severity": "BREACH"}]

    Severity bands:
      OK       < 80%
      WARNING  80-100%
      BREACH   100-150%
      CRITICAL > 150%
    """
    if limits is None:
        limits = _PORTFOLIO.get("risk_limits", DEFAULT_RISK_LIMITS)

    breaches = []

    def _classify(current, limit_val, label, pair=None):
        if limit_val == 0:
            return
        util = abs(current) / abs(limit_val)
        if util < 0.80:
            severity = "OK"
        elif util < 1.00:
            severity = "WARNING"
        elif util < 1.50:
            severity = "BREACH"
        else:
            severity = "CRITICAL"
        if severity != "OK":
            entry = {
                "limit": label,
                "current": current,
                "limit_value": limit_val,
                "utilization": round(util, 4),
                "severity": severity,
            }
            if pair:
                entry["pair"] = pair
            breaches.append(entry)

    totals = risk_totals.get("totals", risk_totals)
    by_pair = risk_totals.get("by_pair", {})

    # Portfolio-level checks
    _classify(abs(totals.get("delta", 0)), limits.get("max_total_delta", 1e18),
              "max_total_delta")
    _classify(abs(totals.get("vega", 0)), limits.get("max_total_vega", 1e18),
              "max_total_vega")
    _classify(totals.get("theta", 0), limits.get("max_daily_theta", -1e18),
              "max_daily_theta")  # theta is negative

    # Per-pair checks
    for pair, pr in by_pair.items():
        _classify(abs(pr.get("delta", 0)), limits.get("max_delta_per_pair", 1e18),
                  "max_delta_per_pair", pair)
        _classify(abs(pr.get("gamma", 0)), limits.get("max_gamma_per_pair", 1e18),
                  "max_gamma_per_pair", pair)
        _classify(abs(pr.get("vega", 0)), limits.get("max_vega_per_pair", 1e18),
                  "max_vega_per_pair", pair)

    # Sort by severity (CRITICAL first)
    sev_order = {"CRITICAL": 0, "BREACH": 1, "WARNING": 2, "OK": 3}
    breaches.sort(key=lambda b: sev_order.get(b["severity"], 4))
    return breaches


def hedge_suggestion(portfolio_risk, target="delta_neutral"):
    """
    Suggest hedges to achieve a risk target.

    target options:
      - 'delta_neutral':  zero net delta per pair via spot
      - 'vega_neutral_3M': flatten 3M vega via ATM straddles
      - 'gamma_neutral':  flatten short gamma via long options

    Returns list of suggested trades.
    """
    suggestions = []
    by_pair = portfolio_risk.get("by_pair", {})

    # Scale thresholds by total portfolio vega (notional proxy)
    total_vega = sum(abs(risk.get("vega", 0)) for risk in by_pair.values())
    scale = max(total_vega / 50_000, 1.0)  # baseline = 50K vega
    delta_threshold = 10_000 * scale
    vega_threshold = 5_000 * scale
    gamma_threshold = 5_000 * scale

    if target == "delta_neutral":
        for pair, risk in by_pair.items():
            net_delta = risk.get("delta", 0.0)
            if not np.isfinite(net_delta) or abs(net_delta) < delta_threshold:
                continue
            direction = "sell" if net_delta > 0 else "buy"
            suggestions.append({
                "pair": pair,
                "instrument": "SPOT",
                "direction": direction,
                "notional": abs(net_delta),
                "rationale": f"Offset {net_delta:+,.0f} delta in {pair}",
            })

    elif target == "vega_neutral_3M":
        for pair, risk in by_pair.items():
            net_vega = risk.get("vega", 0.0)
            if not np.isfinite(net_vega) or abs(net_vega) < vega_threshold:
                continue
            direction = "sell" if net_vega > 0 else "buy"
            # Approximate: 3M ATM straddle vega per 1M notional is ~10k-20k
            approx_notional = abs(net_vega) / 15_000 * 1_000_000
            suggestions.append({
                "pair": pair,
                "instrument": "3M ATM STRADDLE",
                "direction": direction,
                "notional": round(approx_notional, -5),
                "rationale": f"Offset {net_vega:+,.0f} vega in {pair}",
            })

    elif target == "gamma_neutral":
        for pair, risk in by_pair.items():
            net_gamma = risk.get("gamma", 0.0)
            if not np.isfinite(net_gamma) or abs(net_gamma) < gamma_threshold:
                continue
            if net_gamma < 0:
                # Short gamma: buy options to hedge
                approx_notional = abs(net_gamma) / 50_000 * 1_000_000
                suggestions.append({
                    "pair": pair,
                    "instrument": "1M ATM STRADDLE",
                    "direction": "buy",
                    "notional": round(approx_notional, -5),
                    "rationale": f"Offset {net_gamma:+,.0f} short gamma in {pair}",
                })
            else:
                approx_notional = abs(net_gamma) / 50_000 * 1_000_000
                suggestions.append({
                    "pair": pair,
                    "instrument": "1M ATM STRADDLE",
                    "direction": "sell",
                    "notional": round(approx_notional, -5),
                    "rationale": f"Reduce {net_gamma:+,.0f} long gamma in {pair}",
                })

    return suggestions


def what_if_add(positions, new_trade, spots, rates, vol_surfaces):
    """
    Preview the impact of adding a new trade.

    Returns dict with before/after risk comparison.
    """
    # Compute risk before
    before = {k: 0.0 for k in ("delta", "gamma", "vega", "theta", "vanna", "volga", "price")}
    for pos in positions:
        if pos.get("status") != "open":
            continue
        pair = pos["pair"]
        S = spots.get(pair, 1.0) if isinstance(spots, dict) else spots
        r_d = _get_rate(pair, rates, "domestic")
        r_f = _get_rate(pair, rates, "foreign")
        pg = compute_position_greeks(pos, S, r_d, r_f, vol_surfaces)
        for k in before:
            before[k] += pg.get(k, 0.0)

    # Compute Greeks of the new trade
    nt = deepcopy(new_trade)
    nt.setdefault("status", "open")
    nt.setdefault("id", "WHAT_IF")
    pair = nt["pair"]
    S = spots.get(pair, 1.0) if isinstance(spots, dict) else spots
    r_d = _get_rate(pair, rates, "domestic")
    r_f = _get_rate(pair, rates, "foreign")
    new_greeks = compute_position_greeks(nt, S, r_d, r_f, vol_surfaces)

    after = {}
    for k in before:
        after[k] = before[k] + new_greeks.get(k, 0.0)

    change = {}
    for k in before:
        change[k] = after[k] - before[k]

    return {
        "before": before,
        "after": after,
        "change": change,
        "new_trade_greeks": new_greeks,
    }


# ============================================================================
# Portfolio Persistence
# ============================================================================

def _ensure_data_dir():
    d = os.path.dirname(PORTFOLIO_FILE)
    os.makedirs(d, exist_ok=True)


def save_portfolio(filepath=None):
    """Save the current portfolio state to JSON."""
    filepath = filepath or PORTFOLIO_FILE
    _ensure_data_dir()
    try:
        with open(filepath, "w") as f:
            json.dump(_PORTFOLIO, f, indent=2, default=str)
    except (PermissionError, OSError) as exc:
        logger.error("Failed to save portfolio to %s: %s", filepath, exc)


def load_portfolio(filepath=None):
    """Load portfolio state from JSON. Returns the portfolio dict."""
    global _PORTFOLIO
    filepath = filepath or PORTFOLIO_FILE
    _ensure_data_dir()
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                _PORTFOLIO = json.load(f)
        except (json.JSONDecodeError, PermissionError, OSError) as exc:
            logger.error("Failed to load portfolio from %s: %s", filepath, exc)
            _PORTFOLIO = {
                "books": {b: [] for b in BOOKS},
                "trade_history": [],
                "risk_limits": deepcopy(DEFAULT_RISK_LIMITS),
            }
    else:
        create_sample_portfolio()
    return _PORTFOLIO


def get_portfolio():
    """Get the current in-memory portfolio state."""
    return _PORTFOLIO
