"""
Core options pricing engine.

Black-Scholes pricing, full Greeks suite, implied vol solver,
and synthetic vol surface generation.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
import pandas as pd
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Black-Scholes closed-form
# ---------------------------------------------------------------------------

def bs_d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
             option_type: str = "call") -> float:
    """
    Black-Scholes European option price.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to expiry (years)
    r : risk-free rate
    q : continuous dividend yield
    sigma : volatility
    option_type : 'call' or 'put'
    """
    if T <= 0:
        payoff = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return payoff

    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def delta(S, K, T, r, q, sigma, option_type="call"):
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def gamma(S, K, T, r, q, sigma):
    if T <= 0:
        return 0.0
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, q, sigma, option_type="call"):
    if T <= 0:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    common = -(S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
    if option_type == "call":
        return (common - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365.0
    return (common + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365.0


def vega(S, K, T, r, q, sigma):
    if T <= 0:
        return 0.0
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0


def rho(S, K, T, r, q, sigma, option_type="call"):
    if T <= 0:
        return 0.0
    _, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def vanna(S, K, T, r, q, sigma):
    if T <= 0:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


def volga(S, K, T, r, q, sigma):
    """Vomma / Volga — second derivative of price w.r.t. vol."""
    if T <= 0:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    v = vega(S, K, T, r, q, sigma) * 100.0  # un-scale
    return v * d1 * d2 / sigma


def charm(S, K, T, r, q, sigma, option_type="call"):
    """Delta decay — dDelta/dT."""
    if T <= 0:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    charm_val = -np.exp(-q * T) * (
        norm.pdf(d1) * (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    )
    if option_type == "put":
        charm_val += q * np.exp(-q * T)
    else:
        charm_val -= q * np.exp(-q * T)
    return charm_val / 365.0


def speed(S, K, T, r, q, sigma):
    """Third derivative of price w.r.t. spot."""
    if T <= 0:
        return 0.0
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    g = gamma(S, K, T, r, q, sigma)
    return -g / S * (d1 / (sigma * np.sqrt(T)) + 1)


def compute_all_greeks(S, K, T, r, q, sigma, option_type="call") -> dict:
    """Return a dictionary of all Greeks."""
    return {
        "price": bs_price(S, K, T, r, q, sigma, option_type),
        "delta": delta(S, K, T, r, q, sigma, option_type),
        "gamma": gamma(S, K, T, r, q, sigma),
        "theta": theta(S, K, T, r, q, sigma, option_type),
        "vega": vega(S, K, T, r, q, sigma),
        "rho": rho(S, K, T, r, q, sigma, option_type),
        "vanna": vanna(S, K, T, r, q, sigma),
        "volga": volga(S, K, T, r, q, sigma),
        "charm": charm(S, K, T, r, q, sigma, option_type),
        "speed": speed(S, K, T, r, q, sigma),
    }


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float,
                option_type: str = "call") -> Optional[float]:
    """Newton/Brent solver for implied vol."""
    if T <= 0:
        return None
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if option_type == "call" \
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    if price <= intrinsic + 1e-10:
        return None
    try:
        iv = brentq(
            lambda sig: bs_price(S, K, T, r, q, sig, option_type) - price,
            1e-6, 10.0, xtol=1e-8, maxiter=200,
        )
        return iv
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Synthetic vol surface
# ---------------------------------------------------------------------------

def generate_vol_surface(
    S: float = 100.0,
    base_vol: float = 0.20,
    skew_slope: float = -0.15,
    skew_convexity: float = 0.10,
    term_slope: float = 0.02,
    num_strikes: int = 40,
    num_expiries: int = 20,
    strike_range: Tuple[float, float] = (0.70, 1.30),
    expiry_range: Tuple[float, float] = (0.02, 2.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a realistic synthetic implied vol surface with skew and term structure.

    Returns (strikes, expiries, vol_matrix) where vol_matrix is (num_expiries x num_strikes).
    """
    moneyness = np.linspace(strike_range[0], strike_range[1], num_strikes)
    strikes = S * moneyness
    expiries = np.linspace(expiry_range[0], expiry_range[1], num_expiries)

    log_m = np.log(moneyness)
    vol_matrix = np.zeros((num_expiries, num_strikes))

    for i, T in enumerate(expiries):
        # Skew flattens with maturity
        skew_factor = skew_slope / np.sqrt(T + 0.1)
        conv_factor = skew_convexity / (T + 0.1)
        term_adj = term_slope * np.log(T + 0.1)

        vol_matrix[i, :] = (
            base_vol
            + skew_factor * log_m
            + conv_factor * log_m ** 2
            + term_adj
            + np.random.normal(0, 0.002, num_strikes)  # micro-noise
        )

    # Floor at 2% vol
    vol_matrix = np.clip(vol_matrix, 0.02, None)

    return strikes, expiries, vol_matrix


def interpolate_vol_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    vol_matrix: np.ndarray,
) -> RectBivariateSpline:
    """Return a smooth interpolator for the vol surface."""
    return RectBivariateSpline(expiries, strikes, vol_matrix, kx=3, ky=3)


# ---------------------------------------------------------------------------
# Scenario / stress grid
# ---------------------------------------------------------------------------

def scenario_grid(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str = "call",
    spot_shocks: Optional[np.ndarray] = None,
    vol_shocks: Optional[np.ndarray] = None,
    metric: str = "pnl",
) -> pd.DataFrame:
    """
    Build a 2-D scenario grid (spot shock vs vol shock).

    metric: 'pnl', 'price', 'delta', 'gamma', 'theta', 'vega'
    """
    if spot_shocks is None:
        spot_shocks = np.linspace(-0.20, 0.20, 9)
    if vol_shocks is None:
        vol_shocks = np.linspace(-0.10, 0.10, 9)

    base_price = bs_price(S, K, T, r, q, sigma, option_type)
    rows = []

    for ds in spot_shocks:
        row = {}
        S_new = S * (1 + ds)
        for dv in vol_shocks:
            sigma_new = max(sigma + dv, 0.01)
            if metric == "price":
                val = bs_price(S_new, K, T, r, q, sigma_new, option_type)
            elif metric == "pnl":
                val = bs_price(S_new, K, T, r, q, sigma_new, option_type) - base_price
            elif metric == "delta":
                val = delta(S_new, K, T, r, q, sigma_new, option_type)
            elif metric == "gamma":
                val = gamma(S_new, K, T, r, q, sigma_new)
            elif metric == "theta":
                val = theta(S_new, K, T, r, q, sigma_new, option_type)
            elif metric == "vega":
                val = vega(S_new, K, T, r, q, sigma_new)
            else:
                val = bs_price(S_new, K, T, r, q, sigma_new, option_type) - base_price
            row[f"{dv:+.0%} vol"] = round(val, 4)
        rows.append(row)

    df = pd.DataFrame(rows, index=[f"{ds:+.0%} spot" for ds in spot_shocks])
    return df


# ---------------------------------------------------------------------------
# Portfolio-level risk aggregation
# ---------------------------------------------------------------------------

def aggregate_portfolio_greeks(positions: list, S: float, r: float, q: float) -> dict:
    """
    Aggregate Greeks across a portfolio of positions.

    Each position is a dict: {strike, expiry, vol, option_type, quantity, entry_price}
    """
    totals = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0,
              "vanna": 0, "volga": 0, "pnl": 0, "notional": 0}

    for pos in positions:
        qty = pos.get("quantity", 0)
        K = pos["strike"]
        T = pos["expiry"]
        sig = pos["vol"]
        otype = pos.get("option_type", "call")
        entry = pos.get("entry_price", 0)
        multiplier = pos.get("multiplier", 100)

        greeks = compute_all_greeks(S, K, T, r, q, sig, otype)

        totals["delta"] += greeks["delta"] * qty * multiplier
        totals["gamma"] += greeks["gamma"] * qty * multiplier
        totals["theta"] += greeks["theta"] * qty * multiplier
        totals["vega"] += greeks["vega"] * qty * multiplier
        totals["rho"] += greeks["rho"] * qty * multiplier
        totals["vanna"] += greeks["vanna"] * qty * multiplier
        totals["volga"] += greeks["volga"] * qty * multiplier
        totals["pnl"] += (greeks["price"] - entry) * qty * multiplier
        totals["notional"] += abs(qty) * multiplier * S

    return totals
