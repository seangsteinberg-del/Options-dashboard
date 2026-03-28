"""
Core options pricing engine.

Black-Scholes, Monte Carlo, Binomial Tree, SABR vol model,
full Greeks suite, implied vol solver, vol surface generation,
VaR/CVaR, and probability analytics.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.interpolate import RectBivariateSpline
import pandas as pd
from typing import Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Black-Scholes Closed-Form
# ═══════════════════════════════════════════════════════════════════════════

def bs_d1_d2(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 1e-10 or S <= 0 or K <= 0 or not np.isfinite(sigma):
        return 0.0, 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, q, sigma, option_type="call"):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 1e-10:
        if T > 0 and sigma <= 1e-10:
            # Zero-vol with time remaining: use discounted intrinsic
            df_q = np.exp(-q * T)
            df_r = np.exp(-r * T)
            if option_type == "call":
                return max(S * df_q - K * df_r, 0.0)
            return max(K * df_r - S * df_q, 0.0)
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo Engine
# ═══════════════════════════════════════════════════════════════════════════

def monte_carlo_price(S, K, T, r, q, sigma, option_type="call",
                      n_paths=50000, n_steps=100, antithetic=True,
                      return_paths=False, seed=None):
    """
    GBM Monte Carlo pricer with antithetic variance reduction.
    Optionally returns sample paths for visualization.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    dt = T / n_steps
    half = n_paths // 2 if antithetic else n_paths

    Z = rng.standard_normal((half, n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])

    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    log_paths = np.zeros((Z.shape[0], n_steps + 1))
    log_paths[:, 0] = np.log(S)

    for t in range(n_steps):
        log_paths[:, t + 1] = log_paths[:, t] + drift + diffusion * Z[:, t]

    paths = np.exp(log_paths)
    ST = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(len(payoffs))

    result = {"price": price, "std_error": std_err, "n_paths": len(payoffs)}
    if return_paths:
        # Return subset of paths for plotting
        idx = rng.choice(len(paths), min(200, len(paths)), replace=False)
        times = np.linspace(0, T, n_steps + 1)
        result["paths"] = paths[idx]
        result["times"] = times
        result["terminal_prices"] = ST

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Binomial Tree (CRR)
# ═══════════════════════════════════════════════════════════════════════════

def binomial_tree_price(S, K, T, r, q, sigma, option_type="call",
                        n_steps=200, american=False):
    """Cox-Ross-Rubinstein binomial tree. Supports American exercise."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p = np.clip(p, 0.0, 1.0)  # Guard against extreme rate/vol giving p outside [0,1]
    disc = np.exp(-r * dt)

    # Terminal payoffs
    ST = S * u ** np.arange(n_steps, -1, -1) * d ** np.arange(0, n_steps + 1)
    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            S_i = S * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1)
            if option_type == "call":
                V = np.maximum(V, S_i - K)
            else:
                V = np.maximum(V, K - S_i)

    return float(V[0])


# ═══════════════════════════════════════════════════════════════════════════
# Greeks (full suite)
# ═══════════════════════════════════════════════════════════════════════════

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
    if T <= 0 or sigma <= 1e-10:
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
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


def volga(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    v = vega(S, K, T, r, q, sigma) * 100.0
    return v * d1 * d2 / sigma


def charm(S, K, T, r, q, sigma, option_type="call"):
    if T <= 0 or sigma <= 1e-10:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    charm_val = -np.exp(-q * T) * (
        norm.pdf(d1) * (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    )
    if option_type == "call":
        charm_val += q * np.exp(-q * T) * norm.cdf(d1)
    else:
        charm_val -= q * np.exp(-q * T) * norm.cdf(-d1)
    return charm_val / 365.0


def speed(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 1e-10 or S <= 0:
        return 0.0
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    g = gamma(S, K, T, r, q, sigma)
    return -g / S * (d1 / (sigma * np.sqrt(T)) + 1)


def color_greek(S, K, T, r, q, sigma):
    """Rate of change of gamma w.r.t. time."""
    if T <= 0 or sigma <= 1e-10:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    g = gamma(S, K, T, r, q, sigma)
    dd1dT = (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    return g * (q + d1 * dd1dT + 1.0 / (2.0 * T)) / 365.0


def ultima(S, K, T, r, q, sigma):
    """Third derivative of option price w.r.t. vol."""
    if T <= 0 or sigma <= 1e-10:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    v = vega(S, K, T, r, q, sigma) * 100.0
    return -v / (sigma ** 2) * (d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2)


def dual_delta(S, K, T, r, q, sigma, option_type="call"):
    """Derivative of price w.r.t. strike."""
    if T <= 0:
        if option_type == "call":
            return -1.0 if S > K else 0.0
        return 1.0 if K > S else 0.0
    _, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return -np.exp(-r * T) * norm.cdf(d2)
    return np.exp(-r * T) * norm.cdf(-d2)


def compute_all_greeks(S, K, T, r, q, sigma, option_type="call") -> dict:
    if not np.isfinite(sigma) or sigma <= 0 or S <= 0 or K <= 0:
        return {k: 0.0 for k in ("price", "delta", "gamma", "theta", "vega", "rho",
                                   "vanna", "volga", "charm", "speed", "color", "ultima", "dual_delta")}
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
        "color": color_greek(S, K, T, r, q, sigma),
        "ultima": ultima(S, K, T, r, q, sigma),
        "dual_delta": dual_delta(S, K, T, r, q, sigma, option_type),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Implied Volatility
# ═══════════════════════════════════════════════════════════════════════════

def implied_vol(price, S, K, T, r, q, option_type="call"):
    if T <= 0:
        return None
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if option_type == "call" \
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    if price <= intrinsic * (1.0 + 1e-6) + 1e-10:
        return None
    try:
        return brentq(lambda sig: bs_price(S, K, T, r, q, sig, option_type) - price,
                      1e-6, 10.0, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SABR Stochastic Alpha Beta Rho Model
# ═══════════════════════════════════════════════════════════════════════════

def sabr_vol(F, K, T, alpha, beta, rho_sabr, nu):
    """
    Hagan's SABR implied vol approximation.
    F = forward, K = strike, T = expiry, alpha = vol-of-vol base,
    beta = CEV exponent, rho = correlation, nu = vol-of-vol.
    """
    # Use wider ATM band to avoid discontinuity between ATM and off-ATM formulas
    if abs(F - K) / max(F, 1e-10) < 1e-6:
        # ATM formula (Hagan et al. limiting case)
        FK_beta = F ** (1 - beta)
        v = (alpha / FK_beta) * (
            1 + ((1 - beta) ** 2 / 24 * alpha ** 2 / FK_beta ** 2
                 + 0.25 * rho_sabr * beta * nu * alpha / FK_beta
                 + (2 - 3 * rho_sabr ** 2) / 24 * nu ** 2) * T
        )
        return max(v, 1e-6)

    FK = F * K
    FK_beta2 = FK ** ((1 - beta) / 2)
    log_FK = np.log(F / K)

    z = nu / alpha * FK_beta2 * log_FK if abs(alpha) > 1e-12 else 0.0
    denom = 1 - 2 * rho_sabr * z + z ** 2
    if denom < 0:
        denom = 1e-12
    x_z = np.log((np.sqrt(denom) + z - rho_sabr) / max(1 - rho_sabr, 1e-12))

    if abs(x_z) < 1e-12:
        x_z = 1e-12

    prefix = alpha / (FK_beta2 * (1 + (1 - beta) ** 2 / 24 * log_FK ** 2
                                   + (1 - beta) ** 4 / 1920 * log_FK ** 4))

    correction = 1 + (
        (1 - beta) ** 2 / 24 * alpha ** 2 / (FK ** (1 - beta))
        + 0.25 * rho_sabr * beta * nu * alpha / FK_beta2
        + (2 - 3 * rho_sabr ** 2) / 24 * nu ** 2
    ) * T

    return max(prefix * z / x_z * correction, 1e-6)


def fit_sabr(strikes, market_vols, F, T, beta=0.5):
    """Fit SABR parameters (alpha, rho, nu) to market smile for a given expiry."""
    def objective(params):
        alpha, rho_s, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho_s) >= 1:
            return 1e10
        model_vols = np.array([sabr_vol(F, K, T, alpha, beta, rho_s, nu) for K in strikes])
        return np.sum((model_vols - market_vols) ** 2)

    # Initial guess: derive alpha from ATM vol, estimate rho from skew direction
    atm_idx = np.argmin(np.abs(strikes - F))
    alpha0 = max(market_vols[atm_idx] * F ** (1 - beta), 1e-6)
    # Estimate rho sign from skew: negative skew (puts > calls) → negative rho
    rho0 = -0.3 if len(market_vols) < 3 else np.clip(
        -0.5 * np.sign(market_vols[0] - market_vols[-1]), -0.7, 0.7)
    nu0 = max(0.3, 2.0 * np.std(market_vols) / max(np.mean(market_vols), 1e-6))
    result = minimize(objective, [alpha0, rho0, nu0],
                      method="Nelder-Mead",
                      options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-10})
    alpha, rho_s, nu = result.x
    alpha = max(alpha, 1e-8)
    rho_s = np.clip(rho_s, -0.999, 0.999)
    nu = max(nu, 1e-6)
    return {"alpha": alpha, "beta": beta, "rho": rho_s, "nu": nu, "error": result.fun}


# ═══════════════════════════════════════════════════════════════════════════
# Probability Analytics
# ═══════════════════════════════════════════════════════════════════════════

def probability_itm(S, K, T, r, q, sigma, option_type="call"):
    """Risk-neutral probability of finishing ITM."""
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return 1.0 if S < K else 0.0
    _, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return norm.cdf(d2)
    return norm.cdf(-d2)


def probability_of_profit(S, K, T, r, q, sigma, option_type="call", premium=None):
    """Probability that the option trade is profitable at expiry."""
    if premium is None:
        premium = bs_price(S, K, T, r, q, sigma, option_type)
    if T <= 0 or sigma <= 1e-10:
        return 0.0

    if option_type == "call":
        breakeven = K + premium
        d2_be = (np.log(S / breakeven) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d2_be)
    else:
        breakeven = K - premium
        if breakeven <= 0:
            return 0.0
        d2_be = (np.log(S / breakeven) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(-d2_be)


def expected_move(S, T, sigma, confidence=0.68):
    """Expected move at a given confidence level."""
    z = norm.ppf(0.5 + confidence / 2)
    return S * sigma * np.sqrt(T) * z


def probability_touch(S, K, T, r, q, sigma):
    """Probability that spot touches K at any point before expiry."""
    if T <= 0 or sigma <= 1e-10 or S <= 0 or K <= 0:
        return 0.0
    if abs(np.log(S / K)) < 1e-12:
        return 1.0  # Already at barrier
    mu = r - q - 0.5 * sigma ** 2
    sigma_sqrt_T = sigma * np.sqrt(T)
    log_KS = np.log(K / S)

    if K > S:
        # Upward barrier
        p = norm.cdf((-log_KS + mu * T) / sigma_sqrt_T)
        exponent = 2 * mu * log_KS / sigma ** 2
        if abs(exponent) < 100:
            p += np.exp(exponent) * norm.cdf((-log_KS - mu * T) / sigma_sqrt_T)
    else:
        # Downward barrier
        p = norm.cdf((log_KS - mu * T) / sigma_sqrt_T)
        exponent = 2 * mu * log_KS / sigma ** 2
        if abs(exponent) < 100:
            p += np.exp(exponent) * norm.cdf((log_KS + mu * T) / sigma_sqrt_T)
    return min(p, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# VaR and CVaR (Expected Shortfall)
# ═══════════════════════════════════════════════════════════════════════════

def portfolio_var_cvar(positions, S, r, q, horizon_days=1,
                       n_sims=50000, confidence=0.95, seed=42):
    """
    Monte Carlo VaR and CVaR for an options portfolio.
    Returns dict with var, cvar, pnl_distribution.
    """
    rng = np.random.RandomState(seed)
    T_h = horizon_days / 365.0

    if not positions:
        return {"var": 0.0, "cvar": 0.0, "current_value": 0.0, "pnl_distribution": []}

    # Compute current portfolio value
    current_val = 0
    for pos in positions:
        p = bs_price(S, pos["strike"], pos["expiry"], r, q,
                     pos["vol"], pos["option_type"])
        current_val += p * pos["quantity"] * pos.get("multiplier", 100)

    # Simulate spot moves
    avg_vol = np.mean([pos["vol"] for pos in positions])
    Z = rng.standard_normal(n_sims)
    S_new = S * np.exp((r - q - 0.5 * avg_vol ** 2) * T_h + avg_vol * np.sqrt(T_h) * Z)

    # Revalue portfolio at each scenario
    future_vals = np.zeros(n_sims)
    for pos in positions:
        T_new = max(pos["expiry"] - T_h, 1e-6)
        for i, s in enumerate(S_new):
            p = bs_price(s, pos["strike"], T_new, r, q,
                         pos["vol"], pos["option_type"])
            future_vals[i] += p * pos["quantity"] * pos.get("multiplier", 100)

    pnl = future_vals - current_val
    pnl_sorted = np.sort(pnl)

    cutoff = max(1, min(int((1 - confidence) * n_sims), len(pnl_sorted) - 1))
    var = -pnl_sorted[cutoff - 1]
    cvar = -np.mean(pnl_sorted[:cutoff])

    return {
        "var": var,
        "cvar": cvar,
        "current_value": current_val,
        "mean_pnl": np.mean(pnl),
        "std_pnl": np.std(pnl),
        "pnl_distribution": pnl,
        "pnl_percentiles": {
            "1%": np.percentile(pnl, 1),
            "5%": np.percentile(pnl, 5),
            "25%": np.percentile(pnl, 25),
            "50%": np.percentile(pnl, 50),
            "75%": np.percentile(pnl, 75),
            "95%": np.percentile(pnl, 95),
            "99%": np.percentile(pnl, 99),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Vol Surface
# ═══════════════════════════════════════════════════════════════════════════

def generate_vol_surface(S=100.0, base_vol=0.20, skew_slope=-0.15,
                         skew_convexity=0.10, term_slope=0.02,
                         num_strikes=40, num_expiries=20,
                         strike_range=(0.70, 1.30), expiry_range=(0.02, 2.0)):
    """Parametric vol surface for analytical use (no random noise)."""
    moneyness = np.linspace(strike_range[0], strike_range[1], num_strikes)
    strikes = S * moneyness
    expiries = np.linspace(expiry_range[0], expiry_range[1], num_expiries)
    log_m = np.log(moneyness)
    vol_matrix = np.zeros((num_expiries, num_strikes))

    for i, T in enumerate(expiries):
        skew_factor = skew_slope / np.sqrt(T + 0.1)
        conv_factor = skew_convexity / (T + 0.1)
        term_adj = term_slope * np.log(T + 0.1)
        vol_matrix[i, :] = base_vol + skew_factor * log_m + conv_factor * log_m ** 2 + term_adj

    vol_matrix = np.clip(vol_matrix, 0.02, None)
    return strikes, expiries, vol_matrix


def generate_sabr_vol_surface(S=100.0, r=0.05, q=0.015,
                               alpha=0.3, beta=0.5, rho_sabr=-0.3, nu=0.4,
                               num_strikes=50, num_expiries=25,
                               strike_range=(0.70, 1.30), expiry_range=(0.02, 2.0)):
    """Generate vol surface using the SABR model."""
    moneyness = np.linspace(strike_range[0], strike_range[1], num_strikes)
    strikes = S * moneyness
    expiries = np.linspace(expiry_range[0], expiry_range[1], num_expiries)
    vol_matrix = np.zeros((num_expiries, num_strikes))

    for i, T in enumerate(expiries):
        F = S * np.exp((r - q) * T)
        for j, K in enumerate(strikes):
            vol_matrix[i, j] = sabr_vol(F, K, T, alpha, beta, rho_sabr, nu)

    return strikes, expiries, vol_matrix


def interpolate_vol_surface(strikes, expiries, vol_matrix):
    return RectBivariateSpline(expiries, strikes, vol_matrix, kx=3, ky=3)


# ═══════════════════════════════════════════════════════════════════════════
# Realized Volatility Estimators
# ═══════════════════════════════════════════════════════════════════════════


# generate_price_history removed — use Bloomberg historical data via get_fx_historical_spot()


def realized_vol_close_to_close(prices, window=20):
    """Close-to-close realized volatility."""
    log_ret = np.diff(np.log(prices))
    rv = pd.Series(log_ret).rolling(window).std() * np.sqrt(252)
    return rv.values


def realized_vol_parkinson(highs, lows, window=20):
    """Parkinson (high-low) estimator."""
    hl = np.log(highs / lows)
    factor = 1.0 / (4.0 * np.log(2.0))
    var = pd.Series(factor * hl ** 2).rolling(window).mean() * 252
    return np.sqrt(var.values)


def realized_vol_garman_klass(opens, highs, lows, closes, window=20):
    """Garman-Klass estimator using OHLC data."""
    log_hl = np.log(highs / lows) ** 2
    log_co = np.log(closes / opens) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    rv = pd.Series(gk).rolling(window).mean() * 252
    return np.sqrt(rv.values)


# ═══════════════════════════════════════════════════════════════════════════
# Options Chain Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_options_chain(S, r, q, base_vol=0.20, skew=-0.10,
                           expiry_days=30, strike_step=2.5, num_strikes=20):
    """Generate an analytical options chain for structure analysis (no synthetic volume/OI)."""
    T = expiry_days / 365.0
    center = round(S / strike_step) * strike_step
    strikes = np.arange(center - num_strikes * strike_step,
                        center + (num_strikes + 1) * strike_step, strike_step)

    chain = []
    for K in strikes:
        moneyness = np.log(K / S)
        iv = base_vol + skew * moneyness + 0.05 * moneyness ** 2
        iv = max(iv, 0.05)

        for otype in ["call", "put"]:
            mid = bs_price(S, K, T, r, q, iv, otype)
            d = delta(S, K, T, r, q, iv, otype)
            g = gamma(S, K, T, r, q, iv)
            th = theta(S, K, T, r, q, iv, otype)
            v = vega(S, K, T, r, q, iv)

            # Spread widens away from ATM
            spread_mult = 1 + 2 * abs(moneyness)
            half_spread = max(0.01, mid * 0.02 * spread_mult)

            atm_dist = abs(moneyness)
            volume = 0  # Live volume from Bloomberg
            oi = 0      # Live OI from Bloomberg

            chain.append({
                "strike": K,
                "type": otype,
                "expiry_days": expiry_days,
                "bid": max(0.01, round(mid - half_spread, 2)),
                "ask": round(mid + half_spread, 2),
                "mid": round(mid, 2),
                "iv": round(iv * 100, 2),
                "delta": round(d, 4),
                "gamma": round(g, 6),
                "theta": round(th, 4),
                "vega": round(v, 4),
                "volume": volume,
                "open_interest": oi,
                "itm": (otype == "call" and K < S) or (otype == "put" and K > S),
            })

    return chain


# ═══════════════════════════════════════════════════════════════════════════
# Scenario / Stress Grid
# ═══════════════════════════════════════════════════════════════════════════

def scenario_grid(S, K, T, r, q, sigma, option_type="call",
                  spot_shocks=None, vol_shocks=None, metric="pnl"):
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

    return pd.DataFrame(rows, index=[f"{ds:+.0%} spot" for ds in spot_shocks])


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio Risk Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_portfolio_greeks(positions, S, r, q):
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
