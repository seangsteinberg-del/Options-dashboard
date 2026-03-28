"""
Vanna-Volga pricing method for FX options.

Implements the Castagna-Mercurio approach using three reference instruments
(ATM DNS, 25-delta put, 25-delta call) to price and interpolate FX
volatility smiles.  Includes Garman-Kohlhagen Greeks, full VV Greeks via
bump-and-reprice, smile generation, and a SABR comparison utility.

Reference:
    Castagna, A. and Mercurio, F. (2007) "The Vanna-Volga Method for
    Implied Volatilities", Risk, January 2007.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


# =========================================================================
# Garman-Kohlhagen building blocks
# =========================================================================

def _gk_d1_d2(S, K, T, r_d, r_f, sigma):
    """Return (d1, d2) for the Garman-Kohlhagen model."""
    sigma = max(sigma, 1e-10)
    T = max(T, 1e-10)
    S = max(S, 1e-10)
    K = max(K, 1e-10)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def gk_price(S, K, T, r_d, r_f, sigma, cp):
    """Garman-Kohlhagen price for an FX option.

    Parameters
    ----------
    S : float   -- spot FX rate
    K : float   -- strike
    T : float   -- time to expiry in years
    r_d : float -- domestic risk-free rate
    r_f : float -- foreign risk-free rate
    sigma : float -- implied volatility
    cp : int    -- +1 for call, -1 for put
    """
    if T <= 0:
        return max(cp * (S - K), 0.0)
    d1, d2 = _gk_d1_d2(S, K, T, r_d, r_f, sigma)
    df_f = np.exp(-r_f * T)
    df_d = np.exp(-r_d * T)
    return cp * (S * df_f * norm.cdf(cp * d1) - K * df_d * norm.cdf(cp * d2))


def gk_vega(S, K, T, r_d, r_f, sigma):
    """Garman-Kohlhagen vega (dPrice / dSigma)."""
    if T <= 0:
        return 0.0
    d1, _ = _gk_d1_d2(S, K, T, r_d, r_f, sigma)
    return S * np.exp(-r_f * T) * norm.pdf(d1) * np.sqrt(T)


def gk_vanna(S, K, T, r_d, r_f, sigma):
    """Garman-Kohlhagen vanna: d(delta)/d(sigma) = d(vega)/d(spot)."""
    if T <= 0 or sigma <= 1e-10:
        return 0.0
    d1, d2 = _gk_d1_d2(S, K, T, r_d, r_f, sigma)
    return -np.exp(-r_f * T) * norm.pdf(d1) * d2 / sigma


def gk_volga(S, K, T, r_d, r_f, sigma):
    """Garman-Kohlhagen volga: d(vega)/d(sigma) = vega * d1 * d2 / sigma."""
    if T <= 0 or sigma <= 1e-10:
        return 0.0
    d1, d2 = _gk_d1_d2(S, K, T, r_d, r_f, sigma)
    v = gk_vega(S, K, T, r_d, r_f, sigma)
    return v * d1 * d2 / sigma


# =========================================================================
# Delta / strike conversion helpers
# =========================================================================

def _delta_to_strike(S, T, r_d, r_f, sigma, delta_target, cp):
    """Convert a BS delta to a strike via Brent's method.

    For a call delta_target is positive (e.g. 0.25).
    For a put  delta_target is negative (e.g. -0.25).
    """
    F = S * np.exp((r_d - r_f) * T)
    lo = F * 0.3
    hi = F * 3.0

    def obj(K):
        d1, _ = _gk_d1_d2(S, K, T, r_d, r_f, sigma)
        return cp * np.exp(-r_f * T) * norm.cdf(cp * d1) - delta_target

    return brentq(obj, lo, hi, xtol=1e-10, maxiter=200)


def _atm_dns_strike(S, T, r_d, r_f, sigma):
    """ATM Delta-Neutral Straddle strike: where call delta + put delta = 0."""
    F = S * np.exp((r_d - r_f) * T)
    lo = F * 0.3
    hi = F * 3.0

    def obj(K):
        d1, _ = _gk_d1_d2(S, K, T, r_d, r_f, sigma)
        df_f = np.exp(-r_f * T)
        call_delta = df_f * norm.cdf(d1)
        put_delta = -df_f * norm.cdf(-d1)
        return call_delta + put_delta

    return brentq(obj, lo, hi, xtol=1e-10, maxiter=200)


# =========================================================================
# Vanna-Volga core
# =========================================================================

def vv_price(S, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol, cp):
    """Castagna-Mercurio Vanna-Volga price.

    The three market pivots are:
        K1 = 25-delta put strike  (vol = p25_vol)
        K2 = ATM DNS strike       (vol = atm_vol)
        K3 = 25-delta call strike (vol = c25_vol)

    VV price = GK(K, atm_vol) + sum_i w_i * [GK(K_i, sigma_i) - GK(K_i, atm_vol)]

    Weights are determined by matching the portfolio's vanna and volga
    to the target option's vanna and volga.
    """
    if T <= 1e-10:
        return max(cp * (S - K), 0.0)

    # Flat vol surface: no smile adjustment needed
    if np.std([p25_vol, atm_vol, c25_vol]) < 1e-6:
        return gk_price(S, K, T, r_d, r_f, atm_vol, cp)

    # Pivot strikes
    K2 = _atm_dns_strike(S, T, r_d, r_f, atm_vol)
    K1 = _delta_to_strike(S, T, r_d, r_f, p25_vol, -0.25, -1)
    K3 = _delta_to_strike(S, T, r_d, r_f, c25_vol, 0.25, 1)

    sigma_atm = atm_vol

    # Vanna and volga of the target option at ATM vol
    van_K = gk_vanna(S, K, T, r_d, r_f, sigma_atm)
    vol_K = gk_volga(S, K, T, r_d, r_f, sigma_atm)

    # Vanna and volga of the three pivots at ATM vol
    van = [gk_vanna(S, Ki, T, r_d, r_f, sigma_atm) for Ki in (K1, K2, K3)]
    vol = [gk_volga(S, Ki, T, r_d, r_f, sigma_atm) for Ki in (K1, K2, K3)]

    # Solve for weights by inverting the 2x3 system with the log-strike
    # linear constraint.  Full 3x3 system:
    #   sum w_i * vanna_i = vanna_K
    #   sum w_i * volga_i = volga_K
    #   sum w_i * log(K_i) = log(K)       (ensures cost-of-carry consistency)
    A = np.array([
        [van[0], van[1], van[2]],
        [vol[0], vol[1], vol[2]],
        [np.log(K1), np.log(K2), np.log(K3)],
    ])
    b = np.array([van_K, vol_K, np.log(K)])

    try:
        cond = np.linalg.cond(A)
        if cond > 1e10:
            # Near-singular: fall back to flat ATM vol
            return gk_price(S, K, T, r_d, r_f, sigma_atm, cp)
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Degenerate: fall back to flat ATM vol
        return gk_price(S, K, T, r_d, r_f, sigma_atm, cp)

    # Cost-of-carry adjustments for each pivot
    # Each pivot is priced with its natural type: K1=put, K2=straddle(call), K3=call
    sigmas = [p25_vol, atm_vol, c25_vol]
    Ks = [K1, K2, K3]
    pivot_cp = [-1, 1, 1]  # put for 25P, call for ATM & 25C
    adjustment = 0.0
    for i in range(3):
        coc = gk_price(S, Ks[i], T, r_d, r_f, sigmas[i], pivot_cp[i]) \
            - gk_price(S, Ks[i], T, r_d, r_f, sigma_atm, pivot_cp[i])
        adjustment += w[i] * coc

    return gk_price(S, K, T, r_d, r_f, sigma_atm, cp) + adjustment


def vv_implied_vol(S, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol):
    """Invert the VV call price to obtain the VV-interpolated implied vol."""
    if T <= 1e-10:
        return atm_vol

    target = vv_price(S, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol, 1)

    # Guard: if the target is below the intrinsic, return ATM vol
    intrinsic = max(S * np.exp(-r_f * T) - K * np.exp(-r_d * T), 0.0)
    if target <= intrinsic + 1e-8:
        return atm_vol

    try:
        return brentq(
            lambda sig: gk_price(S, K, T, r_d, r_f, sig, 1) - target,
            1e-4, 5.0, xtol=1e-10, maxiter=200,
        )
    except (ValueError, RuntimeError):
        return atm_vol


def vv_smile(S, T, r_d, r_f, atm_vol, p25_vol, c25_vol, n_strikes=50):
    """Generate the full VV smile from ~10-delta put to ~10-delta call.

    Returns
    -------
    dict with keys: strikes, vols, deltas
    """
    F = S * np.exp((r_d - r_f) * T)
    # Bracket: 10-delta put to 10-delta call strikes (wide range)
    K_lo = _delta_to_strike(S, T, r_d, r_f, atm_vol, -0.10, -1)
    K_hi = _delta_to_strike(S, T, r_d, r_f, atm_vol, 0.10, 1)

    if K_lo >= K_hi:
        F = S * np.exp((r_d - r_f) * T)
        K_lo = F * 0.85
        K_hi = F * 1.15

    strikes = np.linspace(K_lo, K_hi, n_strikes)
    vols = np.array([vv_implied_vol(S, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol)
                     for K in strikes])

    # Compute call delta at each strike using its VV vol
    deltas = np.array([
        np.exp(-r_f * T) * norm.cdf(_gk_d1_d2(S, K, T, r_d, r_f, vol)[0])
        for K, vol in zip(strikes, vols)
    ])

    return {"strikes": strikes, "vols": vols, "deltas": deltas}


def vv_greeks(S, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol, cp):
    """Full Greeks under Vanna-Volga pricing (numerical bump-and-reprice).

    Returns
    -------
    dict with: price, delta, gamma, vega, theta, rho_d, rho_f,
               vanna, volga, charm
    """
    def _price(s, k, t, rd, rf, av, pv, cv, flag):
        return vv_price(s, k, max(t, 1e-10), rd, rf, av, pv, cv, flag)

    price = _price(S, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol, cp)

    # Bump sizes -- 0.01 % of value for spot, 0.01 vol pt for vol, 1e-5 for rates
    dS = S * 1e-4
    dV = 1e-4
    dR = 1e-5
    dT = 1.0 / 365.0

    # Delta & gamma (central difference on spot)
    p_up = _price(S + dS, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol, cp)
    p_dn = _price(S - dS, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol, cp)
    delta_val = (p_up - p_dn) / (2 * dS)
    gamma_val = (p_up - 2 * price + p_dn) / (dS ** 2)

    # Vega (bump all three vols by dV)
    p_vup = _price(S, K, T, r_d, r_f, atm_vol + dV, p25_vol + dV, c25_vol + dV, cp)
    p_vdn = _price(S, K, T, r_d, r_f, atm_vol - dV, p25_vol - dV, c25_vol - dV, cp)
    vega_val = (p_vup - p_vdn) / (2 * dV)

    # Theta (one-day decay)
    if T > dT:
        p_t = _price(S, K, T - dT, r_d, r_f, atm_vol, p25_vol, c25_vol, cp)
        theta_val = (p_t - price) / dT
    else:
        theta_val = 0.0

    # Rho domestic
    p_rd_up = _price(S, K, T, r_d + dR, r_f, atm_vol, p25_vol, c25_vol, cp)
    p_rd_dn = _price(S, K, T, r_d - dR, r_f, atm_vol, p25_vol, c25_vol, cp)
    rho_d_val = (p_rd_up - p_rd_dn) / (2 * dR)

    # Rho foreign
    p_rf_up = _price(S, K, T, r_d, r_f + dR, atm_vol, p25_vol, c25_vol, cp)
    p_rf_dn = _price(S, K, T, r_d, r_f - dR, atm_vol, p25_vol, c25_vol, cp)
    rho_f_val = (p_rf_up - p_rf_dn) / (2 * dR)

    # Vanna (d(delta)/d(vol)) via cross bump
    p_su_vu = _price(S + dS, K, T, r_d, r_f, atm_vol + dV, p25_vol + dV, c25_vol + dV, cp)
    p_su_vd = _price(S + dS, K, T, r_d, r_f, atm_vol - dV, p25_vol - dV, c25_vol - dV, cp)
    p_sd_vu = _price(S - dS, K, T, r_d, r_f, atm_vol + dV, p25_vol + dV, c25_vol + dV, cp)
    p_sd_vd = _price(S - dS, K, T, r_d, r_f, atm_vol - dV, p25_vol - dV, c25_vol - dV, cp)
    vanna_val = (p_su_vu - p_su_vd - p_sd_vu + p_sd_vd) / (4 * dS * dV)

    # Volga (d^2 price / d sigma^2)
    volga_val = (p_vup - 2 * price + p_vdn) / (dV ** 2)

    # Charm (d(delta)/d(T)) via finite diff
    if T > dT:
        p_su_t = _price(S + dS, K, T - dT, r_d, r_f, atm_vol, p25_vol, c25_vol, cp)
        p_sd_t = _price(S - dS, K, T - dT, r_d, r_f, atm_vol, p25_vol, c25_vol, cp)
        delta_t = (p_su_t - p_sd_t) / (2 * dS)
        charm_val = (delta_t - delta_val) / dT
    else:
        charm_val = 0.0

    return {
        "price": price,
        "delta": delta_val,
        "gamma": gamma_val,
        "vega": vega_val,
        "theta": theta_val,
        "rho_d": rho_d_val,
        "rho_f": rho_f_val,
        "vanna": vanna_val,
        "volga": volga_val,
        "charm": charm_val,
    }


# =========================================================================
# SABR helper (Hagan et al. 2002 formula)
# =========================================================================

def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """SABR implied volatility via the Hagan et al. (2002) approximation."""
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return 1e-6

    # ATM limit
    if abs(F - K) < 1e-12:
        FK_beta = F ** (1.0 - beta)
        v = (alpha / FK_beta) * (
            1.0 + (
                (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / FK_beta ** 2
                + 0.25 * rho * beta * nu * alpha / FK_beta
                + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
            ) * T
        )
        return max(v, 1e-6)

    FK = F * K
    FK_beta2 = FK ** ((1.0 - beta) / 2.0)
    log_FK = np.log(F / K)

    z = (nu / alpha) * FK_beta2 * log_FK
    disc = np.sqrt(max(1.0 - 2.0 * rho * z + z ** 2, 1e-12))
    x_z = np.log((disc + z - rho) / max(1.0 - rho, 1e-12))
    if abs(x_z) < 1e-12:
        x_z = 1e-12

    prefix = alpha / (
        FK_beta2 * (
            1.0
            + (1.0 - beta) ** 2 / 24.0 * log_FK ** 2
            + (1.0 - beta) ** 4 / 1920.0 * log_FK ** 4
        )
    )

    correction = 1.0 + (
        (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / (FK ** (1.0 - beta))
        + 0.25 * rho * beta * nu * alpha / FK_beta2
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    ) * T

    return max(prefix * z / x_z * correction, 1e-6)


# =========================================================================
# VV vs SABR comparison
# =========================================================================

def vv_vs_sabr(S, T, r_d, r_f, atm_vol, p25_vol, c25_vol,
               sabr_alpha, sabr_rho, sabr_nu, sabr_beta=0.5):
    """Compare the VV smile against a SABR smile.

    Returns a DataFrame with columns:
        delta, strike, vv_vol, sabr_vol, diff
    """
    F = S * np.exp((r_d - r_f) * T)

    smile = vv_smile(S, T, r_d, r_f, atm_vol, p25_vol, c25_vol, n_strikes=50)
    strikes = smile["strikes"]
    vv_vols = smile["vols"]
    deltas = smile["deltas"]

    sabr_vols = np.array([
        sabr_vol(F, K, T, sabr_alpha, sabr_beta, sabr_rho, sabr_nu)
        for K in strikes
    ])

    return pd.DataFrame({
        "delta": np.round(deltas, 4),
        "strike": np.round(strikes, 6),
        "vv_vol": np.round(vv_vols, 6),
        "sabr_vol": np.round(sabr_vols, 6),
        "diff": np.round(vv_vols - sabr_vols, 6),
    })
