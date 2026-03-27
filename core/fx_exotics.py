"""
Exotic FX Options Pricing Engine
=================================
Analytical formulas where available (barriers, digitals, one-touch, forward start,
geometric Asians, floating lookbacks), Monte Carlo simulation otherwise (double
barriers, range accruals, arithmetic Asians, fixed lookbacks, best-of/worst-of,
TARFs).  All MC uses antithetic variates for variance reduction.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Callable


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _gk_d1d2(S, K, T, r_d, r_f, sigma):
    """Garman-Kohlhagen d1, d2."""
    sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / sqrt_T
    d2 = d1 - sqrt_T
    return d1, d2


def _gk_price(S, K, T, r_d, r_f, sigma, cp):
    """Garman-Kohlhagen vanilla FX option price.  cp: 1 = call, -1 = put."""
    if T <= 0:
        return max(cp * (S - K), 0.0)
    d1, d2 = _gk_d1d2(S, K, T, r_d, r_f, sigma)
    return cp * (S * np.exp(-r_f * T) * norm.cdf(cp * d1)
                 - K * np.exp(-r_d * T) * norm.cdf(cp * d2))


def _mc_paths(S, T, r_d, r_f, sigma, n_paths, n_steps, seed=None):
    """Generate GBM paths with antithetic variates.

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps + 1)
        Simulated spot paths including S at t=0.
    dt : float
        Time increment per step.
    """
    rng = np.random.RandomState(seed)
    dt = T / n_steps
    half = n_paths // 2

    Z = rng.standard_normal((half, n_steps))
    Z = np.vstack([Z, -Z])                           # antithetic

    drift = (r_d - r_f - 0.5 * sigma ** 2) * dt
    diff = sigma * np.sqrt(dt)

    log_S = np.zeros((Z.shape[0], n_steps + 1))
    log_S[:, 0] = np.log(S)
    for t in range(n_steps):
        log_S[:, t + 1] = log_S[:, t] + drift + diff * Z[:, t]

    return np.exp(log_S), dt


def _correlated_mc_paths(S1, S2, T, r_d1, r_d2, sigma1, sigma2, rho,
                          n_paths, n_steps, seed=None, r_f1=0.0, r_f2=0.0):
    """Correlated two-asset GBM paths (antithetic).

    Returns paths1, paths2 each of shape (n_paths, n_steps + 1).
    """
    rng = np.random.RandomState(seed)
    dt = T / n_steps
    half = n_paths // 2

    Z1 = rng.standard_normal((half, n_steps))
    Z2_indep = rng.standard_normal((half, n_steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2_indep

    Z1 = np.vstack([Z1, -Z1])
    Z2 = np.vstack([Z2, -Z2])

    def _build(S0, r_d, r_f, sigma, Z):
        drift = (r_d - r_f - 0.5 * sigma ** 2) * dt
        diff = sigma * np.sqrt(dt)
        log_S = np.zeros((Z.shape[0], n_steps + 1))
        log_S[:, 0] = np.log(S0)
        for t in range(n_steps):
            log_S[:, t + 1] = log_S[:, t] + drift + diff * Z[:, t]
        return np.exp(log_S)

    return _build(S1, r_d1, r_f1, sigma1, Z1), _build(S2, r_d2, r_f2, sigma2, Z2)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Single Barrier Options  (Reiner-Rubinstein 1991)
# ═══════════════════════════════════════════════════════════════════════════════

def barrier_price(S, K, B, T, r_d, r_f, sigma, cp, barrier_type, rebate=0.0):
    """Analytical single-barrier FX option price.

    Parameters
    ----------
    cp : int   1 = call, -1 = put.
    barrier_type : str
        One of 'down-and-in-call', 'down-and-out-call', 'up-and-in-call',
        'up-and-out-call', 'down-and-in-put', 'down-and-out-put',
        'up-and-in-put', 'up-and-out-put'.
    rebate : float  Cash rebate paid if knock-out occurs (or if knock-in never triggers).
    """
    if T <= 0 or sigma <= 1e-10:
        return max(cp * (S - K), 0.0)

    mu = (r_d - r_f - 0.5 * sigma ** 2) / (sigma ** 2)
    lam = np.sqrt(mu ** 2 + 2 * r_d / (sigma ** 2))
    sqrt_T = sigma * np.sqrt(T)

    x1 = np.log(S / K) / sqrt_T + (1 + mu) * sqrt_T
    x2 = np.log(S / B) / sqrt_T + (1 + mu) * sqrt_T
    y1 = np.log(B ** 2 / (S * K)) / sqrt_T + (1 + mu) * sqrt_T
    y2 = np.log(B / S) / sqrt_T + (1 + mu) * sqrt_T
    z  = np.log(B / S) / sqrt_T + lam * sqrt_T

    def _A(phi):
        return (phi * S * np.exp(-r_f * T) * norm.cdf(phi * x1)
                - phi * K * np.exp(-r_d * T) * norm.cdf(phi * (x1 - sqrt_T)))

    def _B(phi):
        return (phi * S * np.exp(-r_f * T) * norm.cdf(phi * x2)
                - phi * K * np.exp(-r_d * T) * norm.cdf(phi * (x2 - sqrt_T)))

    def _safe_pow(base, exp):
        """Safe power for barrier ratios — prevent overflow."""
        return np.exp(np.clip(exp * np.log(max(base, 1e-20)), -100, 100))

    def _C(phi, eta):
        fac = _safe_pow(B / S, 2 * (mu + 1))
        fac2 = _safe_pow(B / S, 2 * mu)
        return (phi * S * np.exp(-r_f * T) * fac * norm.cdf(eta * y1)
                - phi * K * np.exp(-r_d * T) * fac2 * norm.cdf(eta * (y1 - sqrt_T)))

    def _D(phi, eta):
        fac = _safe_pow(B / S, 2 * (mu + 1))
        fac2 = _safe_pow(B / S, 2 * mu)
        return (phi * S * np.exp(-r_f * T) * fac * norm.cdf(eta * y2)
                - phi * K * np.exp(-r_d * T) * fac2 * norm.cdf(eta * (y2 - sqrt_T)))

    def _E(eta):
        return (rebate * np.exp(-r_d * T)
                * (norm.cdf(eta * (x2 - sqrt_T))
                   - _safe_pow(B / S, 2 * mu) * norm.cdf(eta * (y2 - sqrt_T))))

    def _F(eta):
        return (rebate * (_safe_pow(B / S, mu + lam) * norm.cdf(eta * z)
                          + _safe_pow(B / S, mu - lam) * norm.cdf(eta * (z - 2 * lam * sqrt_T))))

    bt = barrier_type.lower()

    # --- down-and-in / down-and-out calls  (B < S) ---
    if bt == 'down-and-in-call':
        if S <= B:
            return _gk_price(S, K, T, r_d, r_f, sigma, 1) + _F(1)
        if K >= B:
            return _C(1, 1) + _F(1)
        return _A(1) - _B(1) + _D(1, 1) + _F(1)

    if bt == 'down-and-out-call':
        if S <= B:
            return rebate * np.exp(-r_d * T)
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        di = barrier_price(S, K, B, T, r_d, r_f, sigma, 1, 'down-and-in-call', 0.0)
        return vanilla - di + _E(1)

    # --- up-and-in / up-and-out calls  (B > S) ---
    if bt == 'up-and-in-call':
        if S >= B:
            return _gk_price(S, K, T, r_d, r_f, sigma, 1) + _F(-1)
        if K >= B:
            return _A(1) + _F(-1)
        return _B(1) - _C(1, -1) + _D(1, -1) + _F(-1)

    if bt == 'up-and-out-call':
        if S >= B:
            return rebate * np.exp(-r_d * T)
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        ui = barrier_price(S, K, B, T, r_d, r_f, sigma, 1, 'up-and-in-call', 0.0)
        return vanilla - ui + _E(-1)

    # --- down-and-in / down-and-out puts  (B < S) ---
    if bt == 'down-and-in-put':
        if S <= B:
            return _gk_price(S, K, T, r_d, r_f, sigma, -1) + _F(1)
        if K >= B:
            return _A(-1) - _B(-1) + _D(-1, 1) + _F(1)
        return _C(-1, 1) + _F(1)

    if bt == 'down-and-out-put':
        if S <= B:
            return rebate * np.exp(-r_d * T)
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, -1)
        di = barrier_price(S, K, B, T, r_d, r_f, sigma, -1, 'down-and-in-put', 0.0)
        return vanilla - di + _E(1)

    # --- up-and-in / up-and-out puts  (B > S) ---
    if bt == 'up-and-in-put':
        if S >= B:
            return _gk_price(S, K, T, r_d, r_f, sigma, -1) + _F(-1)
        if K >= B:
            return _A(-1) - _B(-1) + _D(-1, -1) + _C(-1, -1) + _F(-1)
        return _C(-1, -1) + _F(-1)

    if bt == 'up-and-out-put':
        if S >= B:
            return rebate * np.exp(-r_d * T)
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, -1)
        ui = barrier_price(S, K, B, T, r_d, r_f, sigma, -1, 'up-and-in-put', 0.0)
        return vanilla - ui + _E(-1)

    raise ValueError(f"Unknown barrier_type '{barrier_type}'")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Double Barrier Options  (Monte Carlo with Brownian bridge correction)
# ═══════════════════════════════════════════════════════════════════════════════

def double_barrier_price(S, K, B_up, B_down, T, r_d, r_f, sigma, cp,
                          barrier_type='knock-out', n_paths=50000, n_steps=252,
                          seed=None):
    """Double barrier option priced via MC with Brownian-bridge barrier correction.

    barrier_type : 'knock-out' or 'knock-in'
    """
    paths, dt = _mc_paths(S, T, r_d, r_f, sigma, n_paths, n_steps, seed)
    n = paths.shape[0]

    # Brownian-bridge barrier-hit probability between consecutive steps
    alive = np.ones(n, dtype=bool)
    for t in range(n_steps):
        S_t = paths[:, t]
        S_t1 = paths[:, t + 1]
        s_min = np.minimum(S_t, S_t1)
        s_max = np.maximum(S_t, S_t1)

        # Deterministic check
        hit = (s_min <= B_down) | (s_max >= B_up)

        # Bridge correction: probability of hitting barrier between steps
        # P(min < B | S_t, S_{t+1}) = exp(-2 * log(S_t/B)*log(S_{t+1}/B) / (sigma^2 * dt))
        for barrier in [B_down, B_up]:
            if barrier == B_down:
                mask = (~hit) & (s_min > barrier)
            else:
                mask = (~hit) & (s_max < barrier)
            if not np.any(mask):
                continue
            l1 = np.log(S_t[mask] / barrier)
            l2 = np.log(S_t1[mask] / barrier)
            p_cross = np.exp(-2.0 * l1 * l2 / (sigma ** 2 * dt))
            bridge_hit = np.random.random(mask.sum()) < p_cross
            idx = np.where(mask)[0]
            hit[idx[bridge_hit]] = True

        alive &= ~hit

    ST = paths[:, -1]
    payoff = np.maximum(cp * (ST - K), 0.0)

    if barrier_type == 'knock-out':
        payoff *= alive
    else:  # knock-in
        payoff *= ~alive

    price = np.exp(-r_d * T) * np.mean(payoff)
    se = np.exp(-r_d * T) * np.std(payoff) / np.sqrt(n)
    return {'price': price, 'std_error': se, 'n_paths': n}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Digital (Binary) Options
# ═══════════════════════════════════════════════════════════════════════════════

def digital_price(S, K, T, r_d, r_f, sigma, cp, payout=1.0):
    """Cash-or-nothing digital option (analytical).  cp: 1 call, -1 put."""
    if T <= 0:
        return payout if cp * (S - K) > 0 else 0.0
    _, d2 = _gk_d1d2(S, K, T, r_d, r_f, sigma)
    return payout * np.exp(-r_d * T) * norm.cdf(cp * d2)


def digital_greeks(S, K, T, r_d, r_f, sigma, cp, payout=1.0):
    """Full Greeks for a cash-or-nothing digital via analytical derivatives."""
    if T <= 0 or sigma <= 1e-10:
        return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

    d1, d2 = _gk_d1d2(S, K, T, r_d, r_f, sigma)
    sqrt_T = sigma * np.sqrt(T)
    df = np.exp(-r_d * T)
    pdf_d2 = norm.pdf(d2)

    # delta = payout * df * n(d2) / (S * sigma * sqrt(T))
    delta = cp * payout * df * pdf_d2 / (S * sqrt_T)

    # gamma (derivative of delta w.r.t. S)
    gamma = -cp * payout * df * pdf_d2 * d1 / (S ** 2 * sqrt_T * sigma * np.sqrt(T))

    # vega (derivative w.r.t. sigma, expressed per 1% move)
    vega = -cp * payout * df * pdf_d2 * d1 / sigma / 100.0

    # theta (derivative w.r.t. time, per day)
    theta_val = cp * payout * df * (
        r_d * norm.cdf(cp * d2)
        + pdf_d2 * ((r_d - r_f - 0.5 * sigma ** 2) / sqrt_T - d2 / (2 * T))
    )
    theta = -theta_val / 365.0

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. One-Touch / No-Touch
# ═══════════════════════════════════════════════════════════════════════════════

def one_touch_price(S, B, T, r_d, r_f, sigma, payout=1.0):
    """One-touch: pays *payout* if barrier B is ever breached during [0, T].

    Analytical closed form using the reflection principle for GBM.
    """
    if sigma <= 0.001:
        # In near-zero vol regime, barrier is hit iff spot is already past it
        return payout * np.exp(-r_d * T) if abs(B - S) < 1e-12 else 0.0
    if T <= 0:
        # Barrier already breached if spot is at or past the barrier
        if abs(B - S) < 1e-12:
            return payout * np.exp(-r_d * T)
        return 0.0
    if abs(B - S) < 1e-12:
        return payout * np.exp(-r_d * T)

    mu = (r_d - r_f - 0.5 * sigma ** 2) / (sigma ** 2)
    lam = np.sqrt(mu ** 2 + 2 * r_d / (sigma ** 2))
    sqrt_T = sigma * np.sqrt(T)
    log_BS = np.log(B / S)

    eta = 1.0 if B > S else -1.0  # up or down barrier

    term1 = (B / S) ** (mu + lam) * norm.cdf(eta * (log_BS + lam * sigma ** 2 * T) / sqrt_T)
    term2 = (B / S) ** (mu - lam) * norm.cdf(eta * (log_BS - lam * sigma ** 2 * T) / sqrt_T)

    return payout * (term1 + term2)


def no_touch_price(S, B, T, r_d, r_f, sigma, payout=1.0):
    """No-touch: pays if barrier is never hit.  = payout*DF - one_touch."""
    return payout * np.exp(-r_d * T) - one_touch_price(S, B, T, r_d, r_f, sigma, payout)


def double_no_touch_price(S, B_up, B_down, T, r_d, r_f, sigma, payout=1.0,
                           n_paths=50000, n_steps=252, seed=None):
    """Double no-touch: pays if spot stays within [B_down, B_up].  MC simulation
    with Brownian bridge correction for barrier crossing between discrete steps."""
    if abs(B_up - B_down) < 1e-10 or B_up <= B_down:
        return {'price': 0.0, 'std_error': 0.0, 'prob_no_touch': 0.0}
    paths, dt = _mc_paths(S, T, r_d, r_f, sigma, n_paths, n_steps, seed)
    n = paths.shape[0]

    alive = np.ones(n, dtype=bool)
    for t in range(n_steps):
        S_t = paths[:, t]
        S_t1 = paths[:, t + 1]
        s_min = np.minimum(S_t, S_t1)
        s_max = np.maximum(S_t, S_t1)

        # Deterministic check
        hit = (s_min <= B_down) | (s_max >= B_up)

        # Brownian bridge correction: probability of hitting barrier between steps
        for barrier in [B_down, B_up]:
            if barrier == B_down:
                mask = (~hit) & (s_min > barrier)
            else:
                mask = (~hit) & (s_max < barrier)
            if not np.any(mask):
                continue
            l1 = np.log(S_t[mask] / barrier)
            l2 = np.log(S_t1[mask] / barrier)
            p_cross = np.exp(-2.0 * l1 * l2 / (sigma ** 2 * dt))
            bridge_hit = np.random.random(mask.sum()) < p_cross
            idx = np.where(mask)[0]
            hit[idx[bridge_hit]] = True

        alive &= ~hit

    survived = alive
    price = payout * np.exp(-r_d * T) * np.mean(survived)
    se = payout * np.exp(-r_d * T) * np.std(survived) / np.sqrt(n)
    return {'price': price, 'std_error': se, 'prob_no_touch': np.mean(survived)}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Range Accrual
# ═══════════════════════════════════════════════════════════════════════════════

def range_accrual_price(S, B_low, B_high, T, r_d, r_f, sigma, payout=1.0,
                         fixing_freq='daily', n_paths=50000, seed=None):
    """Range accrual: payout proportional to fraction of fixings inside range.

    fixing_freq : 'daily' | 'weekly' | 'monthly'
    """
    freq_map = {'daily': 252, 'weekly': 52, 'monthly': 12}
    n_fixings = max(1, int(freq_map.get(fixing_freq, 252) * T))
    n_steps = max(n_fixings, 252)

    paths, _ = _mc_paths(S, T, r_d, r_f, sigma, n_paths, n_steps, seed)

    # Pick fixing indices evenly spaced through the path
    fix_idx = np.round(np.linspace(1, n_steps, n_fixings)).astype(int)
    fix_spots = paths[:, fix_idx]

    in_range = (fix_spots >= B_low) & (fix_spots <= B_high)
    accrual_frac = in_range.mean(axis=1)

    payoff = payout * accrual_frac
    price = np.exp(-r_d * T) * np.mean(payoff)
    se = np.exp(-r_d * T) * np.std(payoff) / np.sqrt(n_paths)
    return {'price': price, 'std_error': se,
            'expected_accrual': float(np.mean(accrual_frac)),
            'n_fixings': n_fixings}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Asian Options
# ═══════════════════════════════════════════════════════════════════════════════

def asian_geometric_price(S, K, T, r_d, r_f, sigma, cp, n_fixings):
    """Closed-form geometric-average Asian (Kemna-Vorst 1990).

    cp : 1 call, -1 put.
    """
    sigma_a = sigma * np.sqrt((2 * n_fixings + 1) / (6 * (n_fixings + 1)))
    r_a = 0.5 * (r_d - r_f - 0.5 * sigma ** 2
                 + sigma_a ** 2)
    # Price as a standard BS call/put with adjusted vol & rate
    d1 = (np.log(S / K) + (r_a + 0.5 * sigma_a ** 2) * T) / (sigma_a * np.sqrt(T))
    d2 = d1 - sigma_a * np.sqrt(T)
    df = np.exp(-r_d * T)
    if cp == 1:
        return S * np.exp((r_a - r_d) * T) * norm.cdf(d1) - K * df * norm.cdf(d2)
    return K * df * norm.cdf(-d2) - S * np.exp((r_a - r_d) * T) * norm.cdf(-d1)


def asian_price(S, K, T, r_d, r_f, sigma, cp, fixing_freq='monthly',
                average_type='arithmetic', n_paths=50000, seed=None):
    """Asian option (average-price).

    arithmetic : MC (no closed form).
    geometric  : Kemna-Vorst analytical.
    """
    freq_map = {'daily': 252, 'weekly': 52, 'monthly': 12}
    n_fixings = max(1, int(freq_map.get(fixing_freq, 12) * T))

    if average_type == 'geometric':
        return {'price': asian_geometric_price(S, K, T, r_d, r_f, sigma, cp, n_fixings),
                'std_error': 0.0, 'n_fixings': n_fixings, 'method': 'analytical'}

    # Arithmetic Asian via MC with geometric control variate
    n_steps = max(n_fixings, 252)
    paths, _ = _mc_paths(S, T, r_d, r_f, sigma, n_paths, n_steps, seed)

    if n_fixings <= 1:
        fix_idx = np.array([n_steps])  # single fixing uses final spot
    else:
        fix_idx = np.round(np.linspace(1, n_steps, n_fixings)).astype(int)
    fix_spots = paths[:, fix_idx]

    arith_avg = fix_spots.mean(axis=1)
    geom_avg = np.exp(np.log(fix_spots).mean(axis=1))

    payoff_arith = np.maximum(cp * (arith_avg - K), 0.0)
    payoff_geom = np.maximum(cp * (geom_avg - K), 0.0)

    # Control variate: E[geom] is known analytically
    geom_analytic = asian_geometric_price(S, K, T, r_d, r_f, sigma, cp, n_fixings)
    geom_mc = np.exp(-r_d * T) * np.mean(payoff_geom)

    # beta coefficient for control variate
    cov_ag = np.cov(payoff_arith, payoff_geom)[0, 1]
    var_g = np.var(payoff_geom)
    beta_cv = cov_ag / var_g if var_g > 1e-12 else 0.0

    adjusted = payoff_arith - beta_cv * (payoff_geom - np.exp(r_d * T) * geom_analytic)
    price = np.exp(-r_d * T) * np.mean(adjusted)
    se = np.exp(-r_d * T) * np.std(adjusted) / np.sqrt(n_paths)

    return {'price': price, 'std_error': se, 'n_fixings': n_fixings, 'method': 'mc_cv'}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Lookback Options
# ═══════════════════════════════════════════════════════════════════════════════

def _lookback_floating_analytical(S, T, r_d, r_f, sigma, cp):
    """Goldman-Sosin-Gatto (1979) closed form for floating-strike lookback.

    Call payoff: S_T - S_min ;  Put payoff: S_max - S_T.
    """
    b = r_d - r_f
    sqrt_T = sigma * np.sqrt(T)

    if cp == 1:  # floating-strike call  (payoff = S_T - S_min)
        a1 = (b + 0.5 * sigma ** 2) * T / sqrt_T
        a2 = a1 - sqrt_T
        part1 = S * np.exp(-r_f * T) * norm.cdf(a1)
        part2 = S * np.exp(-r_d * T) * norm.cdf(a2)
        if abs(b) > 1e-12:
            part3 = (S * np.exp(-r_d * T) * sigma ** 2 / (2 * b)
                     * (-(S / S) ** (-2 * b / sigma ** 2)
                        * norm.cdf(a1 - 2 * b * np.sqrt(T) / sigma)
                        + np.exp(b * T) * norm.cdf(a1)))
        else:
            part3 = S * np.exp(-r_d * T) * sqrt_T * (norm.pdf(a1) + a1 * norm.cdf(a1))
        return part1 - part2 + part3
    else:  # floating-strike put  (payoff = S_max - S_T)
        a1 = (b + 0.5 * sigma ** 2) * T / sqrt_T
        a2 = a1 - sqrt_T
        part1 = -S * np.exp(-r_f * T) * norm.cdf(-a1)
        part2 = S * np.exp(-r_d * T) * norm.cdf(-a2)
        if abs(b) > 1e-12:
            part3 = (S * np.exp(-r_d * T) * sigma ** 2 / (2 * b)
                     * ((S / S) ** (-2 * b / sigma ** 2)
                        * norm.cdf(-a1 + 2 * b * np.sqrt(T) / sigma)
                        - np.exp(b * T) * norm.cdf(-a1)))
        else:
            part3 = S * np.exp(-r_d * T) * sqrt_T * (norm.pdf(a1) + a1 * (norm.cdf(a1) - 1))
        return -part1 + part2 + part3


def lookback_price(S, T, r_d, r_f, sigma, cp, lookback_type='floating', K=None,
                    n_paths=50000, n_steps=252, seed=None):
    """Lookback option.

    floating strike : analytical (Goldman-Sosin-Gatto).
    fixed strike    : MC simulation.
    """
    if lookback_type == 'floating':
        price = _lookback_floating_analytical(S, T, r_d, r_f, sigma, cp)
        return {'price': price, 'std_error': 0.0, 'method': 'analytical'}

    # Fixed-strike lookback via MC
    if K is None:
        K = S
    paths, _ = _mc_paths(S, T, r_d, r_f, sigma, n_paths, n_steps, seed)
    if cp == 1:
        payoff = np.maximum(paths.max(axis=1) - K, 0.0)
    else:
        payoff = np.maximum(K - paths.min(axis=1), 0.0)

    price = np.exp(-r_d * T) * np.mean(payoff)
    se = np.exp(-r_d * T) * np.std(payoff) / np.sqrt(n_paths)
    return {'price': price, 'std_error': se, 'method': 'mc'}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Forward Start Options  (Rubinstein 1991)
# ═══════════════════════════════════════════════════════════════════════════════

def forward_start_price(S, T_start, T_end, r_d, r_f, sigma, cp, moneyness=1.0):
    """Forward-starting option: strike set at T_start as moneyness * S(T_start).

    Analytical via Rubinstein (1991).  The forward-start call is equivalent to
    e^{-r_f * T_start} vanilla calls with strike = moneyness * S today and
    time to expiry = T_end - T_start.
    """
    T_rem = T_end - T_start
    if T_rem <= 0:
        return 0.0
    K_equiv = moneyness * S
    # Forward-start scaling: the option is worth e^{-r_f * T_start} times a
    # vanilla with remaining maturity T_rem
    return np.exp(-r_f * T_start) * _gk_price(S, K_equiv, T_rem, r_d, r_f, sigma, cp)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Best-of / Worst-of (Rainbow Options)
# ═══════════════════════════════════════════════════════════════════════════════

def best_of_price(S1, S2, K, T, r_d1, r_d2, r_f, sigma1, sigma2, rho, cp,
                   option_type='best-of', n_paths=50000, seed=None):
    """Two-asset rainbow option via correlated MC.

    best-of call  : max(perf1, perf2, 0)  where perf_i = S_i(T)/S_i(0) - 1
    worst-of call : payoff only if BOTH assets perform above strike
    Generalised to use strike K as a performance threshold.
    """
    n_steps = max(int(252 * T), 50)
    paths1, paths2 = _correlated_mc_paths(S1, S2, T, r_d1, r_d2,
                                           sigma1, sigma2, rho,
                                           n_paths, n_steps, seed,
                                           r_f1=r_f, r_f2=r_f)
    perf1 = paths1[:, -1] / S1 - 1.0
    perf2 = paths2[:, -1] / S2 - 1.0

    if option_type == 'best-of':
        perf = np.maximum(perf1, perf2)
    else:  # worst-of
        perf = np.minimum(perf1, perf2)

    payoff = np.maximum(cp * (perf - K), 0.0)
    avg_rd = 0.5 * (r_d1 + r_d2)
    price = np.exp(-avg_rd * T) * np.mean(payoff)
    se = np.exp(-avg_rd * T) * np.std(payoff) / np.sqrt(n_paths)
    return {'price': price, 'std_error': se, 'n_paths': n_paths,
            'avg_perf1': float(np.mean(perf1)), 'avg_perf2': float(np.mean(perf2))}


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TARF  (Target Accrual Redemption Forward)
# ═══════════════════════════════════════════════════════════════════════════════

def tarf_price(S, K, B, T, r_d, r_f, sigma, n_fixings=12, target_profit=0.05,
               leverage=2, n_paths=50000, seed=None):
    """Target Accrual Redemption Forward.

    At each fixing date:
        if S(t) > K : client gains (S(t)/K - 1), accrues toward target
        if S(t) < K : client loses leverage * (1 - S(t)/K)
    Optional knock-out barrier B (upper) triggers early termination.
    Structure terminates when accumulated gains reach target_profit.

    Returns dict with price (MTM from client perspective), probability of
    early termination, expected number of fixings, and expected P&L.
    """
    n_steps = max(n_fixings * 21, 252)
    paths, _ = _mc_paths(S, T, r_d, r_f, sigma, n_paths, n_steps, seed)

    fix_idx = np.round(np.linspace(1, n_steps, n_fixings)).astype(int)
    dt_fix = T / n_fixings
    n = paths.shape[0]

    total_pnl = np.zeros(n)
    terminated = np.zeros(n, dtype=bool)
    fixings_used = np.full(n, n_fixings, dtype=int)
    accrued_gain = np.zeros(n)

    for i, fi in enumerate(fix_idx):
        active = ~terminated
        if not np.any(active):
            break
        spot = paths[active, fi]

        # Knock-out barrier check
        if B is not None and B > K:
            ko = spot >= B
            idx_active = np.where(active)[0]
            terminated[idx_active[ko]] = True
            fixings_used[idx_active[ko]] = i + 1
            active = ~terminated
            if not np.any(active):
                break
            spot = paths[active, fi]

        gain_mask = spot > K
        loss_mask = ~gain_mask

        pnl = np.zeros(active.sum())
        pnl[gain_mask] = spot[gain_mask] / K - 1.0
        pnl[loss_mask] = -leverage * (1.0 - spot[loss_mask] / K)

        idx_active = np.where(active)[0]
        total_pnl[idx_active] += pnl

        # Accrual toward target (only gains count)
        gains_only = np.zeros_like(pnl)
        gains_only[gain_mask] = pnl[gain_mask]
        accrued_gain[idx_active] += gains_only

        # Target termination
        reached = accrued_gain[idx_active] >= target_profit
        terminated[idx_active[reached]] = True
        fixings_used[idx_active[reached]] = np.minimum(
            fixings_used[idx_active[reached]], i + 1)

    # Discount total P&L to present
    avg_time = fixings_used * dt_fix
    disc_pnl = total_pnl * np.exp(-r_d * avg_time)

    price = float(np.mean(disc_pnl))
    prob_early = float(np.mean(fixings_used < n_fixings))
    exp_fix = float(np.mean(fixings_used))
    exp_pnl = float(np.mean(total_pnl))

    return {
        'price': price,
        'std_error': float(np.std(disc_pnl) / np.sqrt(n)),
        'prob_early_termination': prob_early,
        'expected_fixings': exp_fix,
        'expected_pnl': exp_pnl,
        'n_paths': n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical Greeks for any exotic
# ═══════════════════════════════════════════════════════════════════════════════

def exotic_greeks(price_func: Callable, base_params: dict,
                  bumps: Optional[dict] = None) -> dict:
    """Numerical Greeks via central finite differences.

    Parameters
    ----------
    price_func : callable
        Must accept **kwargs and return either a float or a dict with key 'price'.
    base_params : dict
        Base parameters for price_func.
    bumps : dict, optional
        {param_name: bump_size}.  Defaults to standard FX bumps.

    Returns
    -------
    dict with keys delta, gamma, vega, theta, rho_d, rho_f (where applicable).
    """
    if bumps is None:
        bumps = {}
    default_bumps = {
        'S': 0.01,        # 1% spot bump (relative)
        'sigma': 0.01,    # 1 vol point
        'T': 1.0 / 365,   # 1 day
        'r_d': 0.0001,    # 1 bp
        'r_f': 0.0001,
    }
    default_bumps.update(bumps)

    def _price(params):
        result = price_func(**params)
        return result['price'] if isinstance(result, dict) else result

    base_px = _price(base_params)
    greeks = {'price': base_px}

    for param, h in default_bumps.items():
        if param not in base_params:
            continue

        up_params = dict(base_params)
        dn_params = dict(base_params)

        if param == 'S':
            # Relative bump for spot
            S0 = base_params['S']
            up_params['S'] = S0 * (1 + h)
            dn_params['S'] = S0 * (1 - h)
            px_up = _price(up_params)
            px_dn = _price(dn_params)
            greeks['delta'] = (px_up - px_dn) / (2 * S0 * h)
            greeks['gamma'] = (px_up - 2 * base_px + px_dn) / (S0 * h) ** 2
        elif param == 'sigma':
            up_params['sigma'] = base_params['sigma'] + h
            dn_params['sigma'] = max(base_params['sigma'] - h, 0.001)
            greeks['vega'] = (_price(up_params) - _price(dn_params)) / (2 * h) / 100.0
        elif param == 'T':
            dn_params['T'] = max(base_params['T'] - h, 1e-6)
            greeks['theta'] = (_price(dn_params) - base_px) / h / 365.0
        elif param == 'r_d':
            up_params['r_d'] = base_params['r_d'] + h
            dn_params['r_d'] = base_params['r_d'] - h
            greeks['rho_d'] = (_price(up_params) - _price(dn_params)) / (2 * h) / 100.0
        elif param == 'r_f':
            up_params['r_f'] = base_params['r_f'] + h
            dn_params['r_f'] = base_params['r_f'] - h
            greeks['rho_f'] = (_price(up_params) - _price(dn_params)) / (2 * h) / 100.0

    return greeks


# ═══════════════════════════════════════════════════════════════════════════════
# Summary formatter
# ═══════════════════════════════════════════════════════════════════════════════

def exotic_summary(price, greeks: Optional[dict], product_type: str,
                   params: dict) -> dict:
    """Format pricing results into a standardised summary dictionary."""
    summary = {
        'product_type': product_type,
        'price': price if not isinstance(price, dict) else price.get('price', price),
        'params': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                       else v)
                   for k, v in params.items()
                   if not callable(v)},
    }
    if isinstance(price, dict):
        for k in ('std_error', 'prob_early_termination', 'expected_fixings',
                  'expected_pnl', 'expected_accrual', 'prob_no_touch',
                  'avg_perf1', 'avg_perf2', 'method', 'n_fixings', 'n_paths'):
            if k in price:
                summary[k] = price[k]

    if greeks is not None:
        summary['greeks'] = {k: round(v, 8) if isinstance(v, float) else v
                             for k, v in greeks.items() if k != 'price'}
    return summary
