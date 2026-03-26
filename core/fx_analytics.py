"""
FX Analytics Engine
====================
Institutional-grade quantitative analytics for FX options.
Provides vol surface analytics, realized vol cones, forward vol,
smile decomposition, relative value, correlation, carry, positioning,
and risk metrics that power the FX options workstation dashboard.
"""

import logging
import threading
import time as _time
from functools import wraps
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, percentileofscore, jarque_bera, linregress
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

# ── Simple TTL memo for expensive analytics functions ──
_memo_cache = {}
_memo_lock = threading.Lock()

def _ttl_memo(ttl_seconds=120):
    """Decorator: cache result by args for ttl_seconds."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Make args hashable (recursively convert mutable containers)
            def _hashable(x):
                if isinstance(x, list):
                    return tuple(_hashable(i) for i in x)
                if isinstance(x, dict):
                    return tuple(sorted((_hashable(k), _hashable(v)) for k, v in x.items()))
                if isinstance(x, set):
                    return frozenset(_hashable(i) for i in x)
                return x
            key = (fn.__name__,) + tuple(_hashable(a) for a in args) + tuple(sorted(kwargs.items()))
            with _memo_lock:
                entry = _memo_cache.get(key)
                if entry is not None:
                    ts, val = entry
                    if _time.time() - ts < ttl_seconds:
                        return val
            result = fn(*args, **kwargs)
            with _memo_lock:
                _memo_cache[key] = (_time.time(), result)
                # Evict old entries if cache grows too large
                if len(_memo_cache) > 500:
                    cutoff = _time.time() - ttl_seconds * 2
                    stale = [k for k, (t, _) in _memo_cache.items() if t < cutoff]
                    for k in stale:
                        _memo_cache.pop(k, None)
            return result
        return wrapper
    return decorator

from core.fx_conventions import (
    tenor_to_years, tenor_to_days, spot_delta,
    delta_to_strike, bf_rr_to_smile, FX_PAIR_REGISTRY,
)
FX_PAIRS = FX_PAIR_REGISTRY
TENORS = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "5Y"]
from core.bloomberg_fx import (
    get_fx_spots, get_fx_vol_surface, get_fx_historical_vol,
    get_fx_historical_spot, get_fx_rates, get_fx_realized_vol,
    get_fx_correlation, get_cftc_positioning,
)

logger = logging.getLogger(__name__)


def _to_close_array(data):
    """Convert get_fx_historical_spot result to 1D numpy array of close prices."""
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return None
        if "close" in data.columns:
            return data["close"].values
        elif "Close" in data.columns:
            return data["Close"].values
        return data.iloc[:, -1].values
    if isinstance(data, pd.Series):
        return data.values
    arr = np.asarray(data)
    if arr.ndim == 0 or len(arr) == 0:
        return None
    return arr


# Delta grid used for surface analytics
DELTA_GRID = [10, 25, 50, 75, 90]
DELTA_LABELS = ["10P", "25P", "ATM", "25C", "10C"]


# =========================================================================
#  Vol Surface Analytics
# =========================================================================

@_ttl_memo(ttl_seconds=120)
def vol_percentile(pair: str, tenor: str, metric: str = "ATM",
                   lookback_days: int = 252) -> dict:
    """
    Current implied vol level as a percentile of its N-day history.

    Parameters
    ----------
    pair : str       e.g. 'EURUSD'
    tenor : str      e.g. '3M'
    metric : str     'ATM', '25D_RR', '25D_BF', '10D_RR', '10D_BF'
    lookback_days : int  history window (default 252 ~ 1 year)

    Returns
    -------
    dict with current, mean, std, min, max, percentile, rank
    """
    hist = get_fx_historical_vol(pair, tenor, metric, lookback_days)
    if hist is None or len(hist) < 10:
        return None

    current = hist.iloc[-1] if hasattr(hist, 'iloc') else hist[-1]
    pct = percentileofscore(hist, current)

    return {
        "pair": pair,
        "tenor": tenor,
        "metric": metric,
        "current": float(current),
        "mean": float(np.mean(hist)),
        "std": float(np.std(hist)),
        "min": float(np.min(hist)),
        "max": float(np.max(hist)),
        "percentile": float(pct),
        "lookback_days": lookback_days,
    }


@_ttl_memo(ttl_seconds=120)
def vol_zscore(pair: str, tenor: str, metric: str = "ATM",
               lookback_days: int = 252) -> dict:
    """
    Z-score of current implied vol vs its N-day history.

    Returns
    -------
    dict with current, mean, std, zscore, interpretation
    """
    hist = get_fx_historical_vol(pair, tenor, metric, lookback_days)
    if hist is None or len(hist) < 10:
        return None

    current = hist.iloc[-1] if hasattr(hist, 'iloc') else hist[-1]
    mu = np.mean(hist)
    sigma = np.std(hist)
    z = (current - mu) / max(sigma, 1e-6)
    z = float(np.clip(z, -10.0, 10.0))

    if z > 2.0:
        interp = "VERY_HIGH"
    elif z > 1.0:
        interp = "HIGH"
    elif z < -2.0:
        interp = "VERY_LOW"
    elif z < -1.0:
        interp = "LOW"
    else:
        interp = "NORMAL"

    return {
        "pair": pair,
        "tenor": tenor,
        "metric": metric,
        "current": float(current),
        "mean": float(mu),
        "std": float(sigma),
        "zscore": float(z),
        "interpretation": interp,
        "lookback_days": lookback_days,
    }


def vol_percentile_surface(pair: str, lookback: int = 252) -> pd.DataFrame:
    """
    Full tenor x delta percentile grid for the vol surface.

    Returns DataFrame with tenors as rows, delta labels as columns.
    """
    tenors = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"]
    metrics = ["10D_BF", "25D_RR", "ATM", "25D_RR", "10D_BF"]
    metric_names = ["10P", "25P", "ATM", "25C", "10C"]

    rows = []
    for t in tenors:
        row = {"tenor": t}
        for m, label in zip(metrics, metric_names):
            info = vol_percentile(pair, t, m, lookback)
            row[label] = round(info["percentile"], 1) if info is not None else None
        rows.append(row)

    return pd.DataFrame(rows).set_index("tenor")


def vol_zscore_surface(pair: str, lookback: int = 252) -> pd.DataFrame:
    """
    Full tenor x delta z-score grid for the vol surface.

    Returns DataFrame with tenors as rows, delta labels as columns.
    """
    tenors = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"]
    metrics = ["10D_BF", "25D_RR", "ATM", "25D_RR", "10D_BF"]
    metric_names = ["10P", "25P", "ATM", "25C", "10C"]

    rows = []
    for t in tenors:
        row = {"tenor": t}
        for m, label in zip(metrics, metric_names):
            info = vol_zscore(pair, t, m, lookback)
            row[label] = round(info["zscore"], 2) if info is not None else None
        rows.append(row)

    return pd.DataFrame(rows).set_index("tenor")


def vol_change(pair: str, tenor: str, metric: str = "ATM",
               days_ago: int = 1) -> dict:
    """
    Change in implied vol metric from N days ago (absolute and percentage).
    """
    hist = get_fx_historical_vol(pair, tenor, metric, days_ago + 5)
    if hist is None or len(hist) < days_ago + 1:
        return None

    current = hist.iloc[-1] if hasattr(hist, 'iloc') else hist[-1]
    previous = hist.iloc[-(days_ago + 1)] if hasattr(hist, 'iloc') else hist[-(days_ago + 1)]
    abs_change = current - previous
    pct_change = abs_change / max(abs(previous), 1e-6) * 100

    return {
        "pair": pair,
        "tenor": tenor,
        "metric": metric,
        "current": float(current),
        "previous": float(previous),
        "abs_change": float(abs_change),
        "pct_change": float(pct_change),
        "days_ago": days_ago,
    }


def vol_surface_diff(pair: str, days_ago: int = 1) -> pd.DataFrame:
    """
    Difference between current vol surface and the surface N days ago.
    Returns DataFrame of vol changes across tenor x delta grid.
    """
    tenors = ["1W", "1M", "2M", "3M", "6M", "9M", "1Y"]
    metrics = ["10D_BF", "25D_RR", "ATM", "25D_RR", "10D_BF"]
    metric_names = ["10P", "25P", "ATM", "25C", "10C"]

    rows = []
    for t in tenors:
        row = {"tenor": t}
        for m, label in zip(metrics, metric_names):
            ch = vol_change(pair, t, m, days_ago)
            row[label] = round(ch["abs_change"], 2) if ch is not None else None
        rows.append(row)

    return pd.DataFrame(rows).set_index("tenor")


def vol_regime_detect(pair: str, short_window: int = 20,
                      long_window: int = 60) -> dict:
    """
    Detect the current FX-specific vol regime for a currency pair.

    FX-calibrated thresholds (annualised ATM IV):
        LOW < 6%, NORMAL 6-10%, ELEVATED 10-14%, HIGH 14-20%, CRISIS > 20%

    Returns dict with regime, color, description, and supporting metrics.
    """
    rv_hist = get_fx_realized_vol(pair, window=short_window, days=long_window + 50)
    if rv_hist is None or len(rv_hist) < long_window:
        return None

    rv_vals = rv_hist.values if hasattr(rv_hist, 'values') else np.array(rv_hist, dtype=float)
    rv_short = np.mean(rv_vals[-short_window:])
    rv_long = np.mean(rv_vals[-long_window:])
    ratio = rv_short / max(rv_long, 1e-6)

    # Current ATM IV
    atm_info = vol_percentile(pair, "3M", "ATM")
    if atm_info is None:
        return None
    atm_current = atm_info["current"]

    if atm_current > 20.0:
        regime = "CRISIS"
        color = "#ef4444"
        description = "Crisis-level volatility — extreme dislocations, wide spreads"
    elif atm_current > 14.0:
        regime = "HIGH"
        color = "#f97316"
        description = "High vol regime — significant risk events or repricing"
    elif atm_current > 10.0:
        regime = "ELEVATED"
        color = "#f59e0b"
        description = "Elevated volatility — above-average event risk"
    elif atm_current > 6.0:
        regime = "NORMAL"
        color = "#3b82f6"
        description = "Normal trading range — standard market conditions"
    else:
        regime = "LOW"
        color = "#10b981"
        description = "Low vol regime — carry-friendly, compressed risk premia"

    # Trend detection
    if ratio > 1.15:
        trend = "RISING"
    elif ratio < 0.85:
        trend = "FALLING"
    else:
        trend = "STABLE"

    return {
        "pair": pair,
        "regime": regime,
        "color": color,
        "description": description,
        "atm_iv": float(atm_current),
        "rv_short": float(rv_short),
        "rv_long": float(rv_long),
        "ratio": float(ratio),
        "trend": trend,
        "percentile": float(atm_info["percentile"]),
    }


def vol_regime_history(pair: str, lookback: int = 252) -> pd.DataFrame:
    """
    Time series of vol regimes with colour coding, suitable for charting.
    """
    hist = get_fx_historical_vol(pair, "3M", "ATM", lookback)
    if hist is None or len(hist) < 10:
        return pd.DataFrame()

    records = []
    for i, v in enumerate(hist):
        if v > 20:
            regime, color = "CRISIS", "#ef4444"
        elif v > 14:
            regime, color = "HIGH", "#f97316"
        elif v > 10:
            regime, color = "ELEVATED", "#f59e0b"
        elif v > 6:
            regime, color = "NORMAL", "#3b82f6"
        else:
            regime, color = "LOW", "#10b981"
        records.append({"day": i, "vol": float(v), "regime": regime, "color": color})

    return pd.DataFrame(records)


# =========================================================================
#  Vol Cone & Realised Vol Analysis
# =========================================================================

def vol_cone(pair: str,
             windows: List[int] = None,
             lookback: int = 504) -> pd.DataFrame:
    """
    Realised vol cone using three estimators: close-to-close, Parkinson, Garman-Klass.

    Returns DataFrame with percentile bands (min, 10, 25, 50, 75, 90, max)
    and current level for each window.
    """
    if windows is None:
        windows = [5, 10, 20, 60, 90, 120, 252]

    spot_hist_raw = get_fx_historical_spot(pair, lookback + max(windows) + 10)
    if spot_hist_raw is None or (isinstance(spot_hist_raw, pd.DataFrame) and spot_hist_raw.empty):
        return pd.DataFrame()
    spot_hist = _to_close_array(spot_hist_raw)
    if spot_hist is None or len(spot_hist) < max(windows) + 20:
        return pd.DataFrame()

    log_ret = np.diff(np.log(spot_hist))
    records = []

    for w in windows:
        if len(log_ret) < w + 20:
            continue

        # Close-to-close estimator
        c2c = pd.Series(log_ret).rolling(w).std() * np.sqrt(252) * 100
        c2c_clean = c2c.dropna().values

        # Parkinson estimator (simulate high/low from returns)
        high_proxy = spot_hist[1:] * np.exp(np.abs(log_ret) * 0.6)
        low_proxy = spot_hist[1:] * np.exp(-np.abs(log_ret) * 0.6)
        hl_ratio = np.log(high_proxy / low_proxy)
        parkinson_factor = 1.0 / (4.0 * np.log(2.0))
        parkinson_var = pd.Series(parkinson_factor * hl_ratio ** 2).rolling(w).mean() * 252
        parkinson = np.sqrt(parkinson_var.dropna().values) * 100

        # Garman-Klass estimator
        gk_var = 0.5 * hl_ratio ** 2 - (2 * np.log(2) - 1) * log_ret ** 2
        gk_rv = pd.Series(gk_var).rolling(w).mean() * 252
        gk = np.sqrt(np.abs(gk_rv.dropna().values)) * 100

        if len(c2c_clean) < 5:
            continue

        current_c2c = c2c_clean[-1]
        current_park = parkinson[-1] if len(parkinson) > 0 else current_c2c
        current_gk = gk[-1] if len(gk) > 0 else current_c2c

        records.append({
            "window": w,
            "current_c2c": float(current_c2c),
            "current_parkinson": float(current_park),
            "current_gk": float(current_gk),
            "min": float(np.min(c2c_clean)),
            "p10": float(np.percentile(c2c_clean, 10)),
            "p25": float(np.percentile(c2c_clean, 25)),
            "median": float(np.median(c2c_clean)),
            "p75": float(np.percentile(c2c_clean, 75)),
            "p90": float(np.percentile(c2c_clean, 90)),
            "max": float(np.max(c2c_clean)),
            "percentile_rank": float(percentileofscore(c2c_clean, current_c2c)),
        })

    return pd.DataFrame(records)


@_ttl_memo(ttl_seconds=120)
def iv_rv_spread(pair: str, tenor: str = "3M", rv_window: int = 20,
                 lookback: int = 252) -> pd.DataFrame:
    """
    Time series of (ATM IV - Realised Vol) spread.
    A positive spread means IV is rich relative to RV (vol premium).
    """
    iv_hist = get_fx_historical_vol(pair, tenor, "ATM", lookback)
    if iv_hist is None or len(iv_hist) < 20:
        return pd.DataFrame()

    rv_hist = get_fx_realized_vol(pair, window=rv_window, days=lookback)
    if rv_hist is None or len(rv_hist) < 20:
        return pd.DataFrame()

    # Convert to numpy to avoid DatetimeIndex issues
    iv_vals = iv_hist.values if hasattr(iv_hist, 'values') else np.array(iv_hist, dtype=float)
    rv_vals = rv_hist.values if hasattr(rv_hist, 'values') else np.array(rv_hist, dtype=float)

    n = min(len(iv_vals), len(rv_vals))
    iv_arr = iv_vals[-n:].astype(float)
    rv_arr = rv_vals[-n:].astype(float)
    spread = iv_arr - rv_arr

    df = pd.DataFrame({
        "day": np.arange(n),
        "iv": iv_arr,
        "rv": rv_arr,
        "spread": spread,
        "spread_pct": spread / np.maximum(rv_arr, 1e-6) * 100,
    })
    return df


def iv_rv_percentile(pair: str, tenor: str = "3M", rv_window: int = 20,
                     lookback: int = 252) -> dict:
    """
    Percentile rank of the current IV-RV spread relative to its history.
    """
    df = iv_rv_spread(pair, tenor, rv_window, lookback)
    if df.empty:
        return {"percentile": 50.0, "current_spread": 0.0, "mean_spread": 0.0}

    current = df["spread"].iloc[-1]
    pct = percentileofscore(df["spread"].values, current)

    return {
        "pair": pair,
        "tenor": tenor,
        "rv_window": rv_window,
        "current_spread": float(current),
        "mean_spread": float(df["spread"].mean()),
        "std_spread": float(df["spread"].std()),
        "percentile": float(pct),
        "current_iv": float(df["iv"].iloc[-1]),
        "current_rv": float(df["rv"].iloc[-1]),
        "signal": "IV_RICH" if pct > 75 else ("IV_CHEAP" if pct < 25 else "FAIR"),
    }


def breakeven_vol(pair: str, tenor: str, days_to_expiry: int) -> dict:
    """
    What realised vol is needed for a long ATM straddle to break even.
    Accounts for time decay (theta) vs gamma P&L.
    """
    T = days_to_expiry / 365.0
    spots = get_fx_spots([pair])
    spot = spots.get(pair, {}).get("mid", 1.0)

    rates = get_fx_rates(pair)
    r_dom = rates.get("r_dom", 0.03)
    r_for = rates.get("r_for", 0.02)

    surface = get_fx_vol_surface(pair)
    atm_raw = _extract_atm(surface, tenor)
    if atm_raw is None:
        return None
    atm_vol = atm_raw / 100.0

    # Straddle premium as fraction of spot
    fwd = spot * np.exp((r_dom - r_for) * T)
    d1 = (0.5 * atm_vol ** 2 * T) / (atm_vol * np.sqrt(T))
    straddle_pct = 2 * norm.cdf(d1) - 1  # approximate straddle as % of fwd

    # Breakeven daily move = straddle_premium / days
    # In vol terms: breakeven_vol = atm_vol * sqrt(premium_ratio)
    # More precise: need RV such that gamma P&L > theta cost
    # Gamma P&L per day ~ 0.5 * gamma * S^2 * RV^2 / 252
    # Theta per day = known from ATM vol
    # At breakeven: sum of daily gamma PnL = straddle premium
    # Simplified: breakeven_rv ~ atm_vol * sqrt(1 + straddle_premium / (0.5 * vega * atm_vol))
    # For FX ATM straddle: breakeven_rv ~ atm_vol * (1 - small adjustment)
    vega_val = spot * np.sqrt(T) * norm.pdf(d1)
    theta_val = -0.5 * spot * atm_vol * norm.pdf(d1) / np.sqrt(T) / 365.0

    daily_theta = abs(theta_val)
    daily_gamma_per_vol2 = 0.5 * norm.pdf(d1) / (spot * atm_vol * np.sqrt(T)) * spot ** 2 / 252.0

    breakeven_rv_sq = daily_theta / max(daily_gamma_per_vol2, 1e-12)
    breakeven_rv = np.sqrt(max(breakeven_rv_sq, 0)) * 100

    return {
        "pair": pair,
        "tenor": tenor,
        "days_to_expiry": days_to_expiry,
        "atm_iv": float(atm_vol * 100),
        "breakeven_rv": float(breakeven_rv),
        "iv_rv_cushion": float(atm_vol * 100 - breakeven_rv),
        "daily_theta": float(daily_theta),
        "daily_gamma_pnl_at_iv": float(daily_gamma_per_vol2 * (atm_vol ** 2)),
    }


def theta_gamma_ratio(pair: str, tenor: str) -> dict:
    """
    Gamma earned per unit of theta paid.  Higher = more gamma for the theta cost.
    Useful for comparing vega-neutral structures.
    """
    T = tenor_to_years(tenor)
    spots = get_fx_spots([pair])
    spot = spots.get(pair, {}).get("mid", 1.0)

    surface = get_fx_vol_surface(pair)
    atm_raw = _extract_atm(surface, tenor)
    if atm_raw is None:
        return None
    atm_vol = atm_raw / 100.0

    rates = get_fx_rates(pair)
    r_dom = rates.get("r_dom", 0.03)
    r_for = rates.get("r_for", 0.02)

    d1 = (0.5 * atm_vol ** 2 * T) / (atm_vol * np.sqrt(T))

    gamma_val = norm.pdf(d1) / (spot * atm_vol * np.sqrt(T))
    theta_val = -0.5 * spot * atm_vol * norm.pdf(d1) / np.sqrt(T) / 365.0

    ratio = gamma_val / max(abs(theta_val), 1e-12)

    return {
        "pair": pair,
        "tenor": tenor,
        "gamma": float(gamma_val),
        "theta": float(theta_val),
        "gamma_theta_ratio": float(ratio),
        "atm_vol": float(atm_vol * 100),
    }


def vol_carry(pair: str, tenor: str) -> dict:
    """
    Daily vol carry: theta earned per day if vol stays flat.
    Negative carry = cost of holding a long vol position.
    """
    T = tenor_to_years(tenor)
    spots = get_fx_spots([pair])
    spot = spots.get(pair, {}).get("mid", 1.0)

    surface = get_fx_vol_surface(pair)
    atm_raw = _extract_atm(surface, tenor)
    if atm_raw is None:
        return None
    atm_vol = atm_raw / 100.0

    d1 = (0.5 * atm_vol ** 2 * T) / (atm_vol * np.sqrt(T))
    daily_theta = -0.5 * spot * atm_vol * norm.pdf(d1) / np.sqrt(T) / 365.0
    annualised_carry = daily_theta * 365.0

    return {
        "pair": pair,
        "tenor": tenor,
        "daily_theta": float(daily_theta),
        "annualised_carry": float(annualised_carry),
        "carry_as_pct_of_spot": float(annualised_carry / max(spot, 1e-6) * 100),
        "atm_vol": float(atm_vol * 100),
    }


# =========================================================================
#  Forward Volatility
# =========================================================================

def forward_vol(pair: str, T1: str, T2: str) -> dict:
    """
    Implied forward vol for the period [T1, T2] via variance interpolation.

    forward_var = (var_T2 * t2 - var_T1 * t1) / (t2 - t1)
    forward_vol = sqrt(forward_var)
    """
    t1 = tenor_to_years(T1)
    t2 = tenor_to_years(T2)

    if t2 <= t1:
        return {"error": "T2 must be after T1", "forward_vol": 0.0}

    surface = get_fx_vol_surface(pair)
    v1_raw = _extract_atm(surface, T1)
    v2_raw = _extract_atm(surface, T2)
    if v1_raw is None or v2_raw is None:
        return None
    v1 = v1_raw / 100.0
    v2 = v2_raw / 100.0

    var1 = v1 ** 2 * t1
    var2 = v2 ** 2 * t2
    fwd_var = (var2 - var1) / (t2 - t1)

    if fwd_var < 0:
        fwd_var = 0.0
        logger.warning(f"Negative forward variance for {pair} [{T1},{T2}], floored to 0")

    fwd_vol = np.sqrt(fwd_var) * 100

    return {
        "pair": pair,
        "start_tenor": T1,
        "end_tenor": T2,
        "spot_vol_T1": float(v1 * 100),
        "spot_vol_T2": float(v2 * 100),
        "forward_vol": float(fwd_vol),
        "t1_years": float(t1),
        "t2_years": float(t2),
        "variance_T1": float(var1),
        "variance_T2": float(var2),
    }


def forward_vol_curve(pair: str, start_tenor: str = "1M") -> pd.DataFrame:
    """
    Forward vol curve starting from start_tenor out to 2Y.
    Each point is the forward vol from start_tenor to the end tenor.
    """
    end_tenors = ["2M", "3M", "6M", "9M", "1Y", "18M", "2Y"]
    t_start = tenor_to_years(start_tenor)

    records = []
    for end_t in end_tenors:
        t_end = tenor_to_years(end_t)
        if t_end <= t_start:
            continue
        fv = forward_vol(pair, start_tenor, end_t)
        if fv is None:
            continue
        records.append({
            "end_tenor": end_t,
            "end_years": float(t_end),
            "spot_vol": float(fv["spot_vol_T2"]),
            "forward_vol": float(fv["forward_vol"]),
        })

    return pd.DataFrame(records)


def forward_vol_surface(pair: str) -> pd.DataFrame:
    """
    2-D forward vol surface: start_tenor x end_tenor grid of forward vols.
    """
    tenors = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"]
    n = len(tenors)
    data = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i + 1, n):
            fv = forward_vol(pair, tenors[i], tenors[j])
            if fv is not None:
                data[i, j] = fv["forward_vol"]

    df = pd.DataFrame(data, index=tenors, columns=tenors)
    return df


# =========================================================================
#  Smile Analytics
# =========================================================================

def smile_skewness(pair: str, tenor: str) -> dict:
    """
    Smile skew from the 25-delta risk reversal.
    RR > 0 => calls richer => positive skew (upside risk priced higher).
    """
    surface = get_fx_vol_surface(pair)
    rr25 = _extract_metric(surface, tenor, "25D_RR")
    atm = _extract_atm(surface, tenor)

    if rr25 is None or atm is None:
        return None

    normalised = rr25 / max(atm, 1e-6)

    return {
        "pair": pair,
        "tenor": tenor,
        "rr_25d": float(rr25),
        "atm": float(atm),
        "normalised_skew": float(normalised),
        "direction": "CALL_RICH" if rr25 > 0 else "PUT_RICH",
    }


def smile_kurtosis(pair: str, tenor: str) -> dict:
    """
    Smile kurtosis from the 25-delta butterfly.
    Higher BF => fatter tails, more wing demand.
    """
    surface = get_fx_vol_surface(pair)
    bf25 = _extract_metric(surface, tenor, "25D_BF")
    atm = _extract_atm(surface, tenor)

    if bf25 is None or atm is None:
        return None

    normalised = bf25 / max(atm, 1e-6)

    return {
        "pair": pair,
        "tenor": tenor,
        "bf_25d": float(bf25),
        "atm": float(atm),
        "normalised_kurtosis": float(normalised),
        "tail_assessment": "FAT" if normalised > 0.10 else ("THIN" if normalised < 0.03 else "NORMAL"),
    }


def wing_richness(pair: str, tenor: str) -> dict:
    """
    Wing richness indicator: 10D BF / 25D BF.
    Higher ratio => far wings are disproportionately expensive.
    """
    surface = get_fx_vol_surface(pair)
    bf10 = _extract_metric(surface, tenor, "10D_BF")
    bf25 = _extract_metric(surface, tenor, "25D_BF")

    if bf10 is None or bf25 is None:
        return None

    ratio = bf10 / max(bf25, 1e-6)

    return {
        "pair": pair,
        "tenor": tenor,
        "bf_10d": float(bf10),
        "bf_25d": float(bf25),
        "wing_ratio": float(ratio),
        "assessment": "EXPENSIVE_WINGS" if ratio > 3.5 else ("CHEAP_WINGS" if ratio < 2.0 else "NORMAL"),
    }


def smile_asymmetry_index(pair: str, tenor: str) -> dict:
    """
    Smile asymmetry: (|put wing vol - ATM| - |call wing vol - ATM|) / ATM.
    Positive => puts are richer relative to calls.
    """
    surface = get_fx_vol_surface(pair)
    atm = _extract_atm(surface, tenor)
    rr25 = _extract_metric(surface, tenor, "25D_RR")
    bf25 = _extract_metric(surface, tenor, "25D_BF")

    if atm is None or rr25 is None or bf25 is None:
        return None

    # From BF and RR conventions:
    # vol_25c = ATM + BF + 0.5 * RR
    # vol_25p = ATM + BF - 0.5 * RR
    vol_25c = atm + bf25 + 0.5 * rr25
    vol_25p = atm + bf25 - 0.5 * rr25

    put_wing = abs(vol_25p - atm)
    call_wing = abs(vol_25c - atm)
    asymmetry = (put_wing - call_wing) / max(atm, 1e-6)

    return {
        "pair": pair,
        "tenor": tenor,
        "vol_25p": float(vol_25p),
        "vol_25c": float(vol_25c),
        "atm": float(atm),
        "put_wing_spread": float(put_wing),
        "call_wing_spread": float(call_wing),
        "asymmetry_index": float(asymmetry),
    }


def smile_implied_pdf(pair: str, tenor: str,
                      n_points: int = 200) -> pd.DataFrame:
    """
    Risk-neutral probability density via Breeden-Litzenberger.

    Numerically differentiates call prices twice w.r.t. strike to recover
    the implied PDF: f(K) = e^{rT} * d^2C/dK^2
    """
    T = tenor_to_years(tenor)
    spots = get_fx_spots([pair])
    spot = spots.get(pair, {}).get("mid", 1.0)

    rates = get_fx_rates(pair)
    r_dom = rates.get("r_dom", 0.03)
    r_for = rates.get("r_for", 0.02)

    surface = get_fx_vol_surface(pair)
    atm_raw = _extract_atm(surface, tenor)
    rr25_raw = _extract_metric(surface, tenor, "25D_RR")
    bf25_raw = _extract_metric(surface, tenor, "25D_BF")

    if atm_raw is None or rr25_raw is None or bf25_raw is None:
        return pd.DataFrame()

    atm = atm_raw / 100.0
    rr25 = rr25_raw / 100.0
    bf25 = bf25_raw / 100.0

    fwd = spot * np.exp((r_dom - r_for) * T)
    k_min = fwd * np.exp(-4 * atm * np.sqrt(T))
    k_max = fwd * np.exp(4 * atm * np.sqrt(T))
    strikes = np.linspace(k_min, k_max, n_points)
    dk = strikes[1] - strikes[0]

    # Build smile via interpolation
    vol_25p = atm + bf25 - 0.5 * rr25
    vol_25c = atm + bf25 + 0.5 * rr25

    # Parametric smile: quadratic in log-moneyness
    log_m_25 = np.log(fwd / (fwd * 0.96))  # approximate 25D strike shift
    a = bf25 / max(log_m_25 ** 2, 1e-8)
    b = rr25 / max(2 * log_m_25, 1e-8)

    def smile_vol(K):
        lm = np.log(fwd / K)
        return atm + b * lm + a * lm ** 2

    # Compute call prices
    call_prices = np.zeros(n_points)
    for i, K in enumerate(strikes):
        sv = smile_vol(K)
        sv = max(sv, 0.01)
        d1 = (np.log(fwd / K) + 0.5 * sv ** 2 * T) / (sv * np.sqrt(T))
        d2 = d1 - sv * np.sqrt(T)
        call_prices[i] = np.exp(-r_dom * T) * (fwd * norm.cdf(d1) - K * norm.cdf(d2))

    # Second derivative via finite differences
    pdf = np.zeros(n_points)
    pdf[1:-1] = np.exp(r_dom * T) * (call_prices[2:] - 2 * call_prices[1:-1] + call_prices[:-2]) / (dk ** 2)
    pdf = np.maximum(pdf, 0)

    # Normalise to integrate to 1
    total = np.trapezoid(pdf, strikes) if hasattr(np, 'trapezoid') else np.trapz(pdf, strikes)
    if total > 0:
        pdf = pdf / total

    return pd.DataFrame({
        "strike": strikes,
        "pdf": pdf,
        "log_moneyness": np.log(strikes / fwd),
    })


def smile_implied_cdf(pair: str, tenor: str,
                      n_points: int = 200) -> pd.DataFrame:
    """
    Risk-neutral cumulative distribution function derived from the implied PDF.
    """
    pdf_df = smile_implied_pdf(pair, tenor, n_points)
    if pdf_df.empty:
        return pd.DataFrame()
    strikes = pdf_df["strike"].values
    pdf = pdf_df["pdf"].values

    cdf = np.cumsum(pdf)
    dk = strikes[1] - strikes[0] if len(strikes) > 1 else 1.0
    cdf = cdf * dk
    cdf = np.clip(cdf, 0, 1)

    return pd.DataFrame({
        "strike": strikes,
        "cdf": cdf,
        "log_moneyness": pdf_df["log_moneyness"].values,
    })


def tail_probabilities(pair: str, tenor: str,
                       moves: List[float] = None) -> pd.DataFrame:
    """
    Probability of spot moving by given percentages from the implied distribution.
    """
    if moves is None:
        moves = [0.01, 0.02, 0.03, 0.05, 0.10]

    cdf_df = smile_implied_cdf(pair, tenor)
    if cdf_df.empty:
        return pd.DataFrame()
    spots = get_fx_spots([pair])
    spot = spots.get(pair, {}).get("mid", 1.0)

    strikes = cdf_df["strike"].values
    cdf_vals = cdf_df["cdf"].values

    records = []
    for m in moves:
        k_up = spot * (1 + m)
        k_down = spot * (1 - m)

        # Interpolate CDF at those strikes
        prob_below_down = float(np.interp(k_down, strikes, cdf_vals))
        prob_below_up = float(np.interp(k_up, strikes, cdf_vals))

        prob_up = 1 - prob_below_up
        prob_down = prob_below_down

        records.append({
            "move_pct": m * 100,
            "prob_up": round(prob_up * 100, 2),
            "prob_down": round(prob_down * 100, 2),
            "prob_either": round((prob_up + prob_down) * 100, 2),
            "strike_up": round(k_up, 5),
            "strike_down": round(k_down, 5),
        })

    return pd.DataFrame(records)


def smile_pca(pair: str, lookback: int = 252) -> dict:
    """
    PCA decomposition of smile moves into level, skew, curvature, and wings.
    Uses historical daily changes in ATM, 25D RR, 25D BF, 10D RR, 10D BF.
    """
    metrics = ["ATM", "25D_RR", "25D_BF", "10D_RR", "10D_BF"]
    data = {}
    for m in metrics:
        h = get_fx_historical_vol(pair, "3M", m, lookback)
        if h is None or len(h) < 20:
            return None
        data[m] = h

    n = min(len(v) for v in data.values())
    mat = np.column_stack([data[m][-n:] for m in metrics])
    changes = np.diff(mat, axis=0)

    # Demean
    changes_dm = changes - changes.mean(axis=0)

    # Covariance & eigen decomposition
    cov = np.cov(changes_dm.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    explained = eigenvalues / max(total_var, 1e-12) * 100

    pc_names = ["Level", "Skew", "Curvature", "Wings", "Residual"]

    components = {}
    for i in range(min(5, len(eigenvalues))):
        name = pc_names[i] if i < len(pc_names) else f"PC{i+1}"
        components[name] = {
            "eigenvalue": float(eigenvalues[i]),
            "explained_pct": float(explained[i]),
            "loadings": {m: float(eigenvectors[j, i]) for j, m in enumerate(metrics)},
        }

    return {
        "pair": pair,
        "lookback": lookback,
        "components": components,
        "total_variance": float(total_var),
        "cumulative_explained": [float(np.sum(explained[:i+1])) for i in range(min(5, len(explained)))],
    }


def sticky_delta_monitor(pair: str, tenor: str,
                         lookback: int = 60) -> dict:
    """
    Assess whether the smile is moving sticky-delta or sticky-strike.

    Method: Regress daily ATM vol change on daily spot return.
    - Sticky-delta: beta ~ 0 (vol doesn't move with spot)
    - Sticky-strike: beta ~ skew (vol moves as spot changes along smile)
    """
    iv_hist = get_fx_historical_vol(pair, tenor, "ATM", lookback + 5)
    if iv_hist is None or len(iv_hist) < lookback:
        return None

    spot_hist_raw = get_fx_historical_spot(pair, lookback + 5)
    spot_hist = _to_close_array(spot_hist_raw) if spot_hist_raw is not None else None
    if spot_hist is None or len(spot_hist) < lookback:
        return None

    n = min(len(iv_hist), len(spot_hist)) - 1
    iv_changes = np.diff(iv_hist[-n-1:])
    spot_returns = np.diff(np.log(spot_hist[-n-1:]))

    slope, intercept, r_value, p_value, std_err = linregress(spot_returns, iv_changes)

    # Skew for reference
    skew_info = smile_skewness(pair, tenor)
    if skew_info is None:
        return None
    rr = skew_info["rr_25d"]

    # Sticky-delta => slope near 0
    # Sticky-strike => slope near -skew_slope
    if abs(slope) < abs(rr) * 0.3:
        regime = "STICKY_DELTA"
    elif abs(slope) > abs(rr) * 0.7:
        regime = "STICKY_STRIKE"
    else:
        regime = "MIXED"

    return {
        "pair": pair,
        "tenor": tenor,
        "regime": regime,
        "beta": float(slope),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "reference_skew": float(rr),
        "lookback": lookback,
    }


# =========================================================================
#  Relative Value Engine
# =========================================================================

def cross_pair_vol_spread(pair_a: str, pair_b: str, tenor: str = "3M",
                          lookback: int = 252) -> dict:
    """
    Time series of ATM vol spread between two pairs with z-score and signal.
    """
    hist_a = get_fx_historical_vol(pair_a, tenor, "ATM", lookback)
    hist_b = get_fx_historical_vol(pair_b, tenor, "ATM", lookback)
    if hist_a is None or len(hist_a) < 20:
        return None
    if hist_b is None or len(hist_b) < 20:
        return None

    n = min(len(hist_a), len(hist_b))
    a = hist_a[-n:]
    b = hist_b[-n:]
    spread = a - b

    mu = np.mean(spread)
    sigma = np.std(spread)
    current = spread[-1]
    z = (current - mu) / max(sigma, 1e-6)
    pct = percentileofscore(spread, current)

    if z > 1.5:
        signal = f"SELL_{pair_a}_VOL_BUY_{pair_b}_VOL"
    elif z < -1.5:
        signal = f"BUY_{pair_a}_VOL_SELL_{pair_b}_VOL"
    else:
        signal = "NEUTRAL"

    return {
        "pair_a": pair_a,
        "pair_b": pair_b,
        "tenor": tenor,
        "spread_ts": spread.tolist(),
        "mean": float(mu),
        "std": float(sigma),
        "current": float(current),
        "zscore": float(z),
        "percentile": float(pct),
        "signal": signal,
    }


def cross_pair_rr_spread(pair_a: str, pair_b: str, tenor: str = "3M",
                          lookback: int = 252) -> dict:
    """Spread of 25D risk reversals between two pairs."""
    return _cross_pair_metric_spread(pair_a, pair_b, tenor, "25D_RR", lookback)


def cross_pair_bf_spread(pair_a: str, pair_b: str, tenor: str = "3M",
                          lookback: int = 252) -> dict:
    """Spread of 25D butterflies between two pairs."""
    return _cross_pair_metric_spread(pair_a, pair_b, tenor, "25D_BF", lookback)


def cross_pair_term_spread(pair_a: str, pair_b: str,
                           long_tenor: str = "1Y", short_tenor: str = "1M",
                           lookback: int = 252) -> dict:
    """
    Term structure spread comparison between two pairs.
    Computes (long - short) ATM vol for each pair, then takes the cross-pair diff.
    """
    al = get_fx_historical_vol(pair_a, long_tenor, "ATM", lookback)
    a_s = get_fx_historical_vol(pair_a, short_tenor, "ATM", lookback)
    bl = get_fx_historical_vol(pair_b, long_tenor, "ATM", lookback)
    bs = get_fx_historical_vol(pair_b, short_tenor, "ATM", lookback)

    for arr in [al, a_s, bl, bs]:
        if arr is None or len(arr) < 20:
            return None

    n = min(len(al), len(a_s), len(bl), len(bs))
    term_a = al[-n:] - a_s[-n:]
    term_b = bl[-n:] - bs[-n:]
    spread = term_a - term_b

    mu = np.mean(spread)
    sigma = np.std(spread)
    current = spread[-1]
    z = (current - mu) / max(sigma, 1e-6)

    return {
        "pair_a": pair_a,
        "pair_b": pair_b,
        "long_tenor": long_tenor,
        "short_tenor": short_tenor,
        "spread_ts": spread.tolist(),
        "mean": float(mu),
        "std": float(sigma),
        "current": float(current),
        "zscore": float(z),
        "percentile": float(percentileofscore(spread, current)),
    }


def vol_beta(pair_a: str, pair_b: str, tenor: str = "3M",
             lookback: int = 252) -> dict:
    """
    Regression beta of pair_a vol changes on pair_b vol changes.
    Useful for hedge ratios and relative value.
    """
    ha = get_fx_historical_vol(pair_a, tenor, "ATM", lookback)
    hb = get_fx_historical_vol(pair_b, tenor, "ATM", lookback)
    if ha is None or len(ha) < 20:
        return None
    if hb is None or len(hb) < 20:
        return None

    n = min(len(ha), len(hb)) - 1
    da = np.diff(ha[-n-1:])
    db = np.diff(hb[-n-1:])

    slope, intercept, r_value, p_value, std_err = linregress(db, da)
    residuals = da - (slope * db + intercept)

    return {
        "pair_a": pair_a,
        "pair_b": pair_b,
        "tenor": tenor,
        "beta": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "residual_std": float(np.std(residuals)),
        "current_residual": float(residuals[-1]),
        "residual_zscore": float(residuals[-1] / max(np.std(residuals), 1e-6)),
    }


def rv_scanner(pairs: List[str] = None,
               tenors: List[str] = None,
               lookback: int = 252) -> pd.DataFrame:
    """
    Scan all pair/tenor combinations for relative value signals.
    Returns DataFrame sorted by absolute z-score.
    """
    if pairs is None:
        pairs = list(FX_PAIRS.keys())[:8]
    if tenors is None:
        tenors = ["1M", "3M", "1Y"]

    records = []
    for p in pairs:
        for t in tenors:
            info = vol_zscore(p, t, "ATM", lookback)
            pct_info = vol_percentile(p, t, "ATM", lookback)
            iv_rv = iv_rv_percentile(p, t, lookback=lookback)

            if info is None or pct_info is None:
                continue

            records.append({
                "pair": p,
                "tenor": t,
                "atm_vol": info["current"],
                "zscore": info["zscore"],
                "percentile": pct_info["percentile"],
                "iv_rv_spread": iv_rv.get("current_spread", 0.0) if iv_rv else 0.0,
                "iv_rv_pct": iv_rv.get("percentile", 50.0) if iv_rv else 50.0,
                "signal": info["interpretation"],
            })

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["abs_zscore"] = df["zscore"].abs()
    df = df.sort_values("abs_zscore", ascending=False).drop(columns=["abs_zscore"])
    return df.reset_index(drop=True)


def rv_signal_composite(pair: str, lookback: int = 252) -> dict:
    """
    Weighted composite relative value score from multiple signals.
    Score range: -100 (extremely cheap) to +100 (extremely rich).
    """
    # ATM IV z-score (weight 30%)
    atm_z_info = vol_zscore(pair, "3M", "ATM", lookback)
    if atm_z_info is None:
        return None
    atm_z = atm_z_info["zscore"]

    # IV-RV spread percentile (weight 25%)
    ivrv = iv_rv_percentile(pair, "3M", lookback=lookback)
    ivrv_score = ((ivrv["percentile"] - 50) / 50) if ivrv else 0.0

    # Term structure slope z-score (weight 15%)
    ts_1m_info = vol_zscore(pair, "1M", "ATM", lookback)
    ts_1y_info = vol_zscore(pair, "1Y", "ATM", lookback)
    ts_1m = ts_1m_info["zscore"] if ts_1m_info is not None else 0.0
    ts_1y = ts_1y_info["zscore"] if ts_1y_info is not None else 0.0
    ts_z = ts_1m - ts_1y  # front rich => positive

    # BF z-score (weight 15%)
    bf_z_info = vol_zscore(pair, "3M", "25D_BF", lookback)
    bf_z = bf_z_info["zscore"] if bf_z_info is not None else 0.0

    # Regime adjustment (weight 15%)
    regime = vol_regime_detect(pair)
    regime_map = {"LOW": -1.0, "NORMAL": 0.0, "ELEVATED": 0.5, "HIGH": 1.0, "CRISIS": 1.5}
    regime_score = regime_map.get(regime["regime"], 0.0) if regime is not None else 0.0

    composite = (
        0.30 * np.clip(atm_z, -3, 3) / 3.0 +
        0.25 * np.clip(ivrv_score, -1, 1) +
        0.15 * np.clip(ts_z, -3, 3) / 3.0 +
        0.15 * np.clip(bf_z, -3, 3) / 3.0 +
        0.15 * np.clip(regime_score, -1.5, 1.5) / 1.5
    ) * 100

    if composite > 30:
        recommendation = "SELL_VOL"
    elif composite < -30:
        recommendation = "BUY_VOL"
    else:
        recommendation = "NEUTRAL"

    return {
        "pair": pair,
        "composite_score": float(np.clip(composite, -100, 100)),
        "recommendation": recommendation,
        "components": {
            "atm_zscore": float(atm_z),
            "ivrv_score": float(ivrv_score),
            "term_structure_z": float(ts_z),
            "bf_zscore": float(bf_z),
            "regime_score": float(regime_score),
        },
        "weights": {"atm": 0.30, "ivrv": 0.25, "term": 0.15, "bf": 0.15, "regime": 0.15},
    }


def carry_adjusted_rv(pair: str, tenor: str = "3M",
                      lookback: int = 252) -> dict:
    """
    Relative value adjusted for carry (theta cost of holding vol position).
    Cheap vol with negative carry may not be as attractive.
    """
    z = vol_zscore(pair, tenor, "ATM", lookback)
    if z is None:
        return None
    carry = vol_carry(pair, tenor)
    if carry is None:
        return None

    # Carry adjustment: penalise if buying vol but carry is negative (expensive)
    carry_adj = carry["carry_as_pct_of_spot"]
    adjusted_z = z["zscore"] + carry_adj * 0.5  # shift z-score by carry impact

    return {
        "pair": pair,
        "tenor": tenor,
        "raw_zscore": float(z["zscore"]),
        "carry_cost_bps": float(carry_adj),
        "adjusted_zscore": float(adjusted_z),
        "daily_theta": float(carry["daily_theta"]),
        "lookback": lookback,
    }


def implied_correlation(pair_a: str, pair_b: str, cross_pair: str,
                        tenor: str = "3M") -> dict:
    """
    Implied correlation from the vol triangle:
    sigma_cross^2 = sigma_a^2 + sigma_b^2 - 2 * rho * sigma_a * sigma_b

    Solved: rho = (sigma_a^2 + sigma_b^2 - sigma_cross^2) / (2 * sigma_a * sigma_b)
    """
    surface_a = get_fx_vol_surface(pair_a)
    surface_b = get_fx_vol_surface(pair_b)
    surface_c = get_fx_vol_surface(cross_pair)

    va_raw = _extract_atm(surface_a, tenor)
    vb_raw = _extract_atm(surface_b, tenor)
    vc_raw = _extract_atm(surface_c, tenor)

    if va_raw is None or vb_raw is None or vc_raw is None:
        return None

    va = va_raw / 100.0
    vb = vb_raw / 100.0
    vc = vc_raw / 100.0

    denom = 2 * va * vb
    if denom < 1e-10:
        rho = 0.0
    else:
        rho = (va ** 2 + vb ** 2 - vc ** 2) / denom

    rho = float(np.clip(rho, -1.0, 1.0))

    return {
        "pair_a": pair_a,
        "pair_b": pair_b,
        "cross_pair": cross_pair,
        "tenor": tenor,
        "implied_corr": rho,
        "vol_a": float(va * 100),
        "vol_b": float(vb * 100),
        "vol_cross": float(vc * 100),
    }


def correlation_richness(pair_a: str, pair_b: str, cross_pair: str,
                         tenor: str = "3M") -> dict:
    """
    Gap between implied correlation (from vol triangle) and realised correlation.
    Positive gap => implied corr is higher than realised (corr is expensive).
    """
    impl = implied_correlation(pair_a, pair_b, cross_pair, tenor)
    if impl is None:
        return None
    implied_rho = impl["implied_corr"]

    realized_rho = get_fx_correlation(pair_a, pair_b, window=60)
    if realized_rho is None:
        return None

    gap = implied_rho - realized_rho

    return {
        "pair_a": pair_a,
        "pair_b": pair_b,
        "cross_pair": cross_pair,
        "tenor": tenor,
        "implied_corr": float(implied_rho),
        "realized_corr": float(realized_rho),
        "gap": float(gap),
        "signal": "CORR_RICH" if gap > 0.10 else ("CORR_CHEAP" if gap < -0.10 else "FAIR"),
    }


# =========================================================================
#  Correlation Analytics
# =========================================================================

@_ttl_memo(ttl_seconds=180)
def spot_correlation_matrix(pairs: List[str] = None,
                            window: int = 60) -> pd.DataFrame:
    """
    Rolling correlation matrix of spot log-returns for the given FX pairs.
    """
    if pairs is None:
        pairs = list(FX_PAIRS.keys())[:6]

    returns = {}
    for p in pairs:
        hist = _to_close_array(get_fx_historical_spot(p, window + 10))
        if hist is None or len(hist) < window:
            continue
        ret = np.diff(np.log(hist[-(window + 1):]))
        returns[p] = ret[:window]

    if not returns:
        return pd.DataFrame()
    df = pd.DataFrame(returns)
    return df.corr()


def vol_correlation_matrix(pairs: List[str] = None, tenor: str = "3M",
                           window: int = 60) -> pd.DataFrame:
    """
    Correlation matrix of daily ATM vol changes across pairs.
    """
    if pairs is None:
        pairs = list(FX_PAIRS.keys())[:6]

    changes = {}
    for p in pairs:
        hist = get_fx_historical_vol(p, tenor, "ATM", window + 10)
        if hist is None or len(hist) < window:
            continue
        ch = np.diff(hist[-(window + 1):])
        changes[p] = ch[:window]

    if not changes:
        return pd.DataFrame()
    df = pd.DataFrame(changes)
    return df.corr()


def spot_vol_correlation(pair: str, window: int = 60) -> dict:
    """
    Correlation between spot returns and ATM vol changes.
    In FX this is the leverage effect (often weaker than equities).
    """
    spot_hist = _to_close_array(get_fx_historical_spot(pair, window + 10))
    if spot_hist is None or len(spot_hist) < window:
        return None

    vol_hist = get_fx_historical_vol(pair, "3M", "ATM", window + 10)
    if vol_hist is None or len(vol_hist) < window:
        return None

    n = min(len(spot_hist), len(vol_hist)) - 1
    spot_ret = np.diff(np.log(spot_hist[-n-1:]))
    vol_chg = np.diff(vol_hist[-n-1:])

    m = min(len(spot_ret), len(vol_chg))
    corr = float(np.corrcoef(spot_ret[:m], vol_chg[:m])[0, 1])

    return {
        "pair": pair,
        "spot_vol_corr": corr,
        "window": window,
        "interpretation": "STRONG_LEVERAGE" if abs(corr) > 0.5 else
                         ("MODERATE_LEVERAGE" if abs(corr) > 0.3 else "WEAK_LEVERAGE"),
    }


def correlation_regime(pairs: List[str] = None,
                       lookback: int = 252) -> pd.DataFrame:
    """
    Identify correlation breakdowns by comparing short-term and long-term
    correlation matrices.  Flag pairs where corr has deviated significantly.
    """
    if pairs is None:
        pairs = list(FX_PAIRS.keys())[:6]

    corr_short = spot_correlation_matrix(pairs, window=20)
    corr_long = spot_correlation_matrix(pairs, window=min(lookback, 120))

    records = []
    for i, p1 in enumerate(pairs):
        for j, p2 in enumerate(pairs):
            if j <= i:
                continue
            short_c = corr_short.loc[p1, p2]
            long_c = corr_long.loc[p1, p2]
            diff = short_c - long_c

            if abs(diff) > 0.3:
                status = "BREAKDOWN"
            elif abs(diff) > 0.15:
                status = "SHIFTING"
            else:
                status = "STABLE"

            records.append({
                "pair_1": p1,
                "pair_2": p2,
                "corr_short": float(round(short_c, 3)),
                "corr_long": float(round(long_c, 3)),
                "diff": float(round(diff, 3)),
                "status": status,
            })

    return pd.DataFrame(records)


def correlation_term_structure(pair_a: str, pair_b: str,
                               windows: List[int] = None) -> pd.DataFrame:
    """
    Correlation at different rolling windows (term structure of correlation).
    """
    if windows is None:
        windows = [20, 60, 120, 252]

    max_w = max(windows)
    hist_a = _to_close_array(get_fx_historical_spot(pair_a, max_w + 10))
    hist_b = _to_close_array(get_fx_historical_spot(pair_b, max_w + 10))
    if hist_a is None or len(hist_a) < max_w:
        return pd.DataFrame()
    if hist_b is None or len(hist_b) < max_w:
        return pd.DataFrame()

    n = min(len(hist_a), len(hist_b)) - 1
    ret_a = np.diff(np.log(hist_a[-n-1:]))
    ret_b = np.diff(np.log(hist_b[-n-1:]))

    records = []
    for w in windows:
        if w > len(ret_a) or w > len(ret_b):
            continue
        corr = float(np.corrcoef(ret_a[-w:], ret_b[-w:])[0, 1])
        records.append({"window": w, "correlation": round(corr, 4)})

    return pd.DataFrame(records)


def correlation_cone(pair_a: str, pair_b: str,
                     windows: List[int] = None,
                     lookback: int = 504) -> pd.DataFrame:
    """
    Correlation cone: percentile bands for rolling correlation at each window.
    """
    if windows is None:
        windows = [20, 60, 120, 252]

    hist_a = _to_close_array(get_fx_historical_spot(pair_a, lookback + max(windows) + 10))
    hist_b = _to_close_array(get_fx_historical_spot(pair_b, lookback + max(windows) + 10))
    if hist_a is None or len(hist_a) < lookback:
        return pd.DataFrame()
    if hist_b is None or len(hist_b) < lookback:
        return pd.DataFrame()

    n = min(len(hist_a), len(hist_b)) - 1
    ret_a = np.diff(np.log(hist_a[-n-1:]))
    ret_b = np.diff(np.log(hist_b[-n-1:]))

    records = []
    for w in windows:
        if len(ret_a) < w + 20:
            continue
        # Compute rolling correlation series
        corr_series = pd.Series(ret_a).rolling(w).corr(pd.Series(ret_b)).dropna().values

        if len(corr_series) < 5:
            continue

        current = corr_series[-1]
        records.append({
            "window": w,
            "current": float(current),
            "min": float(np.nanmin(corr_series)),
            "p10": float(np.nanpercentile(corr_series, 10)),
            "p25": float(np.nanpercentile(corr_series, 25)),
            "median": float(np.nanmedian(corr_series)),
            "p75": float(np.nanpercentile(corr_series, 75)),
            "p90": float(np.nanpercentile(corr_series, 90)),
            "max": float(np.nanmax(corr_series)),
            "percentile_rank": float(percentileofscore(corr_series, current)),
        })

    return pd.DataFrame(records)


# =========================================================================
#  Carry & Rates
# =========================================================================

def carry_table(pairs: List[str] = None) -> pd.DataFrame:
    """
    Carry table: rate differentials, fwd points, and annualised carry for all pairs.
    """
    if pairs is None:
        pairs = list(FX_PAIRS.keys())[:8]

    records = []
    for p in pairs:
        spots = get_fx_spots([p])
        spot = spots.get(p, {}).get("mid", 1.0)

        rates = get_fx_rates(p)
        r_dom = rates.get("r_dom", 0.03)
        r_for = rates.get("r_for", 0.02)

        diff = r_dom - r_for

        # 3M forward points
        T = 0.25
        fwd = spot * np.exp(diff * T)
        fwd_pts = (fwd - spot) * 10000  # in pips

        # Annualised carry in bps
        ann_carry = diff * 10000

        records.append({
            "pair": p,
            "spot": round(spot, 5),
            "r_dom": round(r_dom * 100, 2),
            "r_for": round(r_for * 100, 2),
            "rate_diff_bps": round(diff * 10000, 1),
            "fwd_3m": round(fwd, 5),
            "fwd_pts_3m": round(fwd_pts, 1),
            "ann_carry_bps": round(ann_carry, 1),
        })

    return pd.DataFrame(records)


def carry_per_vol(pairs: List[str] = None) -> pd.DataFrame:
    """
    Risk-adjusted carry: carry / ATM vol ratio for each pair.
    Higher = better risk-adjusted carry.
    """
    if pairs is None:
        pairs = list(FX_PAIRS.keys())[:8]

    records = []
    for p in pairs:
        rates = get_fx_rates(p)
        r_dom = rates.get("r_dom", 0.03)
        r_for = rates.get("r_for", 0.02)
        diff = abs(r_dom - r_for)

        surface = get_fx_vol_surface(p)
        atm_raw = _extract_atm(surface, "3M")
        if atm_raw is None:
            continue
        atm = atm_raw / 100.0

        ratio = diff / max(atm, 1e-6)
        sharpe_proxy = ratio * np.sqrt(4)  # annualise the 3M ratio

        records.append({
            "pair": p,
            "carry_bps": round(diff * 10000, 1),
            "atm_3m": round(atm * 100, 2),
            "carry_per_vol": round(ratio, 4),
            "sharpe_proxy": round(sharpe_proxy, 2),
            "rank_signal": "ATTRACTIVE" if sharpe_proxy > 0.5 else
                          ("MODERATE" if sharpe_proxy > 0.2 else "UNATTRACTIVE"),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("sharpe_proxy", ascending=False).reset_index(drop=True)
    return df


def carry_momentum(pair: str, lookback: int = 60) -> dict:
    """
    Is carry improving or deteriorating?
    Tracks the change in rate differential over the lookback period.

    Note: Without a Bloomberg historical rate differential series, this
    function can only return the current snapshot. No synthetic data is
    generated to fake a history.
    """
    rates = get_fx_rates(pair)
    r_dom = rates.get("r_dom", 0.03)
    r_for = rates.get("r_for", 0.02)
    current_diff = r_dom - r_for

    # Without real historical rate data we cannot compute momentum.
    # Return the current level with UNKNOWN momentum instead of faking a history.
    return {
        "pair": pair,
        "current_diff_bps": float(current_diff * 10000),
        "change_20d_bps": None,
        "change_60d_bps": None,
        "momentum": "UNKNOWN",
    }


def rate_differential_history(pair: str, lookback: int = 252) -> pd.DataFrame:
    """
    Time series of domestic-foreign rate differential.

    Note: Without a Bloomberg historical rate series, this function
    cannot produce a real time series. Returns an empty DataFrame
    instead of fabricating synthetic data.
    """
    # No real historical rate differential data source is available.
    # Return empty DataFrame so callers know there is no data.
    return pd.DataFrame()


# =========================================================================
#  Positioning & Flow
# =========================================================================

def cftc_positioning_data(pair: str) -> dict:
    """
    CFTC Commitments of Traders positioning data.
    Returns net speculative, commercial, and open interest.
    """
    data = get_cftc_positioning(pair)
    if data is None:
        return None

    return {
        "pair": pair,
        "net_speculative": int(data.get("net_speculative", 0)),
        "net_commercial": int(data.get("net_commercial", 0)),
        "open_interest": int(data.get("open_interest", 0)),
        "spec_long": int(data.get("spec_long", 0)),
        "spec_short": int(data.get("spec_short", 0)),
        "comm_long": int(data.get("comm_long", 0)),
        "comm_short": int(data.get("comm_short", 0)),
        "report_date": data.get("report_date", ""),
    }


def positioning_zscore(pair: str, lookback: int = 156) -> dict:
    """
    Z-score of net speculative positioning vs 3-year weekly history.

    Note: Without real historical CFTC positioning data, this function
    cannot compute a meaningful z-score.  Returns None if no positioning
    data is available.
    """
    current = cftc_positioning_data(pair)
    if current is None:
        return None

    # Without real historical positioning data we cannot compute a z-score.
    # Return the current snapshot with UNKNOWN signal.
    return {
        "pair": pair,
        "net_speculative": int(current["net_speculative"]),
        "mean_3y": None,
        "std_3y": None,
        "zscore": None,
        "lookback_weeks": lookback,
        "signal": "UNKNOWN",
    }


def positioning_extremes(pairs: List[str] = None) -> pd.DataFrame:
    """
    Flag pairs with extreme speculative positioning (|z-score| > 1.5).
    """
    if pairs is None:
        pairs = list(FX_PAIRS.keys())[:8]

    records = []
    for p in pairs:
        z_data = positioning_zscore(p)
        if z_data is None:
            continue
        zscore_val = z_data["zscore"]
        records.append({
            "pair": p,
            "net_speculative": z_data["net_speculative"],
            "zscore": round(zscore_val, 2) if zscore_val is not None else None,
            "signal": z_data["signal"],
            "extreme": abs(zscore_val) > 1.5 if zscore_val is not None else False,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("zscore", key=abs, ascending=False).reset_index(drop=True)
    return df


def positioning_vs_spot(pair: str, lookback: int = 156) -> pd.DataFrame:
    """
    Overlay of positioning and spot for divergence analysis.
    Weekly frequency aligned to CFTC report dates.

    Note: Without real historical CFTC positioning data and sufficient
    spot history, returns an empty DataFrame.
    """
    spot_hist = _to_close_array(get_fx_historical_spot(pair, lookback * 5 + 10))
    if spot_hist is None or len(spot_hist) < lookback:
        return pd.DataFrame()

    current = cftc_positioning_data(pair)
    if current is None:
        return pd.DataFrame()

    # Without real historical positioning time series we cannot build
    # the overlay.  Return empty DataFrame.
    return pd.DataFrame()


# =========================================================================
#  Risk Metrics
# =========================================================================

def historical_var(returns: np.ndarray, confidence: float = 0.95,
                   horizon: int = 1) -> dict:
    """
    Historical simulation Value-at-Risk.
    Scales to horizon using sqrt(T) rule.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 10:
        return {"var": 0.0, "confidence": confidence, "horizon": horizon}

    sorted_ret = np.sort(returns)
    idx = int((1 - confidence) * len(sorted_ret))
    var_1d = -sorted_ret[idx]
    var_horizon = var_1d * np.sqrt(horizon)

    return {
        "var_1d": float(var_1d),
        "var_horizon": float(var_horizon),
        "confidence": confidence,
        "horizon": horizon,
        "n_observations": len(returns),
        "worst_return": float(sorted_ret[0]),
        "best_return": float(sorted_ret[-1]),
    }


def parametric_var(sigma: float, notional: float, confidence: float = 0.95,
                   horizon: int = 1) -> dict:
    """
    Gaussian (parametric) VaR.
    VaR = z * sigma * sqrt(T) * notional
    """
    z = norm.ppf(confidence)
    daily_vol = sigma / np.sqrt(252)
    var_val = z * daily_vol * np.sqrt(horizon) * notional

    return {
        "var": float(var_val),
        "confidence": confidence,
        "horizon": horizon,
        "z_score": float(z),
        "daily_vol": float(daily_vol),
        "sigma": float(sigma),
        "notional": float(notional),
    }


def expected_shortfall(returns: np.ndarray, confidence: float = 0.95,
                       horizon: int = 1) -> dict:
    """
    Conditional VaR (Expected Shortfall / CVaR).
    Average loss in the worst (1-confidence) tail.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 10:
        return {"cvar": 0.0, "var": 0.0, "confidence": confidence}

    sorted_ret = np.sort(returns)
    cutoff = int((1 - confidence) * len(sorted_ret))
    cutoff = max(cutoff, 1)

    var_val = -sorted_ret[cutoff]
    cvar = -np.mean(sorted_ret[:cutoff])
    cvar_horizon = cvar * np.sqrt(horizon)

    return {
        "cvar_1d": float(cvar),
        "cvar_horizon": float(cvar_horizon),
        "var_1d": float(var_val),
        "confidence": confidence,
        "horizon": horizon,
        "n_tail_obs": cutoff,
        "tail_ratio": float(cvar / max(var_val, 1e-10)),
    }


def cornish_fisher_var(returns: np.ndarray, confidence: float = 0.95) -> dict:
    """
    VaR adjusted for skewness and kurtosis using the Cornish-Fisher expansion.
    More accurate than Gaussian VaR for non-normal distributions.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 20:
        return {"cf_var": 0.0, "gaussian_var": 0.0}

    mu = np.mean(returns)
    sigma = np.std(returns)
    s = float(pd.Series(returns).skew())
    k = float(pd.Series(returns).kurtosis())  # excess kurtosis

    z = norm.ppf(confidence)

    # Cornish-Fisher expansion
    z_cf = (z +
            (z ** 2 - 1) * s / 6 +
            (z ** 3 - 3 * z) * k / 24 -
            (2 * z ** 3 - 5 * z) * s ** 2 / 36)

    cf_var = -(mu - z_cf * sigma)
    gaussian_var = -(mu - z * sigma)

    return {
        "cf_var": float(cf_var),
        "gaussian_var": float(gaussian_var),
        "adjustment": float(cf_var - gaussian_var),
        "skewness": float(s),
        "excess_kurtosis": float(k),
        "z_gaussian": float(z),
        "z_cornish_fisher": float(z_cf),
        "confidence": confidence,
    }


def max_drawdown(returns: np.ndarray) -> dict:
    """
    Maximum drawdown from a return series.
    Returns max drawdown percentage, peak/trough indices, and duration.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return {"max_drawdown": 0.0, "peak_idx": 0, "trough_idx": 0}

    # Cumulative wealth
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1

    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(cum[:trough_idx + 1]))
    mdd = float(drawdowns[trough_idx])

    # Recovery index
    recovery_idx = None
    for i in range(trough_idx + 1, len(cum)):
        if cum[i] >= cum[peak_idx]:
            recovery_idx = i
            break

    duration_to_trough = trough_idx - peak_idx
    duration_to_recovery = (recovery_idx - peak_idx) if recovery_idx is not None else None

    return {
        "max_drawdown": float(mdd),
        "max_drawdown_pct": float(mdd * 100),
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "recovery_idx": recovery_idx,
        "duration_to_trough": duration_to_trough,
        "duration_to_recovery": duration_to_recovery,
        "peak_value": float(cum[peak_idx]),
        "trough_value": float(cum[trough_idx]),
    }


def tail_risk_metrics(returns: np.ndarray) -> dict:
    """
    Comprehensive tail risk metrics:
    - Skewness, excess kurtosis
    - Jarque-Bera test for normality
    - Hill tail index estimator
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 20:
        return {
            "skewness": 0.0, "excess_kurtosis": 0.0,
            "jarque_bera": 0.0, "jb_pvalue": 1.0,
            "hill_index": 2.0,
        }

    s = float(pd.Series(returns).skew())
    k = float(pd.Series(returns).kurtosis())

    jb_stat, jb_pvalue = jarque_bera(returns)

    # Hill tail index estimator (for left tail)
    sorted_abs = np.sort(np.abs(returns))[::-1]
    # Use top 10% of observations
    n_tail = max(int(len(sorted_abs) * 0.10), 5)
    threshold = sorted_abs[n_tail - 1]

    if threshold > 0:
        log_exceedances = np.log(sorted_abs[:n_tail] / threshold)
        hill_index = float(n_tail / max(np.sum(log_exceedances), 1e-10))
    else:
        hill_index = 2.0  # default

    # Interpret
    if jb_pvalue < 0.01:
        normality = "REJECTED"
    elif jb_pvalue < 0.05:
        normality = "BORDERLINE"
    else:
        normality = "NOT_REJECTED"

    return {
        "skewness": float(s),
        "excess_kurtosis": float(k),
        "jarque_bera_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
        "normality": normality,
        "hill_tail_index": float(hill_index),
        "tail_assessment": "FAT_TAILS" if hill_index < 3 else "NORMAL_TAILS",
        "n_observations": len(returns),
    }


# =========================================================================
#  Internal Helpers
# =========================================================================

def _cross_pair_metric_spread(pair_a: str, pair_b: str, tenor: str,
                               metric: str, lookback: int) -> dict:
    """Generic cross-pair spread calculation for any vol metric."""
    ha = get_fx_historical_vol(pair_a, tenor, metric, lookback)
    hb = get_fx_historical_vol(pair_b, tenor, metric, lookback)
    if ha is None or len(ha) < 20:
        return None
    if hb is None or len(hb) < 20:
        return None

    n = min(len(ha), len(hb))
    a = ha[-n:]
    b = hb[-n:]
    spread = a - b

    mu = np.mean(spread)
    sigma = np.std(spread)
    current = spread[-1]
    z = (current - mu) / max(sigma, 1e-6)

    return {
        "pair_a": pair_a,
        "pair_b": pair_b,
        "tenor": tenor,
        "metric": metric,
        "spread_ts": spread.tolist(),
        "mean": float(mu),
        "std": float(sigma),
        "current": float(current),
        "zscore": float(z),
        "percentile": float(percentileofscore(spread, current)),
        "signal": "WIDE" if z > 1.5 else ("NARROW" if z < -1.5 else "NORMAL"),
    }


def _extract_atm(surface, tenor: str) -> float:
    """Extract ATM vol from a surface dict. Returns vol in percent, or None if unavailable."""
    if isinstance(surface, dict):
        # bloomberg_fx returns {tenor: {"atm": val, ...}} directly (lowercase keys)
        if tenor in surface and isinstance(surface[tenor], dict):
            val = surface[tenor].get("atm", surface[tenor].get("ATM"))
            if val is not None:
                return val
        # Try nested "tenors" key for alternate format
        tenors = surface.get("tenors", {})
        if tenor in tenors:
            val = tenors[tenor].get("atm", tenors[tenor].get("ATM"))
            if val is not None:
                return val
        # Try nearest tenor
        for t in ["3M", "1M", "6M", "1Y"]:
            if t in surface and isinstance(surface[t], dict):
                val = surface[t].get("atm", surface[t].get("ATM"))
                if val is not None:
                    return val
    return None


_METRIC_KEY_MAP = {
    "ATM": "atm", "25D_RR": "rr25", "25D_BF": "bf25",
    "10D_RR": "rr10", "10D_BF": "bf10",
}


def _extract_metric(surface, tenor: str, metric: str) -> float:
    """Extract a specific metric from the surface dict. Returns vol in percent, or None if unavailable."""
    key = _METRIC_KEY_MAP.get(metric, metric.lower())
    if isinstance(surface, dict):
        # Direct tenor lookup (bloomberg_fx format)
        if tenor in surface and isinstance(surface[tenor], dict):
            val = surface[tenor].get(key, surface[tenor].get(metric))
            if val is not None:
                return val
        # Nested "tenors" key
        tenors = surface.get("tenors", {})
        if tenor in tenors:
            val = tenors[tenor].get(key, tenors[tenor].get(metric))
            if val is not None:
                return val
    return None




