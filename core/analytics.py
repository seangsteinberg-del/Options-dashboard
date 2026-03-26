"""
Advanced Analytics Engine
=========================
Volatility cones, regime detection, cross-asset correlation,
margin estimation, earnings calendar, and statistical analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, percentileofscore
from typing import Dict, List, Tuple

from core.pricing import (
    bs_price, compute_all_greeks, realized_vol_close_to_close,
)


# ═══════════════════════════════════════════════════════════════════════════
# Volatility Cones
# ═══════════════════════════════════════════════════════════════════════════

def volatility_cone(prices: np.ndarray, windows: List[int] = None) -> pd.DataFrame:
    """
    Compute volatility cone showing current RV vs historical percentile bands.
    """
    if windows is None:
        windows = [5, 10, 20, 30, 60, 90, 120, 180, 252]

    log_ret = np.diff(np.log(prices))
    records = []

    for w in windows:
        if len(log_ret) < w + 20:
            continue
        rv_series = pd.Series(log_ret).rolling(w).std() * np.sqrt(252)
        rv_clean = rv_series.dropna().values

        if len(rv_clean) < 5:
            continue

        current = rv_clean[-1] * 100
        records.append({
            "window": w,
            "current": current,
            "min": np.min(rv_clean) * 100,
            "p10": np.percentile(rv_clean, 10) * 100,
            "p25": np.percentile(rv_clean, 25) * 100,
            "median": np.median(rv_clean) * 100,
            "p75": np.percentile(rv_clean, 75) * 100,
            "p90": np.percentile(rv_clean, 90) * 100,
            "max": np.max(rv_clean) * 100,
            "percentile_rank": percentileofscore(rv_clean, rv_clean[-1]),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Volatility Regime Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_vol_regime(prices: np.ndarray, short_window: int = 10,
                      long_window: int = 60) -> dict:
    """
    Classify current volatility regime based on RV ratios and levels.
    Returns regime classification and supporting metrics.
    """
    log_ret = np.diff(np.log(prices))
    rv_short = pd.Series(log_ret).rolling(short_window).std().iloc[-1] * np.sqrt(252)
    rv_long = pd.Series(log_ret).rolling(long_window).std().iloc[-1] * np.sqrt(252)

    ratio = rv_short / max(rv_long, 0.001)
    rv_all = pd.Series(log_ret).rolling(20).std() * np.sqrt(252)
    rv_clean = rv_all.dropna().values
    rv_current = rv_clean[-1] if len(rv_clean) > 0 else rv_short
    rv_pct = percentileofscore(rv_clean, rv_current) if len(rv_clean) > 0 else 50.0

    # Regime classification
    if rv_short > 0.35:
        regime = "CRISIS"
        color = "#ef4444"
    elif ratio > 1.5 and rv_short > 0.25:
        regime = "STRESS"
        color = "#f59e0b"
    elif ratio > 1.2:
        regime = "ELEVATED"
        color = "#f59e0b"
    elif ratio < 0.7 and rv_short < 0.12:
        regime = "SUPPRESSED"
        color = "#06b6d4"
    elif rv_short < 0.15:
        regime = "LOW VOL"
        color = "#10b981"
    else:
        regime = "NORMAL"
        color = "#3b82f6"

    # Trend
    rv_5d = pd.Series(log_ret).rolling(5).std().iloc[-1] * np.sqrt(252)
    rv_20d = pd.Series(log_ret).rolling(20).std().iloc[-1] * np.sqrt(252)
    if rv_5d > rv_20d * 1.15:
        trend = "RISING"
    elif rv_5d < rv_20d * 0.85:
        trend = "FALLING"
    else:
        trend = "STABLE"

    return {
        "regime": regime,
        "color": color,
        "rv_short": rv_short * 100,
        "rv_long": rv_long * 100,
        "ratio": ratio,
        "percentile": rv_pct,
        "trend": trend,
        "rv_5d": rv_5d * 100,
        "rv_20d": rv_20d * 100,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Asset Correlation Matrix
# ═══════════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(price_dict: Dict[str, np.ndarray],
                                window: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation and covariance matrices from price histories.
    Returns (correlation_df, realized_vol_df).
    """
    returns = {}
    for ticker, prices in price_dict.items():
        if len(prices) > window:
            ret = np.diff(np.log(prices[-window - 1:]))
            returns[ticker] = ret[:window]

    if not returns:
        return pd.DataFrame(), pd.DataFrame()

    ret_df = pd.DataFrame(returns)
    corr = ret_df.corr()
    rv = ret_df.std() * np.sqrt(252) * 100  # annualized

    return corr, pd.DataFrame({"ticker": rv.index, "rv_ann": rv.values})


# ═══════════════════════════════════════════════════════════════════════════
# Margin Estimation (SPAN-like)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_margin(positions: List[dict], spot_prices: Dict[str, float],
                    r: float = 0.05, q: float = 0.015) -> dict:
    """
    Estimate portfolio margin using a simplified SPAN-like methodology.
    Scans across spot and vol scenarios to find worst-case loss.
    """
    spot_shocks = np.array([-0.20, -0.15, -0.10, -0.07, -0.05, -0.03,
                             0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20])
    vol_shocks = np.array([-0.15, -0.05, 0, 0.05, 0.15])

    current_val = 0
    for pos in positions:
        S = spot_prices.get(pos.get("ticker", "SPY"), 100)
        p = bs_price(S, pos["strike"], pos["expiry"], r, q, pos["vol"], pos["option_type"])
        current_val += p * pos["quantity"] * pos.get("multiplier", 100)

    worst_loss = 0
    worst_scenario = ""
    scenario_results = []

    for ds in spot_shocks:
        for dv in vol_shocks:
            scenario_val = 0
            for pos in positions:
                S = spot_prices.get(pos.get("ticker", "SPY"), 100)
                S_new = S * (1 + ds)
                sig_new = max(pos["vol"] + dv, 0.01)
                p = bs_price(S_new, pos["strike"], pos["expiry"], r, q, sig_new, pos["option_type"])
                scenario_val += p * pos["quantity"] * pos.get("multiplier", 100)

            loss = current_val - scenario_val
            scenario_results.append({
                "spot_shock": ds, "vol_shock": dv,
                "value": scenario_val, "loss": loss,
            })

            if loss > worst_loss:
                worst_loss = loss
                worst_scenario = f"Spot {ds:+.0%}, Vol {dv:+.0%}"

    # Short option margin add-on
    short_option_margin = 0
    for pos in positions:
        if pos["quantity"] < 0:
            S = spot_prices.get(pos.get("ticker", "SPY"), 100)
            short_option_margin += abs(pos["quantity"]) * pos.get("multiplier", 100) * S * 0.15

    initial_margin = max(worst_loss, short_option_margin) * 1.1
    maintenance_margin = initial_margin * 0.75

    return {
        "initial_margin": initial_margin,
        "maintenance_margin": maintenance_margin,
        "worst_case_loss": worst_loss,
        "worst_scenario": worst_scenario,
        "current_value": current_val,
        "short_option_margin": short_option_margin,
        "margin_utilization": initial_margin / max(abs(current_val), 1) * 100,
        "scenario_count": len(scenario_results),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Greeks Time Evolution (Theta Decay Visualization)
# ═══════════════════════════════════════════════════════════════════════════

def greeks_time_evolution(positions: List[dict], spot_prices: Dict[str, float],
                           r: float = 0.05, q: float = 0.015,
                           days: int = 60) -> pd.DataFrame:
    """Track how portfolio Greeks change as time passes."""
    records = []
    for day in range(days):
        dt = day / 365.0
        row = {"day": day, "delta": 0, "gamma": 0, "theta": 0, "vega": 0, "value": 0}

        for pos in positions:
            S = spot_prices.get(pos.get("ticker", "SPY"), 100)
            T = max(pos["expiry"] - dt, 1e-6)
            g = compute_all_greeks(S, pos["strike"], T, r, q, pos["vol"], pos["option_type"])
            m = pos["quantity"] * pos.get("multiplier", 100)

            row["delta"] += g["delta"] * m
            row["gamma"] += g["gamma"] * m
            row["theta"] += g["theta"] * m
            row["vega"] += g["vega"] * m
            row["value"] += g["price"] * m

        records.append(row)

    return pd.DataFrame(records)
