"""
FX Data Provider
================
Bloomberg API integration for FX markets with comprehensive synthetic fallback.

Provides live FX data when Bloomberg Terminal is available:
  - Spot rates with bid/ask/mid/change across all G10 and EM pairs
  - Implied vol surfaces from OVDV (ATM, risk reversals, butterflies)
  - Deposit rate curves and forward point curves
  - Historical spot, vol, and rate time series
  - FX options chains with full Greeks
  - CFTC positioning data
  - Realized vol, implied correlations, central bank calendars

When Bloomberg is unavailable, falls back to a rich synthetic engine
calibrated with realistic market parameters for 30 currency pairs,
including correlated GBM spot histories, mean-reverting vol dynamics,
and properly shaped vol surfaces with term structure.

All fallback data uses deterministic seeding (hash of pair name) so
results are reproducible across sessions.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Bloomberg connection reuse ───────────────────────────────────────────
try:
    from core.bloomberg import get_connection, is_connected, bdp, bdh, bds
    _HAS_EQUITY_BBG = True
except Exception as _bbg_import_err:
    _HAS_EQUITY_BBG = False
    logger.warning("Failed to import core.bloomberg: %s", _bbg_import_err)

try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# Cache Layer (thread-safe)
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_TTL = {
    "spot": 15,
    "vol_surface": 300,
    "rates": 1800,
    "forwards": 600,
    "historical": 3600,
    "positioning": 86400,
}

_cache: Dict[str, Tuple[float, object]] = {}
_cache_lock = threading.Lock()

# ── Data Integrity Tracking (thread-safe) ─────────────────────────────────
_fetch_errors: list = []
_errors_lock = threading.Lock()
_MAX_ERRORS = 50


def _log_fetch_failure(function: str, pair: str, error: str):
    """Record a Bloomberg fetch failure (only when Bloomberg is connected)."""
    with _errors_lock:
        _fetch_errors.append({
            "time": time.time(),
            "function": function,
            "pair": pair,
            "error": str(error),
        })
        if len(_fetch_errors) > _MAX_ERRORS:
            _fetch_errors.pop(0)
    logger.warning("BBG fetch failed [%s] %s: %s", function, pair, error)


def get_data_mode() -> str:
    """Return the current data mode: 'LIVE', 'SYNTHETIC', or 'DEGRADED'.
    LIVE = Bloomberg connected, fewer than 5 recent failures.
    DEGRADED = Bloomberg connected, but 5+ fetches failing (mixed data).
    SYNTHETIC = No Bloomberg connection, all data is synthetic.
    """
    if not (_HAS_EQUITY_BBG and is_connected()):
        return "SYNTHETIC"
    cutoff = time.time() - 300
    with _errors_lock:
        recent = [e for e in _fetch_errors if e["time"] > cutoff]
    if len(recent) >= 5:
        return "DEGRADED"
    return "LIVE"


def get_recent_errors() -> list:
    """Return recent fetch failures for UI display."""
    cutoff = time.time() - 300
    with _errors_lock:
        return [e for e in _fetch_errors if e["time"] > cutoff]


def clear_errors():
    """Clear the error log."""
    with _errors_lock:
        _fetch_errors.clear()


def _cache_get(key: str, category: str = "spot"):
    """Return cached value if not expired, else None."""
    with _cache_lock:
        if key in _cache:
            ts, val = _cache[key]
            if time.time() - ts < _CACHE_TTL.get(category, 60):
                return val
    return None


def _cache_set(key: str, value, category: str = "spot"):
    with _cache_lock:
        _cache[key] = (time.time(), value)


def cache_clear():
    """Flush the entire FX cache."""
    with _cache_lock:
        _cache.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Tenor Utilities
# ═══════════════════════════════════════════════════════════════════════════

_TENOR_DAYS = {
    "ON": 1, "1W": 7, "2W": 14, "1M": 30, "2M": 60, "3M": 91,
    "6M": 182, "9M": 274, "1Y": 365, "2Y": 730, "3Y": 1095, "5Y": 1825,
}

_ALL_TENORS = list(_TENOR_DAYS.keys())


def tenor_to_years(tenor: str) -> float:
    return _TENOR_DAYS.get(tenor.upper(), 30) / 365.0


# ═══════════════════════════════════════════════════════════════════════════
# FX Pair Correlation Matrix
# ═══════════════════════════════════════════════════════════════════════════

_FX_CORRELATION = {
    # ── G10 Majors vs Majors ──────────────────────────────────────────────
    ("EURUSD", "GBPUSD"): 0.75,
    ("EURUSD", "USDJPY"): -0.30,
    ("EURUSD", "USDCHF"): -0.90,
    ("EURUSD", "AUDUSD"): 0.60,
    ("EURUSD", "NZDUSD"): 0.55,
    ("EURUSD", "USDCAD"): -0.50,
    ("EURUSD", "USDMXN"): -0.55,
    ("EURUSD", "USDBRL"): -0.40,
    ("EURUSD", "USDTRY"): -0.25,
    ("EURUSD", "USDZAR"): -0.45,
    ("EURUSD", "USDCNH"): -0.30,
    ("GBPUSD", "USDJPY"): -0.25,
    ("GBPUSD", "USDCHF"): -0.70,
    ("GBPUSD", "AUDUSD"): 0.55,
    ("GBPUSD", "NZDUSD"): 0.50,
    ("GBPUSD", "USDCAD"): -0.45,
    ("AUDUSD", "USDJPY"): -0.20,
    ("AUDUSD", "USDCNH"): -0.35,
    ("USDJPY", "USDCHF"): 0.35,
    ("USDJPY", "USDCAD"): 0.25,
    ("USDCAD", "USDMXN"): 0.40,
    ("USDBRL", "USDZAR"): 0.50,
    ("EURNOK", "EURSEK"): 0.82,
    ("USDSEK", "USDNOK"): 0.80,
    ("USDSGD", "USDKRW"): 0.55,
    ("USDSGD", "USDCNH"): 0.60,
    ("USDCNH", "USDKRW"): 0.50,
    # ── Cross pairs vs their components ───────────────────────────────────
    ("EURGBP", "EURUSD"): -0.50,
    ("EURGBP", "GBPUSD"): -0.75,
    ("EURJPY", "USDJPY"): 0.65,
    ("EURJPY", "EURUSD"): 0.55,
    ("GBPJPY", "USDJPY"): 0.60,
    ("GBPJPY", "GBPUSD"): 0.55,
    ("AUDJPY", "USDJPY"): 0.55,
    ("AUDJPY", "AUDUSD"): 0.60,
    ("NZDJPY", "USDJPY"): 0.50,
    ("NZDJPY", "NZDUSD"): 0.55,
    ("EURCHF", "EURUSD"): 0.45,
    ("EURCHF", "USDCHF"): -0.65,
    ("EURAUD", "EURUSD"): 0.50,
    ("EURAUD", "AUDUSD"): -0.70,
    ("EURNZD", "EURUSD"): 0.45,
    ("EURNZD", "NZDUSD"): -0.65,
    ("CADCHF", "USDCAD"): -0.55,
    ("CADCHF", "USDCHF"): 0.50,
    ("CADJPY", "USDJPY"): 0.55,
    ("CADJPY", "USDCAD"): -0.50,
    ("AUDNZD", "AUDUSD"): 0.40,
    ("AUDNZD", "NZDUSD"): -0.35,
    # ── Scandie pairs vs majors and each other ────────────────────────────
    ("EURNOK", "EURUSD"): 0.55,
    ("EURSEK", "EURUSD"): 0.55,
    ("USDNOK", "USDCAD"): 0.45,
    ("USDSEK", "USDCAD"): 0.40,
    ("USDNOK", "EURUSD"): -0.50,
    ("USDSEK", "EURUSD"): -0.45,
    ("EURNOK", "USDNOK"): -0.35,
    ("EURSEK", "USDSEK"): -0.30,
    ("EURNOK", "GBPUSD"): 0.40,
    ("EURSEK", "GBPUSD"): 0.38,
    ("USDNOK", "USDJPY"): 0.20,
    ("USDSEK", "USDJPY"): 0.18,
    ("USDNOK", "USDCHF"): 0.35,
    ("USDSEK", "USDCHF"): 0.32,
    ("EURNOK", "EURCHF"): 0.30,
    ("EURSEK", "EURCHF"): 0.28,
    # ── EM correlations ───────────────────────────────────────────────────
    ("USDMXN", "USDZAR"): 0.60,
    ("USDBRL", "USDMXN"): 0.55,
    ("USDTRY", "USDZAR"): 0.50,
    ("USDTRY", "USDMXN"): 0.45,
    ("USDTRY", "USDBRL"): 0.40,
    ("USDBRL", "USDCNH"): 0.25,
    ("USDMXN", "USDCNH"): 0.30,
    ("USDZAR", "USDCNH"): 0.28,
    ("USDMXN", "USDINR"): 0.25,
    ("USDZAR", "USDINR"): 0.22,
    ("USDMXN", "USDKRW"): 0.35,
    ("USDZAR", "USDKRW"): 0.30,
    ("USDBRL", "USDKRW"): 0.28,
    ("USDTRY", "USDCNH"): 0.20,
    ("USDTRY", "USDINR"): 0.18,
    ("USDTRY", "USDKRW"): 0.22,
    ("USDCNH", "USDINR"): 0.40,
    ("USDKRW", "USDINR"): 0.38,
    ("USDSGD", "USDINR"): 0.35,
    # ── Commodity bloc ────────────────────────────────────────────────────
    ("AUDUSD", "USDCAD"): -0.55,
    ("NZDUSD", "USDCAD"): -0.45,
    ("AUDUSD", "NZDUSD"): 0.88,
    ("AUDUSD", "USDMXN"): -0.40,
    ("AUDUSD", "USDZAR"): -0.45,
    ("NZDUSD", "USDMXN"): -0.35,
    ("NZDUSD", "USDZAR"): -0.38,
    ("AUDUSD", "USDBRL"): -0.35,
    ("NZDUSD", "USDBRL"): -0.30,
    ("AUDUSD", "USDNOK"): -0.40,
    ("NZDUSD", "USDNOK"): -0.35,
    ("USDCAD", "USDNOK"): 0.50,
    ("USDCAD", "USDZAR"): 0.35,
    # ── JPY crosses vs each other ─────────────────────────────────────────
    ("EURJPY", "GBPJPY"): 0.80,
    ("AUDJPY", "EURJPY"): 0.60,
    ("AUDJPY", "GBPJPY"): 0.58,
    ("NZDJPY", "EURJPY"): 0.55,
    ("NZDJPY", "GBPJPY"): 0.52,
    ("NZDJPY", "AUDJPY"): 0.82,
    ("CADJPY", "EURJPY"): 0.55,
    ("CADJPY", "GBPJPY"): 0.50,
    ("CADJPY", "AUDJPY"): 0.48,
    ("CADJPY", "NZDJPY"): 0.45,
    # ── CHF crosses ───────────────────────────────────────────────────────
    ("EURCHF", "GBPUSD"): 0.35,
    ("EURCHF", "AUDUSD"): 0.30,
    ("CADCHF", "EURCHF"): 0.40,
    # ── Additional G10 cross linkages ─────────────────────────────────────
    ("EURGBP", "EURCHF"): 0.35,
    ("EURGBP", "EURJPY"): 0.25,
    ("EURAUD", "EURNZD"): 0.75,
    ("EURAUD", "EURGBP"): 0.30,
    ("GBPUSD", "USDMXN"): -0.40,
    ("GBPUSD", "USDZAR"): -0.38,
    ("USDCHF", "USDCAD"): 0.40,
    ("USDCHF", "USDMXN"): 0.35,
    ("USDCHF", "USDZAR"): 0.32,
}


def _get_correlation(pair_a: str, pair_b: str) -> float:
    """Look up pairwise correlation, checking both orderings."""
    if pair_a == pair_b:
        return 1.0
    return _FX_CORRELATION.get(
        (pair_a, pair_b),
        _FX_CORRELATION.get((pair_b, pair_a), 0.15),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Full Fallback Data for 30 Currency Pairs
# ═══════════════════════════════════════════════════════════════════════════

def _make_vol_surface(atm_base, rr25_base, bf25_base, rr10_base, bf10_base,
                      term_slope=1.0):
    """
    Build a full tenor vol surface from base 1M parameters.

    Short tenors are slightly lower (backwardation uncommon in FX vol).
    Long tenors scale upward with term_slope controlling steepness.
    """
    # Multipliers relative to the 1M point
    tenor_mult = {
        "ON": 1.50, "1W": 1.08, "2W": 1.03, "1M": 1.00, "2M": 0.99,
        "3M": 0.98, "6M": 0.97 * term_slope, "9M": 0.97 * term_slope,
        "1Y": 0.96 * term_slope, "2Y": 0.96 * term_slope ** 1.1,
        "3Y": 0.95 * term_slope ** 1.15, "5Y": 0.95 * term_slope ** 1.2,
    }
    # Short tenors get enhanced skew/convexity
    skew_mult = {
        "ON": 1.8, "1W": 1.5, "2W": 1.2, "1M": 1.0, "2M": 0.95,
        "3M": 0.90, "6M": 0.85, "9M": 0.82, "1Y": 0.80,
        "2Y": 0.75, "3Y": 0.72, "5Y": 0.70,
    }
    surface = {}
    for tenor in _ALL_TENORS:
        tm = tenor_mult[tenor]
        sm = skew_mult[tenor]
        surface[tenor] = {
            "atm": round(atm_base * tm, 2),
            "rr25": round(rr25_base * sm, 2),
            "bf25": round(bf25_base * sm, 2),
            "rr10": round(rr10_base * sm, 2),
            "bf10": round(bf10_base * sm, 2),
        }
    return surface


_FX_FALLBACK: Dict[str, dict] = {
    # ── G10 Majors ────────────────────────────────────────────────────────
    "EURUSD": {
        "spot": 1.0850, "r_dom": 0.0530, "r_for": 0.0390,
        "beta_to_dxy": -0.92, "daily_vol": 0.065,
        "vol_surface": _make_vol_surface(7.20, -0.40, 0.25, -0.85, 0.45),
        "vol_history_params": {"mean": 7.20, "std": 1.5, "mr_speed": 0.03, "vol_of_vol": 0.8},
    },
    "USDJPY": {
        "spot": 149.50, "r_dom": -0.0010, "r_for": 0.0530,
        "beta_to_dxy": 0.85, "daily_vol": 0.072,
        "vol_surface": _make_vol_surface(10.80, 0.70, 0.45, 1.40, 0.80),
        "vol_history_params": {"mean": 10.80, "std": 2.0, "mr_speed": 0.025, "vol_of_vol": 1.0},
    },
    "GBPUSD": {
        "spot": 1.2650, "r_dom": 0.0530, "r_for": 0.0520,
        "beta_to_dxy": -0.80, "daily_vol": 0.068,
        "vol_surface": _make_vol_surface(8.50, -0.55, 0.30, -1.10, 0.55),
        "vol_history_params": {"mean": 8.50, "std": 1.6, "mr_speed": 0.03, "vol_of_vol": 0.85},
    },
    "USDCHF": {
        "spot": 0.8820, "r_dom": 0.0175, "r_for": 0.0530,
        "beta_to_dxy": 0.90, "daily_vol": 0.060,
        "vol_surface": _make_vol_surface(7.80, -0.35, 0.22, -0.75, 0.40),
        "vol_history_params": {"mean": 7.80, "std": 1.4, "mr_speed": 0.035, "vol_of_vol": 0.75},
    },
    "AUDUSD": {
        "spot": 0.6520, "r_dom": 0.0530, "r_for": 0.0435,
        "beta_to_dxy": -0.70, "daily_vol": 0.075,
        "vol_surface": _make_vol_surface(9.80, -0.60, 0.35, -1.20, 0.60),
        "vol_history_params": {"mean": 9.80, "std": 1.8, "mr_speed": 0.028, "vol_of_vol": 0.90},
    },
    "NZDUSD": {
        "spot": 0.6080, "r_dom": 0.0530, "r_for": 0.0550,
        "beta_to_dxy": -0.65, "daily_vol": 0.078,
        "vol_surface": _make_vol_surface(10.40, -0.55, 0.35, -1.15, 0.60),
        "vol_history_params": {"mean": 10.40, "std": 1.9, "mr_speed": 0.028, "vol_of_vol": 0.92},
    },
    "USDCAD": {
        "spot": 1.3580, "r_dom": 0.0500, "r_for": 0.0530,
        "beta_to_dxy": 0.65, "daily_vol": 0.058,
        "vol_surface": _make_vol_surface(7.00, -0.30, 0.22, -0.65, 0.40),
        "vol_history_params": {"mean": 7.00, "std": 1.3, "mr_speed": 0.032, "vol_of_vol": 0.72},
    },
    # ── G10 Crosses ───────────────────────────────────────────────────────
    "EURGBP": {
        "spot": 0.8580, "r_dom": 0.0520, "r_for": 0.0390,
        "beta_to_dxy": -0.15, "daily_vol": 0.055,
        "vol_surface": _make_vol_surface(6.80, -0.30, 0.20, -0.65, 0.38),
        "vol_history_params": {"mean": 6.80, "std": 1.2, "mr_speed": 0.035, "vol_of_vol": 0.70},
    },
    "EURJPY": {
        "spot": 162.20, "r_dom": -0.0010, "r_for": 0.0390,
        "beta_to_dxy": -0.10, "daily_vol": 0.080,
        "vol_surface": _make_vol_surface(10.50, 0.55, 0.40, 1.10, 0.70),
        "vol_history_params": {"mean": 10.50, "std": 2.0, "mr_speed": 0.025, "vol_of_vol": 0.95},
    },
    "GBPJPY": {
        "spot": 189.10, "r_dom": -0.0010, "r_for": 0.0520,
        "beta_to_dxy": -0.05, "daily_vol": 0.090,
        "vol_surface": _make_vol_surface(11.80, 0.80, 0.50, 1.60, 0.85),
        "vol_history_params": {"mean": 11.80, "std": 2.2, "mr_speed": 0.022, "vol_of_vol": 1.05},
    },
    "AUDJPY": {
        "spot": 97.45, "r_dom": -0.0010, "r_for": 0.0435,
        "beta_to_dxy": -0.20, "daily_vol": 0.088,
        "vol_surface": _make_vol_surface(11.50, 0.65, 0.45, 1.30, 0.75),
        "vol_history_params": {"mean": 11.50, "std": 2.1, "mr_speed": 0.024, "vol_of_vol": 1.00},
    },
    "EURCHF": {
        "spot": 0.9570, "r_dom": 0.0175, "r_for": 0.0390,
        "beta_to_dxy": -0.05, "daily_vol": 0.050,
        "vol_surface": _make_vol_surface(5.80, -0.45, 0.25, -0.90, 0.45),
        "vol_history_params": {"mean": 5.80, "std": 1.5, "mr_speed": 0.030, "vol_of_vol": 0.80},
    },
    "EURAUD": {
        "spot": 1.6640, "r_dom": 0.0435, "r_for": 0.0390,
        "beta_to_dxy": 0.25, "daily_vol": 0.070,
        "vol_surface": _make_vol_surface(9.20, -0.45, 0.30, -0.95, 0.55),
        "vol_history_params": {"mean": 9.20, "std": 1.7, "mr_speed": 0.028, "vol_of_vol": 0.88},
    },
    "EURNZD": {
        "spot": 1.7845, "r_dom": 0.0550, "r_for": 0.0390,
        "beta_to_dxy": 0.20, "daily_vol": 0.074,
        "vol_surface": _make_vol_surface(9.80, -0.50, 0.32, -1.05, 0.58),
        "vol_history_params": {"mean": 9.80, "std": 1.8, "mr_speed": 0.027, "vol_of_vol": 0.90},
    },
    "NZDJPY": {
        "spot": 90.90, "r_dom": -0.0010, "r_for": 0.0550,
        "beta_to_dxy": -0.18, "daily_vol": 0.085,
        "vol_surface": _make_vol_surface(11.20, 0.60, 0.42, 1.25, 0.72),
        "vol_history_params": {"mean": 11.20, "std": 2.0, "mr_speed": 0.024, "vol_of_vol": 0.98},
    },
    "AUDNZD": {
        "spot": 1.0720, "r_dom": 0.0550, "r_for": 0.0435,
        "beta_to_dxy": -0.05, "daily_vol": 0.048,
        "vol_surface": _make_vol_surface(6.20, -0.20, 0.18, -0.45, 0.32),
        "vol_history_params": {"mean": 6.20, "std": 1.0, "mr_speed": 0.035, "vol_of_vol": 0.65},
    },
    "CADCHF": {
        "spot": 0.6495, "r_dom": 0.0175, "r_for": 0.0500,
        "beta_to_dxy": 0.10, "daily_vol": 0.062,
        "vol_surface": _make_vol_surface(7.50, -0.30, 0.22, -0.65, 0.40),
        "vol_history_params": {"mean": 7.50, "std": 1.3, "mr_speed": 0.032, "vol_of_vol": 0.72},
    },
    "CADJPY": {
        "spot": 110.10, "r_dom": -0.0010, "r_for": 0.0500,
        "beta_to_dxy": 0.15, "daily_vol": 0.082,
        "vol_surface": _make_vol_surface(10.80, 0.55, 0.40, 1.15, 0.70),
        "vol_history_params": {"mean": 10.80, "std": 1.9, "mr_speed": 0.025, "vol_of_vol": 0.95},
    },
    # ── Scandies ──────────────────────────────────────────────────────────
    "EURNOK": {
        "spot": 11.45, "r_dom": 0.0450, "r_for": 0.0390,
        "beta_to_dxy": 0.30, "daily_vol": 0.082,
        "vol_surface": _make_vol_surface(10.20, -0.50, 0.35, -1.05, 0.60),
        "vol_history_params": {"mean": 10.20, "std": 1.8, "mr_speed": 0.028, "vol_of_vol": 0.90},
    },
    "EURSEK": {
        "spot": 11.28, "r_dom": 0.0400, "r_for": 0.0390,
        "beta_to_dxy": 0.25, "daily_vol": 0.070,
        "vol_surface": _make_vol_surface(8.80, -0.40, 0.28, -0.85, 0.50),
        "vol_history_params": {"mean": 8.80, "std": 1.5, "mr_speed": 0.030, "vol_of_vol": 0.82},
    },
    "USDSEK": {
        "spot": 10.40, "r_dom": 0.0400, "r_for": 0.0530,
        "beta_to_dxy": 0.75, "daily_vol": 0.085,
        "vol_surface": _make_vol_surface(10.60, -0.45, 0.32, -0.95, 0.55),
        "vol_history_params": {"mean": 10.60, "std": 1.9, "mr_speed": 0.027, "vol_of_vol": 0.92},
    },
    "USDNOK": {
        "spot": 10.55, "r_dom": 0.0450, "r_for": 0.0530,
        "beta_to_dxy": 0.78, "daily_vol": 0.092,
        "vol_surface": _make_vol_surface(11.40, -0.55, 0.38, -1.15, 0.65),
        "vol_history_params": {"mean": 11.40, "std": 2.0, "mr_speed": 0.026, "vol_of_vol": 0.95},
    },
    # ── Emerging Markets ──────────────────────────────────────────────────
    "USDMXN": {
        "spot": 17.25, "r_dom": 0.1125, "r_for": 0.0530,
        "beta_to_dxy": 0.55, "daily_vol": 0.095,
        "vol_surface": _make_vol_surface(13.50, 1.20, 0.65, 2.40, 1.10, term_slope=1.04),
        "vol_history_params": {"mean": 13.50, "std": 2.5, "mr_speed": 0.020, "vol_of_vol": 1.20},
    },
    "USDBRL": {
        "spot": 4.98, "r_dom": 0.1175, "r_for": 0.0530,
        "beta_to_dxy": 0.50, "daily_vol": 0.105,
        "vol_surface": _make_vol_surface(15.20, 1.50, 0.80, 3.00, 1.35, term_slope=1.05),
        "vol_history_params": {"mean": 15.20, "std": 3.0, "mr_speed": 0.018, "vol_of_vol": 1.40},
    },
    "USDTRY": {
        "spot": 32.50, "r_dom": 0.4500, "r_for": 0.0530,
        "beta_to_dxy": 0.20, "daily_vol": 0.140,
        "vol_surface": _make_vol_surface(24.50, 2.80, 1.50, 5.50, 2.80, term_slope=1.08),
        "vol_history_params": {"mean": 24.50, "std": 5.0, "mr_speed": 0.012, "vol_of_vol": 2.20},
    },
    "USDZAR": {
        "spot": 18.65, "r_dom": 0.0825, "r_for": 0.0530,
        "beta_to_dxy": 0.60, "daily_vol": 0.115,
        "vol_surface": _make_vol_surface(16.80, 1.60, 0.85, 3.20, 1.45, term_slope=1.04),
        "vol_history_params": {"mean": 16.80, "std": 3.2, "mr_speed": 0.018, "vol_of_vol": 1.50},
    },
    "USDCNH": {
        "spot": 7.24, "r_dom": 0.0250, "r_for": 0.0530,
        "beta_to_dxy": 0.72, "daily_vol": 0.048,
        "vol_surface": _make_vol_surface(6.20, 0.40, 0.25, 0.85, 0.45, term_slope=1.02),
        "vol_history_params": {"mean": 6.20, "std": 1.2, "mr_speed": 0.030, "vol_of_vol": 0.70},
    },
    "USDINR": {
        "spot": 83.20, "r_dom": 0.0650, "r_for": 0.0530,
        "beta_to_dxy": 0.55, "daily_vol": 0.038,
        "vol_surface": _make_vol_surface(4.80, 0.60, 0.30, 1.20, 0.55, term_slope=1.03),
        "vol_history_params": {"mean": 4.80, "std": 0.8, "mr_speed": 0.040, "vol_of_vol": 0.55},
    },
    "USDSGD": {
        "spot": 1.3420, "r_dom": 0.0380, "r_for": 0.0530,
        "beta_to_dxy": 0.68, "daily_vol": 0.040,
        "vol_surface": _make_vol_surface(5.50, 0.25, 0.18, 0.55, 0.32, term_slope=1.01),
        "vol_history_params": {"mean": 5.50, "std": 0.9, "mr_speed": 0.035, "vol_of_vol": 0.60},
    },
    "USDKRW": {
        "spot": 1325.0, "r_dom": 0.0350, "r_for": 0.0530,
        "beta_to_dxy": 0.62, "daily_vol": 0.065,
        "vol_surface": _make_vol_surface(8.50, 0.80, 0.45, 1.60, 0.80, term_slope=1.03),
        "vol_history_params": {"mean": 8.50, "std": 1.6, "mr_speed": 0.028, "vol_of_vol": 0.88},
    },
}

_ALL_PAIRS = list(_FX_FALLBACK.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic Seeding
# ═══════════════════════════════════════════════════════════════════════════

def _pair_seed(pair: str, extra: int = 0) -> int:
    """Deterministic seed from pair name for reproducible synthetics."""
    return (abs(hash(pair.upper())) + extra) % (2**31)


# ═══════════════════════════════════════════════════════════════════════════
# BBG Ticker Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fx_bbg_ticker(pair: str) -> str:
    """Convert pair like EURUSD to Bloomberg ticker EURUSD Curncy."""
    return f"{pair.upper()} Curncy"


def _fx_vol_bbg(pair: str) -> str:
    """OVDV vol ticker, e.g., EURUSDV1M Curncy."""
    return f"{pair.upper()}V Curncy"


def _deposit_bbg(ccy: str, tenor: str) -> str:
    """Deposit rate ticker, e.g., USDRC Index for 3M USD deposit.
    Uses Bloomberg deposit rate tickers.
    Format: {CCY_PREFIX}{TENOR_CODE} Index
    Verified against cuemacro/findatapy base_depos_tickers_list.csv.
    """
    ccy_map = {
        "USD": "USD", "EUR": "EUD", "GBP": "BPD", "JPY": "JYD", "CHF": "SFD",
        "AUD": "ADD", "NZD": "NDD", "CAD": "CDD", "SEK": "SKD", "NOK": "NKD",
        "MXN": "MPD", "BRL": "BZD", "TRY": "TYD", "ZAR": "SAD", "CNH": "CCD",
        "INR": "IND", "SGD": "SGD", "KRW": "KRD",
    }
    prefix = ccy_map.get(ccy.upper(), ccy.upper()[:2] + "D")
    tenor_letter = {"1M": "RA", "2M": "RB", "3M": "RC", "6M": "RF",
                    "9M": "RI", "1Y": "R1", "2Y": "R2", "3Y": "R3", "5Y": "R5"}
    suffix = tenor_letter.get(tenor.upper(), "RC")
    return f"{prefix}{suffix} Index"


# ═══════════════════════════════════════════════════════════════════════════
# Live Bloomberg API Functions
# ═══════════════════════════════════════════════════════════════════════════

def get_fx_spots(pairs: List[str] = None) -> Dict[str, dict]:
    """
    Spot rates for FX pairs with bid/ask/mid/change.

    Returns {pair: {bid, ask, mid, change, change_pct, high, low, open, volume_ind}}.
    """
    if pairs is None:
        pairs = _ALL_PAIRS

    cached = _cache_get("spots_" + ",".join(pairs), "spot")
    if cached is not None:
        return cached

    if _HAS_EQUITY_BBG and is_connected():
        try:
            tickers = [_fx_bbg_ticker(p) for p in pairs]
            fields = ["PX_BID", "PX_ASK", "PX_LAST", "PX_MID", "CHG_NET_1D",
                       "CHG_PCT_1D", "PX_HIGH", "PX_LOW", "PX_OPEN", "VOLUME"]
            df = bdp(tickers, fields)
            logger.info("FX spots: BDP returned %d rows for %d pairs. Index: %s",
                        len(df), len(tickers), list(df.index[:5]) if not df.empty else "EMPTY")
            if df.empty:
                logger.warning("FX spots: Bloomberg returned EMPTY — terminal may not "
                               "be logged in or authenticated")
            def _sf(v):
                """Safe float — handles None from Bloomberg null fields."""
                try:
                    return float(v) if v is not None else 0.0
                except (ValueError, TypeError):
                    return 0.0

            # Case-insensitive index lookup
            idx_map = {}
            if not df.empty:
                for iv in df.index:
                    idx_map[iv.upper().strip()] = iv

            result = {}
            missing = []
            for pair, ticker in zip(pairs, tickers):
                actual = ticker if ticker in df.index else idx_map.get(ticker.upper().strip())
                if actual is not None:
                    row = df.loc[actual]
                    bid = _sf(row.get("PX_BID"))
                    ask = _sf(row.get("PX_ASK"))
                    # Best mid: bid/ask average > PX_MID > PX_LAST
                    if bid and ask:
                        mid = round((bid + ask) / 2, 6)
                    else:
                        mid = _sf(row.get("PX_MID")) or _sf(row.get("PX_LAST"))
                    result[pair] = {
                        "bid": bid, "ask": ask,
                        "mid": mid,
                        "change": _sf(row.get("CHG_NET_1D")),
                        "change_pct": _sf(row.get("CHG_PCT_1D")),
                        "high": _sf(row.get("PX_HIGH")),
                        "low": _sf(row.get("PX_LOW")),
                        "open": _sf(row.get("PX_OPEN")),
                        "volume_ind": _sf(row.get("VOLUME")),
                    }
                else:
                    missing.append(pair)
            if missing:
                _log_fetch_failure("get_fx_spots", ",".join(missing[:5]),
                                  f"{len(missing)} pairs missing from BDP response")
            if result:
                _cache_set("spots_" + ",".join(pairs), result, "spot")
                return result
            # All pairs missing — log and return empty
            _log_fetch_failure("get_fx_spots", ",".join(pairs[:3]), "BDP returned no data for any pair")
            return {}
        except Exception as e:
            logger.error(f"FX spots BBG request failed: {e}")
            _log_fetch_failure("get_fx_spots", ",".join(pairs[:3]), str(e))
            return {}

    # SYNTHETIC mode only — Bloomberg not connected
    result = {p: _fallback_spot(p) for p in pairs}
    _cache_set("spots_" + ",".join(pairs), result, "spot")
    return result


def get_fx_vol_surface(pair: str) -> Dict[str, dict]:
    """
    Full vol surface: {tenor: {atm, rr25, bf25, rr10, bf10}}.
    Uses OVDV screen fields when Bloomberg is available.
    """
    ck = f"volsurf_{pair}"
    cached = _cache_get(ck, "vol_surface")
    if cached is not None:
        return cached

    if _HAS_EQUITY_BBG and is_connected():
        try:
            # Use registry prefixes for correct Bloomberg tickers
            from core.fx_conventions import FX_PAIR_REGISTRY
            spec = FX_PAIR_REGISTRY.get(pair.upper())
            pair_u = pair.upper()
            vol_pfx = spec.bb_vol_prefix if spec else f"{pair_u}V"
            rr25_pfx = spec.bb_rr25_prefix if spec else f"{pair_u}25R"
            bf25_pfx = spec.bb_bf25_prefix if spec else f"{pair_u}25B"
            rr10_pfx = spec.bb_rr10_prefix if spec else f"{pair_u}10R"
            bf10_pfx = spec.bb_bf10_prefix if spec else f"{pair_u}10B"

            # Build ALL tickers for ALL tenors in one batch (not per-tenor)
            all_tickers = []
            ticker_map = {}  # ticker -> (tenor, metric_name)
            for tenor in _ALL_TENORS:
                tc = tenor.upper()
                mapping = [
                    (f"{vol_pfx}{tc} Curncy", tenor, "atm"),
                    (f"{rr25_pfx}{tc} Curncy", tenor, "rr25"),
                    (f"{bf25_pfx}{tc} Curncy", tenor, "bf25"),
                    (f"{rr10_pfx}{tc} Curncy", tenor, "rr10"),
                    (f"{bf10_pfx}{tc} Curncy", tenor, "bf10"),
                ]
                for tick, t, m in mapping:
                    all_tickers.append(tick)
                    ticker_map[tick] = (t, m)

            # Single batched bdp call — request PX_LAST and PX_MID
            # Some Bloomberg terminals only populate PX_MID for FX vol
            vol_fields = ["PX_LAST", "PX_MID"]
            logger.info("Vol surface %s: requesting %d tickers, first 3: %s",
                        pair, len(all_tickers), all_tickers[:3])
            df = bdp(all_tickers, vol_fields)
            logger.info("Vol surface %s: BDP returned %d rows. Index values: %s",
                        pair, len(df),
                        list(df.index[:10]) if not df.empty else "EMPTY")
            if not df.empty:
                # Log actual values for first few tickers so we can see what BBG sends
                sample = df.head(5)
                for idx_val in sample.index:
                    px_last = sample.loc[idx_val, "PX_LAST"] if "PX_LAST" in sample.columns else "N/A"
                    px_mid = sample.loc[idx_val, "PX_MID"] if "PX_MID" in sample.columns else "N/A"
                    logger.info("  BDP row: %r -> PX_LAST=%r, PX_MID=%r", idx_val, px_last, px_mid)

            # Build a case-insensitive lookup from whatever Bloomberg returned
            idx_map = {}
            if not df.empty:
                for idx_val in df.index:
                    idx_map[idx_val.upper().strip()] = idx_val

            def _best_vol_value(row):
                """Extract best available vol value: PX_LAST > PX_MID."""
                for fld in ("PX_LAST", "PX_MID"):
                    if fld not in df.columns:
                        continue
                    v = row[fld] if isinstance(row, pd.Series) else df.loc[row, fld]
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    try:
                        fv = float(v)
                        if fv != 0:
                            return fv
                    except (TypeError, ValueError):
                        continue
                return None

            matched = 0
            null_count = 0
            surface = {}
            for tick, (tenor, metric_name) in ticker_map.items():
                # Try exact match first, then case-insensitive
                actual = tick if tick in df.index else idx_map.get(tick.upper().strip())
                if actual is not None:
                    val = _best_vol_value(df.loc[actual])
                    if val is not None:
                        if tenor not in surface:
                            surface[tenor] = {}
                        surface[tenor][metric_name] = val
                        matched += 1
                    else:
                        null_count += 1
            if null_count > 0 and matched == 0:
                logger.warning("Vol surface %s: BDP returned %d rows but ALL values "
                               "were None/NaN/0 — Bloomberg may not be fully authenticated "
                               "or lacks FX vol data subscription", pair, len(df))

            # Only keep tenors that have at least ATM
            surface = {t: v for t, v in surface.items() if "atm" in v}
            # Fill missing metrics with 0
            for t in surface:
                for m in ("atm", "rr25", "bf25", "rr10", "bf10"):
                    surface[t].setdefault(m, 0.0)

            if surface:
                logger.info("Vol surface %s: %d tenors from %d/%d tickers",
                            pair, len(surface), matched, len(all_tickers))
                _cache_set(ck, surface, "vol_surface")
                return surface
            logger.warning("Vol surface %s: 0 ATM tenors (BDP returned %d rows, "
                           "matched %d/%d tickers). Sample tickers sent: %s",
                           pair, len(df), matched, len(all_tickers),
                           all_tickers[:3])
            return {}
        except Exception as e:
            logger.error(f"FX vol surface BBG request failed: {e}")
            _log_fetch_failure("get_fx_vol_surface", pair, str(e))
            return {}

    # SYNTHETIC mode only — Bloomberg not connected
    surface = _fallback_vol_surface(pair)
    _cache_set(ck, surface, "vol_surface")
    return surface


def get_fx_vol_point(pair: str, tenor: str = "1M",
                     metric: str = "atm") -> float:
    """Single vol point from the surface."""
    surface = get_fx_vol_surface(pair)
    if tenor in surface and metric in surface[tenor]:
        return surface[tenor][metric]
    # Fallback: return ATM vol from nearest available tenor instead of
    # a hardcoded value, so the result scales with the pair's vol regime.
    if surface:
        req_days = _TENOR_DAYS.get(tenor.upper(), 30)
        best_tenor = min(surface.keys(),
                         key=lambda t: abs(_TENOR_DAYS.get(t, 9999) - req_days))
        val = surface[best_tenor].get("atm")
        if val is not None:
            return val
    return None


def get_fx_rates(pair: str) -> dict:
    """
    Domestic and foreign risk-free rates plus differential.
    Returns {r_dom, r_for, rate_diff}.
    """
    ck = f"rates_{pair}"
    cached = _cache_get(ck, "rates")
    if cached is not None:
        return cached

    if _HAS_EQUITY_BBG and is_connected():
        try:
            # FX options convention: domestic = quote currency, foreign = base currency
            # For EURUSD: domestic=USD (quote), foreign=EUR (base)
            ccy_dom = pair[3:]   # quote currency (pricing currency)
            ccy_for = pair[:3]   # base currency (underlying asset)
            # Simplification: use 3M deposit rates
            dom_tick = _deposit_bbg(ccy_dom, "3M")
            for_tick = _deposit_bbg(ccy_for, "3M")
            df = bdp([dom_tick, for_tick], ["PX_LAST", "PX_MID"])
            # Case-insensitive lookup
            idx_map = {iv.upper().strip(): iv for iv in df.index} if not df.empty else {}
            dom_actual = dom_tick if dom_tick in df.index else idx_map.get(dom_tick.upper().strip())
            for_actual = for_tick if for_tick in df.index else idx_map.get(for_tick.upper().strip())
            if dom_actual is None or for_actual is None:
                raise ValueError(f"Missing rate data: dom={dom_actual is not None}, for={for_actual is not None}")

            def _rate_val(actual_idx):
                """Extract rate from PX_LAST or PX_MID."""
                for fld in ("PX_LAST", "PX_MID"):
                    if fld in df.columns:
                        v = df.loc[actual_idx, fld]
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            return float(v) / 100.0
                raise ValueError(f"No rate data for {actual_idx}")

            r_dom = _rate_val(dom_actual)
            r_for = _rate_val(for_actual)
            res = {"r_dom": r_dom, "r_for": r_for,
                   "rate_diff": round(r_dom - r_for, 4)}
            _cache_set(ck, res, "rates")
            return res
        except Exception as e:
            logger.error(f"FX rates BBG request failed: {e}")
            _log_fetch_failure("get_fx_rates", pair, str(e))
            return {}

    # SYNTHETIC mode only — Bloomberg not connected
    res = _fallback_rates(pair)
    _cache_set(ck, res, "rates")
    return res


def get_fx_rate_curve(ccy: str) -> Dict[str, float]:
    """
    Deposit/swap rate curve for a single currency.
    Returns {tenor: rate}.
    """
    ck = f"ratecurve_{ccy}"
    cached = _cache_get(ck, "rates")
    if cached is not None:
        return cached

    if _HAS_EQUITY_BBG and is_connected():
        try:
            tenors = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y"]
            tickers = [_deposit_bbg(ccy, t) for t in tenors]
            df = bdp(tickers, ["PX_LAST", "PX_MID"])  # Single batched call
            idx_map = {iv.upper().strip(): iv for iv in df.index} if not df.empty else {}
            curve = {}
            for tenor, tick in zip(tenors, tickers):
                actual = tick if tick in df.index else idx_map.get(tick.upper().strip())
                if actual is not None:
                    # Try PX_LAST then PX_MID
                    val = None
                    for fld in ("PX_LAST", "PX_MID"):
                        if fld in df.columns:
                            v = df.loc[actual, fld]
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                val = v
                                break
                    if val is not None:
                        curve[tenor] = float(val) / 100.0
            if curve:
                _cache_set(ck, curve, "rates")
                return curve
            _log_fetch_failure("get_fx_rate_curve", ccy, "BDP returned no rate data")
            return {}
        except Exception as e:
            logger.error(f"FX rate curve BBG request failed: {e}")
            _log_fetch_failure("get_fx_rate_curve", ccy, str(e))
            return {}

    # SYNTHETIC mode only — Bloomberg not connected
    curve = _fallback_rate_curve(ccy)
    _cache_set(ck, curve, "rates")
    return curve


def get_fx_forward_curve(pair: str) -> Dict[str, dict]:
    """
    Forward point curve.
    Returns {tenor: {fwd_points, outright, implied_rate_diff}}.
    """
    ck = f"fwdcurve_{pair}"
    cached = _cache_get(ck, "forwards")
    if cached is not None:
        return cached

    if _HAS_EQUITY_BBG and is_connected():
        try:
            # Forward tickers use registry prefix (handles cross pairs + JPY correctly)
            # e.g., EURUSD → "EUR", USDJPY → "JPY", EURGBP → "EURGBP"
            from core.fx_conventions import FX_PAIR_REGISTRY
            fwd_spec = FX_PAIR_REGISTRY.get(pair.upper())
            # bb_fwd_prefix stores the base form like "EUR Curncy" or "EURGBP Curncy"
            # We extract the symbol part before " Curncy" to build tenor tickers
            if fwd_spec and fwd_spec.bb_fwd_prefix:
                fwd_sym = fwd_spec.bb_fwd_prefix.replace(" Curncy", "").strip()
            else:
                fwd_sym = pair[:3].upper()
            fwd_tenor_map = {"ON": "ON", "1W": "1W", "2W": "2W", "1M": "1M", "2M": "2M",
                             "3M": "3M", "6M": "6M", "9M": "9M", "1Y": "12M", "2Y": "2Y",
                             "3Y": "3Y", "5Y": "5Y"}
            tickers = [f"{fwd_sym}{fwd_tenor_map.get(t, t)} Curncy" for t in _ALL_TENORS]
            df = bdp(tickers, ["PX_LAST", "PX_MID"])
            idx_map = {iv.upper().strip(): iv for iv in df.index} if not df.empty else {}
            spot_data = get_fx_spots([pair])
            if pair not in spot_data:
                raise ValueError(f"No spot data for {pair}")
            spot = spot_data[pair]["mid"]
            # Forward points divisor: JPY pairs use 100, others use 10000
            pip_size = fwd_spec.pip if fwd_spec else 0.0001
            pts_divisor = 1.0 / pip_size  # e.g., 10000 for 0.0001 pip, 100 for 0.01 pip
            curve = {}
            for tenor, tick in zip(_ALL_TENORS, tickers):
                actual = tick if tick in df.index else idx_map.get(tick.upper().strip())
                if actual is not None:
                    pts_raw = None
                    for fld in ("PX_LAST", "PX_MID"):
                        if fld in df.columns:
                            v = df.loc[actual, fld]
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                pts_raw = v
                                break
                    if pts_raw is None:
                        continue
                    pts = float(pts_raw)
                    outright = spot + pts / pts_divisor
                    ty = tenor_to_years(tenor)
                    impl_diff = np.log(outright / spot) / ty if ty > 0 else 0.0
                    curve[tenor] = {
                        "fwd_points": pts,
                        "outright": round(outright, 6),
                        "implied_rate_diff": round(impl_diff, 4),
                    }
            if curve:
                _cache_set(ck, curve, "forwards")
                return curve
            _log_fetch_failure("get_fx_forward_curve", pair, "BDP returned no forward data")
            return {}
        except Exception as e:
            logger.error(f"FX forward curve BBG request failed: {e}")
            _log_fetch_failure("get_fx_forward_curve", pair, str(e))
            return {}

    # SYNTHETIC mode only — Bloomberg not connected
    curve = _fallback_forward_curve(pair)
    _cache_set(ck, curve, "forwards")
    return curve


def get_fx_historical_spot(pair: str, days: int = 252) -> pd.DataFrame:
    """
    Historical OHLC spot data.
    Returns DataFrame with columns: open, high, low, close.
    """
    ck = f"histspot_{pair}_{days}"
    cached = _cache_get(ck, "historical")
    if cached is not None:
        return cached

    if _HAS_EQUITY_BBG and is_connected():
        try:
            start = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
            ticker = _fx_bbg_ticker(pair)
            fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"]
            df = bdh(ticker, fields, start)
            if not df.empty:
                df = df.rename(columns={
                    "PX_OPEN": "open", "PX_HIGH": "high",
                    "PX_LOW": "low", "PX_LAST": "close",
                })
                df = df.tail(days)
                _cache_set(ck, df, "historical")
                return df
            _log_fetch_failure("get_fx_historical_spot", pair, "BDH returned empty dataframe")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"FX historical spot BBG request failed: {e}")
            _log_fetch_failure("get_fx_historical_spot", pair, str(e))
            return pd.DataFrame()

    # SYNTHETIC mode only — Bloomberg not connected
    df = _generate_spot_history(pair, days)
    _cache_set(ck, df, "historical")
    return df


def get_fx_historical_vol(pair: str, tenor: str = "1M",
                          metric: str = "atm", days: int = 252) -> pd.Series:
    """Historical vol time series for a given tenor/metric."""
    ck = f"histvol_{pair}_{tenor}_{metric}_{days}"
    cached = _cache_get(ck, "historical")
    if cached is not None:
        return cached

    if _HAS_EQUITY_BBG and is_connected():
        try:
            # Build the correct ticker using registry prefixes
            from core.fx_conventions import FX_PAIR_REGISTRY
            m = metric.upper() if isinstance(metric, str) else "ATM"
            pair_u = pair.upper()
            spec = FX_PAIR_REGISTRY.get(pair_u)
            tc = tenor.upper()
            # Map metric names from both conventions: "rr25"/"RR25" and "25D_RR"
            if m in ("ATM", ""):
                pfx = spec.bb_vol_prefix if spec else f"{pair_u}V"
            elif m in ("25D_RR", "RR25"):
                pfx = spec.bb_rr25_prefix if spec else f"{pair_u}25R"
            elif m in ("25D_BF", "BF25"):
                pfx = spec.bb_bf25_prefix if spec else f"{pair_u}25B"
            elif m in ("10D_RR", "RR10"):
                pfx = spec.bb_rr10_prefix if spec else f"{pair_u}10R"
            elif m in ("10D_BF", "BF10"):
                pfx = spec.bb_bf10_prefix if spec else f"{pair_u}10B"
            else:
                pfx = spec.bb_vol_prefix if spec else f"{pair_u}V"
            ticker = f"{pfx}{tc} Curncy"

            start = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
            df = bdh(ticker, ["PX_LAST", "PX_MID"], start)
            if not df.empty:
                # Use PX_LAST if available, else PX_MID
                col = "PX_LAST" if "PX_LAST" in df.columns and df["PX_LAST"].notna().any() else "PX_MID"
                if col in df.columns:
                    series = df[col].tail(days)
                    series.name = f"{pair}_{tenor}_{metric}"
                    _cache_set(ck, series, "historical")
                    return series
            _log_fetch_failure("get_fx_historical_vol", f"{pair}/{tenor}/{metric}", "BDH returned empty")
            return pd.Series(dtype=float)
        except Exception as e:
            logger.error(f"FX historical vol BBG request failed: {e}")
            _log_fetch_failure("get_fx_historical_vol", f"{pair}/{tenor}/{metric}", str(e))
            return pd.Series(dtype=float)

    # SYNTHETIC mode only — Bloomberg not connected
    series = _generate_vol_history(pair, tenor, metric, days)
    _cache_set(ck, series, "historical")
    return series


def get_fx_option_chain(pair: str, tenor: str = "1M") -> pd.DataFrame:
    """
    FX options chain for a pair and tenor.
    Returns DataFrame: strike, type, bid, ask, mid, iv, delta, gamma, vega, theta, oi.
    """
    ck = f"optchain_{pair}_{tenor}"
    cached = _cache_get(ck, "vol_surface")
    if cached is not None:
        return cached

    if _HAS_EQUITY_BBG and is_connected():
        try:
            ticker = _fx_bbg_ticker(pair)
            overrides = {"OPT_CHAIN_TENOR": tenor.upper()}
            chain_df = bds(ticker, "OPT_CHAIN", **overrides)
            if not chain_df.empty:
                opt_tickers = chain_df.iloc[:, 0].tolist()
                fields = ["PX_BID", "PX_ASK", "PX_LAST", "IVOL_MID", "DELTA",
                           "GAMMA", "THETA", "VEGA", "OPEN_INT", "STRIKE_PX",
                           "OPT_PUT_CALL"]
                df = bdp(opt_tickers, fields)
                df = df.reset_index().rename(columns={
                    "security": "option_ticker", "STRIKE_PX": "strike",
                    "OPT_PUT_CALL": "type", "PX_LAST": "mid", "PX_BID": "bid",
                    "PX_ASK": "ask", "IVOL_MID": "iv", "DELTA": "delta",
                    "GAMMA": "gamma", "THETA": "theta", "VEGA": "vega",
                    "OPEN_INT": "oi",
                })
                _cache_set(ck, df, "vol_surface")
                return df
            logger.warning(f"FX OPT_CHAIN empty for {pair}/{tenor}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"FX option chain BBG request failed: {e}")
            _log_fetch_failure("get_fx_option_chain", pair, str(e))
            return pd.DataFrame()

    # SYNTHETIC mode only — Bloomberg not connected
    df = _fallback_option_chain(pair, tenor)
    _cache_set(ck, df, "vol_surface")
    return df


def get_fx_deposit_rates(ccy: str, tenors: List[str] = None) -> Dict[str, float]:
    """Deposit rate curve for a currency."""
    if tenors is None:
        tenors = ["1M", "3M", "6M", "1Y"]
    return get_fx_rate_curve(ccy)


def get_cftc_positioning(pair: str) -> dict:
    """
    CFTC Commitments of Traders positioning data.
    Returns {net_spec, commercial, non_reportable, total_oi}.
    """
    ck = f"cftc_{pair}"
    cached = _cache_get(ck, "positioning")
    if cached is not None:
        return cached

    if _HAS_EQUITY_BBG and is_connected():
        # CFTC data not available via Bloomberg real-time — return empty
        return {}

    # SYNTHETIC mode only — Bloomberg not connected
    result = _fallback_positioning(pair)
    _cache_set(ck, result, "positioning")
    return result


def get_fx_realized_vol(pair: str, window: int = 20,
                        days: int = 252) -> pd.Series:
    """
    Historical realized volatility series.
    Computed as annualized stdev of log returns over rolling window.
    """
    hist = get_fx_historical_spot(pair, days + window)
    if hist.empty or len(hist) < window + 1:
        return pd.Series(dtype=float)
    log_ret = np.log(hist["close"] / hist["close"].shift(1)).dropna()
    rv = log_ret.rolling(window).std() * np.sqrt(252) * 100.0
    rv.name = f"{pair}_RV{window}"
    return rv.dropna()


def get_fx_correlation(pair_a: str, pair_b: str, window: int = 60,
                       days: int = 252) -> pd.Series:
    """Rolling correlation between two FX pairs."""
    ha = get_fx_historical_spot(pair_a, days + window)
    hb = get_fx_historical_spot(pair_b, days + window)
    if ha.empty or hb.empty:
        return pd.Series(dtype=float)
    # Use positional alignment (not index) to avoid DatetimeIndex mismatch
    ca = ha["close"].values
    cb = hb["close"].values
    n = min(len(ca), len(cb))
    ra = pd.Series(np.diff(np.log(ca[-n:])))
    rb = pd.Series(np.diff(np.log(cb[-n:])))
    if len(ra) < window + 1:
        return pd.Series(dtype=float)
    corr = ra.rolling(window).corr(rb)
    corr.name = f"corr_{pair_a}_{pair_b}"
    return corr.dropna()


def get_fx_implied_correlation(pair_a: str, pair_b: str, cross: str,
                               tenor: str = "1M") -> float:
    """
    Triangle implied correlation from vol of cross vs two legs.

    sigma_cross^2 = sigma_a^2 + sigma_b^2 - 2*rho*sigma_a*sigma_b
    => rho = (sigma_a^2 + sigma_b^2 - sigma_cross^2) / (2*sigma_a*sigma_b)
    """
    va = get_fx_vol_point(pair_a, tenor, "atm")
    vb = get_fx_vol_point(pair_b, tenor, "atm")
    vc = get_fx_vol_point(cross, tenor, "atm")
    if not va or not vb or not vc or va <= 0 or vb <= 0:
        return 0.0
    rho = (va**2 + vb**2 - vc**2) / (2.0 * va * vb)
    return round(max(min(rho, 1.0), -1.0), 4)


def get_central_bank_dates(bank: str) -> List[dict]:
    """
    Upcoming central bank meeting dates with consensus.
    Returns [{date, bank, consensus, prev_rate}].
    """
    _CB_SCHEDULE = {
        "FED": {"dates": [7, 14, 21, 49, 77, 112, 147, 175],
                "rate": 5.30, "consensus": "hold"},
        "ECB": {"dates": [10, 42, 70, 105, 140, 175],
                "rate": 3.90, "consensus": "-25bp"},
        "BOJ": {"dates": [12, 56, 91, 140, 182],
                "rate": -0.10, "consensus": "hold"},
        "BOE": {"dates": [14, 49, 84, 119, 154, 182],
                "rate": 5.20, "consensus": "hold"},
        "RBA": {"dates": [8, 42, 77, 112, 147, 175],
                "rate": 4.35, "consensus": "hold"},
        "RBNZ": {"dates": [21, 63, 105, 147, 182],
                 "rate": 5.50, "consensus": "hold"},
        "BOC": {"dates": [10, 42, 77, 112, 147, 175],
                "rate": 5.00, "consensus": "-25bp"},
        "SNB": {"dates": [35, 91, 147, 182],
                "rate": 1.75, "consensus": "hold"},
        "RIKSBANK": {"dates": [28, 70, 119, 168],
                     "rate": 4.00, "consensus": "-25bp"},
        "NORGES": {"dates": [21, 63, 112, 161],
                   "rate": 4.50, "consensus": "hold"},
        "CBRT": {"dates": [14, 42, 70, 98, 126, 154, 182],
                 "rate": 45.00, "consensus": "hold"},
        "SARB": {"dates": [21, 77, 133, 175],
                 "rate": 8.25, "consensus": "hold"},
        "BANXICO": {"dates": [14, 56, 98, 140, 175],
                    "rate": 11.25, "consensus": "-25bp"},
    }
    now = datetime.now()
    info = _CB_SCHEDULE.get(bank.upper(), {"dates": [30, 90, 150],
                                            "rate": 3.0, "consensus": "hold"})
    meetings = []
    for d in info["dates"]:
        meeting_date = now + timedelta(days=d)
        meetings.append({
            "date": meeting_date.strftime("%Y-%m-%d"),
            "bank": bank.upper(),
            "consensus": info["consensus"],
            "prev_rate": info["rate"],
        })
    return meetings


# ═══════════════════════════════════════════════════════════════════════════
# Fallback Data Generators
# ═══════════════════════════════════════════════════════════════════════════

def _fallback_spot(pair: str) -> dict:
    """Generate realistic spot data from calibrated parameters."""
    fb = _FX_FALLBACK.get(pair.upper())
    if fb is None:
        fb = {"spot": 1.0, "daily_vol": 0.06}
    rng = np.random.RandomState(_pair_seed(pair))
    spot = fb["spot"]
    dv = fb["daily_vol"] / 100.0  # daily_vol is annualized pct / sqrt(252)
    change = spot * rng.normal(0, dv)

    # Compute realistic spread (tighter for majors, wider for EM)
    if spot > 100:
        spread = spot * 0.0003  # JPY, KRW
    elif spot > 10:
        spread = spot * 0.0004  # NOK, SEK, MXN, ZAR, TRY
    elif spot > 5:
        spread = spot * 0.0003  # CNH, BRL
    else:
        spread = spot * 0.0001  # G10 majors

    mid = round(spot + change, 6)
    return {
        "bid": round(mid - spread / 2, 6),
        "ask": round(mid + spread / 2, 6),
        "mid": mid,
        "change": round(change, 6),
        "change_pct": round(change / spot * 100, 4),
        "high": round(mid + abs(change) * 1.2, 6),
        "low": round(mid - abs(change) * 1.3, 6),
        "open": round(spot - change * 0.4, 6),
        "volume_ind": int(rng.lognormal(8, 1.5)),
    }


def _fallback_vol_surface(pair: str) -> Dict[str, dict]:
    """Return the calibrated vol surface for a pair."""
    fb = _FX_FALLBACK.get(pair.upper())
    if fb is None:
        return _make_vol_surface(8.0, -0.3, 0.25, -0.65, 0.45)
    return fb["vol_surface"]


def _fallback_rates(pair: str) -> dict:
    """Calibrated domestic/foreign rate pair."""
    fb = _FX_FALLBACK.get(pair.upper())
    if fb is None:
        return {"r_dom": 0.05, "r_for": 0.03, "rate_diff": 0.02}
    return {
        "r_dom": fb["r_dom"],
        "r_for": fb["r_for"],
        "rate_diff": round(fb["r_dom"] - fb["r_for"], 4),
    }


def _fallback_rate_curve(ccy: str) -> Dict[str, float]:
    """Synthetic deposit rate curve for a currency."""
    # Find a pair that uses this currency to get a base rate
    # FX convention: pair[:3] = base/foreign ccy → rate is r_for
    #                pair[3:] = quote/domestic ccy → rate is r_dom
    base_rate = 0.05
    for pair, fb in _FX_FALLBACK.items():
        if pair[:3] == ccy.upper():
            base_rate = fb["r_for"]
            break
        elif pair[3:] == ccy.upper():
            base_rate = fb["r_dom"]
            break

    rng = np.random.RandomState(_pair_seed(ccy, extra=42))
    # Upward-sloping curve with small noise
    tenors = {"1M": 0.97, "3M": 1.00, "6M": 1.02, "1Y": 1.04,
              "2Y": 1.06, "3Y": 1.07, "5Y": 1.08}
    curve = {}
    for t, mult in tenors.items():
        noise = rng.normal(0, 0.001)
        curve[t] = round(base_rate * mult + noise, 4)
    return curve


def _fallback_forward_curve(pair: str) -> Dict[str, dict]:
    """Synthetic forward points from rate differential."""
    fb = _FX_FALLBACK.get(pair.upper())
    if fb is None:
        fb = {"spot": 1.0, "r_dom": 0.05, "r_for": 0.03}
    spot = fb["spot"]
    r_d = fb["r_dom"]
    r_f = fb["r_for"]
    curve = {}
    for tenor in _ALL_TENORS:
        ty = tenor_to_years(tenor)
        # Interest rate parity: F = S * exp((r_dom - r_for) * T)
        outright = spot * np.exp((r_d - r_f) * ty)
        from core.fx_conventions import FX_PAIR_REGISTRY
        fwd_spec = FX_PAIR_REGISTRY.get(pair.upper())
        pip_size = fwd_spec.pip if fwd_spec else 0.0001
        fwd_pts = (outright - spot) / pip_size
        implied_diff = np.log(outright / spot) / ty if ty > 0 else 0.0
        curve[tenor] = {
            "fwd_points": round(fwd_pts, 2),
            "outright": round(outright, 6),
            "implied_rate_diff": round(implied_diff, 4),
        }
    return curve


def _fallback_option_chain(pair: str, tenor: str = "1M") -> pd.DataFrame:
    """Generate synthetic FX option chain from vol surface."""
    fb = _FX_FALLBACK.get(pair.upper())
    if fb is None:
        fb = {"spot": 1.0, "r_dom": 0.05, "r_for": 0.03,
              "vol_surface": _make_vol_surface(8.0, -0.3, 0.25, -0.65, 0.45)}

    spot = fb["spot"]
    r_d = fb["r_dom"]
    r_f = fb["r_for"]
    T = tenor_to_years(tenor)
    vol_data = fb["vol_surface"].get(tenor, fb["vol_surface"].get("1M"))
    atm_vol = vol_data["atm"] / 100.0

    rng = np.random.RandomState(_pair_seed(pair, extra=100))

    # Generate strikes around the ATM forward
    fwd = spot * np.exp((r_d - r_f) * T)
    if spot > 100:
        step = max(0.5, round(spot * 0.005, 1))
    elif spot > 10:
        step = max(0.05, round(spot * 0.005, 2))
    else:
        step = max(0.005, round(spot * 0.005, 4))

    n_strikes = 15
    strikes = np.array([fwd + (i - n_strikes // 2) * step
                        for i in range(n_strikes)])
    strikes = np.round(strikes, 4)

    rows = []
    for K in strikes:
        for opt_type in ["call", "put"]:
            # Simple smile parameterization from vol surface data
            moneyness = np.log(K / fwd) / (atm_vol * np.sqrt(T)) if T > 0 else 0
            rr = vol_data["rr25"] / 100.0
            bf = vol_data["bf25"] / 100.0
            iv = atm_vol + rr * moneyness * 0.5 + bf * moneyness ** 2 * 0.5
            iv = max(iv, atm_vol * 0.4)

            # Garman-Kohlhagen pricing
            d1 = (np.log(spot / K) + (r_d - r_f + 0.5 * iv**2) * T) / (iv * np.sqrt(T)) if T > 0 else 0
            d2 = d1 - iv * np.sqrt(T)
            from scipy.stats import norm as sp_norm
            if opt_type == "call":
                price = spot * np.exp(-r_f * T) * sp_norm.cdf(d1) - K * np.exp(-r_d * T) * sp_norm.cdf(d2)
                delta = np.exp(-r_f * T) * sp_norm.cdf(d1)
            else:
                price = K * np.exp(-r_d * T) * sp_norm.cdf(-d2) - spot * np.exp(-r_f * T) * sp_norm.cdf(-d1)
                delta = -np.exp(-r_f * T) * sp_norm.cdf(-d1)

            gamma = np.exp(-r_f * T) * sp_norm.pdf(d1) / (spot * iv * np.sqrt(T)) if T > 0 else 0
            vega = spot * np.exp(-r_f * T) * sp_norm.pdf(d1) * np.sqrt(T) / 100.0
            theta = -(spot * iv * np.exp(-r_f * T) * sp_norm.pdf(d1)) / (2 * np.sqrt(T)) / 365.0 if T > 0 else 0

            spread = price * rng.uniform(0.02, 0.06)
            oi = int(rng.lognormal(6, 1.5))

            rows.append({
                "strike": K,
                "type": opt_type,
                "bid": round(max(price - spread / 2, 0), 6),
                "ask": round(price + spread / 2, 6),
                "mid": round(price, 6),
                "iv": round(iv * 100, 2),
                "delta": round(delta, 4),
                "gamma": round(gamma, 6),
                "vega": round(vega, 6),
                "theta": round(theta, 6),
                "oi": oi,
            })

    return pd.DataFrame(rows)


def _fallback_positioning(pair: str) -> dict:
    """Synthetic CFTC-like positioning data."""
    rng = np.random.RandomState(_pair_seed(pair, extra=200))
    total_oi = int(rng.lognormal(11, 0.8))
    net_spec = int(rng.normal(0, total_oi * 0.15))
    commercial = -net_spec + int(rng.normal(0, total_oi * 0.05))
    non_rep = -(net_spec + commercial)
    return {
        "net_spec": net_spec,
        "commercial": commercial,
        "non_reportable": non_rep,
        "total_oi": total_oi,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic History Generators
# ═══════════════════════════════════════════════════════════════════════════

def _generate_spot_history(pair: str, days: int = 252,
                           seed: int = None) -> pd.DataFrame:
    """
    Correlated GBM spot history using pair-specific parameters.

    Uses Geometric Brownian Motion with drift calibrated from rate
    differential and daily volatility from the fallback table.
    """
    if seed is None:
        seed = _pair_seed(pair)
    rng = np.random.RandomState(seed)
    fb = _FX_FALLBACK.get(pair.upper())
    if fb is None:
        fb = {"spot": 1.0, "r_dom": 0.05, "r_for": 0.03, "daily_vol": 0.06}

    spot = fb["spot"]
    ann_vol = fb["daily_vol"]
    drift = (fb["r_dom"] - fb["r_for"]) / 252.0
    daily_sigma = ann_vol / np.sqrt(252)

    # Generate log returns with small mean-reversion toward current spot
    log_prices = np.zeros(days)
    log_prices[0] = np.log(spot)
    mr_strength = 0.002  # gentle mean reversion

    for i in range(1, days):
        deviation = log_prices[i - 1] - np.log(spot)
        mr = -mr_strength * deviation
        log_prices[i] = (log_prices[i - 1] + drift + mr
                         + daily_sigma * rng.normal())

    prices = np.exp(log_prices)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    daily_range = daily_sigma * prices
    return pd.DataFrame({
        "open": np.round(prices * (1 + rng.uniform(-0.002, 0.002, days)), 6),
        "high": np.round(prices + np.abs(daily_range * rng.uniform(0.3, 1.2, days)), 6),
        "low": np.round(prices - np.abs(daily_range * rng.uniform(0.3, 1.2, days)), 6),
        "close": np.round(prices, 6),
    }, index=dates)


def _generate_vol_history(pair: str, tenor: str = "1M",
                          metric: str = "atm", days: int = 252,
                          seed: int = None) -> pd.Series:
    """
    Mean-reverting Ornstein-Uhlenbeck vol time series.

    vol(t+1) = vol(t) + mr_speed * (mean - vol(t)) + vol_of_vol * N(0,1)

    Parameters are calibrated per pair from vol_history_params.
    """
    if seed is None:
        seed = _pair_seed(pair, extra=300 + hash(tenor) % 1000)
    rng = np.random.RandomState(seed)

    fb = _FX_FALLBACK.get(pair.upper())
    if fb is None:
        params = {"mean": 8.0, "std": 1.5, "mr_speed": 0.03, "vol_of_vol": 0.8}
    else:
        params = fb["vol_history_params"]

    # Adjust mean based on metric
    vol_surface = fb["vol_surface"] if fb else _make_vol_surface(8.0, -0.3, 0.25, -0.65, 0.45)
    tenor_data = vol_surface.get(tenor, vol_surface.get("1M", {}))
    base_level = tenor_data.get(metric, params["mean"])
    if metric != "atm":
        # RR and BF fluctuate around their calibrated level
        mean_val = base_level
        vol_of_vol = params["vol_of_vol"] * 0.3
    else:
        mean_val = base_level
        vol_of_vol = params["vol_of_vol"]

    mr_speed = params["mr_speed"]

    series = np.zeros(days)
    series[0] = mean_val + rng.normal(0, params["std"] * 0.3)

    for i in range(1, days):
        innovation = vol_of_vol * rng.normal() / np.sqrt(252)
        series[i] = series[i - 1] + mr_speed * (mean_val - series[i - 1]) + innovation

    # Ensure vols stay positive for ATM; RR/BF can be negative
    if metric == "atm":
        series = np.maximum(series, 0.5)
    elif metric.startswith("bf"):
        series = np.maximum(series, 0.01)

    dates = pd.date_range(end=datetime.now(), periods=len(series), freq="B")
    result = pd.Series(np.round(series, 2), index=dates,
                       name=f"{pair}_{tenor}_{metric}")
    return result


def _generate_correlated_histories(pairs: List[str], days: int = 252,
                                   seed: int = None) -> Dict[str, pd.DataFrame]:
    """
    Multi-pair correlated spot simulation using Cholesky decomposition.

    Builds a correlation matrix from _FX_CORRELATION, applies Cholesky
    decomposition, and generates correlated normal innovations for GBM.
    """
    if seed is None:
        seed = _pair_seed("_".join(sorted(pairs)))
    rng = np.random.RandomState(seed)
    n = len(pairs)

    # Build correlation matrix
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = _get_correlation(pairs[i], pairs[j])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # Ensure positive semi-definite via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-6)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize diagonal to 1.0
    d = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(d, d)

    # Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)

    # Generate correlated normals
    Z = rng.standard_normal((days, n))
    corr_Z = Z @ L.T

    results = {}
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    for idx, pair in enumerate(pairs):
        fb = _FX_FALLBACK.get(pair.upper())
        if fb is None:
            fb = {"spot": 1.0, "r_dom": 0.05, "r_for": 0.03, "daily_vol": 0.06}
        spot = fb["spot"]
        drift = (fb["r_dom"] - fb["r_for"]) / 252.0
        daily_sigma = fb["daily_vol"] / np.sqrt(252)

        log_prices = np.zeros(days)
        log_prices[0] = np.log(spot)
        for i in range(1, days):
            log_prices[i] = (log_prices[i - 1] + drift
                             + daily_sigma * corr_Z[i, idx])

        prices = np.exp(log_prices)
        daily_range = daily_sigma * prices
        results[pair] = pd.DataFrame({
            "open": np.round(prices * (1 + rng.uniform(-0.002, 0.002, days)), 6),
            "high": np.round(prices + np.abs(daily_range * rng.uniform(0.3, 1.2, days)), 6),
            "low": np.round(prices - np.abs(daily_range * rng.uniform(0.3, 1.2, days)), 6),
            "close": np.round(prices, 6),
        }, index=dates)

    return results


def _generate_rate_history(ccy: str, days: int = 252,
                           seed: int = None) -> pd.DataFrame:
    """
    Slowly drifting rate curve history for a currency.

    Rates follow a slow random walk with central bank jump events and
    mean reversion toward the calibrated base rate.
    """
    if seed is None:
        seed = _pair_seed(ccy, extra=500)
    rng = np.random.RandomState(seed)

    # Find base rate for this currency
    # FX convention: pair[:3] = base/foreign ccy → rate is r_for
    #                pair[3:] = quote/domestic ccy → rate is r_dom
    base_rate = 0.05
    for pair, fb in _FX_FALLBACK.items():
        if pair[:3] == ccy.upper():
            base_rate = fb["r_for"]
            break
        elif pair[3:] == ccy.upper():
            base_rate = fb["r_dom"]
            break

    tenors = ["1M", "3M", "6M", "1Y", "2Y", "5Y"]
    tenor_spreads = [0.97, 1.00, 1.02, 1.04, 1.06, 1.08]

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    data = {}

    for tenor, spread in zip(tenors, tenor_spreads):
        rates = np.zeros(days)
        rates[0] = base_rate * spread + rng.normal(0, 0.002)

        mr_speed = 0.01
        daily_vol = 0.0008

        # Insert occasional "jump" days simulating CB meetings
        jump_days = rng.choice(days, size=max(1, days // 45), replace=False)
        jumps = np.zeros(days)
        for jd in jump_days:
            jumps[jd] = rng.choice([-0.0025, 0, 0, 0.0025])

        for i in range(1, days):
            rates[i] = (rates[i - 1]
                        + mr_speed * (base_rate * spread - rates[i - 1])
                        + daily_vol * rng.normal()
                        + jumps[i])
            rates[i] = max(rates[i], -0.01)  # floor at -1%

        data[tenor] = np.round(rates, 4)

    return pd.DataFrame(data, index=dates)


def _generate_positioning_history(pair: str, days: int = 252,
                                  seed: int = None) -> pd.DataFrame:
    """
    Mean-reverting CFTC-style positioning history.

    Positioning oscillates around zero with pair-specific amplitude,
    mimicking weekly CFTC releases (values change every 5 business days).
    """
    if seed is None:
        seed = _pair_seed(pair, extra=700)
    rng = np.random.RandomState(seed)

    fb = _FX_FALLBACK.get(pair.upper())
    # Scale amplitude by how "liquid" the pair is
    if fb is None or fb["daily_vol"] > 0.10:
        amplitude = 30000
    elif fb["daily_vol"] > 0.07:
        amplitude = 60000
    else:
        amplitude = 120000

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Generate weekly data (every 5 business days) and forward-fill
    weekly_points = days // 5 + 1
    net_spec = np.zeros(weekly_points)
    net_spec[0] = rng.normal(0, amplitude * 0.3)
    mr_speed = 0.08

    for i in range(1, weekly_points):
        net_spec[i] = (net_spec[i - 1]
                       + mr_speed * (0 - net_spec[i - 1])
                       + amplitude * 0.15 * rng.normal())

    net_spec = np.clip(net_spec, -amplitude, amplitude).astype(int)

    # Expand to daily by forward-filling
    daily_net = np.repeat(net_spec, 5)[:days]
    commercial = -daily_net + (rng.normal(0, amplitude * 0.05, days)).astype(int)
    non_rep = -(daily_net + commercial)
    total_oi = np.abs(daily_net) + np.abs(commercial) + np.abs(non_rep)

    return pd.DataFrame({
        "net_spec": daily_net,
        "commercial": commercial,
        "non_reportable": non_rep,
        "total_oi": total_oi,
    }, index=dates)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience & Aggregate Functions
# ═══════════════════════════════════════════════════════════════════════════

def get_all_pairs() -> List[str]:
    """Return the list of all supported FX pairs."""
    return list(_ALL_PAIRS)


def get_fx_snapshot(pair: str) -> dict:
    """
    Complete snapshot for a single pair: spot, vol surface, rates, forwards.
    Useful for populating a full pricer panel.
    """
    spot_data = get_fx_spots([pair]).get(pair, {})
    vol_surface = get_fx_vol_surface(pair)
    rates = get_fx_rates(pair)
    fwd_curve = get_fx_forward_curve(pair)
    positioning = get_cftc_positioning(pair)
    return {
        "pair": pair,
        "spot": spot_data,
        "vol_surface": vol_surface,
        "rates": rates,
        "forwards": fwd_curve,
        "positioning": positioning,
    }


def get_fx_board(pairs: List[str] = None) -> pd.DataFrame:
    """
    FX quote board — spots plus 1M ATM vol for multiple pairs.
    Returns DataFrame suitable for display.
    """
    if pairs is None:
        pairs = _ALL_PAIRS
    spots = get_fx_spots(pairs)
    rows = []
    for pair in pairs:
        sd = spots.get(pair, {})
        vol_1m = get_fx_vol_point(pair, "1M", "atm")
        rates = get_fx_rates(pair) or {}
        rows.append({
            "pair": pair,
            "bid": sd.get("bid", 0),
            "ask": sd.get("ask", 0),
            "mid": sd.get("mid", 0),
            "chg": sd.get("change", 0),
            "chg_pct": sd.get("change_pct", 0),
            "high": sd.get("high", 0),
            "low": sd.get("low", 0),
            "vol_1m": vol_1m or 0,
            "r_dom": rates.get("r_dom", 0),
            "r_for": rates.get("r_for", 0),
        })
    return pd.DataFrame(rows)


def get_fx_vol_matrix(pairs: List[str] = None,
                      tenors: List[str] = None) -> pd.DataFrame:
    """
    ATM vol matrix: pairs x tenors. Returns DataFrame.
    """
    if pairs is None:
        pairs = _ALL_PAIRS[:10]
    if tenors is None:
        tenors = ["1W", "1M", "3M", "6M", "1Y"]

    data = {}
    for pair in pairs:
        surface = get_fx_vol_surface(pair)
        row = {}
        for tenor in tenors:
            row[tenor] = surface.get(tenor, {}).get("atm", 0)
        data[pair] = row
    return pd.DataFrame(data).T


def get_fx_correlation_matrix(pairs: List[str] = None) -> pd.DataFrame:
    """
    Cross-pair correlation matrix using static calibrated correlations.
    Returns DataFrame with pairs as both index and columns.
    """
    if pairs is None:
        pairs = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD",
                 "NZDUSD", "USDCAD"]
    n = len(pairs)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = _get_correlation(pairs[i], pairs[j])
            matrix[i, j] = rho
            matrix[j, i] = rho
    return pd.DataFrame(matrix, index=pairs, columns=pairs)


def get_fx_risk_reversal_monitor(pairs: List[str] = None,
                                 tenor: str = "1M") -> pd.DataFrame:
    """
    Risk reversal monitor: 25D and 10D RR for multiple pairs.
    Useful for skew monitoring.
    """
    if pairs is None:
        pairs = _ALL_PAIRS
    rows = []
    for pair in pairs:
        surface = get_fx_vol_surface(pair)
        td = surface.get(tenor, {})
        rows.append({
            "pair": pair,
            "atm": td.get("atm", 0),
            "rr25": td.get("rr25", 0),
            "rr10": td.get("rr10", 0),
            "bf25": td.get("bf25", 0),
            "bf10": td.get("bf10", 0),
        })
    return pd.DataFrame(rows)


def get_fx_carry_rankings(pairs: List[str] = None) -> pd.DataFrame:
    """
    Rank pairs by carry (rate differential) for carry trade analysis.
    """
    if pairs is None:
        pairs = _ALL_PAIRS
    rows = []
    for pair in pairs:
        rates = get_fx_rates(pair)
        if not rates:
            continue
        rows.append({
            "pair": pair,
            "r_dom": rates.get("r_dom", 0),
            "r_for": rates.get("r_for", 0),
            "carry": round(rates.get("rate_diff", 0) * 100, 2),
        })
    df = pd.DataFrame(rows).sort_values("carry", ascending=False)
    df = df.reset_index(drop=True)
    return df


def get_fx_term_structure(pair: str) -> pd.DataFrame:
    """
    Full vol term structure for a single pair: all tenors with all metrics.
    """
    surface = get_fx_vol_surface(pair)
    rows = []
    for tenor in _ALL_TENORS:
        td = surface.get(tenor, {})
        rows.append({
            "tenor": tenor,
            "days": _TENOR_DAYS[tenor],
            "atm": td.get("atm", 0),
            "rr25": td.get("rr25", 0),
            "bf25": td.get("bf25", 0),
            "rr10": td.get("rr10", 0),
            "bf10": td.get("bf10", 0),
            "25d_call": round(td.get("atm", 0) + td.get("rr25", 0) / 2
                              + td.get("bf25", 0), 2),
            "25d_put": round(td.get("atm", 0) - td.get("rr25", 0) / 2
                             + td.get("bf25", 0), 2),
        })
    return pd.DataFrame(rows)
