"""
Data Fetch Helpers
==================
Safe wrappers around bloomberg_fx functions that handle the try/except/None
pattern once, so panels don't each need to repeat it.
"""

import logging
from core.bloomberg_fx import (
    get_fx_spots, get_fx_vol_surface, get_fx_rates,
    get_fx_historical_spot, get_fx_historical_vol,
)

logger = logging.getLogger(__name__)


def fetch_spots_safe(pairs):
    """Fetch spot rates for a list of pairs. Returns dict (may be empty)."""
    try:
        result = get_fx_spots(pairs)
        return result if result else {}
    except Exception as e:
        logger.error("Spot fetch failed for %s: %s", pairs, e)
        return {}


def fetch_vol_surface_safe(pair):
    """Fetch vol surface for a pair. Returns dict or None."""
    try:
        result = get_fx_vol_surface(pair)
        return result if result else None
    except Exception as e:
        logger.error("Vol surface fetch failed for %s: %s", pair, e)
        return None


def fetch_rates_safe(pair):
    """Fetch interest rates for a pair. Returns dict or None."""
    try:
        result = get_fx_rates(pair)
        return result if result else None
    except Exception as e:
        logger.error("Rates fetch failed for %s: %s", pair, e)
        return None


def fetch_spot_and_rates(pair):
    """Convenience: fetch spot + rates together. Returns (spot, r_dom, r_for) tuple.

    spot is a float (or None if unavailable).
    r_dom, r_for are floats (default 0.0 if unavailable).
    """
    spots = fetch_spots_safe([pair])
    spot_data = spots.get(pair, {})
    spot = spot_data.get("mid") or spot_data.get("last")

    rates = fetch_rates_safe(pair) or {}
    r_dom = rates.get("r_dom", 0.0)
    r_for = rates.get("r_for", 0.0)
    return spot, r_dom, r_for


def fetch_historical_spot_safe(pair, days=800):
    """Fetch historical spot data. Returns DataFrame or None."""
    try:
        result = get_fx_historical_spot(pair, days=days)
        if result is not None and not result.empty:
            return result
        return None
    except Exception as e:
        logger.error("Historical spot fetch failed for %s: %s", pair, e)
        return None


def fetch_historical_vol_safe(pair, tenor, metric, days=800):
    """Fetch historical vol data. Returns Series or None."""
    try:
        result = get_fx_historical_vol(pair, tenor, metric, days)
        if result is not None and len(result) > 0:
            return result
        return None
    except Exception as e:
        logger.error("Historical vol fetch failed for %s %s %s: %s", pair, tenor, metric, e)
        return None
