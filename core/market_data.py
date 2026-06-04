"""
core/market_data.py — Strict, no-fabrication market-data access.

This module enforces ONE rule for the whole workstation:

    A number shown to a trader is EITHER a real Bloomberg value (or an
    analytic derived purely from real Bloomberg values), OR it is visibly
    marked unavailable.  No plausible-looking placeholder — no default
    spot of 1.0, no "typical" 8% vol, no 4%/2% rates — may ever reach the
    screen dressed up as market data.

Accessors come in two flavours:

  * require_*  -> returns a real value or raises MarketDataUnavailable.
                 Use when the whole computation is meaningless without it;
                 the caller catches the exception and renders NO DATA.

  * get_*      -> returns a real value or None.  Use when one missing input
                 should blank a single cell, not the whole view.

NEVER add a ``default=`` parameter that returns a fabricated market level.
The only acceptable defaults are for genuinely cosmetic, non-market values.

All getters read exclusively through ``core.bloomberg_fx`` (the cache layer),
which itself returns empty on failure and never fabricates.  This module
adds the discipline of refusing to invent a value when the cache is empty.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical display token for a value that is genuinely unavailable.
# Use this everywhere a missing number would otherwise be rendered, so the
# trader always sees the same unambiguous "no data" mark.
NA = "—"


class MarketDataUnavailable(Exception):
    """Raised when a required real market input is missing.

    Callers MUST NOT swallow this by substituting a default value.  They
    should surface it: render :data:`NA`, a NO-DATA figure, or a degraded
    badge.  Letting this propagate is the correct, honest behaviour.
    """

    def __init__(self, kind: str, pair: Optional[str] = None,
                 tenor: Optional[str] = None, detail: str = ""):
        self.kind = kind
        self.pair = pair
        self.tenor = tenor
        self.detail = detail
        loc = " ".join(x for x in (pair, tenor) if x)
        msg = f"{kind} unavailable"
        if loc:
            msg += f" for {loc}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


def normalize_vol(raw) -> Optional[float]:
    """Normalize a raw ATM quote to a decimal vol, or ``None`` if not real.

    Bloomberg quotes vols in points (e.g. ``8.5`` == 8.5%); this returns
    ``0.085``.  A value already in decimal form (``0.085``) is passed
    through.  Returns ``None`` for ``None`` / non-numeric / non-positive
    inputs — which signals *"no real vol"*, NOT *"zero vol"*.  This is the
    single place that distinguishes a true 0 from a missing quote, so no
    caller is tempted to treat a missing surface as a flat 0 smile.
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v / 100.0 if v > 1.0 else v


# ---------------------------------------------------------------------------
# Spot
# ---------------------------------------------------------------------------

def get_spot(pair: str) -> Optional[float]:
    """Return the real mid spot for ``pair``, or ``None`` if unavailable."""
    try:
        from core.bloomberg_fx import get_fx_spots
        data = (get_fx_spots([pair]) or {}).get(pair)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("spot fetch failed for %s: %s", pair, exc)
        return None
    if isinstance(data, dict):
        val = data.get("mid", data.get("close"))
    else:
        val = data
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    return val if (math.isfinite(val) and val > 0) else None


def require_spot(pair: str) -> float:
    """Return the real mid spot for ``pair`` or raise MarketDataUnavailable."""
    spot = get_spot(pair)
    if spot is None:
        raise MarketDataUnavailable("spot", pair)
    return spot


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def get_rates(pair: str) -> Optional[Tuple[float, float]]:
    """Return ``(r_dom, r_for)`` as decimals for ``pair``, or ``None``.

    Returns ``None`` unless BOTH legs are real — a one-sided rate cannot
    price an FX option, so half-data is treated as no data rather than
    quietly completed with a default.
    """
    try:
        from core.bloomberg_fx import get_fx_rates
        r = get_fx_rates(pair)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("rates fetch failed for %s: %s", pair, exc)
        return None
    if not isinstance(r, dict):
        return None
    r_dom = r.get("r_dom")
    r_for = r.get("r_for")
    if r_dom is None or r_for is None:
        return None
    try:
        r_dom = float(r_dom)
        r_for = float(r_for)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(r_dom) and math.isfinite(r_for)):
        return None
    return r_dom, r_for


def require_rates(pair: str) -> Tuple[float, float]:
    """Return ``(r_dom, r_for)`` for ``pair`` or raise MarketDataUnavailable."""
    rates = get_rates(pair)
    if rates is None:
        raise MarketDataUnavailable("rates", pair)
    return rates


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def get_atm_vol(pair: str, tenor: str) -> Optional[float]:
    """Return the real ATM vol (decimal) for ``pair``/``tenor``, or ``None``."""
    try:
        from core.bloomberg_fx import get_fx_vol_surface
        surf = get_fx_vol_surface(pair) or {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("vol fetch failed for %s: %s", pair, exc)
        return None
    node = surf.get(tenor)
    if isinstance(node, dict):
        return normalize_vol(node.get("atm"))
    return normalize_vol(node)


def require_atm_vol(pair: str, tenor: str) -> float:
    """Return the real ATM vol for ``pair``/``tenor`` or raise."""
    vol = get_atm_vol(pair, tenor)
    if vol is None:
        raise MarketDataUnavailable("vol", pair, tenor)
    return vol


def atm_vol_from_surface(surface, tenor: Optional[str] = None) -> Optional[float]:
    """Extract a real decimal ATM vol from an already-fetched surface dict.

    Mirrors :func:`get_atm_vol` but operates on a surface the caller already
    holds (avoids a second fetch).  Returns ``None`` — never a default — when
    no real ATM quote is present.  When ``tenor`` is given it is preferred;
    otherwise the nearest liquid tenor with a real quote is used.
    """
    if surface is None:
        return None
    if isinstance(surface, (int, float)):
        return normalize_vol(surface)
    if not isinstance(surface, dict):
        return None
    search = ([tenor] if tenor else []) + ["3M", "1M", "6M", "1Y", "2M", "9M", "2W", "1W"]
    seen = set()
    for tn in search:
        if tn is None or tn in seen:
            continue
        seen.add(tn)
        node = surface.get(tn)
        if isinstance(node, dict):
            v = normalize_vol(node.get("atm"))
            if v is not None:
                return v
        elif node is not None:
            v = normalize_vol(node)
            if v is not None:
                return v
    # Last resort: any dict-valued entry carrying a real atm quote.
    for node in surface.values():
        if isinstance(node, dict) and "atm" in node:
            v = normalize_vol(node.get("atm"))
            if v is not None:
                return v
    return None


__all__ = [
    "NA",
    "MarketDataUnavailable",
    "normalize_vol",
    "get_spot",
    "require_spot",
    "get_rates",
    "require_rates",
    "get_atm_vol",
    "require_atm_vol",
    "atm_vol_from_surface",
]
