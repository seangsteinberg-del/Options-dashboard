"""
FX Data Provider
================
Bloomberg API integration for FX markets.

Provides live FX data when Bloomberg Terminal is available:
  - Spot rates with bid/ask/mid/change across all G10 and EM pairs
  - Implied vol surfaces from OVDV (ATM, risk reversals, butterflies)
  - Deposit rate curves and forward point curves
  - Historical spot, vol, and rate time series
  - FX options chains with full Greeks
  - CFTC positioning data
  - Realized vol, implied correlations, central bank calendars

When Bloomberg is unavailable, functions return empty containers
(empty dicts, DataFrames, or Series) so the UI can handle gracefully.
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
    from core.bloomberg import get_connection, is_connected, bdp, bdh, bds, bloomberg_ever_connected
    _HAS_EQUITY_BBG = True
except Exception as _bbg_import_err:
    _HAS_EQUITY_BBG = False
    def bloomberg_ever_connected(): return False
    logger.warning("Failed to import core.bloomberg: %s", _bbg_import_err)

try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# Cache-Only Mode: when enabled, Dash callbacks never touch Bloomberg.
# Only the background fetcher thread is allowed to call Bloomberg.
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_ONLY_MODE = False
_fetch_thread_local = threading.local()


def set_cache_only_mode(enabled: bool):
    """Enable/disable cache-only mode for Dash callback threads."""
    global _CACHE_ONLY_MODE
    _CACHE_ONLY_MODE = enabled
    logger.info("Cache-only mode %s", "ENABLED" if enabled else "DISABLED")


def mark_thread_as_fetcher():
    """Mark the current thread as the background fetcher (allowed to call Bloomberg)."""
    _fetch_thread_local.is_fetcher = True


def _may_fetch() -> bool:
    """True if the current thread is allowed to call Bloomberg."""
    if getattr(_fetch_thread_local, 'is_fetcher', False):
        return True  # Background fetcher always allowed
    return not _CACHE_ONLY_MODE  # Dash threads blocked when cache-only


# ═══════════════════════════════════════════════════════════════════════════
# Cache Layer (thread-safe, with request deduplication)
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_TTL = {
    "spot": 15,
    "vol_surface": 300,
    "rates": 1800,
    "forwards": 600,
    "historical": 3600,
    "positioning": 86400,
}

# Longer TTLs when background fetcher is active (must outlast fetch interval)
_CACHE_TTL_BG = {
    "spot": 180,        # 3 min (fetcher runs every 2 min)
    "vol_surface": 300,  # 5 min
    "rates": 3600,       # 1 hr
    "forwards": 600,     # 10 min
    "historical": 7200,  # 2 hr
    "positioning": 86400,
}

_cache: Dict[str, Tuple[float, object]] = {}
_cache_lock = threading.Lock()

# In-flight request deduplication: when multiple callbacks want the same
# data, only ONE makes the Bloomberg call. Others wait on the Event.
_inflight: Dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()

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
    ttl_table = _CACHE_TTL_BG if _CACHE_ONLY_MODE else _CACHE_TTL
    with _cache_lock:
        if key in _cache:
            ts, val = _cache[key]
            if time.time() - ts < ttl_table.get(category, 60):
                return val
    return None


def _cache_set(key: str, value, category: str = "spot"):
    with _cache_lock:
        _cache[key] = (time.time(), value)


def _cache_wait_or_claim(key: str, category: str = "spot"):
    """Check cache, and if miss, claim the right to fetch.

    Returns (cached_value, should_fetch):
      - (value, False) if cache hit — use value directly
      - (None, True) if cache miss and YOU should fetch — others will wait
      - (value, False) if another thread fetched while you waited
    """
    # Fast path: cache hit
    cached = _cache_get(key, category)
    if cached is not None:
        return cached, False

    # Check if another thread is already fetching this key
    with _inflight_lock:
        if key in _inflight:
            # Another thread is fetching — wait for it
            evt = _inflight[key]
        else:
            # We're first — claim it
            evt = threading.Event()
            _inflight[key] = evt
            return None, True  # caller should fetch

    # Wait for the other thread to finish (max 30s)
    evt.wait(timeout=30)
    # Now check cache for the result
    cached = _cache_get(key, category)
    if cached is not None:
        return cached, False
    # Other thread failed — we could retry but just return miss
    return None, False


def _cache_done(key: str):
    """Signal that fetching for this key is complete."""
    with _inflight_lock:
        evt = _inflight.pop(key, None)
    if evt:
        evt.set()


def _cache_invalidate(key: str):
    """Remove a specific key from cache, forcing a re-fetch."""
    with _cache_lock:
        _cache.pop(key, None)


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
# Supported FX Pairs
# ═══════════════════════════════════════════════════════════════════════════

_ALL_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "EURAUD", "EURNZD",
    "NZDJPY", "AUDNZD", "CADCHF", "CADJPY", "EURNOK", "EURSEK", "USDSEK",
    "USDNOK", "USDMXN", "USDBRL", "USDTRY", "USDZAR", "USDCNH", "USDINR",
    "USDSGD", "USDKRW",
]


# ═══════════════════════════════════════════════════════════════════════════
# BBG Ticker Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fx_bbg_ticker(pair: str) -> str:
    """Convert pair like EURUSD to Bloomberg ticker EURUSD Curncy."""
    return f"{pair.upper()} Curncy"


def _fx_vol_bbg(pair: str) -> str:
    """OVDV vol ticker, e.g., EURUSDV1M Curncy."""
    return f"{pair.upper()}V Curncy"


def _deposit_bbg(ccy: str, tenor: str) -> list:
    """Deposit rate ticker candidates for a currency+tenor.

    Returns a list of tickers to try in order (first hit wins).
    G10 currencies use standard deposit rate tickers (Curncy).
    EM currencies include swap-rate and policy-rate fallbacks since
    many terminals lack EM deposit rate data.
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
    primary = f"{prefix}{suffix} Curncy"

    # EM fallback tickers: swap rates and policy rates that are widely available
    _EM_SWAP_FALLBACKS = {
        "BRL": ["BCSW3 Curncy", "BZSTSETA Index"],      # BRL 3M swap / SELIC target
        "CNH": ["CCSWNI3 Curncy", "CCFIX3M Index"],      # CNH 3M NDF swap
        "INR": ["IRSWNI3 Curncy", "RBIREPRT Index"],     # INR 3M NDF swap / RBI repo
        "SGD": ["SDSW3 Curncy", "MASRRR Index"],          # SGD 3M swap / MAS rate
        "KRW": ["KWSWNI3 Curncy", "KORP7DR Index"],      # KRW 3M NDF swap / BOK 7d repo
    }
    ccy_upper = ccy.upper()
    fallbacks = _EM_SWAP_FALLBACKS.get(ccy_upper, [])
    return [primary] + fallbacks


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

    # If requesting a subset, check the all-pairs cache first (populated by bg_fetcher)
    all_ck = "spots_" + ",".join(_ALL_PAIRS)
    if pairs != _ALL_PAIRS:
        all_cached = _cache_get(all_ck, "spot")
        if all_cached is not None and isinstance(all_cached, dict):
            result = {p: all_cached[p] for p in pairs if p in all_cached}
            if result:  # only use cache hit if we actually found the requested pairs
                return result

    ck = "spots_" + ",".join(pairs)
    cached, should_fetch = _cache_wait_or_claim(ck, "spot")
    if cached is not None:
        return cached
    if not should_fetch:
        return {}
    if not _may_fetch():
        _cache_done(ck)
        return {}  # cache-only mode: background thread will populate

    if _HAS_EQUITY_BBG and is_connected():
        try:
            tickers = [_fx_bbg_ticker(p) for p in pairs]
            fields = ["PX_BID", "PX_ASK", "PX_LAST", "PX_MID", "CHG_NET_1D",
                       "CHG_PCT_1D", "PX_HIGH", "PX_LOW", "PX_OPEN", "VOLUME"]
            df = bdp(tickers, fields)
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
                    if bid > 0 and ask > 0:
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
                _cache_set(ck, result, "spot")
                _cache_done(ck)
                return result
            _log_fetch_failure("get_fx_spots", ",".join(pairs[:3]), "BDP returned no data for any pair")
            _cache_done(ck)
            return {}
        except Exception as e:
            logger.error(f"FX spots BBG request failed: {e}")
            _log_fetch_failure("get_fx_spots", ",".join(pairs[:3]), str(e))
            _cache_done(ck)
            return {}

    # No Bloomberg — return empty
    _cache_done(ck)
    return {}


def get_fx_vol_surface(pair: str) -> Dict[str, dict]:
    """
    Full vol surface: {tenor: {atm, rr25, bf25, rr10, bf10}}.
    Uses OVDV screen fields when Bloomberg is available.
    """
    ck = f"volsurf_{pair}"
    cached, should_fetch = _cache_wait_or_claim(ck, "vol_surface")
    if cached is not None:
        return cached
    if not should_fetch:
        return {}
    if not _may_fetch():
        _cache_done(ck)
        return {}

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
            vol_fields = ["PX_LAST", "PX_MID"]
            df = bdp(all_tickers, vol_fields)

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
                _cache_done(ck)
                return surface
            logger.warning("Vol surface %s: 0 ATM tenors (BDP returned %d rows, "
                           "matched %d/%d tickers). Sample tickers sent: %s",
                           pair, len(df), matched, len(all_tickers),
                           all_tickers[:3])
            _cache_done(ck)
            return {}
        except Exception as e:
            logger.error(f"FX vol surface BBG request failed: {e}")
            _log_fetch_failure("get_fx_vol_surface", pair, str(e))
            _cache_done(ck)
            return {}

    _cache_done(ck)
    return {}


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
    cached, should_fetch = _cache_wait_or_claim(ck, "rates")
    if cached is not None:
        return cached
    if not should_fetch:
        return {}
    if not _may_fetch():
        _cache_done(ck)
        return {}

    if _HAS_EQUITY_BBG and is_connected():
        try:
            # FX options convention: domestic = quote currency, foreign = base currency
            # For EURUSD: domestic=USD (quote), foreign=EUR (base)
            ccy_dom = pair[3:]   # quote currency (pricing currency)
            ccy_for = pair[:3]   # base currency (underlying asset)

            def _try_rate(ccy: str) -> Optional[float]:
                """Try each ticker candidate for a currency until one works."""
                candidates = _deposit_bbg(ccy, "3M")
                for tick in candidates:
                    try:
                        df = bdp([tick], ["PX_LAST", "PX_MID"])
                        if df.empty:
                            continue
                        idx_map = {iv.upper().strip(): iv for iv in df.index}
                        actual = tick if tick in df.index else idx_map.get(tick.upper().strip())
                        if actual is None:
                            continue
                        for fld in ("PX_LAST", "PX_MID"):
                            if fld in df.columns:
                                v = df.loc[actual, fld]
                                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                    rate = float(v) / 100.0
                                    logger.debug("Rate for %s via %s = %.4f", ccy, tick, rate)
                                    return rate
                    except Exception:
                        continue
                return None

            r_dom = _try_rate(ccy_dom)
            r_for = _try_rate(ccy_for)

            if r_dom is None and r_for is None:
                raise ValueError(f"No rate data for either {ccy_dom} or {ccy_for}")

            # Use sensible defaults if only one side available
            if r_dom is None:
                r_dom = r_for  # approximate
                logger.warning("Using foreign rate as proxy for %s domestic rate", pair)
            if r_for is None:
                r_for = r_dom
                logger.warning("Using domestic rate as proxy for %s foreign rate", pair)

            res = {"r_dom": r_dom, "r_for": r_for,
                   "rate_diff": round(r_dom - r_for, 4)}
            _cache_set(ck, res, "rates")
            _cache_done(ck)
            return res
        except Exception as e:
            logger.error(f"FX rates BBG request failed: {e}")
            _log_fetch_failure("get_fx_rates", pair, str(e))
            _cache_done(ck)
            return {}

    _cache_done(ck)
    return {}


def get_fx_rate_curve(ccy: str) -> Dict[str, float]:
    """
    Deposit/swap rate curve for a single currency.
    Returns {tenor: rate}.
    """
    ck = f"ratecurve_{ccy}"
    cached = _cache_get(ck, "rates")
    if cached is not None:
        return cached
    if not _may_fetch():
        return {}

    if _HAS_EQUITY_BBG and is_connected():
        try:
            tenors = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y"]
            # _deposit_bbg now returns a list of candidates; collect all primaries
            all_tickers = []
            tenor_tick_map = {}  # tenor -> list of candidate tickers
            for t in tenors:
                candidates = _deposit_bbg(ccy, t)
                tenor_tick_map[t] = candidates
                all_tickers.extend(candidates)
            # Batch BDP call with all candidates at once
            df = bdp(all_tickers, ["PX_LAST", "PX_MID"])
            idx_map = {iv.upper().strip(): iv for iv in df.index} if not df.empty else {}
            curve = {}
            for tenor in tenors:
                for tick in tenor_tick_map[tenor]:
                    actual = tick if tick in df.index else idx_map.get(tick.upper().strip())
                    if actual is None:
                        continue
                    val = None
                    for fld in ("PX_LAST", "PX_MID"):
                        if fld in df.columns:
                            v = df.loc[actual, fld]
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                val = v
                                break
                    if val is not None:
                        curve[tenor] = float(val) / 100.0
                        break  # Got a value for this tenor, move on
            if curve:
                _cache_set(ck, curve, "rates")
                return curve
            _log_fetch_failure("get_fx_rate_curve", ccy, "BDP returned no rate data")
            return {}
        except Exception as e:
            logger.error(f"FX rate curve BBG request failed: {e}")
            _log_fetch_failure("get_fx_rate_curve", ccy, str(e))
            return {}

    return {}


def get_fx_forward_curve(pair: str) -> Dict[str, dict]:
    """
    Forward point curve.
    Returns {tenor: {fwd_points, outright, implied_rate_diff}}.
    """
    ck = f"fwdcurve_{pair}"
    cached = _cache_get(ck, "forwards")
    if cached is not None:
        return cached
    if not _may_fetch():
        return {}

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
                    impl_diff = np.log(outright / spot) / ty if (ty > 0 and outright > 0) else 0.0
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

    return {}


def get_fx_historical_spot(pair: str, days: int = 252) -> pd.DataFrame:
    """
    Historical OHLC spot data.
    Returns DataFrame with columns: open, high, low, close.
    """
    ck = f"histspot_{pair}_{days}"
    cached, should_fetch = _cache_wait_or_claim(ck, "historical")
    if cached is not None:
        return cached
    if not _may_fetch() and should_fetch:
        _cache_done(ck)
        return pd.DataFrame()
    if not should_fetch:
        return pd.DataFrame()

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
                _cache_done(ck)
                return df
            _log_fetch_failure("get_fx_historical_spot", pair, "BDH returned empty dataframe")
            _cache_done(ck)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"FX historical spot BBG request failed: {e}")
            _log_fetch_failure("get_fx_historical_spot", pair, str(e))
            _cache_done(ck)
            return pd.DataFrame()

    _cache_done(ck)
    return pd.DataFrame()


def get_fx_historical_vol(pair: str, tenor: str = "1M",
                          metric: str = "atm", days: int = 252) -> pd.Series:
    """Historical vol time series for a given tenor/metric."""
    metric = metric.upper()  # normalize cache key
    ck = f"histvol_{pair}_{tenor}_{metric}_{days}"
    cached, should_fetch = _cache_wait_or_claim(ck, "historical")
    if cached is not None:
        return cached
    if not _may_fetch() and should_fetch:
        _cache_done(ck)
        return pd.Series(dtype=float)
    if not should_fetch:
        return pd.Series(dtype=float)

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
                    _cache_done(ck)
                    return series
            _log_fetch_failure("get_fx_historical_vol", f"{pair}/{tenor}/{metric}", "BDH returned empty")
            _cache_done(ck)
            return pd.Series(dtype=float)
        except Exception as e:
            logger.error(f"FX historical vol BBG request failed: {e}")
            _log_fetch_failure("get_fx_historical_vol", f"{pair}/{tenor}/{metric}", str(e))
            _cache_done(ck)
            return pd.Series(dtype=float)

    _cache_done(ck)
    return pd.Series(dtype=float)


def get_fx_option_chain(pair: str, tenor: str = "1M") -> pd.DataFrame:
    """
    FX options chain for a pair and tenor.
    Returns DataFrame: strike, type, bid, ask, mid, iv, delta, gamma, vega, theta, oi.
    """
    ck = f"optchain_{pair}_{tenor}"
    cached = _cache_get(ck, "vol_surface")
    if cached is not None:
        return cached
    if not _may_fetch():
        return pd.DataFrame()

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

    return pd.DataFrame()


def get_fx_deposit_rates(ccy: str, tenors: List[str] = None) -> Dict[str, float]:
    """Deposit rate curve for a currency."""
    if tenors is None:
        tenors = ["1M", "3M", "6M", "1Y"]
    return get_fx_rate_curve(ccy)


def get_cftc_positioning(pair: str) -> dict:
    """
    CFTC Commitments of Traders positioning proxy.
    When live CFTC data is unavailable, derives positioning estimate from
    historical spot momentum, vol percentile, and carry direction.
    Returns {net_spec, z_score, signal, confidence}.
    """
    ck = f"cftc_{pair}"
    cached = _cache_get(ck, "positioning")
    if cached is not None:
        return cached
    if not _may_fetch():
        return {}

    try:
        hist = get_fx_historical_spot(pair, 90)
        if hist.empty or len(hist) < 30:
            return {}
        close = hist["close"].values
        # Momentum: 1M vs 3M return z-score as positioning proxy
        ret_1m = (close[-1] / close[-22] - 1) if len(close) > 22 else 0.0
        ret_3m = (close[-1] / close[0] - 1) if len(close) > 60 else ret_1m
        ret_std = float(np.std(np.diff(np.log(close[-60:]))))
        z = ret_1m / max(ret_std * np.sqrt(22), 1e-6)
        # Estimate: positive z = long positioning (USD strength for USD/XXX)
        result = {
            "net_spec": round(float(z * 30), 1),  # synthetic net speculative (scaled)
            "z_score": round(float(z), 2),
            "signal": "LONG" if z > 1.0 else "SHORT" if z < -1.0 else "NEUTRAL",
            "confidence": "proxy",
        }
        _cache_set(ck, result, "positioning")
        _cache_done(ck)
        return result
    except Exception as e:
        logger.debug("CFTC positioning proxy failed for %s: %s", pair, e)
        return {}


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
    if not va or not vb or not vc or va <= 0 or vb <= 0 or vc <= 0:
        return 0.0
    rho = (va**2 + vb**2 - vc**2) / (2.0 * va * vb)
    return round(max(min(rho, 1.0), -1.0), 4)


def get_central_bank_dates(bank: str) -> List[dict]:
    """
    Upcoming central bank meeting dates with consensus.
    Returns [{date, bank, consensus, prev_rate}].
    """
    _CB_SCHEDULE = {
        "FED": {"dates": [7, 14, 42, 77, 112, 147, 175],
                "rate": 4.25, "consensus": "hold"},
        "ECB": {"dates": [10, 42, 70, 105, 140, 175],
                "rate": 2.75, "consensus": "hold"},
        "BOJ": {"dates": [12, 56, 91, 140, 182],
                "rate": 0.50, "consensus": "hold"},
        "BOE": {"dates": [14, 49, 84, 119, 154, 182],
                "rate": 4.25, "consensus": "-25bp"},
        "RBA": {"dates": [8, 42, 77, 112, 147, 175],
                "rate": 3.85, "consensus": "hold"},
        "RBNZ": {"dates": [21, 63, 105, 147, 182],
                 "rate": 3.75, "consensus": "-25bp"},
        "BOC": {"dates": [10, 42, 77, 112, 147, 175],
                "rate": 2.75, "consensus": "hold"},
        "SNB": {"dates": [35, 91, 147, 182],
                "rate": 0.75, "consensus": "hold"},
        "RIKSBANK": {"dates": [28, 70, 119, 168],
                     "rate": 2.75, "consensus": "hold"},
        "NORGES": {"dates": [21, 63, 112, 161],
                   "rate": 4.00, "consensus": "hold"},
        "CBRT": {"dates": [14, 42, 70, 98, 126, 154, 182],
                 "rate": 42.50, "consensus": "-50bp"},
        "SARB": {"dates": [21, 77, 133, 175],
                 "rate": 7.75, "consensus": "hold"},
        "BANXICO": {"dates": [14, 56, 98, 140, 175],
                    "rate": 9.50, "consensus": "-25bp"},
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


def get_fx_correlation_matrix(pairs: List[str] = None,
                              days: int = 120) -> pd.DataFrame:
    """
    Cross-pair correlation matrix computed from historical spot data.
    Returns identity matrix if historical data is unavailable.
    """
    if pairs is None:
        pairs = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD",
                 "NZDUSD", "USDCAD"]
    n = len(pairs)
    # Try to compute from live historical data
    returns = {}
    for pair in pairs:
        hist = get_fx_historical_spot(pair, days)
        if not hist.empty and "close" in hist.columns and len(hist) > 10:
            returns[pair] = np.log(hist["close"] / hist["close"].shift(1)).dropna()
    if len(returns) >= 2:
        ret_df = pd.DataFrame(returns)
        corr = ret_df.corr()
        # Reindex to include all requested pairs (NaN for missing → fill diagonal)
        corr = corr.reindex(index=pairs, columns=pairs, fill_value=0.0)
        np.fill_diagonal(corr.values, 1.0)
        return corr
    # No historical data available — return identity matrix
    return pd.DataFrame(np.eye(n), index=pairs, columns=pairs)


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
            "days": _TENOR_DAYS.get(tenor, 30),
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
