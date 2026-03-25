"""
Bloomberg Data Provider
=======================
Wraps blpapi to pull live and historical data from Bloomberg Terminal.

Provides:
  - Real-time & delayed spot prices, bid/ask
  - Full options chains with Greeks and IV
  - Implied vol surfaces (OVDV)
  - Historical OHLCV bars
  - Realized volatility
  - Dividend yields, risk-free rates
  - Corporate actions / earnings dates

Falls back to synthetic data if Bloomberg is unavailable, so the
dashboard always works even without a terminal connection.
"""

import logging
import threading
import time as _time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Try to import blpapi ──────────────────────────────────────────────────
try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False
    logger.warning("blpapi not installed — running in synthetic/fallback mode. "
                   "Install with: pip install blpapi")


class BloombergConnection:
    """Manages the blpapi session lifecycle.

    Thread-safe: a lock serialises all sendRequest/nextEvent calls so
    concurrent Dash callbacks never interleave on the same session.
    """

    def __init__(self, host="localhost", port=8194):
        self.host = host
        self.port = port
        self.session = None
        self.ref_data_service = None
        self.connected = False
        self._lock = threading.Lock()  # serialises ALL blpapi calls

    def connect(self) -> bool:
        if not BLPAPI_AVAILABLE:
            logger.info("blpapi not available, using fallback data")
            return False

        try:
            # Tear down any stale session first
            if self.session:
                try:
                    self.session.stop()
                except Exception:
                    pass
                self.session = None
                self.ref_data_service = None

            opts = blpapi.SessionOptions()
            opts.setServerHost(self.host)
            opts.setServerPort(self.port)
            opts.setAutoRestartOnDisconnection(True)
            opts.setConnectTimeout(5000)  # 5s connection timeout

            self.session = blpapi.Session(opts)
            if not self.session.start():
                logger.error("Failed to start Bloomberg session")
                return False

            if not self.session.openService("//blp/refdata"):
                logger.error("Failed to open //blp/refdata")
                return False

            self.ref_data_service = self.session.getService("//blp/refdata")
            self.connected = True
            logger.info(f"Connected to Bloomberg Terminal at {self.host}:{self.port}")

            # Quick health check: verify Bloomberg actually returns data
            try:
                request = self.ref_data_service.createRequest("ReferenceDataRequest")
                request.append("securities", "EURUSD Curncy")
                request.append("fields", "PX_LAST")
                request.append("fields", "PX_MID")
                responses = self._send_request(request)
                has_data = False
                for msg in responses:
                    if msg.hasElement("securityData"):
                        sd = msg.getElement("securityData")
                        for i in range(sd.numValues()):
                            sec = sd.getValueAsElement(i)
                            if sec.hasElement("fieldData"):
                                fd = sec.getElement("fieldData")
                                for fld in ("PX_LAST", "PX_MID"):
                                    if fd.hasElement(fld) and not fd.getElement(fld).isNull():
                                        has_data = True
                if has_data:
                    logger.info("Bloomberg health check PASSED — EURUSD data available")
                else:
                    logger.warning("Bloomberg health check WARNING — connected but EURUSD "
                                   "returned no price data. Terminal may not be logged in "
                                   "or may lack FX data permissions.")
            except Exception as hc_err:
                logger.warning("Bloomberg health check failed: %s", hc_err)
            return True

        except Exception as e:
            logger.error(f"Bloomberg connection failed: {e}")
            self.connected = False
            return False

    def reconnect(self) -> bool:
        """Tear down and rebuild the session. Returns True on success."""
        logger.info("Bloomberg reconnecting...")
        self.connected = False
        return self.connect()

    def disconnect(self):
        if self.session:
            try:
                self.session.stop()
            except Exception:
                pass
            self.connected = False
            logger.info("Disconnected from Bloomberg")

    _next_cid = 1  # class-level counter for correlation IDs

    def _send_request(self, request):
        """Send request and collect all response events.

        Thread-safe: holds `_lock` for the entire send→receive cycle.
        Uses a unique CorrelationId per request so stale events from
        previous requests are discarded instead of being mixed in.
        """
        with self._lock:
            if not self.connected or not self.session:
                return []

            # Assign a unique correlation ID to this request
            cid_val = BloombergConnection._next_cid
            BloombergConnection._next_cid += 1
            cid = blpapi.CorrelationId(cid_val)

            # Drain any stale events sitting in the queue before sending
            while True:
                stale = self.session.nextEvent(timeout=0)
                if stale.eventType() == blpapi.Event.TIMEOUT:
                    break  # queue is empty

            try:
                self.session.sendRequest(request, correlationId=cid)
            except Exception as e:
                logger.error("Bloomberg sendRequest failed: %s", e)
                self.connected = False
                return []

            data = []
            max_attempts = 6  # 6 × 5s = 30s max per request
            for _ in range(max_attempts):
                try:
                    event = self.session.nextEvent(timeout=5000)
                except Exception as e:
                    logger.error("Bloomberg nextEvent failed: %s", e)
                    self.connected = False
                    return data
                ev_type = event.eventType()
                if ev_type == blpapi.Event.TIMEOUT:
                    logger.warning("Bloomberg request timed out (cid=%d)", cid_val)
                    break
                if ev_type in (blpapi.Event.REQUEST_STATUS,):
                    for msg in event:
                        logger.error("Bloomberg request status error: %s", msg)
                    break

                # Only accept messages that match OUR correlation ID
                for msg in event:
                    if msg.correlationId() == cid:
                        data.append(msg)
                    else:
                        logger.debug("Discarding stale event (expected cid=%d, got %s)",
                                     cid_val, msg.correlationId())

                if ev_type == blpapi.Event.RESPONSE:
                    break
            return data


# ── Singleton connection (thread-safe) ────────────────────────────────────
_connection: Optional[BloombergConnection] = None
_conn_lock = threading.Lock()

# Once Bloomberg has connected successfully even once, this flag is set True
# and NEVER reset. This prevents ANY synthetic/fallback data from leaking
# through if the session temporarily drops.
_bloomberg_ever_connected = False
_last_connect_attempt = 0.0  # monotonic time of last failed connection attempt
_RECONNECT_COOLDOWN = 30.0   # seconds before retrying a failed connection


def get_connection() -> BloombergConnection:
    """Return the singleton BloombergConnection, creating it on first call.

    Uses a cooldown to avoid blocking all callbacks with repeated failed
    connection attempts.
    """
    global _connection, _bloomberg_ever_connected, _last_connect_attempt
    # Fast path — already connected, no lock needed
    if _connection is not None and _connection.connected:
        return _connection
    # If we recently failed to connect, don't retry yet (avoid blocking)
    if _last_connect_attempt and (_time.monotonic() - _last_connect_attempt < _RECONNECT_COOLDOWN):
        if _connection is None:
            _connection = BloombergConnection()
        return _connection
    with _conn_lock:
        # Double-check inside the lock
        if _connection is not None and _connection.connected:
            return _connection
        if _connection is None:
            _connection = BloombergConnection()
        _connection.connect()
        if _connection.connected:
            _bloomberg_ever_connected = True
            _last_connect_attempt = 0.0
        else:
            _last_connect_attempt = _time.monotonic()
        return _connection


def bloomberg_ever_connected() -> bool:
    """True if Bloomberg has EVER connected in this process.

    Once True, synthetic/fallback data must NEVER be returned — even if
    the session drops temporarily. Use this to gate fallback paths.
    """
    return _bloomberg_ever_connected


def is_connected() -> bool:
    """Check if Bloomberg session is alive.

    Does NOT aggressively reconnect — relies on get_connection() cooldown
    to avoid blocking all callbacks with repeated failed attempts.
    """
    global _last_connect_attempt
    conn = get_connection()
    if not conn.connected:
        return False
    if conn.session and conn.ref_data_service:
        try:
            _ = conn.ref_data_service.name()
            return True
        except Exception:
            logger.warning("Bloomberg health check failed — marking disconnected")
            conn.connected = False
            _last_connect_attempt = _time.monotonic()
            return False
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Reference Data (BDP / BDH / BDS)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_value(element):
    """Safely extract a Python value from a blpapi Element."""
    try:
        if element.isNull():
            return None
        dtype = element.datatype()
        # blpapi datatypes: FLOAT64=6, INT32=2, INT64=3, STRING=8, DATE=10, DATETIME=12, BOOL=1
        if dtype in (5, 6):     # FLOAT32, FLOAT64
            return element.getValueAsFloat()
        if dtype in (2, 3):     # INT32, INT64
            return element.getValueAsInteger()
        if dtype in (10, 12):   # DATE, DATETIME
            return element.getValueAsDatetime()
        if dtype in (1,):       # BOOL
            return element.getValueAsBool()
        return element.getValueAsString()
    except Exception:
        try:
            return element.getValueAsString()
        except Exception:
            return None


def bdp(securities: List[str], fields: List[str]) -> pd.DataFrame:
    """Bloomberg Data Point — single-point reference data.

    When connected: returns real data or EMPTY DataFrame on failure.
    If Bloomberg was ever connected, NEVER returns synthetic data.
    """
    conn = get_connection()
    if not conn.connected:
        if _bloomberg_ever_connected:
            logger.warning("BDP: Bloomberg disconnected but was previously connected "
                           "— returning EMPTY (no synthetic data)")
            return pd.DataFrame()
        return _fallback_bdp(securities, fields)

    try:
        if not conn.ref_data_service:
            logger.error("BDP: ref_data_service is None — session broken")
            conn.connected = False
            return pd.DataFrame()
        request = conn.ref_data_service.createRequest("ReferenceDataRequest")
        for sec in securities:
            request.append("securities", sec)
        for fld in fields:
            request.append("fields", fld)

        responses = conn._send_request(request)
        if not responses:
            logger.warning("BDP got no response messages for %s", securities[:3])
            return pd.DataFrame()  # Empty — do NOT inject fake data
        rows = []
        for msg in responses:
            # Check for request-level errors
            if msg.hasElement("responseError"):
                err = msg.getElement("responseError")
                logger.error("BDP responseError: %s",
                             err.getElementAsString("message") if err.hasElement("message") else str(err))
                continue
            if not msg.hasElement("securityData"):
                continue
            security_data = msg.getElement("securityData")

            # securityData is an array for ReferenceDataRequest
            for i in range(security_data.numValues()):
                try:
                    sec = security_data.getValueAsElement(i)
                except Exception:
                    continue

                # Check for errors on this security
                if sec.hasElement("securityError"):
                    err = sec.getElement("securityError")
                    logger.warning("BDP security error for %s: %s",
                                   sec.getElementAsString("security") if sec.hasElement("security") else "?",
                                   err.getElementAsString("message") if err.hasElement("message") else "unknown")
                    continue

                try:
                    ticker = sec.getElementAsString("security")
                except Exception:
                    continue

                if not sec.hasElement("fieldData"):
                    continue
                field_data = sec.getElement("fieldData")

                # Log field-level exceptions (invalid fields for this security)
                if sec.hasElement("fieldExceptions"):
                    fe = sec.getElement("fieldExceptions")
                    for fi in range(fe.numValues()):
                        try:
                            fex = fe.getValueAsElement(fi)
                            fld_id = fex.getElementAsString("fieldId") if fex.hasElement("fieldId") else "?"
                            err_info = fex.getElement("errorInfo") if fex.hasElement("errorInfo") else None
                            sub = err_info.getElementAsString("subcategory") if err_info and err_info.hasElement("subcategory") else "?"
                            logger.debug("BDP field exception %s/%s: %s", ticker, fld_id, sub)
                        except Exception:
                            pass

                row = {"security": ticker}
                for fld in fields:
                    if field_data.hasElement(fld):
                        row[fld] = _extract_value(field_data.getElement(fld))
                    else:
                        row[fld] = None
                rows.append(row)

        if not rows:
            logger.warning("BDP returned no data for %s", securities[:3])
            return pd.DataFrame()  # Empty — do NOT inject fake data
        result_df = pd.DataFrame(rows).set_index("security")
        logger.debug("BDP result: %d rows, index=%s, columns=%s, sample=%s",
                      len(result_df), list(result_df.index[:5]),
                      list(result_df.columns),
                      result_df.head(3).to_dict() if len(result_df) <= 10
                      else f"{len(result_df)} rows")
        return result_df

    except Exception as e:
        logger.error(f"BDP request failed: {e}")
        conn.connected = False  # Mark broken so next call triggers reconnect
        return pd.DataFrame()  # Empty — do NOT inject fake data


def bdh(security: str, fields: List[str], start_date: str, end_date: str = None,
        **overrides) -> pd.DataFrame:
    """Bloomberg Data History — historical time series.

    If Bloomberg was ever connected, NEVER returns synthetic data.
    """
    conn = get_connection()
    if not conn.connected:
        if _bloomberg_ever_connected:
            logger.warning("BDH: Bloomberg disconnected but was previously connected "
                           "— returning EMPTY (no synthetic data)")
            return pd.DataFrame()
        return _fallback_bdh(security, fields, start_date, end_date)

    try:
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        if not conn.ref_data_service:
            conn.connected = False
            return pd.DataFrame()
        request = conn.ref_data_service.createRequest("HistoricalDataRequest")
        request.append("securities", security)
        for fld in fields:
            request.append("fields", fld)
        request.set("startDate", start_date.replace("-", ""))
        request.set("endDate", end_date.replace("-", ""))
        request.set("periodicitySelection", overrides.get("periodicity", "DAILY"))

        responses = conn._send_request(request)
        if not responses:
            logger.warning("BDH got no response messages for %s", security)
            return pd.DataFrame()  # Empty — do NOT inject fake data
        rows = []
        for msg in responses:
            # Check for request-level errors
            if msg.hasElement("responseError"):
                err = msg.getElement("responseError")
                logger.error("BDH responseError: %s",
                             err.getElementAsString("message") if err.hasElement("message") else str(err))
                continue
            if not msg.hasElement("securityData"):
                continue
            security_data = msg.getElement("securityData")

            # Check for security-level errors
            if security_data.hasElement("securityError"):
                err = security_data.getElement("securityError")
                logger.warning("BDH security error for %s: %s", security,
                               err.getElementAsString("message") if err.hasElement("message") else "unknown")
                continue

            if not security_data.hasElement("fieldData"):
                continue
            field_data_array = security_data.getElement("fieldData")

            for j in range(field_data_array.numValues()):
                fd = field_data_array.getValueAsElement(j)
                row = {}
                # Extract date
                if fd.hasElement("date"):
                    dt = _extract_value(fd.getElement("date"))
                    if hasattr(dt, 'year'):
                        row["date"] = datetime(dt.year, dt.month, dt.day)
                    else:
                        row["date"] = str(dt) if dt else None
                else:
                    row["date"] = None

                # Extract fields
                for fld in fields:
                    if fd.hasElement(fld):
                        val = _extract_value(fd.getElement(fld))
                        try:
                            row[fld] = float(val) if val is not None else None
                        except (ValueError, TypeError):
                            row[fld] = None
                    else:
                        row[fld] = None
                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df.set_index("date", inplace=True)
        return df

    except Exception as e:
        logger.error(f"BDH request failed: {e}")
        conn.connected = False
        return pd.DataFrame()  # Empty — do NOT inject fake data


def bds(security: str, field: str, **overrides) -> pd.DataFrame:
    """Bloomberg Data Set — bulk data (options chains, etc.)."""
    conn = get_connection()
    if not conn.connected:
        return pd.DataFrame()

    try:
        if not conn.ref_data_service:
            conn.connected = False
            return pd.DataFrame()
        request = conn.ref_data_service.createRequest("ReferenceDataRequest")
        request.append("securities", security)
        request.append("fields", field)

        if overrides:
            ov = request.getElement("overrides")
            for k, v in overrides.items():
                o = ov.appendElement()
                o.setElement("fieldId", k)
                o.setElement("value", str(v))

        responses = conn._send_request(request)
        rows = []
        for msg in responses:
            if not msg.hasElement("securityData"):
                continue
            security_data = msg.getElement("securityData")

            for i in range(security_data.numValues()):
                try:
                    sec = security_data.getValueAsElement(i)
                except Exception:
                    continue

                if not sec.hasElement("fieldData"):
                    continue
                fd = sec.getElement("fieldData")

                if not fd.hasElement(field):
                    continue
                bulk = fd.getElement(field)

                for j in range(bulk.numValues()):
                    row_elem = bulk.getValueAsElement(j)
                    row = {}
                    for k in range(row_elem.numElements()):
                        el = row_elem.getElement(k)
                        row[str(el.name())] = _extract_value(el)
                    rows.append(row)

        return pd.DataFrame(rows)

    except Exception as e:
        logger.error(f"BDS request failed: {e}")
        conn.connected = False
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# High-Level Data Functions (used by panels)
# ═══════════════════════════════════════════════════════════════════════════

def get_spot_prices(tickers: List[str]) -> Dict[str, dict]:
    """
    Get current spot prices, changes, bid/ask for a list of tickers.
    Returns {ticker: {price, change, change_pct, bid, ask, volume, ...}}
    """
    bbg_tickers = [_to_bbg_ticker(t) for t in tickers]
    fields = ["PX_LAST", "CHG_NET_1D", "CHG_PCT_1D", "PX_BID", "PX_ASK",
              "VOLUME", "PX_HIGH", "PX_LOW", "PX_OPEN",
              "CUR_MKT_CAP", "VOLATILITY_30D"]

    df = bdp(bbg_tickers, fields)

    result = {}
    for orig, bbg in zip(tickers, bbg_tickers):
        if bbg in df.index:
            row = df.loc[bbg]
            result[orig] = {
                "price": _safe_float(row.get("PX_LAST")),
                "change": _safe_float(row.get("CHG_NET_1D")),
                "change_pct": _safe_float(row.get("CHG_PCT_1D")),
                "bid": _safe_float(row.get("PX_BID")),
                "ask": _safe_float(row.get("PX_ASK")),
                "volume": _safe_float(row.get("VOLUME")),
                "high": _safe_float(row.get("PX_HIGH")),
                "low": _safe_float(row.get("PX_LOW")),
                "open": _safe_float(row.get("PX_OPEN")),
                "mkt_cap": _safe_float(row.get("CUR_MKT_CAP")),
                "rv30": _safe_float(row.get("VOLATILITY_30D")),
            }
        else:
            logger.warning(f"Ticker {orig} ({bbg}) missing from BDP response — skipped")

    return result


def get_options_chain(underlying: str, expiry: str = None,
                      option_type: str = "both") -> pd.DataFrame:
    """
    Pull a full options chain from Bloomberg.

    Parameters
    ----------
    underlying : ticker like 'SPY', 'AAPL'
    expiry : 'YYYY-MM-DD' format, or None for nearest monthly
    option_type : 'call', 'put', or 'both'
    """
    conn = get_connection()
    bbg = _to_bbg_ticker(underlying)

    if not conn.connected:
        return _fallback_options_chain(underlying, expiry)

    try:
        # Get chain tickers via OPT_CHAIN
        overrides = {}
        if expiry:
            overrides["OPTION_CHAIN_EXPIRY_DT"] = expiry.replace("-", "")

        chain_df = bds(bbg, "OPT_CHAIN", **overrides)
        if chain_df.empty:
            logger.warning(f"OPT_CHAIN empty for {bbg} — returning empty DataFrame")
            return pd.DataFrame()

        opt_tickers = chain_df.iloc[:, 0].tolist()

        # Pull data for all option tickers
        fields = ["PX_LAST", "PX_BID", "PX_ASK", "IVOL_MID", "DELTA", "GAMMA",
                   "THETA", "VEGA", "OPEN_INT", "VOLUME", "STRIKE_PX",
                   "OPT_EXPIRE_DT", "OPT_PUT_CALL"]

        df = bdp(opt_tickers, fields)
        df = df.reset_index().rename(columns={"security": "option_ticker"})

        # Normalize column names
        col_map = {
            "STRIKE_PX": "strike", "OPT_EXPIRE_DT": "expiry",
            "OPT_PUT_CALL": "type", "PX_LAST": "mid", "PX_BID": "bid",
            "PX_ASK": "ask", "IVOL_MID": "iv", "DELTA": "delta",
            "GAMMA": "gamma", "THETA": "theta", "VEGA": "vega",
            "OPEN_INT": "open_interest", "VOLUME": "volume",
        }
        df = df.rename(columns=col_map)

        if option_type == "call":
            df = df[df["type"].str.upper() == "CALL"]
        elif option_type == "put":
            df = df[df["type"].str.upper() == "PUT"]

        return df.sort_values("strike").reset_index(drop=True)

    except Exception as e:
        logger.error(f"Options chain request failed: {e}")
        return pd.DataFrame()


def get_vol_surface(underlying: str, r: float = 0.05, q: float = 0.015) -> Tuple:
    """
    Pull implied vol surface from Bloomberg (OVDV).
    Returns (strikes, expiries, vol_matrix) just like generate_vol_surface().
    """
    conn = get_connection()
    bbg = _to_bbg_ticker(underlying)

    if not conn.connected:
        from core.pricing import generate_vol_surface
        spot_data = _fallback_spot(underlying)
        return generate_vol_surface(S=spot_data["price"])

    try:
        # Pull OVDV surface via bulk data
        surface_df = bds(bbg, "OVDV_SURF_MID")

        if surface_df.empty:
            # Fallback: build from chain
            return _build_vol_surface_from_chain(underlying, r, q)

        # Parse the OVDV surface into strikes x expiries grid
        # Bloomberg returns: Strike, Expiry, Vol
        strikes = np.sort(surface_df["Strike"].unique().astype(float))
        expiry_labels = surface_df["Expiry"].unique()
        expiries_years = np.array([_expiry_to_years(e) for e in expiry_labels])
        sort_idx = np.argsort(expiries_years)
        expiries_years = expiries_years[sort_idx]
        expiry_labels = expiry_labels[sort_idx]

        vol_matrix = np.zeros((len(expiries_years), len(strikes)))
        for i, exp in enumerate(expiry_labels):
            for j, K in enumerate(strikes):
                row = surface_df[(surface_df["Expiry"] == exp) &
                                  (surface_df["Strike"].astype(float) == K)]
                if not row.empty:
                    vol_matrix[i, j] = float(row.iloc[0]["Vol"]) / 100.0

        return strikes, expiries_years, vol_matrix

    except Exception as e:
        logger.error(f"Vol surface request failed: {e}")
        return np.array([]), np.array([]), np.array([[]])


def get_historical_prices(ticker: str, days: int = 252) -> pd.DataFrame:
    """Pull historical OHLCV data from Bloomberg."""
    start = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
    fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "VOLUME"]
    df = bdh(_to_bbg_ticker(ticker), fields, start)

    if df.empty:
        logger.warning(f"BDH empty for {ticker} — returning empty DataFrame")
        return pd.DataFrame()

    df = df.rename(columns={
        "PX_OPEN": "open", "PX_HIGH": "high", "PX_LOW": "low",
        "PX_LAST": "close", "VOLUME": "volume",
    })
    return df.tail(days)


def get_dividend_yield(ticker: str) -> float:
    """Get current dividend yield from Bloomberg."""
    df = bdp([_to_bbg_ticker(ticker)], ["EQY_DVD_YLD_IND"])
    if not df.empty:
        val = df.iloc[0, 0]
        return _safe_float(val) / 100.0 if val else 0.015
    return 0.015


def get_risk_free_rate() -> float:
    """Get current risk-free rate (3M T-Bill)."""
    df = bdp(["GB3 Govt"], ["PX_LAST"])
    if not df.empty:
        val = df.iloc[0, 0]
        return _safe_float(val) / 100.0 if val else 0.05
    return 0.05


def get_earnings_dates(ticker: str) -> List[str]:
    """Get upcoming earnings dates."""
    df = bds(_to_bbg_ticker(ticker), "EARN_ANN_DT_TIME_HIST_WITH_EPS")
    if not df.empty:
        return df.iloc[:5, 0].tolist()
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_bbg_ticker(ticker: str) -> str:
    """Convert short ticker to Bloomberg format."""
    ticker = ticker.upper().strip()
    if " " in ticker:  # Already in BBG format like "SPY US Equity"
        return ticker
    # Index ETFs and equities
    return f"{ticker} US Equity"


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _expiry_to_years(expiry_str):
    try:
        exp_date = pd.to_datetime(expiry_str)
        return max((exp_date - pd.Timestamp.now()).days / 365.0, 0.01)
    except Exception:
        return 0.25


def _build_vol_surface_from_chain(underlying, r, q):
    """Build vol surface from options chain data when OVDV isn't available."""
    from core.pricing import generate_vol_surface

    chain = get_options_chain(underlying)
    if chain.empty or "iv" not in chain.columns:
        logger.warning(f"No chain data for {underlying} — returning empty surface")
        return np.array([]), np.array([]), np.array([[]])

    # Group by expiry and build surface
    calls = chain[chain["type"].str.upper() == "CALL"]
    if calls.empty:
        logger.warning(f"No call data for {underlying} — returning empty surface")
        return np.array([]), np.array([]), np.array([[]])

    expiries = sorted(calls["expiry"].unique())
    strikes = np.sort(calls["strike"].unique().astype(float))

    expiry_years = [_expiry_to_years(e) for e in expiries]
    vol_matrix = np.full((len(expiries), len(strikes)), np.nan)

    for i, exp in enumerate(expiries):
        exp_data = calls[calls["expiry"] == exp]
        for j, K in enumerate(strikes):
            row = exp_data[exp_data["strike"].astype(float) == K]
            if not row.empty and pd.notna(row.iloc[0].get("iv")):
                vol_matrix[i, j] = float(row.iloc[0]["iv"]) / 100.0

    # Interpolate NaN gaps
    from scipy.interpolate import griddata
    valid = ~np.isnan(vol_matrix)
    if valid.sum() > 4:
        yi, xi = np.meshgrid(range(len(expiry_years)), range(len(strikes)), indexing="ij")
        vol_matrix = griddata(
            (yi[valid], xi[valid]), vol_matrix[valid],
            (yi, xi), method="cubic", fill_value=0.20,
        )

    vol_matrix = np.clip(vol_matrix, 0.02, 5.0)
    return np.array(strikes), np.array(expiry_years), vol_matrix


# ═══════════════════════════════════════════════════════════════════════════
# Fallback Data (when Bloomberg is not connected)
# ═══════════════════════════════════════════════════════════════════════════

_FALLBACK_TICKERS = {
    "SPY":  {"price": 521.40, "vol": 0.16}, "QQQ":  {"price": 446.80, "vol": 0.20},
    "AAPL": {"price": 178.50, "vol": 0.24}, "MSFT": {"price": 416.20, "vol": 0.22},
    "NVDA": {"price": 882.30, "vol": 0.45}, "TSLA": {"price": 176.40, "vol": 0.55},
    "AMZN": {"price": 186.50, "vol": 0.28}, "META": {"price": 506.70, "vol": 0.32},
    "JPM":  {"price": 198.30, "vol": 0.20}, "GS":   {"price": 468.50, "vol": 0.25},
    "XOM":  {"price": 118.40, "vol": 0.22}, "GLD":  {"price": 215.60, "vol": 0.15},
}


def _fallback_spot(ticker):
    info = _FALLBACK_TICKERS.get(ticker.upper(), {"price": 100.0, "vol": 0.20})
    rng = np.random.RandomState(hash(ticker) % 2**31)
    change = info["price"] * rng.uniform(-0.02, 0.02)
    return {
        "price": round(info["price"] + change, 2),
        "change": round(change, 2),
        "change_pct": round(change / info["price"] * 100, 2),
        "bid": round(info["price"] + change - 0.02, 2),
        "ask": round(info["price"] + change + 0.02, 2),
        "volume": int(rng.lognormal(16, 1)),
        "high": round(info["price"] + abs(change) * 1.5, 2),
        "low": round(info["price"] - abs(change) * 1.5, 2),
        "open": round(info["price"] - change * 0.3, 2),
        "mkt_cap": 0,
        "rv30": round(info["vol"] * 100, 1),
    }


def _fallback_bdp(securities, fields):
    rows = []
    for sec in securities:
        short = sec.replace(" US Equity", "").replace(" Govt", "")
        info = _fallback_spot(short)
        row = {"security": sec}
        field_map = {
            "PX_LAST": info["price"], "CHG_NET_1D": info["change"],
            "CHG_PCT_1D": info["change_pct"], "PX_BID": info["bid"],
            "PX_ASK": info["ask"], "VOLUME": info["volume"],
            "PX_HIGH": info["high"], "PX_LOW": info["low"],
            "PX_OPEN": info["open"], "CUR_MKT_CAP": info["mkt_cap"],
            "VOLATILITY_30D": info["rv30"], "EQY_DVD_YLD_IND": 1.5,
        }
        for fld in fields:
            row[fld] = field_map.get(fld, None)
        rows.append(row)
    return pd.DataFrame(rows).set_index("security")


def _fallback_bdh(security, fields, start_date, end_date):
    short = security.replace(" US Equity", "").replace(" Govt", "")
    return _fallback_historical(short, 252)


def _fallback_historical(ticker, days=252):
    from core.pricing import generate_price_history
    info = _FALLBACK_TICKERS.get(ticker.upper(), {"price": 100.0, "vol": 0.20})
    prices = generate_price_history(S=info["price"], sigma=info["vol"], days=days,
                                     seed=hash(ticker) % 2**31)
    rng = np.random.RandomState(hash(ticker) % 2**31)
    n = len(prices)
    dates = pd.date_range(end=datetime.now(), periods=n, freq="B")
    return pd.DataFrame({
        "open": prices * (1 + rng.uniform(-0.005, 0.005, n)),
        "high": prices * (1 + rng.uniform(0, 0.015, n)),
        "low": prices * (1 - rng.uniform(0, 0.015, n)),
        "close": prices,
        "volume": rng.lognormal(16, 1, n).astype(int),
    }, index=dates)


def _fallback_options_chain(underlying, expiry=None):
    from core.pricing import generate_options_chain
    info = _FALLBACK_TICKERS.get(underlying.upper(), {"price": 100.0, "vol": 0.20})
    chain = generate_options_chain(
        S=info["price"], r=0.05, q=0.015,
        base_vol=info["vol"], expiry_days=30,
        strike_step=max(1, round(info["price"] / 40)),
        num_strikes=25,
    )
    return pd.DataFrame(chain)
