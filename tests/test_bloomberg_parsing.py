"""
Tests for the blpapi RESPONSE PARSING layer (core/bloomberg.py: bdp/bdh/bds).

This is the one layer that normally only runs against a live Bloomberg terminal.
Here we mock the blpapi SDK's Element/Message objects with faithful stand-ins and
push synthetic-but-realistic Bloomberg responses through the REAL parsers, then
assert they produce the expected DataFrames. This closes the gap of "is the
response parsing correct" without needing a live server (the only thing left to a
live terminal is whether the server is actually connected and serving — an
operational fact, not a code-correctness one).

blpapi datatype codes used by _extract_value:
  BOOL=1, INT32=2, INT64=3, FLOAT32=5, FLOAT64=6, STRING=8, DATE=10, DATETIME=12
"""
from datetime import datetime

import pandas as pd
import pytest

import core.bloomberg as bbg
from core.bloomberg import bdp, bdh, bds, _extract_value


# ---------------------------------------------------------------------------
# Faithful blpapi Element / Message / Session stand-ins
# ---------------------------------------------------------------------------
class FE:
    """Stand-in for a blpapi Element supporting the full surface the parsers use:
    named children (getElement(str)/hasElement/getElementAsString), arrays
    (numValues/getValueAsElement), indexed children (numElements/getElement(int)/
    name) and scalars (datatype/getValueAs*)."""

    def __init__(self, name="", children=None, values=None, scalar=None, dtype=8,
                 is_null=False):
        self._name = name
        self._children = children or {}
        self._ordered = list((children or {}).items())
        self._values = values
        self._scalar = scalar
        self._dtype = dtype
        self._is_null = is_null

    # scalar API
    def isNull(self):
        return self._is_null or (self._scalar is None and not self._children
                                 and self._values is None)
    def datatype(self):
        return self._dtype
    def getValueAsFloat(self):
        return float(self._scalar)
    def getValueAsInteger(self):
        return int(self._scalar)
    def getValueAsBool(self):
        return bool(self._scalar)
    def getValueAsString(self):
        return str(self._scalar)
    def getValueAsDatetime(self):
        return self._scalar

    # named-children API
    def hasElement(self, n):
        return n in self._children
    def getElement(self, key):
        if isinstance(key, int):
            return self._ordered[key][1]
        return self._children[key]
    def getElementAsString(self, n):
        return str(self._children[n]._scalar)
    def getElementAsFloat(self, n):
        return float(self._children[n]._scalar)
    def numElements(self):
        return len(self._ordered)
    def name(self):
        return self._name

    # array API
    def numValues(self):
        return len(self._values) if self._values is not None else 0
    def getValueAsElement(self, i):
        return self._values[i]


def _dtype_for(v):
    if isinstance(v, bool):
        return 1
    if isinstance(v, int):
        return 2
    if isinstance(v, float):
        return 6
    if hasattr(v, "year"):
        return 12
    return 8


def sc(name, v):
    return FE(name=name, scalar=v, dtype=_dtype_for(v))


def container(name="", **named):
    return FE(name=name, children={k: (v if isinstance(v, FE) else sc(k, v))
                                   for k, v in named.items()})


class _FakeReq:
    def append(self, *a, **k):
        pass
    def set(self, *a, **k):
        pass
    def getElement(self, *a, **k):
        return _FakeReq()
    def appendElement(self):
        return _FakeReq()
    def setElement(self, *a, **k):
        pass


class _FakeService:
    def createRequest(self, name):
        return _FakeReq()


class _FakeConn:
    def __init__(self, messages):
        self.connected = True
        self.ref_data_service = _FakeService()
        self._messages = messages
    def _send_request(self, request):
        return self._messages


@pytest.fixture
def patch_conn(monkeypatch):
    def _install(messages):
        monkeypatch.setattr(bbg, "get_connection", lambda: _FakeConn(messages))
        monkeypatch.setattr(bbg, "_bloomberg_ever_connected", True, raising=False)
    return _install


# ---------------------------------------------------------------------------
# _extract_value: datatype dispatch
# ---------------------------------------------------------------------------
class TestExtractValue:
    def test_float(self):
        assert _extract_value(FE(scalar=1.0855, dtype=6)) == pytest.approx(1.0855)
    def test_int(self):
        assert _extract_value(FE(scalar=42, dtype=2)) == 42
    def test_string(self):
        assert _extract_value(FE(scalar="EURUSD", dtype=8)) == "EURUSD"
    def test_bool(self):
        assert _extract_value(FE(scalar=True, dtype=1)) is True
    def test_datetime(self):
        d = datetime(2026, 6, 1)
        assert _extract_value(FE(scalar=d, dtype=12)) == d
    def test_null(self):
        assert _extract_value(FE(is_null=True)) is None


# ---------------------------------------------------------------------------
# bdp — ReferenceDataRequest (spots, vol points, rates)
# ---------------------------------------------------------------------------
class TestBDP:
    def _msg(self, secs):
        # secs: list of (ticker, {field: value})
        sec_vals = []
        for ticker, fields in secs:
            field_data = FE(name="fieldData",
                            children={f: sc(f, v) for f, v in fields.items()})
            sec_vals.append(FE(name="security", children={
                "security": sc("security", ticker),
                "fieldData": field_data,
            }))
        security_data = FE(name="securityData", values=sec_vals)
        return FE(children={"securityData": security_data})

    def test_parses_points(self, patch_conn):
        patch_conn([self._msg([
            ("EURUSD Curncy", {"PX_BID": 1.0849, "PX_ASK": 1.0851, "PX_LAST": 1.0850}),
            ("USDJPY Curncy", {"PX_BID": 150.10, "PX_ASK": 150.14, "PX_LAST": 150.12}),
        ])])
        df = bdp(["EURUSD Curncy", "USDJPY Curncy"], ["PX_BID", "PX_ASK", "PX_LAST"])
        assert list(df.index) == ["EURUSD Curncy", "USDJPY Curncy"]
        assert df.loc["EURUSD Curncy", "PX_LAST"] == pytest.approx(1.0850)
        assert df.loc["USDJPY Curncy", "PX_BID"] == pytest.approx(150.10)
        assert set(df.columns) == {"PX_BID", "PX_ASK", "PX_LAST"}

    def test_missing_field_is_none(self, patch_conn):
        patch_conn([self._msg([("EURUSD Curncy", {"PX_LAST": 1.085})])])
        df = bdp(["EURUSD Curncy"], ["PX_LAST", "PX_BID"])
        assert df.loc["EURUSD Curncy", "PX_LAST"] == pytest.approx(1.085)
        assert df.loc["EURUSD Curncy", "PX_BID"] is None

    def test_security_error_skipped(self, patch_conn):
        bad = FE(name="security", children={
            "security": sc("security", "BADTICKER"),
            "securityError": container("securityError", message="Invalid security"),
        })
        good = FE(name="security", children={
            "security": sc("security", "EURUSD Curncy"),
            "fieldData": FE(name="fieldData", children={"PX_LAST": sc("PX_LAST", 1.085)}),
        })
        patch_conn([FE(children={"securityData": FE(name="securityData", values=[bad, good])})])
        df = bdp(["BADTICKER", "EURUSD Curncy"], ["PX_LAST"])
        assert list(df.index) == ["EURUSD Curncy"]  # bad one skipped, not fabricated

    def test_disconnected_returns_empty(self, monkeypatch):
        class Down:
            connected = False
            ref_data_service = None
        monkeypatch.setattr(bbg, "get_connection", lambda: Down())
        monkeypatch.setattr(bbg, "_bloomberg_ever_connected", True, raising=False)
        df = bdp(["EURUSD Curncy"], ["PX_LAST"])
        assert df.empty  # no fabricated data on disconnect

    def test_response_error_returns_empty(self, patch_conn):
        patch_conn([FE(children={"responseError": container("responseError",
                                                            message="Service unavailable")})])
        df = bdp(["EURUSD Curncy"], ["PX_LAST"])
        assert df.empty


# ---------------------------------------------------------------------------
# bdh — HistoricalDataRequest (spot/vol time series)
# ---------------------------------------------------------------------------
class TestBDH:
    def _msg(self, ticker, bars):
        # bars: list of dicts incl 'date' (datetime) + fields
        fd_vals = []
        for bar in bars:
            fd_vals.append(FE(name="fieldData",
                              children={k: sc(k, v) for k, v in bar.items()}))
        field_data_array = FE(name="fieldData", values=fd_vals)
        security_data = FE(name="securityData", children={
            "security": sc("security", ticker),
            "fieldData": field_data_array,
        })
        return FE(children={"securityData": security_data})

    def test_parses_timeseries(self, patch_conn):
        bars = [
            {"date": datetime(2026, 5, 28), "PX_LAST": 1.0840},
            {"date": datetime(2026, 5, 29), "PX_LAST": 1.0852},
            {"date": datetime(2026, 6, 1), "PX_LAST": 1.0861},
        ]
        patch_conn([self._msg("EURUSD Curncy", bars)])
        df = bdh("EURUSD Curncy", ["PX_LAST"], "20260528", "20260601")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 3
        assert list(df["PX_LAST"]) == pytest.approx([1.0840, 1.0852, 1.0861])
        assert df.index[0] == pd.Timestamp("2026-05-28")

    def test_empty_history(self, patch_conn):
        patch_conn([self._msg("EURUSD Curncy", [])])
        df = bdh("EURUSD Curncy", ["PX_LAST"], "20260101")
        assert df.empty


# ---------------------------------------------------------------------------
# bds — bulk (e.g. options chain members)
# ---------------------------------------------------------------------------
class TestBDS:
    def _msg(self, security, field, rows):
        # rows: list of dicts (ordered field -> value)
        bulk_vals = []
        for r in rows:
            bulk_vals.append(FE(name="row",
                                children={k: sc(k, v) for k, v in r.items()}))
        bulk = FE(name=field, values=bulk_vals)
        sec = FE(name="security", children={
            "fieldData": FE(name="fieldData", children={field: bulk}),
        })
        return FE(children={"securityData": FE(name="securityData", values=[sec])})

    def test_parses_bulk(self, patch_conn):
        rows = [
            {"Strike": 1.08, "Ticker": "EURUSD 1.08 C"},
            {"Strike": 1.10, "Ticker": "EURUSD 1.10 C"},
        ]
        patch_conn([self._msg("EURUSD Curncy", "OPT_CHAIN", rows)])
        df = bds("EURUSD Curncy", "OPT_CHAIN")
        assert len(df) == 2
        assert set(df.columns) == {"Strike", "Ticker"}
        assert df.iloc[0]["Strike"] == pytest.approx(1.08)
        assert df.iloc[1]["Ticker"] == "EURUSD 1.10 C"

    def test_disconnected_returns_empty(self, monkeypatch):
        class Down:
            connected = False
            ref_data_service = None
        monkeypatch.setattr(bbg, "get_connection", lambda: Down())
        df = bds("EURUSD Curncy", "OPT_CHAIN")
        assert df.empty


# ---------------------------------------------------------------------------
# FULL CHAIN: simulated blpapi wire -> bdp parse -> get_fx_spots transform ->
# panel-facing dict. This is the complete live-data path end to end.
# ---------------------------------------------------------------------------
def test_end_to_end_get_fx_spots_from_simulated_bloomberg(monkeypatch):
    import core.bloomberg_fx as bfx

    ticker = bfx._fx_bbg_ticker("EURUSD")  # exact ticker the getter requests
    field_data = FE(name="fieldData", children={
        "PX_BID": sc("PX_BID", 1.0849), "PX_ASK": sc("PX_ASK", 1.0851),
        "PX_LAST": sc("PX_LAST", 1.0850), "PX_MID": sc("PX_MID", 1.0850),
        "CHG_NET_1D": sc("CHG_NET_1D", 0.0012), "CHG_PCT_1D": sc("CHG_PCT_1D", 0.11),
        "PX_HIGH": sc("PX_HIGH", 1.0875), "PX_LOW": sc("PX_LOW", 1.0820),
        "PX_OPEN": sc("PX_OPEN", 1.0838),
    })
    sec = FE(name="security", children={
        "security": sc("security", ticker), "fieldData": field_data})
    msg = FE(children={"securityData": FE(name="securityData", values=[sec])})

    monkeypatch.setattr(bbg, "get_connection", lambda: _FakeConn([msg]))
    monkeypatch.setattr(bbg, "_bloomberg_ever_connected", True, raising=False)
    monkeypatch.setattr(bfx, "is_connected", lambda: True)
    monkeypatch.setattr(bfx, "_HAS_EQUITY_BBG", True, raising=False)
    monkeypatch.setattr(bfx, "_CACHE_ONLY_MODE", False, raising=False)
    bfx._cache_invalidate_prefix("spots_")  # ensure a live fetch, not a cache hit

    out = bfx.get_fx_spots(["EURUSD"])
    assert "EURUSD" in out, "transform dropped the pair"
    rec = out["EURUSD"]
    assert rec["bid"] == pytest.approx(1.0849)
    assert rec["ask"] == pytest.approx(1.0851)
    assert rec["mid"] == pytest.approx(1.0850, abs=1e-4)
    bfx._cache_invalidate_prefix("spots_")  # don't leak into other tests
