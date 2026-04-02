"""
Unit tests for core/fx_portfolio.py — FX Options Portfolio Engine.

Tests cover:
  - Position validation (_validate_position)
  - Adding positions (add_position) with defaults
  - Closing positions (close_position)
  - Partial closes (partial_close)
  - Rolling positions (roll_position)
  - Querying positions (get_positions, get_all_positions)
  - Portfolio state (get_portfolio, create_sample_portfolio)
  - GK pricing helpers (_gk_d1d2, _gk_price, _gk_greeks)
  - Tenor bucketing (_tenor_bucket)
  - Time to expiry (_years_to_expiry)
  - Vol lookup (_lookup_vol)
  - Rate lookup (_get_rate)
  - Risk limits checking (check_risk_limits)
  - Hedge suggestions (hedge_suggestion)
  - Exposure summary (exposure_summary)
  - P&L attribution (pnl_attribution)
  - JSON persistence (save_portfolio / load_portfolio)
  - Portfolio deep copy safety (get_portfolio)
"""

import json
import os
from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest

from core.fx_portfolio import (
    BOOKS,
    DEFAULT_RISK_LIMITS,
    TENOR_BUCKETS,
    _gk_d1d2,
    _gk_greeks,
    _gk_price,
    _get_rate,
    _lookup_vol,
    _tenor_bucket,
    _validate_position,
    _years_to_expiry,
    add_position,
    check_risk_limits,
    close_position,
    compute_position_greeks,
    create_sample_portfolio,
    exposure_summary,
    get_all_positions,
    get_portfolio,
    get_positions,
    hedge_suggestion,
    load_portfolio,
    partial_close,
    pnl_attribution,
    roll_position,
    save_portfolio,
    what_if_add,
)
import core.fx_portfolio as portfolio_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_position(**overrides):
    """Build a minimal valid position dict with sensible defaults."""
    pos = {
        "pair": "EURUSD",
        "option_type": "call",
        "direction": "buy",
        "strike": 1.1000,
        "expiry": (date.today() + timedelta(days=90)).isoformat(),
        "notional": 1_000_000,
        "book": "G10_FLOW",
    }
    pos.update(overrides)
    return pos


@pytest.fixture(autouse=True)
def reset_portfolio():
    """Reset the global _PORTFOLIO before each test so tests are isolated."""
    create_sample_portfolio()
    yield
    create_sample_portfolio()


# ============================================================================
# _validate_position
# ============================================================================

class TestValidatePosition:

    def test_valid_position_passes(self):
        pos = _make_position()
        _validate_position(pos)  # should not raise

    @pytest.mark.parametrize("missing_field", [
        "pair", "option_type", "direction", "strike", "expiry", "notional", "book",
    ])
    def test_missing_required_field(self, missing_field):
        pos = _make_position()
        del pos[missing_field]
        with pytest.raises(ValueError, match="Missing required field"):
            _validate_position(pos)

    def test_invalid_option_type(self):
        pos = _make_position(option_type="straddle")
        with pytest.raises(ValueError, match="option_type must be"):
            _validate_position(pos)

    def test_invalid_direction(self):
        pos = _make_position(direction="hold")
        with pytest.raises(ValueError, match="direction must be"):
            _validate_position(pos)

    def test_negative_strike(self):
        pos = _make_position(strike=-1.10)
        with pytest.raises(ValueError, match="strike must be positive"):
            _validate_position(pos)

    def test_zero_strike(self):
        pos = _make_position(strike=0)
        with pytest.raises(ValueError, match="strike must be positive"):
            _validate_position(pos)

    def test_negative_notional(self):
        pos = _make_position(notional=-500_000)
        with pytest.raises(ValueError, match="notional must be positive"):
            _validate_position(pos)

    def test_zero_notional(self):
        pos = _make_position(notional=0)
        with pytest.raises(ValueError, match="notional must be positive"):
            _validate_position(pos)

    def test_put_option_type_valid(self):
        pos = _make_position(option_type="put")
        _validate_position(pos)  # should not raise

    def test_sell_direction_valid(self):
        pos = _make_position(direction="sell")
        _validate_position(pos)  # should not raise


# ============================================================================
# add_position
# ============================================================================

class TestAddPosition:

    def test_returns_uuid_string(self):
        pos_id = add_position("G10_FLOW", _make_position())
        assert isinstance(pos_id, str)
        assert len(pos_id) == 36  # UUID4 format

    def test_position_appears_in_book(self):
        add_position("G10_FLOW", _make_position())
        positions = get_positions(book="G10_FLOW")
        assert len(positions) == 1
        assert positions[0]["pair"] == "EURUSD"

    def test_default_fields_populated(self):
        add_position("G10_PROP", _make_position())
        positions = get_positions(book="G10_PROP")
        pos = positions[0]
        assert pos["status"] == "open"
        assert pos["notional_ccy"] == "EUR"  # first 3 chars of EURUSD
        assert pos["delta_at_entry"] == 0.0
        assert pos["entry_vol"] == 0.10
        assert pos["entry_premium"] == 0.0
        assert pos["cut"] == "NY"
        assert pos["counterparty"] == "INTERBANK"
        assert pos["strategy"] == "PROP"
        assert pos["tags"] == []
        assert pos["notes"] == ""

    def test_custom_fields_not_overridden(self):
        pos = _make_position(cut="TOK", counterparty="JPM", strategy="RV")
        add_position("G10_FLOW", pos)
        result = get_positions(book="G10_FLOW")[0]
        assert result["cut"] == "TOK"
        assert result["counterparty"] == "JPM"
        assert result["strategy"] == "RV"

    def test_notional_ccy_from_pair(self):
        pos = _make_position(pair="GBPUSD")
        add_position("G10_FLOW", pos)
        result = get_positions(book="G10_FLOW")[0]
        assert result["notional_ccy"] == "GBP"

    def test_position_added_to_new_book(self):
        """Adding to a book not in BOOKS should dynamically create it."""
        pos_id = add_position("CUSTOM_BOOK", _make_position())
        positions = get_positions(book="CUSTOM_BOOK")
        assert len(positions) == 1
        assert positions[0]["id"] == pos_id

    def test_trade_history_recorded(self):
        add_position("G10_FLOW", _make_position())
        portfolio = get_portfolio()
        assert len(portfolio["trade_history"]) == 1
        entry = portfolio["trade_history"][0]
        assert entry["action"] == "OPEN"
        assert entry["book"] == "G10_FLOW"

    def test_invalid_position_raises(self):
        bad_pos = _make_position()
        del bad_pos["pair"]
        with pytest.raises(ValueError):
            add_position("G10_FLOW", bad_pos)

    def test_multiple_positions_in_same_book(self):
        add_position("G10_FLOW", _make_position(pair="EURUSD"))
        add_position("G10_FLOW", _make_position(pair="GBPUSD"))
        positions = get_positions(book="G10_FLOW")
        assert len(positions) == 2
        pairs = {p["pair"] for p in positions}
        assert pairs == {"EURUSD", "GBPUSD"}


# ============================================================================
# close_position
# ============================================================================

class TestClosePosition:

    def test_close_returns_true(self):
        pos_id = add_position("G10_FLOW", _make_position())
        result = close_position("G10_FLOW", pos_id, close_price=0.05)
        assert result is True

    def test_closed_position_status(self):
        pos_id = add_position("G10_FLOW", _make_position())
        close_position("G10_FLOW", pos_id, close_price=0.05)
        # Should no longer appear in open positions
        open_positions = get_positions(book="G10_FLOW", status="open")
        assert len(open_positions) == 0
        # Should appear in closed positions
        closed = get_positions(book="G10_FLOW", status="closed")
        assert len(closed) == 1
        assert closed[0]["close_price"] == 0.05

    def test_close_default_date(self):
        pos_id = add_position("G10_FLOW", _make_position())
        close_position("G10_FLOW", pos_id)
        closed = get_positions(book="G10_FLOW", status="closed")
        assert closed[0]["close_date"] == date.today().isoformat()

    def test_close_custom_date(self):
        pos_id = add_position("G10_FLOW", _make_position())
        close_position("G10_FLOW", pos_id, close_date="2025-06-15")
        closed = get_positions(book="G10_FLOW", status="closed")
        assert closed[0]["close_date"] == "2025-06-15"

    def test_close_nonexistent_returns_false(self):
        result = close_position("G10_FLOW", "nonexistent-id-123")
        assert result is False

    def test_close_wrong_book_returns_false(self):
        pos_id = add_position("G10_FLOW", _make_position())
        result = close_position("G10_PROP", pos_id)
        assert result is False

    def test_close_already_closed_returns_false(self):
        pos_id = add_position("G10_FLOW", _make_position())
        close_position("G10_FLOW", pos_id)
        result = close_position("G10_FLOW", pos_id)
        assert result is False

    def test_close_adds_trade_history(self):
        pos_id = add_position("G10_FLOW", _make_position())
        close_position("G10_FLOW", pos_id)
        portfolio = get_portfolio()
        close_entries = [h for h in portfolio["trade_history"] if h["action"] == "CLOSE"]
        assert len(close_entries) == 1
        assert close_entries[0]["position_id"] == pos_id


# ============================================================================
# partial_close
# ============================================================================

class TestPartialClose:

    def test_partial_close_reduces_notional(self):
        pos_id = add_position("G10_FLOW", _make_position(notional=1_000_000))
        result = partial_close("G10_FLOW", pos_id, 400_000)
        assert result is True
        positions = get_positions(book="G10_FLOW")
        assert len(positions) == 1
        assert positions[0]["notional"] == 600_000

    def test_partial_close_full_amount_closes(self):
        pos_id = add_position("G10_FLOW", _make_position(notional=1_000_000))
        result = partial_close("G10_FLOW", pos_id, 1_000_000)
        assert result is True
        open_positions = get_positions(book="G10_FLOW", status="open")
        assert len(open_positions) == 0

    def test_partial_close_exceeding_notional_closes(self):
        pos_id = add_position("G10_FLOW", _make_position(notional=500_000))
        result = partial_close("G10_FLOW", pos_id, 1_000_000)
        assert result is True
        open_positions = get_positions(book="G10_FLOW", status="open")
        assert len(open_positions) == 0

    def test_partial_close_nonexistent_returns_false(self):
        result = partial_close("G10_FLOW", "bad-id", 100_000)
        assert result is False

    def test_partial_close_trade_history(self):
        pos_id = add_position("G10_FLOW", _make_position(notional=1_000_000))
        partial_close("G10_FLOW", pos_id, 300_000)
        portfolio = get_portfolio()
        pc_entries = [h for h in portfolio["trade_history"] if h["action"] == "PARTIAL_CLOSE"]
        assert len(pc_entries) == 1
        assert pc_entries[0]["closed_notional"] == 300_000
        assert pc_entries[0]["remaining_notional"] == 700_000

    def test_successive_partial_closes(self):
        pos_id = add_position("G10_FLOW", _make_position(notional=1_000_000))
        partial_close("G10_FLOW", pos_id, 200_000)
        partial_close("G10_FLOW", pos_id, 300_000)
        positions = get_positions(book="G10_FLOW", status="open")
        assert len(positions) == 1
        assert positions[0]["notional"] == 500_000


# ============================================================================
# roll_position
# ============================================================================

class TestRollPosition:

    def test_roll_creates_new_position(self):
        pos_id = add_position("G10_FLOW", _make_position(strike=1.10))
        new_expiry = (date.today() + timedelta(days=180)).isoformat()
        new_id = roll_position("G10_FLOW", pos_id, new_expiry)
        assert new_id is not None
        assert new_id != pos_id

    def test_roll_closes_old_position(self):
        pos_id = add_position("G10_FLOW", _make_position())
        new_expiry = (date.today() + timedelta(days=180)).isoformat()
        roll_position("G10_FLOW", pos_id, new_expiry)
        closed = get_positions(book="G10_FLOW", status="closed")
        assert len(closed) == 1
        assert closed[0]["id"] == pos_id

    def test_roll_new_position_has_updated_expiry(self):
        pos_id = add_position("G10_FLOW", _make_position())
        new_expiry = (date.today() + timedelta(days=180)).isoformat()
        new_id = roll_position("G10_FLOW", pos_id, new_expiry)
        open_positions = get_positions(book="G10_FLOW", status="open")
        assert len(open_positions) == 1
        assert open_positions[0]["id"] == new_id
        assert open_positions[0]["expiry"] == new_expiry

    def test_roll_with_new_strike(self):
        pos_id = add_position("G10_FLOW", _make_position(strike=1.10))
        new_expiry = (date.today() + timedelta(days=180)).isoformat()
        new_id = roll_position("G10_FLOW", pos_id, new_expiry, new_strike=1.15)
        positions = get_positions(book="G10_FLOW", status="open")
        assert positions[0]["strike"] == 1.15

    def test_roll_nonexistent_returns_none(self):
        result = roll_position("G10_FLOW", "bad-id", "2025-12-31")
        assert result is None

    def test_roll_preserves_other_fields(self):
        pos = _make_position(pair="USDJPY", option_type="put", direction="sell",
                             notional=5_000_000)
        pos_id = add_position("G10_PROP", pos)
        new_expiry = (date.today() + timedelta(days=270)).isoformat()
        roll_position("G10_PROP", pos_id, new_expiry)
        open_pos = get_positions(book="G10_PROP", status="open")
        assert open_pos[0]["pair"] == "USDJPY"
        assert open_pos[0]["option_type"] == "put"
        assert open_pos[0]["direction"] == "sell"
        assert open_pos[0]["notional"] == 5_000_000


# ============================================================================
# get_positions / get_all_positions
# ============================================================================

class TestGetPositions:

    def test_empty_portfolio_returns_empty(self):
        assert get_positions() == []
        assert get_all_positions() == []

    def test_filter_by_book(self):
        add_position("G10_FLOW", _make_position(pair="EURUSD"))
        add_position("G10_PROP", _make_position(pair="GBPUSD"))
        flow = get_positions(book="G10_FLOW")
        assert len(flow) == 1
        assert flow[0]["pair"] == "EURUSD"

    def test_filter_by_pair(self):
        add_position("G10_FLOW", _make_position(pair="EURUSD"))
        add_position("G10_FLOW", _make_position(pair="GBPUSD"))
        eur = get_positions(pair="EURUSD")
        assert len(eur) == 1
        assert eur[0]["pair"] == "EURUSD"

    def test_filter_by_status(self):
        pos_id = add_position("G10_FLOW", _make_position())
        add_position("G10_FLOW", _make_position(pair="GBPUSD"))
        close_position("G10_FLOW", pos_id)
        open_pos = get_positions(status="open")
        assert len(open_pos) == 1
        closed_pos = get_positions(status="closed")
        assert len(closed_pos) == 1

    def test_get_all_positions_across_books(self):
        add_position("G10_FLOW", _make_position(pair="EURUSD"))
        add_position("G10_PROP", _make_position(pair="GBPUSD"))
        add_position("EM_FLOW", _make_position(pair="USDMXN"))
        all_pos = get_all_positions()
        assert len(all_pos) == 3

    def test_returns_deep_copies(self):
        add_position("G10_FLOW", _make_position())
        positions = get_positions(book="G10_FLOW")
        positions[0]["notional"] = 999
        # Original should be unchanged
        original = get_positions(book="G10_FLOW")
        assert original[0]["notional"] == 1_000_000

    def test_status_none_returns_all(self):
        pos_id = add_position("G10_FLOW", _make_position())
        add_position("G10_FLOW", _make_position(pair="GBPUSD"))
        close_position("G10_FLOW", pos_id)
        all_pos = get_positions(book="G10_FLOW", status=None)
        assert len(all_pos) == 2


# ============================================================================
# get_portfolio / create_sample_portfolio
# ============================================================================

class TestPortfolioState:

    def test_create_sample_portfolio_has_all_books(self):
        portfolio = create_sample_portfolio()
        for book in BOOKS:
            assert book in portfolio["books"]

    def test_create_sample_portfolio_empty_books(self):
        portfolio = create_sample_portfolio()
        for book in BOOKS:
            assert portfolio["books"][book] == []

    def test_create_sample_portfolio_has_risk_limits(self):
        portfolio = create_sample_portfolio()
        assert portfolio["risk_limits"] == DEFAULT_RISK_LIMITS

    def test_get_portfolio_returns_deep_copy(self):
        add_position("G10_FLOW", _make_position())
        p1 = get_portfolio()
        p1["books"]["G10_FLOW"] = []  # mutate copy
        p2 = get_portfolio()
        assert len(p2["books"]["G10_FLOW"]) == 1  # original unaffected


# ============================================================================
# GK Pricing Helpers
# ============================================================================

class TestGKPricing:

    def test_d1d2_returns_two_floats(self):
        d1, d2 = _gk_d1d2(1.10, 1.10, 0.25, 0.04, 0.02, 0.10)
        assert isinstance(d1, float)
        assert isinstance(d2, float)
        assert d1 > d2  # d1 = d2 + sigma*sqrt(T) > d2

    def test_gk_price_call_positive(self):
        price = _gk_price(1.10, 1.10, 0.25, 0.04, 0.02, 0.10, 1)
        assert price > 0

    def test_gk_price_put_positive(self):
        price = _gk_price(1.10, 1.10, 0.25, 0.04, 0.02, 0.10, -1)
        assert price > 0

    def test_gk_put_call_parity(self):
        """GK parity: C - P = S*exp(-r_f*T) - K*exp(-r_d*T)"""
        S, K, T, r_d, r_f, sigma = 1.10, 1.12, 0.25, 0.04, 0.02, 0.08
        call = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        put = _gk_price(S, K, T, r_d, r_f, sigma, -1)
        parity = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)
        assert abs((call - put) - parity) < 1e-10

    def test_gk_price_zero_time_call_itm(self):
        price = _gk_price(1.20, 1.10, 0.0, 0.04, 0.02, 0.10, 1)
        assert abs(price - 0.10) < 1e-8

    def test_gk_price_zero_time_call_otm(self):
        price = _gk_price(1.00, 1.10, 0.0, 0.04, 0.02, 0.10, 1)
        assert abs(price) < 1e-10

    def test_gk_greeks_returns_all_keys(self):
        greeks = _gk_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.10, 1)
        expected_keys = {"price", "delta", "gamma", "vega", "theta",
                         "rho_d", "rho_f", "vanna", "volga"}
        assert set(greeks.keys()) == expected_keys

    def test_gk_greeks_call_delta_between_0_and_1(self):
        greeks = _gk_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.10, 1)
        assert 0 < greeks["delta"] < 1

    def test_gk_greeks_put_delta_between_neg1_and_0(self):
        greeks = _gk_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.10, -1)
        assert -1 < greeks["delta"] < 0

    def test_gk_greeks_gamma_positive(self):
        greeks = _gk_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.10, 1)
        assert greeks["gamma"] > 0

    def test_gk_greeks_vega_positive(self):
        greeks = _gk_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.10, 1)
        assert greeks["vega"] > 0

    def test_gk_greeks_zero_time_expiry(self):
        greeks = _gk_greeks(1.10, 1.05, 0.0, 0.04, 0.02, 0.10, 1)
        assert greeks["gamma"] == 0
        assert greeks["vega"] == 0
        assert greeks["price"] == pytest.approx(0.05, abs=1e-6)


# ============================================================================
# _tenor_bucket
# ============================================================================

class TestTenorBucket:

    @pytest.mark.parametrize("days, expected", [
        (7, "0-1M"),
        (30, "0-1M"),
        (31, "0-1M"),
        (32, "1-3M"),
        (60, "1-3M"),
        (91, "1-3M"),
        (92, "3-6M"),
        (120, "3-6M"),
        (182, "3-6M"),
        (183, "6-12M"),
        (365, "6-12M"),
        (366, "1-2Y"),
        (730, "1-2Y"),
        (731, "2-5Y"),
        (1500, "2-5Y"),
    ])
    def test_bucket_mapping(self, days, expected):
        T = days / 365.0
        assert _tenor_bucket(T) == expected


# ============================================================================
# _years_to_expiry
# ============================================================================

class TestYearsToExpiry:

    def test_numeric_passthrough(self):
        assert _years_to_expiry(0.5) == 0.5
        assert _years_to_expiry(1) == 1.0

    def test_date_string(self):
        future = (date.today() + timedelta(days=365)).isoformat()
        T = _years_to_expiry(future)
        assert abs(T - 1.0) < 0.01

    def test_past_date_returns_minimum(self):
        past = (date.today() - timedelta(days=10)).isoformat()
        T = _years_to_expiry(past)
        assert T == pytest.approx(1e-6)

    def test_invalid_string_returns_fallback(self):
        T = _years_to_expiry("not-a-date")
        assert T == 0.25

    def test_none_returns_fallback(self):
        T = _years_to_expiry(None)
        assert T == 0.25


# ============================================================================
# _lookup_vol
# ============================================================================

class TestLookupVol:

    def test_flat_vol(self):
        surfaces = {"EURUSD": 0.08}
        vol = _lookup_vol("EURUSD", 1.10, 0.25, 1.10, surfaces)
        assert vol == 0.08

    def test_callable_surface(self):
        surfaces = {"EURUSD": lambda K, T: 0.10 + 0.01 * T}
        vol = _lookup_vol("EURUSD", 1.10, 0.5, 1.10, surfaces)
        assert vol == pytest.approx(0.105)

    def test_dict_with_atm(self):
        surfaces = {"EURUSD": {"atm": 0.09}}
        vol = _lookup_vol("EURUSD", 1.10, 0.25, 1.10, surfaces)
        assert vol == 0.09

    def test_missing_pair_returns_default(self):
        surfaces = {"GBPUSD": 0.07}
        vol = _lookup_vol("EURUSD", 1.10, 0.25, 1.10, surfaces)
        assert vol == 0.10

    def test_empty_surface_returns_default(self):
        vol = _lookup_vol("EURUSD", 1.10, 0.25, 1.10, {})
        assert vol == 0.10

    def test_dict_with_ATM_key(self):
        surfaces = {"EURUSD": {"ATM": 0.075}}
        vol = _lookup_vol("EURUSD", 1.10, 0.25, 1.10, surfaces)
        assert vol == 0.075


# ============================================================================
# _get_rate
# ============================================================================

class TestGetRate:

    def test_scalar_rate(self):
        assert _get_rate("EURUSD", 0.05, "domestic") == 0.05
        assert _get_rate("EURUSD", 0.05, "foreign") == 0.05

    def test_tuple_rate(self):
        rates = {"EURUSD": (0.04, 0.03)}
        assert _get_rate("EURUSD", rates, "domestic") == 0.04
        assert _get_rate("EURUSD", rates, "foreign") == 0.03

    def test_dict_rate(self):
        rates = {"EURUSD": {"r_d": 0.045, "r_f": 0.025}}
        assert _get_rate("EURUSD", rates, "domestic") == 0.045
        assert _get_rate("EURUSD", rates, "foreign") == 0.025

    def test_missing_pair_returns_default(self):
        rates = {"GBPUSD": (0.04, 0.03)}
        assert _get_rate("EURUSD", rates, "domestic") == 0.04  # default

    def test_single_element_tuple(self):
        rates = {"EURUSD": (0.05,)}
        assert _get_rate("EURUSD", rates, "domestic") == 0.05

    def test_empty_tuple_returns_default(self):
        rates = {"EURUSD": ()}
        assert _get_rate("EURUSD", rates, "domestic") == 0.04


# ============================================================================
# Persistence: save_portfolio / load_portfolio
# ============================================================================

class TestPersistence:

    def test_save_and_load_round_trip(self, tmp_path):
        filepath = str(tmp_path / "test_portfolio.json")
        add_position("G10_FLOW", _make_position(pair="EURUSD"))
        add_position("G10_PROP", _make_position(pair="GBPUSD"))
        save_portfolio(filepath)

        # Reset and load
        create_sample_portfolio()
        assert len(get_all_positions()) == 0

        load_portfolio(filepath)
        positions = get_all_positions()
        assert len(positions) == 2

    def test_saved_file_is_valid_json(self, tmp_path):
        filepath = str(tmp_path / "test_portfolio.json")
        add_position("G10_FLOW", _make_position())
        save_portfolio(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        assert "books" in data
        assert "trade_history" in data
        assert "risk_limits" in data

    def test_load_nonexistent_file_returns_empty(self, tmp_path):
        filepath = str(tmp_path / "nonexistent.json")
        result = load_portfolio(filepath)
        assert "books" in result
        # All books should be empty
        for book in BOOKS:
            assert result["books"].get(book, []) == []

    def test_load_corrupted_file_returns_empty(self, tmp_path):
        filepath = str(tmp_path / "corrupted.json")
        with open(filepath, "w") as f:
            f.write("{bad json::::")
        result = load_portfolio(filepath)
        assert "books" in result
        for book in BOOKS:
            assert result["books"].get(book, []) == []

    def test_save_to_existing_dir(self, tmp_path):
        """save_portfolio writes successfully to an existing directory."""
        filepath = str(tmp_path / "portfolio.json")
        add_position("G10_FLOW", _make_position())
        save_portfolio(filepath)
        assert os.path.exists(filepath)

    def test_save_to_nonexistent_subdir_logs_error(self, tmp_path):
        """save_portfolio logs error when custom path parent doesn't exist
        (_ensure_data_dir only creates the default data dir)."""
        filepath = str(tmp_path / "nonexistent_subdir" / "portfolio.json")
        # Should not raise, but the file won't be created
        save_portfolio(filepath)
        assert not os.path.exists(filepath)


# ============================================================================
# check_risk_limits
# ============================================================================

class TestCheckRiskLimits:

    def test_no_breach_on_zero_risk(self):
        risk = {"totals": {"delta": 0, "vega": 0, "theta": 0}, "by_pair": {}}
        breaches = check_risk_limits(risk)
        assert breaches == []

    def test_delta_breach(self):
        risk = {
            "totals": {"delta": 25_000_000, "vega": 0, "theta": 0},
            "by_pair": {},
        }
        limits = {"max_total_delta": 20_000_000}
        breaches = check_risk_limits(risk, limits)
        delta_breaches = [b for b in breaches if b["limit"] == "max_total_delta"]
        assert len(delta_breaches) == 1
        assert delta_breaches[0]["severity"] in ("BREACH", "CRITICAL")

    def test_warning_severity(self):
        risk = {
            "totals": {"delta": 17_000_000, "vega": 0, "theta": 0},
            "by_pair": {},
        }
        limits = {"max_total_delta": 20_000_000}
        breaches = check_risk_limits(risk, limits)
        delta_breaches = [b for b in breaches if b["limit"] == "max_total_delta"]
        assert len(delta_breaches) == 1
        assert delta_breaches[0]["severity"] == "WARNING"

    def test_critical_severity(self):
        risk = {
            "totals": {"delta": 35_000_000, "vega": 0, "theta": 0},
            "by_pair": {},
        }
        limits = {"max_total_delta": 20_000_000}
        breaches = check_risk_limits(risk, limits)
        delta_breaches = [b for b in breaches if b["limit"] == "max_total_delta"]
        assert len(delta_breaches) == 1
        assert delta_breaches[0]["severity"] == "CRITICAL"

    def test_per_pair_vega_breach(self):
        risk = {
            "totals": {"delta": 0, "vega": 0, "theta": 0},
            "by_pair": {"EURUSD": {"delta": 0, "gamma": 0, "vega": 250_000}},
        }
        limits = {"max_vega_per_pair": 200_000}
        breaches = check_risk_limits(risk, limits)
        vega_breaches = [b for b in breaches if b["limit"] == "max_vega_per_pair"]
        assert len(vega_breaches) == 1
        assert vega_breaches[0]["pair"] == "EURUSD"

    def test_theta_breach_only_when_negative(self):
        # Positive theta (net long theta from short options) should not breach
        risk = {
            "totals": {"delta": 0, "vega": 0, "theta": 60_000},
            "by_pair": {},
        }
        limits = {"max_daily_theta": -50_000}
        breaches = check_risk_limits(risk, limits)
        theta_breaches = [b for b in breaches if b["limit"] == "max_daily_theta"]
        assert len(theta_breaches) == 0

    def test_theta_breach_when_negative(self):
        risk = {
            "totals": {"delta": 0, "vega": 0, "theta": -75_000},
            "by_pair": {},
        }
        limits = {"max_daily_theta": -50_000}
        breaches = check_risk_limits(risk, limits)
        theta_breaches = [b for b in breaches if b["limit"] == "max_daily_theta"]
        assert len(theta_breaches) == 1

    def test_sorted_by_severity(self):
        risk = {
            "totals": {"delta": 25_000_000, "vega": 1_500_000, "theta": 0},
            "by_pair": {},
        }
        limits = {
            "max_total_delta": 20_000_000,
            "max_total_vega": 500_000,
        }
        breaches = check_risk_limits(risk, limits)
        # CRITICAL should come before BREACH
        if len(breaches) >= 2:
            sev_order = {"CRITICAL": 0, "BREACH": 1, "WARNING": 2, "OK": 3}
            for i in range(len(breaches) - 1):
                assert sev_order[breaches[i]["severity"]] <= sev_order[breaches[i + 1]["severity"]]


# ============================================================================
# hedge_suggestion
# ============================================================================

class TestHedgeSuggestion:

    def test_delta_neutral_suggestion(self):
        risk = {
            "by_pair": {
                "EURUSD": {"delta": 3_000_000, "vega": 0, "gamma": 0},
            }
        }
        suggestions = hedge_suggestion(risk, target="delta_neutral")
        assert len(suggestions) >= 1
        s = suggestions[0]
        assert s["pair"] == "EURUSD"
        assert s["instrument"] == "SPOT"
        assert s["direction"] == "sell"  # positive delta -> sell to flatten

    def test_delta_neutral_negative_delta(self):
        risk = {
            "by_pair": {
                "EURUSD": {"delta": -3_000_000, "vega": 0, "gamma": 0},
            }
        }
        suggestions = hedge_suggestion(risk, target="delta_neutral")
        assert len(suggestions) >= 1
        assert suggestions[0]["direction"] == "buy"

    def test_delta_neutral_small_delta_ignored(self):
        """Deltas below the threshold should produce no suggestion."""
        risk = {
            "by_pair": {
                "EURUSD": {"delta": 100, "vega": 0, "gamma": 0},
            }
        }
        suggestions = hedge_suggestion(risk, target="delta_neutral")
        assert len(suggestions) == 0

    def test_vega_neutral_suggestion(self):
        risk = {
            "by_pair": {
                "EURUSD": {"delta": 0, "vega": 100_000, "gamma": 0},
            }
        }
        suggestions = hedge_suggestion(risk, target="vega_neutral_3M")
        assert len(suggestions) >= 1
        s = suggestions[0]
        assert s["instrument"] == "3M ATM STRADDLE"
        assert s["direction"] == "sell"

    def test_gamma_neutral_short_gamma(self):
        risk = {
            "by_pair": {
                "EURUSD": {"delta": 0, "vega": 50_000, "gamma": -100_000},
            }
        }
        suggestions = hedge_suggestion(risk, target="gamma_neutral")
        assert len(suggestions) >= 1
        s = suggestions[0]
        assert s["instrument"] == "1M ATM STRADDLE"
        assert s["direction"] == "buy"

    def test_gamma_neutral_long_gamma(self):
        risk = {
            "by_pair": {
                "EURUSD": {"delta": 0, "vega": 50_000, "gamma": 100_000},
            }
        }
        suggestions = hedge_suggestion(risk, target="gamma_neutral")
        assert len(suggestions) >= 1
        assert suggestions[0]["direction"] == "sell"

    def test_empty_portfolio_no_suggestions(self):
        risk = {"by_pair": {}}
        suggestions = hedge_suggestion(risk, target="delta_neutral")
        assert suggestions == []


# ============================================================================
# exposure_summary (pure function, no Bloomberg needed)
# ============================================================================

class TestExposureSummary:

    def _build_positions(self):
        """Build a list of position dicts for testing exposure_summary."""
        return [
            {"pair": "EURUSD", "book": "G10_FLOW", "strategy": "PROP",
             "notional": 5_000_000, "status": "open", "option_type": "call",
             "direction": "buy", "strike": 1.10, "expiry": "2026-06-01"},
            {"pair": "USDMXN", "book": "EM_FLOW", "strategy": "RV",
             "notional": 3_000_000, "status": "open", "option_type": "put",
             "direction": "sell", "strike": 17.50, "expiry": "2026-06-01"},
            {"pair": "GBPUSD", "book": "G10_FLOW", "strategy": "PROP",
             "notional": 2_000_000, "status": "open", "option_type": "call",
             "direction": "buy", "strike": 1.30, "expiry": "2026-06-01"},
            {"pair": "EURUSD", "book": "G10_PROP", "strategy": "PROP",
             "notional": 1_000_000, "status": "closed", "option_type": "call",
             "direction": "buy", "strike": 1.10, "expiry": "2026-06-01"},
        ]

    def test_total_notional_excludes_closed(self):
        positions = self._build_positions()
        result = exposure_summary(positions, {}, 0.04, {})
        assert result["total_notional"] == 10_000_000  # 5M + 3M + 2M

    def test_by_pair_aggregation(self):
        positions = self._build_positions()
        result = exposure_summary(positions, {}, 0.04, {})
        assert result["by_pair"]["EURUSD"] == 5_000_000
        assert result["by_pair"]["USDMXN"] == 3_000_000
        assert result["by_pair"]["GBPUSD"] == 2_000_000

    def test_em_vs_g10_split(self):
        positions = self._build_positions()
        result = exposure_summary(positions, {}, 0.04, {})
        assert result["em_notional"] == 3_000_000
        assert result["g10_notional"] == 7_000_000
        assert result["em_pct"] == pytest.approx(0.30)

    def test_by_book(self):
        positions = self._build_positions()
        result = exposure_summary(positions, {}, 0.04, {})
        assert result["by_book"]["G10_FLOW"] == 7_000_000
        assert result["by_book"]["EM_FLOW"] == 3_000_000

    def test_by_strategy(self):
        positions = self._build_positions()
        result = exposure_summary(positions, {}, 0.04, {})
        assert result["by_strategy"]["PROP"] == 7_000_000
        assert result["by_strategy"]["RV"] == 3_000_000

    def test_empty_positions(self):
        result = exposure_summary([], {}, 0.04, {})
        assert result["total_notional"] == 0.0
        assert result["em_pct"] == 0.0


# ============================================================================
# compute_position_greeks (uses GK but no Bloomberg)
# ============================================================================

class TestComputePositionGreeks:

    def test_long_call_positive_delta(self):
        pos = {
            "id": "test1", "pair": "EURUSD", "option_type": "call",
            "direction": "buy", "strike": 1.10,
            "expiry": 0.25,  # numeric T
            "notional": 1_000_000, "status": "open",
        }
        greeks = compute_position_greeks(pos, 1.10, 0.04, 0.02, {"EURUSD": 0.10})
        assert greeks["delta"] > 0

    def test_short_call_negative_delta(self):
        pos = {
            "id": "test2", "pair": "EURUSD", "option_type": "call",
            "direction": "sell", "strike": 1.10,
            "expiry": 0.25,
            "notional": 1_000_000, "status": "open",
        }
        greeks = compute_position_greeks(pos, 1.10, 0.04, 0.02, {"EURUSD": 0.10})
        assert greeks["delta"] < 0

    def test_long_put_negative_delta(self):
        pos = {
            "id": "test3", "pair": "EURUSD", "option_type": "put",
            "direction": "buy", "strike": 1.10,
            "expiry": 0.25,
            "notional": 1_000_000, "status": "open",
        }
        greeks = compute_position_greeks(pos, 1.10, 0.04, 0.02, {"EURUSD": 0.10})
        assert greeks["delta"] < 0

    def test_greeks_output_keys(self):
        pos = {
            "id": "test4", "pair": "EURUSD", "option_type": "call",
            "direction": "buy", "strike": 1.10,
            "expiry": 0.25,
            "notional": 1_000_000, "status": "open",
        }
        greeks = compute_position_greeks(pos, 1.10, 0.04, 0.02, {"EURUSD": 0.10})
        for key in ("delta", "gamma", "vega", "theta", "rho_d", "rho_f",
                     "vanna", "volga", "price", "pair", "position_id", "sigma",
                     "T", "bucket"):
            assert key in greeks

    def test_bucket_assigned(self):
        pos = {
            "id": "test5", "pair": "EURUSD", "option_type": "call",
            "direction": "buy", "strike": 1.10,
            "expiry": 0.25,  # ~91 days -> "1-3M"
            "notional": 1_000_000, "status": "open",
        }
        greeks = compute_position_greeks(pos, 1.10, 0.04, 0.02, {"EURUSD": 0.10})
        assert greeks["bucket"] in TENOR_BUCKETS


# ============================================================================
# pnl_attribution (pure Taylor decomposition, no Bloomberg)
# ============================================================================

class TestPnLAttribution:

    def _sample_position(self):
        return {
            "id": "pnl-test",
            "pair": "EURUSD",
            "option_type": "call",
            "direction": "buy",
            "strike": 1.10,
            "expiry": 0.25,
            "notional": 1_000_000,
            "status": "open",
        }

    def test_zero_move_near_zero_pnl(self):
        """With no spot or vol change over 0 days, P&L should be ~0."""
        pos = self._sample_position()
        attr = pnl_attribution(
            [pos],
            spots_old={"EURUSD": 1.10},
            spots_new={"EURUSD": 1.10},
            surfaces_old={"EURUSD": 0.10},
            surfaces_new={"EURUSD": 0.10},
            rates=0.04,
            dt=0.0,
        )
        assert abs(attr["total_pnl"]) < 1.0  # essentially zero

    def test_spot_up_call_positive_pnl(self):
        pos = self._sample_position()
        attr = pnl_attribution(
            [pos],
            spots_old={"EURUSD": 1.10},
            spots_new={"EURUSD": 1.12},
            surfaces_old={"EURUSD": 0.10},
            surfaces_new={"EURUSD": 0.10},
            rates=0.04,
            dt=0.0,
        )
        assert attr["delta_pnl"] > 0
        assert attr["total_pnl"] > 0

    def test_vol_up_long_vega_positive(self):
        pos = self._sample_position()
        attr = pnl_attribution(
            [pos],
            spots_old={"EURUSD": 1.10},
            spots_new={"EURUSD": 1.10},
            surfaces_old={"EURUSD": 0.10},
            surfaces_new={"EURUSD": 0.12},
            rates=0.04,
            dt=0.0,
        )
        assert attr["vega_pnl"] > 0

    def test_closed_positions_excluded(self):
        pos = self._sample_position()
        pos["status"] = "closed"
        attr = pnl_attribution(
            [pos],
            spots_old={"EURUSD": 1.10},
            spots_new={"EURUSD": 1.15},
            surfaces_old={"EURUSD": 0.10},
            surfaces_new={"EURUSD": 0.10},
            rates=0.04,
            dt=0.0,
        )
        assert attr["total_pnl"] == 0.0

    def test_by_position_breakdown(self):
        pos = self._sample_position()
        attr = pnl_attribution(
            [pos],
            spots_old={"EURUSD": 1.10},
            spots_new={"EURUSD": 1.12},
            surfaces_old={"EURUSD": 0.10},
            surfaces_new={"EURUSD": 0.10},
            rates=0.04,
            dt=0.0,
        )
        assert "by_position" in attr
        assert len(attr["by_position"]) == 1
        assert attr["by_position"][0]["position_id"] == "pnl-test"


# ============================================================================
# what_if_add (pure function, no Bloomberg)
# ============================================================================

class TestWhatIfAdd:

    def test_what_if_structure(self):
        existing = [{
            "id": "existing-1", "pair": "EURUSD", "option_type": "call",
            "direction": "buy", "strike": 1.10, "expiry": 0.25,
            "notional": 1_000_000, "status": "open",
        }]
        new_trade = {
            "pair": "EURUSD", "option_type": "put", "direction": "sell",
            "strike": 1.08, "expiry": 0.25, "notional": 500_000,
        }
        result = what_if_add(existing, new_trade, {"EURUSD": 1.10}, 0.04, {"EURUSD": 0.10})
        assert "before" in result
        assert "after" in result
        assert "change" in result
        assert "new_trade_greeks" in result

    def test_what_if_delta_change(self):
        existing = []
        new_trade = {
            "pair": "EURUSD", "option_type": "call", "direction": "buy",
            "strike": 1.10, "expiry": 0.25, "notional": 1_000_000,
        }
        result = what_if_add(existing, new_trade, {"EURUSD": 1.10}, 0.04, {"EURUSD": 0.10})
        # Adding a long call should increase delta
        assert result["change"]["delta"] > 0
        assert result["before"]["delta"] == 0.0


# ============================================================================
# Constants
# ============================================================================

class TestConstants:

    def test_books_nonempty(self):
        assert len(BOOKS) > 0

    def test_tenor_buckets_nonempty(self):
        assert len(TENOR_BUCKETS) > 0
        assert "0-1M" in TENOR_BUCKETS
        assert "2-5Y" in TENOR_BUCKETS

    def test_default_risk_limits_keys(self):
        expected_keys = {
            "max_delta_per_pair", "max_total_delta",
            "max_gamma_per_pair", "max_vega_per_pair",
            "max_vega_per_tenor_bucket", "max_total_vega",
            "max_daily_theta", "max_var_95_1d",
            "max_notional_per_pair", "max_em_notional_pct",
        }
        assert set(DEFAULT_RISK_LIMITS.keys()) == expected_keys
