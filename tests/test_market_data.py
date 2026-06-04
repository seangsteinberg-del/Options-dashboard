"""
Tests for core.market_data — the strict, no-fabrication accessors.

The invariant: a missing real value yields None (get_*) or raises
MarketDataUnavailable (require_*).  It must NEVER return a fabricated default.
"""
import pytest
from unittest.mock import patch

from core.market_data import (
    NA, MarketDataUnavailable, normalize_vol,
    get_spot, require_spot, get_rates, require_rates,
    get_atm_vol, require_atm_vol, atm_vol_from_surface,
)


class TestNormalizeVol:
    def test_points_to_decimal(self):
        assert normalize_vol(8.5) == pytest.approx(0.085)

    def test_decimal_passthrough(self):
        assert normalize_vol(0.085) == pytest.approx(0.085)

    def test_none(self):
        assert normalize_vol(None) is None

    def test_zero_is_missing_not_zero_vol(self):
        assert normalize_vol(0) is None
        assert normalize_vol(0.0) is None

    def test_negative(self):
        assert normalize_vol(-1) is None

    def test_non_numeric(self):
        assert normalize_vol("x") is None


class TestGetSpot:
    def test_real(self):
        with patch("core.bloomberg_fx.get_fx_spots", return_value={"EURUSD": {"mid": 1.085}}):
            assert get_spot("EURUSD") == pytest.approx(1.085)

    def test_missing_returns_none(self):
        with patch("core.bloomberg_fx.get_fx_spots", return_value={}):
            assert get_spot("EURUSD") is None

    def test_zero_mid_is_none(self):
        with patch("core.bloomberg_fx.get_fx_spots", return_value={"EURUSD": {"mid": 0}}):
            assert get_spot("EURUSD") is None

    def test_exception_returns_none(self):
        with patch("core.bloomberg_fx.get_fx_spots", side_effect=Exception("down")):
            assert get_spot("EURUSD") is None

    def test_require_raises(self):
        with patch("core.bloomberg_fx.get_fx_spots", return_value={}):
            with pytest.raises(MarketDataUnavailable):
                require_spot("EURUSD")


class TestGetRates:
    def test_real(self):
        with patch("core.bloomberg_fx.get_fx_rates", return_value={"r_dom": 0.04, "r_for": 0.02}):
            assert get_rates("EURUSD") == (0.04, 0.02)

    def test_one_sided_is_none(self):
        with patch("core.bloomberg_fx.get_fx_rates", return_value={"r_dom": 0.04}):
            assert get_rates("EURUSD") is None

    def test_empty_is_none(self):
        with patch("core.bloomberg_fx.get_fx_rates", return_value={}):
            assert get_rates("EURUSD") is None

    def test_require_raises(self):
        with patch("core.bloomberg_fx.get_fx_rates", return_value={}):
            with pytest.raises(MarketDataUnavailable):
                require_rates("EURUSD")


class TestGetAtmVol:
    def test_real(self):
        surf = {"3M": {"atm": 9.0}}
        with patch("core.bloomberg_fx.get_fx_vol_surface", return_value=surf):
            assert get_atm_vol("EURUSD", "3M") == pytest.approx(0.09)

    def test_missing_tenor_none(self):
        surf = {"1M": {"atm": 9.0}}
        with patch("core.bloomberg_fx.get_fx_vol_surface", return_value=surf):
            assert get_atm_vol("EURUSD", "3M") is None

    def test_empty_surface_none(self):
        with patch("core.bloomberg_fx.get_fx_vol_surface", return_value={}):
            assert get_atm_vol("EURUSD", "3M") is None

    def test_require_raises(self):
        with patch("core.bloomberg_fx.get_fx_vol_surface", return_value={}):
            with pytest.raises(MarketDataUnavailable):
                require_atm_vol("EURUSD", "3M")


class TestAtmVolFromSurface:
    def test_dict_preferred_tenor(self):
        surf = {"1M": {"atm": 7.0}, "3M": {"atm": 9.0}}
        assert atm_vol_from_surface(surf, "3M") == pytest.approx(0.09)

    def test_dict_nearest_when_no_tenor(self):
        surf = {"3M": {"atm": 9.0}}
        assert atm_vol_from_surface(surf) == pytest.approx(0.09)

    def test_scalar(self):
        assert atm_vol_from_surface(8.0) == pytest.approx(0.08)

    def test_none_surface(self):
        assert atm_vol_from_surface(None) is None

    def test_empty(self):
        assert atm_vol_from_surface({}) is None

    def test_missing_atm_in_node(self):
        assert atm_vol_from_surface({"3M": {"rr25": -0.4}}) is None


def test_na_token():
    assert NA == "—"
