"""
Unit tests for core/data_helpers.py — safe Bloomberg fetch wrappers.

Tests verify the wrapper functions handle errors gracefully
without requiring an actual Bloomberg connection.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.data_helpers import (
    fetch_spots_safe,
    fetch_vol_surface_safe,
    fetch_rates_safe,
    fetch_spot_and_rates,
    fetch_historical_spot_safe,
    fetch_historical_vol_safe,
)


class TestFetchSpotsSafe:

    @patch("core.data_helpers.get_fx_spots")
    def test_returns_dict_on_success(self, mock_spots):
        mock_spots.return_value = {"EURUSD": {"mid": 1.10}}
        result = fetch_spots_safe(["EURUSD"])
        assert result == {"EURUSD": {"mid": 1.10}}

    @patch("core.data_helpers.get_fx_spots")
    def test_returns_empty_on_exception(self, mock_spots):
        mock_spots.side_effect = Exception("Bloomberg down")
        result = fetch_spots_safe(["EURUSD"])
        assert result == {}

    @patch("core.data_helpers.get_fx_spots")
    def test_returns_empty_on_none(self, mock_spots):
        mock_spots.return_value = None
        result = fetch_spots_safe(["EURUSD"])
        assert result == {}


class TestFetchVolSurfaceSafe:

    @patch("core.data_helpers.get_fx_vol_surface")
    def test_returns_dict_on_success(self, mock_vol):
        mock_vol.return_value = {"1M": {"atm": 8.5}}
        result = fetch_vol_surface_safe("EURUSD")
        assert result == {"1M": {"atm": 8.5}}

    @patch("core.data_helpers.get_fx_vol_surface")
    def test_returns_none_on_exception(self, mock_vol):
        mock_vol.side_effect = Exception("Bloomberg down")
        result = fetch_vol_surface_safe("EURUSD")
        assert result is None

    @patch("core.data_helpers.get_fx_vol_surface")
    def test_returns_none_on_empty(self, mock_vol):
        mock_vol.return_value = {}
        result = fetch_vol_surface_safe("EURUSD")
        assert result is None


class TestFetchRatesSafe:

    @patch("core.data_helpers.get_fx_rates")
    def test_returns_dict_on_success(self, mock_rates):
        mock_rates.return_value = {"r_dom": 0.04, "r_for": 0.02}
        result = fetch_rates_safe("EURUSD")
        assert result == {"r_dom": 0.04, "r_for": 0.02}

    @patch("core.data_helpers.get_fx_rates")
    def test_returns_none_on_exception(self, mock_rates):
        mock_rates.side_effect = Exception("Bloomberg down")
        result = fetch_rates_safe("EURUSD")
        assert result is None


class TestFetchSpotAndRates:

    @patch("core.data_helpers.get_fx_rates")
    @patch("core.data_helpers.get_fx_spots")
    def test_returns_tuple(self, mock_spots, mock_rates):
        mock_spots.return_value = {"EURUSD": {"mid": 1.10}}
        mock_rates.return_value = {"r_dom": 0.04, "r_for": 0.02}
        spot, r_dom, r_for = fetch_spot_and_rates("EURUSD")
        assert spot == 1.10
        assert r_dom == 0.04
        assert r_for == 0.02

    @patch("core.data_helpers.get_fx_rates")
    @patch("core.data_helpers.get_fx_spots")
    def test_defaults_on_failure(self, mock_spots, mock_rates):
        mock_spots.side_effect = Exception("down")
        mock_rates.side_effect = Exception("down")
        spot, r_dom, r_for = fetch_spot_and_rates("EURUSD")
        assert spot is None
        assert r_dom == 0.0
        assert r_for == 0.0


class TestFetchHistoricalSafe:

    @patch("core.data_helpers.get_fx_historical_spot")
    def test_spot_returns_none_on_exception(self, mock_hist):
        mock_hist.side_effect = Exception("down")
        result = fetch_historical_spot_safe("EURUSD")
        assert result is None

    @patch("core.data_helpers.get_fx_historical_spot")
    def test_spot_returns_none_on_empty(self, mock_hist):
        import pandas as pd
        mock_hist.return_value = pd.DataFrame()
        result = fetch_historical_spot_safe("EURUSD")
        assert result is None

    @patch("core.data_helpers.get_fx_historical_vol")
    def test_vol_returns_none_on_exception(self, mock_hist):
        mock_hist.side_effect = Exception("down")
        result = fetch_historical_vol_safe("EURUSD", "3M", "ATM")
        assert result is None

    @patch("core.data_helpers.get_fx_historical_vol")
    def test_vol_returns_none_on_empty(self, mock_hist):
        mock_hist.return_value = []
        result = fetch_historical_vol_safe("EURUSD", "3M", "ATM")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
