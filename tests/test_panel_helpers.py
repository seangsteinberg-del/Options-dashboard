"""
Regression tests for shared panel-helper bugs:

1. `_empty_fig(message)` was aliased to `no_data_fig` whose first positional
   argument is `height`. Calling `_empty_fig("RR/BF SPREAD: ...")` therefore
   crashed inside Plotly with `Invalid value of type 'str' received for the
   'height' property`. Every panel-local _empty_fig must now accept a
   string-first call signature.

2. `cross_pair_rr_spread` (and sibling cross-pair spreads) called
   `spread[-1]` on a pandas Series whose underlying index is a DatetimeIndex,
   triggering `KeyError: -1`. The fix uses .iloc / numpy coercion.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch


# ─────────────────────────── _empty_fig signature ───────────────────────────

@pytest.mark.parametrize("module_name", [
    "panels.relative_value_plus",
    "panels.backtest",
    "panels.market_dashboard",
    "panels.vol_scanner_unified",
    "panels.exotics_pricer",
])
def test_empty_fig_accepts_string_message(module_name):
    """The string-first call site is the one that broke production."""
    mod = __import__(module_name, fromlist=["_empty_fig"])
    fig = mod._empty_fig("Some message that should not crash")
    assert fig is not None
    # Plotly figure carries an annotation with the message
    assert any("Some message" in (a.text or "") for a in fig.layout.annotations)


def test_market_dashboard_empty_fig_accepts_three_args():
    """market_dashboard uniquely calls _empty_fig(title, height, msg)."""
    from panels.market_dashboard import _empty_fig
    fig = _empty_fig("VOL RICHNESS", 220, "No percentile data")
    assert fig is not None
    assert fig.layout.height == 220
    assert any("No percentile data" in (a.text or "")
               for a in fig.layout.annotations)


# ─────────────────────────── cross-pair DatetimeIndex ───────────────────────

def _series_with_datetime_index(values):
    idx = pd.date_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx)


def test_cross_pair_rr_spread_handles_datetime_indexed_series():
    """Reproduces the production KeyError: -1 from `spread[-1]` on a
    Series-with-DatetimeIndex returned by get_fx_historical_vol."""
    from core import fx_analytics
    a = _series_with_datetime_index(np.random.randn(120) * 0.5 + 8.0)
    b = _series_with_datetime_index(np.random.randn(120) * 0.5 + 7.5)
    with patch.object(fx_analytics, "get_fx_historical_vol",
                       side_effect=[a, b]):
        result = fx_analytics.cross_pair_rr_spread("EURUSD", "USDJPY",
                                                    "3M", 120)
    assert result is not None
    assert np.isfinite(result["current"])
    assert np.isfinite(result["zscore"])


def test_cross_pair_bf_spread_handles_datetime_indexed_series():
    from core import fx_analytics
    a = _series_with_datetime_index(np.random.randn(120) * 0.3 + 0.5)
    b = _series_with_datetime_index(np.random.randn(120) * 0.3 + 0.6)
    with patch.object(fx_analytics, "get_fx_historical_vol",
                       side_effect=[a, b]):
        result = fx_analytics.cross_pair_bf_spread("EURUSD", "USDJPY",
                                                    "3M", 120)
    assert result is not None
    assert np.isfinite(result["current"])


def test_cross_pair_term_spread_handles_datetime_indexed_series():
    """Same DatetimeIndex bug pattern in cross_pair_term_spread."""
    from core import fx_analytics
    series = [_series_with_datetime_index(np.random.randn(120) * 0.5 + 8.0)
              for _ in range(4)]
    with patch.object(fx_analytics, "get_fx_historical_vol",
                       side_effect=series):
        result = fx_analytics.cross_pair_term_spread("EURUSD", "USDJPY",
                                                      "1Y", "1M", 120)
    assert result is not None
    assert np.isfinite(result["current"])
    assert np.isfinite(result["zscore"])
