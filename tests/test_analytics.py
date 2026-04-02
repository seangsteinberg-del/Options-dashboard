"""
Unit tests for core/fx_analytics.py — pure computation functions.

Tests cover functions that don't require Bloomberg data:
  - _to_close_array conversion
  - historical_var, parametric_var, expected_shortfall
  - cornish_fisher_var
  - max_drawdown
  - omega_ratio, ulcer_index, max_consecutive, recovery_factor
  - tail_risk_metrics
"""

import numpy as np
import pandas as pd
import pytest
from core.fx_analytics import (
    _to_close_array,
    historical_var,
    parametric_var,
    expected_shortfall,
    cornish_fisher_var,
    max_drawdown,
    omega_ratio,
    ulcer_index,
    max_consecutive,
    recovery_factor,
    tail_risk_metrics,
)


# ── _to_close_array ──────────────────────────────────────────────────────

class TestToCloseArray:

    def test_none_input(self):
        assert _to_close_array(None) is None

    def test_empty_dataframe(self):
        assert _to_close_array(pd.DataFrame()) is None

    def test_series_input(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = _to_close_array(s)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_dataframe_with_close_column(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "open": [0.9, 1.9, 2.9]})
        result = _to_close_array(df)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_dataframe_single_column(self):
        df = pd.DataFrame({"price": [10.0, 20.0]})
        result = _to_close_array(df)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_close_array(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)


# ── Historical VaR ───────────────────────────────────────────────────────

class TestHistoricalVaR:

    def test_basic_var(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        result = historical_var(returns, confidence=0.95)
        assert result["var_1d"] > 0
        assert result["var_horizon"] > 0

    def test_higher_confidence_higher_var(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        var95 = historical_var(returns, confidence=0.95)
        var99 = historical_var(returns, confidence=0.99)
        assert var99["var_1d"] > var95["var_1d"]

    def test_zero_returns(self):
        returns = np.zeros(100)
        result = historical_var(returns, confidence=0.95)
        assert result["var_1d"] == 0.0

    def test_horizon_scaling(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        r1 = historical_var(returns, confidence=0.95, horizon=1)
        r10 = historical_var(returns, confidence=0.95, horizon=10)
        ratio = r10["var_horizon"] / r1["var_horizon"]
        assert abs(ratio - np.sqrt(10)) < 0.01

    def test_small_sample(self):
        returns = np.array([0.01, -0.02, 0.005])
        result = historical_var(returns, confidence=0.95)
        # Less than 10 observations → returns fallback
        assert result["n_observations"] == 0

    def test_result_keys(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)
        result = historical_var(returns)
        for key in ("var_1d", "var_horizon", "confidence", "horizon",
                     "n_observations", "worst_return", "best_return"):
            assert key in result


# ── Parametric VaR ───────────────────────────────────────────────────────

class TestParametricVaR:

    def test_basic(self):
        result = parametric_var(0.10, 1_000_000, confidence=0.95)
        assert result["var"] > 0

    def test_higher_vol_higher_var(self):
        r1 = parametric_var(0.10, 1_000_000)
        r2 = parametric_var(0.20, 1_000_000)
        assert r2["var"] > r1["var"]

    def test_zero_vol(self):
        result = parametric_var(0.0, 1_000_000)
        assert result["var"] == 0.0

    def test_notional_scaling(self):
        r1 = parametric_var(0.10, 1_000_000)
        r2 = parametric_var(0.10, 2_000_000)
        assert abs(r2["var"] - 2 * r1["var"]) < 0.01

    def test_horizon_scaling(self):
        r1 = parametric_var(0.10, 1_000_000, horizon=1)
        r10 = parametric_var(0.10, 1_000_000, horizon=10)
        ratio = r10["var"] / r1["var"]
        assert abs(ratio - np.sqrt(10)) < 0.01

    def test_percentage_auto_convert(self):
        """Sigma > 1.0 is auto-converted from percentage to decimal."""
        r_pct = parametric_var(10.0, 1_000_000)  # 10% passed as percentage
        r_dec = parametric_var(0.10, 1_000_000)   # 10% passed as decimal
        assert abs(r_pct["var"] - r_dec["var"]) < 0.01


# ── Expected Shortfall ───────────────────────────────────────────────────

class TestExpectedShortfall:

    def test_cvar_greater_than_var(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        es = expected_shortfall(returns, confidence=0.95)
        assert es["cvar_1d"] >= es["var_1d"]

    def test_basic_positive(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        es = expected_shortfall(returns, confidence=0.95)
        assert es["cvar_1d"] > 0

    def test_tail_ratio(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        es = expected_shortfall(returns, confidence=0.95)
        assert es["tail_ratio"] >= 1.0  # CVaR ≥ VaR always

    def test_result_keys(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)
        es = expected_shortfall(returns)
        for key in ("cvar_1d", "cvar_horizon", "var_1d", "confidence",
                     "horizon", "n_tail_obs", "tail_ratio"):
            assert key in es

    def test_small_sample(self):
        returns = np.array([0.01, -0.02])
        es = expected_shortfall(returns)
        assert es["n_tail_obs"] == 0


# ── Cornish-Fisher VaR ───────────────────────────────────────────────────

class TestCornishFisherVaR:

    def test_basic(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        result = cornish_fisher_var(returns)
        assert "cf_var" in result
        assert "skewness" in result
        assert "excess_kurtosis" in result
        assert result["cf_var"] > 0

    def test_normal_returns_close_to_gaussian(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 10000)
        cf = cornish_fisher_var(returns, confidence=0.95)
        # For normal returns, CF VaR ≈ Gaussian VaR
        assert abs(cf["cf_var"] - cf["gaussian_var"]) < 0.002

    def test_small_sample_fallback(self):
        returns = np.array([0.01, -0.02, 0.005])
        result = cornish_fisher_var(returns)
        assert result["cf_var"] == 0.0


# ── Max Drawdown ─────────────────────────────────────────────────────────

class TestMaxDrawdown:

    def test_basic(self):
        returns = np.array([0.01, 0.02, -0.05, 0.01, -0.03, 0.02])
        result = max_drawdown(returns)
        assert result["max_drawdown"] < 0  # Drawdowns are negative
        assert result["max_drawdown_pct"] < 0

    def test_all_positive_zero_dd(self):
        returns = np.array([0.01, 0.02, 0.03, 0.01])
        result = max_drawdown(returns)
        assert result["max_drawdown"] == 0.0

    def test_all_negative(self):
        returns = np.array([-0.01, -0.02, -0.01, -0.03])
        result = max_drawdown(returns)
        assert result["max_drawdown"] < 0

    def test_peak_trough_indices(self):
        returns = np.array([0.05, 0.05, -0.10, -0.10, 0.05, 0.05])
        result = max_drawdown(returns)
        assert "peak_idx" in result
        assert "trough_idx" in result
        assert result["trough_idx"] >= result["peak_idx"]

    def test_single_element_fallback(self):
        returns = np.array([0.01])
        result = max_drawdown(returns)
        assert result["max_drawdown"] == 0.0

    def test_recovery_tracked(self):
        returns = np.array([0.05, 0.05, -0.10, 0.10, 0.10])
        result = max_drawdown(returns)
        # Should find a recovery point
        assert result["peak_value"] > 0
        assert result["trough_value"] > 0


# ── Omega Ratio ──────────────────────────────────────────────────────────

class TestOmegaRatio:

    def test_positive_pnl(self):
        pnl = np.array([100, 200, -50, 150, -30, 80])
        omega = omega_ratio(pnl, mar=0.0)
        assert omega > 1.0  # Net positive → omega > 1

    def test_negative_pnl(self):
        pnl = np.array([-100, -200, 50, -150, 30, -80])
        omega = omega_ratio(pnl, mar=0.0)
        assert omega < 1.0  # Net negative → omega < 1

    def test_all_positive_capped(self):
        pnl = np.array([100, 200, 300])
        omega = omega_ratio(pnl, mar=0.0)
        assert omega == 10.0  # Capped at 10 when no losses

    def test_zero_pnl(self):
        pnl = np.zeros(10)
        omega = omega_ratio(pnl, mar=0.0)
        assert np.isfinite(omega)

    def test_single_element(self):
        omega = omega_ratio(np.array([100]), mar=0.0)
        assert omega == 0.0  # Less than 2 elements → 0


# ── Ulcer Index ──────────────────────────────────────────────────────────

class TestUlcerIndex:

    def test_monotonic_increasing_low(self):
        pnl = np.array([100, 200, 300, 400])
        ui = ulcer_index(pnl)
        assert ui >= 0

    def test_volatile_pnl(self):
        pnl = np.array([100, -200, 300, -400, 500])
        ui = ulcer_index(pnl)
        assert ui > 0

    def test_flat_pnl(self):
        pnl = np.array([100, 100, 100, 100])
        ui = ulcer_index(pnl)
        assert ui == 0.0

    def test_single_element(self):
        ui = ulcer_index(np.array([100]))
        assert ui == 0.0


# ── Max Consecutive ──────────────────────────────────────────────────────

class TestMaxConsecutive:

    def test_consecutive_losses(self):
        pnl = np.array([100, -50, -30, -20, 100, -10])
        assert max_consecutive(pnl, direction="loss") == 3

    def test_consecutive_wins(self):
        pnl = np.array([100, 200, 300, -50, 100])
        assert max_consecutive(pnl, direction="win") == 3

    def test_no_losses(self):
        pnl = np.array([100, 200, 300])
        assert max_consecutive(pnl, direction="loss") == 0

    def test_all_losses(self):
        pnl = np.array([-100, -200, -300, -50])
        assert max_consecutive(pnl, direction="loss") == 4

    def test_single_element(self):
        assert max_consecutive(np.array([-100]), direction="loss") == 1
        assert max_consecutive(np.array([100]), direction="win") == 1


# ── Recovery Factor ──────────────────────────────────────────────────────

class TestRecoveryFactor:

    def test_basic(self):
        rf = recovery_factor(1000, 200)
        assert rf == 5.0

    def test_zero_drawdown_returns_zero(self):
        rf = recovery_factor(1000, 0)
        assert rf == 0.0  # Guards against division by zero

    def test_negative_pnl(self):
        rf = recovery_factor(-500, 1000)
        assert rf < 0


# ── Tail Risk Metrics ────────────────────────────────────────────────────

class TestTailRiskMetrics:

    def test_basic_structure(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        result = tail_risk_metrics(returns)
        assert isinstance(result, dict)
        for key in ("skewness", "excess_kurtosis", "jarque_bera_stat",
                     "jb_pvalue", "normality"):
            assert key in result

    def test_normal_returns(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 10000)
        result = tail_risk_metrics(returns)
        # Large normal sample → skew near 0, excess kurtosis near 0
        assert abs(result["skewness"]) < 0.2
        assert abs(result["excess_kurtosis"]) < 0.5

    def test_skewed_returns(self):
        np.random.seed(42)
        returns = np.random.exponential(0.01, 500) - 0.01
        result = tail_risk_metrics(returns)
        assert result["skewness"] > 0

    def test_normality_not_rejected_for_normal(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 5000)
        result = tail_risk_metrics(returns)
        assert result["normality"] == "NOT_REJECTED"

    def test_hill_tail_index(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        result = tail_risk_metrics(returns)
        assert "hill_tail_index" in result
        assert result["hill_tail_index"] > 0

    def test_small_sample_fallback(self):
        returns = np.array([0.01, -0.02, 0.005])
        result = tail_risk_metrics(returns)
        assert result["n_observations"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
