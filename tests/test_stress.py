"""
Unit tests for core/fx_stress.py — stress scenarios, shock helpers, position stressing.

Tests cover:
  - Scenario registry (get_scenarios, get_scenario, get_pair_shock)
  - Shock application helpers (apply_spot_shock, apply_vol_shock, apply_rate_shock)
  - Single position stress testing
  - Stress test edge cases
"""

import numpy as np
import pytest
from core.fx_stress import (
    get_scenarios, get_scenario, get_pair_shock,
    apply_spot_shock, apply_vol_shock, apply_rate_shock,
    stress_single_position, FX_STRESS_SCENARIOS,
    _default,
)


# ── Scenario Registry ────────────────────────────────────────────────────

class TestScenarioRegistry:

    def test_get_scenarios_returns_list(self):
        scenarios = get_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 10  # At least 10 named scenarios

    def test_scenario_has_required_fields(self):
        for sc in get_scenarios():
            assert "name" in sc
            assert "display_name" in sc
            assert "severity" in sc

    def test_get_scenario_known(self):
        sc = get_scenario("SNB_CHF_UNPEGGING")
        assert sc is not None
        assert "name" in sc
        assert "shocks" in sc

    def test_get_scenario_unknown(self):
        sc = get_scenario("NONEXISTENT_SCENARIO")
        assert sc is None

    def test_all_scenarios_have_shocks(self):
        for key, sc in FX_STRESS_SCENARIOS.items():
            assert "shocks" in sc, f"{key} missing shocks"
            assert isinstance(sc["shocks"], dict)

    def test_all_scenarios_have_severity(self):
        valid = {"EXTREME", "SEVERE", "MODERATE", "HIGH", "MEDIUM", "LOW"}
        for key, sc in FX_STRESS_SCENARIOS.items():
            assert "severity" in sc, f"{key} missing severity"
            assert sc["severity"] in valid, f"{key} has unexpected severity: {sc['severity']}"


# ── Get Pair Shock ────────────────────────────────────────────────────────

class TestGetPairShock:

    def test_known_pair_in_scenario(self):
        shock = get_pair_shock("SNB_CHF_UNPEGGING", "USDCHF")
        assert isinstance(shock, dict)
        assert "spot" in shock
        assert "vol_mult" in shock
        assert "rate" in shock

    def test_unknown_pair_gets_default(self):
        shock = get_pair_shock("SNB_CHF_UNPEGGING", "AUDJPY")
        assert isinstance(shock, dict)
        assert "spot" in shock

    def test_unknown_scenario(self):
        shock = get_pair_shock("NONEXISTENT", "EURUSD")
        assert shock["spot"] == 0.0
        assert shock["vol_mult"] == 1.0
        assert shock["rate"] == 0

    def test_default_helper(self):
        d = _default(0.05, 1.5, 50)
        assert d == {"spot": 0.05, "vol_mult": 1.5, "rate": 50}


# ── Shock Application Helpers ─────────────────────────────────────────────

class TestShockHelpers:

    def test_spot_shock_positive(self):
        result = apply_spot_shock(1.10, 0.05)
        assert abs(result - 1.155) < 1e-10

    def test_spot_shock_negative(self):
        result = apply_spot_shock(1.10, -0.10)
        assert abs(result - 0.99) < 1e-10

    def test_spot_shock_zero(self):
        result = apply_spot_shock(1.10, 0.0)
        assert result == 1.10

    def test_vol_shock_increase(self):
        result = apply_vol_shock(0.10, 1.5)
        assert abs(result - 0.15) < 1e-10

    def test_vol_shock_decrease(self):
        result = apply_vol_shock(0.10, 0.5)
        assert abs(result - 0.05) < 1e-10

    def test_vol_shock_floor(self):
        result = apply_vol_shock(0.10, 0.0)
        assert result == 0.001  # Floor at 0.1%

    def test_rate_shock_positive(self):
        result = apply_rate_shock(0.05, 100)
        assert abs(result - 0.06) < 1e-10  # +100bp = +1%

    def test_rate_shock_negative(self):
        result = apply_rate_shock(0.05, -50)
        assert abs(result - 0.045) < 1e-10  # -50bp = -0.5%

    def test_rate_shock_zero(self):
        result = apply_rate_shock(0.05, 0)
        assert result == 0.05


# ── Single Position Stress Test ──────────────────────────────────────────

class TestStressSinglePosition:

    @pytest.fixture
    def sample_position(self):
        return {
            "pair": "EURUSD",
            "strike": 1.10,
            "expiry": 0.25,  # 3 months
            "option_type": "call",
            "notional": 1_000_000,
            "direction": "buy",
        }

    def test_basic_stress(self, sample_position):
        result = stress_single_position(
            sample_position, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="SNB_CHF_UNPEGGING",
        )
        assert "base_value" in result
        assert "stressed_value" in result
        assert "pnl_impact" in result
        assert "pnl_pct" in result
        assert np.isfinite(result["pnl_impact"])

    def test_stress_returns_greeks(self, sample_position):
        result = stress_single_position(
            sample_position, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="SNB_CHF_UNPEGGING",
        )
        assert "stressed_greeks" in result
        greeks = result["stressed_greeks"]
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks

    def test_stress_put(self, sample_position):
        sample_position["option_type"] = "put"
        result = stress_single_position(
            sample_position, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="SNB_CHF_UNPEGGING",
        )
        assert np.isfinite(result["pnl_impact"])

    def test_stress_sell_direction(self, sample_position):
        result_buy = stress_single_position(
            sample_position, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="SNB_CHF_UNPEGGING",
        )
        sample_position["direction"] = "sell"
        result_sell = stress_single_position(
            sample_position, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="SNB_CHF_UNPEGGING",
        )
        # Buy and sell should have opposite P&L
        assert result_buy["pnl_impact"] * result_sell["pnl_impact"] <= 0

    def test_stress_string_expiry(self):
        pos = {
            "pair": "EURUSD",
            "strike": 1.10,
            "expiry": "2027-06-15",
            "option_type": "call",
            "notional": 1_000_000,
            "direction": "buy",
        }
        result = stress_single_position(
            pos, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="SNB_CHF_UNPEGGING",
        )
        assert np.isfinite(result["pnl_impact"])

    def test_zero_notional(self, sample_position):
        sample_position["notional"] = 0
        result = stress_single_position(
            sample_position, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="SNB_CHF_UNPEGGING",
        )
        assert result["pnl_impact"] == 0.0

    def test_unknown_scenario_no_shock(self, sample_position):
        """Unknown scenario → zero shocks → P&L should be ~0."""
        result = stress_single_position(
            sample_position, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="NONEXISTENT",
        )
        assert abs(result["pnl_impact"]) < 0.01

    def test_spot_values_in_result(self, sample_position):
        result = stress_single_position(
            sample_position, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="SNB_CHF_UNPEGGING",
        )
        assert result["spot_base"] == 1.10
        assert result["spot_stressed"] != 1.10  # CHF scenario has spot shock

    def test_vol_values_in_result(self, sample_position):
        result = stress_single_position(
            sample_position, spot=1.10, r_d=0.04, r_f=0.02,
            vol=0.08, scenario_name="SNB_CHF_UNPEGGING",
        )
        assert result["vol_base"] == 0.08
        assert result["vol_stressed"] > 0  # Should be positive after shock


# ── Edge Cases ────────────────────────────────────────────────────────────

class TestStressEdgeCases:

    def test_very_deep_itm(self):
        pos = {
            "pair": "EURUSD", "strike": 0.80, "expiry": 0.25,
            "option_type": "call", "notional": 1_000_000, "direction": "buy",
        }
        result = stress_single_position(pos, 1.10, 0.04, 0.02, 0.08, "SNB_CHF_UNPEGGING")
        assert np.isfinite(result["pnl_impact"])

    def test_very_deep_otm(self):
        pos = {
            "pair": "EURUSD", "strike": 2.00, "expiry": 0.25,
            "option_type": "call", "notional": 1_000_000, "direction": "buy",
        }
        result = stress_single_position(pos, 1.10, 0.04, 0.02, 0.08, "SNB_CHF_UNPEGGING")
        assert np.isfinite(result["pnl_impact"])

    def test_near_expiry(self):
        pos = {
            "pair": "EURUSD", "strike": 1.10, "expiry": 0.001,
            "option_type": "call", "notional": 1_000_000, "direction": "buy",
        }
        result = stress_single_position(pos, 1.10, 0.04, 0.02, 0.08, "SNB_CHF_UNPEGGING")
        assert np.isfinite(result["pnl_impact"])

    def test_high_vol(self):
        pos = {
            "pair": "USDTRY", "strike": 30.0, "expiry": 0.25,
            "option_type": "call", "notional": 1_000_000, "direction": "buy",
        }
        result = stress_single_position(pos, 30.0, 0.40, 0.05, 0.30, "SNB_CHF_UNPEGGING")
        assert np.isfinite(result["pnl_impact"])

    def test_all_named_scenarios(self):
        """Every named scenario should work without crashing."""
        pos = {
            "pair": "EURUSD", "strike": 1.10, "expiry": 0.25,
            "option_type": "call", "notional": 1_000_000, "direction": "buy",
        }
        for sc_name in FX_STRESS_SCENARIOS:
            result = stress_single_position(pos, 1.10, 0.04, 0.02, 0.08, sc_name)
            assert np.isfinite(result["pnl_impact"]), f"Failed for {sc_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
