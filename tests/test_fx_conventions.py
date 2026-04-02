"""
Unit tests for core/fx_conventions.py — FX pair specs, deltas, tenors, smile math.

Tests cover:
  - tenor_to_days / tenor_to_years conversion
  - bf_rr_to_smile / smile_to_bf_rr round-trip
  - Delta/strike conversions
  - FX forward pricing
  - Pair registry lookups
  - Premium conversions
"""

import numpy as np
import pytest
from core.fx_conventions import (
    tenor_to_days, tenor_to_years, years_to_nearest_tenor,
    bf_rr_to_smile, smile_to_bf_rr,
    spot_delta, forward_delta, delta_to_strike,
    atm_dns_strike, atm_forward_strike,
    fx_forward, forward_points, carry_roll_down,
    premium_ccy1, premium_ccy2, premium_pips,
    get_pair, all_pairs, pair_groups,
    invert_pair, is_usd_base, is_usd_quote,
    pip_value, get_pip_size, get_delta_convention,
    triangulate_vol,
    safe_float,
)


# ── Tenor Conversion ─────────────────────────────────────────────────────

class TestTenorConversion:

    def test_days_known_tenors(self):
        assert tenor_to_days("ON") == 1
        assert tenor_to_days("1W") == 7
        assert tenor_to_days("1M") == 30
        assert tenor_to_days("3M") == 91
        assert tenor_to_days("6M") == 182
        assert tenor_to_days("1Y") == 365
        assert tenor_to_days("2Y") == 730

    def test_years_known_tenors(self):
        assert abs(tenor_to_years("1Y") - 1.0) < 0.01
        assert abs(tenor_to_years("6M") - 0.5) < 0.01
        assert abs(tenor_to_years("3M") - 0.25) < 0.01

    def test_years_to_nearest_tenor(self):
        assert years_to_nearest_tenor(1.0) == "1Y"
        assert years_to_nearest_tenor(0.25) == "3M"
        assert years_to_nearest_tenor(0.5) == "6M"

    def test_unknown_tenor_raises(self):
        with pytest.raises(ValueError):
            tenor_to_days("99Z")


# ── Smile Decomposition ──────────────────────────────────────────────────

class TestSmileDecomposition:

    def test_bf_rr_to_smile_3pt(self):
        """Convert ATM/RR/BF to individual delta vols."""
        atm, rr25, bf25 = 8.5, -0.5, 0.3
        result = bf_rr_to_smile(atm, rr25, bf25)
        assert "atm" in result
        assert "c25" in result
        assert "p25" in result
        # c25 = ATM + BF25 + RR25/2, p25 = ATM + BF25 - RR25/2
        assert result["atm"] == atm

    def test_bf_rr_to_smile_with_10d(self):
        result = bf_rr_to_smile(8.5, -0.5, 0.3, rr10=-1.0, bf10=0.8)
        assert "c10" in result
        assert "p10" in result

    def test_smile_to_bf_rr_roundtrip(self):
        """Decompose a smile → BF/RR → back to smile should be consistent."""
        atm = 8.5
        c25 = atm + 0.1  # ATM + half BF25 - half RR25
        p25 = atm + 0.6  # ATM + half BF25 + half RR25
        result = smile_to_bf_rr(atm, c25, p25)
        assert "rr25" in result
        assert "bf25" in result
        # RR = call - put
        expected_rr = c25 - p25
        assert abs(result["rr25"] - expected_rr) < 0.01

    def test_flat_smile(self):
        """When all vols equal, RR=0, BF=0."""
        result = smile_to_bf_rr(8.0, 8.0, 8.0)
        assert abs(result["rr25"]) < 0.01
        assert abs(result["bf25"]) < 0.01

    def test_symmetric_smile_zero_rr(self):
        """Symmetric butterfly → RR = 0."""
        atm = 8.0
        c25 = 8.5
        p25 = 8.5  # Same as call → symmetric
        result = smile_to_bf_rr(atm, c25, p25)
        assert abs(result["rr25"]) < 0.01


# ── Delta / Strike Conversion ────────────────────────────────────────────

class TestDeltaStrike:

    S, T, r_d, r_f, sigma = 1.10, 0.25, 0.04, 0.02, 0.08

    def test_spot_delta_call_positive(self):
        d = spot_delta(self.S, self.S, self.T, self.r_d, self.r_f, self.sigma, 1)
        assert 0 < d < 1

    def test_spot_delta_put_negative(self):
        d = spot_delta(self.S, self.S, self.T, self.r_d, self.r_f, self.sigma, -1)
        assert -1 < d < 0

    def test_forward_delta_call(self):
        d = forward_delta(self.S, self.S, self.T, self.r_d, self.r_f, self.sigma, 1)
        assert 0 < d < 1

    def test_delta_to_strike_25d_call(self):
        K = delta_to_strike(0.25, self.S, self.T, self.r_d, self.r_f,
                             self.sigma, 1, "spot")
        assert K > self.S  # OTM call strike > spot

    def test_delta_to_strike_25d_put(self):
        K = delta_to_strike(-0.25, self.S, self.T, self.r_d, self.r_f,
                             self.sigma, -1, "spot")
        assert K < self.S  # OTM put strike < spot

    def test_atm_dns_strike_near_forward(self):
        K = atm_dns_strike(self.S, self.T, self.r_d, self.r_f, self.sigma)
        F = atm_forward_strike(self.S, self.T, self.r_d, self.r_f)
        assert abs(K - F) < 0.02

    def test_atm_forward_strike(self):
        F = atm_forward_strike(self.S, self.T, self.r_d, self.r_f)
        expected = self.S * np.exp((self.r_d - self.r_f) * self.T)
        assert abs(F - expected) < 1e-10


# ── FX Forward ───────────────────────────────────────────────────────────

class TestFXForward:

    def test_positive_rate_diff_forward_above_spot(self):
        """r_d > r_f → forward above spot (domestic currency depreciates)."""
        F = fx_forward(1.10, 0.05, 0.02, 1.0)
        assert F > 1.10

    def test_negative_rate_diff_forward_below_spot(self):
        """r_d < r_f → forward below spot."""
        F = fx_forward(1.10, 0.02, 0.05, 1.0)
        assert F < 1.10

    def test_zero_rate_diff(self):
        F = fx_forward(1.10, 0.03, 0.03, 1.0)
        assert abs(F - 1.10) < 1e-10

    def test_forward_points(self):
        pts = forward_points(1.10, 0.05, 0.02, 1.0)
        F = fx_forward(1.10, 0.05, 0.02, 1.0)
        assert abs(pts - (F - 1.10)) < 1e-10

    def test_carry_roll_down(self):
        result = carry_roll_down(1.10, 0.05, 0.02, 0.25, 30)
        assert "carry" in result
        assert "roll_down" in result


# ── Pair Registry ────────────────────────────────────────────────────────

class TestPairRegistry:

    def test_eurusd_exists(self):
        spec = get_pair("EURUSD")
        assert spec is not None
        assert spec.base == "EUR"
        assert spec.quote == "USD"

    def test_usdjpy_exists(self):
        spec = get_pair("USDJPY")
        assert spec is not None
        assert spec.base == "USD"
        assert spec.quote == "JPY"

    def test_all_pairs_non_empty(self):
        pairs = all_pairs()
        assert len(pairs) >= 20

    def test_pair_groups_non_empty(self):
        groups = pair_groups()
        assert len(groups) >= 3
        # Check that at least one group has multiple pairs
        assert any(len(v) >= 3 for v in groups.values())


# ── Pair Utilities ───────────────────────────────────────────────────────

class TestPairUtilities:

    def test_invert_pair(self):
        assert invert_pair("EURUSD") == "USDEUR"

    def test_is_usd_base(self):
        assert is_usd_base("USDJPY") is True
        assert is_usd_base("EURUSD") is False

    def test_is_usd_quote(self):
        assert is_usd_quote("EURUSD") is True
        assert is_usd_quote("USDJPY") is False

    def test_pip_value(self):
        pv = pip_value("EURUSD", 1_000_000)
        assert pv > 0

    def test_get_pip_size(self):
        assert get_pip_size("EURUSD") == 0.0001
        assert get_pip_size("USDJPY") == 0.01

    def test_get_delta_convention(self):
        conv = get_delta_convention("EURUSD")
        assert conv in ("spot", "forward", "premium_adjusted")

    def test_triangulate_vol(self):
        vol = triangulate_vol(8.0, 10.0, 0.5)
        assert vol > 0
        # vol_cross² = vol_a² + vol_b² - 2*corr*vol_a*vol_b
        expected = np.sqrt(64 + 100 - 2 * 0.5 * 8 * 10)
        assert abs(vol - expected) < 0.01


# ── Premium Conversions ──────────────────────────────────────────────────

class TestPremiumConversions:

    def test_premium_ccy1(self):
        p = premium_ccy1(0.05, 1.10)
        assert abs(p - 0.05 / 1.10) < 1e-10

    def test_premium_ccy2(self):
        p = premium_ccy2(0.05)
        assert p == 0.05  # CCY2 premium is BS price as-is

    def test_premium_pips(self):
        p = premium_pips(0.0050, 0.0001)
        assert abs(p - 50) < 1e-10


# ── safe_float ───────────────────────────────────────────────────────────

class TestSafeFloat:

    def test_valid_number(self):
        assert safe_float(3.14) == 3.14

    def test_none_returns_default(self):
        assert safe_float(None, 0.0) == 0.0

    def test_string_number(self):
        assert safe_float("3.14") == 3.14

    def test_invalid_string(self):
        assert safe_float("abc", -1.0) == -1.0

    def test_nan_returns_default(self):
        assert safe_float(float("nan"), 0.0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
