"""
Unit tests for core/vanna_volga.py — GK pricing, VV smile, SABR model.

Tests cover:
  - GK pricing (calls, puts, put-call parity, edge cases)
  - GK Greeks (vega, vanna, volga)
  - Delta/strike conversion
  - ATM DNS strike
  - VV pricing and implied vol
  - VV smile generation
  - SABR implied vol
  - VV vs SABR comparison
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from core.vanna_volga import (
    gk_price, gk_vega, gk_vanna, gk_volga,
    _gk_d1_d2, _delta_to_strike, _atm_dns_strike,
    vv_price, vv_implied_vol, vv_smile, vv_greeks,
    sabr_vol, vv_vs_sabr,
)


# ── Garman-Kohlhagen Pricing ────────────────────────────────────────────

class TestGKPricing:

    def test_call_positive(self):
        price = gk_price(1.10, 1.10, 0.25, 0.04, 0.02, 0.08, 1)
        assert price > 0

    def test_put_positive(self):
        price = gk_price(1.10, 1.10, 0.25, 0.04, 0.02, 0.08, -1)
        assert price > 0

    def test_put_call_parity(self):
        """C - P = S*exp(-r_f*T) - K*exp(-r_d*T)"""
        S, K, T, r_d, r_f, sigma = 1.10, 1.12, 0.25, 0.04, 0.02, 0.08
        call = gk_price(S, K, T, r_d, r_f, sigma, 1)
        put = gk_price(S, K, T, r_d, r_f, sigma, -1)
        parity = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)
        assert abs((call - put) - parity) < 1e-10

    def test_deep_itm_call(self):
        price = gk_price(1.50, 1.00, 0.5, 0.04, 0.02, 0.10, 1)
        intrinsic = 1.50 * np.exp(-0.02 * 0.5) - 1.00 * np.exp(-0.04 * 0.5)
        assert price >= intrinsic - 0.001

    def test_deep_otm_call_near_zero(self):
        price = gk_price(1.00, 2.00, 0.1, 0.04, 0.02, 0.08, 1)
        assert price < 0.001

    def test_zero_time_call(self):
        call = gk_price(1.15, 1.10, 0.0, 0.04, 0.02, 0.08, 1)
        assert abs(call - 0.05) < 0.001  # Intrinsic

    def test_zero_time_put_otm(self):
        put = gk_price(1.15, 1.10, 0.0, 0.04, 0.02, 0.08, -1)
        assert put == 0.0  # OTM put at expiry

    def test_symmetric_around_atm(self):
        """ATM call and put should be roughly similar."""
        call = gk_price(1.10, 1.10, 0.25, 0.04, 0.02, 0.10, 1)
        put = gk_price(1.10, 1.10, 0.25, 0.04, 0.02, 0.10, -1)
        assert abs(call - put) < 0.01


# ── GK Greeks ────────────────────────────────────────────────────────────

class TestGKGreeks:

    def test_vega_positive(self):
        v = gk_vega(1.10, 1.10, 0.25, 0.04, 0.02, 0.08)
        assert v > 0

    def test_vega_zero_at_expiry(self):
        v = gk_vega(1.10, 1.10, 0.0, 0.04, 0.02, 0.08)
        assert v == 0.0

    def test_vega_peaks_atm(self):
        v_atm = gk_vega(1.10, 1.10, 0.25, 0.04, 0.02, 0.08)
        v_otm = gk_vega(1.10, 1.30, 0.25, 0.04, 0.02, 0.08)
        assert v_atm > v_otm

    def test_vanna_finite(self):
        va = gk_vanna(1.10, 1.10, 0.25, 0.04, 0.02, 0.08)
        assert np.isfinite(va)

    def test_vanna_zero_at_expiry(self):
        va = gk_vanna(1.10, 1.10, 0.0, 0.04, 0.02, 0.08)
        assert va == 0.0

    def test_volga_finite(self):
        vo = gk_volga(1.10, 1.10, 0.25, 0.04, 0.02, 0.08)
        assert np.isfinite(vo)

    def test_volga_zero_at_expiry(self):
        vo = gk_volga(1.10, 1.10, 0.0, 0.04, 0.02, 0.08)
        assert vo == 0.0


# ── Delta/Strike Conversion ──────────────────────────────────────────────

class TestDeltaStrikeConversion:

    def test_call_25_delta(self):
        K = _delta_to_strike(1.10, 0.25, 0.04, 0.02, 0.08, 0.25, 1)
        assert K > 1.10  # 25-delta call strike > spot for positive vol

    def test_put_25_delta(self):
        K = _delta_to_strike(1.10, 0.25, 0.04, 0.02, 0.08, -0.25, -1)
        assert K < 1.10  # 25-delta put strike < spot

    def test_atm_delta_near_spot(self):
        K = _delta_to_strike(1.10, 0.25, 0.04, 0.02, 0.08, 0.50, 1)
        # ATM call delta ≈ 0.50, strike should be near spot
        assert abs(K - 1.10) < 0.05

    def test_atm_dns_strike(self):
        K = _atm_dns_strike(1.10, 0.25, 0.04, 0.02, 0.08)
        # DNS strike should be near forward
        F = 1.10 * np.exp((0.04 - 0.02) * 0.25)
        assert abs(K - F) < 0.02


# ── Vanna-Volga Pricing ──────────────────────────────────────────────────

class TestVVPricing:

    # Typical EURUSD-like parameters
    S = 1.10
    T = 0.25
    r_d = 0.04
    r_f = 0.02
    atm = 0.08
    p25 = 0.085  # 25D put vol (higher due to skew)
    c25 = 0.078  # 25D call vol

    def test_vv_call_positive(self):
        price = vv_price(self.S, self.S, self.T, self.r_d, self.r_f,
                          self.atm, self.p25, self.c25, 1)
        assert price > 0

    def test_vv_put_positive(self):
        price = vv_price(self.S, self.S, self.T, self.r_d, self.r_f,
                          self.atm, self.p25, self.c25, -1)
        assert price > 0

    def test_vv_at_pivot_matches_market(self):
        """VV price at ATM strike should be very close to GK with ATM vol."""
        K_atm = _atm_dns_strike(self.S, self.T, self.r_d, self.r_f, self.atm)
        vv = vv_price(self.S, K_atm, self.T, self.r_d, self.r_f,
                       self.atm, self.p25, self.c25, 1)
        gk = gk_price(self.S, K_atm, self.T, self.r_d, self.r_f, self.atm, 1)
        assert abs(vv - gk) < 0.001

    def test_vv_zero_time(self):
        price = vv_price(self.S, self.S - 0.05, 0.0, self.r_d, self.r_f,
                          self.atm, self.p25, self.c25, 1)
        assert abs(price - 0.05) < 0.001

    def test_vv_implied_vol_roundtrip(self):
        """VV price → VV IV should recover something reasonable."""
        K = self.S * 1.02
        iv = vv_implied_vol(self.S, K, self.T, self.r_d, self.r_f,
                             self.atm, self.p25, self.c25)
        assert iv is not None
        assert 0.01 < iv < 1.0  # Reasonable IV range


# ── VV Smile ─────────────────────────────────────────────────────────────

class TestVVSmile:

    def test_smile_returns_dict(self):
        smile = vv_smile(1.10, 0.25, 0.04, 0.02, 0.08, 0.085, 0.078)
        assert isinstance(smile, dict)
        assert "strikes" in smile
        assert "vols" in smile
        assert "deltas" in smile

    def test_smile_length(self):
        smile = vv_smile(1.10, 0.25, 0.04, 0.02, 0.08, 0.085, 0.078, n_strikes=50)
        assert len(smile["strikes"]) == 50
        assert len(smile["vols"]) == 50
        assert len(smile["deltas"]) == 50

    def test_smile_vols_positive(self):
        smile = vv_smile(1.10, 0.25, 0.04, 0.02, 0.08, 0.085, 0.078)
        for v in smile["vols"]:
            assert v > 0

    def test_smile_u_shape(self):
        """Smile should be roughly U-shaped: wings higher than center."""
        smile = vv_smile(1.10, 0.25, 0.04, 0.02, 0.08, 0.085, 0.078, n_strikes=50)
        vols = smile["vols"]
        center_vol = vols[len(vols) // 2]
        # At least one wing should be higher
        assert max(vols[0], vols[-1]) >= center_vol

    def test_flat_smile_when_symmetric(self):
        """When p25_vol == c25_vol == atm_vol, smile should be flat."""
        smile = vv_smile(1.10, 0.25, 0.04, 0.02, 0.10, 0.10, 0.10, n_strikes=20)
        vols = smile["vols"]
        for v in vols:
            assert abs(v - 0.10) < 0.005  # All vols near ATM


# ── VV Greeks ────────────────────────────────────────────────────────────

class TestVVGreeks:

    def test_greeks_returns_dict(self):
        g = vv_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.08, 0.085, 0.078, 1)
        assert isinstance(g, dict)
        for key in ("delta", "gamma", "vega", "theta", "rho_d", "rho_f",
                     "vanna", "volga"):
            assert key in g
            assert np.isfinite(g[key])

    def test_call_delta_positive(self):
        g = vv_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.08, 0.085, 0.078, 1)
        assert g["delta"] > 0

    def test_put_delta_negative(self):
        g = vv_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.08, 0.085, 0.078, -1)
        assert g["delta"] < 0

    def test_gamma_positive(self):
        g = vv_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.08, 0.085, 0.078, 1)
        assert g["gamma"] > 0

    def test_vega_positive(self):
        g = vv_greeks(1.10, 1.10, 0.25, 0.04, 0.02, 0.08, 0.085, 0.078, 1)
        assert g["vega"] > 0


# ── SABR Implied Vol ─────────────────────────────────────────────────────

class TestSABR:

    def test_atm_vol(self):
        F = 1.10
        vol = sabr_vol(F, F, 0.25, 0.08, 0.5, -0.2, 0.4)
        assert vol > 0
        assert vol < 1.0

    def test_otm_vol(self):
        vol = sabr_vol(1.10, 1.20, 0.25, 0.08, 0.5, -0.2, 0.4)
        assert vol > 0
        assert vol < 1.0

    def test_negative_rho_puts_higher(self):
        """Negative rho → put-side vol higher than call-side."""
        F = 1.10
        vol_put_side = sabr_vol(F, F * 0.95, 0.25, 0.08, 0.5, -0.3, 0.4)
        vol_call_side = sabr_vol(F, F * 1.05, 0.25, 0.08, 0.5, -0.3, 0.4)
        assert vol_put_side > vol_call_side

    def test_zero_vol_of_vol(self):
        """With nu=0, SABR ATM vol should still be positive."""
        F = 1.10
        vol_atm = sabr_vol(F, F, 0.25, 0.08, 0.5, 0.0, 0.0)
        assert vol_atm > 0

    def test_invalid_inputs(self):
        vol = sabr_vol(0, 1.10, 0.25, 0.08, 0.5, -0.2, 0.4)
        assert vol == 1e-6
        vol = sabr_vol(1.10, 1.10, 0.0, 0.08, 0.5, -0.2, 0.4)
        assert vol == 1e-6

    def test_high_vol_of_vol(self):
        """High nu shouldn't crash."""
        vol = sabr_vol(1.10, 1.10, 0.25, 0.08, 0.5, -0.2, 2.0)
        assert np.isfinite(vol) and vol > 0


# ── VV vs SABR ───────────────────────────────────────────────────────────

class TestVVvsSABR:

    def test_comparison_returns_dataframe(self):
        result = vv_vs_sabr(
            S=1.10, T=0.25, r_d=0.04, r_f=0.02,
            atm_vol=0.08, p25_vol=0.085, c25_vol=0.078,
            sabr_alpha=0.08, sabr_rho=-0.2, sabr_nu=0.4,
        )
        assert len(result) == 50
        for col in ("delta", "strike", "vv_vol", "sabr_vol", "diff"):
            assert col in result.columns

    def test_vols_all_positive(self):
        result = vv_vs_sabr(
            S=1.10, T=0.25, r_d=0.04, r_f=0.02,
            atm_vol=0.08, p25_vol=0.085, c25_vol=0.078,
            sabr_alpha=0.08, sabr_rho=-0.2, sabr_nu=0.4,
        )
        assert (result["vv_vol"] > 0).all()
        assert (result["sabr_vol"] > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
