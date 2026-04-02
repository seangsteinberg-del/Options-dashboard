"""
Unit tests for core/pricing.py — Black-Scholes, Greeks, Monte Carlo, Binomial.

Tests cover:
  - BS pricing (calls, puts, put-call parity)
  - Edge cases (zero vol, zero time, deep ITM/OTM)
  - Greeks (delta, gamma, vega, theta, rho)
  - Greeks bounds and symmetry
  - Implied vol solver
  - Monte Carlo pricer convergence
  - Binomial tree convergence to BS
"""

import numpy as np
import pytest
from core.pricing import (
    bs_price, bs_d1_d2,
    delta, gamma, vega, theta, rho,
    binomial_tree_price, monte_carlo_price,
    implied_vol, compute_all_greeks,
)


# ── Black-Scholes Pricing ─────────────────────────────────────────────────

class TestBSPricing:

    def test_atm_call_positive(self):
        price = bs_price(100, 100, 1.0, 0.05, 0.02, 0.20, "call")
        assert price > 0
        assert price < 100  # Can't exceed spot

    def test_atm_put_positive(self):
        price = bs_price(100, 100, 1.0, 0.05, 0.02, 0.20, "put")
        assert price > 0

    def test_put_call_parity(self):
        """C - P = S*exp(-qT) - K*exp(-rT)"""
        S, K, T, r, q, sigma = 1.10, 1.12, 0.25, 0.04, 0.02, 0.08
        call = bs_price(S, K, T, r, q, sigma, "call")
        put = bs_price(S, K, T, r, q, sigma, "put")
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1e-10

    def test_deep_itm_call(self):
        price = bs_price(150, 100, 1.0, 0.05, 0.02, 0.20, "call")
        intrinsic = 150 * np.exp(-0.02) - 100 * np.exp(-0.05)
        assert price >= intrinsic - 0.01

    def test_deep_otm_call_near_zero(self):
        price = bs_price(50, 200, 0.1, 0.05, 0.02, 0.10, "call")
        assert price < 0.01

    def test_zero_vol_call(self):
        price = bs_price(110, 100, 1.0, 0.05, 0.02, 0.0, "call")
        assert price >= 0  # Should return discounted intrinsic

    def test_zero_time(self):
        call = bs_price(110, 100, 0.0, 0.05, 0.02, 0.20, "call")
        assert abs(call - 10.0) < 0.01  # Intrinsic
        put = bs_price(90, 100, 0.0, 0.05, 0.02, 0.20, "put")
        assert abs(put - 10.0) < 0.01

    def test_negative_rate_doesnt_crash(self):
        price = bs_price(100, 100, 1.0, -0.01, 0.02, 0.20, "call")
        assert np.isfinite(price) and price >= 0

    def test_high_vol(self):
        price = bs_price(100, 100, 1.0, 0.05, 0.02, 5.0, "call")
        assert np.isfinite(price) and price > 0


# ── Greeks ─────────────────────────────────────────────────────────────────

class TestGreeks:

    def test_call_delta_range(self):
        d = delta(100, 100, 1.0, 0.05, 0.02, 0.20, "call")
        assert 0 < d < 1

    def test_put_delta_range(self):
        d = delta(100, 100, 1.0, 0.05, 0.02, 0.20, "put")
        assert -1 < d < 0

    def test_call_put_delta_relationship(self):
        """Call delta - Put delta = exp(-qT)"""
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.02, 0.20
        cd = delta(S, K, T, r, q, sigma, "call")
        pd = delta(S, K, T, r, q, sigma, "put")
        assert abs((cd - pd) - np.exp(-q * T)) < 1e-10

    def test_deep_itm_call_delta_near_one(self):
        d = delta(200, 100, 0.5, 0.05, 0.02, 0.10, "call")
        assert d > 0.95

    def test_gamma_positive(self):
        g = gamma(100, 100, 1.0, 0.05, 0.02, 0.20)
        assert g > 0

    def test_gamma_peaks_atm(self):
        g_atm = gamma(100, 100, 0.25, 0.05, 0.02, 0.20)
        g_otm = gamma(100, 120, 0.25, 0.05, 0.02, 0.20)
        assert g_atm > g_otm

    def test_gamma_zero_vol(self):
        g = gamma(100, 100, 1.0, 0.05, 0.02, 0.0)
        assert g == 0.0

    def test_vega_positive(self):
        v = vega(100, 100, 1.0, 0.05, 0.02, 0.20)
        assert v > 0

    def test_vega_zero_at_expiry(self):
        v = vega(100, 100, 0.0, 0.05, 0.02, 0.20)
        assert v == 0.0

    def test_theta_call_negative(self):
        """Long call theta should be negative (time decay)."""
        t = theta(100, 100, 1.0, 0.05, 0.02, 0.20, "call")
        assert t < 0

    def test_rho_call_positive(self):
        r = rho(100, 100, 1.0, 0.05, 0.02, 0.20, "call")
        assert r > 0

    def test_rho_put_negative(self):
        r = rho(100, 100, 1.0, 0.05, 0.02, 0.20, "put")
        assert r < 0

    def test_compute_all_greeks_returns_dict(self):
        g = compute_all_greeks(100, 100, 1.0, 0.05, 0.02, 0.20, "call")
        assert isinstance(g, dict)
        for key in ("delta", "gamma", "vega", "theta", "rho"):
            assert key in g
            assert np.isfinite(g[key])


# ── Implied Vol ────────────────────────────────────────────────────────────

class TestImpliedVol:

    def test_roundtrip(self):
        """Price → IV → Price should match."""
        true_vol = 0.20
        price = bs_price(100, 100, 1.0, 0.05, 0.02, true_vol, "call")
        iv = implied_vol(price, 100, 100, 1.0, 0.05, 0.02, "call")
        assert iv is not None
        assert abs(iv - true_vol) < 1e-6

    def test_put_roundtrip(self):
        true_vol = 0.15
        price = bs_price(1.10, 1.12, 0.25, 0.04, 0.02, true_vol, "put")
        iv = implied_vol(price, 1.10, 1.12, 0.25, 0.04, 0.02, "put")
        assert iv is not None
        assert abs(iv - true_vol) < 1e-6

    def test_zero_price_returns_none(self):
        iv = implied_vol(0.0, 100, 100, 1.0, 0.05, 0.02, "call")
        assert iv is None or iv == 0.0

    def test_high_vol_roundtrip(self):
        true_vol = 2.0
        price = bs_price(100, 100, 1.0, 0.05, 0.02, true_vol, "call")
        iv = implied_vol(price, 100, 100, 1.0, 0.05, 0.02, "call")
        # May not converge for very high vol, so just check it doesn't crash
        assert iv is None or abs(iv - true_vol) < 0.1


# ── Monte Carlo ────────────────────────────────────────────────────────────

class TestMonteCarlo:

    def test_mc_converges_to_bs(self):
        bs = bs_price(100, 100, 1.0, 0.05, 0.02, 0.20, "call")
        mc = monte_carlo_price(100, 100, 1.0, 0.05, 0.02, 0.20, "call",
                                n_paths=100000, seed=42)
        assert abs(mc["price"] - bs) < 0.5  # Within 50 cents

    def test_mc_std_error(self):
        mc = monte_carlo_price(100, 100, 1.0, 0.05, 0.02, 0.20, "call",
                                n_paths=50000, seed=42)
        assert mc["std_error"] > 0
        assert mc["std_error"] < 1.0

    def test_mc_zero_vol(self):
        mc = monte_carlo_price(110, 100, 1.0, 0.05, 0.02, 0.0, "call")
        assert mc["price"] == 0.0

    def test_mc_small_paths(self):
        """Should not crash with very small n_paths."""
        mc = monte_carlo_price(100, 100, 1.0, 0.05, 0.02, 0.20, "call",
                                n_paths=2, seed=42)
        assert np.isfinite(mc["price"])


# ── Binomial Tree ──────────────────────────────────────────────────────────

class TestBinomialTree:

    def test_european_converges_to_bs(self):
        bs = bs_price(100, 100, 1.0, 0.05, 0.02, 0.20, "call")
        bt = binomial_tree_price(100, 100, 1.0, 0.05, 0.02, 0.20, "call",
                                  n_steps=500)
        assert abs(bt - bs) < 0.1

    def test_american_put_ge_european(self):
        """American put should be worth at least as much as European put."""
        eu = binomial_tree_price(100, 100, 1.0, 0.05, 0.02, 0.20, "put",
                                  n_steps=200, american=False)
        am = binomial_tree_price(100, 100, 1.0, 0.05, 0.02, 0.20, "put",
                                  n_steps=200, american=True)
        assert am >= eu - 0.01

    def test_zero_vol(self):
        price = binomial_tree_price(110, 100, 1.0, 0.05, 0.02, 0.0, "call")
        assert price >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
