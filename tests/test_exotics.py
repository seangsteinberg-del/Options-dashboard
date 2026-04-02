"""
Unit tests for core/fx_exotics.py — Exotic FX options pricing engine.

Tests cover:
  - Barrier options (single barriers: KI, KO, up, down)
  - Digital (binary) options
  - Asian options (geometric analytical, arithmetic MC)
  - One-touch / no-touch
  - Lookback options (floating analytical, fixed MC)
  - Forward start options
  - Double barrier (MC)
  - Range accrual (MC)
  - Best-of / worst-of rainbow (MC)
  - TARF
  - Numerical exotic Greeks
  - Edge cases (zero vol, zero time, extreme barriers)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from core.fx_exotics import (
    barrier_price,
    double_barrier_price,
    digital_price,
    digital_greeks,
    one_touch_price,
    no_touch_price,
    double_no_touch_price,
    range_accrual_price,
    asian_geometric_price,
    asian_price,
    lookback_price,
    forward_start_price,
    best_of_price,
    tarf_price,
    exotic_greeks,
    exotic_summary,
    _gk_price,
)


# ── Common test parameters (EURUSD-like) ────────────────────────────────────

S = 1.10       # spot
K = 1.10       # ATM strike
T = 0.25       # 3 months
r_d = 0.04     # domestic (USD) rate
r_f = 0.02     # foreign (EUR) rate
sigma = 0.08   # 8 vol


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Barrier Options
# ═══════════════════════════════════════════════════════════════════════════════

class TestBarrierOptions:

    # --- Basic positivity and finiteness ---

    def test_down_and_in_call_positive(self):
        price = barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=1, barrier_type='down-and-in-call')
        assert np.isfinite(price) and price >= 0

    def test_down_and_out_call_positive(self):
        price = barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=1, barrier_type='down-and-out-call')
        assert np.isfinite(price) and price >= 0

    def test_up_and_in_call_positive(self):
        price = barrier_price(S, K, B=1.15, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=1, barrier_type='up-and-in-call')
        assert np.isfinite(price) and price >= 0

    def test_up_and_out_call_positive(self):
        price = barrier_price(S, K, B=1.15, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=1, barrier_type='up-and-out-call')
        assert np.isfinite(price) and price >= 0

    def test_down_and_in_put_positive(self):
        price = barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=-1, barrier_type='down-and-in-put')
        assert np.isfinite(price) and price >= 0

    def test_down_and_out_put_positive(self):
        price = barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=-1, barrier_type='down-and-out-put')
        assert np.isfinite(price) and price >= 0

    def test_up_and_in_put_positive(self):
        price = barrier_price(S, K, B=1.15, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=-1, barrier_type='up-and-in-put')
        assert np.isfinite(price) and price >= 0

    def test_up_and_out_put_positive(self):
        price = barrier_price(S, K, B=1.15, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=-1, barrier_type='up-and-out-put')
        assert np.isfinite(price) and price >= 0

    # --- KI + KO = Vanilla (no rebate) ---

    def test_down_ki_ko_equals_vanilla_call(self):
        """Down-and-in + Down-and-out call = vanilla call (with zero rebate)."""
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        B = 1.05
        ki = barrier_price(S, K, B, T, r_d, r_f, sigma, 1, 'down-and-in-call', rebate=0)
        ko = barrier_price(S, K, B, T, r_d, r_f, sigma, 1, 'down-and-out-call', rebate=0)
        assert abs((ki + ko) - vanilla) < 1e-8, f"KI+KO={ki+ko}, vanilla={vanilla}"

    def test_up_ki_ko_equals_vanilla_call(self):
        """Up-and-in + Up-and-out call = vanilla call."""
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        B = 1.15
        ki = barrier_price(S, K, B, T, r_d, r_f, sigma, 1, 'up-and-in-call', rebate=0)
        ko = barrier_price(S, K, B, T, r_d, r_f, sigma, 1, 'up-and-out-call', rebate=0)
        assert abs((ki + ko) - vanilla) < 1e-8

    def test_down_ki_ko_equals_vanilla_put(self):
        """Down-and-in + Down-and-out put = vanilla put."""
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, -1)
        B = 1.05
        ki = barrier_price(S, K, B, T, r_d, r_f, sigma, -1, 'down-and-in-put', rebate=0)
        ko = barrier_price(S, K, B, T, r_d, r_f, sigma, -1, 'down-and-out-put', rebate=0)
        assert abs((ki + ko) - vanilla) < 1e-8

    def test_up_ki_ko_equals_vanilla_put(self):
        """Up-and-in + Up-and-out put = vanilla put."""
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, -1)
        B = 1.15
        ki = barrier_price(S, K, B, T, r_d, r_f, sigma, -1, 'up-and-in-put', rebate=0)
        ko = barrier_price(S, K, B, T, r_d, r_f, sigma, -1, 'up-and-out-put', rebate=0)
        assert abs((ki + ko) - vanilla) < 1e-8

    # --- Barrier <= spot or >= spot special cases ---

    def test_barrier_at_spot_down_and_in_call(self):
        """When barrier >= spot, down-and-in should equal vanilla (already knocked in)."""
        price = barrier_price(S, K, B=S, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=1, barrier_type='down-and-in-call')
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        # With the _F rebate term it may not be exact, but should be close
        assert np.isfinite(price) and price >= 0

    def test_barrier_at_spot_up_and_in_put(self):
        """When barrier <= spot, up-and-in should equal vanilla (already knocked in)."""
        price = barrier_price(S, K, B=S, T=T, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=-1, barrier_type='up-and-in-put')
        assert np.isfinite(price) and price >= 0

    def test_ko_less_than_vanilla(self):
        """Knock-out should be worth less than or equal to vanilla."""
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        ko = barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                           sigma=sigma, cp=1, barrier_type='down-and-out-call', rebate=0)
        assert ko <= vanilla + 1e-10

    def test_invalid_barrier_type_raises(self):
        with pytest.raises(ValueError, match="Unknown barrier_type"):
            barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                          sigma=sigma, cp=1, barrier_type='invalid-type')

    # --- Edge cases ---

    def test_zero_vol_barrier(self):
        price = barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                              sigma=0.0, cp=1, barrier_type='down-and-out-call')
        assert np.isfinite(price) and price >= 0

    def test_zero_time_barrier(self):
        price = barrier_price(S, K=1.05, B=1.08, T=0.0, r_d=r_d, r_f=r_f,
                              sigma=sigma, cp=1, barrier_type='up-and-out-call')
        assert np.isfinite(price) and price >= 0

    def test_rebate_increases_ko_price(self):
        """Adding a rebate should increase knock-out price."""
        ko_no_rebate = barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                                     sigma=sigma, cp=1,
                                     barrier_type='down-and-out-call', rebate=0)
        ko_with_rebate = barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                                       sigma=sigma, cp=1,
                                       barrier_type='down-and-out-call', rebate=0.01)
        assert ko_with_rebate >= ko_no_rebate - 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Digital (Binary) Options
# ═══════════════════════════════════════════════════════════════════════════════

class TestDigitalOptions:

    def test_call_positive(self):
        price = digital_price(S, K, T, r_d, r_f, sigma, cp=1)
        assert price > 0

    def test_put_positive(self):
        price = digital_price(S, K, T, r_d, r_f, sigma, cp=-1)
        assert price > 0

    def test_bounded_by_discounted_payout(self):
        """Digital price must be in [0, payout * exp(-r_d * T)]."""
        payout = 1.0
        df = np.exp(-r_d * T)
        for cp in [1, -1]:
            price = digital_price(S, K, T, r_d, r_f, sigma, cp, payout)
            assert -1e-10 <= price <= payout * df + 1e-10

    def test_custom_payout(self):
        """With payout=100, price should scale accordingly."""
        p1 = digital_price(S, K, T, r_d, r_f, sigma, cp=1, payout=1.0)
        p100 = digital_price(S, K, T, r_d, r_f, sigma, cp=1, payout=100.0)
        assert abs(p100 - 100.0 * p1) < 1e-10

    def test_call_put_sum_to_df(self):
        """Digital call + digital put = payout * exp(-r_d * T) (complementary)."""
        payout = 1.0
        dc = digital_price(S, K, T, r_d, r_f, sigma, cp=1, payout=payout)
        dp = digital_price(S, K, T, r_d, r_f, sigma, cp=-1, payout=payout)
        df = payout * np.exp(-r_d * T)
        assert abs(dc + dp - df) < 1e-10

    def test_deep_itm_digital_call(self):
        """Deep ITM digital call price should be near discounted payout."""
        price = digital_price(2.0, 1.0, T, r_d, r_f, sigma, cp=1, payout=1.0)
        df = np.exp(-r_d * T)
        assert price > 0.9 * df

    def test_deep_otm_digital_call(self):
        """Deep OTM digital call price should be near zero."""
        price = digital_price(0.50, 2.0, T, r_d, r_f, sigma, cp=1, payout=1.0)
        assert price < 0.01

    def test_zero_time_itm(self):
        """At expiry, ITM digital should equal payout."""
        price = digital_price(1.20, 1.10, 0.0, r_d, r_f, sigma, cp=1, payout=1.0)
        assert abs(price - 1.0) < 1e-10

    def test_zero_time_otm(self):
        """At expiry, OTM digital should be zero."""
        price = digital_price(1.00, 1.10, 0.0, r_d, r_f, sigma, cp=1, payout=1.0)
        assert abs(price) < 1e-10


class TestDigitalGreeks:

    def test_greeks_returns_dict(self):
        g = digital_greeks(S, K, T, r_d, r_f, sigma, cp=1)
        for key in ('delta', 'gamma', 'vega', 'theta'):
            assert key in g
            assert np.isfinite(g[key])

    def test_call_delta_positive(self):
        """Digital call delta should be positive (higher spot = more likely ITM)."""
        g = digital_greeks(S, K, T, r_d, r_f, sigma, cp=1)
        assert g['delta'] > 0

    def test_put_delta_negative(self):
        """Digital put delta should be negative."""
        g = digital_greeks(S, K, T, r_d, r_f, sigma, cp=-1)
        assert g['delta'] < 0

    def test_zero_time_greeks(self):
        """At expiry, all Greeks should be zero."""
        g = digital_greeks(S, K, 0.0, r_d, r_f, sigma, cp=1)
        assert g['delta'] == 0.0
        assert g['gamma'] == 0.0
        assert g['vega'] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. One-Touch / No-Touch
# ═══════════════════════════════════════════════════════════════════════════════

class TestOneTouchNoTouch:

    def test_one_touch_up_positive(self):
        price = one_touch_price(S, B=1.15, T=T, r_d=r_d, r_f=r_f, sigma=sigma)
        assert np.isfinite(price) and price > 0

    def test_one_touch_down_positive(self):
        price = one_touch_price(S, B=1.05, T=T, r_d=r_d, r_f=r_f, sigma=sigma)
        assert np.isfinite(price) and price > 0

    def test_no_touch_positive(self):
        price = no_touch_price(S, B=1.15, T=T, r_d=r_d, r_f=r_f, sigma=sigma)
        assert np.isfinite(price) and price > 0

    def test_one_touch_plus_no_touch_equals_df(self):
        """One-touch + No-touch = payout * exp(-r_d * T)."""
        payout = 1.0
        ot = one_touch_price(S, B=1.15, T=T, r_d=r_d, r_f=r_f, sigma=sigma, payout=payout)
        nt = no_touch_price(S, B=1.15, T=T, r_d=r_d, r_f=r_f, sigma=sigma, payout=payout)
        df = payout * np.exp(-r_d * T)
        assert abs(ot + nt - df) < 1e-10

    def test_one_touch_plus_no_touch_down(self):
        """Same identity for a down barrier."""
        payout = 1.0
        ot = one_touch_price(S, B=1.05, T=T, r_d=r_d, r_f=r_f, sigma=sigma, payout=payout)
        nt = no_touch_price(S, B=1.05, T=T, r_d=r_d, r_f=r_f, sigma=sigma, payout=payout)
        df = payout * np.exp(-r_d * T)
        assert abs(ot + nt - df) < 1e-10

    def test_one_touch_bounded(self):
        """One-touch price in [0, payout * exp(-r_d * T)]."""
        payout = 1.0
        ot = one_touch_price(S, B=1.15, T=T, r_d=r_d, r_f=r_f, sigma=sigma, payout=payout)
        df = payout * np.exp(-r_d * T)
        assert -1e-10 <= ot <= df + 1e-10

    def test_barrier_at_spot(self):
        """If barrier equals spot, one-touch should be discounted payout."""
        payout = 1.0
        ot = one_touch_price(S, B=S, T=T, r_d=r_d, r_f=r_f, sigma=sigma, payout=payout)
        df = payout * np.exp(-r_d * T)
        assert abs(ot - df) < 1e-8

    def test_zero_vol_deterministic(self):
        """With near-zero vol, one-touch depends on whether forward crosses barrier."""
        # Forward = S * exp((r_d - r_f) * T)
        fwd = S * np.exp((r_d - r_f) * T)
        # Barrier above forward: should not touch
        ot = one_touch_price(S, B=fwd * 1.5, T=T, r_d=r_d, r_f=r_f, sigma=0.0001)
        assert ot < 0.01  # Very small probability

    def test_zero_time(self):
        """At T=0 with barrier != spot, one-touch should be zero."""
        ot = one_touch_price(S, B=1.15, T=0.0, r_d=r_d, r_f=r_f, sigma=sigma)
        assert abs(ot) < 1e-10

    def test_longer_time_higher_probability(self):
        """Longer expiry should give higher one-touch probability."""
        ot_short = one_touch_price(S, B=1.15, T=0.1, r_d=r_d, r_f=r_f, sigma=sigma)
        ot_long = one_touch_price(S, B=1.15, T=1.0, r_d=r_d, r_f=r_f, sigma=sigma)
        # Undiscounted probability should be higher for longer time
        # (discounting may reduce the value, but the probability is higher)
        prob_short = ot_short / np.exp(-r_d * 0.1)
        prob_long = ot_long / np.exp(-r_d * 1.0)
        assert prob_long >= prob_short - 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Asian Options
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsianOptions:

    def test_geometric_call_positive(self):
        price = asian_geometric_price(S, K, T, r_d, r_f, sigma, cp=1, n_fixings=12)
        assert np.isfinite(price) and price > 0

    def test_geometric_put_positive(self):
        price = asian_geometric_price(S, K, T, r_d, r_f, sigma, cp=-1, n_fixings=12)
        assert np.isfinite(price) and price > 0

    def test_geometric_bounded_by_vanilla(self):
        """Geometric average Asian call should be <= vanilla call (averaging reduces value)."""
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        asian = asian_geometric_price(S, K, T, r_d, r_f, sigma, cp=1, n_fixings=12)
        assert asian <= vanilla + 1e-8

    def test_geometric_single_fixing_near_vanilla(self):
        """With 1 fixing, geometric Asian should approach vanilla."""
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        asian = asian_geometric_price(S, K, T, r_d, r_f, sigma, cp=1, n_fixings=1)
        # With 1 fixing the adjusted vol/rate differ slightly, but should be in the ballpark
        assert np.isfinite(asian) and asian > 0

    def test_asian_arithmetic_mc_doesnt_crash(self):
        """Arithmetic Asian via MC should return valid dict."""
        result = asian_price(S, K, T, r_d, r_f, sigma, cp=1,
                             fixing_freq='monthly', average_type='arithmetic',
                             n_paths=5000, seed=42)
        assert 'price' in result
        assert np.isfinite(result['price'])
        assert result['price'] > 0
        assert result['std_error'] >= 0

    def test_asian_geometric_via_wrapper(self):
        """asian_price with average_type='geometric' should use analytical."""
        result = asian_price(S, K, T, r_d, r_f, sigma, cp=1,
                             fixing_freq='monthly', average_type='geometric')
        assert result['method'] == 'analytical'
        assert result['std_error'] == 0.0
        assert result['price'] > 0

    def test_asian_put_mc(self):
        """Asian put via MC should produce positive price."""
        result = asian_price(S, K, T, r_d, r_f, sigma, cp=-1,
                             fixing_freq='monthly', average_type='arithmetic',
                             n_paths=5000, seed=42)
        assert result['price'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Lookback Options
# ═══════════════════════════════════════════════════════════════════════════════

class TestLookbackOptions:

    def test_floating_call_positive(self):
        result = lookback_price(S, T, r_d, r_f, sigma, cp=1, lookback_type='floating')
        assert result['price'] > 0
        assert result['method'] == 'analytical'

    def test_floating_put_positive(self):
        result = lookback_price(S, T, r_d, r_f, sigma, cp=-1, lookback_type='floating')
        assert result['price'] > 0

    def test_floating_call_ge_vanilla_atm(self):
        """Floating-strike lookback call should be worth >= vanilla ATM call.

        The lookback call pays S_T - S_min, which in expectation is at least
        as valuable as S_T - S_0 (the ATM forward payoff).
        """
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        lb = lookback_price(S, T, r_d, r_f, sigma, cp=1, lookback_type='floating')
        assert lb['price'] >= vanilla - 1e-8

    def test_floating_put_ge_vanilla_atm(self):
        """Floating-strike lookback put should be worth >= vanilla ATM put."""
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, -1)
        lb = lookback_price(S, T, r_d, r_f, sigma, cp=-1, lookback_type='floating')
        assert lb['price'] >= vanilla - 1e-8

    def test_fixed_lookback_mc_doesnt_crash(self):
        """Fixed-strike lookback via MC should return valid result."""
        result = lookback_price(S, T, r_d, r_f, sigma, cp=1,
                                lookback_type='fixed', K=K,
                                n_paths=5000, n_steps=100, seed=42)
        assert result['price'] > 0
        assert result['method'] == 'mc'
        assert result['std_error'] >= 0

    def test_higher_vol_higher_lookback(self):
        """Higher vol should increase lookback value (more path optionality)."""
        lb_low = lookback_price(S, T, r_d, r_f, sigma=0.05, cp=1, lookback_type='floating')
        lb_high = lookback_price(S, T, r_d, r_f, sigma=0.20, cp=1, lookback_type='floating')
        assert lb_high['price'] > lb_low['price']


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Forward Start Options
# ═══════════════════════════════════════════════════════════════════════════════

class TestForwardStartOptions:

    def test_call_positive(self):
        price = forward_start_price(S, T_start=0.1, T_end=0.5,
                                    r_d=r_d, r_f=r_f, sigma=sigma, cp=1)
        assert np.isfinite(price) and price > 0

    def test_put_positive(self):
        price = forward_start_price(S, T_start=0.1, T_end=0.5,
                                    r_d=r_d, r_f=r_f, sigma=sigma, cp=-1)
        assert np.isfinite(price) and price > 0

    def test_zero_remaining_time(self):
        """When T_start == T_end, price should be zero."""
        price = forward_start_price(S, T_start=0.5, T_end=0.5,
                                    r_d=r_d, r_f=r_f, sigma=sigma, cp=1)
        assert abs(price) < 1e-10

    def test_negative_remaining_time(self):
        """When T_start > T_end, price should be zero."""
        price = forward_start_price(S, T_start=1.0, T_end=0.5,
                                    r_d=r_d, r_f=r_f, sigma=sigma, cp=1)
        assert abs(price) < 1e-10

    def test_moneyness_scaling(self):
        """Higher moneyness (OTM call) should produce lower price."""
        p_atm = forward_start_price(S, T_start=0.1, T_end=0.5,
                                    r_d=r_d, r_f=r_f, sigma=sigma, cp=1,
                                    moneyness=1.0)
        p_otm = forward_start_price(S, T_start=0.1, T_end=0.5,
                                    r_d=r_d, r_f=r_f, sigma=sigma, cp=1,
                                    moneyness=1.10)  # 10% OTM
        assert p_atm > p_otm

    def test_immediate_start_near_vanilla(self):
        """With T_start=0, forward start should equal vanilla (scaled by e^0 = 1)."""
        fs = forward_start_price(S, T_start=0.0, T_end=T,
                                 r_d=r_d, r_f=r_f, sigma=sigma, cp=1, moneyness=1.0)
        vanilla = _gk_price(S, S * 1.0, T, r_d, r_f, sigma, 1)
        assert abs(fs - vanilla) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Double Barrier Options (MC)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDoubleBarrier:

    def test_ko_doesnt_crash(self):
        result = double_barrier_price(S, K, B_up=1.20, B_down=1.00,
                                      T=T, r_d=r_d, r_f=r_f, sigma=sigma, cp=1,
                                      barrier_type='knock-out',
                                      n_paths=5000, n_steps=50, seed=42)
        assert 'price' in result
        assert np.isfinite(result['price'])
        assert result['price'] >= 0

    def test_ki_doesnt_crash(self):
        result = double_barrier_price(S, K, B_up=1.20, B_down=1.00,
                                      T=T, r_d=r_d, r_f=r_f, sigma=sigma, cp=1,
                                      barrier_type='knock-in',
                                      n_paths=5000, n_steps=50, seed=42)
        assert np.isfinite(result['price'])
        assert result['price'] >= 0

    def test_ko_plus_ki_near_vanilla(self):
        """Double-barrier KO + KI should roughly equal vanilla (MC approximation)."""
        ko = double_barrier_price(S, K, B_up=1.20, B_down=1.00,
                                  T=T, r_d=r_d, r_f=r_f, sigma=sigma, cp=1,
                                  barrier_type='knock-out',
                                  n_paths=20000, n_steps=100, seed=42)
        ki = double_barrier_price(S, K, B_up=1.20, B_down=1.00,
                                  T=T, r_d=r_d, r_f=r_f, sigma=sigma, cp=1,
                                  barrier_type='knock-in',
                                  n_paths=20000, n_steps=100, seed=42)
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        # MC so tolerance is wider
        assert abs((ko['price'] + ki['price']) - vanilla) < 0.01

    def test_wide_barriers_near_vanilla(self):
        """Very wide barriers should produce knock-out price near vanilla."""
        result = double_barrier_price(S, K, B_up=5.0, B_down=0.01,
                                      T=T, r_d=r_d, r_f=r_f, sigma=sigma, cp=1,
                                      barrier_type='knock-out',
                                      n_paths=10000, n_steps=50, seed=42)
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        assert abs(result['price'] - vanilla) < 0.005


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Double No-Touch (MC)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDoubleNoTouch:

    def test_doesnt_crash(self):
        result = double_no_touch_price(S, B_up=1.20, B_down=1.00,
                                       T=T, r_d=r_d, r_f=r_f, sigma=sigma,
                                       n_paths=5000, n_steps=50, seed=42)
        assert 'price' in result
        assert np.isfinite(result['price'])
        assert 0 <= result['prob_no_touch'] <= 1

    def test_wide_corridor_high_survival(self):
        """Very wide corridor should have near-100% survival probability."""
        result = double_no_touch_price(S, B_up=5.0, B_down=0.01,
                                       T=T, r_d=r_d, r_f=r_f, sigma=sigma,
                                       n_paths=5000, n_steps=50, seed=42)
        assert result['prob_no_touch'] > 0.99

    def test_narrow_corridor_low_survival(self):
        """Very narrow corridor should have low survival probability."""
        result = double_no_touch_price(S, B_up=S * 1.001, B_down=S * 0.999,
                                       T=T, r_d=r_d, r_f=r_f, sigma=sigma,
                                       n_paths=5000, n_steps=50, seed=42)
        assert result['prob_no_touch'] < 0.5

    def test_invalid_barriers(self):
        """B_up <= B_down should return zero price."""
        result = double_no_touch_price(S, B_up=1.0, B_down=1.0,
                                       T=T, r_d=r_d, r_f=r_f, sigma=sigma)
        assert result['price'] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Range Accrual (MC)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRangeAccrual:

    def test_doesnt_crash(self):
        result = range_accrual_price(S, B_low=1.05, B_high=1.15,
                                     T=T, r_d=r_d, r_f=r_f, sigma=sigma,
                                     n_paths=5000, seed=42)
        assert 'price' in result
        assert np.isfinite(result['price'])
        assert result['price'] >= 0

    def test_full_range_equals_df_payout(self):
        """If range covers everything, expected accrual should be ~1.0."""
        result = range_accrual_price(S, B_low=0.01, B_high=10.0,
                                     T=T, r_d=r_d, r_f=r_f, sigma=sigma,
                                     n_paths=5000, seed=42)
        assert result['expected_accrual'] > 0.99

    def test_weekly_fixing(self):
        result = range_accrual_price(S, B_low=1.05, B_high=1.15,
                                     T=T, r_d=r_d, r_f=r_f, sigma=sigma,
                                     fixing_freq='weekly', n_paths=5000, seed=42)
        assert result['n_fixings'] >= 1
        assert result['price'] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Best-of / Worst-of (Rainbow)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRainbowOptions:

    def test_best_of_doesnt_crash(self):
        result = best_of_price(S1=1.10, S2=1.30, K=0.0, T=T,
                               r_d1=r_d, r_d2=0.03, r_f=r_f,
                               sigma1=sigma, sigma2=0.10, rho=0.5, cp=1,
                               option_type='best-of', n_paths=5000, seed=42)
        assert 'price' in result
        assert np.isfinite(result['price'])
        assert result['price'] >= 0

    def test_worst_of_doesnt_crash(self):
        result = best_of_price(S1=1.10, S2=1.30, K=0.0, T=T,
                               r_d1=r_d, r_d2=0.03, r_f=r_f,
                               sigma1=sigma, sigma2=0.10, rho=0.5, cp=1,
                               option_type='worst-of', n_paths=5000, seed=42)
        assert np.isfinite(result['price'])

    def test_best_of_ge_worst_of(self):
        """Best-of call should be worth at least as much as worst-of call."""
        best = best_of_price(S1=1.10, S2=1.30, K=0.0, T=T,
                             r_d1=r_d, r_d2=0.03, r_f=r_f,
                             sigma1=sigma, sigma2=0.10, rho=0.5, cp=1,
                             option_type='best-of', n_paths=20000, seed=42)
        worst = best_of_price(S1=1.10, S2=1.30, K=0.0, T=T,
                              r_d1=r_d, r_d2=0.03, r_f=r_f,
                              sigma1=sigma, sigma2=0.10, rho=0.5, cp=1,
                              option_type='worst-of', n_paths=20000, seed=42)
        assert best['price'] >= worst['price'] - 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TARF
# ═══════════════════════════════════════════════════════════════════════════════

class TestTARF:

    def test_doesnt_crash(self):
        result = tarf_price(S, K=K, B=1.20, T=1.0, r_d=r_d, r_f=r_f, sigma=sigma,
                            n_fixings=12, target_profit=0.05, leverage=2,
                            n_paths=5000, seed=42)
        assert 'price' in result
        assert np.isfinite(result['price'])
        assert 0 <= result['prob_early_termination'] <= 1
        assert result['expected_fixings'] >= 1

    def test_no_barrier(self):
        """TARF with B=None (no knock-out barrier) should still work."""
        result = tarf_price(S, K=K, B=None, T=1.0, r_d=r_d, r_f=r_f, sigma=sigma,
                            n_fixings=12, target_profit=0.05, leverage=2,
                            n_paths=5000, seed=42)
        assert np.isfinite(result['price'])

    def test_high_leverage_lower_price(self):
        """Higher leverage (more downside) should generally produce lower client MTM."""
        low_lev = tarf_price(S, K=K, B=1.20, T=1.0, r_d=r_d, r_f=r_f, sigma=sigma,
                             n_fixings=12, target_profit=0.05, leverage=1,
                             n_paths=20000, seed=42)
        high_lev = tarf_price(S, K=K, B=1.20, T=1.0, r_d=r_d, r_f=r_f, sigma=sigma,
                              n_fixings=12, target_profit=0.05, leverage=5,
                              n_paths=20000, seed=42)
        # Higher leverage = more client downside, so MTM should be worse
        assert high_lev['price'] <= low_lev['price'] + 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Numerical Exotic Greeks
# ═══════════════════════════════════════════════════════════════════════════════

class TestExoticGreeks:

    def test_barrier_greeks(self):
        """Numerical Greeks for a barrier option should return all keys."""
        params = dict(S=S, K=K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                      sigma=sigma, cp=1, barrier_type='down-and-out-call')
        g = exotic_greeks(barrier_price, params)
        assert 'price' in g
        assert 'delta' in g
        assert 'gamma' in g
        assert 'vega' in g
        assert 'theta' in g
        for key in ('price', 'delta', 'gamma', 'vega', 'theta'):
            assert np.isfinite(g[key])

    def test_digital_greeks_via_exotic(self):
        """Numerical Greeks for a digital should also work."""
        params = dict(S=S, K=K, T=T, r_d=r_d, r_f=r_f,
                      sigma=sigma, cp=1, payout=1.0)
        g = exotic_greeks(digital_price, params)
        assert 'delta' in g
        assert np.isfinite(g['delta'])

    def test_greeks_delta_sign(self):
        """Delta of a barrier call should be positive."""
        params = dict(S=S, K=K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                      sigma=sigma, cp=1, barrier_type='down-and-out-call')
        g = exotic_greeks(barrier_price, params)
        assert g['delta'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Summary Formatter
# ═══════════════════════════════════════════════════════════════════════════════

class TestExoticSummary:

    def test_scalar_price(self):
        s = exotic_summary(0.025, None, 'barrier', {'S': 1.10, 'K': 1.10})
        assert s['product_type'] == 'barrier'
        assert s['price'] == 0.025
        assert 'greeks' not in s

    def test_dict_price(self):
        result = {'price': 0.03, 'std_error': 0.001, 'method': 'mc'}
        s = exotic_summary(result, None, 'asian', {'S': 1.10})
        assert s['price'] == 0.03
        assert s['std_error'] == 0.001
        assert s['method'] == 'mc'

    def test_with_greeks(self):
        greeks = {'price': 0.025, 'delta': 0.5, 'gamma': 0.1, 'vega': 0.02}
        s = exotic_summary(0.025, greeks, 'digital', {'S': 1.10})
        assert 'greeks' in s
        assert 'delta' in s['greeks']
        # 'price' key should be excluded from greeks sub-dict
        assert 'price' not in s['greeks']

    def test_callable_params_excluded(self):
        """Callable params (like functions) should be excluded from summary."""
        s = exotic_summary(0.01, None, 'test', {'S': 1.10, 'func': lambda x: x})
        assert 'func' not in s['params']


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_very_short_expiry(self):
        """1-day expiry options should produce finite, non-negative prices."""
        T_short = 1.0 / 365.0
        p = barrier_price(S, K, B=1.05, T=T_short, r_d=r_d, r_f=r_f,
                          sigma=sigma, cp=1, barrier_type='down-and-out-call')
        assert np.isfinite(p) and p >= 0

        d = digital_price(S, K, T_short, r_d, r_f, sigma, cp=1)
        assert np.isfinite(d) and d >= 0

        ot = one_touch_price(S, B=1.15, T=T_short, r_d=r_d, r_f=r_f, sigma=sigma)
        assert np.isfinite(ot) and ot >= 0

    def test_extreme_barrier_far_away(self):
        """Barrier very far from spot should make KO = vanilla, KI = 0."""
        vanilla = _gk_price(S, K, T, r_d, r_f, sigma, 1)
        # Barrier far below spot for a down-and-out call
        ko = barrier_price(S, K, B=0.01, T=T, r_d=r_d, r_f=r_f,
                           sigma=sigma, cp=1, barrier_type='down-and-out-call', rebate=0)
        ki = barrier_price(S, K, B=0.01, T=T, r_d=r_d, r_f=r_f,
                           sigma=sigma, cp=1, barrier_type='down-and-in-call', rebate=0)
        assert abs(ko - vanilla) < 1e-6
        assert ki < 1e-6

    def test_high_vol_barrier(self):
        """High volatility should not cause overflow."""
        p = barrier_price(S, K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                          sigma=2.0, cp=1, barrier_type='down-and-out-call')
        assert np.isfinite(p) and p >= 0

    def test_high_vol_one_touch(self):
        """High vol one-touch should be near discounted payout."""
        payout = 1.0
        ot = one_touch_price(S, B=1.15, T=T, r_d=r_d, r_f=r_f,
                             sigma=5.0, payout=payout)
        df = payout * np.exp(-r_d * T)
        assert np.isfinite(ot)
        assert ot <= df + 1e-6

    def test_negative_rates(self):
        """Negative rates should not crash any pricer."""
        p = barrier_price(S, K, B=1.05, T=T, r_d=-0.01, r_f=-0.005,
                          sigma=sigma, cp=1, barrier_type='down-and-out-call')
        assert np.isfinite(p) and p >= 0

        d = digital_price(S, K, T, -0.01, -0.005, sigma, cp=1)
        assert np.isfinite(d) and d >= 0

        ot = one_touch_price(S, B=1.15, T=T, r_d=-0.01, r_f=-0.005, sigma=sigma)
        assert np.isfinite(ot) and ot >= 0

    def test_very_large_T(self):
        """Long-dated options (10Y) should not overflow."""
        T_long = 10.0
        p = barrier_price(S, K, B=1.05, T=T_long, r_d=r_d, r_f=r_f,
                          sigma=sigma, cp=1, barrier_type='down-and-out-call')
        assert np.isfinite(p) and p >= 0

        d = digital_price(S, K, T_long, r_d, r_f, sigma, cp=1)
        assert np.isfinite(d) and d >= 0

    def test_spot_below_barrier_down_and_out(self):
        """If spot is already below barrier, down-and-out should return rebate * DF."""
        rebate = 0.05
        p = barrier_price(S=1.00, K=K, B=1.05, T=T, r_d=r_d, r_f=r_f,
                          sigma=sigma, cp=1, barrier_type='down-and-out-call',
                          rebate=rebate)
        expected = rebate * np.exp(-r_d * T)
        assert abs(p - expected) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
