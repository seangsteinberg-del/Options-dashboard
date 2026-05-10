"""
Regression tests for multi-tenor (calendar/diagonal) handling in
panels.structure_builder. Before the fix, the payoff chart and aggregates
collapsed all legs to a single expiry T, producing monotonically-decreasing
'always-loses' P&L for calendars. These tests pin down the corrected
behaviour: at the FRONT leg's expiry, longer-dated legs still carry
remaining time value.
"""
import numpy as np
import pytest

from panels.structure_builder import (
    _build_payoff_chart, _compute_aggregates, _compute_expected_value,
    _build_pnl_heatmap, _build_scenario_table,
)


def _calendar_legs(strike=1.10, T_back=0.5, T_front=0.0833, vol=0.10):
    """Long calendar: buy back-month call, sell front-month call, same strike."""
    return [
        {"cp_sign": 1, "strike": strike, "vol": vol, "side_sign": 1,
         "ratio": 1, "T": T_back, "price_unit": 0.040,
         "premium_total": 40000, "premium_pips": 400.0,
         "delta": 0.5, "gamma": 0.01, "vega": 100, "theta": -0.5,
         "vanna": 0, "volga": 0},
        {"cp_sign": 1, "strike": strike, "vol": vol, "side_sign": -1,
         "ratio": 1, "T": T_front, "price_unit": 0.020,
         "premium_total": -20000, "premium_pips": -200.0,
         "delta": -0.5, "gamma": -0.02, "vega": -50, "theta": 1.0,
         "vanna": 0, "volga": 0},
    ]


def _strangle_legs(spot=1.0, vol=0.10, T=0.5):
    """Long strangle: same expiry on both legs."""
    return [
        {"cp_sign": -1, "strike": 0.95, "vol": vol, "side_sign": 1,
         "ratio": 1, "T": T, "price_unit": 0.01,
         "premium_total": 10000, "premium_pips": 100.0,
         "delta": -0.1, "gamma": 0.05, "vega": 30, "theta": -0.05,
         "vanna": 0, "volga": 0},
        {"cp_sign": 1, "strike": 1.05, "vol": vol, "side_sign": 1,
         "ratio": 1, "T": T, "price_unit": 0.01,
         "premium_total": 10000, "premium_pips": 100.0,
         "delta": 0.1, "gamma": 0.05, "vega": 30, "theta": -0.05,
         "vanna": 0, "volga": 0},
    ]


class TestPayoffChartMultiTenor:
    def test_calendar_payoff_is_tent_shaped_not_monotonic(self):
        """Calendar at front expiry should peak near the strike — not be a
        monotonically-decreasing line ending in the loss of total premium."""
        legs = _calendar_legs()
        fig = _build_payoff_chart(legs, {"breakevens": []}, S=1.10, T=0.0833,
                                  r_d=0.05, r_f=0.03, notional=1_000_000,
                                  atm_vol=0.10)
        expiry_trace = next(t for t in fig.data if "Expiry" in t.name)
        y = np.array(expiry_trace.y)

        is_monotone_decr = all(y[i] >= y[i + 1] for i in range(len(y) - 1))
        assert not is_monotone_decr, "Calendar payoff regressed to monotonic loss"
        argmax = int(np.argmax(y))
        # Peak should sit in the middle 1/3 of the spot range (near the strike)
        assert len(y) // 3 < argmax < 2 * len(y) // 3

    def test_calendar_label_says_front_expiry(self):
        legs = _calendar_legs()
        fig = _build_payoff_chart(legs, {"breakevens": []}, S=1.10, T=0.0833,
                                  r_d=0.05, r_f=0.03, notional=1_000_000,
                                  atm_vol=0.10)
        names = {t.name for t in fig.data}
        assert "At Front Expiry" in names
        assert "At Expiry" not in names  # plain label is single-tenor only

    def test_strangle_payoff_unchanged_label(self):
        """Single-expiry payoff still uses the plain 'At Expiry' label."""
        legs = _strangle_legs()
        fig = _build_payoff_chart(legs, {"breakevens": []}, S=1.0, T=0.5,
                                  r_d=0.05, r_f=0.03, notional=1_000_000,
                                  atm_vol=0.10)
        names = {t.name for t in fig.data}
        assert "At Expiry" in names
        assert "At Front Expiry" not in names

    def test_strangle_payoff_remains_u_shape(self):
        legs = _strangle_legs()
        fig = _build_payoff_chart(legs, {"breakevens": []}, S=1.0, T=0.5,
                                  r_d=0.05, r_f=0.03, notional=1_000_000,
                                  atm_vol=0.10)
        y = np.array(next(t for t in fig.data if "Expiry" in t.name).y)
        argmin = int(np.argmin(y))
        # U-shape: minimum sits in middle 1/3
        assert len(y) // 3 < argmin < 2 * len(y) // 3
        # Min equals total premium paid (-$20k)
        assert y[argmin] == pytest.approx(-20_000, rel=0.05)


class TestAggregatesMultiTenor:
    def test_calendar_max_loss_bounded(self):
        """Calendar max loss should not approach the full notional."""
        legs = _calendar_legs()
        agg = _compute_aggregates(legs, S=1.10, T=0.0833, r_d=0.05,
                                  r_f=0.03, notional=1_000_000,
                                  pip_size=0.0001)
        # Max loss should be finite and small relative to notional
        assert np.isfinite(agg["max_loss"])
        assert agg["max_loss"] > -100_000
        # Max profit should be POSITIVE (calendar peaks at strike)
        assert agg["max_profit"] > 0
        # POP should be a sensible fraction
        assert 0 < agg["pop"] < 100

    def test_calendar_has_two_breakevens(self):
        """Tent-shaped payoff crosses zero on both sides of the peak."""
        legs = _calendar_legs()
        agg = _compute_aggregates(legs, S=1.10, T=0.0833, r_d=0.05,
                                  r_f=0.03, notional=1_000_000,
                                  pip_size=0.0001)
        assert len(agg["breakevens"]) == 2
        be_lo, be_hi = sorted(agg["breakevens"])
        assert be_lo < 1.10 < be_hi  # bracket the strike


class TestExpectedValueMultiTenor:
    def test_calendar_ev_finite(self):
        legs = _calendar_legs()
        ev = _compute_expected_value(
            legs, S=1.10, T=0.0833, r_d=0.05, r_f=0.03,
            pair="EURUSD", tenor="1M", notional=1_000_000,
            pip_size=0.0001, vol_surface={},
        )
        # EV may be positive or negative but should be finite, not catastrophic
        assert ev is not None
        assert np.isfinite(ev["ev"])
        assert abs(ev["ev"]) < 100_000  # premium scale, not full notional


class TestHeatmapMultiTenor:
    def test_calendar_heatmap_uses_per_leg_T(self):
        """Heatmap MTM grid is non-degenerate for calendar (was previously
        showing the same underpriced loss for all back-leg cells)."""
        legs = _calendar_legs()
        fig = _build_pnl_heatmap(legs, S=1.10, T=0.0833, r_d=0.05, r_f=0.03,
                                 notional=1_000_000, atm_vol=0.10)
        z = np.array(fig.data[0].z)
        # The heatmap should have variation across both spot AND vol dims
        assert z.max() > z.min()
        # Vol-shock direction should produce variation (back-leg vega)
        col_mid = z[:, z.shape[1] // 2]
        assert col_mid.max() - col_mid.min() > 0


class TestScenarioTableMultiTenor:
    def test_calendar_scenario_pnl_finite(self):
        legs = _calendar_legs()
        rows = _build_scenario_table(legs, S=1.10, T=0.0833, r_d=0.05,
                                     r_f=0.03, notional=1_000_000)
        for r in rows:
            assert np.isfinite(r["pnl"])
        # Scenario at 0% spot move should be near max profit (peak at strike)
        # — i.e., better than -10% / +10% scenarios
        center_pnl = next(r["pnl"] for r in rows if r["shock"] == 0)
        edge_pnl = next(r["pnl"] for r in rows if r["shock"] == 0.10)
        assert center_pnl > edge_pnl
