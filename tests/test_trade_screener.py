"""
Unit tests for core/trade_screener.py.

Covers:
  - Score helpers (`_score_from_pct`, `_score_from_z`)
  - Pair / tenor universe filters
  - Each screen with patched analytics + cache primitives
  - top_n_per_category ranking
  - scan_all returns a list (and respects cache)
  - Cross-validation: every CATEGORY_DEFAULTS entry references a real
    preset in panels.structure_builder.PRESETS and a real view in
    panels.vol_surface_fx.VIEW_PRESETS
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core import trade_screener as ts
from core.trade_screener import (
    TradeIdea, ALL_CATEGORIES, CATEGORY_DEFAULTS,
    _score_from_pct, _score_from_z,
    filter_pairs, filter_tenors,
    screen_cheap_vol, screen_rich_vol,
    screen_iv_rv_rich, screen_iv_rv_cheap,
    screen_skew_extreme, screen_wings_extreme,
    screen_term_inversion, screen_term_kink,
    screen_carry_per_vol, screen_vol_of_vol,
    screen_tail_underpriced,
    top_n_per_category, scan_all, clear_cache,
)


# ============================================================================
# Score helpers
# ============================================================================

class TestScoreFromPct:

    def test_extreme_low_scores_max(self):
        assert _score_from_pct(0.0) == pytest.approx(10.0)

    def test_extreme_high_scores_max(self):
        assert _score_from_pct(100.0) == pytest.approx(10.0)

    def test_at_threshold_scores_five(self):
        assert _score_from_pct(20.0) == pytest.approx(5.0)
        assert _score_from_pct(80.0) == pytest.approx(5.0)

    def test_midpoint_scores_zero(self):
        assert _score_from_pct(50.0) == pytest.approx(0.0)

    def test_none_returns_zero(self):
        assert _score_from_pct(None) == 0.0

    def test_nan_returns_zero(self):
        assert _score_from_pct(float("nan")) == 0.0


class TestScoreFromZ:

    def test_at_threshold_scores_five(self):
        assert _score_from_z(2.0, threshold=2.0) == pytest.approx(5.0)
        assert _score_from_z(-2.0, threshold=2.0) == pytest.approx(5.0)

    def test_double_threshold_scores_max(self):
        assert _score_from_z(4.0, threshold=2.0) == pytest.approx(10.0)

    def test_zero_scores_zero(self):
        assert _score_from_z(0.0) == 0.0

    def test_clamped_at_ten(self):
        assert _score_from_z(20.0, threshold=2.0) == pytest.approx(10.0)

    def test_none_returns_zero(self):
        assert _score_from_z(None) == 0.0


# ============================================================================
# Universe filters
# ============================================================================

class TestFilterPairs:

    def test_all_returns_full_universe(self):
        from core.fx_conventions import FX_PAIR_REGISTRY
        assert set(filter_pairs("ALL")) == set(FX_PAIR_REGISTRY.keys())

    def test_g10_excludes_em(self):
        g10 = filter_pairs("G10")
        from core.fx_conventions import FX_PAIR_REGISTRY
        for p in g10:
            assert FX_PAIR_REGISTRY[p].group == "G10"

    def test_em_only(self):
        em = filter_pairs("EM")
        from core.fx_conventions import FX_PAIR_REGISTRY
        for p in em:
            assert FX_PAIR_REGISTRY[p].group != "G10"

    def test_unknown_falls_back_to_all(self):
        assert len(filter_pairs("UNKNOWN")) > 0


class TestFilterTenors:

    def test_front_belly_back(self):
        assert "1W" in filter_tenors("FRONT")
        assert "1M" in filter_tenors("BELLY")
        assert "1Y" in filter_tenors("BACK")

    def test_all_includes_full_set(self):
        ts_all = filter_tenors("ALL")
        for t in ("1W", "1M", "3M", "6M", "1Y"):
            assert t in ts_all


# ============================================================================
# Cross-validation: CATEGORY_DEFAULTS references real presets / views
# ============================================================================

class TestCategoryDefaults:
    """Each (preset, view) pair must exist in the destination panels."""

    def test_all_presets_exist_in_structure_builder(self):
        from panels.structure_builder import PRESETS
        for cat, (preset, _view) in CATEGORY_DEFAULTS.items():
            assert preset in PRESETS, (
                f"Category {cat!r} maps to preset {preset!r} which is "
                f"not in panels.structure_builder.PRESETS"
            )

    def test_all_views_exist_in_vol_surface(self):
        from panels.vol_surface_fx import VIEW_PRESETS
        for cat, (_preset, view) in CATEGORY_DEFAULTS.items():
            assert view in VIEW_PRESETS, (
                f"Category {cat!r} maps to vol_view {view!r} which is "
                f"not in panels.vol_surface_fx.VIEW_PRESETS"
            )

    def test_all_categories_in_alllist(self):
        for cat in CATEGORY_DEFAULTS:
            assert cat in ALL_CATEGORIES


# ============================================================================
# Individual screens (with mocked analytics)
# ============================================================================

class TestScreenCheapVol:

    def test_emits_when_below_threshold(self):
        def fake(pair, tenor, metric="ATM", **_kw):
            return {"percentile": 10.0, "current": 5.5, "mean": 8.0}
        with patch("core.trade_screener.vol_percentile", side_effect=fake):
            ideas = screen_cheap_vol(["EURUSD"], ["3M"])
        assert len(ideas) == 1
        idea = ideas[0]
        assert idea.category == "Cheap Vol"
        assert idea.direction == "long_vol"
        assert idea.score > 5.0
        assert idea.structure_preset == "Straddle"
        assert idea.vol_view == "Surface"

    def test_skips_when_above_threshold(self):
        def fake(pair, tenor, metric="ATM", **_kw):
            return {"percentile": 50.0, "current": 8.0, "mean": 8.0}
        with patch("core.trade_screener.vol_percentile", side_effect=fake):
            ideas = screen_cheap_vol(["EURUSD"], ["3M"])
        assert ideas == []

    def test_skips_when_no_data(self):
        with patch("core.trade_screener.vol_percentile", return_value=None):
            ideas = screen_cheap_vol(["EURUSD"], ["3M"])
        assert ideas == []


class TestScreenRichVol:

    def test_emits_when_above_threshold(self):
        def fake(pair, tenor, metric="ATM", **_kw):
            return {"percentile": 92.0, "current": 14.0, "mean": 9.0}
        with patch("core.trade_screener.vol_percentile", side_effect=fake):
            ideas = screen_rich_vol(["EURUSD"], ["3M"])
        assert len(ideas) == 1
        assert ideas[0].direction == "short_vol"
        assert ideas[0].score > 5.0


class TestScreenIvRvRich:

    def test_emits_when_iv_rich(self):
        def fake(pair, tenor, **_kw):
            return {"percentile": 90.0, "current_iv": 12.0,
                    "current_rv": 7.0, "current_spread": 5.0}
        with patch("core.trade_screener.iv_rv_percentile", side_effect=fake):
            ideas = screen_iv_rv_rich(["EURUSD"], ["1M", "3M"])
        assert len(ideas) == 2   # 1M and 3M (belly tenors)
        assert all(i.direction == "short_vol" for i in ideas)

    def test_ignores_non_belly_tenors(self):
        def fake(pair, tenor, **_kw):
            return {"percentile": 90.0, "current_iv": 12.0,
                    "current_rv": 7.0, "current_spread": 5.0}
        with patch("core.trade_screener.iv_rv_percentile", side_effect=fake):
            ideas = screen_iv_rv_rich(["EURUSD"], ["1W", "1Y"])
        assert ideas == []


class TestScreenIvRvCheap:

    def test_emits_when_iv_cheap(self):
        def fake(pair, tenor, **_kw):
            return {"percentile": 10.0, "current_iv": 6.0,
                    "current_rv": 9.0, "current_spread": -3.0}
        with patch("core.trade_screener.iv_rv_percentile", side_effect=fake):
            ideas = screen_iv_rv_cheap(["EURUSD"], ["3M"])
        assert len(ideas) == 1
        assert ideas[0].direction == "long_vol"


class TestScreenSkewExtreme:

    def test_emits_call_skew_rich(self):
        def fake(pair, tenor, metric="25D_RR", **_kw):
            return {"zscore": 2.5, "current": 1.8, "mean": 0.0}
        with patch("core.trade_screener.vol_zscore", side_effect=fake):
            ideas = screen_skew_extreme(["EURUSD"], ["3M"])
        assert len(ideas) == 1
        assert ideas[0].category == "Skew Rich (Calls)"
        assert ideas[0].direction == "short_skew"

    def test_emits_put_skew_rich(self):
        def fake(pair, tenor, metric="25D_RR", **_kw):
            return {"zscore": -2.5, "current": -1.8, "mean": 0.0}
        with patch("core.trade_screener.vol_zscore", side_effect=fake):
            ideas = screen_skew_extreme(["EURUSD"], ["3M"])
        assert len(ideas) == 1
        assert ideas[0].category == "Skew Rich (Puts)"
        assert ideas[0].direction == "long_skew"

    def test_skips_within_threshold(self):
        def fake(pair, tenor, metric="25D_RR", **_kw):
            return {"zscore": 1.0, "current": 0.5, "mean": 0.0}
        with patch("core.trade_screener.vol_zscore", side_effect=fake):
            assert screen_skew_extreme(["EURUSD"], ["3M"]) == []


class TestScreenWingsExtreme:

    def test_wings_rich(self):
        def fake(pair, tenor, metric="25D_BF", **_kw):
            return {"zscore": 2.3, "current": 0.8, "mean": 0.3}
        with patch("core.trade_screener.vol_zscore", side_effect=fake):
            ideas = screen_wings_extreme(["EURUSD"], ["3M"])
        assert len(ideas) == 1
        assert ideas[0].category == "Wings Rich"

    def test_wings_cheap(self):
        def fake(pair, tenor, metric="25D_BF", **_kw):
            return {"zscore": -2.3, "current": 0.1, "mean": 0.5}
        with patch("core.trade_screener.vol_zscore", side_effect=fake):
            ideas = screen_wings_extreme(["EURUSD"], ["3M"])
        assert len(ideas) == 1
        assert ideas[0].category == "Wings Cheap"
        assert ideas[0].direction == "tail"


class TestScreenTermInversion:

    def test_emits_when_inverted(self):
        # 1M ATM 12, 6M ATM 9 → 3pt inversion
        surf = {"1M": {"atm": 12.0}, "6M": {"atm": 9.0}}
        with patch("core.trade_screener.get_fx_vol_surface", return_value=surf):
            ideas = screen_term_inversion(["EURUSD"])
        assert len(ideas) == 1
        assert ideas[0].direction == "calendar"
        assert ideas[0].metrics["inversion"] == pytest.approx(3.0)

    def test_skips_when_normal_term(self):
        surf = {"1M": {"atm": 8.0}, "6M": {"atm": 9.0}}
        with patch("core.trade_screener.get_fx_vol_surface", return_value=surf):
            assert screen_term_inversion(["EURUSD"]) == []

    def test_skips_when_no_surface(self):
        with patch("core.trade_screener.get_fx_vol_surface", return_value=None):
            assert screen_term_inversion(["EURUSD"]) == []


class TestScreenTermKink:

    def test_emits_when_3m_above_interp(self):
        # Linear interp at 3M from 1M=8 to 6M=10 is 8.8
        # Set 3M to 9.5 → kink of 0.7
        surf = {"1M": {"atm": 8.0}, "3M": {"atm": 9.5}, "6M": {"atm": 10.0}}
        with patch("core.trade_screener.get_fx_vol_surface", return_value=surf):
            ideas = screen_term_kink(["EURUSD"])
        assert len(ideas) == 1
        assert ideas[0].metrics["kink"] > 0

    def test_skips_when_no_kink(self):
        surf = {"1M": {"atm": 8.0}, "3M": {"atm": 8.8}, "6M": {"atm": 10.0}}
        with patch("core.trade_screener.get_fx_vol_surface", return_value=surf):
            assert screen_term_kink(["EURUSD"]) == []


class TestScreenCarryPerVol:

    def test_emits_when_sharpe_high(self):
        df = pd.DataFrame([
            {"pair": "AUDUSD", "carry_bps": 150, "atm_3m": 8.0,
             "carry_per_vol": 0.18, "sharpe_proxy": 0.6,
             "rank_signal": "ATTRACTIVE"},
        ])
        with patch("core.trade_screener.carry_per_vol", return_value=df):
            ideas = screen_carry_per_vol(["AUDUSD"])
        assert len(ideas) == 1
        assert ideas[0].direction == "carry"

    def test_skips_when_sharpe_low(self):
        df = pd.DataFrame([
            {"pair": "AUDUSD", "carry_bps": 50, "atm_3m": 8.0,
             "carry_per_vol": 0.06, "sharpe_proxy": 0.2,
             "rank_signal": "MODERATE"},
        ])
        with patch("core.trade_screener.carry_per_vol", return_value=df):
            assert screen_carry_per_vol(["AUDUSD"]) == []


class TestScreenVolOfVol:

    def test_emits_when_z_extreme(self):
        # Construct a vov series with last value at high z
        rng = np.random.default_rng(0)
        baseline = rng.normal(0.5, 0.05, 100)
        spike = np.array([1.0])    # ~10 sigma above baseline
        vov_series = np.concatenate([baseline, spike])
        df = pd.DataFrame({"vov_60d": vov_series})
        with patch("core.trade_screener.vol_of_vol", return_value=df):
            ideas = screen_vol_of_vol(["EURUSD"], ["3M"])
        assert len(ideas) == 1
        assert ideas[0].metrics["zscore"] > 1.5

    def test_skips_when_short_history(self):
        df = pd.DataFrame({"vov_60d": [0.1, 0.2, 0.3]})
        with patch("core.trade_screener.vol_of_vol", return_value=df):
            assert screen_vol_of_vol(["EURUSD"], ["3M"]) == []

    def test_skips_when_empty(self):
        with patch("core.trade_screener.vol_of_vol", return_value=pd.DataFrame()):
            assert screen_vol_of_vol(["EURUSD"], ["3M"]) == []


class TestScreenTailUnderpriced:

    def test_emits_when_10d_bf_z_low(self):
        def fake(pair, tenor, metric="10D_BF", **_kw):
            return {"zscore": -2.0, "current": 0.05, "mean": 0.4}
        with patch("core.trade_screener.vol_zscore", side_effect=fake):
            ideas = screen_tail_underpriced(["EURUSD"], ["3M"])
        assert len(ideas) == 1
        assert ideas[0].category == "Tail Underpriced"
        assert ideas[0].direction == "tail"

    def test_skips_when_z_above_negative_threshold(self):
        def fake(pair, tenor, metric="10D_BF", **_kw):
            return {"zscore": 0.5, "current": 0.5, "mean": 0.4}
        with patch("core.trade_screener.vol_zscore", side_effect=fake):
            assert screen_tail_underpriced(["EURUSD"], ["3M"]) == []


# ============================================================================
# Top-N ranking
# ============================================================================

class TestTopNPerCategory:

    def _ideas(self, scores_per_cat):
        out = []
        for cat, scores in scores_per_cat.items():
            preset, view = CATEGORY_DEFAULTS[cat]
            for s in scores:
                out.append(TradeIdea(
                    pair="EURUSD", tenor="3M", category=cat,
                    direction="long_vol", score=s,
                    structure_preset=preset, vol_view=view,
                    thesis="test",
                ))
        return out

    def test_takes_top_n_per_category(self):
        ideas = self._ideas({
            "Cheap Vol": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
            "Rich Vol":  [9.5, 8.5, 7.5, 6.5, 5.5],
        })
        out = top_n_per_category(ideas, n=3)
        # 3 per category × 2 categories = 6
        assert len(out) == 6
        cheap = [i for i in out if i.category == "Cheap Vol"]
        assert [i.score for i in cheap] == [9.0, 8.0, 7.0]

    def test_respects_category_order(self):
        ideas = self._ideas({
            "Rich Vol":  [9.5],
            "Cheap Vol": [9.0],
        })
        # Rich Vol comes after Cheap Vol in ALL_CATEGORIES, so Cheap appears first
        out = top_n_per_category(ideas, n=5)
        assert out[0].category == "Cheap Vol"


# ============================================================================
# Top-level scan
# ============================================================================

class TestScanAll:

    def test_returns_list(self):
        # Patch every screen to return [], so scan_all wires together cleanly
        clear_cache()
        with patch.object(ts, "screen_cheap_vol",        return_value=[]), \
             patch.object(ts, "screen_rich_vol",         return_value=[]), \
             patch.object(ts, "screen_iv_rv_rich",       return_value=[]), \
             patch.object(ts, "screen_iv_rv_cheap",      return_value=[]), \
             patch.object(ts, "screen_skew_extreme",     return_value=[]), \
             patch.object(ts, "screen_wings_extreme",    return_value=[]), \
             patch.object(ts, "screen_term_inversion",   return_value=[]), \
             patch.object(ts, "screen_term_kink",        return_value=[]), \
             patch.object(ts, "screen_carry_per_vol",    return_value=[]), \
             patch.object(ts, "screen_vol_of_vol",       return_value=[]), \
             patch.object(ts, "screen_tail_underpriced", return_value=[]):
            result = scan_all(region="ALL", tenor_bucket="ALL")
        assert isinstance(result, list)

    def test_swallows_screen_exceptions(self):
        clear_cache()
        # Use a real function (not Mock) so the logger.exception()
        # call inside scan_all can read fn.__name__.
        def _boom(*_a, **_kw):
            raise RuntimeError("boom")
        with patch.object(ts, "screen_cheap_vol",        _boom), \
             patch.object(ts, "screen_rich_vol",         return_value=[]), \
             patch.object(ts, "screen_iv_rv_rich",       return_value=[]), \
             patch.object(ts, "screen_iv_rv_cheap",      return_value=[]), \
             patch.object(ts, "screen_skew_extreme",     return_value=[]), \
             patch.object(ts, "screen_wings_extreme",    return_value=[]), \
             patch.object(ts, "screen_term_inversion",   return_value=[]), \
             patch.object(ts, "screen_term_kink",        return_value=[]), \
             patch.object(ts, "screen_carry_per_vol",    return_value=[]), \
             patch.object(ts, "screen_vol_of_vol",       return_value=[]), \
             patch.object(ts, "screen_tail_underpriced", return_value=[]):
            # Should not raise
            result = scan_all()
        assert isinstance(result, list)

    def test_cache_returns_same_object(self):
        clear_cache()
        with patch.object(ts, "screen_cheap_vol",        return_value=[]), \
             patch.object(ts, "screen_rich_vol",         return_value=[]), \
             patch.object(ts, "screen_iv_rv_rich",       return_value=[]), \
             patch.object(ts, "screen_iv_rv_cheap",      return_value=[]), \
             patch.object(ts, "screen_skew_extreme",     return_value=[]), \
             patch.object(ts, "screen_wings_extreme",    return_value=[]), \
             patch.object(ts, "screen_term_inversion",   return_value=[]), \
             patch.object(ts, "screen_term_kink",        return_value=[]), \
             patch.object(ts, "screen_carry_per_vol",    return_value=[]), \
             patch.object(ts, "screen_vol_of_vol",       return_value=[]), \
             patch.object(ts, "screen_tail_underpriced", return_value=[]):
            a = scan_all(region="G10", tenor_bucket="BELLY")
            b = scan_all(region="G10", tenor_bucket="BELLY")
        assert a is b


# ============================================================================
# TradeIdea dataclass
# ============================================================================

class TestTradeIdea:

    def test_to_row_omits_metrics(self):
        idea = TradeIdea(
            pair="EURUSD", tenor="3M", category="Cheap Vol",
            direction="long_vol", score=8.5,
            structure_preset="Straddle", vol_view="Surface",
            thesis="test",
            metrics={"percentile": 10},
        )
        row = idea.to_row()
        assert "metrics" not in row
        assert row["score"] == 8.5
        assert row["pair"] == "EURUSD"
