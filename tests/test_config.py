"""
Unit tests for core/config.py — verify centralized constants are consistent.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.config import (
    TENORS_FULL, TENORS_LIQUID, TENORS_TRADING, TENORS_SURFACE,
    TENORS_HEATMAP, TENORS_HEATMAP_COMPACT,
    DELTA_LABELS, DELTA_NUMERIC,
    VOL_METRICS, HISTORY_METRICS, METRIC_TO_KEY,
)


class TestTenorConsistency:

    def test_liquid_subset_of_full(self):
        for t in TENORS_LIQUID:
            assert t in TENORS_FULL, f"{t} in LIQUID but not in FULL"

    def test_trading_subset_of_full(self):
        for t in TENORS_TRADING:
            assert t in TENORS_FULL, f"{t} in TRADING but not in FULL"

    def test_surface_subset_of_full(self):
        for t in TENORS_SURFACE:
            assert t in TENORS_FULL, f"{t} in SURFACE but not in FULL"

    def test_heatmap_subset_of_liquid(self):
        for t in TENORS_HEATMAP:
            assert t in TENORS_LIQUID, f"{t} in HEATMAP but not in LIQUID"

    def test_compact_subset_of_heatmap(self):
        for t in TENORS_HEATMAP_COMPACT:
            assert t in TENORS_HEATMAP, f"{t} in COMPACT but not in HEATMAP"

    def test_no_duplicates(self):
        for name, lst in [
            ("FULL", TENORS_FULL), ("LIQUID", TENORS_LIQUID),
            ("TRADING", TENORS_TRADING), ("SURFACE", TENORS_SURFACE),
        ]:
            assert len(lst) == len(set(lst)), f"Duplicates in TENORS_{name}"


class TestDeltaConsistency:

    def test_labels_and_numeric_same_length(self):
        assert len(DELTA_LABELS) == len(DELTA_NUMERIC)

    def test_delta_numeric_sorted(self):
        assert DELTA_NUMERIC == sorted(DELTA_NUMERIC)

    def test_atm_in_center(self):
        assert "ATM" in DELTA_LABELS
        idx = DELTA_LABELS.index("ATM")
        assert DELTA_NUMERIC[idx] == 0.0


class TestMetrics:

    def test_metric_to_key_covers_metrics(self):
        for m in VOL_METRICS:
            assert m in METRIC_TO_KEY, f"{m} missing from METRIC_TO_KEY"

    def test_history_metrics_match_vol_metrics(self):
        assert set(HISTORY_METRICS) == set(VOL_METRICS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
