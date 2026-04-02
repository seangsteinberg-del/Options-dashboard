"""
Centralized Configuration Constants
====================================
All shared constants live here. Panels and core modules import from this
file instead of defining their own copies.
"""

# ── Tenor Lists ──────────────────────────────────────────────────────────────
# Full tenor ladder (all available tenors, ON → 5Y)
TENORS_FULL = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "5Y"]

# Liquid tenors (most commonly used in vol surfaces and analytics)
TENORS_LIQUID = ["1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]

# Trading tenors (structure builder, blotter)
TENORS_TRADING = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]

# Vol surface tenors (includes ON, 1W for term structure)
TENORS_SURFACE = ["ON", "1W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "5Y"]

# Heatmap tenors (compact grid view)
TENORS_HEATMAP = ["1M", "2M", "3M", "6M", "1Y", "2Y"]

# Heatmap tenors (relative value — even more compact)
TENORS_HEATMAP_COMPACT = ["1M", "3M", "6M", "1Y"]


# ── Delta Grid ───────────────────────────────────────────────────────────────
DELTA_LABELS = ["10P", "25P", "ATM", "25C", "10C"]
DELTA_NUMERIC = [-0.50, -0.25, 0.0, 0.25, 0.50]


# ── Vol Metrics ──────────────────────────────────────────────────────────────
VOL_METRICS = ["ATM", "25D_RR", "25D_BF", "10D_RR", "10D_BF"]


# ── History Mode Metrics ─────────────────────────────────────────────────────
HISTORY_METRICS = ["ATM", "25D_RR", "25D_BF", "10D_RR", "10D_BF"]
METRIC_TO_KEY = {
    "ATM": "atm", "25D_RR": "rr25", "25D_BF": "bf25",
    "10D_RR": "rr10", "10D_BF": "bf10",
}


# ── Default Lookback Windows ────────────────────────────────────────────────
RV_WINDOWS = [5, 10, 20, 60, 90, 120, 252]
PERCENTILE_LOOKBACK = 252
HISTORY_LOOKBACK_DEFAULT = 800


# ── Monospace Font ───────────────────────────────────────────────────────────
MONO_FONT = "'JetBrains Mono', monospace"
