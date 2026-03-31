"""
Strategy Backtester Panel
=========================
Historical backtest engine for FX option strategies with Garman-Kohlhagen
pricing, regime-aware entry signals, configurable exit rules, and
comprehensive performance analytics.

Supports straddles, strangles, risk reversals, butterflies, and calendar
spreads with mark-to-market tracking and full trade logging.
"""

import dash
from dash import html, dcc, Input, Output, State, no_update, dash_table, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from core.theme import (
    COLORS, CARD_STYLE, CHART_TEMPLATE, STAT_BOX_STYLE,
    LABEL_STYLE, DROPDOWN_STYLE, INPUT_STYLE, BUTTON_STYLE,
    CARD_HEADER_STYLE, TABLE_HEADER_STYLE, TABLE_CELL_STYLE,
    make_stat_style, chart_layout, CSV_BTN_STYLE, no_data_fig,
)
from core.csv_export import export_csv
from core.bloomberg_fx import (
    get_fx_vol_surface, get_fx_spots, get_fx_rates, get_all_pairs,
    get_fx_historical_spot, get_fx_historical_vol,
)
from core.fx_conventions import (
    FX_PAIR_REGISTRY, tenor_to_years, tenor_to_days,
)
from core.fx_analytics import vol_regime_detect, vol_zscore, vol_percentile


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_STRATEGIES = [
    "Long Straddle", "Short Straddle",
    "Long Strangle", "Short Strangle",
    "Long RR (buy call sell put)", "Short RR",
    "Long BF", "Short BF",
    "Long Calendar (buy far sell near)", "Short Calendar",
]

_PAIRS = sorted(FX_PAIR_REGISTRY.keys())

_TENORS = [
    {"label": "1M", "value": "1M"},
    {"label": "3M", "value": "3M"},
    {"label": "6M", "value": "6M"},
]

_DELTAS = [
    {"label": "25-delta", "value": 0.25},
    {"label": "10-delta", "value": 0.10},
]

_LOOKBACKS = [
    {"label": "1 Year",  "value": 1},
    {"label": "2 Years", "value": 2},
    {"label": "3 Years", "value": 3},
    {"label": "5 Years", "value": 5},
]

_ENTRY_SIGNALS = [
    {"label": "Fixed interval (every month)",  "value": "fixed"},
    {"label": "Vol cheap (ATM %ile < 20)",     "value": "vol_cheap"},
    {"label": "Vol rich (ATM %ile > 80)",      "value": "vol_rich"},
    {"label": "Skew extreme (|RR z| > 1.5)",   "value": "skew_extreme"},
    {"label": "Vol momentum (RV accel > 20%)", "value": "vol_momentum"},
]

_EXIT_RULES = [
    {"label": "Hold to expiry",           "value": "hold"},
    {"label": "Exit at +50% profit",      "value": "tp50"},
    {"label": "Exit at -30% stop loss",   "value": "sl30"},
    {"label": "Exit at +100% / -50%",     "value": "tp100_sl50"},
    {"label": "Trailing stop 20%",        "value": "trailing_20"},
]

_REGIME_LABELS = ["LOW", "NORMAL", "ELEVATED", "HIGH", "CRISIS"]
_REGIME_COLORS = {
    "LOW": COLORS["accent_green"],
    "NORMAL": COLORS["accent_blue"],
    "ELEVATED": COLORS["accent_orange"],
    "HIGH": COLORS["accent_red"],
    "CRISIS": COLORS["accent_purple"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Garman-Kohlhagen Pricing Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _gk_price(S, K, T, r_d, r_f, sigma, cp):
    """Garman-Kohlhagen vanilla price.  cp = +1 call, -1 put."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(cp * (S - K), 0.0)
    from scipy.stats import norm as _norm
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(
        cp * (S * np.exp(-r_f * T) * _norm.cdf(cp * d1)
              - K * np.exp(-r_d * T) * _norm.cdf(cp * d2))
    )


def _gk_delta(S, K, T, r_d, r_f, sigma, cp):
    """GK spot delta for strike computation."""
    if T <= 0 or sigma <= 0:
        return float(cp) if cp * (S - K) > 0 else 0.0
    from scipy.stats import norm as _norm
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(cp * np.exp(-r_f * T) * _norm.cdf(cp * d1))


def _strike_from_delta(target_delta, S, T, r_d, r_f, sigma, cp):
    """Invert delta to find strike via Newton-Raphson."""
    from scipy.stats import norm as _norm
    if T <= 1e-8 or sigma <= 1e-8 or S <= 0:
        return S  # fallback to ATM
    F = S * np.exp((r_d - r_f) * T)
    # Initial guess from simplified Black-Scholes inversion
    ppf_arg = np.clip(abs(target_delta) * np.exp(r_f * T), 1e-8, 1 - 1e-8)
    K = F * np.exp(-cp * _norm.ppf(ppf_arg) * sigma * np.sqrt(T)
                   + 0.5 * sigma ** 2 * T)
    sqrtT = np.sqrt(T)
    for _ in range(50):
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        calc = cp * np.exp(-r_f * T) * _norm.cdf(cp * d1)
        dd = -np.exp(-r_f * T) * _norm.pdf(cp * d1) / (K * sigma * sqrtT)
        if abs(dd) < 1e-15:
            break
        K -= (calc - target_delta) / dd
        K = max(K, 1e-10)
        if abs(calc - target_delta) < 1e-10:
            break
    return float(K)


def _atm_strike(S, T, r_d, r_f, sigma):
    """ATM DNS strike."""
    F = S * np.exp((r_d - r_f) * T)
    return F * np.exp(0.5 * sigma ** 2 * T)


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Historical Data Generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_backtest_data(pair, lookback_years):
    """
    Build spot + vol histories for the backtest period from Bloomberg data.
    Returns a DataFrame indexed by date with columns:
        spot, atm_vol, rr25, bf25, r_d, r_f
    Returns None when Bloomberg data is insufficient.
    """
    n_days = int(lookback_years * 252)

    # --- Spot path from Bloomberg ---
    spots = get_fx_historical_spot(pair, days=n_days)
    if spots is None or spots.empty or len(spots) < n_days * 0.5:
        return None
    col = "close" if "close" in spots.columns else ("Close" if "Close" in spots.columns else (spots.columns[-1] if len(spots.columns) > 0 else None))
    if col is None:
        return None
    spot_series = spots[col].values[-n_days:]
    # If Bloomberg returned fewer days than requested, just use what we have
    actual_days = len(spot_series)

    # --- Vol history from Bloomberg ---
    vol_raw = get_fx_historical_vol(pair, "1M", "ATM", actual_days)
    if vol_raw is not None:
        if isinstance(vol_raw, dict):
            vol_raw = list(vol_raw.values()) if vol_raw else None
    if vol_raw is None or len(vol_raw) < actual_days * 0.5:
        return None
    vol_series = np.array(vol_raw, dtype=float)[-actual_days:]

    # Ensure spot and vol are the same length
    common_len = min(len(spot_series), len(vol_series))
    spot_series = spot_series[-common_len:]
    vol_series = vol_series[-common_len:]

    # RR25 and BF25: fetch from Bloomberg historical vol (required)
    rr25_raw = get_fx_historical_vol(pair, "1M", "25D_RR", actual_days)
    bf25_raw = get_fx_historical_vol(pair, "1M", "25D_BF", actual_days)
    if rr25_raw is not None and len(rr25_raw) >= common_len // 2:
        rr25_arr = np.array(rr25_raw, dtype=float)[-common_len:]
        if len(rr25_arr) < common_len:
            rr25_arr = np.pad(rr25_arr, (common_len - len(rr25_arr), 0), mode='edge')
        rr25_series = np.nan_to_num(rr25_arr, nan=0.0)
    else:
        # No Bloomberg RR data — use flat skew (0)
        rr25_series = np.zeros(common_len)
    if bf25_raw is not None and len(bf25_raw) >= common_len // 2:
        bf25_arr = np.array(bf25_raw, dtype=float)[-common_len:]
        if len(bf25_arr) < common_len:
            bf25_arr = np.pad(bf25_arr, (common_len - len(bf25_arr), 0), mode='edge')
        bf25_series = np.nan_to_num(bf25_arr, nan=0.25)
    else:
        # No Bloomberg BF data — use flat butterfly (0.25)
        bf25_series = np.full(common_len, 0.25)

    # Rates
    rates_data = get_fx_rates(pair) or {}
    r_d = rates_data.get("r_dom", 0.04)
    r_f = rates_data.get("r_for", 0.03)
    rd_series = np.full(common_len, r_d)
    rf_series = np.full(common_len, r_f)

    # Build date index
    end_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_date, periods=common_len)

    df = pd.DataFrame({
        "spot": spot_series,
        "atm_vol": vol_series,
        "rr25": rr25_series,
        "bf25": bf25_series,
        "r_d": rd_series,
        "r_f": rf_series,
    }, index=dates[-common_len:])

    # Compute realised vol (short and long windows) for regime detection
    log_ret = np.log(df["spot"] / df["spot"].shift(1))
    df["rv_20d"] = log_ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100
    df["rv_60d"] = log_ret.rolling(60, min_periods=30).std() * np.sqrt(252) * 100
    df["rv_20d"] = df["rv_20d"].ffill().bfill()
    df["rv_60d"] = df["rv_60d"].ffill().bfill()

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Backtest Engine
# ═══════════════════════════════════════════════════════════════════════════

def _compute_regime(atm_vol, rv_short=None, rv_long=None):
    """Classify vol regime using ATM level + RV trend (matching fx_analytics.vol_regime_detect logic).

    Parameters
    ----------
    atm_vol : float — current ATM implied vol (annualised %)
    rv_short : float | None — short-window (20d) realised vol
    rv_long : float | None — long-window (60d) realised vol

    Returns regime string. When RV data is available, the regime can be
    upgraded/downgraded based on the short/long RV ratio:
      ratio > 1.25 → vol accelerating → upgrade one level
      ratio < 0.75 → vol decelerating → downgrade one level
    """
    # Base regime from ATM IV level (FX-calibrated thresholds)
    levels = ["LOW", "NORMAL", "ELEVATED", "HIGH", "CRISIS"]
    if atm_vol > 20.0:
        base_idx = 4  # CRISIS
    elif atm_vol > 14.0:
        base_idx = 3  # HIGH
    elif atm_vol > 10.0:
        base_idx = 2  # ELEVATED
    elif atm_vol > 6.0:
        base_idx = 1  # NORMAL
    else:
        base_idx = 0  # LOW

    # Adjust based on RV trend if available
    if rv_short is not None and rv_long is not None and rv_long > 1e-6:
        ratio = rv_short / rv_long
        if ratio > 1.25:
            base_idx = min(base_idx + 1, 4)   # vol accelerating → upgrade
        elif ratio < 0.75:
            base_idx = max(base_idx - 1, 0)   # vol decelerating → downgrade

    return levels[base_idx]


def _compute_stats_from_trades(trades_list):
    """Compute all backtest stats from a list of trade dicts. Reusable for filtering."""
    if not trades_list:
        return None
    trades_df = pd.DataFrame(trades_list)
    pnls = trades_df["pnl"].values
    cum_pnl = np.cumsum(pnls)
    total_pnl = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0
    n_trades = len(trades_list)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / max(n_trades, 1) * 100
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    max_dd_idx = int(np.argmin(drawdown)) if len(drawdown) > 0 else 0

    avg_hold = float(np.mean(trades_df["hold_days"])) if n_trades > 0 else 0.0

    if len(pnls) > 1 and np.std(pnls) > 1e-10:
        trades_per_year = 252.0 / max(avg_hold, 1)
        sharpe = float(np.clip((np.mean(pnls) / np.std(pnls)) * np.sqrt(trades_per_year), -10, 10))
    else:
        sharpe = 0.0

    if len(pnls) > 1:
        neg_pnls = pnls[pnls < 0]
        downside_std = np.std(neg_pnls) if len(neg_pnls) > 1 else (np.std(pnls) if len(pnls) > 1 else 1e-6)
        tpy = 252.0 / max(avg_hold, 1)
        sortino = float((np.mean(pnls) / max(downside_std, 1e-6)) * np.sqrt(tpy))
    else:
        sortino = 0.0

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / max(gross_loss, 1e-6)

    calmar = total_pnl / abs(max_dd) if abs(max_dd) > 0 else 0.0
    best_trade = float(np.max(pnls)) if len(pnls) > 0 else 0.0
    worst_trade = float(np.min(pnls)) if len(pnls) > 0 else 0.0

    # Extended risk metrics
    from core.fx_analytics import omega_ratio, ulcer_index, max_consecutive, recovery_factor
    omega = omega_ratio(pnls) if len(pnls) > 1 else 0.0
    ulcer = ulcer_index(pnls) if len(pnls) > 1 else 0.0
    max_streak_wins = max_consecutive(pnls, "win") if len(pnls) > 0 else 0
    max_streak_losses = max_consecutive(pnls, "loss") if len(pnls) > 0 else 0
    rec_factor = recovery_factor(total_pnl, max_dd) if abs(max_dd) > 0 else 0.0

    # Monthly P&L
    trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["month"] = trades_df["entry_dt"].dt.to_period("M").astype(str)
    monthly_pnl = trades_df.groupby("month")["pnl"].sum().reset_index()

    # Regime stats
    regime_labels = _REGIME_LABELS
    regime_stats = {}
    for reg in regime_labels:
        mask = trades_df["regime"] == reg
        if mask.sum() > 0:
            reg_wins = (trades_df.loc[mask, "pnl"] > 0).sum()
            regime_stats[reg] = {"n": int(mask.sum()), "win_rate": float(reg_wins / mask.sum() * 100)}
        else:
            regime_stats[reg] = {"n": 0, "win_rate": 0.0}

    return {
        "trades": trades_df,
        "cum_pnl": cum_pnl,
        "drawdown": drawdown,
        "max_dd": max_dd,
        "max_dd_idx": max_dd_idx,
        "total_pnl": total_pnl,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe": sharpe,
        "calmar": calmar,
        "avg_hold": avg_hold,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "monthly_pnl": monthly_pnl,
        "regime_stats": regime_stats,
        "sortino": sortino,
        "profit_factor": profit_factor,
        "omega": omega,
        "ulcer_index": ulcer,
        "max_streak_wins": max_streak_wins,
        "max_streak_losses": max_streak_losses,
        "recovery_factor": rec_factor,
    }


def _build_strategy_legs(strategy, S, T, r_d, r_f, atm_vol, rr25, bf25, delta):
    """
    Build option legs for a strategy.
    Returns list of dicts: {strike, cp, qty, vol, premium}
    atm_vol, rr25, bf25 are expected in vol-points (e.g. 8.5 = 8.5%).
    """
    # Defensive: if atm_vol looks like it is already in decimal, convert
    if atm_vol < 1.0:
        atm_vol = atm_vol * 100.0
    K_atm = _atm_strike(S, T, r_d, r_f, atm_vol / 100)
    vol_atm = atm_vol / 100

    # Compute wing vols from BF/RR quotes
    vol_call = vol_atm + bf25 / 100 + 0.5 * rr25 / 100
    vol_put = vol_atm + bf25 / 100 - 0.5 * rr25 / 100
    vol_call = max(vol_call, 0.01)
    vol_put = max(vol_put, 0.01)

    K_call = _strike_from_delta(delta, S, T, r_d, r_f, vol_call, +1)
    K_put = _strike_from_delta(-delta, S, T, r_d, r_f, vol_put, -1)

    # Far tenor for calendar spreads (double the near tenor)
    T_far = T * 2.0

    legs = []

    if strategy == "Long Straddle":
        legs = [
            {"strike": K_atm, "cp": +1, "qty": +1, "vol": vol_atm, "T": T},
            {"strike": K_atm, "cp": -1, "qty": +1, "vol": vol_atm, "T": T},
        ]
    elif strategy == "Short Straddle":
        legs = [
            {"strike": K_atm, "cp": +1, "qty": -1, "vol": vol_atm, "T": T},
            {"strike": K_atm, "cp": -1, "qty": -1, "vol": vol_atm, "T": T},
        ]
    elif strategy == "Long Strangle":
        legs = [
            {"strike": K_call, "cp": +1, "qty": +1, "vol": vol_call, "T": T},
            {"strike": K_put,  "cp": -1, "qty": +1, "vol": vol_put,  "T": T},
        ]
    elif strategy == "Short Strangle":
        legs = [
            {"strike": K_call, "cp": +1, "qty": -1, "vol": vol_call, "T": T},
            {"strike": K_put,  "cp": -1, "qty": -1, "vol": vol_put,  "T": T},
        ]
    elif strategy == "Long RR (buy call sell put)":
        legs = [
            {"strike": K_call, "cp": +1, "qty": +1, "vol": vol_call, "T": T},
            {"strike": K_put,  "cp": -1, "qty": -1, "vol": vol_put,  "T": T},
        ]
    elif strategy == "Short RR":
        legs = [
            {"strike": K_call, "cp": +1, "qty": -1, "vol": vol_call, "T": T},
            {"strike": K_put,  "cp": -1, "qty": +1, "vol": vol_put,  "T": T},
        ]
    elif strategy == "Long BF":
        legs = [
            {"strike": K_call, "cp": +1, "qty": +1, "vol": vol_call, "T": T},
            {"strike": K_atm,  "cp": +1, "qty": -1, "vol": vol_atm,  "T": T},
            {"strike": K_atm,  "cp": -1, "qty": -1, "vol": vol_atm,  "T": T},
            {"strike": K_put,  "cp": -1, "qty": +1, "vol": vol_put,  "T": T},
        ]
    elif strategy == "Short BF":
        legs = [
            {"strike": K_call, "cp": +1, "qty": -1, "vol": vol_call, "T": T},
            {"strike": K_atm,  "cp": +1, "qty": +1, "vol": vol_atm,  "T": T},
            {"strike": K_atm,  "cp": -1, "qty": +1, "vol": vol_atm,  "T": T},
            {"strike": K_put,  "cp": -1, "qty": -1, "vol": vol_put,  "T": T},
        ]
    elif strategy == "Long Calendar (buy far sell near)":
        legs = [
            {"strike": K_atm, "cp": +1, "qty": +1, "vol": vol_atm * 0.97, "T": T_far},
            {"strike": K_atm, "cp": +1, "qty": -1, "vol": vol_atm,        "T": T},
        ]
    elif strategy == "Short Calendar":
        legs = [
            {"strike": K_atm, "cp": +1, "qty": -1, "vol": vol_atm * 0.97, "T": T_far},
            {"strike": K_atm, "cp": +1, "qty": +1, "vol": vol_atm,        "T": T},
        ]

    # Price each leg
    for leg in legs:
        leg["premium"] = _gk_price(S, leg["strike"], leg["T"], r_d, r_f, leg["vol"], leg["cp"])

    return legs


def _value_strategy(legs, S, T_remaining, r_d, r_f, atm_vol_new):
    """
    Mark-to-market value of a strategy at a later date.
    T_remaining: fraction of original T left for each leg.
    atm_vol_new: current ATM vol (decimal).
    """
    if not legs:
        return 0.0

    # If atm_vol_new looks like vol-points (> 1.0), convert to decimal
    if atm_vol_new > 1.0:
        atm_vol_new = atm_vol_new / 100.0

    total = 0.0
    ref_T = legs[0]["T"]
    ref_vol = max(legs[0]["vol"], 0.005)

    for leg in legs:
        # Scale remaining time proportionally for each leg (T_remaining is in years)
        T_r = T_remaining * leg["T"] / ref_T if ref_T > 0 else 0.0
        # T_remaining is already in years; T_r preserves proportional scaling for multi-leg

        # At or past expiry for this leg: use intrinsic value
        if T_r <= 1e-6:
            px = max(leg["cp"] * (S - leg["strike"]), 0.0)
        else:
            # Shift vol proportionally to new ATM level, clamped to
            # avoid extreme ratios in stressed scenarios
            vol_shift = atm_vol_new / ref_vol
            vol_shift = max(min(vol_shift, 3.0), 0.33)  # clamp to 3x range
            adj_vol = leg["vol"] * vol_shift
            adj_vol = max(adj_vol, 0.005)
            px = _gk_price(S, leg["strike"], T_r, r_d, r_f, adj_vol, leg["cp"])
        total += leg["qty"] * px
    return total


def run_backtest(strategy, pair, tenor, delta, lookback_years,
                 entry_signal, exit_rule, notional):
    """
    Execute the full backtest.

    1. Generate historical spot + vol data for lookback period
    2. Identify entry dates based on signal
    3. For each entry:
       a. Get market data (spot, vol surface, rates)
       b. Construct the strategy legs (compute strikes from deltas)
       c. Price entry using GK
       d. Simulate holding period:
          - Regenerate spot/vol path from entry to expiry
          - Track daily mark-to-market
          - Apply exit rules (profit target, stop loss)
       e. Record exit date, exit price, P&L
    4. Compile results into analytics and trade log
    """
    T = tenor_to_years(tenor)
    hold_days = tenor_to_days(tenor)

    data = _generate_backtest_data(pair, lookback_years)
    if data is None or len(data) < hold_days + 20:
        return None

    # --- Parse exit rule thresholds ---
    tp_pct, sl_pct, trailing_pct = None, None, None
    if exit_rule == "tp50":
        tp_pct = 0.50
    elif exit_rule == "sl30":
        sl_pct = -0.30
    elif exit_rule == "tp100_sl50":
        tp_pct = 1.00
        sl_pct = -0.50
    elif exit_rule == "trailing_20":
        trailing_pct = 0.20

    # --- Identify entry points ---
    entry_indices = []
    monthly_gap = max(hold_days, 21)  # at least 1 month between entries

    if entry_signal == "fixed":
        idx = 0
        while idx < len(data) - hold_days:
            entry_indices.append(idx)
            idx += monthly_gap
    else:
        # Compute rolling vol percentile and RR z-score for signal logic
        window = 60
        vol_arr = data["atm_vol"].values
        rr_arr = data["rr25"].values
        for idx in range(window, len(data) - hold_days):
            lookback_slice = vol_arr[max(0, idx - 252):idx]
            current_vol = vol_arr[idx]

            if entry_signal == "vol_cheap":
                pct = (np.sum(lookback_slice < current_vol) / max(len(lookback_slice), 1)) * 100
                if pct < 20:
                    if not entry_indices or (idx - entry_indices[-1]) >= monthly_gap:
                        entry_indices.append(idx)
            elif entry_signal == "vol_rich":
                pct = (np.sum(lookback_slice < current_vol) / max(len(lookback_slice), 1)) * 100
                if pct > 80:
                    if not entry_indices or (idx - entry_indices[-1]) >= monthly_gap:
                        entry_indices.append(idx)
            elif entry_signal == "skew_extreme":
                rr_slice = rr_arr[max(0, idx - 252):idx]
                rr_mu = np.mean(rr_slice)
                rr_sigma = max(np.std(rr_slice), 0.01)
                z = (rr_arr[idx] - rr_mu) / rr_sigma
                if abs(z) > 1.5:
                    if not entry_indices or (idx - entry_indices[-1]) >= monthly_gap:
                        entry_indices.append(idx)
            elif entry_signal == "vol_momentum":
                # Enter when short-term vol accelerates vs long-term
                if idx >= 60:
                    _spot_short = np.maximum(data["spot"].values[idx-10:idx], 1e-10)
                    _spot_long = np.maximum(data["spot"].values[idx-60:idx], 1e-10)
                    rv_short = np.std(np.diff(np.log(_spot_short))) * np.sqrt(252) * 100
                    rv_long = np.std(np.diff(np.log(_spot_long))) * np.sqrt(252) * 100
                    if rv_long > 1e-6 and rv_short / rv_long > 1.20:
                        if not entry_indices or (idx - entry_indices[-1]) >= monthly_gap:
                            entry_indices.append(idx)

    if not entry_indices:
        return None

    # --- Run trades ---
    trades = []
    daily_pnl = pd.Series(0.0, index=data.index, dtype=float)

    for entry_idx in entry_indices:
        if entry_idx + hold_days >= len(data):
            continue

        # Entry market data
        row_entry = data.iloc[entry_idx]
        S_entry = row_entry["spot"]
        vol_entry = row_entry["atm_vol"]
        rr_entry = row_entry["rr25"]
        bf_entry = row_entry["bf25"]
        rd_entry = row_entry["r_d"]
        rf_entry = row_entry["r_f"]
        rv_short = row_entry.get("rv_20d") if "rv_20d" in row_entry.index else None
        rv_long = row_entry.get("rv_60d") if "rv_60d" in row_entry.index else None
        regime = _compute_regime(vol_entry, rv_short, rv_long)

        # Build strategy legs at entry
        legs = _build_strategy_legs(
            strategy, S_entry, T, rd_entry, rf_entry,
            vol_entry, rr_entry, bf_entry, delta,
        )
        if not legs:
            continue
        entry_premium = sum(leg["qty"] * leg["premium"] for leg in legs)
        entry_cost = entry_premium * notional

        # --- Simulate holding period with MtM ---
        exit_idx = entry_idx + hold_days
        exit_reason = "expiry"
        best_mtm = 0.0   # track best MtM P&L (starts at 0, not entry_cost)
        worst_mtm = 0.0   # track worst MtM P&L (starts at 0, not entry_cost)

        for d in range(1, hold_days + 1):
            day_idx = entry_idx + d
            if day_idx >= len(data):
                exit_idx = day_idx - 1
                break

            row_d = data.iloc[day_idx]
            S_d = row_d["spot"]
            vol_d = row_d["atm_vol"] / 100 if np.isfinite(row_d["atm_vol"]) else 0.10
            T_rem = max((hold_days - d) / 365.0, 1e-6)

            mtm_value = _value_strategy(legs, S_d, T_rem, row_d["r_d"], row_d["r_f"], vol_d)
            mtm_pnl = (mtm_value - entry_premium) * notional

            daily_pnl.iloc[day_idx] += mtm_pnl / max(hold_days, 1)
            best_mtm = max(best_mtm, mtm_pnl)
            worst_mtm = min(worst_mtm, mtm_pnl)

            # Check exit rules — use max of |entry_cost| and a notional-based
            # floor so near-zero-premium structures (risk reversals) don't
            # produce extreme pnl_pct values that trigger exits immediately
            cost_denom = max(abs(entry_cost), notional * 0.001, 1.0)
            pnl_pct = mtm_pnl / cost_denom

            if tp_pct is not None and pnl_pct >= tp_pct:
                exit_idx = day_idx
                exit_reason = f"TP {tp_pct*100:.0f}%"
                break
            if sl_pct is not None and pnl_pct <= sl_pct:
                exit_idx = day_idx
                exit_reason = f"SL {sl_pct*100:.0f}%"
                break
            if trailing_pct is not None and best_mtm > 1e-10:
                drawdown_from_peak = (best_mtm - mtm_pnl) / best_mtm
                if drawdown_from_peak >= trailing_pct:
                    exit_idx = day_idx
                    exit_reason = f"Trail {trailing_pct*100:.0f}%"
                    break

        exit_idx = min(exit_idx, len(data) - 1)

        # Exit valuation
        row_exit = data.iloc[exit_idx]
        S_exit = row_exit["spot"]
        vol_exit = row_exit["atm_vol"]
        T_exit = max((hold_days - (exit_idx - entry_idx)) / 365.0, 0.0)

        if T_exit <= 0:
            # At expiry: intrinsic value
            exit_value = sum(
                leg["qty"] * max(leg["cp"] * (S_exit - leg["strike"]), 0.0)
                for leg in legs
            )
        else:
            exit_value = _value_strategy(
                legs, S_exit, T_exit, row_exit["r_d"], row_exit["r_f"],
                vol_exit / 100,
            )

        # Transaction costs: bid-ask spread ~ 0.3 vega for vanilla strategies
        # Approximation: 0.1% of notional per leg at entry + exit
        n_legs = len(legs)
        tc_per_trade = notional * 0.001 * n_legs * 2  # entry + exit
        trade_pnl = (exit_value - entry_premium) * notional - tc_per_trade
        hold_d = exit_idx - entry_idx

        # Determine entry signal label
        sig_labels = {
            "fixed": "Monthly", "vol_cheap": "Vol Cheap", "vol_rich": "Vol Rich",
            "skew_extreme": "Skew Extreme", "vol_momentum": "Vol Momentum",
        }
        sig_label = sig_labels.get(entry_signal, entry_signal.replace("_", " ").title())

        trades.append({
            "entry_date": data.index[entry_idx].strftime("%Y-%m-%d"),
            "exit_date": data.index[exit_idx].strftime("%Y-%m-%d"),
            "entry_spot": round(S_entry, 5),
            "exit_spot": round(S_exit, 5),
            "entry_vol": round(vol_entry, 2),
            "exit_vol": round(vol_exit, 2),
            "entry_premium": round(entry_premium * notional, 2),
            "exit_value": round(exit_value * notional, 2),
            "pnl": round(trade_pnl, 2),
            "pnl_pct": round(trade_pnl / max(abs(entry_cost), notional * 0.001, 1.0) * 100, 2),
            "spot_move_pct": round((S_exit - S_entry) / max(abs(S_entry), 1e-10) * 100, 2),
            "mae": round(worst_mtm, 0),
            "mfe": round(best_mtm, 0),
            "regime": regime,
            "signal": sig_label,
            "hold_days": hold_d,
            "exit_reason": exit_reason,
        })

    if not trades:
        return None

    # --- Compile results ---
    results = _compute_stats_from_trades(trades)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

_INPUT_PANEL_STYLE = {
    **CARD_STYLE,
    "width": "350px",
    "minWidth": "350px",
    "maxWidth": "350px",
    "flexShrink": "0",
    "overflowY": "auto",
    "maxHeight": "calc(100vh - 100px)",
}

_RESULTS_PANEL_STYLE = {
    "flex": "1",
    "minWidth": "0",
    "display": "flex",
    "flexDirection": "column",
    "gap": "16px",
}


def layout():
    return html.Div([
        dcc.Download(id="bt-csv-download"),
        html.Div([
            # ── Left: Input Panel ─────────────────────────────────────
            html.Div([
                html.Div("STRATEGY BACKTESTER", style=CARD_HEADER_STYLE),

                html.Label("STRATEGY", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="bt-strategy",
                    options=[{"label": s, "value": s} for s in _STRATEGIES],
                    value="Long Straddle",
                    clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                html.Label("PAIR", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="bt-pair",
                    options=[{"label": p, "value": p} for p in _PAIRS],
                    value="EURUSD",
                    clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                html.Label("TENOR", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="bt-tenor",
                    options=_TENORS,
                    value="1M",
                    clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                html.Label("DELTA (STRANGLE / RR)", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="bt-delta",
                    options=_DELTAS,
                    value=0.25,
                    clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                html.Label("LOOKBACK PERIOD", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="bt-lookback",
                    options=_LOOKBACKS,
                    value=2,
                    clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                html.Label("ENTRY SIGNAL", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="bt-entry-signal",
                    options=_ENTRY_SIGNALS,
                    value="fixed",
                    clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                html.Label("EXIT RULE", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="bt-exit-rule",
                    options=_EXIT_RULES,
                    value="hold",
                    clearable=False,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),

                html.Label("NOTIONAL PER TRADE", style=LABEL_STYLE),
                dcc.Input(
                    id="bt-notional",
                    type="number",
                    value=1_000_000,
                    step=100_000,
                    min=10_000,
                    style={**INPUT_STYLE, "marginBottom": "20px"},
                    debounce=True,
                ),

                html.Button(
                    "RUN BACKTEST",
                    id="bt-run-btn",
                    n_clicks=0,
                    style={
                        **BUTTON_STYLE,
                        "width": "100%",
                        "padding": "14px",
                        "fontSize": "13px",
                        "letterSpacing": "2px",
                        "background": f"linear-gradient(135deg, {COLORS['accent_blue']}, {COLORS['accent_purple']})",
                        "boxShadow": f"0 6px 20px rgba(255,136,0,0.25)",
                    },
                ),

                html.Div([
                    html.Div("FILTER BY VOL REGIME", style={
                        "color": "#808080", "fontSize": "9px", "fontWeight": "600",
                        "textTransform": "uppercase", "marginBottom": "4px",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                    dcc.Dropdown(
                        id="bt-regime-filter",
                        options=[
                            {"label": "ALL REGIMES", "value": "ALL"},
                            {"label": "LOW vol only", "value": "LOW"},
                            {"label": "NORMAL vol only", "value": "NORMAL"},
                            {"label": "ELEVATED vol only", "value": "ELEVATED"},
                            {"label": "HIGH vol only", "value": "HIGH"},
                        ],
                        value="ALL",
                        clearable=False,
                        style={"backgroundColor": "#000000", "color": "#d4d4d4"},
                    ),
                ], style={"marginTop": "12px"}),

                dcc.Store(id="bt-trades-store", data=[]),
            ], style=_INPUT_PANEL_STYLE, className="dashboard-card"),

            # ── Right: Results Panel ──────────────────────────────────
            html.Div([
                # Stats summary row
                dcc.Loading(
                    id="bt-loading",
                    type="dot",
                    color=COLORS["accent_cyan"],
                    children=html.Div(id="bt-stats-row", style={
                        "display": "flex",
                        "gap": "10px",
                        "flexWrap": "wrap",
                        "marginBottom": "4px",
                    }),
                ),

                # Charts 2x2 grid
                html.Div([
                    html.Div([
                        html.Button("CSV", id="bt-csv-equity", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="bt-equity-curve", style={"height": "300px"}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px", "padding": "12px"},
                       className="dashboard-card"),
                    html.Div([
                        html.Button("CSV", id="bt-csv-pnl", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="bt-pnl-dist", style={"height": "300px"}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px", "padding": "12px"},
                       className="dashboard-card"),
                ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

                html.Div([
                    html.Div([
                        html.Button("CSV", id="bt-csv-monthly", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="bt-monthly-returns", style={"height": "300px"}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px", "padding": "12px"},
                       className="dashboard-card"),
                    html.Div([
                        html.Button("CSV", id="bt-csv-regime", n_clicks=0, style=CSV_BTN_STYLE),
                        dcc.Graph(id="bt-regime-winrate", style={"height": "300px"}),
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "400px", "padding": "12px"},
                       className="dashboard-card"),
                ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

                # Trade log table
                html.Div([
                    html.Div("TRADE LOG", style=CARD_HEADER_STYLE),
                    html.Div(id="bt-trade-log"),
                ], style=CARD_STYLE, className="dashboard-card"),
            ], style=_RESULTS_PANEL_STYLE),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Chart Builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_equity_curve(results):
    """Cumulative P&L with drawdown shading."""
    cum = results["cum_pnl"]
    dd = results["drawdown"]
    n = len(cum)
    x = list(range(n))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Drawdown fill area
    fig.add_trace(
        go.Scatter(
            x=x, y=dd, fill="tozeroy",
            fillcolor="rgba(255,51,51,0.10)",
            line=dict(color="rgba(255,51,51,0.35)", width=1),
            name="Drawdown",
            hovertemplate="DD: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Cumulative P&L line
    fig.add_trace(
        go.Scatter(
            x=x, y=cum,
            line=dict(color=COLORS["accent_cyan"], width=2.5),
            name="Cumulative P&L",
            hovertemplate="P&L: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Max drawdown annotation
    dd_idx = results["max_dd_idx"]
    fig.add_annotation(
        x=dd_idx, y=dd[dd_idx],
        text=f"Max DD: ${results['max_dd']:,.0f}",
        showarrow=True, arrowhead=2,
        font=dict(color=COLORS["accent_red"], size=10),
        arrowcolor=COLORS["accent_red"],
        yref="y2",
    )

    fig.update_layout(
        **chart_layout(
        title=dict(text="EQUITY CURVE", font=dict(size=13, color=COLORS["text_primary"])),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=9, color=COLORS["text_secondary"])),
        margin=dict(l=50, r=20, t=45, b=30),
    ))
    fig.update_yaxes(title_text="Cumulative P&L ($)", secondary_y=False,
                     gridcolor="#1a1a30", tickfont=dict(size=9))
    fig.update_yaxes(title_text="Drawdown ($)", secondary_y=True,
                     gridcolor="#1a1a30", tickfont=dict(size=9))
    fig.update_xaxes(title_text="Trade #", tickfont=dict(size=9))

    return fig


def _build_pnl_distribution(results):
    """Histogram of per-trade P&L with win/loss coloring."""
    pnls = results["trades"]["pnl"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    fig = go.Figure()

    if len(losses) > 0:
        fig.add_trace(go.Histogram(
            x=losses, name="Losses",
            marker_color=COLORS["accent_red"],
            opacity=0.75,
            hovertemplate="P&L: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
        ))
    if len(wins) > 0:
        fig.add_trace(go.Histogram(
            x=wins, name="Wins",
            marker_color=COLORS["accent_green"],
            opacity=0.75,
            hovertemplate="P&L: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
        ))

    # Mean and median lines
    mean_pnl = float(np.mean(pnls))
    median_pnl = float(np.median(pnls))

    fig.add_vline(x=mean_pnl, line_dash="dash",
                  line_color=COLORS["accent_cyan"], line_width=1.5,
                  annotation_text=f"Mean: ${mean_pnl:,.0f}",
                  annotation_font=dict(size=9, color=COLORS["accent_cyan"]))
    fig.add_vline(x=median_pnl, line_dash="dot",
                  line_color=COLORS["accent_orange"], line_width=1.5,
                  annotation_text=f"Median: ${median_pnl:,.0f}",
                  annotation_font=dict(size=9, color=COLORS["accent_orange"]))

    fig.update_layout(
        **chart_layout(
        title=dict(text="TRADE P&L DISTRIBUTION", font=dict(size=13, color=COLORS["text_primary"])),
        barmode="overlay",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=9, color=COLORS["text_secondary"])),
        margin=dict(l=50, r=20, t=45, b=30),
        xaxis_title="P&L per Trade ($)",
        yaxis_title="Count",
    ))

    return fig


def _build_monthly_returns(results):
    """Bar chart of monthly P&L, colored by sign."""
    monthly = results["monthly_pnl"]
    colors = [COLORS["accent_green"] if v > 0 else COLORS["accent_red"]
              for v in monthly["pnl"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month"],
        y=monthly["pnl"],
        marker_color=colors,
        marker_line_color=colors,
        marker_line_width=0.5,
        hovertemplate="%{x}<br>P&L: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        **chart_layout(
        title=dict(text="MONTHLY RETURNS", font=dict(size=13, color=COLORS["text_primary"])),
        showlegend=False,
        margin=dict(l=50, r=20, t=45, b=60),
        xaxis_title="Month",
        yaxis_title="P&L ($)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
    ))

    return fig


def _build_regime_winrate(results):
    """Grouped bar chart of win rate by vol regime."""
    rs = results["regime_stats"]
    regimes = _REGIME_LABELS
    win_rates = [rs.get(r, {"win_rate": 0})["win_rate"] for r in regimes]
    counts = [rs.get(r, {"n": 0})["n"] for r in regimes]
    bar_colors = [_REGIME_COLORS[r] for r in regimes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=regimes,
        y=win_rates,
        marker_color=bar_colors,
        marker_line_color=bar_colors,
        marker_line_width=0.5,
        text=[f"{wr:.0f}%<br>(n={n})" for wr, n in zip(win_rates, counts)],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_secondary"]),
        hovertemplate="%{x}<br>Win Rate: %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(y=50, line_dash="dash",
                  line_color=COLORS["text_muted"], line_width=1,
                  annotation_text="50%",
                  annotation_font=dict(size=9, color=COLORS["text_muted"]))

    fig.update_layout(
        **chart_layout(
        title=dict(text="WIN RATE BY VOL REGIME", font=dict(size=13, color=COLORS["text_primary"])),
        showlegend=False,
        margin=dict(l=50, r=20, t=45, b=30),
        yaxis_title="Win Rate (%)",
        xaxis_title="Vol Regime",
        yaxis=dict(range=[0, 110]),
    ))

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Stats Summary Builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_stat_box(label, value, fmt=",.0f", color=None):
    """Single stat box with optional color accent."""
    if color is None:
        color = COLORS["accent_cyan"]

    if isinstance(value, float):
        display = f"{value:{fmt}}"
    else:
        display = str(value)

    return html.Div([
        html.Div(label, style={
            **LABEL_STYLE,
            "fontSize": "9px",
            "marginBottom": "4px",
            "color": COLORS["text_muted"],
        }),
        html.Div(display, style={
            "fontSize": "16px",
            "fontWeight": "700",
            "color": color,
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ], style=make_stat_style(color))


def _build_stats_row(results):
    """Build the row of summary stat boxes."""
    pnl_color = COLORS["accent_green"] if results["total_pnl"] >= 0 else COLORS["accent_red"]
    wr_color = COLORS["accent_green"] if results["win_rate"] >= 50 else COLORS["accent_orange"]
    dd_color = COLORS["accent_red"]

    sr_color = COLORS["accent_purple"] if results["sharpe"] > 0.5 else COLORS["accent_orange"]
    return [
        _build_stat_box("TOTAL P&L", results["total_pnl"], ",.0f", pnl_color),
        _build_stat_box("# TRADES", results["n_trades"], ".0f", COLORS["accent_cyan"]),
        _build_stat_box("WIN RATE", results["win_rate"], ".1f", wr_color),
        _build_stat_box("PROFIT FACTOR", results.get("profit_factor", 0), ".2f", pnl_color),
        _build_stat_box("SHARPE", results["sharpe"], ".2f", sr_color),
        _build_stat_box("SORTINO", results.get("sortino", 0), ".2f", sr_color),
        _build_stat_box("MAX DD", results["max_dd"], ",.0f", dd_color),
        _build_stat_box("CALMAR", results["calmar"], ".2f", COLORS["accent_teal"]),
        _build_stat_box("AVG WIN", results["avg_win"], ",.0f", COLORS["accent_green"]),
        _build_stat_box("AVG LOSS", results["avg_loss"], ",.0f", COLORS["accent_red"]),
        _build_stat_box("AVG HOLD (D)", results["avg_hold"], ".0f", COLORS["accent_blue"]),
        _build_stat_box("BEST TRADE", results["best_trade"], ",.0f", COLORS["accent_green"]),
        _build_stat_box("WORST TRADE", results["worst_trade"], ",.0f", COLORS["accent_red"]),
        _build_stat_box("OMEGA", results.get("omega", 0), ".2f", COLORS["accent_teal"]),
        _build_stat_box("RECOVERY", results.get("recovery_factor", 0), ".1f", COLORS["accent_blue"]),
        _build_stat_box("WIN STREAK", results.get("max_streak_wins", 0), ".0f", COLORS["accent_green"]),
        _build_stat_box("LOSS STREAK", results.get("max_streak_losses", 0), ".0f", COLORS["accent_red"]),
        _build_stat_box("ULCER IDX", results.get("ulcer_index", 0), ".2f", COLORS["accent_purple"]),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Trade Log Table Builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_trade_log(results):
    """Build the DataTable of individual trades."""
    df = results["trades"].copy()
    df.insert(0, "#", range(1, len(df) + 1))

    columns = [
        {"name": "#",              "id": "#",              "type": "numeric"},
        {"name": "Entry Date",     "id": "entry_date",     "type": "text"},
        {"name": "Exit Date",      "id": "exit_date",      "type": "text"},
        {"name": "Entry Spot",     "id": "entry_spot",     "type": "numeric", "format": dash_table.Format.Format(precision=5)},
        {"name": "Exit Spot",      "id": "exit_spot",      "type": "numeric", "format": dash_table.Format.Format(precision=5)},
        {"name": "Entry Vol",      "id": "entry_vol",      "type": "numeric", "format": dash_table.Format.Format(precision=2)},
        {"name": "Exit Vol",       "id": "exit_vol",       "type": "numeric", "format": dash_table.Format.Format(precision=2)},
        {"name": "Entry Premium",  "id": "entry_premium",  "type": "numeric", "format": dash_table.Format.Format(precision=0, group=True)},
        {"name": "Exit Value",     "id": "exit_value",     "type": "numeric", "format": dash_table.Format.Format(precision=0, group=True)},
        {"name": "P&L",            "id": "pnl",            "type": "numeric", "format": dash_table.Format.Format(precision=0, group=True)},
        {"name": "P&L %",          "id": "pnl_pct",        "type": "numeric", "format": dash_table.Format.Format(precision=2)},
        {"name": "Hold Days",      "id": "hold_days",      "type": "numeric"},
        {"name": "Exit Reason",    "id": "exit_reason",    "type": "text"},
        {"name": "Regime",         "id": "regime",         "type": "text"},
        {"name": "Signal",         "id": "signal",         "type": "text"},
    ]

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=columns,
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={
            "overflowX": "auto",
            "borderRadius": "0px",
            "border": f"1px solid {COLORS['border_subtle']}",
        },
        style_header={
            **TABLE_HEADER_STYLE,
            "textAlign": "center",
        },
        style_cell={
            **TABLE_CELL_STYLE,
            "textAlign": "center",
            "minWidth": "80px",
            "maxWidth": "140px",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": "{pnl} > 0", "column_id": "pnl"},
                "color": COLORS["accent_green"],
                "fontWeight": "700",
            },
            {
                "if": {"filter_query": "{pnl} <= 0", "column_id": "pnl"},
                "color": COLORS["accent_red"],
                "fontWeight": "700",
            },
            {
                "if": {"filter_query": "{pnl_pct} > 0", "column_id": "pnl_pct"},
                "color": COLORS["accent_green"],
            },
            {
                "if": {"filter_query": "{pnl_pct} <= 0", "column_id": "pnl_pct"},
                "color": COLORS["accent_red"],
            },
            {
                "if": {"filter_query": '{exit_reason} contains "TP"', "column_id": "exit_reason"},
                "color": COLORS["accent_green"],
                "fontWeight": "700",
            },
            {
                "if": {"filter_query": '{exit_reason} contains "SL"', "column_id": "exit_reason"},
                "color": COLORS["accent_red"],
                "fontWeight": "700",
            },
            {
                "if": {"filter_query": '{exit_reason} = "expiry"', "column_id": "exit_reason"},
                "color": COLORS["text_muted"],
            },
            {
                "if": {"state": "active"},
                "backgroundColor": COLORS["bg_card_hover"],
                "border": f"1px solid {COLORS['accent_blue']}",
            },
        ],
        style_as_list_view=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Empty figure helper
# ═══════════════════════════════════════════════════════════════════════════

def _empty_fig(message="Run backtest to see results"):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=13, color=COLORS["text_muted"]),
    )
    fig.update_layout(
        **chart_layout(
        margin=dict(l=20, r=20, t=30, b=20),
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app):

    @app.callback(
        [
            Output("bt-stats-row", "children"),
            Output("bt-equity-curve", "figure"),
            Output("bt-pnl-dist", "figure"),
            Output("bt-monthly-returns", "figure"),
            Output("bt-regime-winrate", "figure"),
            Output("bt-trade-log", "children"),
            Output("bt-trades-store", "data"),
        ],
        Input("bt-run-btn", "n_clicks"),
        [
            State("bt-strategy", "value"),
            State("bt-pair", "value"),
            State("bt-tenor", "value"),
            State("bt-delta", "value"),
            State("bt-lookback", "value"),
            State("bt-entry-signal", "value"),
            State("bt-exit-rule", "value"),
            State("bt-notional", "value"),
        ],
        prevent_initial_call=True,
    )
    def _run_backtest(n_clicks, strategy, pair, tenor, delta, lookback,
                      entry_signal, exit_rule, notional):
        if not n_clicks:
            return (no_update,) * 7

        # Validate inputs
        if not strategy or not pair or not tenor:
            empty = _empty_fig("Select parameters and run backtest")
            return (
                [html.Div("Configure parameters and click RUN BACKTEST",
                          style={"color": COLORS["text_muted"], "padding": "20px"})],
                empty, empty, empty, empty,
                html.Div("No results yet.",
                         style={"color": COLORS["text_muted"], "padding": "20px"}),
                [],
            )

        notional = notional or 1_000_000
        delta = delta or 0.25
        lookback = lookback or 2

        results = run_backtest(
            strategy, pair, tenor, delta, lookback,
            entry_signal, exit_rule, notional,
        )

        if results is None:
            empty = no_data_fig(msg="NO DATA — insufficient Bloomberg history or no trades generated")
            return (
                [html.Div("No data or no trades generated with these parameters.",
                          style={"color": COLORS["accent_orange"], "padding": "20px"})],
                empty, empty, empty, empty,
                html.Div("No trades to display.",
                         style={"color": COLORS["text_muted"], "padding": "20px"}),
                [],
            )

        # Serialize trades for the store (raw dicts for regime filtering)
        trades_data = results["trades"].to_dict("records") if results and "trades" in results else []

        # Build all outputs
        stats = _build_stats_row(results)
        eq_fig = _build_equity_curve(results)
        dist_fig = _build_pnl_distribution(results)
        monthly_fig = _build_monthly_returns(results)
        regime_fig = _build_regime_winrate(results)
        trade_table = _build_trade_log(results)

        return stats, eq_fig, dist_fig, monthly_fig, regime_fig, trade_table, trades_data

    # ── CSV Export ──────────────────────────────────────────────────────
    @app.callback(
        Output("bt-csv-download", "data"),
        [Input("bt-csv-equity", "n_clicks"),
         Input("bt-csv-pnl", "n_clicks"),
         Input("bt-csv-monthly", "n_clicks"),
         Input("bt-csv-regime", "n_clicks")],
        [State("bt-equity-curve", "figure"),
         State("bt-pnl-dist", "figure"),
         State("bt-monthly-returns", "figure"),
         State("bt-regime-winrate", "figure")],
        prevent_initial_call=True,
    )
    def bt_csv_export(n1, n2, n3, n4, fig1, fig2, fig3, fig4):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        mapping = {
            "bt-csv-equity": (fig1, "Backtest", "EquityCurve"),
            "bt-csv-pnl": (fig2, "Backtest", "PnLDist"),
            "bt-csv-monthly": (fig3, "Backtest", "MonthlyReturns"),
            "bt-csv-regime": (fig4, "Backtest", "RegimeWinrate"),
        }
        if btn not in mapping:
            return no_update
        fig, panel, chart_type = mapping[btn]
        if not fig:
            return no_update
        return export_csv(fig, panel, chart_type)

    # ── Regime Filter ──────────────────────────────────────────────────
    @app.callback(
        [Output("bt-stats-row", "children", allow_duplicate=True),
         Output("bt-equity-curve", "figure", allow_duplicate=True),
         Output("bt-pnl-dist", "figure", allow_duplicate=True),
         Output("bt-monthly-returns", "figure", allow_duplicate=True),
         Output("bt-regime-winrate", "figure", allow_duplicate=True),
         Output("bt-trade-log", "children", allow_duplicate=True)],
        Input("bt-regime-filter", "value"),
        State("bt-trades-store", "data"),
        prevent_initial_call=True,
    )
    def _filter_by_regime(regime, trades_data):
        if not trades_data:
            raise PreventUpdate

        trades = trades_data
        if regime and regime != "ALL":
            trades = [t for t in trades_data if t.get("regime") == regime]

        if not trades:
            empty_fig = go.Figure()
            empty_fig.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000",
                                    font=dict(color="#808080"))
            return (
                [html.Div("No trades in selected regime", style={"color": "#808080"})],
                empty_fig, empty_fig, empty_fig, empty_fig,
                html.Div("No trades", style={"color": "#808080"})
            )

        results = _compute_stats_from_trades(trades)
        return (
            _build_stats_row(results),
            _build_equity_curve(results),
            _build_pnl_distribution(results),
            _build_monthly_returns(results),
            _build_regime_winrate(results),
            _build_trade_log(results),
        )
