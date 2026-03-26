"""
Market Data Simulator
=====================
Generates realistic live-feeling market data: ticking prices, vol moves,
order flow, and intraday patterns for the dashboard.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List


# ── Ticker Universe ───────────────────────────────────────────────────────
TICKERS = {
    "SPY":  {"spot": 521.40, "vol": 0.16, "beta": 1.00, "sector": "Index"},
    "QQQ":  {"spot": 446.80, "vol": 0.20, "beta": 1.15, "sector": "Index"},
    "AAPL": {"spot": 178.50, "vol": 0.24, "beta": 1.10, "sector": "Tech"},
    "MSFT": {"spot": 416.20, "vol": 0.22, "beta": 0.95, "sector": "Tech"},
    "NVDA": {"spot": 882.30, "vol": 0.45, "beta": 1.60, "sector": "Tech"},
    "TSLA": {"spot": 176.40, "vol": 0.55, "beta": 1.80, "sector": "Auto"},
    "AMZN": {"spot": 186.50, "vol": 0.28, "beta": 1.20, "sector": "Tech"},
    "META": {"spot": 506.70, "vol": 0.32, "beta": 1.25, "sector": "Tech"},
    "JPM":  {"spot": 198.30, "vol": 0.20, "beta": 0.90, "sector": "Finance"},
    "GS":   {"spot": 468.50, "vol": 0.25, "beta": 1.10, "sector": "Finance"},
    "XOM":  {"spot": 118.40, "vol": 0.22, "beta": 0.75, "sector": "Energy"},
    "GLD":  {"spot": 215.60, "vol": 0.15, "beta": 0.05, "sector": "Commodity"},
}

EXPIRY_DATES = [
    "2026-03-27", "2026-04-03", "2026-04-17", "2026-05-15",
    "2026-06-19", "2026-09-18", "2026-12-18", "2027-01-15",
    "2027-03-19", "2027-06-18", "2027-12-17",
]


def simulate_intraday_prices(ticker="SPY", minutes=390, seed=None):
    """Simulate a full trading day of minute-bar data with realistic intraday patterns."""
    rng = np.random.RandomState(seed)
    info = TICKERS.get(ticker, TICKERS["SPY"])
    S = info["spot"]
    sigma = info["vol"]

    # Intraday U-shape volume pattern
    t = np.linspace(0, 1, minutes)
    vol_multiplier = 1.5 * (1 + 2 * (t - 0.5) ** 2)  # higher at open/close

    dt = 1 / (252 * 390)
    prices = [S]
    volumes = []

    for i in range(minutes):
        local_vol = sigma * vol_multiplier[i]
        ret = (0 - 0.5 * local_vol ** 2) * dt + local_vol * np.sqrt(dt) * rng.standard_normal()
        prices.append(prices[-1] * np.exp(ret))
        base_vol = int(50000 * vol_multiplier[i] * rng.lognormal(0, 0.5))
        volumes.append(base_vol)

    prices = np.array(prices[1:])
    volumes = np.array(volumes)

    # Generate OHLC from minute bars
    opens = prices * (1 + rng.uniform(-0.001, 0.001, minutes))
    highs = np.maximum(prices, opens) * (1 + rng.uniform(0, 0.003, minutes))
    lows = np.minimum(prices, opens) * (1 - rng.uniform(0, 0.003, minutes))

    start = datetime(2026, 3, 20, 9, 30)
    timestamps = [start + timedelta(minutes=i) for i in range(minutes)]

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    })


def simulate_tick_update(current_prices: Dict[str, float], seed=None) -> Dict:
    """Simulate a single tick update for all tickers. Used for live-refresh."""
    rng = np.random.RandomState(seed)
    updates = {}

    # Correlated market move
    market_move = rng.standard_normal() * 0.001

    for ticker, info in TICKERS.items():
        current = current_prices.get(ticker, info["spot"])
        beta = info["beta"]
        sigma = info["vol"]
        idio = rng.standard_normal() * sigma / np.sqrt(252 * 390)
        move = beta * market_move + idio
        new_price = current * (1 + move)

        change = new_price - info["spot"]
        change_pct = (new_price / info["spot"] - 1) * 100

        updates[ticker] = {
            "price": round(new_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "bid": round(new_price - rng.uniform(0.01, 0.05), 2),
            "ask": round(new_price + rng.uniform(0.01, 0.05), 2),
            "volume": int(rng.lognormal(10, 1.5)),
        }

    return updates


def generate_portfolio_history(positions, S_init, sigma, r, q, days=60, seed=42):
    """Generate historical portfolio P&L series."""
    rng = np.random.RandomState(seed)
    from core.pricing import bs_price

    daily_pnl = []
    cumulative = []
    spot_path = []
    total = 0
    S = S_init

    for day in range(days):
        ret = (r - q - 0.5 * sigma ** 2) / 252 + sigma / np.sqrt(252) * rng.standard_normal()
        S_new = S * np.exp(ret)

        day_pnl = 0
        for pos in positions:
            T_new = max(pos["expiry"] - day / 365.0, 1e-6)
            T_old = max(pos["expiry"] - (day - 1) / 365.0, 1e-6)
            p_new = bs_price(S_new, pos["strike"], T_new, r, q, pos["vol"], pos["option_type"])
            p_old = bs_price(S, pos["strike"], T_old, r, q, pos["vol"], pos["option_type"])
            day_pnl += (p_new - p_old) * pos["quantity"] * pos.get("multiplier", 100)

        daily_pnl.append(day_pnl)
        total += day_pnl
        cumulative.append(total)
        spot_path.append(S_new)
        S = S_new

    dates = [datetime(2026, 1, 19) + timedelta(days=i) for i in range(days)]
    return pd.DataFrame({
        "date": dates,
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative,
        "spot": spot_path,
    })


def generate_greeks_timeseries(S, K, T_start, r, q, sigma, option_type="call", days=60):
    """How Greeks evolve as time passes (theta decay visualization)."""
    from core.pricing import compute_all_greeks

    records = []
    for day in range(days):
        T = max(T_start - day / 365.0, 1e-6)
        greeks = compute_all_greeks(S, K, T, r, q, sigma, option_type)
        greeks["day"] = day
        greeks["dte"] = max(int(T * 365), 0)
        records.append(greeks)

    return pd.DataFrame(records)
