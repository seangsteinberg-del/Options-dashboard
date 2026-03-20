"""
Portfolio Manager
=================
Persistent multi-book portfolio management with trade history,
P&L tracking, Greeks aggregation, and position lifecycle.

Stores state in a local JSON file so positions survive restarts.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from copy import deepcopy

import numpy as np
import pandas as pd

from core.pricing import bs_price, compute_all_greeks

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "portfolio.json")


def _ensure_data_dir():
    d = os.path.dirname(PORTFOLIO_FILE)
    os.makedirs(d, exist_ok=True)


def _load() -> dict:
    _ensure_data_dir()
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return _default_state()


def _save(state: dict):
    _ensure_data_dir()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _default_state() -> dict:
    """Create a default portfolio state with sample positions."""
    return {
        "books": {
            "MAIN": {
                "name": "MAIN",
                "description": "Primary Options Book",
                "created": datetime.now().isoformat(),
                "positions": [
                    {"id": "POS-001", "ticker": "SPY", "strike": 515, "expiry": 0.08,
                     "option_type": "call", "quantity": 20, "entry_price": 8.50,
                     "entry_date": "2026-03-10", "vol": 0.18, "multiplier": 100,
                     "tags": ["momentum"], "notes": "Long delta bet"},
                    {"id": "POS-002", "ticker": "SPY", "strike": 525, "expiry": 0.08,
                     "option_type": "call", "quantity": -15, "entry_price": 3.20,
                     "entry_date": "2026-03-10", "vol": 0.17, "multiplier": 100,
                     "tags": ["spread"], "notes": "Cap upside"},
                    {"id": "POS-003", "ticker": "SPY", "strike": 510, "expiry": 0.25,
                     "option_type": "put", "quantity": -10, "entry_price": 4.80,
                     "entry_date": "2026-03-05", "vol": 0.19, "multiplier": 100,
                     "tags": ["income"], "notes": "Short put income"},
                    {"id": "POS-004", "ticker": "SPY", "strike": 500, "expiry": 0.25,
                     "option_type": "put", "quantity": 10, "entry_price": 2.90,
                     "entry_date": "2026-03-05", "vol": 0.21, "multiplier": 100,
                     "tags": ["hedge"], "notes": "Tail hedge"},
                    {"id": "POS-005", "ticker": "NVDA", "strike": 880, "expiry": 0.25,
                     "option_type": "call", "quantity": 5, "entry_price": 45.00,
                     "entry_date": "2026-03-01", "vol": 0.45, "multiplier": 100,
                     "tags": ["directional"], "notes": "Earnings play"},
                    {"id": "POS-006", "ticker": "NVDA", "strike": 920, "expiry": 0.25,
                     "option_type": "call", "quantity": -5, "entry_price": 28.00,
                     "entry_date": "2026-03-01", "vol": 0.43, "multiplier": 100,
                     "tags": ["spread"], "notes": "Bull call spread"},
                    {"id": "POS-007", "ticker": "AAPL", "strike": 175, "expiry": 0.50,
                     "option_type": "put", "quantity": -8, "entry_price": 5.20,
                     "entry_date": "2026-02-20", "vol": 0.24, "multiplier": 100,
                     "tags": ["income"], "notes": "Cash secured put"},
                    {"id": "POS-008", "ticker": "TSLA", "strike": 180, "expiry": 0.17,
                     "option_type": "call", "quantity": 10, "entry_price": 12.80,
                     "entry_date": "2026-03-12", "vol": 0.55, "multiplier": 100,
                     "tags": ["momentum", "earnings"], "notes": "Pre-earnings vol play"},
                    {"id": "POS-009", "ticker": "TSLA", "strike": 165, "expiry": 0.17,
                     "option_type": "put", "quantity": 10, "entry_price": 8.50,
                     "entry_date": "2026-03-12", "vol": 0.58, "multiplier": 100,
                     "tags": ["straddle"], "notes": "Long straddle"},
                    {"id": "POS-010", "ticker": "QQQ", "strike": 445, "expiry": 0.33,
                     "option_type": "call", "quantity": 15, "entry_price": 14.20,
                     "entry_date": "2026-03-08", "vol": 0.20, "multiplier": 100,
                     "tags": ["index"], "notes": "Tech beta"},
                    {"id": "POS-011", "ticker": "META", "strike": 500, "expiry": 0.42,
                     "option_type": "call", "quantity": -6, "entry_price": 22.50,
                     "entry_date": "2026-03-15", "vol": 0.32, "multiplier": 100,
                     "tags": ["covered"], "notes": "Covered call"},
                    {"id": "POS-012", "ticker": "MSFT", "strike": 410, "expiry": 0.50,
                     "option_type": "put", "quantity": 5, "entry_price": 8.90,
                     "entry_date": "2026-03-02", "vol": 0.22, "multiplier": 100,
                     "tags": ["hedge"], "notes": "Portfolio hedge"},
                ],
            },
            "HEDGE": {
                "name": "HEDGE",
                "description": "Tail Risk Hedges",
                "created": datetime.now().isoformat(),
                "positions": [
                    {"id": "POS-H01", "ticker": "SPY", "strike": 470, "expiry": 0.50,
                     "option_type": "put", "quantity": 25, "entry_price": 3.40,
                     "entry_date": "2026-03-01", "vol": 0.25, "multiplier": 100,
                     "tags": ["crash-put"], "notes": "-10% crash put"},
                    {"id": "POS-H02", "ticker": "SPY", "strike": 450, "expiry": 1.00,
                     "option_type": "put", "quantity": 15, "entry_price": 5.80,
                     "entry_date": "2026-02-15", "vol": 0.27, "multiplier": 100,
                     "tags": ["tail-risk"], "notes": "Deep OTM 1y put"},
                    {"id": "POS-H03", "ticker": "QQQ", "strike": 400, "expiry": 0.50,
                     "option_type": "put", "quantity": 10, "entry_price": 4.20,
                     "entry_date": "2026-03-01", "vol": 0.23, "multiplier": 100,
                     "tags": ["tail-risk"], "notes": "Tech crash hedge"},
                ],
            },
        },
        "trade_history": [],
        "risk_limits": {
            "max_delta": 5000,
            "max_gamma": 500,
            "max_vega": 10000,
            "max_theta": -2000,
            "max_var_95": 50000,
            "max_notional": 5000000,
            "max_single_name_pct": 0.30,
        },
        "settings": {
            "default_book": "MAIN",
            "base_currency": "USD",
            "risk_free_rate": 0.05,
            "default_div_yield": 0.015,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio API
# ═══════════════════════════════════════════════════════════════════════════

def get_state() -> dict:
    return _load()


def get_books() -> List[str]:
    return list(_load()["books"].keys())


def get_positions(book: str = "MAIN") -> List[dict]:
    state = _load()
    return state["books"].get(book, {}).get("positions", [])


def get_all_positions() -> List[dict]:
    """All positions across all books."""
    state = _load()
    all_pos = []
    for book_name, book in state["books"].items():
        for pos in book.get("positions", []):
            p = deepcopy(pos)
            p["book"] = book_name
            all_pos.append(p)
    return all_pos


def add_position(book: str, position: dict) -> str:
    state = _load()
    if book not in state["books"]:
        state["books"][book] = {"name": book, "description": "", "created": datetime.now().isoformat(), "positions": []}

    pos_id = f"POS-{uuid.uuid4().hex[:6].upper()}"
    position["id"] = pos_id
    position.setdefault("entry_date", datetime.now().strftime("%Y-%m-%d"))
    position.setdefault("multiplier", 100)
    position.setdefault("tags", [])
    position.setdefault("notes", "")

    state["books"][book]["positions"].append(position)

    # Log trade
    state["trade_history"].append({
        "id": f"TRD-{uuid.uuid4().hex[:6].upper()}",
        "timestamp": datetime.now().isoformat(),
        "action": "OPEN",
        "book": book,
        "position_id": pos_id,
        "details": deepcopy(position),
    })

    _save(state)
    return pos_id


def close_position(book: str, position_id: str, close_price: float = None):
    state = _load()
    positions = state["books"].get(book, {}).get("positions", [])
    remaining = []
    closed = None
    for pos in positions:
        if pos["id"] == position_id:
            closed = deepcopy(pos)
        else:
            remaining.append(pos)

    if closed:
        state["books"][book]["positions"] = remaining
        state["trade_history"].append({
            "id": f"TRD-{uuid.uuid4().hex[:6].upper()}",
            "timestamp": datetime.now().isoformat(),
            "action": "CLOSE",
            "book": book,
            "position_id": position_id,
            "close_price": close_price,
            "details": closed,
        })
        _save(state)


def update_position(book: str, position_id: str, updates: dict):
    state = _load()
    for pos in state["books"].get(book, {}).get("positions", []):
        if pos["id"] == position_id:
            pos.update(updates)
            break
    _save(state)


def get_risk_limits() -> dict:
    return _load().get("risk_limits", {})


def set_risk_limits(limits: dict):
    state = _load()
    state["risk_limits"].update(limits)
    _save(state)


def get_trade_history() -> List[dict]:
    return _load().get("trade_history", [])


def create_book(name: str, description: str = ""):
    state = _load()
    if name not in state["books"]:
        state["books"][name] = {
            "name": name, "description": description,
            "created": datetime.now().isoformat(), "positions": [],
        }
        _save(state)


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio Analytics
# ═══════════════════════════════════════════════════════════════════════════

def compute_book_risk(book: str, spot_prices: Dict[str, float], r: float = 0.05, q: float = 0.015) -> dict:
    """Compute full risk metrics for a book."""
    positions = get_positions(book)
    if not positions:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0,
                "vanna": 0, "volga": 0, "pnl": 0, "notional": 0,
                "by_ticker": {}, "by_expiry": {}, "by_type": {"call": {}, "put": {}}}

    totals = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0,
              "vanna": 0, "volga": 0, "pnl": 0, "notional": 0}
    by_ticker = {}
    by_expiry = {}

    for pos in positions:
        ticker = pos.get("ticker", "SPY")
        S = spot_prices.get(ticker, 100)
        K = pos["strike"]
        T = pos["expiry"]
        sig = pos["vol"]
        otype = pos["option_type"]
        qty = pos["quantity"]
        mult = pos.get("multiplier", 100)
        entry = pos.get("entry_price", 0)

        greeks = compute_all_greeks(S, K, T, r, q, sig, otype)
        pos_pnl = (greeks["price"] - entry) * qty * mult

        for g in ["delta", "gamma", "theta", "vega", "rho", "vanna", "volga"]:
            totals[g] += greeks[g] * qty * mult
        totals["pnl"] += pos_pnl
        totals["notional"] += abs(qty) * mult * S

        # By ticker
        if ticker not in by_ticker:
            by_ticker[ticker] = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "pnl": 0, "notional": 0, "positions": 0}
        by_ticker[ticker]["delta"] += greeks["delta"] * qty * mult
        by_ticker[ticker]["gamma"] += greeks["gamma"] * qty * mult
        by_ticker[ticker]["theta"] += greeks["theta"] * qty * mult
        by_ticker[ticker]["vega"] += greeks["vega"] * qty * mult
        by_ticker[ticker]["pnl"] += pos_pnl
        by_ticker[ticker]["notional"] += abs(qty) * mult * S
        by_ticker[ticker]["positions"] += 1

        # By expiry bucket
        if T <= 0.08:
            bucket = "0-1M"
        elif T <= 0.25:
            bucket = "1-3M"
        elif T <= 0.50:
            bucket = "3-6M"
        elif T <= 1.0:
            bucket = "6-12M"
        else:
            bucket = "12M+"
        if bucket not in by_expiry:
            by_expiry[bucket] = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "pnl": 0}
        by_expiry[bucket]["delta"] += greeks["delta"] * qty * mult
        by_expiry[bucket]["vega"] += greeks["vega"] * qty * mult
        by_expiry[bucket]["theta"] += greeks["theta"] * qty * mult
        by_expiry[bucket]["pnl"] += pos_pnl

    totals["by_ticker"] = by_ticker
    totals["by_expiry"] = by_expiry
    return totals


def pnl_attribution(positions: List[dict], S_old: Dict[str, float],
                     S_new: Dict[str, float], r: float = 0.05, q: float = 0.015,
                     dt: float = 1/365) -> dict:
    """
    Decompose P&L into delta, gamma, theta, vega, and higher-order contributions.

    Uses Taylor expansion:
      dP ≈ Δ·dS + ½Γ·dS² + Θ·dt + V·dσ + residual
    """
    attr = {"delta_pnl": 0, "gamma_pnl": 0, "theta_pnl": 0,
            "vega_pnl": 0, "cross_pnl": 0, "total_pnl": 0, "by_ticker": {}}

    for pos in positions:
        ticker = pos.get("ticker", "SPY")
        S0 = S_old.get(ticker, 100)
        S1 = S_new.get(ticker, S0)
        K = pos["strike"]
        T = pos["expiry"]
        sig = pos["vol"]
        otype = pos["option_type"]
        qty = pos["quantity"]
        mult = pos.get("multiplier", 100)

        dS = S1 - S0
        greeks = compute_all_greeks(S0, K, T, r, q, sig, otype)

        d_pnl = greeks["delta"] * dS * qty * mult
        g_pnl = 0.5 * greeks["gamma"] * dS ** 2 * qty * mult
        t_pnl = greeks["theta"] * qty * mult  # already per day
        # For vega, estimate vol change from spot change (leverage effect)
        dv = -0.1 * dS / S0  # rough: vol increases when spot drops
        v_pnl = greeks["vega"] * dv * 100 * qty * mult

        # Actual P&L
        p_old = bs_price(S0, K, T, r, q, sig, otype)
        p_new = bs_price(S1, K, max(T - dt, 1e-6), r, q, max(sig + dv, 0.01), otype)
        actual = (p_new - p_old) * qty * mult
        residual = actual - d_pnl - g_pnl - t_pnl - v_pnl

        attr["delta_pnl"] += d_pnl
        attr["gamma_pnl"] += g_pnl
        attr["theta_pnl"] += t_pnl
        attr["vega_pnl"] += v_pnl
        attr["cross_pnl"] += residual
        attr["total_pnl"] += actual

        if ticker not in attr["by_ticker"]:
            attr["by_ticker"][ticker] = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "cross": 0, "total": 0}
        attr["by_ticker"][ticker]["delta"] += d_pnl
        attr["by_ticker"][ticker]["gamma"] += g_pnl
        attr["by_ticker"][ticker]["theta"] += t_pnl
        attr["by_ticker"][ticker]["vega"] += v_pnl
        attr["by_ticker"][ticker]["cross"] += residual
        attr["by_ticker"][ticker]["total"] += actual

    return attr


def check_risk_limits(totals: dict) -> List[dict]:
    """Check portfolio against risk limits. Returns list of breaches."""
    limits = get_risk_limits()
    breaches = []

    checks = [
        ("max_delta", abs(totals.get("delta", 0)), "Delta"),
        ("max_gamma", abs(totals.get("gamma", 0)), "Gamma"),
        ("max_vega", abs(totals.get("vega", 0)), "Vega"),
        ("max_theta", -totals.get("theta", 0), "Theta (loss)"),
        ("max_notional", totals.get("notional", 0), "Notional"),
    ]

    for limit_key, current_val, label in checks:
        limit = limits.get(limit_key, float("inf"))
        if current_val > limit:
            breaches.append({
                "metric": label,
                "current": current_val,
                "limit": limit,
                "severity": "CRITICAL" if current_val > limit * 1.2 else "WARNING",
                "utilization": current_val / limit * 100,
            })
        elif current_val > limit * 0.8:
            breaches.append({
                "metric": label,
                "current": current_val,
                "limit": limit,
                "severity": "NEAR",
                "utilization": current_val / limit * 100,
            })

    # Single-name concentration
    by_ticker = totals.get("by_ticker", {})
    total_notional = totals.get("notional", 1)
    max_pct = limits.get("max_single_name_pct", 0.30)
    for ticker, data in by_ticker.items():
        tk_pct = data.get("notional", 0) / max(total_notional, 1)
        if tk_pct > max_pct:
            breaches.append({
                "metric": f"{ticker} Concentration",
                "current": tk_pct * 100,
                "limit": max_pct * 100,
                "severity": "WARNING",
                "utilization": tk_pct / max_pct * 100,
            })

    return breaches


# Initialize with defaults on first load
if not os.path.exists(PORTFOLIO_FILE):
    _save(_default_state())
