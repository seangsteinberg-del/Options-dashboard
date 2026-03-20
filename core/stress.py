"""
Stress Testing Engine
=====================
Named historical scenarios, custom shocks, and portfolio impact analysis.
Revalues positions under stressed market conditions using BS model.
"""

import numpy as np
from typing import Dict, List

from core.pricing import bs_price, compute_all_greeks


# ═══════════════════════════════════════════════════════════════════════════
# Named Historical Scenarios
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "2008 GFC": {
        "description": "Global Financial Crisis — Lehman collapse, credit freeze",
        "spot_shock": -0.38,
        "vol_shock": 1.50,      # +150% relative increase in IV
        "rate_shock": -0.02,    # -200bp
        "severity": "EXTREME",
        "color": "#ef4444",
    },
    "2020 COVID": {
        "description": "COVID-19 pandemic crash — fastest bear market in history",
        "spot_shock": -0.34,
        "vol_shock": 2.00,
        "rate_shock": -0.015,
        "severity": "EXTREME",
        "color": "#ef4444",
    },
    "2018 Volmageddon": {
        "description": "XIV blow-up — VIX spike from 11 to 50, short-vol unwind",
        "spot_shock": -0.10,
        "vol_shock": 3.00,
        "rate_shock": 0.0,
        "severity": "SEVERE",
        "color": "#f59e0b",
    },
    "Rate Shock +200bp": {
        "description": "Sudden 200bp rate hike — bond selloff, equity repricing",
        "spot_shock": -0.08,
        "vol_shock": 0.30,
        "rate_shock": 0.02,
        "severity": "MODERATE",
        "color": "#f59e0b",
    },
    "Flash Crash": {
        "description": "2010-style flash crash — rapid liquidity withdrawal",
        "spot_shock": -0.07,
        "vol_shock": 1.00,
        "rate_shock": 0.0,
        "severity": "MODERATE",
        "color": "#f59e0b",
    },
    "Bull Melt-Up": {
        "description": "Euphoric rally — vol compression, FOMO buying",
        "spot_shock": 0.15,
        "vol_shock": -0.30,
        "rate_shock": 0.005,
        "severity": "LOW",
        "color": "#10b981",
    },
    "Stagflation": {
        "description": "Rising inflation + slowing growth — 1970s replay",
        "spot_shock": -0.12,
        "vol_shock": 0.50,
        "rate_shock": 0.015,
        "severity": "SEVERE",
        "color": "#f59e0b",
    },
    "EM Contagion": {
        "description": "Emerging market crisis — capital flight, dollar strength",
        "spot_shock": -0.15,
        "vol_shock": 0.60,
        "rate_shock": -0.005,
        "severity": "MODERATE",
        "color": "#f59e0b",
    },
    "Taper Tantrum": {
        "description": "2013-style — central bank tightening surprise",
        "spot_shock": -0.06,
        "vol_shock": 0.40,
        "rate_shock": 0.01,
        "severity": "MODERATE",
        "color": "#f59e0b",
    },
    "Black Monday": {
        "description": "1987 crash — single-day 22% drop",
        "spot_shock": -0.22,
        "vol_shock": 2.50,
        "rate_shock": -0.01,
        "severity": "EXTREME",
        "color": "#ef4444",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Stress Test Runner
# ═══════════════════════════════════════════════════════════════════════════

def _apply_stress(positions: List[dict], spot_prices: Dict[str, float],
                  spot_shock: float, vol_shock: float, rate_shock: float,
                  r: float = 0.05, q: float = 0.015) -> dict:
    """
    Core stress test logic. Revalues every position under shocked conditions.

    vol_shock is a relative multiplier on each position's IV.
    spot_shock is a % move in the underlying.
    rate_shock is an absolute shift in the risk-free rate.
    """
    results = []
    total_base = 0.0
    total_stressed = 0.0

    r_stressed = max(r + rate_shock, 0.001)

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

        # Base valuation
        base_price = bs_price(S, K, T, r, q, sig, otype)
        base_greeks = compute_all_greeks(S, K, T, r, q, sig, otype)

        # Stressed valuation
        S_stressed = S * (1 + spot_shock)
        sig_stressed = max(sig * (1 + vol_shock), 0.01)
        stressed_price = bs_price(S_stressed, K, T, r_stressed, q, sig_stressed, otype)
        stressed_greeks = compute_all_greeks(S_stressed, K, T, r_stressed, q, sig_stressed, otype)

        base_val = base_price * qty * mult
        stressed_val = stressed_price * qty * mult
        pnl_impact = stressed_val - base_val

        total_base += base_val
        total_stressed += stressed_val

        results.append({
            "id": pos.get("id", ""),
            "ticker": ticker,
            "option_type": otype,
            "strike": K,
            "expiry": T,
            "quantity": qty,
            "base_price": base_price,
            "stressed_price": stressed_price,
            "pnl_impact": pnl_impact,
            "base_delta": base_greeks["delta"] * qty * mult,
            "stressed_delta": stressed_greeks["delta"] * qty * mult,
            "base_vega": base_greeks["vega"] * qty * mult,
            "stressed_vega": stressed_greeks["vega"] * qty * mult,
            "pct_change": (stressed_price / max(base_price, 0.001) - 1) * 100,
        })

    # Sort by absolute P&L impact (worst first)
    results.sort(key=lambda x: x["pnl_impact"])

    return {
        "positions": results,
        "total_base_value": total_base,
        "total_stressed_value": total_stressed,
        "total_pnl_impact": total_stressed - total_base,
        "worst_position": results[0] if results else None,
        "best_position": results[-1] if results else None,
        "spot_shock": spot_shock,
        "vol_shock": vol_shock,
        "rate_shock": rate_shock,
        "n_positions": len(results),
        "n_winners": sum(1 for r in results if r["pnl_impact"] > 0),
        "n_losers": sum(1 for r in results if r["pnl_impact"] < 0),
    }


def run_stress_test(positions: List[dict], spot_prices: Dict[str, float],
                    scenario_name: str, r: float = 0.05, q: float = 0.015) -> dict:
    """Run a named stress scenario against positions."""
    scenario = SCENARIOS.get(scenario_name)
    if not scenario:
        return {"error": f"Unknown scenario: {scenario_name}"}

    result = _apply_stress(
        positions, spot_prices,
        scenario["spot_shock"], scenario["vol_shock"], scenario["rate_shock"],
        r, q,
    )
    result["scenario_name"] = scenario_name
    result["description"] = scenario["description"]
    result["severity"] = scenario["severity"]
    result["color"] = scenario["color"]
    return result


def run_custom_stress(positions: List[dict], spot_prices: Dict[str, float],
                      spot_shock: float, vol_shock: float, rate_shock: float,
                      r: float = 0.05, q: float = 0.015) -> dict:
    """Run a custom stress scenario."""
    result = _apply_stress(positions, spot_prices, spot_shock, vol_shock, rate_shock, r, q)
    result["scenario_name"] = "Custom"
    result["description"] = f"Spot {spot_shock:+.0%}, Vol {vol_shock:+.0%}, Rate {rate_shock:+.0%}"
    result["severity"] = "CUSTOM"
    result["color"] = "#8b5cf6"
    return result


def compare_scenarios(positions: List[dict], spot_prices: Dict[str, float],
                      scenario_names: List[str] = None,
                      r: float = 0.05, q: float = 0.015) -> List[dict]:
    """Run multiple scenarios and return comparison summary."""
    if scenario_names is None:
        scenario_names = list(SCENARIOS.keys())

    comparisons = []
    for name in scenario_names:
        result = run_stress_test(positions, spot_prices, name, r, q)
        if "error" not in result:
            comparisons.append({
                "scenario": name,
                "severity": result["severity"],
                "color": result["color"],
                "spot_shock": result["spot_shock"] * 100,
                "vol_shock": result["vol_shock"] * 100,
                "rate_shock": result["rate_shock"] * 100,
                "total_pnl": result["total_pnl_impact"],
                "worst_position_pnl": result["worst_position"]["pnl_impact"] if result["worst_position"] else 0,
                "n_losers": result["n_losers"],
                "description": result["description"],
            })

    # Sort by P&L impact (worst first)
    comparisons.sort(key=lambda x: x["total_pnl"])
    return comparisons
