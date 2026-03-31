"""
FX Stress Testing Engine
=========================
15 named FX-specific stress scenarios based on historical events,
custom scenario building, reverse stress testing, and scenario
sensitivity analysis.

Uses Garman-Kohlhagen pricing for FX option revaluation under
stressed market parameters.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, List, Optional


# =====================================================================
# Major FX Pairs (used as the universe for default shocks)
# =====================================================================

MAJOR_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD",
    "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "USDCNH",
    "USDZAR", "USDMXN", "USDTRY", "USDBRL", "USDNOK", "USDSEK",
    "AUDJPY", "GBPCHF", "EURAUD", "EURNZD", "USDSGD", "USDINR",
]


# =====================================================================
# Garman-Kohlhagen Pricing
# =====================================================================

def _gk_price(S, K, T, r_d, r_f, sigma, cp):
    """
    Garman-Kohlhagen FX option price.

    Parameters
    ----------
    S : float   -- spot exchange rate
    K : float   -- strike
    T : float   -- time to expiry in years
    r_d : float -- domestic risk-free rate
    r_f : float -- foreign risk-free rate
    sigma : float -- implied volatility
    cp : str    -- 'call' or 'put'

    Returns
    -------
    float -- option premium in domestic currency units per 1 unit of foreign notional
    """
    if T <= 0:
        return max(S - K, 0.0) if cp == "call" else max(K - S, 0.0)
    sigma = max(sigma, 1e-6)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if cp == "call":
        return S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
    return K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)


def _gk_greeks(S, K, T, r_d, r_f, sigma, cp):
    """Full Greeks under Garman-Kohlhagen."""
    if T <= 0:
        price = max(S - K, 0.0) if cp == "call" else max(K - S, 0.0)
        d = (1.0 if S > K else 0.0) if cp == "call" else (-1.0 if S < K else 0.0)
        return {"price": price, "delta": d, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "rho_d": 0.0, "rho_f": 0.0}
    sigma = max(sigma, 1e-6)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df_d = np.exp(-r_d * T)
    df_f = np.exp(-r_f * T)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    npd1 = norm.pdf(d1)

    if cp == "call":
        price = S * df_f * nd1 - K * df_d * nd2
        delta_val = df_f * nd1
        theta_val = (-(S * df_f * npd1 * sigma) / (2 * sqrtT)
                     - r_d * K * df_d * nd2 + r_f * S * df_f * nd1) / 365.0
        rho_d_val = K * T * df_d * nd2 / 10000.0
        rho_f_val = -S * T * df_f * nd1 / 10000.0
    else:
        price = K * df_d * norm.cdf(-d2) - S * df_f * norm.cdf(-d1)
        delta_val = df_f * (nd1 - 1)
        theta_val = (-(S * df_f * npd1 * sigma) / (2 * sqrtT)
                     + r_d * K * df_d * norm.cdf(-d2) - r_f * S * df_f * norm.cdf(-d1)) / 365.0
        rho_d_val = -K * T * df_d * norm.cdf(-d2) / 10000.0
        rho_f_val = S * T * df_f * norm.cdf(-d1) / 10000.0

    gamma_val = df_f * npd1 / (S * sigma * sqrtT)
    vega_val = S * df_f * npd1 * sqrtT / 100.0

    return {"price": price, "delta": delta_val, "gamma": gamma_val,
            "vega": vega_val, "theta": theta_val, "rho_d": rho_d_val,
            "rho_f": rho_f_val}


# =====================================================================
# 15 Named FX Stress Scenarios
# =====================================================================

def _default(spot, vol, rate):
    """Helper to build a shock dict."""
    return {"spot": spot, "vol_mult": vol, "rate": rate}


FX_STRESS_SCENARIOS = {
    "SNB_CHF_UNPEGGING": {
        "name": "SNB CHF Unpegging (2015)",
        "description": "Swiss National Bank removes EUR/CHF floor. EURCHF crashes 30%, massive CHF appreciation.",
        "severity": "EXTREME",
        "date": "2015-01-15",
        "shocks": {
            "EURCHF": _default(-0.30, 4.0, 0),
            "USDCHF": _default(-0.25, 3.5, 0),
            "GBPCHF": _default(-0.22, 3.0, 0),
            "EURUSD": _default(-0.05, 1.8, 0),
            "USDJPY": _default(-0.03, 1.5, 0),
            "GBPUSD": _default(-0.02, 1.5, 0),
            "AUDUSD": _default(-0.03, 1.4, 0),
            "USDCAD": _default(0.02, 1.3, 0),
            "NZDUSD": _default(-0.03, 1.4, 0),
            "EURGBP": _default(-0.03, 1.6, 0),
            "EURJPY": _default(-0.08, 2.0, 0),
            "GBPJPY": _default(-0.05, 1.8, 0),
        },
        "default_shock": _default(-0.02, 1.5, 0),
    },

    "BREXIT_VOTE": {
        "name": "Brexit Vote (2016)",
        "description": "UK votes to leave the EU. GBP flash-crashes 8%, safe-haven bid in JPY and CHF.",
        "severity": "SEVERE",
        "date": "2016-06-24",
        "shocks": {
            "GBPUSD": _default(-0.08, 3.0, -25),
            "EURGBP": _default(0.06, 2.5, 0),
            "GBPJPY": _default(-0.10, 3.0, -25),
            "GBPCHF": _default(-0.09, 2.8, -25),
            "EURUSD": _default(-0.03, 1.8, 0),
            "USDJPY": _default(-0.03, 1.5, 0),
            "USDCHF": _default(-0.02, 1.4, 0),
            "AUDUSD": _default(-0.02, 1.5, 0),
            "USDCAD": _default(0.01, 1.3, 0),
            "NZDUSD": _default(-0.02, 1.4, 0),
            "EURJPY": _default(-0.05, 1.8, 0),
            "EURCHF": _default(-0.02, 1.4, 0),
        },
        "default_shock": _default(-0.01, 1.3, 0),
    },

    "COVID_LIQUIDITY": {
        "name": "COVID-19 Liquidity Crisis (2020)",
        "description": "Pandemic triggers EM currency collapse, dollar liquidity squeeze, vol explosion across all pairs.",
        "severity": "EXTREME",
        "date": "2020-03-19",
        "shocks": {
            "USDZAR": _default(0.25, 3.5, 100),
            "USDMXN": _default(0.20, 3.5, 75),
            "USDTRY": _default(0.18, 3.0, 150),
            "USDBRL": _default(0.30, 3.5, 200),
            "USDINR": _default(0.08, 2.5, 50),
            "USDCNH": _default(0.03, 2.0, 0),
            "AUDUSD": _default(-0.05, 2.5, -50),
            "NZDUSD": _default(-0.06, 2.5, -50),
            "USDCAD": _default(0.05, 2.5, -50),
            "USDNOK": _default(0.15, 3.0, -25),
            "EURUSD": _default(-0.03, 2.0, -25),
            "GBPUSD": _default(-0.05, 2.2, -50),
            "USDJPY": _default(-0.03, 2.0, -25),
            "USDCHF": _default(-0.02, 1.8, -25),
        },
        "default_shock": _default(0.03, 2.0, 0),
    },

    "BOJ_INTERVENTION": {
        "name": "BOJ FX Intervention",
        "description": "Bank of Japan intervenes to sell USD/JPY. Sharp yen rally, JPY crosses tumble.",
        "severity": "SEVERE",
        "date": "2022-09-22",
        "shocks": {
            "USDJPY": _default(-0.06, 2.5, -15),
            "EURJPY": _default(-0.05, 2.2, 0),
            "GBPJPY": _default(-0.05, 2.2, 0),
            "AUDJPY": _default(-0.04, 2.0, 0),
            "EURUSD": _default(0.01, 1.3, 0),
            "GBPUSD": _default(0.01, 1.3, 0),
            "AUDUSD": _default(0.01, 1.2, 0),
            "USDCHF": _default(-0.02, 1.3, 0),
            "USDCAD": _default(-0.01, 1.2, 0),
            "NZDUSD": _default(0.01, 1.2, 0),
            "EURGBP": _default(0.00, 1.1, 0),
            "EURCHF": _default(-0.01, 1.2, 0),
        },
        "default_shock": _default(0.00, 1.2, 0),
    },

    "FED_EMERGENCY_CUT": {
        "name": "Fed Emergency Rate Cut",
        "description": "Fed delivers inter-meeting 50bp cut. USD weakens broadly, EM rallies, vol moderately elevated.",
        "severity": "MODERATE",
        "date": "2020-03-03",
        "shocks": {
            "EURUSD": _default(0.03, 1.5, -50),
            "GBPUSD": _default(0.02, 1.5, -50),
            "USDJPY": _default(-0.04, 1.8, -50),
            "USDCHF": _default(-0.03, 1.5, -50),
            "AUDUSD": _default(0.02, 1.4, -50),
            "NZDUSD": _default(0.02, 1.4, -50),
            "USDCAD": _default(-0.02, 1.4, -50),
            "USDZAR": _default(-0.03, 1.5, -50),
            "USDMXN": _default(-0.03, 1.5, -50),
            "USDCNH": _default(-0.01, 1.3, -50),
            "EURGBP": _default(0.01, 1.2, 0),
            "EURJPY": _default(-0.01, 1.3, -50),
        },
        "default_shock": _default(-0.02, 1.3, -50),
    },

    "EM_CONTAGION_97": {
        "name": "EM Contagion (1997 Asia / 1998 Russia)",
        "description": "Emerging-market crisis contagion. EM currencies collapse, G10 safe havens bid.",
        "severity": "EXTREME",
        "date": "1997-07-02",
        "shocks": {
            "USDZAR": _default(0.20, 2.5, 200),
            "USDMXN": _default(0.15, 2.5, 150),
            "USDTRY": _default(0.25, 3.0, 300),
            "USDBRL": _default(0.20, 2.5, 250),
            "USDINR": _default(0.10, 2.0, 100),
            "USDCNH": _default(0.05, 1.8, 50),
            "USDSGD": _default(0.08, 2.0, 75),
            "USDJPY": _default(-0.05, 2.0, -10),
            "USDCHF": _default(-0.04, 1.8, -10),
            "EURUSD": _default(0.02, 1.5, 0),
            "GBPUSD": _default(0.01, 1.4, 0),
            "AUDUSD": _default(-0.08, 2.0, 25),
            "NZDUSD": _default(-0.07, 2.0, 25),
            "USDCAD": _default(0.03, 1.5, 10),
        },
        "default_shock": _default(0.05, 1.8, 50),
    },

    "CARRY_UNWIND_07": {
        "name": "Carry Trade Unwind (2007-08)",
        "description": "Massive JPY carry unwind. High-yielders collapse, JPY rockets, correlation spike.",
        "severity": "SEVERE",
        "date": "2007-08-16",
        "shocks": {
            "USDJPY": _default(-0.08, 2.5, -25),
            "EURJPY": _default(-0.10, 2.5, -25),
            "GBPJPY": _default(-0.12, 2.8, -25),
            "AUDJPY": _default(-0.15, 3.0, -25),
            "NZDUSD": _default(-0.10, 2.5, 0),
            "AUDUSD": _default(-0.10, 2.5, 0),
            "USDZAR": _default(0.15, 2.5, 100),
            "USDMXN": _default(0.10, 2.0, 50),
            "USDTRY": _default(0.15, 2.5, 150),
            "USDBRL": _default(0.12, 2.5, 100),
            "EURUSD": _default(0.02, 1.5, 0),
            "GBPUSD": _default(-0.02, 1.5, 0),
            "USDCHF": _default(-0.05, 2.0, -10),
            "USDCAD": _default(0.03, 1.5, 10),
        },
        "default_shock": _default(0.02, 1.8, 25),
    },

    "CORRELATION_BREAKDOWN": {
        "name": "Correlation Breakdown",
        "description": "Normal FX correlations flip. Historically correlated pairs diverge, hedges fail.",
        "severity": "SEVERE",
        "date": "2015-08-24",
        "shocks": {
            "EURUSD": _default(0.04, 2.0, 0),
            "GBPUSD": _default(-0.03, 2.0, 0),
            "USDJPY": _default(0.03, 2.0, 0),
            "USDCHF": _default(0.03, 2.0, 0),
            "AUDUSD": _default(0.04, 2.0, 0),
            "NZDUSD": _default(-0.04, 2.0, 0),
            "USDCAD": _default(-0.03, 2.0, 0),
            "EURGBP": _default(0.06, 2.2, 0),
            "EURJPY": _default(0.06, 2.2, 0),
            "GBPJPY": _default(-0.05, 2.2, 0),
            "EURCHF": _default(0.05, 2.0, 0),
            "AUDJPY": _default(0.05, 2.0, 0),
        },
        "default_shock": _default(0.02, 2.0, 0),
    },

    "USD_FLASH_CRASH": {
        "name": "USD Flash Crash",
        "description": "Sudden 5% USD devaluation across the board. Liquidity vanishes, vol triples.",
        "severity": "EXTREME",
        "date": "2024-01-01",
        "shocks": {
            "EURUSD": _default(0.05, 3.0, -25),
            "GBPUSD": _default(0.05, 3.0, -25),
            "AUDUSD": _default(0.05, 3.0, -25),
            "NZDUSD": _default(0.05, 3.0, -25),
            "USDJPY": _default(-0.05, 3.0, -25),
            "USDCHF": _default(-0.05, 3.0, -25),
            "USDCAD": _default(-0.05, 3.0, -25),
            "USDCNH": _default(-0.05, 3.0, -25),
            "USDZAR": _default(-0.05, 3.0, -25),
            "USDMXN": _default(-0.05, 3.0, -25),
            "USDTRY": _default(-0.05, 3.0, -25),
            "USDBRL": _default(-0.05, 3.0, -25),
            "USDNOK": _default(-0.05, 3.0, -25),
            "USDSEK": _default(-0.05, 3.0, -25),
            "EURGBP": _default(0.00, 1.5, 0),
            "EURJPY": _default(-0.01, 1.5, 0),
            "GBPJPY": _default(-0.01, 1.5, 0),
        },
        "default_shock": _default(-0.05, 3.0, -25),
    },

    "GEOPOLITICAL_SHOCK": {
        "name": "Geopolitical Shock / Risk-Off",
        "description": "Major geopolitical event. Classic risk-off: JPY/CHF bid, EM sold aggressively.",
        "severity": "SEVERE",
        "date": "2022-02-24",
        "shocks": {
            "USDJPY": _default(-0.04, 2.0, -15),
            "USDCHF": _default(-0.03, 1.8, -15),
            "EURUSD": _default(-0.02, 1.5, 0),
            "GBPUSD": _default(-0.02, 1.5, 0),
            "AUDUSD": _default(-0.03, 1.5, 0),
            "NZDUSD": _default(-0.03, 1.5, 0),
            "USDCAD": _default(0.02, 1.4, 0),
            "USDZAR": _default(0.10, 2.2, 50),
            "USDMXN": _default(0.08, 2.0, 40),
            "USDTRY": _default(0.12, 2.5, 75),
            "USDBRL": _default(0.10, 2.2, 60),
            "USDCNH": _default(0.02, 1.5, 10),
            "USDINR": _default(0.05, 1.8, 30),
            "EURJPY": _default(-0.05, 1.8, -15),
            "GBPJPY": _default(-0.05, 1.8, -15),
        },
        "default_shock": _default(0.03, 1.5, 10),
    },

    "HAWKISH_FED": {
        "name": "Hawkish Fed Surprise",
        "description": "Fed signals aggressive tightening. USD strengthens 3-5% broadly, EM hammered.",
        "severity": "MODERATE",
        "date": "2022-06-15",
        "shocks": {
            "EURUSD": _default(-0.04, 1.5, 50),
            "GBPUSD": _default(-0.03, 1.5, 50),
            "USDJPY": _default(0.03, 1.5, 50),
            "USDCHF": _default(0.02, 1.3, 50),
            "AUDUSD": _default(-0.04, 1.5, 50),
            "NZDUSD": _default(-0.04, 1.5, 50),
            "USDCAD": _default(0.02, 1.3, 50),
            "USDZAR": _default(0.08, 2.0, 50),
            "USDMXN": _default(0.06, 1.8, 50),
            "USDTRY": _default(0.10, 2.0, 50),
            "USDBRL": _default(0.07, 1.8, 50),
            "USDCNH": _default(0.02, 1.5, 50),
            "EURGBP": _default(-0.01, 1.2, 0),
            "EURJPY": _default(-0.01, 1.3, 50),
        },
        "default_shock": _default(0.03, 1.5, 50),
    },

    "CHINA_DEVAL": {
        "name": "China Devaluation",
        "description": "PBOC allows sharp CNH depreciation. Asian EM sells off, AUD hit hard, safe-haven bid.",
        "severity": "SEVERE",
        "date": "2015-08-11",
        "shocks": {
            "USDCNH": _default(0.05, 2.5, 25),
            "USDSGD": _default(0.04, 2.0, 15),
            "USDINR": _default(0.05, 2.0, 25),
            "AUDUSD": _default(-0.08, 2.5, 0),
            "NZDUSD": _default(-0.06, 2.2, 0),
            "USDJPY": _default(-0.03, 1.8, -10),
            "USDCHF": _default(-0.02, 1.5, -10),
            "EURUSD": _default(0.01, 1.3, 0),
            "GBPUSD": _default(-0.01, 1.3, 0),
            "USDCAD": _default(0.03, 1.5, 0),
            "USDZAR": _default(0.07, 2.0, 30),
            "USDMXN": _default(0.05, 1.8, 20),
            "EURJPY": _default(-0.02, 1.5, -10),
            "AUDJPY": _default(-0.10, 2.5, 0),
        },
        "default_shock": _default(0.02, 1.5, 10),
    },

    "EUROPEAN_DEBT_CRISIS": {
        "name": "European Debt Crisis (2011-12)",
        "description": "Sovereign debt contagion in periphery Europe. EUR collapses, EUR crosses follow.",
        "severity": "SEVERE",
        "date": "2011-11-01",
        "shocks": {
            "EURUSD": _default(-0.10, 2.5, 25),
            "EURGBP": _default(-0.07, 2.2, 25),
            "EURJPY": _default(-0.12, 2.5, 25),
            "EURCHF": _default(-0.08, 2.5, 25),
            "EURAUD": _default(-0.06, 2.0, 25),
            "EURNZD": _default(-0.05, 2.0, 25),
            "GBPUSD": _default(-0.03, 1.5, 0),
            "USDJPY": _default(-0.02, 1.3, -10),
            "USDCHF": _default(-0.03, 1.5, -10),
            "AUDUSD": _default(-0.02, 1.5, 0),
            "NZDUSD": _default(-0.02, 1.5, 0),
            "USDCAD": _default(0.01, 1.3, 0),
            "USDZAR": _default(0.05, 1.8, 30),
            "USDMXN": _default(0.04, 1.6, 20),
        },
        "default_shock": _default(-0.02, 1.5, 10),
    },

    "OIL_SHOCK": {
        "name": "Oil Price Shock",
        "description": "Crude collapses 30%. Commodity FX (CAD, NOK) hammered, USD strengthens.",
        "severity": "MODERATE",
        "date": "2014-11-27",
        "shocks": {
            "USDCAD": _default(0.05, 2.0, -15),
            "USDNOK": _default(0.08, 2.5, -15),
            "USDZAR": _default(0.06, 2.0, 25),
            "USDMXN": _default(0.05, 1.8, 20),
            "USDBRL": _default(0.05, 1.8, 20),
            "AUDUSD": _default(-0.04, 1.8, 0),
            "NZDUSD": _default(-0.03, 1.5, 0),
            "EURUSD": _default(-0.01, 1.3, 0),
            "GBPUSD": _default(-0.01, 1.3, 0),
            "USDJPY": _default(0.01, 1.2, 0),
            "USDCHF": _default(0.01, 1.2, 0),
            "EURGBP": _default(0.00, 1.1, 0),
            "EURJPY": _default(-0.01, 1.2, 0),
            "USDCNH": _default(0.01, 1.2, 0),
        },
        "default_shock": _default(0.01, 1.3, 0),
    },

    "VOLMAGEDDON_FX": {
        "name": "FX Volmageddon (Pure Vega Stress)",
        "description": "Spot barely moves but implied vol triples. Short-vol books destroyed, long-gamma wins.",
        "severity": "SEVERE",
        "date": "2018-02-05",
        "shocks": {
            "EURUSD": _default(0.00, 3.0, 0),
            "GBPUSD": _default(0.00, 3.0, 0),
            "USDJPY": _default(0.00, 3.0, 0),
            "USDCHF": _default(0.00, 3.0, 0),
            "AUDUSD": _default(0.00, 3.0, 0),
            "NZDUSD": _default(0.00, 3.0, 0),
            "USDCAD": _default(0.00, 3.0, 0),
            "EURGBP": _default(0.00, 3.0, 0),
            "EURJPY": _default(0.00, 3.0, 0),
            "GBPJPY": _default(0.00, 3.0, 0),
            "EURCHF": _default(0.00, 3.0, 0),
            "USDCNH": _default(0.00, 2.5, 0),
            "USDZAR": _default(0.00, 3.0, 0),
            "USDMXN": _default(0.00, 3.0, 0),
            "USDTRY": _default(0.00, 3.0, 0),
            "USDBRL": _default(0.00, 3.0, 0),
        },
        "default_shock": _default(0.00, 3.0, 0),
    },

    # -----------------------------------------------------------------
    # Uniform stress scenarios (apply same shock to all pairs)
    # -----------------------------------------------------------------

    "VOL_SPIKE_50": {
        "name": "Vol Spike (+50%)",
        "description": "Implied volatility jumps 50% across all pairs. Tests short-vega exposure.",
        "severity": "MODERATE",
        "date": "",
        "shocks": {},
        "default_shock": _default(0.0, 1.5, 0),
    },

    "VOL_SPIKE_100": {
        "name": "Vol Spike (+100%)",
        "description": "Implied volatility doubles across all pairs. Extreme vega stress.",
        "severity": "SEVERE",
        "date": "",
        "shocks": {},
        "default_shock": _default(0.0, 2.0, 0),
    },

    "VOL_COLLAPSE_30": {
        "name": "Vol Collapse (-30%)",
        "description": "Implied volatility drops 30% across all pairs. Tests long-vega exposure.",
        "severity": "MODERATE",
        "date": "",
        "shocks": {},
        "default_shock": _default(0.0, 0.7, 0),
    },

    "RISK_OFF_JPY": {
        "name": "Risk-Off (JPY rally)",
        "description": "Classic risk-off move: spot down 5%, vol nearly doubles, rates fall 50bp.",
        "severity": "SEVERE",
        "date": "",
        "shocks": {
            "USDJPY": _default(-0.08, 2.2, -50),
            "EURJPY": _default(-0.07, 2.0, -50),
            "GBPJPY": _default(-0.07, 2.0, -50),
            "AUDJPY": _default(-0.06, 2.0, -50),
            "AUDUSD": _default(-0.04, 1.8, -50),
            "NZDUSD": _default(-0.04, 1.8, -50),
        },
        "default_shock": _default(-0.05, 1.8, -50),
    },

    "RISK_ON_EM": {
        "name": "Risk-On (EM rally)",
        "description": "EM rally: spot up 3%, vol compresses, rates rise 25bp on growth optimism.",
        "severity": "MODERATE",
        "date": "",
        "shocks": {
            "USDZAR": _default(-0.05, 0.7, 25),
            "USDMXN": _default(-0.04, 0.7, 25),
            "USDTRY": _default(-0.03, 0.8, 25),
            "USDBRL": _default(-0.04, 0.7, 25),
            "USDCNH": _default(-0.02, 0.8, 25),
            "AUDUSD": _default(0.03, 0.8, 25),
            "NZDUSD": _default(0.03, 0.8, 25),
        },
        "default_shock": _default(0.03, 0.8, 25),
    },

    "RATES_UP_100": {
        "name": "Rates +100bp",
        "description": "Parallel +100bp rate shock across all pairs. Pure rho stress test.",
        "severity": "MODERATE",
        "date": "",
        "shocks": {},
        "default_shock": _default(0.0, 1.0, 100),
    },

    "RATES_DOWN_100": {
        "name": "Rates -100bp",
        "description": "Parallel -100bp rate shock across all pairs. Pure rho stress test.",
        "severity": "MODERATE",
        "date": "",
        "shocks": {},
        "default_shock": _default(0.0, 1.0, -100),
    },

    "SPOT_UP_5": {
        "name": "Spot +5%",
        "description": "Uniform 5% spot appreciation. Tests delta and gamma exposure.",
        "severity": "MODERATE",
        "date": "",
        "shocks": {},
        "default_shock": _default(0.05, 1.0, 0),
    },

    "SPOT_DOWN_5": {
        "name": "Spot -5%",
        "description": "Uniform 5% spot depreciation. Tests delta and gamma exposure.",
        "severity": "MODERATE",
        "date": "",
        "shocks": {},
        "default_shock": _default(-0.05, 1.0, 0),
    },

    "SPOT_UP_10": {
        "name": "Spot +10%",
        "description": "Uniform 10% spot appreciation. Large directional stress.",
        "severity": "SEVERE",
        "date": "",
        "shocks": {},
        "default_shock": _default(0.10, 1.0, 0),
    },

    "SPOT_DOWN_10": {
        "name": "Spot -10%",
        "description": "Uniform 10% spot depreciation. Large directional stress.",
        "severity": "SEVERE",
        "date": "",
        "shocks": {},
        "default_shock": _default(-0.10, 1.0, 0),
    },

    "TAIL_RISK": {
        "name": "Tail Risk",
        "description": "Combined tail event: spot -15%, vol 3.5x, rates -150bp. Worst-case scenario.",
        "severity": "EXTREME",
        "date": "",
        "shocks": {},
        "default_shock": _default(-0.15, 3.5, -150),
    },
}


# =====================================================================
# Scenario Query Functions
# =====================================================================

def get_scenarios():
    """Return list of all scenario names and descriptions."""
    return [
        {
            "name": key,
            "display_name": sc["name"],
            "description": sc["description"],
            "severity": sc["severity"],
            "date": sc.get("date", ""),
        }
        for key, sc in FX_STRESS_SCENARIOS.items()
    ]


def get_scenario(name):
    """Return full scenario dict by name. Returns None if not found."""
    return FX_STRESS_SCENARIOS.get(name)


def get_pair_shock(scenario_name, pair):
    """
    Get shock vector for a specific pair in a scenario.
    Falls back to the scenario's default_shock if the pair is not
    explicitly defined.

    Returns dict: {"spot": float, "vol_mult": float, "rate": float}
    """
    sc = FX_STRESS_SCENARIOS.get(scenario_name)
    if sc is None:
        return _default(0.0, 1.0, 0)
    return sc["shocks"].get(pair, sc.get("default_shock", _default(0.0, 1.0, 0)))


# =====================================================================
# Shock Application Helpers
# =====================================================================

def apply_spot_shock(current_spot, shock_pct):
    """Apply percentage spot shock: S_new = S * (1 + shock_pct)."""
    return current_spot * (1.0 + shock_pct)


def apply_vol_shock(current_vol, vol_multiplier):
    """Apply vol multiplier: vol_new = vol * multiplier. Floors at 0.1%."""
    return max(current_vol * vol_multiplier, 0.001)


def apply_rate_shock(current_rate, shock_bps):
    """Apply rate shock in basis points: rate_new = rate + shock_bps / 10000."""
    return current_rate + shock_bps / 10000.0


# =====================================================================
# Single Position Stress Test
# =====================================================================

def _get_shock_from_scenario(scenario_dict, pair):
    """Get shock vector for a pair from a scenario dict (not the global registry).

    Falls back to the scenario's default_shock if the pair is not in shocks.
    """
    return scenario_dict["shocks"].get(pair,
                                        scenario_dict.get("default_shock", _default(0.0, 1.0, 0)))


def stress_single_position(position, spot, r_d, r_f, vol, scenario_name):
    """
    Stress test a single FX option position.

    Parameters
    ----------
    position : dict
        Must contain: pair, strike, expiry (years), option_type ('call'/'put'),
        notional, and optionally entry_price.
    spot : float   -- current spot rate
    r_d : float    -- domestic risk-free rate
    r_f : float    -- foreign risk-free rate
    vol : float    -- current implied volatility (decimal)
    scenario_name : str  -- key into FX_STRESS_SCENARIOS

    Returns
    -------
    dict with base_value, stressed_value, pnl_impact, pnl_pct, stressed_greeks
    """
    pair = position.get("pair", "EURUSD")
    K = position["strike"]
    # Expiry can be a date string or float years — normalise to float
    _exp = position["expiry"]
    if isinstance(_exp, str):
        from datetime import datetime
        try:
            T = max((datetime.strptime(_exp, "%Y-%m-%d") - datetime.now()).days / 365.0, 1e-6)
        except ValueError:
            T = 0.25  # fallback to 3M
    else:
        T = float(_exp) if _exp else 0.25
    cp = position.get("option_type", "call")
    notional = position.get("notional", 1_000_000)
    direction = position.get("direction", "buy")
    dir_sign = 1.0 if direction == "buy" else -1.0

    shock = get_pair_shock(scenario_name, pair)

    S_stressed = apply_spot_shock(spot, shock["spot"])
    vol_stressed = apply_vol_shock(vol, shock["vol_mult"])
    # Shock both domestic and foreign rates (rate shock applies to the differential)
    # Convention: positive rate shock strengthens domestic currency
    r_d_stressed = apply_rate_shock(r_d, shock["rate"])
    r_f_stressed = apply_rate_shock(r_f, -shock["rate"] * 0.3)  # partial offset on foreign

    base_greeks = _gk_greeks(spot, K, T, r_d, r_f, vol, cp)
    stressed_greeks = _gk_greeks(S_stressed, K, T, r_d_stressed, r_f_stressed, vol_stressed, cp)

    base_value = base_greeks["price"] * notional * dir_sign
    stressed_value = stressed_greeks["price"] * notional * dir_sign
    pnl_impact = stressed_value - base_value
    pnl_pct = pnl_impact / max(abs(base_value), 1e-10)

    return {
        "pair": pair,
        "strike": K,
        "expiry": T,
        "option_type": cp,
        "notional": notional,
        "base_value": base_value,
        "stressed_value": stressed_value,
        "pnl_impact": pnl_impact,
        "pnl_pct": pnl_pct,
        "spot_base": spot,
        "spot_stressed": S_stressed,
        "vol_base": vol,
        "vol_stressed": vol_stressed,
        "stressed_greeks": {
            k: v * notional * dir_sign
            for k, v in stressed_greeks.items()
        },
    }


# =====================================================================
# Portfolio Stress Test
# =====================================================================

def _stress_portfolio_with_scenario(positions, spots, rates, vol_surfaces, scenario_dict):
    """Stress test using a scenario dict directly (avoids mutating global state).

    scenario_dict must have keys: shocks, default_shock, and optionally name/severity.
    """
    by_position = []
    by_pair = {}
    total_base = 0.0
    total_stressed = 0.0

    for pos in positions:
        pair = pos.get("pair", "EURUSD")
        spot = spots.get(pair, 1.0)
        rate_info = rates.get(pair, {"r_d": 0.03, "r_f": 0.02})
        if isinstance(rate_info, dict):
            r_d = rate_info.get("r_d", 0.03)
            r_f = rate_info.get("r_f", 0.02)
        elif isinstance(rate_info, (tuple, list)) and len(rate_info) >= 2:
            r_d, r_f = rate_info[0], rate_info[1]
        else:
            r_d, r_f = 0.03, 0.02

        vol_entry = vol_surfaces.get(pair, 0.10)
        if callable(vol_entry):
            vol = vol_entry(pos["strike"], pos["expiry"])
        else:
            vol = vol_entry

        K = pos["strike"]
        _exp = pos["expiry"]
        if isinstance(_exp, str):
            from datetime import datetime
            try:
                T = max((datetime.strptime(_exp, "%Y-%m-%d") - datetime.now()).days / 365.0, 1e-6)
            except ValueError:
                T = 0.25
        else:
            T = float(_exp) if _exp else 0.25

        cp = pos.get("option_type", "call")
        notional = pos.get("notional", 1_000_000)
        direction = pos.get("direction", "buy")
        dir_sign = 1.0 if direction == "buy" else -1.0

        shock = _get_shock_from_scenario(scenario_dict, pair)

        S_stressed = apply_spot_shock(spot, shock["spot"])
        vol_stressed = apply_vol_shock(vol, shock["vol_mult"])
        r_d_stressed = apply_rate_shock(r_d, shock["rate"])
        r_f_stressed = apply_rate_shock(r_f, -shock["rate"] * 0.3)

        base_greeks = _gk_greeks(spot, K, T, r_d, r_f, vol, cp)
        stressed_greeks = _gk_greeks(S_stressed, K, T, r_d_stressed, r_f_stressed, vol_stressed, cp)

        base_value = base_greeks["price"] * notional * dir_sign
        stressed_value = stressed_greeks["price"] * notional * dir_sign
        pnl_impact = stressed_value - base_value
        pnl_pct = pnl_impact / max(abs(base_value), 1e-10)

        result = {
            "pair": pair,
            "strike": K,
            "expiry": T,
            "option_type": cp,
            "notional": notional,
            "base_value": base_value,
            "stressed_value": stressed_value,
            "pnl_impact": pnl_impact,
            "pnl_pct": pnl_pct,
            "spot_base": spot,
            "spot_stressed": S_stressed,
            "vol_base": vol,
            "vol_stressed": vol_stressed,
            "stressed_greeks": {
                k: v * notional * dir_sign
                for k, v in stressed_greeks.items()
            },
        }
        by_position.append(result)

        total_base += result["base_value"]
        total_stressed += result["stressed_value"]

        if pair not in by_pair:
            by_pair[pair] = 0.0
        by_pair[pair] += result["pnl_impact"]

    total_pnl = total_stressed - total_base
    worst_positions = sorted(by_position, key=lambda x: x["pnl_impact"])

    limit_breaches = []
    if total_pnl < -1_000_000:
        limit_breaches.append({
            "limit": "MAX_STRESS_LOSS",
            "threshold": -1_000_000,
            "actual": total_pnl,
            "severity": "CRITICAL" if total_pnl < -5_000_000 else "WARNING",
        })
    for pair, pair_pnl in by_pair.items():
        if abs(pair_pnl) > 500_000:
            limit_breaches.append({
                "limit": f"CONCENTRATION_{pair}",
                "threshold": 500_000,
                "actual": pair_pnl,
                "severity": "WARNING",
            })

    return {
        "scenario_name": scenario_dict.get("name", "Custom"),
        "scenario_display": scenario_dict.get("name", "Custom"),
        "severity": scenario_dict.get("severity", "CUSTOM"),
        "total_base_value": total_base,
        "total_stressed_value": total_stressed,
        "total_pnl": total_pnl,
        "by_pair": by_pair,
        "by_position": by_position,
        "worst_positions": worst_positions[:10],
        "limit_breaches": limit_breaches,
        "n_positions": len(positions),
        "n_losers": sum(1 for p in by_position if p["pnl_impact"] < 0),
        "n_winners": sum(1 for p in by_position if p["pnl_impact"] > 0),
    }


def stress_portfolio(positions, spots, rates, vol_surfaces, scenario_name):
    """
    Stress test an entire FX options portfolio.

    Parameters
    ----------
    positions : list of dict
        Each must have: pair, strike, expiry, option_type, notional.
    spots : dict
        {pair: current_spot}
    rates : dict
        {pair: {"r_d": float, "r_f": float}}
    vol_surfaces : dict
        {pair: float}  -- ATM vol per pair (simplified) or callable(K, T)->vol
    scenario_name : str

    Returns
    -------
    dict with total_base_value, total_stressed_value, total_pnl, by_pair, by_position,
    worst_positions, limit_breaches.
    """
    by_position = []
    by_pair = {}
    total_base = 0.0
    total_stressed = 0.0

    for pos in positions:
        pair = pos.get("pair", "EURUSD")
        spot = spots.get(pair, 1.0)
        rate_info = rates.get(pair, {"r_d": 0.03, "r_f": 0.02})
        if isinstance(rate_info, dict):
            r_d = rate_info.get("r_d", 0.03)
            r_f = rate_info.get("r_f", 0.02)
        elif isinstance(rate_info, (tuple, list)) and len(rate_info) >= 2:
            r_d, r_f = rate_info[0], rate_info[1]
        else:
            r_d, r_f = 0.03, 0.02

        vol_entry = vol_surfaces.get(pair, 0.10)
        if callable(vol_entry):
            vol = vol_entry(pos["strike"], pos["expiry"])
        else:
            vol = vol_entry

        result = stress_single_position(pos, spot, r_d, r_f, vol, scenario_name)
        by_position.append(result)

        total_base += result["base_value"]
        total_stressed += result["stressed_value"]

        if pair not in by_pair:
            by_pair[pair] = 0.0
        by_pair[pair] += result["pnl_impact"]

    total_pnl = total_stressed - total_base

    worst_positions = sorted(by_position, key=lambda x: x["pnl_impact"])

    # Simple limit checks under stress
    limit_breaches = []
    if total_pnl < -1_000_000:
        limit_breaches.append({
            "limit": "MAX_STRESS_LOSS",
            "threshold": -1_000_000,
            "actual": total_pnl,
            "severity": "CRITICAL" if total_pnl < -5_000_000 else "WARNING",
        })
    for pair, pair_pnl in by_pair.items():
        if abs(pair_pnl) > 500_000:
            limit_breaches.append({
                "limit": f"CONCENTRATION_{pair}",
                "threshold": 500_000,
                "actual": pair_pnl,
                "severity": "WARNING",
            })

    scenario_info = FX_STRESS_SCENARIOS.get(scenario_name, {})

    return {
        "scenario_name": scenario_name,
        "scenario_display": scenario_info.get("name", scenario_name),
        "severity": scenario_info.get("severity", "UNKNOWN"),
        "total_base_value": total_base,
        "total_stressed_value": total_stressed,
        "total_pnl": total_pnl,
        "by_pair": by_pair,
        "by_position": by_position,
        "worst_positions": worst_positions[:10],
        "limit_breaches": limit_breaches,
        "n_positions": len(positions),
        "n_losers": sum(1 for p in by_position if p["pnl_impact"] < 0),
        "n_winners": sum(1 for p in by_position if p["pnl_impact"] > 0),
    }


# =====================================================================
# Scenario Comparison
# =====================================================================

def compare_scenarios(positions, spots, rates, vol_surfaces, scenario_names=None):
    """
    Compare all (or selected) scenarios side by side.

    Returns a pandas DataFrame with columns:
    Scenario, Severity, Total PnL, Worst Pair, Worst Position PnL, # Losing
    Sorted by Total PnL (worst first).
    """
    if scenario_names is None:
        scenario_names = list(FX_STRESS_SCENARIOS.keys())

    rows = []
    for name in scenario_names:
        result = stress_portfolio(positions, spots, rates, vol_surfaces, name)
        worst_pair = min(result["by_pair"], key=result["by_pair"].get) if result["by_pair"] else ""
        worst_pos_pnl = result["worst_positions"][0]["pnl_impact"] if result["worst_positions"] else 0.0

        rows.append({
            "Scenario": FX_STRESS_SCENARIOS[name]["name"],
            "Severity": result["severity"],
            "Total PnL": result["total_pnl"],
            "Worst Pair": worst_pair,
            "Worst Pair PnL": result["by_pair"].get(worst_pair, 0.0),
            "Worst Position PnL": worst_pos_pnl,
            "# Losing": result["n_losers"],
            "# Winning": result["n_winners"],
        })

    df = pd.DataFrame(rows).sort_values("Total PnL", ascending=True).reset_index(drop=True)
    return df


# =====================================================================
# Custom Stress
# =====================================================================

def custom_stress(positions, spots, rates, vol_surfaces, custom_shocks):
    """
    Run a user-defined custom stress scenario.

    Parameters
    ----------
    custom_shocks : dict
        {pair: {"spot": -0.05, "vol_mult": 2.0, "rate": -50}}
        spot is a decimal fraction, vol_mult is a multiplier, rate is in bps.

    Returns
    -------
    Same structure as stress_portfolio.
    """
    # Build a temporary scenario WITHOUT mutating the global dict (thread-safe).
    # Previous code inserted into FX_STRESS_SCENARIOS which caused race conditions
    # when concurrent callbacks or reverse_stress_test optimisation loops ran.
    temp_scenario = {
        "name": "Custom Scenario",
        "description": "User-defined custom stress",
        "severity": "CUSTOM",
        "date": "",
        "shocks": {
            pair: _default(sh.get("spot", 0.0), sh.get("vol_mult", 1.0), sh.get("rate", 0))
            for pair, sh in custom_shocks.items()
        },
        "default_shock": _default(0.0, 1.0, 0),
    }
    result = _stress_portfolio_with_scenario(positions, spots, rates, vol_surfaces, temp_scenario)

    result["scenario_name"] = "Custom"
    result["scenario_display"] = "Custom Scenario"
    return result


# =====================================================================
# Reverse Stress Test
# =====================================================================

def reverse_stress_test(positions, spots, rates, vol_surfaces,
                        target_loss, metric="total_pnl"):
    """
    Find the shock vector that produces a given target loss.

    Uses numerical optimization over a uniform spot shock and vol multiplier
    applied to all pairs. Returns the shock parameters that cause the
    portfolio to lose approximately target_loss.

    Parameters
    ----------
    target_loss : float -- negative number representing the target P&L
    metric : str        -- 'total_pnl' (only supported metric currently)

    Returns
    -------
    dict with optimal spot_shock, vol_multiplier, achieved_pnl, and
    per-pair shock details.
    """
    pairs_in_portfolio = list({pos.get("pair", "EURUSD") for pos in positions})

    def objective(params):
        spot_shock, vol_mult = params
        vol_mult = max(vol_mult, 0.1)
        custom = {pair: {"spot": spot_shock, "vol_mult": vol_mult, "rate": 0}
                  for pair in pairs_in_portfolio}
        result = custom_stress(positions, spots, rates, vol_surfaces, custom)
        achieved = result["total_pnl"]
        return (achieved - target_loss) ** 2

    best_result = None
    best_cost = float("inf")

    # Multi-start optimization: try several initial guesses
    starts = [
        (-0.05, 1.5),
        (-0.10, 2.0),
        (-0.15, 2.5),
        (0.05, 1.5),
        (0.10, 2.0),
        (-0.02, 3.0),
    ]

    for x0 in starts:
        try:
            res = minimize(objective, x0, method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-4})
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        return {"error": "Optimization failed to converge", "target_loss": target_loss}

    opt_spot, opt_vol = best_result.x
    opt_vol = max(opt_vol, 0.1)

    # Compute the final result with the optimal parameters
    final_shocks = {pair: {"spot": opt_spot, "vol_mult": opt_vol, "rate": 0}
                    for pair in pairs_in_portfolio}
    final_result = custom_stress(positions, spots, rates, vol_surfaces, final_shocks)

    return {
        "target_loss": target_loss,
        "achieved_pnl": final_result["total_pnl"],
        "residual": abs(final_result["total_pnl"] - target_loss),
        "spot_shock": opt_spot,
        "vol_multiplier": opt_vol,
        "per_pair_shocks": final_shocks,
        "portfolio_result": final_result,
    }


# =====================================================================
# Scenario Sensitivity (Scaling)
# =====================================================================

def scenario_sensitivity(positions, spots, rates, vol_surfaces,
                         scenario_name, scale_range=None):
    """
    Scale a named scenario from 0% to 200% intensity and compute P&L
    at each scale level. Useful for understanding non-linear risk exposure.

    The scaling works as follows:
      - spot shock is linearly scaled: shock * scale
      - vol multiplier is interpolated: 1 + (mult - 1) * scale
      - rate shock is linearly scaled: shock * scale

    Parameters
    ----------
    scale_range : array-like or None
        Scale factors to evaluate. Default is 0% to 200% in 5% steps.

    Returns
    -------
    pandas DataFrame with columns: Scale, Total_PnL, and per-pair PnL columns.
    """
    if scale_range is None:
        scale_range = np.linspace(0.0, 2.0, 41)

    sc = FX_STRESS_SCENARIOS.get(scenario_name)
    if sc is None:
        return pd.DataFrame()

    base_shocks = sc["shocks"]
    default_sh = sc.get("default_shock", _default(0.0, 1.0, 0))
    pairs_in_portfolio = list({pos.get("pair", "EURUSD") for pos in positions})

    records = []

    for scale in scale_range:
        scaled_shocks = {}
        for pair in pairs_in_portfolio:
            ref = base_shocks.get(pair, default_sh)
            scaled_shocks[pair] = {
                "spot": ref["spot"] * scale,
                "vol_mult": 1.0 + (ref["vol_mult"] - 1.0) * scale,
                "rate": ref["rate"] * scale,
            }

        result = custom_stress(positions, spots, rates, vol_surfaces, scaled_shocks)

        row = {
            "Scale": scale,
            "Scale_Pct": f"{scale:.0%}",
            "Total_PnL": result["total_pnl"],
        }
        for pair, pnl in result["by_pair"].items():
            row[f"PnL_{pair}"] = pnl

        records.append(row)

    return pd.DataFrame(records)
