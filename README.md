# FX Options Workstation

A Bloomberg Terminal-style analytics dashboard built for FX options strategists. Covers the full workflow from market monitoring and volatility surface analysis to trade structuring, execution, portfolio risk management, and backtesting.

Built with Dash/Plotly and Python. Requires a live Bloomberg Terminal connection for all market data.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Dash](https://img.shields.io/badge/dash-2.14%2B-orange)
![Bloomberg](https://img.shields.io/badge/data-Bloomberg%20Terminal-green)
<img width="1402" height="790" alt="image" src="https://github.com/user-attachments/assets/907471fd-7f71-458a-ac7f-895e80f73bb8" />


---

## Quick Start

### Prerequisites

- Python 3.9+
- Bloomberg Terminal running on `localhost:8194`
- Bloomberg C++ SDK installed (`WAPI<GO>` on terminal)
- `BLPAPI_ROOT` environment variable set

### Install & Run

```bash
pip install -r requirements.txt
python app.py
```

The app launches on `localhost:8765` via pywebview (desktop window) or falls back to `localhost:8050` in the browser. Missing pip packages are auto-installed on first run.

On Windows, you can also double-click `start.bat` which handles Python detection and dependency installation automatically.

### Bloomberg Connectivity Test

```bash
python test_bbg.py
```

Runs through FX spot, vol (ATM/RR/BF), deposit rates, and forward points to verify your Bloomberg connection is working. Copy the output for debugging if anything fails.

---

## Workspaces & Panels

The dashboard is organized into 4 workspaces with 8 panels total. Use the workspace buttons at the top or press `Ctrl+K` to open the command palette for quick navigation.

### DESK

| Panel | Description |
|-------|-------------|
| **Dashboard** | Market overview with live FX ticker tape, KPI summary, top movers table, vol/skew/term structure charts, and central bank event calendar |

### VOL

| Panel | Description |
|-------|-------------|
| **Vol Scanner** | 30-pair x 6-tenor heatmap grid showing ATM vol, risk reversals, butterflies, and vol metrics across the full FX universe at a glance |
| **Vol Surface** | Flagship panel with 14+ chart types including 3D volatility surface, smile evolution, term structure, vol cone, IV-RV analysis, forward vol, skew dynamics, and implied probability distributions |
| **Relative Value** | Cross-pair analytics: vol spread analysis, correlation monitoring, macro regime overlays, carry rankings, and positioning data |

### TRADE

| Panel | Description |
|-------|-------------|
| **Trade Workshop** | Structure builder with 25 preset strategies plus fully custom multi-leg construction. Garman-Kohlhagen pricing, Greeks, payoff diagrams, breakeven analysis, expected value calculator, and structure comparison |
| **Blotter** | Trade execution and booking. Flow analytics, position logging, and portfolio integration with automatic Greeks computation at entry |

### RISK

| Panel | Description |
|-------|-------------|
| **Risk Dashboard** | Portfolio-level Greeks aggregation, Value-at-Risk (parametric, historical, Cornish-Fisher), stress testing against 26 historical/hypothetical scenarios, P&L attribution, vega bucketing, and what-if analysis |
| **Backtest** | Historical strategy backtester with 10 option strategies, regime-aware entry signals, full P&L tracking, and performance analytics |

---

## Supported Instruments

### FX Pairs (30)

**G10 Majors:** EURUSD, USDJPY, GBPUSD, USDCHF, AUDUSD, NZDUSD, USDCAD

**G10 Crosses:** EURGBP, EURJPY, GBPJPY, AUDJPY, EURCHF, EURAUD, EURNZD, NZDJPY, AUDNZD, CADCHF, CADJPY

**Scandies:** EURNOK, EURSEK, USDSEK, USDNOK

**Emerging Markets:** USDMXN, USDBRL, USDTRY, USDZAR, USDCNH, USDINR, USDSGD, USDKRW

Each pair has full convention support: delta conventions (spot, forward, premium-adjusted), premium currency, pip size, spot date rules (T+1/T+2), cut times (NY/TKY/LDN), and liquidity tiers.

### Tenors

ON, 1W, 2W, 1M, 2M, 3M, 4M, 5M, 6M, 9M, 1Y, 18M, 2Y, 3Y, 4Y, 5Y

### Vol Quoting

ATM (delta-neutral straddle), 25-delta and 10-delta risk reversals and butterflies. Full 5-point smile construction per tenor.

---

## Structure Presets (25)

The Trade Workshop includes these preset structures:

| Category | Structures |
|----------|-----------|
| **Vanillas** | Call, Put |
| **Vertical Spreads** | Call Spread, Put Spread |
| **Volatility** | Risk Reversal, 25D Risk Reversal, Straddle, Strangle, 25D Strangle, 10D Strangle |
| **Butterflies & Condors** | Butterfly, Iron Butterfly, Iron Condor, Broken Wing Butterfly |
| **Exotic Spreads** | Seagull, Collar, Fence, Participating Forward, Leveraged Forward |
| **Ratio Structures** | 1x2 Call Spread, 1x2 Put Spread, 1x3 Call Spread, 1x3 Put Spread, Jade Lizard |
| **Calendar/Diagonal** | Calendar Spread, Diagonal Spread |
| **Multi-Leg** | Christmas Tree, Ladder |
| **Custom** | Fully custom up to 8 legs |

All structures are priced using Garman-Kohlhagen with live Bloomberg vol surfaces and deposit rates.

---

## Exotic Products (11)

The Exotics Pricer panel supports:

| Product | Pricing Method |
|---------|---------------|
| **Barrier (KI/KO)** | Closed-form (down/up-and-in/out) with rebate |
| **Double Barrier** | Monte Carlo simulation |
| **Digital / Binary** | Closed-form with overhedge |
| **One-Touch / No-Touch** | Closed-form |
| **Double No-Touch** | Monte Carlo simulation |
| **Range Accrual** | Monte Carlo with daily/weekly/monthly fixings |
| **Asian (Fix/Float)** | Geometric closed-form + arithmetic Monte Carlo |
| **Lookback** | Analytical (floating) + Monte Carlo (fixed) |
| **Forward Start** | Closed-form |
| **Best-of / Worst-of** | Correlated Monte Carlo (2 assets) |
| **TARF** | Monte Carlo with target profit and leverage |

All exotics include full Greeks via finite-difference bumping and scenario analysis.

---

## Stress Testing (26 Scenarios)

### Historical Scenarios

| Scenario | Severity | Date |
|----------|----------|------|
| SNB CHF Unpegging | EXTREME | Jan 2015 |
| Brexit Vote | SEVERE | Jun 2016 |
| COVID-19 Liquidity Crisis | EXTREME | Mar 2020 |
| BOJ FX Intervention | SEVERE | Sep 2022 |
| EM Contagion (Asia/Russia) | EXTREME | 1997-98 |
| Carry Trade Unwind | SEVERE | 2007-08 |
| European Debt Crisis | SEVERE | 2011-12 |

### Hypothetical Scenarios

| Scenario | Severity |
|----------|----------|
| Fed Emergency Rate Cut | MODERATE |
| Correlation Breakdown | SEVERE |
| USD Flash Crash | EXTREME |
| Geopolitical Shock / Risk-Off | SEVERE |
| Hawkish Fed Surprise | MODERATE |
| China Devaluation | SEVERE |
| Oil Price Shock | MODERATE |
| FX Volmageddon | SEVERE |

### Parametric Shocks

Vol Spike (+50% / +100%), Vol Collapse (-30%), Risk-Off, Risk-On, Rates +/-100bp, Spot +/-5%, Spot +/-10%, Tail Risk

Custom stress scenarios with user-defined spot/vol/rate shocks per pair are also supported, plus reverse stress testing to find the scenario that causes a target P&L loss.

---

## Backtest Strategies (10)

| Strategy | Description |
|----------|-------------|
| Long/Short Straddle | ATM call + ATM put |
| Long/Short Strangle | OTM put + OTM call |
| Long/Short Risk Reversal | Sell put, buy call (or inverse) |
| Long/Short Butterfly | Buy wings, sell body (or inverse) |
| Long/Short Calendar | Buy far-dated, sell near-dated (or inverse) |

Configurable parameters: pair, tenor (1M/3M/6M), delta (25D/10D), lookback (1-5 years), and entry signals (fixed interval, vol z-score, regime-based, percentile-based).

---

## Architecture

### Stack

- **Frontend:** Dash/Plotly with pure-black Bloomberg Terminal aesthetic
- **Backend:** Python with Garman-Kohlhagen, SABR, Vanna-Volga, and Monte Carlo pricing
- **Data:** Bloomberg Terminal via blpapi (localhost:8194)
- **Desktop:** pywebview wrapper (falls back to browser)

### Data Flow

```
Bloomberg Terminal (localhost:8194)
        |
        v
bg_fetcher.py (daemon thread, 2-min cycle)
        |
        v
bloomberg_fx.py (in-memory cache, TTL-based)
        ^
        |
Dash callbacks (panel files)
```

Callbacks never hit Bloomberg directly. The cache layer serves all requests with configurable TTLs:
- **Spots:** 15-second TTL
- **Vol surfaces:** 300-second TTL
- **Rates:** 1800-second TTL

### Entry Point

`app.py` is the single entry point. On startup it:
1. Auto-installs any missing pip packages
2. Connects to Bloomberg Terminal
3. Starts the background data fetcher thread
4. Pre-renders all panel layouts into the DOM
5. Registers all panel callbacks
6. Launches via pywebview or falls back to browser

Tab switching toggles `display:none/block` -- panels are never destroyed or recreated, preserving state.

### Core Modules

| Module | Purpose |
|--------|---------|
| `core/bloomberg.py` | blpapi session wrapper (BDP/BDH/BDS requests) |
| `core/bloomberg_fx.py` | FX-specific data provider with in-memory cache and cache-only mode |
| `core/bg_fetcher.py` | Background daemon thread that pre-populates the cache on a 2-minute cycle |
| `core/fx_conventions.py` | `FXPairSpec` registry for 30 pairs, delta systems (spot/forward/premium-adjusted), tenor utilities, cut times, calendar functions |
| `core/fx_analytics.py` | Vol percentile/z-score, regime detection, vol cone, IV-RV spread, correlation analysis, carry analytics, positioning, VaR |
| `core/pricing.py` | Black-Scholes, Garman-Kohlhagen, Monte Carlo, binomial tree, full Greeks chain (delta through ultima), implied vol solver, SABR fitting |
| `core/vanna_volga.py` | Vanna-Volga smile model and SABR volatility model with comparison tools |
| `core/fx_exotics.py` | 11 exotic product pricers with Greeks via finite-difference bumping |
| `core/fx_portfolio.py` | Position management, GK Greeks, P&L attribution, vega/gamma bucketing, hedge suggestions, what-if analysis |
| `core/fx_stress.py` | 26 stress scenarios, custom shock builder, reverse stress testing, scenario sensitivity |
| `core/theme.py` | Bloomberg Terminal aesthetic: colors, layout constants, chart template, reusable component styles |
| `core/csv_export.py` | Figure-to-DataFrame extraction and CSV download via `dcc.Download` |

### Panel Architecture

Every panel in `panels/` exports two functions:
- `layout()` -- returns the Dash component tree
- `register_callbacks(app)` -- registers all `@app.callback` decorators

### Global State

Cross-panel state is managed via Dash `dcc.Store` components:
- `global-pair` -- currently selected FX pair (default: EURUSD)
- `global-tenor` -- currently selected tenor (default: 3M)
- `global-portfolio-version` -- integer incremented on trade execution, triggers risk panel refresh
- `watchlist-store` -- user's custom pair watchlist

---

## Pricing Models

### Garman-Kohlhagen (Primary)

The standard FX extension of Black-Scholes, accounting for foreign and domestic interest rates. Used for all vanilla pricing, Greeks computation, and delta-to-strike conversions.

**Greeks computed:** Delta (spot/forward/premium-adjusted), Gamma, Theta, Vega, Rho, Vanna, Volga, Charm, Speed, Color, Ultima, Dual Delta

### SABR

Stochastic Alpha Beta Rho model for smile dynamics. Calibrated to market quotes (ATM + 25D/10D RR/BF). Used for strike interpolation and smile extrapolation.

### Vanna-Volga

Three-point smile model using ATM, 25D put, and 25D call vols. Provides fast smile interpolation consistent with market quotes. Includes comparison tools against SABR.

### Monte Carlo

Path-dependent pricing for exotic products. Supports:
- Single-asset and correlated multi-asset simulation
- Configurable path count and time steps
- Antithetic variance reduction
- Used for barriers, Asians, lookbacks, TARFs, and other path-dependent payoffs

### Realized Volatility

Three estimators: Close-to-Close, Parkinson (high-low), and Garman-Klass (OHLC).

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Open command palette (search panels, pairs) |

---

## Project Structure

```
Options-dashboard/
  app.py                    # Single entry point (~1400 lines)
  requirements.txt          # Python dependencies
  start.bat                 # Windows launcher
  test_bbg.py               # Bloomberg connectivity test
  assets/
    custom.css              # Ticker tape animation, scrollbar styling
  core/
    bloomberg.py            # blpapi session wrapper
    bloomberg_fx.py         # FX data provider + cache
    bg_fetcher.py           # Background data fetcher thread
    fx_conventions.py       # FX pair registry + conventions
    fx_analytics.py         # Volatility & correlation analytics
    pricing.py              # GK, BS, MC, binomial, Greeks, SABR
    vanna_volga.py          # Vanna-Volga + SABR smile models
    fx_exotics.py           # 11 exotic product pricers
    fx_portfolio.py         # Position management + risk
    fx_stress.py            # Stress scenarios + reverse stress
    theme.py                # Colors, styles, chart template
    csv_export.py           # CSV download utilities
  panels/
    market_dashboard.py     # DESK: Market overview
    vol_scanner_unified.py  # VOL: Heatmap scanner
    vol_surface_fx.py       # VOL: Flagship vol surface
    relative_value_plus.py  # VOL: Cross-pair RV
    structure_builder.py    # TRADE: Structure builder
    blotter_fx.py           # TRADE: Execution blotter
    risk_fx.py              # RISK: Portfolio risk
    backtest.py             # RISK: Strategy backtester
    exotics_pricer.py       # Exotic products pricer
    chart_lab.py            # Analytical workstation
  downloads/                # CSV export destination
```

---

## Configuration

### Bloomberg Connection

The app connects to Bloomberg Terminal on `localhost:8194` (default blpapi port). The terminal must be running before launching the app. There is no synthetic data fallback.

### Data Refresh

- Background fetcher runs on a 2-minute cycle
- Ticker tape refreshes every 2 minutes
- Data source status indicator shows LIVE / DEGRADED / DISCONNECTED in the header

### CSV Export

All charts support CSV export. Files are saved to the `downloads/` directory inside the project root.

---

## Requirements

```
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=5.18.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
pywebview>=4.0 (optional, for desktop mode)
blpapi>=3.19.0 (requires Bloomberg C++ SDK)
```
