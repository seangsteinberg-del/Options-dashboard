# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

FX Options Workstation — a Bloomberg Terminal-style analytics dashboard built for an FX options strategist. Covers the full workflow: market monitoring, vol surface analysis, trade structuring & pricing (Garman-Kohlhagen), execution, portfolio risk management, and backtesting. Requires a live Bloomberg Terminal connection for all market data.

## Running the App

```bash
python app.py
```

Launches on `localhost:8765` via pywebview (desktop) or falls back to `localhost:8050` in browser. Auto-installs missing pip packages on first run. No build step required.

Bloomberg Terminal on localhost:8194 is **required** — the app has no synthetic data fallback.

## Testing

```bash
python -m pytest tests/ -v    # 522 unit tests (no Bloomberg required)
python test_bbg.py             # Bloomberg connectivity test (requires Terminal running)
```

Unit tests cover pricing (BS, GK, Monte Carlo, binomial), exotic options (11 products), analytics (VaR, drawdown, tail risk), Vanna-Volga/SABR models, FX conventions (tenor math, delta/strike, forwards, arbitrage checks), stress scenarios (15+ named scenarios), portfolio management (CRUD, validation, persistence), CSV export (figure-to-DataFrame extraction), data helpers (mocked Bloomberg wrappers), and config validation. All tests are pure computation — no Bloomberg connection needed.

## Architecture

**Stack:** Dash/Plotly Python app with Bloomberg Terminal data integration and Garman-Kohlhagen FX options pricing.

### Entry Point & Panel Lifecycle

`app.py` is the single entry point (~1400 lines). It:
1. Connects to Bloomberg Terminal (required)
2. Starts a background data fetcher thread (`core/bg_fetcher.py`) on a 2-min cycle
3. Pre-renders ALL panel layouts into the DOM at startup
4. Tab switching toggles `display:none/block` — panels are never destroyed/recreated

Every panel in `panels/` exports two functions:
- `layout()` — returns the Dash component tree
- `register_callbacks(app)` — registers all `@app.callback` decorators

### Data Flow

```
Bloomberg Terminal → bg_fetcher.py (daemon thread, 2-min cycle)
                       ↓ writes to
                   bloomberg_fx.py (in-memory cache, TTL-based)
                       ↑ reads from
                   Dash callbacks (in panel files)
```

Callbacks never hit Bloomberg directly. The cache layer (`bloomberg_fx.py`) serves all requests with TTLs: 15s (spots), 300s (vol surfaces), 1800s (rates).

### Core Modules

| Module | Purpose |
|--------|---------|
| `core/bloomberg.py` | blpapi session wrapper (BDP/BDH/BDS) |
| `core/bloomberg_fx.py` | FX data provider with cache-only mode (Bloomberg required) |
| `core/bg_fetcher.py` | Background daemon thread pre-populating cache |
| `core/fx_conventions.py` | `FXPairSpec` registry for 30 pairs, delta systems, tenor utils, cut times |
| `core/fx_analytics.py` | Vol percentile, z-score, regime, cone, IV-RV, correlation, carry, VaR |
| `core/pricing.py` | Black-Scholes, Garman-Kohlhagen, Monte Carlo, binomial, full Greeks chain |
| `core/vanna_volga.py` | Vanna-Volga and SABR smile models |
| `core/fx_exotics.py` | 11 exotic product pricers (barriers, digitals, Asians, TARFs, etc.) |
| `core/fx_portfolio.py` | Position management, GK Greeks, P&L attribution, vega bucketing, hedge suggestions |
| `core/fx_stress.py` | Stress scenarios and custom shock builder |
| `core/theme.py` | All styling: colors, layout constants, chart template, reusable component styles |
| `core/csv_export.py` | Figure-to-DataFrame extraction and CSV download via `dcc.Download` |

### Panels (4 workspaces)

- **DESK:** `market_dashboard.py` — KPIs, movers table, vol/skew/term charts
- **VOL:** `vol_scanner_unified.py` (30x6 heatmaps), `vol_surface_fx.py` (flagship, 14+ chart types, 3D surface), `relative_value_plus.py` (cross-pair, correlation, macro, carry)
- **TRADE:** `structure_builder.py` (25 preset structures, GK pricing, solver, expected-value calc), `blotter_fx.py` (trade execution, flow analytics)
- **RISK:** `risk_fx.py` (Greeks, VaR, stress, P&L attribution, what-if), `backtest.py` (10 strategies, regime-aware)
- **Additional:** `chart_lab.py` (4-slot analytical workstation), `exotics_pricer.py` (11 exotic products)

### Global State

Dash `dcc.Store` components in `app.py` provide cross-panel state:
- `global-pair` (default "EURUSD"), `global-tenor` (default "3M")
- `global-portfolio-version` — integer incremented on trade execution, used by risk panel to detect changes
- `watchlist-store` — user's custom pair watchlist

## Theming Rules

Pure-black Bloomberg Terminal aesthetic defined in `core/theme.py`. Key constraints:

- **Backgrounds:** Always `#000000`. Never dark gray, never gradients.
- **Accent color:** Bloomberg orange `#ff8800` for all interactive elements, highlights, selections.
- **Semantic colors:** Green `#00cc66` (profit/positive), Red `#ff3333` (loss/negative).
- **Borders:** `#222240` (structural blue-gray).
- **Text:** `#d4d4d4` (data), `#808080` (labels/muted), `#ffffff` (section headers only).
- **Font:** JetBrains Mono monospace everywhere. No sans-serif.
- **Heatmap colorscales:** Dark midpoint (`#0e0e0e`) that blends with background. Extremes use theme colors (blue for cold/cheap, orange/red for hot/rich). Never use white midpoints or Plotly defaults (Plasma, Viridis, RdBu).
- **Colorbars:** Always include `outlinewidth=0, bgcolor="rgba(0,0,0,0)"`.
- **Cell gaps:** `xgap=2, ygap=2` on heatmaps (black bg shows through as grid).
- **No rounded corners, no box shadows, no blur effects.**

All reusable styles are exported from `core/theme.py`: `CARD_STYLE`, `BUTTON_STYLE`, `INPUT_STYLE`, `DROPDOWN_STYLE`, `STAT_BOX_STYLE`, `TAB_STYLE`, `CHART_TEMPLATE`, etc. Use `chart_layout(**overrides)` to build figure layouts.

## Key Patterns

- **CSV export:** `core/csv_export.py` extracts trace data from Plotly figure dicts. Each panel has a `dcc.Download` component and a callback that calls `export_csv(fig, panel_name, chart_type)`.
- **Analytics caching:** `fx_analytics.py` uses a TTL memoization decorator (120s) for expensive computations.
- **No synthetic data:** All market data comes from Bloomberg Terminal. Without a connection, panels show empty states.
- **Panel sizing:** `CHART_SM=240`, `CHART_MD=320`, `CHART_LG=400` (px heights). Spacing: `GAP="8px"`, `SECTION_GAP="16px"`.
