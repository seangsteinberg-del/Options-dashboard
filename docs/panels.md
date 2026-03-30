# Panel Reference

Detailed documentation for each panel in the FX Options Workstation.

---

## DESK Workspace

### Market Dashboard (`panels/market_dashboard.py`)

The landing page and market overview panel.

**Features:**
- **Live Ticker Tape** -- Scrolling FX spot prices with change percentages for 12 major pairs, auto-refreshes every 2 minutes
- **KPI Summary** -- Key metrics: top mover, highest vol, widest spread, etc.
- **Top Movers Table** -- Sortable table of FX pairs ranked by daily change, vol level, or spread
- **Vol/Skew/Term Charts** -- Quick-view charts for the selected pair showing ATM vol history, risk reversal skew, and term structure
- **Central Bank Calendar** -- Upcoming policy meetings for Fed, ECB, BOE, BOJ, RBA, RBNZ, BOC with estimated day-of-year offsets
- **CSV Export** -- All chart data exportable

**Data Sources:** `get_fx_spots()`, `get_fx_vol_surface()`, `get_fx_board()`

---

## VOL Workspace

### Vol Scanner (`panels/vol_scanner_unified.py`)

A 30-pair x 6-tenor heatmap view of the entire FX vol universe.

**Features:**
- **ATM Vol Heatmap** -- Color-coded grid showing ATM implied vol for every pair/tenor combination
- **Risk Reversal Heatmap** -- 25-delta RR levels across the grid
- **Butterfly Heatmap** -- 25-delta BF levels
- **Breakeven Analysis** -- IV-RV cushion and breakeven realized vol for each cell
- **Vol Percentile Overlay** -- Historical percentile ranking (1Y lookback) shown as cell annotations
- **Click-to-Navigate** -- Click any cell to jump to that pair/tenor in the Vol Surface panel

**Data Sources:** `get_fx_vol_matrix()`, `breakeven_vol()`, `vol_percentile()`

---

### Vol Surface (`panels/vol_surface_fx.py`)

The flagship analytical panel with 14+ chart types for deep volatility analysis.

**Chart Types:**
1. **3D Vol Surface** -- Interactive 3D surface plot (strike x tenor x vol)
2. **Smile** -- Volatility smile for a single tenor (vol vs delta)
3. **Smile Evolution** -- How the smile has changed over time
4. **Term Structure** -- ATM vol across tenors
5. **Vol Cone** -- Historical vol percentile bands (min/25th/median/75th/max) vs current IV
6. **IV-RV Spread** -- Implied vs realized vol spread over time
7. **IV-RV Percentile** -- Historical percentile of the IV-RV spread
8. **Forward Vol** -- Forward-starting implied vol curve
9. **Vol Z-Score** -- Standardized vol level relative to history
10. **Skew Dynamics** -- Risk reversal evolution and regime detection
11. **Implied PDF** -- Implied probability density function from option prices
12. **Implied CDF** -- Cumulative distribution function
13. **Smile PCA** -- Principal component analysis of smile movements
14. **Breakeven Analysis** -- Required realized vol to break even on straddle positions

**Additional Features:**
- Pair and tenor selectors synced with global state
- "Build Trade" button to jump to Trade Workshop with current pair
- CSV export for all charts

**Data Sources:** `get_fx_vol_surface()`, `vol_cone()`, `iv_rv_spread()`, `forward_vol_curve()`, `smile_implied_pdf()`, `smile_pca()`

---

### Relative Value (`panels/relative_value_plus.py`)

Cross-pair analytics for identifying relative value opportunities.

**Features:**
- **Vol Spread Analysis** -- Compare ATM vol levels between two pairs over time
- **RR/BF Spread** -- Cross-pair risk reversal and butterfly differentials
- **Correlation Monitor** -- Spot and vol correlation matrices with regime detection
- **Implied Correlation** -- Derived from triangle arbitrage (e.g., EURUSD + USDJPY vs EURJPY)
- **Correlation Cone** -- Historical correlation percentile bands
- **Carry Rankings** -- Carry-per-vol ratio rankings across all pairs
- **Carry Momentum** -- Rate differential trend analysis
- **Positioning Data** -- CFTC futures positioning with z-scores and extremes

**Data Sources:** `cross_pair_vol_spread()`, `implied_correlation()`, `correlation_cone()`, `carry_table()`, `carry_per_vol()`, `cftc_positioning_data()`

---

## TRADE Workspace

### Trade Workshop (`panels/structure_builder.py`)

Full-featured options structure builder with 25 presets.

**Features:**
- **Preset Structures** -- 25 pre-configured strategies from vanilla calls to jade lizards
- **Custom Builder** -- Up to 8 legs with independent call/put, buy/sell, delta, and ratio settings
- **Garman-Kohlhagen Pricing** -- Live pricing using Bloomberg vol surfaces and deposit rates
- **Greeks Display** -- Full Greeks chain for each leg and aggregate
- **Payoff Diagram** -- Interactive P&L at expiry chart
- **Breakeven Analysis** -- Breakeven spot levels and vol levels
- **Expected Value** -- Monte Carlo expected P&L and probability of profit
- **Structure Comparison** -- Save structure A and B, compare side-by-side
- **Efficiency Analysis** -- Auto-suggests alternative structures for the same market view
- **Send to Blotter** -- One-click execution to book the trade

**Pricing Flow:**
1. User selects pair, tenor, and structure
2. Bloomberg vol surface and rates are fetched from cache
3. Delta-to-strike conversion using pair's delta convention (spot/forward/premium-adjusted)
4. GK pricing for each leg
5. Aggregate Greeks, net premium, and payoff profile computed

**Data Sources:** `get_fx_vol_surface()`, `get_fx_rates()`, `get_fx_spots()`, `delta_to_strike()`, `_gk_price()`

---

### Blotter (`panels/blotter_fx.py`)

Trade execution, booking, and flow analytics.

**Features:**
- **Trade Entry** -- Book single-leg or multi-leg trades with pair, tenor, delta, notional, cut time, strategy tag, and book assignment
- **Position Table** -- Live view of all open positions with Greeks, P&L, and status
- **Flow Analytics** -- Aggregated flow statistics: volume by pair, by strategy, by book
- **Book Management** -- Assign trades to books: G10_FLOW, G10_PROP, EM_FLOW, EM_PROP, STRUCTURED, HEDGE
- **Auto-Greeks** -- Delta, gamma, vega, theta computed at trade entry using GK
- **Portfolio Integration** -- Trades are written to `fx_portfolio` and trigger risk dashboard updates

**Data Sources:** `get_fx_vol_surface()`, `get_fx_spots()`, `get_fx_rates()`, `add_position()`, `get_all_positions()`

---

## RISK Workspace

### Risk Dashboard (`panels/risk_fx.py`)

Portfolio-level risk management and analysis.

**Features:**
- **Greeks Aggregation** -- Portfolio delta, gamma, vega, theta by pair and book
- **Vega Bucketing** -- Vega exposure by tenor bucket (1M/3M/6M/1Y/2Y+)
- **Gamma Bucketing** -- Gamma exposure by tenor
- **Delta by Pair** -- Net delta exposure across all pairs
- **Value-at-Risk** -- Parametric, Historical, and Cornish-Fisher VaR at 95%/99% confidence
- **P&L Attribution** -- Decompose P&L into delta, gamma, vega, theta, and cross-gamma components
- **Stress Testing** -- Apply any of 26 scenarios to the portfolio, see P&L impact per position
- **What-If Analysis** -- Add hypothetical trades and see the impact on portfolio risk before execution
- **Hedge Suggestions** -- Auto-generated delta-neutral and vega-neutral hedge recommendations
- **Risk Limits** -- Configurable limits with breach alerts (delta, gamma, vega, VaR)
- **Exposure Summary** -- Notional exposure by pair, currency, and book

**Data Sources:** `compute_portfolio_risk()`, `stress_portfolio()`, `pnl_attribution()`, `vega_by_bucket()`, `hedge_suggestion()`

---

### Backtest (`panels/backtest.py`)

Historical strategy backtester with regime awareness.

**Features:**
- **10 Strategies** -- Long/short straddle, strangle, risk reversal, butterfly, and calendar
- **Configurable Parameters:**
  - Pair: Any of the 30 registered pairs
  - Tenor: 1M, 3M, 6M
  - Delta: 25-delta or 10-delta
  - Lookback: 1-5 years
  - Entry signal: Fixed interval, vol z-score, vol percentile, regime-based
- **P&L Tracking** -- Entry price, mark-to-market, and exit P&L for each trade
- **Performance Metrics** -- Total P&L, win rate, Sharpe ratio, max drawdown, Omega ratio
- **Regime Overlay** -- Vol regime detection (low/normal/high/crisis) overlaid on P&L chart
- **Trade Table** -- Detailed log of every trade with entry/exit dates, prices, and P&L

**Data Sources:** `get_fx_historical_spot()`, `get_fx_historical_vol()`, `vol_regime_detect()`

---

## Additional Panels

### Exotics Pricer (`panels/exotics_pricer.py`)

Pricing engine for 11 exotic FX option products.

**Supported Products:**
- Barrier (KI/KO) with 4 barrier types (down/up-and-in/out)
- Double Barrier
- Digital / Binary
- One-Touch / No-Touch
- Double No-Touch
- Range Accrual (daily/weekly/monthly fixings)
- Asian (arithmetic/geometric averaging)
- Lookback (fixed/floating strike)
- Forward Start
- Best-of / Worst-of (2-asset correlated)
- TARF (Target Redemption Forward)

**For each product:**
- Price in premium currency
- Full Greeks via finite-difference bumping
- Scenario analysis (spot/vol/rate bumps)
- Comparison to vanilla equivalent

**Data Sources:** `get_fx_spots()`, `get_fx_vol_surface()`, `get_fx_rates()`, exotic pricing functions from `core/fx_exotics.py`

---

### Chart Lab (`panels/chart_lab.py`)

4-slot analytical workstation for custom chart arrangements.

**Features:**
- 4 independent chart slots, each configurable with different chart types
- Select pair and tenor independently per slot
- Available chart types include vol cone, IV-RV, term structure, breakeven, and more
- Useful for side-by-side comparison of different pairs or metrics
