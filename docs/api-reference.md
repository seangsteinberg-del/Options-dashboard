# Core API Reference

Public functions and classes exported by each core module.

---

## core/pricing.py

Black-Scholes, Garman-Kohlhagen, Monte Carlo, and binomial pricing with full Greeks chain.

### Pricing Functions

```python
bs_price(S, K, T, r, q, sigma, option_type="call") -> float
```
Black-Scholes option price. `r` = risk-free rate, `q` = dividend yield (or foreign rate for FX).

```python
monte_carlo_price(S, K, T, r, q, sigma, option_type="call", n_paths=50000, n_steps=252, seed=None) -> dict
```
Monte Carlo simulation. Returns `{"price", "std_error", "ci_lower", "ci_upper"}`.

```python
binomial_tree_price(S, K, T, r, q, sigma, option_type="call", n_steps=200) -> float
```
Cox-Ross-Rubinstein binomial tree.

```python
implied_vol(price, S, K, T, r, q, option_type="call") -> float
```
Newton-Raphson implied volatility solver.

### Greeks

All Greeks follow the signature `(S, K, T, r, q, sigma, option_type="call") -> float` unless noted.

| Function | Description |
|----------|-------------|
| `delta()` | Option delta (dV/dS) |
| `gamma(S, K, T, r, q, sigma)` | Gamma (d2V/dS2), option-type independent |
| `theta()` | Theta (dV/dt), per calendar day |
| `vega(S, K, T, r, q, sigma)` | Vega (dV/dsigma), per 1% vol move |
| `rho()` | Rho (dV/dr) |
| `vanna(S, K, T, r, q, sigma)` | Vanna (d2V/dS dsigma) |
| `volga(S, K, T, r, q, sigma)` | Volga/Vomma (d2V/dsigma2) |
| `charm()` | Charm (d delta / dt) |
| `speed(S, K, T, r, q, sigma)` | Speed (d gamma / dS) |
| `color_greek(S, K, T, r, q, sigma)` | Color (d gamma / dt) |
| `ultima(S, K, T, r, q, sigma)` | Ultima (d3V/dsigma3) |
| `dual_delta()` | Dual delta (dV/dK) |

```python
compute_all_greeks(S, K, T, r, q, sigma, option_type="call") -> dict
```
Returns all Greeks in a single dict.

### Probability & Risk

```python
probability_itm(S, K, T, r, q, sigma, option_type="call") -> float
probability_of_profit(S, K, T, r, q, sigma, option_type="call", premium=None) -> float
expected_move(S, T, sigma, confidence=0.68) -> (lower, upper)
probability_touch(S, K, T, r, q, sigma) -> float
```

### SABR

```python
sabr_vol(F, K, T, alpha, beta, rho_sabr, nu) -> float
```
Hagan's SABR implied vol formula.

```python
fit_sabr(strikes, market_vols, F, T, beta=0.5) -> dict
```
Calibrate SABR parameters to market data. Returns `{"alpha", "beta", "rho", "nu", "rmse"}`.

### Vol Surface Generation

```python
generate_vol_surface(S, base_vol, skew_slope, term_slope, n_strikes, n_expiries) -> (strikes, expiries, vol_matrix)
generate_sabr_vol_surface(S, r, q, ...) -> (strikes, expiries, vol_matrix)
interpolate_vol_surface(strikes, expiries, vol_matrix) -> RectBivariateSpline
```

### Realized Volatility Estimators

```python
realized_vol_close_to_close(prices, window=20) -> pd.Series
realized_vol_parkinson(highs, lows, window=20) -> pd.Series
realized_vol_garman_klass(opens, highs, lows, closes, window=20) -> pd.Series
```

### Portfolio Functions

```python
scenario_grid(S, K, T, r, q, sigma, option_type, spot_range, vol_range) -> pd.DataFrame
aggregate_portfolio_greeks(positions, S, r, q) -> dict
portfolio_var_cvar(positions, S, r, q, horizon_days=1, confidence=0.95, n_sims=10000) -> dict
```

---

## core/fx_conventions.py

FX pair registry, delta systems, vol quoting, forward calculations, premium conversions, and calendar utilities.

### Pair Registry

```python
class FXPairSpec:
    pair: str           # "EURUSD"
    base: str           # "EUR"
    quote: str          # "USD"
    bb_spot: str        # Bloomberg spot ticker
    bb_vol: str         # Bloomberg vol ticker prefix
    pip: float          # 0.0001 or 0.01 (JPY pairs)
    group: str          # "G10" or "EM"
    subgroup: str       # "Majors", "Crosses", "Scandies", "LatAm", "CEEMEA", "Asia"
    cut_default: str    # "NY", "TKY", or "LDN"
    spot_date_rule: int # 1 (USDCAD) or 2 (all others)
    premium_ccy: str    # Currency for premium quotation
    delta_convention: str  # "spot", "forward", or "premium-adjusted"
    vol_quote: str      # "pips" or "%"
    notional_default: float  # Default notional size
    liquidity_tier: int # 1 (most liquid) to 4
```

```python
get_pair(pair: str) -> FXPairSpec       # Lookup, accepts "EURUSD" or "EUR/USD"
all_pairs() -> List[str]                # All 30 registered pairs
pair_groups() -> Dict[str, List[str]]   # Pairs grouped by group/subgroup
```

### Delta & Strike Conversions

```python
spot_delta(S, K, T, r_d, r_f, sigma, cp) -> float
forward_delta(S, K, T, r_d, r_f, sigma, cp) -> float
premium_adjusted_delta(S, K, T, r_d, r_f, sigma, cp) -> float

delta_to_strike(target_delta, S, T, r_d, r_f, sigma, cp, convention="spot") -> float
strike_to_delta(K, S, T, r_d, r_f, sigma, cp, convention="spot") -> float

atm_dns_strike(S, T, r_d, r_f, sigma) -> float     # Delta-neutral straddle strike
atm_forward_strike(S, T, r_d, r_f) -> float         # ATM forward strike
```

### Vol Smile Construction

```python
bf_rr_to_smile(atm, rr25, bf25, rr10=None, bf10=None) -> dict
```
Convert market quotes (ATM + RR + BF) to individual vol points. Returns `{"ATM", "25C", "25P", "10C", "10P"}`.

```python
build_smile_spline(atm, c25, p25, c10, p10, S, T, r_d, r_f) -> CubicSpline
build_full_vol_surface(vol_quotes_by_tenor, S, r_d, r_f) -> dict
vol_at_delta_tenor(surface, delta_val, tenor) -> float
vol_at_strike_tenor(surface, K, T, S, r_d, r_f, convention="spot") -> float
```

### Forward & Carry

```python
fx_forward(S, r_d, r_f, T) -> float           # Covered interest rate parity
forward_points(S, r_d, r_f, T) -> float        # Forward points (F - S)
forward_curve(S, r_d_curve, r_f_curve, tenors) -> dict
carry_roll_down(S, r_d, r_f, T, days) -> dict  # Carry and roll-down decomposition
```

### Premium Conversions

```python
premium_ccy1(bs_price_val, S) -> float         # Premium in base currency
premium_ccy2(bs_price_val) -> float            # Premium in quote currency
premium_pips(bs_price_val, pip_size) -> float   # Premium in pips
premium_pct_notional(bs_price_val, S, notional_ccy="base") -> float
```

### Tenor Utilities

```python
tenor_to_days(tenor: str) -> int       # "3M" -> 91
tenor_to_years(tenor: str) -> float    # "3M" -> 0.249
years_to_nearest_tenor(T: float) -> str  # 0.25 -> "3M"
```

### Calendar Functions

```python
fx_spot_date(trade_date, pair=None) -> date      # T+2 (T+1 for USDCAD)
fx_expiry_date(trade_date, tenor, pair=None) -> date
fx_delivery_date(expiry_date, pair=None) -> date
business_days_between(date1, date2) -> int
is_fx_holiday(dt, pair=None) -> bool
```

### Pair Utilities

```python
invert_pair(pair: str) -> str                    # "EURUSD" -> "USDEUR"
is_usd_base(pair: str) -> bool
is_usd_quote(pair: str) -> bool
cross_from_legs(pair_a, pair_b) -> Optional[str] # Derive cross from two legs
triangulate_vol(vol_a, vol_b, corr) -> float     # Cross vol from legs + correlation
pip_value(pair, notional) -> float
```

---

## core/fx_analytics.py

Volatility analytics, correlation analysis, carry metrics, positioning, and risk measures.

### Vol Analytics

```python
vol_percentile(pair, tenor, metric="ATM", lookback=252) -> dict
vol_zscore(pair, tenor, metric="ATM", lookback=252) -> dict
vol_percentile_surface(pair, lookback=252) -> pd.DataFrame
vol_zscore_surface(pair, lookback=252) -> pd.DataFrame
vol_change(pair, tenor, metric="ATM", periods=[1,5,21]) -> dict
vol_surface_diff(pair, days_ago=1) -> pd.DataFrame
```

### Vol Regime

```python
vol_regime_detect(pair, short_window=20, long_window=60) -> dict
```
Returns `{"regime": "low|normal|high|crisis", "short_vol", "long_vol", "ratio"}`.

```python
vol_regime_history(pair, lookback=252) -> pd.DataFrame
```

### Vol Cone

```python
vol_cone(pair, windows=[5,10,20,40,60], lookback=504) -> dict
```
Returns min/25th/median/75th/max realized vol for each window, plus current IV overlay.

### IV-RV Analysis

```python
iv_rv_spread(pair, tenor="3M", rv_window=20, lookback=252) -> dict
iv_rv_percentile(pair, tenor="3M", rv_window=20, lookback=252) -> dict
breakeven_vol(pair, tenor, days_to_expiry) -> dict
theta_gamma_ratio(pair, tenor) -> dict
vol_carry(pair, tenor) -> dict
```

### Forward Vol

```python
forward_vol(pair, T1, T2) -> dict                       # Forward vol between two tenors
forward_vol_curve(pair, start_tenor="1M") -> pd.DataFrame
forward_vol_surface(pair) -> pd.DataFrame
```

### Smile Analytics

```python
smile_skewness(pair, tenor) -> dict
smile_kurtosis(pair, tenor) -> dict
wing_richness(pair, tenor) -> dict
smile_asymmetry_index(pair, tenor) -> dict
smile_implied_pdf(pair, tenor, n_points=200) -> dict
smile_implied_cdf(pair, tenor, n_points=200) -> dict
tail_probabilities(pair, tenor, thresholds=[1,2,3]) -> dict
smile_pca(pair, lookback=252) -> dict
sticky_delta_monitor(pair, tenor, lookback=60) -> dict
```

### Cross-Pair / Relative Value

```python
cross_pair_vol_spread(pair_a, pair_b, tenor="3M", lookback=252) -> dict
cross_pair_rr_spread(pair_a, pair_b, tenor="3M", lookback=252) -> dict
cross_pair_bf_spread(pair_a, pair_b, tenor="3M", lookback=252) -> dict
vol_beta(pair_a, pair_b, tenor="3M", lookback=252) -> dict
rv_scanner(pairs=None, tenors=["1M","3M","6M"]) -> pd.DataFrame
rv_signal_composite(pair, lookback=252) -> dict
carry_adjusted_rv(pair, tenor="3M", lookback=252) -> dict
```

### Correlation

```python
implied_correlation(pair_a, pair_b, cross_pair, tenor="3M") -> dict
correlation_richness(pair_a, pair_b, cross_pair, tenor="3M", lookback=252) -> dict
spot_correlation_matrix(pairs=None, window=60) -> pd.DataFrame
vol_correlation_matrix(pairs=None, tenor="3M", window=60) -> pd.DataFrame
spot_vol_correlation(pair, window=60) -> dict
correlation_regime(pairs=None, window=60) -> dict
correlation_term_structure(pair_a, pair_b, windows=[20,40,60,120,252]) -> dict
correlation_cone(pair_a, pair_b, lookback=504) -> dict
```

### Carry & Positioning

```python
carry_table(pairs=None) -> pd.DataFrame
carry_per_vol(pairs=None) -> pd.DataFrame
carry_momentum(pair, lookback=60) -> dict
rate_differential_history(pair, lookback=252) -> pd.DataFrame
cftc_positioning_data(pair) -> dict
positioning_zscore(pair, lookback=156) -> dict
positioning_extremes(pairs=None) -> pd.DataFrame
positioning_vs_spot(pair, lookback=156) -> pd.DataFrame
```

### Risk Measures

```python
historical_var(returns, confidence=0.95, horizon_days=1) -> dict
parametric_var(sigma, notional, confidence=0.95, horizon_days=1) -> dict
expected_shortfall(returns, confidence=0.95, horizon_days=1) -> dict
cornish_fisher_var(returns, confidence=0.95) -> dict
max_drawdown(returns) -> dict
omega_ratio(pnl_array, mar=0.0) -> float
ulcer_index(pnl_array) -> float
max_consecutive(pnl_array, direction="loss") -> int
tail_risk_metrics(returns) -> dict
```

---

## core/vanna_volga.py

Vanna-Volga and SABR smile models.

### Vanna-Volga

```python
vv_price(S, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol, cp) -> float
vv_implied_vol(S, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol) -> float
vv_smile(S, T, r_d, r_f, atm_vol, p25_vol, c25_vol, n_strikes=50) -> dict
vv_greeks(S, K, T, r_d, r_f, atm_vol, p25_vol, c25_vol, cp) -> dict
```

### SABR

```python
sabr_vol(F, K, T, alpha, beta, rho, nu) -> float
```

### Comparison

```python
vv_vs_sabr(S, T, r_d, r_f, atm_vol, p25_vol, c25_vol, ...) -> dict
```
Compare Vanna-Volga and SABR smile fits.

---

## core/fx_exotics.py

11 exotic product pricers. All use Garman-Kohlhagen inputs: `S` (spot), `K` (strike), `T` (time to expiry in years), `r_d` (domestic rate), `r_f` (foreign rate), `sigma` (vol), `cp` (+1 call / -1 put).

### Products

```python
barrier_price(S, K, B, T, r_d, r_f, sigma, cp, barrier_type, rebate=0.0) -> float
double_barrier_price(S, K, B_up, B_down, T, r_d, r_f, sigma, cp, knock_type="out", n_paths=50000) -> float
digital_price(S, K, T, r_d, r_f, sigma, cp, payout=1.0) -> float
digital_greeks(S, K, T, r_d, r_f, sigma, cp, payout=1.0) -> dict
one_touch_price(S, B, T, r_d, r_f, sigma, payout=1.0) -> float
no_touch_price(S, B, T, r_d, r_f, sigma, payout=1.0) -> float
double_no_touch_price(S, B_up, B_down, T, r_d, r_f, sigma, payout=1.0, n_paths=50000) -> float
range_accrual_price(S, B_low, B_high, T, r_d, r_f, sigma, payout=1.0, fixing_freq="daily", n_paths=50000) -> float
asian_price(S, K, T, r_d, r_f, sigma, cp, fixing_freq="monthly", avg_type="arithmetic", n_paths=50000) -> float
lookback_price(S, T, r_d, r_f, sigma, cp, lookback_type="floating", K=None, n_paths=50000) -> float
forward_start_price(S, T_start, T_end, r_d, r_f, sigma, cp, moneyness=1.0) -> float
best_of_price(S1, S2, K, T, r_d1, r_d2, r_f, sigma1, sigma2, rho, cp, n_paths=50000) -> float
tarf_price(S, K, B, T, r_d, r_f, sigma, n_fixings=12, target_profit=0.05, leverage=2, n_paths=50000) -> dict
```

### Generic Greeks & Summary

```python
exotic_greeks(price_func, base_params, bumps=None) -> dict
```
Finite-difference Greeks for any exotic. Bumps spot, vol, rates, and time.

```python
exotic_summary(price, greeks, product_type, params) -> dict
```
Formatted summary with price, Greeks, and risk metrics.

---

## core/fx_portfolio.py

Position management, risk computation, and hedge generation.

### Position Management

```python
add_position(book, position_dict) -> str              # Returns position ID
close_position(book, position_id, close_price=None, close_date=None) -> dict
roll_position(book, position_id, new_expiry, new_strike=None) -> dict
partial_close(book, position_id, close_notional) -> dict
get_positions(book=None, pair=None, status="open") -> list
get_all_positions() -> list
```

**Books:** `G10_FLOW`, `G10_PROP`, `EM_FLOW`, `EM_PROP`, `STRUCTURED`, `HEDGE`

### Risk Computation

```python
compute_position_greeks(pos, spot, r_d, r_f, vol_surface) -> dict
compute_book_risk(book, spots, rates, vol_surfaces) -> dict
compute_portfolio_risk(spots, rates, vol_surfaces) -> dict
```

### Analytics

```python
pnl_attribution(positions, spots_old, spots_new, surfaces_old, surfaces_new, rates) -> dict
vega_by_bucket(positions, spots, rates, vol_surfaces) -> dict
gamma_by_bucket(positions, spots, rates, vol_surfaces) -> dict
delta_by_pair(positions, spots, rates, vol_surfaces) -> dict
exposure_summary(positions, spots, rates, vol_surfaces) -> dict
```

### Hedging

```python
check_risk_limits(risk_totals, limits=None) -> dict
hedge_suggestion(portfolio_risk, target="delta_neutral") -> list
what_if_add(positions, new_trade, spots, rates, vol_surfaces) -> dict
```

### Persistence

```python
save_portfolio(filepath=None) -> str
load_portfolio(filepath=None) -> dict
get_portfolio() -> dict
```

---

## core/fx_stress.py

Stress testing framework with 26 pre-built scenarios.

### Scenario Access

```python
get_scenarios() -> list                          # All scenario metadata
get_scenario(name) -> dict                       # Single scenario detail
get_pair_shock(scenario_name, pair) -> dict       # Shock for a specific pair
```

### Shock Application

```python
apply_spot_shock(current_spot, shock_pct) -> float
apply_vol_shock(current_vol, vol_multiplier) -> float
apply_rate_shock(current_rate, shock_bps) -> float
```

### Portfolio Stress

```python
stress_single_position(position, spot, r_d, r_f, vol, scenario_name) -> dict
stress_portfolio(positions, spots, rates, vol_surfaces, scenario_name) -> dict
compare_scenarios(positions, spots, rates, vol_surfaces, scenario_names=None) -> dict
custom_stress(positions, spots, rates, vol_surfaces, custom_shocks) -> dict
```

### Advanced

```python
reverse_stress_test(positions, spots, rates, vol_surfaces, target_pnl, ...) -> dict
scenario_sensitivity(positions, spots, rates, vol_surfaces, scenario_name, scale_range) -> dict
```

---

## core/bloomberg_fx.py

FX-specific Bloomberg data provider with caching.

### Configuration

```python
set_cache_only_mode(enabled: bool)    # Restrict to cache reads only
mark_thread_as_fetcher()              # Mark current thread as the fetcher
cache_clear()                          # Clear all cached data
```

### Data Functions

```python
get_fx_spots(pairs=None) -> Dict[str, dict]
get_fx_vol_surface(pair) -> Dict[str, dict]
get_fx_vol_point(pair, tenor="1M", metric="ATM") -> float
get_fx_rates(pair) -> dict
get_fx_rate_curve(ccy) -> Dict[str, float]
get_fx_forward_curve(pair) -> Dict[str, dict]
get_fx_historical_spot(pair, days=252) -> pd.DataFrame
get_fx_historical_vol(pair, tenor="1M", days=252) -> pd.DataFrame
get_fx_option_chain(pair, tenor="1M") -> pd.DataFrame
get_fx_realized_vol(pair, window=20, method="close") -> float
get_fx_correlation(pair_a, pair_b, window=60, method="spot") -> float
get_fx_implied_correlation(pair_a, pair_b, cross, tenor="3M") -> float
```

### Aggregated Views

```python
get_all_pairs() -> List[str]
get_fx_snapshot(pair) -> dict
get_fx_board(pairs=None) -> pd.DataFrame
get_fx_vol_matrix(pairs=None, tenors=None) -> pd.DataFrame
get_fx_correlation_matrix(pairs=None, window=60) -> pd.DataFrame
get_fx_risk_reversal_monitor(pairs=None, tenors=None) -> pd.DataFrame
get_fx_carry_rankings(pairs=None) -> pd.DataFrame
get_fx_term_structure(pair) -> pd.DataFrame
```

### Status

```python
get_data_mode() -> str                 # "LIVE", "DEGRADED", or "DISCONNECTED"
get_recent_errors() -> list
get_data_age_seconds(pair=None) -> dict
format_data_age(pair=None) -> str
```

---

## core/bloomberg.py

Low-level blpapi session wrapper.

```python
class BloombergConnection:
    """Manages a blpapi.Session with auto-reconnect."""
    connect() -> bool
    disconnect()
    is_connected() -> bool
    session -> blpapi.Session

get_connection() -> BloombergConnection   # Singleton
is_connected() -> bool                     # Quick check
bloomberg_ever_connected() -> bool         # True if ever connected this session
```

### Request Functions

```python
bdp(securities, fields) -> pd.DataFrame       # Reference data (BDP)
bdh(security, fields, start_date, end_date=None) -> pd.DataFrame  # Historical data (BDH)
bds(security, field, **overrides) -> pd.DataFrame  # Bulk data (BDS)
```

---

## core/theme.py

Styling constants and utilities.

### Key Exports

```python
COLORS: dict          # All color constants
CHART_TEMPLATE: dict  # Plotly figure template
chart_layout(**overrides) -> dict  # Build figure layout with Bloomberg aesthetic defaults

# Reusable component styles (dicts)
CARD_STYLE, BUTTON_STYLE, INPUT_STYLE, DROPDOWN_STYLE
STAT_BOX_STYLE, TAB_STYLE, TAB_SELECTED_STYLE

# Helper functions
no_data_fig(height=300, msg="NO DATA") -> go.Figure
skeleton_chart(height=300, label="LOADING") -> go.Figure
status_color() -> str
status_text() -> str
make_stat_style(color=None) -> dict
clickable_stat(value, label, pair, metric, tenor, color=None) -> html.Div
grid_cell(children, **kwargs) -> html.Div
section_header(text) -> html.Div
```

### Chart Sizing Constants

```python
CHART_SM = 240   # px
CHART_MD = 320   # px
CHART_LG = 400   # px
GAP = "8px"
SECTION_GAP = "16px"
```
