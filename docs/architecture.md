# Architecture Guide

## Overview

The FX Options Workstation is a single-process Dash/Plotly application that connects to Bloomberg Terminal for all market data. It runs as a desktop app via pywebview or in the browser.

## System Diagram

```
+-----------------------------------------------------------+
|                    pywebview / Browser                      |
|                  (localhost:8765)                           |
+-----------------------------------------------------------+
|                     Dash App (app.py)                       |
|  +-------+  +-------+  +-------+  +-------+  +-------+    |
|  | DESK  |  |  VOL  |  | TRADE |  | RISK  |  |EXTRAS |    |
|  |-------+  |-------+  |-------+  |-------+  |-------+    |
|  |Dashbrd|  |Scanner|  |Workshop|  |Risk   |  |Exotics|    |
|  |       |  |Surface|  |Blotter |  |Backtest| |ChrtLab|    |
|  |       |  |RelVal |  |       |  |       |  |       |    |
|  +-------+  +-------+  +-------+  +-------+  +-------+    |
+-----------------------------------------------------------+
|                   Core Engine Layer                         |
|  +-------------+  +-------------+  +--------------+        |
|  | pricing.py  |  |fx_analytics |  | fx_portfolio |        |
|  | vanna_volga |  |fx_stress.py |  | fx_exotics   |        |
|  +-------------+  +-------------+  +--------------+        |
+-----------------------------------------------------------+
|                    Data Layer                                |
|  +------------------+  +-------------------------------+   |
|  | bloomberg_fx.py  |  |     bg_fetcher.py             |   |
|  | (cache + API)    |<-|  (daemon thread, 2-min cycle) |   |
|  +------------------+  +-------------------------------+   |
|           |                          |                      |
|  +------------------+                |                      |
|  | bloomberg.py     |<---------------+                      |
|  | (blpapi wrapper) |                                       |
|  +------------------+                                       |
+-----------------------------------------------------------+
            |
            v
   Bloomberg Terminal (localhost:8194)
```

## Startup Sequence

1. **`_ensure_packages()`** -- Auto-installs missing pip packages (dash, plotly, numpy, scipy, pandas, pywebview, blpapi)

2. **Import panels** -- All 8+ panel modules are imported, which triggers their module-level code

3. **Import core** -- FX pair registry is built, theme constants are loaded

4. **Build layout** -- `app.layout` is constructed with all panel layouts pre-rendered in the DOM

5. **Register callbacks** -- Each panel's `register_callbacks(app)` is called

6. **Bloomberg connect** -- `is_connected()` checks if Bloomberg Terminal is reachable

7. **Start bg_fetcher** -- If Bloomberg is connected, a `BloombergFetcher` daemon thread starts:
   - First cycle runs immediately (up to 180s timeout)
   - If successful, `set_cache_only_mode(True)` is called
   - Subsequent cycles run every 2 minutes

8. **Launch server** -- pywebview creates a desktop window, or falls back to browser mode

## Panel Lifecycle

### Pre-rendering

All panels are rendered into the DOM at startup inside hidden `div` elements:

```python
for tab_id, module in _TAB_MODULE_MAP.items():
    panels.append(html.Div(
        _safe_panel_layout(tab_id, module),
        id=f"panel-{tab_id}",
        style={"display": "none"}
    ))
```

### Tab Switching

Tab switching toggles `display: none/block` via a clientside callback. Panels are never destroyed or recreated, which:
- Preserves panel state (dropdown selections, chart zoom levels)
- Avoids re-triggering initial callbacks
- Keeps all `dcc.Interval` components running in the background

### Error Handling

`_safe_panel_layout()` wraps each `layout()` call in a try/except. If a panel fails to render, it shows a red error message with the traceback instead of crashing the entire app.

## Data Flow

### Cache Architecture

`bloomberg_fx.py` implements a thread-safe in-memory cache with category-based TTLs:

| Category | TTL | Examples |
|----------|-----|---------|
| `spot` | 15s | FX spot prices |
| `vol` | 300s | Vol surfaces, ATM vols |
| `rates` | 1800s | Deposit rates, forward curves |
| `hist` | 3600s | Historical time series |

### Cache-Only Mode

After the first successful fetch cycle, `set_cache_only_mode(True)` is called. In this mode:
- Dash callbacks only read from cache (never trigger Bloomberg requests)
- Only the background fetcher thread is allowed to call Bloomberg
- This prevents callback timeouts and Bloomberg request contention

### Fetch Priority

The background fetcher runs in this order each cycle:
1. FX spot prices (all 30 pairs)
2. Vol surfaces (all pairs, all tenors)
3. Deposit rates (all currencies)
4. Forward curves
5. Historical data (hourly cycle, not every 2-min cycle)

### Error Tracking

`bloomberg_fx.py` tracks recent fetch failures in a rolling list. The header bar shows:
- **ALL DATA LIVE** (green) -- no recent errors
- **DEGRADED** (red) -- some data failing, with error summary
- **BLOOMBERG DISCONNECTED** (red) -- no connection

## Global State Management

### dcc.Store Components

| Store ID | Type | Purpose |
|----------|------|---------|
| `global-pair` | memory | Currently selected FX pair, synced across panels |
| `global-tenor` | memory | Currently selected tenor |
| `global-portfolio-version` | memory | Integer counter, incremented on trade execution |
| `watchlist-store` | local | User's custom watchlist (persists in browser localStorage) |

### Cross-Panel Communication

When a user executes a trade in the Blotter:
1. Trade is added to `fx_portfolio` (in-memory position store)
2. `global-portfolio-version` is incremented
3. Risk Dashboard detects the version change and recomputes portfolio risk

When a user clicks "Build Trade" on the Vol Surface:
1. Callback returns the selected pair
2. Workspace switches to TRADE, tab switches to Trade Workshop
3. `global-pair` is updated

## Threading Model

The app uses a single Python process with:
- **Main thread**: Dash/Flask server handling HTTP requests and callbacks
- **bg_fetcher thread**: Daemon thread running Bloomberg data collection
- **pywebview thread** (if desktop): Runs the native window event loop

The cache in `bloomberg_fx.py` uses `threading.Lock` for thread safety. The `_cache_wait_or_claim` mechanism prevents duplicate Bloomberg requests when multiple callbacks request the same data simultaneously.

## Styling System

All visual styling is centralized in `core/theme.py`:

### Color Constants (`COLORS` dict)

```python
bg_primary      = "#000000"     # Pure black background
accent_orange   = "#ff8800"     # Bloomberg orange (interactive elements)
accent_green    = "#00cc66"     # Profit / positive values
accent_red      = "#ff3333"     # Loss / negative values
border          = "#222240"     # Structural borders
text_primary    = "#d4d4d4"     # Data text
text_secondary  = "#808080"     # Labels, muted text
text_header     = "#ffffff"     # Section headers only
```

### Reusable Styles

| Export | Usage |
|--------|-------|
| `CARD_STYLE` | Panel section containers |
| `BUTTON_STYLE` | All buttons |
| `INPUT_STYLE` | Text inputs |
| `DROPDOWN_STYLE` | Dropdown selectors |
| `STAT_BOX_STYLE` | KPI stat boxes |
| `TAB_STYLE` / `TAB_SELECTED_STYLE` | Tab buttons |
| `CHART_TEMPLATE` | Plotly figure template |
| `chart_layout(**overrides)` | Build figure layouts with defaults |

### Chart Sizing

```python
CHART_SM = 240   # Small charts (px height)
CHART_MD = 320   # Medium charts
CHART_LG = 400   # Large charts
GAP = "8px"      # Spacing between elements
SECTION_GAP = "16px"  # Spacing between sections
```

## CSV Export

`core/csv_export.py` provides a generic mechanism to export any Plotly figure to CSV:

1. `figure_to_dataframe(fig_dict)` extracts trace data (x, y, z, text, customdata) from a Plotly figure dict and returns a pandas DataFrame
2. `export_csv(fig_dict, panel_name, chart_type)` writes the DataFrame to the `downloads/` directory and returns the content for `dcc.Download`

Each panel includes a `dcc.Download` component and a callback that calls `export_csv()` when the user clicks the export button.
