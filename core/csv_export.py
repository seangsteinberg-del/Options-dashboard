"""CSV export utility for Plotly figures."""

import logging
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from dash import dcc, no_update

logger = logging.getLogger(__name__)


def figure_to_dataframe(fig_dict):
    """Extract data from a Plotly figure dict into a pandas DataFrame.

    Handles scatter/bar (columnar), heatmap/surface (matrix), and histogram traces.
    Multi-trace figures are merged on shared x-axis via outer join.
    """
    if not fig_dict or "data" not in fig_dict:
        return pd.DataFrame()

    traces = fig_dict["data"]
    if not traces:
        return pd.DataFrame()

    # Detect trace types
    trace_types = {t.get("type", "scatter") for t in traces}

    # ── Heatmap / Surface → matrix format ──
    if trace_types & {"heatmap", "surface", "heatmapgl"}:
        t = next((tr for tr in traces if tr.get("type") in ("heatmap", "surface", "heatmapgl")), None)
        if t is None:
            return pd.DataFrame()
        z = t.get("z") or []
        if not z or not isinstance(z, (list, np.ndarray)) or len(z) == 0:
            return pd.DataFrame()
        first_row = z[0] if isinstance(z[0], (list, np.ndarray)) else []
        x = t.get("x", list(range(len(first_row))))
        y = t.get("y", list(range(len(z))))
        return pd.DataFrame(z, index=y, columns=x)

    # ── Histogram → bin edges + counts ──
    if trace_types == {"histogram"}:
        frames = []
        for t in traces:
            name = t.get("name", "value")
            vals = t.get("x") or t.get("y") or []
            if not len(vals):
                continue
            arr = np.array(vals, dtype=float)
            counts, edges = np.histogram(arr, bins="auto")
            frames.append(pd.DataFrame({
                "bin_start": edges[:-1],
                "bin_end": edges[1:],
                name: counts,
            }))
        if not frames:
            return pd.DataFrame()
        df = frames[0]
        for f in frames[1:]:
            df = df.merge(f, on=["bin_start", "bin_end"], how="outer")
        return df

    # ── Scatter / Bar / Line → columnar, merged on x ──
    frames = []
    for t in traces:
        ttype = t.get("type", "scatter")
        if ttype in ("heatmap", "surface", "heatmapgl"):
            continue
        x = t.get("x")
        y = t.get("y")
        if x is None or y is None:
            continue
        name = t.get("name") or f"trace_{len(frames)}"
        # Deduplicate column names
        existing = {f.columns[-1] for f in frames if len(f.columns) > 0}
        if name in existing:
            name = f"{name}_{len(frames)}"
        s = pd.DataFrame({"x": x, name: y})
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    # Ensure consistent x-column dtype across traces to avoid int/float merge warnings
    all_numeric = all(pd.api.types.is_numeric_dtype(f["x"]) for f in frames)
    if all_numeric:
        for f in frames:
            f["x"] = f["x"].astype(float)

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="x", how="outer")
    df = df.set_index("x").sort_index()
    # Preserve a meaningful index name from axis title if available
    xaxis = fig_dict.get("layout", {}).get("xaxis", {})
    x_title = xaxis.get("title", {})
    if isinstance(x_title, dict):
        x_title = x_title.get("text", "")
    df.index.name = x_title if x_title else ""
    return df


def export_csv(fig_dict, panel_name, chart_type):
    """Return a dcc.send_data_frame dict for downloading CSV from a figure.

    Parameters
    ----------
    fig_dict : dict  — Plotly figure dictionary
    panel_name : str — e.g. "VolSurface"
    chart_type : str — e.g. "ATM_History"

    Returns
    -------
    dict suitable for dcc.Download data property
    """
    try:
        df = figure_to_dataframe(fig_dict)
        if df.empty:
            logger.warning("CSV export: empty DataFrame for %s/%s", panel_name, chart_type)
            return no_update
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{panel_name}_{chart_type}_{ts}.csv"
        csv_string = df.to_csv()
        return dcc.send_string(csv_string, filename)
    except Exception:
        traceback.print_exc()
        return no_update
