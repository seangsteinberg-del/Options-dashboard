"""CSV export utility for Plotly figures."""

from datetime import datetime
import numpy as np
import pandas as pd
from dash import dcc


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
        t = next(tr for tr in traces if tr.get("type") in ("heatmap", "surface", "heatmapgl"))
        z = t.get("z", [])
        x = t.get("x", list(range(len(z[0]) if z else 0)))
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
        existing = {f.columns[-1] for f in frames}
        if name in existing:
            name = f"{name}_{len(frames)}"
        s = pd.DataFrame({"x": x, name: y})
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="x", how="outer")
    df = df.set_index("x").sort_index()
    df.index.name = ""
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
    df = figure_to_dataframe(fig_dict)
    if df.empty:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{panel_name}_{chart_type}_{ts}.csv"
    return dcc.send_data_frame(df.to_csv, filename)
