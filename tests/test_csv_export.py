"""Unit tests for core.csv_export module."""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from core.csv_export import (
    _trace_get,
    export_csv,
    figure_to_dataframe,
    set_downloads_dir,
)


# ---------------------------------------------------------------------------
# _trace_get helper
# ---------------------------------------------------------------------------

class TestTraceGet:
    """Tests for the _trace_get helper that abstracts dict / graph-object access."""

    def test_dict_existing_key(self):
        assert _trace_get({"name": "foo"}, "name") == "foo"

    def test_dict_missing_key_returns_default(self):
        assert _trace_get({"a": 1}, "b") is None

    def test_dict_missing_key_custom_default(self):
        assert _trace_get({"a": 1}, "b", "fallback") == "fallback"

    def test_graph_object_attribute(self):
        trace = go.Scatter(x=[1, 2], y=[3, 4], name="series_a")
        assert _trace_get(trace, "name") == "series_a"

    def test_graph_object_missing_attribute_returns_default(self):
        trace = go.Scatter(x=[1], y=[2])
        # "customdata" was never set, so attribute returns None -> treated as default
        assert _trace_get(trace, "customdata", "fallback") == "fallback"

    def test_graph_object_none_treated_as_missing(self):
        """Plotly returns None for unset optional attrs; _trace_get should map to default."""
        trace = go.Scatter(x=[1], y=[2])
        # 'text' is unset -> None -> should return default
        assert _trace_get(trace, "text", "default_text") == "default_text"


# ---------------------------------------------------------------------------
# figure_to_dataframe – scatter traces
# ---------------------------------------------------------------------------

class TestScatterTraces:
    """Scatter / line chart extraction."""

    def test_single_scatter_dict(self):
        fig = {
            "data": [{"type": "scatter", "x": [1, 2, 3], "y": [10, 20, 30], "name": "vals"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert list(df.columns) == ["vals"]
        assert list(df["vals"]) == [10, 20, 30]

    def test_scatter_implicit_type(self):
        """Traces with no explicit type default to 'scatter'."""
        fig = {
            "data": [{"x": [1, 2], "y": [5, 6], "name": "s"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert not df.empty
        assert "s" in df.columns

    def test_scatter_index_is_x_values(self):
        fig = {
            "data": [{"type": "scatter", "x": [10, 20], "y": [1, 2], "name": "a"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert list(df.index) == [10.0, 20.0]

    def test_scatter_x_title_propagates_to_index_name(self):
        fig = {
            "data": [{"x": [1], "y": [2], "name": "v"}],
            "layout": {"xaxis": {"title": {"text": "Date"}}},
        }
        df = figure_to_dataframe(fig)
        assert df.index.name == "Date"

    def test_scatter_x_title_string_format(self):
        """Some figures store title as a plain string instead of dict."""
        fig = {
            "data": [{"x": [1], "y": [2], "name": "v"}],
            "layout": {"xaxis": {"title": "Tenor"}},
        }
        df = figure_to_dataframe(fig)
        assert df.index.name == "Tenor"


# ---------------------------------------------------------------------------
# figure_to_dataframe – bar traces
# ---------------------------------------------------------------------------

class TestBarTraces:
    """Bar chart extraction."""

    def test_single_bar(self):
        fig = {
            "data": [{"type": "bar", "x": ["A", "B", "C"], "y": [10, 20, 30], "name": "bars"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert list(df.columns) == ["bars"]
        assert list(df["bars"]) == [10, 20, 30]
        assert list(df.index) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# figure_to_dataframe – heatmap traces
# ---------------------------------------------------------------------------

class TestHeatmapTraces:
    """Heatmap / surface / heatmapgl extraction."""

    def test_basic_heatmap(self):
        z = [[1, 2], [3, 4]]
        fig = {
            "data": [{"type": "heatmap", "z": z, "x": ["a", "b"], "y": ["r0", "r1"]}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["a", "b"]
        assert list(df.index) == ["r0", "r1"]
        assert df.iloc[0, 0] == 1
        assert df.iloc[1, 1] == 4

    def test_heatmap_auto_axis_labels(self):
        """When x/y are omitted, integer range labels should be generated."""
        z = [[10, 20, 30], [40, 50, 60]]
        fig = {
            "data": [{"type": "heatmap", "z": z}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.shape == (2, 3)
        assert list(df.columns) == [0, 1, 2]
        assert list(df.index) == [0, 1]

    def test_surface_treated_as_heatmap(self):
        z = [[5, 6], [7, 8]]
        fig = {
            "data": [{"type": "surface", "z": z, "x": ["c1", "c2"], "y": ["r1", "r2"]}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.shape == (2, 2)
        assert df.iloc[1, 0] == 7

    def test_heatmapgl_treated_as_heatmap(self):
        z = [[1]]
        fig = {
            "data": [{"type": "heatmapgl", "z": z, "x": ["x"], "y": ["y"]}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.shape == (1, 1)

    def test_heatmap_empty_z(self):
        fig = {
            "data": [{"type": "heatmap", "z": []}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.empty

    def test_heatmap_z_none(self):
        fig = {
            "data": [{"type": "heatmap", "z": None}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.empty

    @pytest.mark.xfail(
        reason="Bug: `z = t.get('z') or []` raises ValueError on numpy arrays "
               "because numpy truth value is ambiguous. Should use "
               "`z = t.get('z'); if z is None: z = []`",
        raises=ValueError,
        strict=True,
    )
    def test_heatmap_numpy_z_raw(self):
        """Passing a raw numpy array as z triggers a ValueError (known bug)."""
        z = np.array([[1.5, 2.5], [3.5, 4.5]])
        fig = {
            "data": [{"type": "heatmap", "z": z, "x": ["a", "b"], "y": ["r0", "r1"]}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.shape == (2, 2)

    def test_heatmap_numpy_z_as_list(self):
        """Numpy z converted to list works fine (workaround for the raw-array bug)."""
        z = np.array([[1.5, 2.5], [3.5, 4.5]]).tolist()
        fig = {
            "data": [{"type": "heatmap", "z": z, "x": ["a", "b"], "y": ["r0", "r1"]}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.shape == (2, 2)
        assert df.iloc[0, 1] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# figure_to_dataframe – multiple traces
# ---------------------------------------------------------------------------

class TestMultipleTraces:
    """Multi-trace figures merged on shared x-axis."""

    def test_two_scatter_same_x(self):
        fig = {
            "data": [
                {"x": [1, 2, 3], "y": [10, 20, 30], "name": "A"},
                {"x": [1, 2, 3], "y": [100, 200, 300], "name": "B"},
            ],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert set(df.columns) == {"A", "B"}
        assert len(df) == 3
        assert df.loc[2.0, "B"] == 200

    def test_two_scatter_different_x(self):
        """Outer join should produce NaN where x values don't overlap."""
        fig = {
            "data": [
                {"x": [1, 2], "y": [10, 20], "name": "A"},
                {"x": [2, 3], "y": [200, 300], "name": "B"},
            ],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert len(df) == 3  # x = 1, 2, 3
        assert pd.isna(df.loc[1.0, "B"])
        assert pd.isna(df.loc[3.0, "A"])
        assert df.loc[2.0, "A"] == 20
        assert df.loc[2.0, "B"] == 200

    def test_duplicate_trace_names_deduplicated(self):
        fig = {
            "data": [
                {"x": [1], "y": [10], "name": "series"},
                {"x": [1], "y": [20], "name": "series"},
            ],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert len(df.columns) == 2
        # Second column should be renamed to avoid collision
        assert "series" in df.columns

    def test_mixed_bar_and_scatter(self):
        fig = {
            "data": [
                {"type": "scatter", "x": [1, 2], "y": [10, 20], "name": "line"},
                {"type": "bar", "x": [1, 2], "y": [5, 15], "name": "bars"},
            ],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert set(df.columns) == {"line", "bars"}


# ---------------------------------------------------------------------------
# figure_to_dataframe – empty / invalid input
# ---------------------------------------------------------------------------

class TestEmptyAndInvalid:
    """Edge cases for missing, empty, or invalid figure inputs."""

    def test_none_input(self):
        df = figure_to_dataframe(None)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_dict(self):
        df = figure_to_dataframe({})
        assert df.empty

    def test_dict_without_data_key(self):
        df = figure_to_dataframe({"layout": {}})
        assert df.empty

    def test_empty_data_list(self):
        df = figure_to_dataframe({"data": [], "layout": {}})
        assert df.empty

    def test_trace_with_no_x(self):
        """A trace missing x should be silently skipped."""
        fig = {
            "data": [{"y": [1, 2], "name": "no_x"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.empty

    def test_trace_with_no_y(self):
        """A trace missing y should be silently skipped."""
        fig = {
            "data": [{"x": [1, 2], "name": "no_y"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.empty

    def test_trace_with_no_name(self):
        """Unnamed traces should get auto-generated names like trace_0."""
        fig = {
            "data": [{"x": [1, 2], "y": [3, 4]}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert not df.empty
        assert "trace_0" in df.columns


# ---------------------------------------------------------------------------
# figure_to_dataframe – go.Figure objects
# ---------------------------------------------------------------------------

class TestGoFigureObjects:
    """Ensure go.Figure objects (not just dicts) are handled correctly."""

    def test_go_figure_scatter(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 20, 30], name="test"))
        df = figure_to_dataframe(fig)
        assert "test" in df.columns
        assert list(df["test"]) == [10, 20, 30]

    def test_go_figure_bar(self):
        fig = go.Figure(data=[go.Bar(x=["a", "b"], y=[5, 10], name="bars")])
        df = figure_to_dataframe(fig)
        assert "bars" in df.columns
        assert list(df["bars"]) == [5, 10]

    def test_go_figure_heatmap(self):
        fig = go.Figure(data=go.Heatmap(
            z=[[1, 2], [3, 4]],
            x=["c1", "c2"],
            y=["r1", "r2"],
        ))
        df = figure_to_dataframe(fig)
        assert df.shape == (2, 2)

    def test_go_figure_multiple_traces(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[10, 20], name="alpha"))
        fig.add_trace(go.Scatter(x=[1, 2], y=[30, 40], name="beta"))
        df = figure_to_dataframe(fig)
        assert set(df.columns) == {"alpha", "beta"}

    def test_go_figure_with_layout_title(self):
        fig = go.Figure(data=[go.Scatter(x=[1], y=[2], name="s")])
        fig.update_layout(xaxis_title="Strike")
        df = figure_to_dataframe(fig)
        assert df.index.name == "Strike"

    def test_go_figure_empty(self):
        fig = go.Figure()
        df = figure_to_dataframe(fig)
        assert df.empty


# ---------------------------------------------------------------------------
# figure_to_dataframe – histogram traces
# ---------------------------------------------------------------------------

class TestHistogramTraces:
    """Histogram traces: the module re-bins raw values via numpy."""

    def test_basic_histogram(self):
        fig = {
            "data": [{"type": "histogram", "x": [1, 1, 2, 3, 3, 3], "name": "counts"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert "bin_start" in df.columns
        assert "bin_end" in df.columns
        assert "counts" in df.columns
        assert df["counts"].sum() == 6  # total count equals input length

    def test_histogram_empty_values(self):
        fig = {
            "data": [{"type": "histogram", "x": [], "name": "empty"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.empty


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------

class TestExportCsv:
    """Tests for export_csv (writes CSV to disk and returns dcc.send_string)."""

    def test_export_writes_file_to_disk(self):
        fig = {
            "data": [{"x": [1, 2], "y": [10, 20], "name": "vals"}],
            "layout": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            set_downloads_dir(tmpdir)
            try:
                result = export_csv(fig, "TestPanel", "TestChart")
                # A file should have been created
                files = os.listdir(tmpdir)
                assert len(files) == 1
                assert files[0].startswith("TestPanel_TestChart_")
                assert files[0].endswith(".csv")
                # Verify content
                content = open(os.path.join(tmpdir, files[0])).read()
                assert "vals" in content
            finally:
                set_downloads_dir(None)

    def test_export_empty_figure_returns_no_update(self):
        from dash import no_update

        result = export_csv({"data": [], "layout": {}}, "P", "C")
        assert result is no_update

    def test_export_none_figure_returns_no_update(self):
        from dash import no_update

        result = export_csv(None, "P", "C")
        assert result is no_update

    def test_export_returns_dict_for_dcc_download(self):
        fig = {
            "data": [{"x": [1], "y": [2], "name": "v"}],
            "layout": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            set_downloads_dir(tmpdir)
            try:
                result = export_csv(fig, "Panel", "Chart")
                # dcc.send_string returns a dict with 'content', 'filename', etc.
                assert isinstance(result, dict)
                assert "content" in result
                assert "filename" in result
                assert result["filename"].startswith("Panel_Chart_")
            finally:
                set_downloads_dir(None)

    def test_export_without_downloads_dir(self):
        """When no download directory is set, should still return dcc data."""
        fig = {
            "data": [{"x": [1], "y": [2], "name": "v"}],
            "layout": {},
        }
        set_downloads_dir(None)
        result = export_csv(fig, "P", "C")
        assert isinstance(result, dict)
        assert "content" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_traces_with_different_lengths(self):
        """Traces with different x lengths should merge via outer join."""
        fig = {
            "data": [
                {"x": [1, 2, 3, 4], "y": [10, 20, 30, 40], "name": "long"},
                {"x": [2, 3], "y": [200, 300], "name": "short"},
            ],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert len(df) == 4
        assert pd.isna(df.loc[1.0, "short"])
        assert pd.isna(df.loc[4.0, "short"])
        assert df.loc[2.0, "short"] == 200

    def test_string_x_axis(self):
        """Non-numeric x values (dates, labels) should work."""
        fig = {
            "data": [{"x": ["2024-01-01", "2024-02-01"], "y": [1, 2], "name": "ts"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert len(df) == 2
        assert "ts" in df.columns

    def test_single_point_trace(self):
        fig = {
            "data": [{"x": [42], "y": [99], "name": "single"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert len(df) == 1
        assert df["single"].iloc[0] == 99

    def test_large_heatmap(self):
        z = np.random.rand(50, 30).tolist()
        fig = {
            "data": [{"type": "heatmap", "z": z}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert df.shape == (50, 30)

    def test_heatmap_with_mixed_non_heatmap_traces(self):
        """When heatmap is present alongside scatter, only heatmap is extracted."""
        fig = {
            "data": [
                {"type": "heatmap", "z": [[1, 2], [3, 4]], "x": ["a", "b"], "y": ["r0", "r1"]},
                {"type": "scatter", "x": [1], "y": [2], "name": "overlay"},
            ],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        # Heatmap takes priority
        assert df.shape == (2, 2)

    def test_no_layout_key(self):
        """Figure dict with data but no layout key should still work."""
        fig = {
            "data": [{"x": [1, 2], "y": [3, 4], "name": "v"}],
        }
        df = figure_to_dataframe(fig)
        assert not df.empty
        # index name falls back to "" when layout missing
        assert df.index.name == ""

    def test_output_is_sorted_by_x(self):
        """Result should be sorted by x index even if input is unsorted."""
        fig = {
            "data": [{"x": [3, 1, 2], "y": [30, 10, 20], "name": "unsorted"}],
            "layout": {},
        }
        df = figure_to_dataframe(fig)
        assert list(df.index) == [1.0, 2.0, 3.0]
        assert list(df["unsorted"]) == [10, 20, 30]
