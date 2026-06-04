"""
Data-integrity indicator tests.

This is the mechanism that protects a trader from a flaky / disconnected
terminal: the header must show LIVE only when data is genuinely live, and must
visibly warn (DEGRADED / DISCONNECTED) otherwise. Verifying it closes the
"connected-but-flaky terminal could mislead" risk — the data-state is always
transparent, so a trader never acts on silently-stale or absent data.
"""
import time
from unittest.mock import patch

import core.bloomberg_fx as bfx


class TestDataMode:
    def test_disconnected_when_not_connected(self, monkeypatch):
        monkeypatch.setattr(bfx, "is_connected", lambda: False)
        assert bfx.get_data_mode() == "DISCONNECTED"

    def test_live_when_connected_and_no_errors(self, monkeypatch):
        monkeypatch.setattr(bfx, "is_connected", lambda: True)
        monkeypatch.setattr(bfx, "_HAS_EQUITY_BBG", True, raising=False)
        bfx.clear_errors()
        assert bfx.get_data_mode() == "LIVE"

    def test_degraded_when_connected_but_many_failures(self, monkeypatch):
        monkeypatch.setattr(bfx, "is_connected", lambda: True)
        monkeypatch.setattr(bfx, "_HAS_EQUITY_BBG", True, raising=False)
        bfx.clear_errors()
        now = time.time()
        with bfx._errors_lock:
            for i in range(6):
                bfx._fetch_errors.append({"time": now, "function": "get_fx_spots",
                                          "pair": "EURUSD", "error": "timeout"})
        try:
            assert bfx.get_data_mode() == "DEGRADED"
        finally:
            bfx.clear_errors()

    def test_stale_errors_do_not_degrade(self, monkeypatch):
        """Errors older than the 300s window must not keep the desk in DEGRADED."""
        monkeypatch.setattr(bfx, "is_connected", lambda: True)
        monkeypatch.setattr(bfx, "_HAS_EQUITY_BBG", True, raising=False)
        bfx.clear_errors()
        old = time.time() - 1000
        with bfx._errors_lock:
            for i in range(6):
                bfx._fetch_errors.append({"time": old, "function": "get_fx_spots",
                                          "pair": "EURUSD", "error": "timeout"})
        try:
            assert bfx.get_data_mode() == "LIVE"
        finally:
            bfx.clear_errors()


class TestStatusBadge:
    """The header callback must translate each mode into a clear, correctly
    coloured trader-facing warning — never a silent or 'live' label when not."""

    def _badge_text(self, mode, errors=None):
        import app as appmod
        import dash
        app2 = dash.Dash("status", suppress_callback_exceptions=True)
        # find the data-source-status callback on the real app
        key = next(k for k in appmod.app.callback_map if "data-source-status.children" in k)
        fn = appmod.app.callback_map[key]["callback"]
        fn = getattr(fn, "__wrapped__", fn)
        with patch.object(bfx, "get_data_mode", lambda: mode), \
             patch.object(bfx, "get_recent_errors", lambda: errors or []):
            children, ts = fn(1)
        # flatten to text
        import json
        return json.dumps(_to_plain(children))

    def test_live_badge(self):
        txt = self._badge_text("LIVE")
        assert "LIVE" in txt and "DEGRADED" not in txt and "DISCONNECTED" not in txt

    def test_degraded_badge_warns(self):
        errs = [{"function": "get_fx_spots", "pair": "EURUSD", "error": "timeout"}] * 6
        txt = self._badge_text("DEGRADED", errs)
        assert "DEGRADED" in txt and "failures" in txt

    def test_disconnected_badge_warns(self):
        txt = self._badge_text("DISCONNECTED")
        assert "DISCONNECTED" in txt and "NO LIVE DATA" in txt


def _to_plain(component):
    """Recursively extract text/props from a Dash component tree for assertions."""
    if isinstance(component, (list, tuple)):
        return [_to_plain(c) for c in component]
    if hasattr(component, "to_plotly_json"):
        j = component.to_plotly_json()
        props = j.get("props", {})
        return {"children": _to_plain(props.get("children")),
                "style": props.get("style", {}).get("color", "")}
    return component
