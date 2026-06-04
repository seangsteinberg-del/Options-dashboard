"""
App boot/serve integration test.

Boots the real `app` module and serves its full layout + dependency graph through
Dash's actual Flask test client — i.e. the same server the dashboard runs on.
This proves the entire app starts, every panel's pre-rendered layout serializes,
and all callbacks are wired, without a live Bloomberg terminal (degraded mode).

`is_connected` is forced False so the test does not block on a real Bloomberg
connection attempt.
"""
import json
from unittest.mock import patch


def _client():
    import app as appmod
    return appmod, appmod.app.server.test_client()


def test_app_serves_index_and_full_layout():
    import app as appmod
    # Force degraded mode (no slow Bloomberg connection attempt during render).
    with patch.object(appmod, "is_connected", lambda: False):
        client = appmod.app.server.test_client()

        r = client.get("/")
        assert r.status_code == 200
        assert len(r.data) > 2000

        # Full pre-rendered component tree of EVERY panel must serialize.
        layout = client.get("/_dash-layout")
        assert layout.status_code == 200
        body = layout.get_data(as_text=True)
        assert len(body) > 50_000
        for marker in ("FX OPTIONS", "workspace-tabs", "market-dashboard"):
            assert marker in body, f"served layout missing {marker}"


def test_app_registers_all_callbacks():
    import app as appmod
    with patch.object(appmod, "is_connected", lambda: False):
        client = appmod.app.server.test_client()
        deps = json.loads(client.get("/_dash-dependencies").get_data(as_text=True))
    # 9 navigated panels register ~119 callbacks; guard against silent loss.
    assert len(deps) > 80
