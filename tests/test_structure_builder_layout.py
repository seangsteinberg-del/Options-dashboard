"""
Layout-stability regression tests for the structure_builder leg row.

Bug history:
  - The leg row was redesigned into 3 stacked sub-rows
    (TYPE/SIDE/DELTA, RATIO/TNR×, STRIKE/VOL/PREM) for use in a 350px sidebar.
  - But `toggle_leg_rows` (a runtime callback) was hardcoded to set
    `display: flex` (row direction) on the outer container — which placed
    the 3 sub-rows side-by-side after every Add Leg / Remove Leg / preset
    change, breaking the new vertical layout.
  - The fix factored the outer style into `_leg_row_outer_style` so both
    the initial render and the runtime callback share one source of truth.
"""
from panels.structure_builder import (
    _make_leg_row, _leg_row_outer_style, _leg_row_styles, MAX_LEGS,
)


class TestLegRowOuterStyle:
    def test_visible_is_block(self):
        s = _leg_row_outer_style(0, visible=True)
        assert s["display"] == "block"

    def test_hidden_is_none(self):
        s = _leg_row_outer_style(0, visible=False)
        assert s["display"] == "none"

    def test_visible_carries_chrome(self):
        s = _leg_row_outer_style(0, visible=True)
        assert "padding" in s
        assert "borderLeft" in s
        assert "marginBottom" in s

    def test_alternating_background(self):
        even = _leg_row_outer_style(0, visible=True)["backgroundColor"]
        odd = _leg_row_outer_style(1, visible=True)["backgroundColor"]
        assert even != odd  # alternating row stripes

    def test_initial_render_uses_same_style(self):
        """_make_leg_row's outer container must apply _leg_row_outer_style
        — the runtime toggle_leg_rows callback uses the same."""
        leg = _make_leg_row(0, cp="call", side="buy", delta=0.5,
                            ratio=1, tenor_mult=1.0, visible=True)
        expected = _leg_row_outer_style(0, visible=True)
        assert dict(leg.style) == expected


class TestLegRowStylesList:
    def test_correct_length(self):
        styles = _leg_row_styles(2)
        assert len(styles) == MAX_LEGS

    def test_visibility_matches_num_legs(self):
        styles = _leg_row_styles(3)
        assert styles[0]["display"] == "block"
        assert styles[1]["display"] == "block"
        assert styles[2]["display"] == "block"
        for s in styles[3:]:
            assert s["display"] == "none"

    def test_clamps_above_max(self):
        styles = _leg_row_styles(99)
        assert all(s["display"] == "block" for s in styles)

    def test_clamps_below_one(self):
        styles = _leg_row_styles(0)
        assert styles[0]["display"] == "block"
        for s in styles[1:]:
            assert s["display"] == "none"

    def test_handles_none_input(self):
        styles = _leg_row_styles(None)
        # None should be treated like 0/1 — at least one visible
        assert styles[0]["display"] == "block"
