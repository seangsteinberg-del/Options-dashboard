"""
FX Trade Idea Screener
======================
Scans the cached vol surfaces, IV-RV histories, and rate differentials across
the FX pair registry and surfaces ranked tactical trade candidates.

Each screen is a pure function over `core.fx_analytics` primitives. Screens
return `List[TradeIdea]` records that include:
    - the pair / tenor / category / direction
    - a normalised 0-10 score (higher = stronger signal)
    - a `structure_preset` key into `panels.structure_builder.PRESETS`
    - a `vol_view`     key into `panels.vol_surface_fx.VIEW_PRESETS`
    - a one-line thesis and a small `metrics` dict for the table

The engine is intentionally cache-only: it never hits Bloomberg directly.
All upstream calls are routed through `core.fx_analytics`, which reads
`core.bloomberg_fx`'s in-memory cache (filled by `core.bg_fetcher` on a
2-minute cycle). This keeps `scan_all()` cheap enough to be invoked from
every panel callback that wants fresh ideas.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.fx_analytics import (
    vol_percentile, vol_zscore, iv_rv_percentile,
    carry_per_vol, vol_of_vol,
)
from core.fx_conventions import FX_PAIR_REGISTRY
from core.bloomberg_fx import get_fx_vol_surface

logger = logging.getLogger(__name__)


# ============================================================================
# Output type
# ============================================================================

@dataclass
class TradeIdea:
    """A single trade candidate surfaced by one of the screens."""
    pair: str
    tenor: str
    category: str            # human-readable screen name
    direction: str           # "long_vol", "short_vol", "long_skew", "short_skew",
                             # "calendar", "carry", "tail"
    score: float             # normalised 0-10 (higher = stronger)
    structure_preset: str    # PRESETS key in panels.structure_builder
    vol_view: str            # VIEW_PRESETS key in panels.vol_surface_fx
    thesis: str              # one-sentence rationale
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flat dict suitable for a Dash DataTable row."""
        d = asdict(self)
        d.pop("metrics", None)
        d["score"] = round(self.score, 2)
        return d


# ============================================================================
# Category → Structure preset & VOL view defaults
# ============================================================================
#
# Each category has a default structure recommendation and a VOL-panel view
# preset. The structure preset names must match keys in
# `panels.structure_builder.PRESETS`; the vol_view names must match keys in
# `panels.vol_surface_fx.VIEW_PRESETS`. Both are checked at unit-test time.

CATEGORY_DEFAULTS = {
    # category                  preset                   vol_view
    "Cheap Vol":               ("Straddle",              "Surface"),
    "Rich Vol":                ("Iron Condor",           "Surface"),
    "IV Rich vs RV":           ("Strangle",              "RV Analysis"),
    "IV Cheap vs RV":          ("Straddle",              "RV Analysis"),
    "Skew Rich (Calls)":       ("Risk Reversal",         "Skew"),
    "Skew Rich (Puts)":        ("Risk Reversal",         "Skew"),
    "Wings Rich":              ("Iron Condor",           "Smile Deep"),
    "Wings Cheap":             ("10D Strangle",          "Smile Deep"),
    "Term Inversion":          ("Calendar Spread",       "Term"),
    "Term Kink":               ("Calendar Spread",       "Term"),
    "Carry / Vol":             ("Risk Reversal",         "Carry"),
    "Vol-of-Vol Extreme":      ("Strangle",              "Vol-of-Vol"),
    "Tail Underpriced":        ("10D Strangle",          "Tail Risk"),
}

ALL_CATEGORIES = list(CATEGORY_DEFAULTS.keys())


# ============================================================================
# Pair universe helpers
# ============================================================================

def default_pair_universe() -> List[str]:
    """All registered pairs ordered by liquidity tier then alphabetical."""
    return sorted(FX_PAIR_REGISTRY.keys(),
                  key=lambda p: (FX_PAIR_REGISTRY[p].liquidity_tier, p))


def filter_pairs(region: str = "ALL") -> List[str]:
    """Filter the universe by region label used in the UI."""
    region = (region or "ALL").upper()
    pairs = default_pair_universe()
    if region == "ALL":
        return pairs
    if region == "G10":
        return [p for p in pairs if FX_PAIR_REGISTRY[p].group == "G10"]
    if region == "EM":
        return [p for p in pairs if FX_PAIR_REGISTRY[p].group != "G10"]
    if region == "MAJORS":
        return [p for p in pairs if FX_PAIR_REGISTRY[p].subgroup == "Majors"]
    return pairs


def filter_tenors(tenor_bucket: str = "ALL") -> List[str]:
    """Map a UI bucket to the tenors each screen should iterate over."""
    bucket = (tenor_bucket or "ALL").upper()
    if bucket == "FRONT":
        return ["1W", "2W", "1M"]
    if bucket == "BELLY":
        return ["1M", "2M", "3M"]
    if bucket == "BACK":
        return ["6M", "9M", "1Y"]
    return ["1W", "1M", "3M", "6M", "1Y"]


# ============================================================================
# Score helpers
# ============================================================================

def _score_from_pct(pct: float, low: float = 20.0, high: float = 80.0) -> float:
    """
    Map a percentile to a 0-10 score where the extremes (0 / 100) score 10
    and `low` / `high` thresholds score 5. Inside (low..high) the score
    fades linearly to 0 at the midpoint.
    """
    if pct is None or not np.isfinite(pct):
        return 0.0
    if pct <= low:
        # 0 → 10, low → 5
        return 5.0 + 5.0 * (low - pct) / max(low, 1e-6)
    if pct >= high:
        # 100 → 10, high → 5
        return 5.0 + 5.0 * (pct - high) / max(100.0 - high, 1e-6)
    # Inside the band: 5 at the threshold edge, fading to 0 at midpoint
    mid = 50.0
    width = max(mid - low, high - mid, 1e-6)
    return max(0.0, 5.0 * abs(pct - mid) / width)


def _score_from_z(z: float, threshold: float = 2.0) -> float:
    """
    Map a z-score to 0-10. z=threshold scores 5, z=2*threshold scores 10,
    saturating thereafter.
    """
    if z is None or not np.isfinite(z):
        return 0.0
    az = abs(z)
    if az < threshold:
        return max(0.0, 5.0 * az / max(threshold, 1e-6))
    return min(10.0, 5.0 + 5.0 * (az - threshold) / max(threshold, 1e-6))


# ============================================================================
# Individual screens
# ============================================================================
#
# Each screen takes (pairs, tenors) and returns List[TradeIdea]. They all
# tolerate missing data — a None or short history just means "no idea here".


def screen_cheap_vol(pairs: List[str], tenors: List[str],
                     percentile_threshold: float = 20.0) -> List[TradeIdea]:
    """ATM vol percentile is in the bottom tail → buy gamma."""
    out: List[TradeIdea] = []
    preset, view = CATEGORY_DEFAULTS["Cheap Vol"]
    for pair in pairs:
        for tenor in tenors:
            info = vol_percentile(pair, tenor, "ATM")
            if not info:
                continue
            pct = info.get("percentile")
            if pct is None or pct >= percentile_threshold:
                continue
            score = _score_from_pct(pct, low=percentile_threshold, high=80.0)
            atm = info.get("current", 0.0)
            out.append(TradeIdea(
                pair=pair, tenor=tenor, category="Cheap Vol",
                direction="long_vol", score=score,
                structure_preset=preset, vol_view=view,
                thesis=(f"{tenor} ATM vol at {pct:.0f}th pctile "
                        f"({atm:.1f}%) — sub-{percentile_threshold:.0f}th. "
                        f"Long gamma cheaply funded; mean-reversion bias."),
                metrics={"percentile": pct, "atm_vol": atm,
                         "mean": info.get("mean", 0.0)},
            ))
    return out


def screen_rich_vol(pairs: List[str], tenors: List[str],
                    percentile_threshold: float = 80.0) -> List[TradeIdea]:
    """ATM vol percentile is in the top tail → sell premium."""
    out: List[TradeIdea] = []
    preset, view = CATEGORY_DEFAULTS["Rich Vol"]
    for pair in pairs:
        for tenor in tenors:
            info = vol_percentile(pair, tenor, "ATM")
            if not info:
                continue
            pct = info.get("percentile")
            if pct is None or pct <= percentile_threshold:
                continue
            score = _score_from_pct(pct, low=20.0, high=percentile_threshold)
            atm = info.get("current", 0.0)
            out.append(TradeIdea(
                pair=pair, tenor=tenor, category="Rich Vol",
                direction="short_vol", score=score,
                structure_preset=preset, vol_view=view,
                thesis=(f"{tenor} ATM vol at {pct:.0f}th pctile "
                        f"({atm:.1f}%) — above-{percentile_threshold:.0f}th. "
                        f"Sell premium / iron condor; expect compression."),
                metrics={"percentile": pct, "atm_vol": atm,
                         "mean": info.get("mean", 0.0)},
            ))
    return out


def screen_iv_rv_rich(pairs: List[str], tenors: List[str]) -> List[TradeIdea]:
    """IV-RV spread sits in the top quartile → IV is rich vs realized."""
    out: List[TradeIdea] = []
    preset, view = CATEGORY_DEFAULTS["IV Rich vs RV"]
    for pair in pairs:
        # IV-RV is most actionable in the belly — restrict to those tenors
        for tenor in [t for t in tenors if t in ("1M", "2M", "3M")]:
            info = iv_rv_percentile(pair, tenor, rv_window=20)
            if not info or info.get("percentile") is None:
                continue
            pct = float(info["percentile"])
            if pct <= 75.0:
                continue
            score = _score_from_pct(pct, low=20.0, high=75.0)
            iv = info.get("current_iv", 0.0)
            rv = info.get("current_rv", 0.0)
            spread = info.get("current_spread", iv - rv)
            out.append(TradeIdea(
                pair=pair, tenor=tenor, category="IV Rich vs RV",
                direction="short_vol", score=score,
                structure_preset=preset, vol_view=view,
                thesis=(f"{tenor} IV {iv:.1f}% vs 20d RV {rv:.1f}% "
                        f"(spread {spread:+.1f} pts, {pct:.0f}th pctile). "
                        f"Sell vol; harvest premium decay."),
                metrics={"iv": iv, "rv": rv, "spread": spread,
                         "percentile": pct},
            ))
    return out


def screen_iv_rv_cheap(pairs: List[str], tenors: List[str]) -> List[TradeIdea]:
    """IV-RV spread sits in the bottom quartile → IV is cheap vs realized."""
    out: List[TradeIdea] = []
    preset, view = CATEGORY_DEFAULTS["IV Cheap vs RV"]
    for pair in pairs:
        for tenor in [t for t in tenors if t in ("1M", "2M", "3M")]:
            info = iv_rv_percentile(pair, tenor, rv_window=20)
            if not info or info.get("percentile") is None:
                continue
            pct = float(info["percentile"])
            if pct >= 25.0:
                continue
            score = _score_from_pct(pct, low=25.0, high=80.0)
            iv = info.get("current_iv", 0.0)
            rv = info.get("current_rv", 0.0)
            spread = info.get("current_spread", iv - rv)
            out.append(TradeIdea(
                pair=pair, tenor=tenor, category="IV Cheap vs RV",
                direction="long_vol", score=score,
                structure_preset=preset, vol_view=view,
                thesis=(f"{tenor} IV {iv:.1f}% vs 20d RV {rv:.1f}% "
                        f"(spread {spread:+.1f} pts, {pct:.0f}th pctile). "
                        f"Long gamma; realised exceeds implied."),
                metrics={"iv": iv, "rv": rv, "spread": spread,
                         "percentile": pct},
            ))
    return out


def screen_skew_extreme(pairs: List[str], tenors: List[str],
                        z_threshold: float = 2.0) -> List[TradeIdea]:
    """25D RR z-score outside ±threshold → skew rich vs history."""
    out: List[TradeIdea] = []
    for pair in pairs:
        for tenor in tenors:
            info = vol_zscore(pair, tenor, "25D_RR")
            if not info:
                continue
            z = info.get("zscore")
            if z is None or abs(z) < z_threshold:
                continue
            score = _score_from_z(z, threshold=z_threshold)
            rr = info.get("current", 0.0)
            if z > 0:
                cat = "Skew Rich (Calls)"
                direction = "short_skew"
                thesis = (f"{tenor} 25D RR z={z:+.2f} (call-skew "
                          f"premium at {rr:+.2f}). Sell RR — fade call skew.")
            else:
                cat = "Skew Rich (Puts)"
                direction = "long_skew"
                thesis = (f"{tenor} 25D RR z={z:+.2f} (put-skew "
                          f"premium at {rr:+.2f}). Buy RR — fade put skew.")
            preset, view = CATEGORY_DEFAULTS[cat]
            out.append(TradeIdea(
                pair=pair, tenor=tenor, category=cat,
                direction=direction, score=score,
                structure_preset=preset, vol_view=view,
                thesis=thesis,
                metrics={"zscore": z, "rr": rr,
                         "mean": info.get("mean", 0.0)},
            ))
    return out


def screen_wings_extreme(pairs: List[str], tenors: List[str],
                         z_threshold: float = 2.0) -> List[TradeIdea]:
    """25D BF z-score outside ±threshold → wings expensive or cheap."""
    out: List[TradeIdea] = []
    for pair in pairs:
        for tenor in tenors:
            info = vol_zscore(pair, tenor, "25D_BF")
            if not info:
                continue
            z = info.get("zscore")
            if z is None or abs(z) < z_threshold:
                continue
            score = _score_from_z(z, threshold=z_threshold)
            bf = info.get("current", 0.0)
            if z > 0:
                cat = "Wings Rich"
                direction = "short_vol"
                thesis = (f"{tenor} 25D BF z={z:+.2f} (BF at {bf:+.2f}). "
                          f"Sell wings — convexity rich vs history.")
            else:
                cat = "Wings Cheap"
                direction = "tail"
                thesis = (f"{tenor} 25D BF z={z:+.2f} (BF at {bf:+.2f}). "
                          f"Buy 10D wings — convexity cheap.")
            preset, view = CATEGORY_DEFAULTS[cat]
            out.append(TradeIdea(
                pair=pair, tenor=tenor, category=cat,
                direction=direction, score=score,
                structure_preset=preset, vol_view=view,
                thesis=thesis,
                metrics={"zscore": z, "bf": bf,
                         "mean": info.get("mean", 0.0)},
            ))
    return out


def screen_term_inversion(pairs: List[str]) -> List[TradeIdea]:
    """Front-month ATM > 6M ATM by a meaningful margin → buy back, sell front."""
    out: List[TradeIdea] = []
    preset, view = CATEGORY_DEFAULTS["Term Inversion"]
    for pair in pairs:
        surf = get_fx_vol_surface(pair)
        if not surf:
            continue
        front = _atm(surf, "1M")
        back = _atm(surf, "6M")
        if front is None or back is None:
            continue
        diff = front - back
        if diff <= 0.4:    # less than 40bps inversion = noise
            continue
        # Score: 1pt inversion = 7, 2pt = 10
        score = float(np.clip(5.0 + diff * 2.5, 0.0, 10.0))
        out.append(TradeIdea(
            pair=pair, tenor="1M-6M", category="Term Inversion",
            direction="calendar", score=score,
            structure_preset=preset, vol_view=view,
            thesis=(f"1M ATM {front:.1f}% > 6M ATM {back:.1f}% "
                    f"(inverted by {diff:+.2f} pts). "
                    f"Sell 1M / buy 6M — calendar."),
            metrics={"front": front, "back": back, "inversion": diff},
        ))
    return out


def screen_term_kink(pairs: List[str]) -> List[TradeIdea]:
    """A local min/max in the 1M-3M-6M ATM curve flags a calendar opportunity."""
    out: List[TradeIdea] = []
    preset, view = CATEGORY_DEFAULTS["Term Kink"]
    for pair in pairs:
        surf = get_fx_vol_surface(pair)
        if not surf:
            continue
        m1, m3, m6 = _atm(surf, "1M"), _atm(surf, "3M"), _atm(surf, "6M")
        if m1 is None or m3 is None or m6 is None:
            continue
        # Linear interp expectation at 3M
        expected_3m = m1 + (m6 - m1) * (3 - 1) / (6 - 1)
        kink = m3 - expected_3m
        if abs(kink) < 0.3:    # less than 30bps deviation = no kink
            continue
        score = float(np.clip(5.0 + abs(kink) * 5.0, 0.0, 10.0))
        if kink > 0:
            thesis = (f"3M ATM {m3:.1f}% sits {kink:+.2f}pts above "
                      f"1M-6M interp ({expected_3m:.1f}%). "
                      f"Sell 3M / buy 1M+6M butterfly.")
        else:
            thesis = (f"3M ATM {m3:.1f}% sits {kink:+.2f}pts below "
                      f"1M-6M interp ({expected_3m:.1f}%). "
                      f"Buy 3M / sell 1M+6M.")
        out.append(TradeIdea(
            pair=pair, tenor="1M-3M-6M", category="Term Kink",
            direction="calendar", score=score,
            structure_preset=preset, vol_view=view,
            thesis=thesis,
            metrics={"m1": m1, "m3": m3, "m6": m6, "kink": kink},
        ))
    return out


def screen_carry_per_vol(pairs: List[str],
                         sharpe_threshold: float = 0.5) -> List[TradeIdea]:
    """High carry / vol Sharpe proxy → carry trade via short OTM put."""
    out: List[TradeIdea] = []
    preset, view = CATEGORY_DEFAULTS["Carry / Vol"]
    df = carry_per_vol(pairs)
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        sharpe = float(row.get("sharpe_proxy", 0.0))
        if sharpe < sharpe_threshold:
            continue
        score = float(np.clip(5.0 + (sharpe - sharpe_threshold) * 5.0, 0.0, 10.0))
        carry_bps = float(row.get("carry_bps", 0.0))
        atm = float(row.get("atm_3m", 0.0))
        # Direction depends on carry sign (which the analytics drops into abs)
        direction = "carry"
        out.append(TradeIdea(
            pair=str(row["pair"]), tenor="3M", category="Carry / Vol",
            direction=direction, score=score,
            structure_preset=preset, vol_view=view,
            thesis=(f"Carry {carry_bps:.0f}bps vs 3M ATM {atm:.1f}% — "
                    f"Sharpe-proxy {sharpe:.2f}. Express via RR or short "
                    f"OTM put for funded directional exposure."),
            metrics={"carry_bps": carry_bps, "atm_3m": atm,
                     "sharpe_proxy": sharpe},
        ))
    return out


def screen_vol_of_vol(pairs: List[str],
                      tenors: List[str],
                      vov_z_threshold: float = 1.5) -> List[TradeIdea]:
    """Vol-of-vol z-score very high → range-bound trade via strangle."""
    out: List[TradeIdea] = []
    preset, view = CATEGORY_DEFAULTS["Vol-of-Vol Extreme"]
    for pair in pairs:
        # vol_of_vol is most relevant in the belly
        for tenor in [t for t in tenors if t in ("1M", "3M")]:
            try:
                df = vol_of_vol(pair, tenor)
            except Exception:
                continue
            if df is None or getattr(df, "empty", True):
                continue
            # vol_of_vol returns a DataFrame with vov_<window>d columns.
            # Compute z-score of latest 60d vov vs its own history.
            col = "vov_60d" if "vov_60d" in df.columns else df.columns[-1]
            series = df[col].dropna().values
            if len(series) < 30:
                continue
            current = float(series[-1])
            mu = float(np.mean(series))
            sigma = float(np.std(series))
            if sigma <= 1e-9:
                continue
            z = (current - mu) / sigma
            if not np.isfinite(z) or abs(z) < vov_z_threshold:
                continue
            score = _score_from_z(z, threshold=vov_z_threshold)
            out.append(TradeIdea(
                pair=pair, tenor=tenor, category="Vol-of-Vol Extreme",
                direction="short_vol" if z > 0 else "long_vol",
                score=score,
                structure_preset=preset, vol_view=view,
                thesis=(f"{tenor} vol-of-vol z={z:+.2f} "
                        f"(60d vov {current:.2f} vs mean {mu:.2f}). "
                        f"{'Vol regime unstable — strangle for whip' if z > 0 else 'Vol regime calm — short straddle'}."),
                metrics={"zscore": z, "vov_60d": current, "vov_mean": mu},
            ))
    return out


def screen_tail_underpriced(pairs: List[str], tenors: List[str]) -> List[TradeIdea]:
    """10D BF z-score below -1.5 → tail vol cheap; long convexity in 10D wings."""
    out: List[TradeIdea] = []
    preset, view = CATEGORY_DEFAULTS["Tail Underpriced"]
    for pair in pairs:
        for tenor in tenors:
            info = vol_zscore(pair, tenor, "10D_BF")
            if not info:
                continue
            z = info.get("zscore")
            if z is None or z >= -1.5:
                continue
            score = _score_from_z(z, threshold=1.5)
            bf = info.get("current", 0.0)
            out.append(TradeIdea(
                pair=pair, tenor=tenor, category="Tail Underpriced",
                direction="tail", score=score,
                structure_preset=preset, vol_view=view,
                thesis=(f"{tenor} 10D BF z={z:+.2f} (BF {bf:+.2f}). "
                        f"Tail convexity historically cheap — buy 10D strangle."),
                metrics={"zscore": z, "bf_10d": bf,
                         "mean": info.get("mean", 0.0)},
            ))
    return out


# ============================================================================
# Internal helpers
# ============================================================================

def _atm(surface: dict, tenor: str) -> Optional[float]:
    """
    Strict ATM extraction: returns None if the exact tenor isn't present.
    (`extract_surface_atm` falls back to nearest tenor — that's wrong for
    term-structure comparisons where we need the specific tenor.)
    """
    if not isinstance(surface, dict):
        return None
    bucket = surface.get(tenor)
    if not isinstance(bucket, dict):
        # try nested "tenors" key for alternate format
        bucket = surface.get("tenors", {}).get(tenor)
    if not isinstance(bucket, dict):
        return None
    val = bucket.get("atm", bucket.get("ATM"))
    if val is None or not np.isfinite(val):
        return None
    return float(val)


# ============================================================================
# Top-level scan
# ============================================================================

# Cheap module-level cache. The 2-min TTL is aligned to bg_fetcher's cycle.
_SCAN_TTL = 120.0
_scan_cache: Dict[Tuple, Tuple[float, List[TradeIdea]]] = {}
_scan_lock = threading.Lock()


def scan_all(region: str = "ALL",
             tenor_bucket: str = "ALL") -> List[TradeIdea]:
    """
    Run every screen, return a flat list of `TradeIdea`s.

    Cached for `_SCAN_TTL` seconds keyed by (region, tenor_bucket) — every
    panel callback hitting the same bucket reuses the same result during a
    single bg_fetcher cycle.
    """
    key = (region or "ALL", tenor_bucket or "ALL")
    now = _time.time()
    with _scan_lock:
        cached = _scan_cache.get(key)
        if cached is not None and (now - cached[0]) < _SCAN_TTL:
            return cached[1]

    pairs = filter_pairs(region)
    tenors = filter_tenors(tenor_bucket)
    ideas: List[TradeIdea] = []

    # Each screen is wrapped so a single failure doesn't kill the scan.
    for fn, args in [
        (screen_cheap_vol,        (pairs, tenors)),
        (screen_rich_vol,         (pairs, tenors)),
        (screen_iv_rv_rich,       (pairs, tenors)),
        (screen_iv_rv_cheap,      (pairs, tenors)),
        (screen_skew_extreme,     (pairs, tenors)),
        (screen_wings_extreme,    (pairs, tenors)),
        (screen_term_inversion,   (pairs,)),
        (screen_term_kink,        (pairs,)),
        (screen_carry_per_vol,    (pairs,)),
        (screen_vol_of_vol,       (pairs, tenors)),
        (screen_tail_underpriced, (pairs, tenors)),
    ]:
        try:
            ideas.extend(fn(*args))
        except Exception:
            logger.exception("screen %s failed", fn.__name__)

    with _scan_lock:
        _scan_cache[key] = (now, ideas)
    return ideas


def top_n_per_category(ideas: List[TradeIdea], n: int = 5,
                       categories: Optional[List[str]] = None) -> List[TradeIdea]:
    """
    Group `ideas` by category, sort each group by score desc, take top n.
    Returns a flat list ordered by category (in `categories` order if given).
    """
    by_cat: Dict[str, List[TradeIdea]] = {}
    for idea in ideas:
        by_cat.setdefault(idea.category, []).append(idea)

    cat_order = categories if categories else ALL_CATEGORIES
    out: List[TradeIdea] = []
    for cat in cat_order:
        bucket = by_cat.get(cat, [])
        bucket.sort(key=lambda x: x.score, reverse=True)
        out.extend(bucket[:n])
    return out


def clear_cache() -> None:
    """Drop the scan cache (used by tests)."""
    with _scan_lock:
        _scan_cache.clear()


__all__ = [
    "TradeIdea",
    "ALL_CATEGORIES",
    "CATEGORY_DEFAULTS",
    "default_pair_universe",
    "filter_pairs",
    "filter_tenors",
    "scan_all",
    "top_n_per_category",
    "clear_cache",
    # individual screens (importable for unit tests)
    "screen_cheap_vol", "screen_rich_vol",
    "screen_iv_rv_rich", "screen_iv_rv_cheap",
    "screen_skew_extreme", "screen_wings_extreme",
    "screen_term_inversion", "screen_term_kink",
    "screen_carry_per_vol", "screen_vol_of_vol",
    "screen_tail_underpriced",
]
