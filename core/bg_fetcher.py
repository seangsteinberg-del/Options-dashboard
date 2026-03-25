"""
Bloomberg Background Fetcher
=============================
A daemon thread that periodically fetches ALL FX data from Bloomberg
and writes it to the in-memory cache. Dash callbacks never touch
Bloomberg directly — they only read from cache.

This completely decouples the UI from Bloomberg latency.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class BloombergFetcher(threading.Thread):
    """Background thread that pre-populates the FX data cache."""

    def __init__(self, interval: int = 120, historical_interval: int = 3600):
        """
        Parameters
        ----------
        interval : seconds between spot/vol/rates refresh cycles (default 2 min)
        historical_interval : seconds between historical BDH refreshes (default 1 hr)
        """
        super().__init__(daemon=True, name="bbg-fetcher")
        self.interval = interval
        self.historical_interval = historical_interval
        self._stop_event = threading.Event()
        self._first_cycle_done = threading.Event()
        self._last_historical = 0.0

    def wait_for_first_cycle(self, timeout: float = 180):
        """Block until the first fetch cycle completes."""
        logger.info("Waiting for first Bloomberg fetch cycle (max %ds)...", timeout)
        self._first_cycle_done.wait(timeout=timeout)
        if self._first_cycle_done.is_set():
            logger.info("First fetch cycle complete — cache is warm")
        else:
            logger.warning("First fetch cycle timed out after %ds", timeout)

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self):
        from core.bloomberg_fx import (
            mark_thread_as_fetcher, get_all_pairs, _cache_invalidate,
            get_fx_spots, get_fx_vol_surface, get_fx_rates,
            get_fx_historical_spot, get_fx_historical_vol,
        )
        from core.bloomberg import is_connected

        mark_thread_as_fetcher()
        pairs = get_all_pairs()
        logger.info("Background fetcher started: %d pairs, interval=%ds",
                     len(pairs), self.interval)

        first_cycle = True
        while not self._stop_event.is_set():
            if not is_connected():
                logger.warning("BG fetcher: Bloomberg not connected, skipping cycle")
                self._sleep(10)
                continue

            try:
                include_hist = first_cycle or self._historical_due()
                self._fetch_cycle(pairs, include_hist)
            except Exception as e:
                logger.error("BG fetcher cycle failed: %s", e, exc_info=True)

            if first_cycle:
                first_cycle = False
                self._first_cycle_done.set()

            self._sleep(self.interval)

        logger.info("Background fetcher stopped")

    def _historical_due(self) -> bool:
        return (time.monotonic() - self._last_historical) > self.historical_interval

    def _sleep(self, seconds: float):
        """Sleep in 1-second increments for responsive shutdown."""
        for _ in range(int(seconds)):
            if self._stop_event.is_set():
                return
            self._stop_event.wait(1.0)

    def _fetch_cycle(self, pairs, include_historical=False):
        from core.bloomberg_fx import (
            _cache_invalidate, get_fx_spots, get_fx_vol_surface,
            get_fx_rates, get_fx_historical_spot, get_fx_historical_vol,
        )

        t0 = time.monotonic()

        # 1. Spots — single BDP call for all 30 pairs
        try:
            ck = "spots_" + ",".join(pairs)
            _cache_invalidate(ck)
            result = get_fx_spots(pairs)
            logger.info("BG: spots OK (%d pairs)", len(result))
        except Exception as e:
            logger.error("BG: spots failed: %s", e)

        # 2. Vol surfaces — one BDP per pair (most expensive)
        ok, fail = 0, 0
        for pair in pairs:
            try:
                _cache_invalidate(f"volsurf_{pair}")
                surface = get_fx_vol_surface(pair)
                if surface:
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
        logger.info("BG: vol surfaces %d/%d OK", ok, ok + fail)

        # 3. Rates — one BDP per pair
        ok, fail = 0, 0
        for pair in pairs:
            try:
                _cache_invalidate(f"rates_{pair}")
                result = get_fx_rates(pair)
                if result:
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
        logger.info("BG: rates %d/%d OK", ok, ok + fail)

        # 4. Historical data (only when due — expensive BDH calls)
        if include_historical:
            ok = 0
            for pair in pairs:
                try:
                    _cache_invalidate(f"histspot_{pair}_252")
                    df = get_fx_historical_spot(pair, days=252)
                    if not df.empty:
                        ok += 1
                except Exception:
                    pass
            logger.info("BG: historical spots %d/%d OK", ok, len(pairs))

            ok = 0
            for pair in pairs:
                try:
                    _cache_invalidate(f"histvol_{pair}_1M_atm_252")
                    series = get_fx_historical_vol(pair, "1M", "atm", 252)
                    if len(series) > 0:
                        ok += 1
                except Exception:
                    pass
            logger.info("BG: historical vols %d/%d OK", ok, len(pairs))
            self._last_historical = time.monotonic()

        elapsed = time.monotonic() - t0
        logger.info("BG: fetch cycle complete in %.1fs", elapsed)
